"""Train AnySole V1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import (
    AnySoleDataset,
    collate_windows,
    load_split_ids,
    sample_config_ids,
)
from anysole.diffusion import GaussianDiffusion
from anysole.losses import compute_losses
from anysole.models import AnySoleModel, MODEL_NAMES, MODEL_ANYSOLEV1
from anysole.types import ANYSOLE_ROOT, GAIT_ROOT, CONFIG_PROBS, CONFIG_T, CONFIG_V, assert_batch_shapes
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.geometry import fk_pose6d
from anysole.losses import soft_contact_from_keypoints
from anysole.types import CONFIG_NAMES


DEFAULT_CONFIG = {
    "d_model": 256,
    "tw": 20,
    "batch_size": 256,
    "lr": 1.0e-3,
    "epochs": 200,
    "num_workers": 4,
    "diffusion_train_steps": 1000,
    "diffusion_sample_steps": 50,
    "lambda_pose": 1.0,
    "lambda_traj": 1.0,
    "lambda_trec": 0.1,
    "lambda_vrec": 0.1,
    "lambda_kp": 1.0,
    "lambda_con": 0.1,
    "traj_velocity_w": 1.0,
    "traj_delta_w": 1.0,
    "traj_delta_weight_power": 1.0,
    "traj_deltas": [2, 4, 8, 19],
    "config_probs": list(CONFIG_PROBS),
    "cam_id": 3,
    "seq_root": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/derived/MotionPRO/sequences/cam3",
    "split_csv": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/splits/default/splits.csv",
    "cache_root": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3",
    "out_dir": "/data/fangyuxuan/projects/gait/AnySole/outputs/v1",
    "use_insole_drift": False,
    "modal": MODEL_ANYSOLEV1,
    "template_path": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/calibration/insole_templates.json",
    "wandb_mode": "disabled",
    "wandb_project": "Anysole",
    "wandb_entity": "davyfangyuxuan-nanjing-university-of-aeronautics-and-ast",
    "wandb_eval_interval": 5,
}


def load_config(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read %s" % path) from exc
    loaded = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping: %s" % path)
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def condition_inputs(batch: dict, config_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero absent modalities without modifying reconstruction targets in batch."""
    drop_v = (config_id == CONFIG_T).view(-1, 1, 1)
    drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
    v_feat = torch.where(drop_v, torch.zeros_like(batch["V_feat"]), batch["V_feat"])
    t_raw = torch.where(drop_t, torch.zeros_like(batch["T_raw"]), batch["T_raw"])
    t_phys = torch.where(drop_t, torch.zeros_like(batch["T_phys"]), batch["T_phys"])
    return v_feat, t_raw, t_phys


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnySole V1.")
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--config-probs",
        type=float,
        nargs=3,
        metavar=("VT", "V", "T"),
        default=None,
        help="Sampling probabilities for VT, V-only, and T-only configs.",
    )
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu.")
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help="Model variant to train.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override checkpoint output directory.")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_eval_interval", type=int, default=None)
    return parser.parse_args(argv)


def _evaluate(model, diffusion, loader, config, device):
    """Return the epoch-level VT/V/T metrics used by the training dashboard."""
    result = {}
    with torch.inference_mode():
        for config_value, config_name in zip((0, 1, 2), CONFIG_NAMES):
            totals = {"mpjpe": 0.0, "root_ate": 0.0, "contact_f1": 0.0, "foot_slide": 0.0, "n": 0}
            tau0_mpjpe = 0.0
            first_batch = True
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                bsz = batch["pose_gt"].shape[0]
                cid = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                v_feat, t_raw, t_phys = condition_inputs(batch, cid)
                pred_pose = diffusion.ddim_sample_loop(
                    model, tau_related_kwargs={"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                    "config_id": cid, "session_id": batch.get("session_id")},
                    shape=(bsz, int(config["tw"]), int(batch["pose_gt"].shape[-1])),
                    steps=int(config["diffusion_sample_steps"]), eta=0.0, device=device)
                zero = torch.zeros(bsz, device=device, dtype=torch.long)
                out = model(v_feat, t_raw, t_phys, pred_pose, zero, cid, batch.get("session_id"))
                anchor = batch["trans_anchor"][:, None, :]
                pred_trans = out["trans_hat"] + anchor
                gt_trans = batch["trans_gt"] + anchor
                pred_kp = fk_pose6d(pred_pose, pred_trans, batch["offsets"], batch["parents"])
                n = bsz
                totals["mpjpe"] += float(torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1).mean().item()) * n * 1000.0
                if first_batch:
                    # tau=0 clean-input reconstruction MPJPE on one batch (no
                    # diffusion sampling): tells apart "cannot fit the training
                    # data" from "cannot denoise", on the same joint scope as
                    # eval.py (all 23 joints).
                    out0 = model(v_feat, t_raw, t_phys, batch["pose_gt"], zero, cid, batch.get("session_id"))
                    kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
                    tau0_mpjpe = float(torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0
                    first_batch = False
                pred_rel = pred_trans - pred_trans[:, :1]
                gt_rel = gt_trans - gt_trans[:, :1]
                totals["root_ate"] += float(torch.linalg.vector_norm(pred_rel - gt_rel, dim=-1).mean().item()) * n
                pred_contact = soft_contact_from_keypoints(pred_kp) > 0.5
                gt_contact = batch["contact_gt"] > 0.5
                tp = (pred_contact & gt_contact).sum().item()
                fp = (pred_contact & ~gt_contact).sum().item()
                fn = (~pred_contact & gt_contact).sum().item()
                totals.setdefault("tp", 0); totals.setdefault("fp", 0); totals.setdefault("fn", 0)
                totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
                foot = pred_kp[:, :, (17, 18, 21, 22), :][:, :, :, [0, 2]]
                # Keypoints are meters; report horizontal contact speed in cm/s.
                speed = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1).mean(dim=-1) * 40.0 * 100.0
                contact_frames = pred_contact[:, 1:].any(dim=-1)
                slide = speed[contact_frames]
                if slide.numel():
                    totals["foot_slide"] += float(slide.mean().item()) * n
                totals["n"] += n
            count = max(totals["n"], 1)
            p = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fp", 0), 1)
            r = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fn", 0), 1)
            result.update({f"val/mpjpe/{config_name}": totals["mpjpe"] / count,
                           f"val/root_ate/{config_name}": totals["root_ate"] / count,
                           f"val/contact_f1/{config_name}": 2 * p * r / max(p + r, 1e-8),
                           f"val/foot_slide/{config_name}": totals["foot_slide"] / count,
                           f"val/tau0_mpjpe/{config_name}": tau0_mpjpe})
    result["val/gap_mpjpe_dropT"] = result["val/mpjpe/V"] - result["val/mpjpe/VT"]
    result["val/gap_mpjpe_dropV"] = result["val/mpjpe/T"] - result["val/mpjpe/VT"]
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.config_probs is not None:
        config["config_probs"] = args.config_probs
    if args.modal is not None:
        config["modal"] = args.modal
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)
    for key in ("wandb_mode", "wandb_project", "wandb_entity", "wandb_eval_interval"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config.get("modal", MODEL_ANYSOLEV1) not in MODEL_NAMES:
        raise ValueError("Unknown modal %r; expected one of %s" % (config.get("modal"), MODEL_NAMES))

    session_ids = None
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = load_split_ids(Path(config["split_csv"]), "train")[: args.limit_sessions]

    try:
        dataset = AnySoleDataset(
            mode="train",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=int(config["tw"]),
            session_ids=session_ids,
        )
    except FileNotFoundError as exc:
        if "HRNet cache" in str(exc):
            raise FileNotFoundError(
                "%s Run `python -m anysole.data.extract_hrnet --cam-id %d`; "
                "training never extracts features on demand."
                % (exc, int(config["cam_id"]))
            ) from exc
        raise
    if len(dataset) == 0:
        raise RuntimeError("Training dataset contains no valid windows")

    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    try:
        val_dataset = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]), window_length=int(config["tw"]))
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_windows, pin_memory=device.type == "cuda")
    except (FileNotFoundError, RuntimeError) as exc:
        print("validation disabled: %s" % exc)
    wandb_run = None
    if str(config.get("wandb_mode", "disabled")) != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(mode=str(config["wandb_mode"]), project=str(config.get("wandb_project") or "Anysole"), entity=config.get("wandb_entity"), config=dict(config))
        except ImportError as exc:
            raise RuntimeError("W&B monitoring requested but wandb is not installed") from exc
    model_kw = {}
    modal = str(config.get("modal", MODEL_ANYSOLEV1))
    # Keep model variants independent in the centralized results tree. An
    # explicit --out-dir remains authoritative for custom experiments.
    if args.out_dir is None:
        results_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results")))
        config["out_dir"] = str(results_root / "AnySole" / modal / "checkpoints")
    if modal == "anysolev1_insole_drift" or bool(config.get("use_insole_drift", False)):
        templates, subject_map = load_template_bank(config["template_path"])
        model_kw.update(templates=templates, subject_to_index=subject_map)
    model = AnySoleModel(d=int(config["d_model"]), tw=int(config["tw"]), modal=modal, use_insole_drift=bool(config.get("use_insole_drift", False)), **model_kw).to(device)
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_start_step = global_step
        nonfinite_skips = 0
        epoch_sums = {"loss_total": 0.0, "loss_pose": 0.0, "loss_traj": 0.0, "loss_con": 0.0, "loss_kp": 0.0, "loss_Trec_Tmissing": 0.0, "loss_Vrec_Vmissing": 0.0, "grad_norm": 0.0, "n": 0, "n_tmissing": 0, "n_vmissing": 0}
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            batch_size = batch["pose_gt"].shape[0]
            config_id = sample_config_ids(batch_size, probs=config["config_probs"]).to(device)
            batch["config_id"] = config_id
            assert_batch_shapes(batch, batch_size)
            v_feat, t_raw, t_phys = condition_inputs(batch, config_id)

            tau = torch.randint(
                0,
                int(config["diffusion_train_steps"]),
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            x_tau = diffusion.q_sample(batch["pose_gt"], tau)
            optimizer.zero_grad(set_to_none=True)
            out = model(v_feat, t_raw, t_phys, x_tau, tau, config_id, batch.get("session_id"))
            losses = compute_losses(out, batch, config_id, config)

            finite = torch.isfinite(losses["loss"]) and torch.isfinite(out["F"]).all()
            if not bool(finite):
                for group in optimizer.param_groups:
                    group["lr"] = 1.0e-4
                optimizer.zero_grad(set_to_none=True)
                print("non-finite fusion/loss at step %d; lr set to 1e-4, batch skipped" % global_step)
                nonfinite_skips += 1
                global_step += 1
                continue

            losses["loss"].backward()
            grad_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq = grad_sq + parameter.grad.detach().square().sum()
            epoch_sums["grad_norm"] += float(torch.sqrt(grad_sq).item()) * batch_size
            optimizer.step()
            n = batch_size
            epoch_sums["n"] += n
            for key, name in (("loss", "loss_total"), ("L_pose", "loss_pose"), ("L_traj", "loss_traj"), ("L_con", "loss_con"), ("L_kp", "loss_kp")):
                epoch_sums[name] += float(losses[key].detach().item()) * n
            n_tmissing = int((config_id == CONFIG_T).sum().item())
            n_vmissing = int((config_id == CONFIG_V).sum().item())
            if n_tmissing:
                epoch_sums["loss_Trec_Tmissing"] += float(losses["L_Trec"].detach().item()) * n_tmissing
                epoch_sums["n_tmissing"] += n_tmissing
            if n_vmissing:
                epoch_sums["loss_Vrec_Vmissing"] += float(losses["L_Vrec"].detach().item()) * n_vmissing
                epoch_sums["n_vmissing"] += n_vmissing
            if global_step % 50 == 0:
                values = " ".join("%s=%.6f" % (key, value.detach().item()) for key, value in losses.items())
                print("epoch=%d step=%d %s" % (epoch + 1, global_step, values))
            global_step += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": dict(config),
        }
        torch.save(checkpoint, out_dir / "ckpt_last.pt")
        print("saved %s (epoch %d)" % (out_dir / "ckpt_last.pt", epoch + 1))
        if wandb_run is not None:
            denom = max(epoch_sums["n"], 1)
            log = {"epoch": epoch + 1}
            log["train/global_step"] = global_step
            log["train/steps_per_epoch"] = global_step - epoch_start_step
            log["train/nonfinite_skips"] = nonfinite_skips
            for key, value in epoch_sums.items():
                if key in ("n", "n_tmissing", "n_vmissing"):
                    continue
                metric_denom = denom
                if key == "loss_Trec_Tmissing": metric_denom = max(epoch_sums["n_tmissing"], 1)
                if key == "loss_Vrec_Vmissing": metric_denom = max(epoch_sums["n_vmissing"], 1)
                log[f"train/{key}"] = value / metric_denom
            log["train/lr"] = optimizer.param_groups[0]["lr"]
            if val_loader is not None and (epoch + 1) % max(int(config.get("wandb_eval_interval", 5)), 1) == 0:
                model.eval(); log.update(_evaluate(model, diffusion, val_loader, config, device))
            wandb_run.log(log, step=epoch + 1)
    if wandb_run is not None:
        wandb_run.finish()
    try:
        from anysole.eval import main as eval_main
        eval_main(["--config", str(args.config), "--ckpt", str(out_dir / "ckpt_last.pt"),
                   "--modal", modal, "--split", "test", "--device", str(device)])
    except Exception as exc:
        print("automatic test evaluation failed: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
