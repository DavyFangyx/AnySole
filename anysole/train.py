"""Train AnySole V1."""

from __future__ import annotations

import argparse
import math
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
from anysole.types import ANYSOLE_ROOT, CONFIG_PROBS, CONFIG_T, CONFIG_V, POSE_DIM, anysole_model_dir, assert_batch_shapes
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.geometry import fk_pose6d
from anysole.losses import soft_contact_from_keypoints
from anysole.types import CONFIG_NAMES


DEFAULT_CONFIG = {
    "d_model": 256,
    "dropout": 0.1,
    "tw": 20,
    "batch_size": 256,
    "lr": 1.0e-3,
    "epochs": 200,
    "num_workers": 4,
    "diffusion_train_steps": 1000,
    "diffusion_sample_steps": 50,
    "pose_layers": 6,
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
    "contact_method": "tactile_abs",
    "modal": MODEL_ANYSOLEV1,
    "template_path": "/data/fangyuxuan/projects/gait/AnysoleWorkspace/calibration/insole_templates.json",
    "wandb_mode": "disabled",
    "wandb_project": "Anysole",
    "wandb_entity": "davyfangyuxuan-nanjing-university-of-aeronautics-and-ast",
    "wandb_eval_interval": 5,
    "wandb_experiment_tag": "anysole_v1",
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


def fit_pose_stats(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """One pass over the training loader: per-dim (POSE_DIM,) mean/std of pose_gt.

    The pose head normalizes its 6D input/output with these fixed stats (MDM/RoHM
    normalize the motion representation before diffusing), so the values are
    fitted once on the training set and frozen for the whole run.
    """
    sums = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    sumsq = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            pose = raw_batch["pose_gt"].to(device=device, dtype=torch.float64, non_blocking=True)
            flat = pose.reshape(-1, POSE_DIM)
            sums += flat.sum(dim=0)
            sumsq += flat.square().sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit pose stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    return mean.float(), var.sqrt().float()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnySole V1.")
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Model dropout rate (0 disables dropout; used by Test9 overfitting and retrain experiments). Saved into the checkpoint config.",
    )
    parser.add_argument(
        "--config-probs",
        type=float,
        nargs=3,
        metavar=("VT", "V", "T"),
        default=None,
        help="Sampling probabilities for VT, V-only, and T-only configs.",
    )
    parser.add_argument(
        "--dropoutVT",
        dest="dropout_vt",
        type=str,
        nargs="+",
        metavar=("DROP_V_PCT", "DROP_T_PCT"),
        default=None,
        help="Modality dropout as percentages: V is dropped DROP_V_PCT%% of steps, T is dropped "
        "DROP_T_PCT%% (config_probs becomes [1-v-t, t, v] in VT/V/T order; overrides --config-probs). "
        "Accepts '--dropoutVT 20 30' or '--dropoutVT 20,30'; '0,0' disables modality dropout.",
    )
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="Upper bound for training-step sampling: tau ~ U(0, tau_max) instead of "
        "U(0, diffusion_train_steps). Test11 low-noise band training (e.g. --tau-max 100).",
    )
    parser.add_argument(
        "--tau-fixed",
        type=int,
        default=None,
        help="Fix tau to this value for every training step. Test11 tau0 identity training "
        "(--tau-fixed 0: x_tau = x0 clean input, task degenerates to the identity map).",
    )
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu.")
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help="Model variant to train.")
    parser.add_argument(
        "--contact-method",
        default=None,
        help="Test5 contact-label scheme for contact_gt (see results_display/README.md). "
        "Default: config contact_method, falling back to tactile_abs (原 contact.npy).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override checkpoint output directory.")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_eval_interval", type=int, default=None,
                        help="Epochs between validation evals (3 condition configs x full val set, "
                             "DDIM sampling). One eval costs ~16 normal epochs of wall-clock, so this "
                             "knob dominates total time: 10 -> 57s stall every 10 epochs (~64%% overhead), "
                             "50 -> ~26%% overhead. Runs regardless of wandb mode.")
    parser.add_argument("--wandb_experiment_tag", default=None, help="Batch tag used by Wandb_Analyzer to group runs.")
    parser.add_argument("--loss-cap", type=float, default=None,
                        help="If set, clip the logged train loss metrics (loss_total, loss_pose, loss_traj, "
                             "loss_con, loss_kp, loss_Trec_Tmissing, loss_Vrec_Vmissing, grad_norm) at this "
                             "value, keeping their original names (no *_cap duplicates). Raw charts get "
                             "y-squashed by the first epochs' spike (loss_total 30 -> 0.1, loss_pose 1.5 -> "
                             "0.01), hiding convergence; e.g. --loss-cap 1.0 pins the y-axis to 0-1.")
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
    if args.dropout is not None:
        config["dropout"] = float(args.dropout)
    if args.config_probs is not None:
        config["config_probs"] = args.config_probs
    if args.dropout_vt is not None:
        tokens = [item for item in " ".join(args.dropout_vt).replace(",", " ").split() if item]
        if len(tokens) != 2:
            raise ValueError("--dropoutVT needs two percentages (e.g. '--dropoutVT 20 30' or '--dropoutVT 20,30')")
        drop_v, drop_t = float(tokens[0]) / 100.0, float(tokens[1]) / 100.0
        if drop_v < 0.0 or drop_t < 0.0 or drop_v + drop_t > 1.0:
            raise ValueError("--dropoutVT percentages must be non-negative and sum to at most 100")
        config["config_probs"] = [1.0 - drop_v - drop_t, drop_t, drop_v]
        config["dropout_vt"] = [drop_v * 100.0, drop_t * 100.0]
    if args.modal is not None:
        config["modal"] = args.modal
    if args.contact_method is not None:
        config["contact_method"] = args.contact_method
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)
    for key in ("wandb_mode", "wandb_project", "wandb_entity", "wandb_eval_interval", "wandb_experiment_tag"):
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

    if args.tau_max is not None:
        if not 1 <= args.tau_max <= int(config["diffusion_train_steps"]):
            raise ValueError("--tau-max must be in [1, diffusion_train_steps]")
        config["tau_max"] = int(args.tau_max)
    if args.tau_fixed is not None:
        if not 0 <= args.tau_fixed < int(config["diffusion_train_steps"]):
            raise ValueError("--tau-fixed must be in [0, diffusion_train_steps)")
        config["tau_fixed"] = int(args.tau_fixed)
    if args.loss_cap is not None:
        if not args.loss_cap > 0.0:
            raise ValueError("--loss-cap must be positive")
        config["loss_cap"] = float(args.loss_cap)

    try:
        dataset = AnySoleDataset(
            mode="train",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=int(config["tw"]),
            session_ids=session_ids,
            contact_method=str(config["contact_method"]),
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
        val_dataset = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]), window_length=int(config["tw"]), contact_method=str(config["contact_method"]))
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_windows, pin_memory=device.type == "cuda")
    except (FileNotFoundError, RuntimeError) as exc:
        print("validation disabled: %s" % exc)
    wandb_run = None
    if str(config.get("wandb_mode", "disabled")) != "disabled":
        try:
            import wandb
            # Identity fields consumed by Wandb_Analyzer: experiment_tag groups a
            # training batch, run_kind/model/stage/fold decide the raw/ directory
            # tree. Single-stage, non-cross-validation runs use stage=single,
            # fold=fold0.
            wandb_identity = {
                "experiment_tag": str(config.get("wandb_experiment_tag") or "anysole_v1"),
                "run_kind": "train",
                "model": str(config.get("modal", MODEL_ANYSOLEV1)),
                "stage": "single",
                "fold": "fold0",
            }
            wandb_run = wandb.init(
                mode=str(config["wandb_mode"]),
                project=str(config.get("wandb_project") or "Anysole"),
                entity=config.get("wandb_entity"),
                name="%s_%s_%s" % (wandb_identity["model"], wandb_identity["stage"], wandb_identity["fold"]),
                group="%s_%s" % (wandb_identity["experiment_tag"], wandb_identity["run_kind"]),
                job_type="train",
                tags=[wandb_identity["experiment_tag"], wandb_identity["run_kind"], wandb_identity["model"], wandb_identity["stage"], wandb_identity["fold"]],
                config={**dict(config), **wandb_identity},
            )
        except ImportError as exc:
            raise RuntimeError("W&B monitoring requested but wandb is not installed") from exc
    model_kw = {}
    modal = str(config.get("modal", MODEL_ANYSOLEV1))
    # Keep model variants independent in the centralized results tree. An
    # explicit --out-dir remains authoritative for custom experiments.
    if args.out_dir is None:
        # Model dir naming: anysole_{version/ablation}_{contact_method}. Every
        # contact-label scheme gets its own dir so sweeps never clobber each
        # other's checkpoints/metrics.
        contact_method = str(config.get("contact_method", "tactile_abs"))
        config["out_dir"] = str(anysole_model_dir(modal, contact_method) / "checkpoints")
    if modal == "anysolev1_insole_drift" or bool(config.get("use_insole_drift", False)):
        templates, subject_map = load_template_bank(config["template_path"])
        model_kw.update(templates=templates, subject_to_index=subject_map)
    model = AnySoleModel(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        modal=modal,
        use_insole_drift=bool(config.get("use_insole_drift", False)),
        pose_layers=int(config.get("pose_layers", 6)),
        **model_kw,
    ).to(device)
    # Frozen pose-normalization stats (recorded so eval/infer rebuild the same depth).
    config["pose_layers"] = int(config.get("pose_layers", 6))
    # E4: training noise is scaled by pose_std (see q_sample below).  The flag
    # is saved in every checkpoint so eval/infer can match the sampling init
    # scale; checkpoints trained without it (E3 and earlier) sample unscaled.
    config["noise_scaled"] = True
    model.noise_scaled = True
    pose_mean, pose_std = fit_pose_stats(loader, device)
    model.pose_head.set_stats(pose_mean, pose_std)
    print("pose stats: mean %.4f +/- %.4f, std %.4f +/- %.4f (dims floored at 1e-2: %d/%d)"
          % (pose_mean.mean().item(), pose_mean.std().item(), pose_std.mean().item(),
             pose_std.std().item(), int((pose_std <= 1e-2).sum().item()), POSE_DIM))
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        # E4: cosine LR decay over the whole run (was constant lr; E3 showed
        # val metrics still improving at epoch 800 with lr pinned at 1e-3).
        progress = epoch / float(config["epochs"])
        cos_lr = float(config["lr"]) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = cos_lr
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

            # Test11 τ training-regime knobs: --tau-fixed pins every step to one
            # noise level (0 = identity-mapping training), --tau-max narrows the
            # sampling band to U(0, tau_max) (low-noise region training).
            if "tau_fixed" in config:
                tau = torch.full(
                    (batch_size,),
                    int(config["tau_fixed"]),
                    device=device,
                    dtype=torch.long,
                )
            else:
                tau = torch.randint(
                    0,
                    int(config.get("tau_max", config["diffusion_train_steps"])),
                    (batch_size,),
                    device=device,
                    dtype=torch.long,
                )
            # E4: diffuse with per-dim noise scaled by the pose std (Step2Motion
            # normalizes the data before diffusing; scaling the noise is the
            # same thing here since the head normalizes its input internally).
            # Without this, N(0,1) noise lands on a signal of std ~0.118 (8.5x
            # SNR mismatch) and the DDIM chain injects per-frame jitter.
            noise = torch.randn_like(batch["pose_gt"]) * model.pose_head.pose_std.view(1, 1, -1)
            x_tau = diffusion.q_sample(batch["pose_gt"], tau, noise=noise)
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
        # Atomic replace: a reader (auto-eval torch.load) must never see a
        # half-written file when several runs share one out_dir.  Each process
        # writes its own temp file, so a concurrent save cannot corrupt it.
        # Re-mkdir so an externally deleted results dir cannot kill a long run.
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "ckpt_last.pt"
        tmp_path = out_dir / ("ckpt_last.pt.tmp%d" % os.getpid())
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)
        print("saved %s (epoch %d)" % (ckpt_path, epoch + 1))
        # Val eval cadence is the wall-clock knob: one _evaluate pass (3
        # condition configs x full val set, DDIM sampling) costs ~16 normal
        # epochs of wall-clock.  Triggered by --wandb_eval_interval but runs
        # regardless of wandb mode; results go to the console and, when a
        # wandb run is active, to its log.
        eval_results = None
        eval_interval = max(int(config.get("wandb_eval_interval", 5)), 1)
        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            model.eval()
            eval_results = _evaluate(model, diffusion, val_loader, config, device)
            print("epoch %d eval: %s" % (epoch + 1,
                  "  ".join("%s=%.4f" % (key, value) for key, value in sorted(eval_results.items()))))
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
            # Optional capped logging (--loss-cap): the first epochs spike far
            # above the converged range (loss_total 30 -> 0.1, loss_pose 1.5 ->
            # 0.01), which stretches the raw charts' y-axis and hides
            # convergence.  When set, clip the train losses in place under
            # their original names so the wandb view stays within 0-cap and
            # Wandb_Analyzer picks up the clipped series directly.
            loss_cap = config.get("loss_cap")
            if loss_cap is not None:
                for key in ("loss_total", "loss_pose", "loss_traj", "loss_con", "loss_kp",
                            "loss_Trec_Tmissing", "loss_Vrec_Vmissing", "grad_norm"):
                    full_key = "train/%s" % key
                    if full_key in log:
                        log[full_key] = min(float(log[full_key]), float(loss_cap))
            if eval_results is not None:
                log.update(eval_results)
            wandb_run.log(log, step=epoch + 1)
    if wandb_run is not None:
        wandb_run.finish()
    try:
        from anysole.eval import main as eval_main
        # Training-time auto-eval stays lean: metrics only, no BVH export
        # (run eval again later to refresh predictions/eval_bvh).
        eval_main(["--config", str(args.config), "--ckpt", str(out_dir / "ckpt_last.pt"),
                   "--modal", modal, "--split", "test", "--device", str(device),
                   "--contact-method", str(config["contact_method"]), "--no-write-bvh"])
    except Exception as exc:
        print("automatic test evaluation failed: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
