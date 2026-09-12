"""Train AnySole V1."""

from __future__ import annotations

import argparse
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
from anysole.types import ANYSOLE_ROOT, CONFIG_PROBS, CONFIG_T, CONFIG_V, assert_batch_shapes
from anysole.ablations.insole_drift.templates import load_template_bank


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
    return parser.parse_args(argv)


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
    model_kw = {}
    modal = str(config.get("modal", MODEL_ANYSOLEV1))
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
                global_step += 1
                continue

            losses["loss"].backward()
            optimizer.step()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
