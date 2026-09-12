"""Evaluate AnySole V1 under VT, V-only, and T-only conditioning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.bvh_io import pose_trans_to_motion, write_bvh
from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import fk_pose6d, rot6d_to_rotmat
from anysole.losses import soft_contact_from_keypoints
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_NAMES
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    FPS,
    N_JOINTS,
    POSE_DIM,
    assert_batch_shapes,
)


def _pa_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-joint errors after per-frame similarity Procrustes alignment."""
    n_joints = pred.shape[-2]
    x = pred.reshape(-1, n_joints, 3)
    y = target.reshape(-1, n_joints, 3)
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x0 = x - x_mean
    y0 = y - y_mean
    covariance = x0.transpose(1, 2) @ y0
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[:, -1] = torch.where(
        torch.det(u @ vh) < 0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    rotation = u @ torch.diag_embed(correction) @ vh
    variance = x0.square().sum(dim=(1, 2)).clamp_min(1.0e-8)
    scale = (singular * correction).sum(dim=1) / variance
    aligned = scale[:, None, None] * (x0 @ rotation) + y_mean
    return torch.linalg.vector_norm(aligned - y, dim=-1)


def _rotation_error_deg(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    pred_rot = rot6d_to_rotmat(pred_pose.reshape(*pred_pose.shape[:2], N_JOINTS, 6))
    gt_rot = rot6d_to_rotmat(gt_pose.reshape(*gt_pose.shape[:2], N_JOINTS, 6))
    relative = pred_rot.transpose(-1, -2) @ gt_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


class MetricSums:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {name: 0.0 for name in ("MPJPE", "PA-MPJPE", "MPJRE", "traj_ATE", "contact_acc")}
        self.counts: Dict[str, int] = {name: 0 for name in self.sums}

    def add(self, name: str, values: torch.Tensor, scale: float = 1.0) -> None:
        self.sums[name] += float(values.double().sum().item()) * scale
        self.counts[name] += values.numel()

    def means(self) -> Dict[str, float]:
        return {name: self.sums[name] / max(self.counts[name], 1) for name in self.sums}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnySole V1 with DDIM sampling.")
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-bvh", type=Path, default=None, metavar="DIR")
    return parser.parse_args(argv)


def _load_model(checkpoint: dict, config: dict, device: torch.device) -> AnySoleModel:
    saved_config = checkpoint.get("config", {})
    d_model = int(saved_config.get("d_model", config["d_model"]))
    tw = int(saved_config.get("tw", config["tw"]))
    if tw != int(config["tw"]):
        raise ValueError("Checkpoint tw=%d differs from evaluation config tw=%d" % (tw, int(config["tw"])))
    modal = str(saved_config.get("modal", MODEL_ANYSOLEV1))
    if modal not in MODEL_NAMES:
        raise ValueError("Unknown checkpoint modal %r" % modal)
    use_drift = modal == MODEL_ANYSOLEV1_INSOLE_DRIFT
    model_kw = {}
    if use_drift:
        template_path = saved_config.get("template_path", config.get("template_path"))
        if not template_path:
            raise ValueError("template_path is required for drift-enabled evaluation")
        templates, subject_map = load_template_bank(template_path)
        model_kw.update(templates=templates, subject_to_index=subject_map)
    model = AnySoleModel(d=d_model, tw=tw, modal=modal, use_insole_drift=use_drift, **model_kw).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    model = _load_model(checkpoint, config, device)
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])

    session_ids = None
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = load_split_ids(Path(config["split_csv"]), "val")[: args.limit_sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        session_ids=session_ids,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset contains no valid windows")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )

    for config_value in (CONFIG_VT, CONFIG_V, CONFIG_T):
        metrics = MetricSums()
        exports = {}
        with torch.inference_mode():
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                batch_size = batch["pose_gt"].shape[0]
                config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                batch["config_id"] = config_id
                assert_batch_shapes(batch, batch_size)
                v_feat, t_raw, t_phys = condition_inputs(batch, config_id)
                cond = {
                    "V_feat": v_feat,
                    "T_raw": t_raw,
                    "T_phys": t_phys,
                    "config_id": config_id,
                    "session_id": batch.get("session_id"),
                }
                pred_pose = diffusion.ddim_sample_loop(
                    model,
                    tau_related_kwargs=cond,
                    shape=(batch_size, int(config["tw"]), POSE_DIM),
                    steps=sample_steps,
                    eta=0.0,
                    device=device,
                )
                tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id, batch.get("session_id"))
                pred_trans = out["trans_hat"]
                anchor = batch["trans_anchor"][:, None, :]
                pred_trans_world = pred_trans + anchor
                gt_trans_world = batch["trans_gt"] + anchor
                pred_kp = fk_pose6d(pred_pose, pred_trans_world, batch["offsets"], batch["parents"])

                metrics.add("MPJPE", torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1), 1000.0)
                metrics.add("PA-MPJPE", _pa_error(pred_kp, batch["kp_gt"]), 1000.0)
                metrics.add("MPJRE", _rotation_error_deg(pred_pose, batch["pose_gt"]))
                metrics.add("traj_ATE", torch.linalg.vector_norm(pred_trans_world - gt_trans_world, dim=-1), 1000.0)
                pred_contact = soft_contact_from_keypoints(pred_kp) > 0.5
                metrics.add("contact_acc", (pred_contact == (batch["contact_gt"] > 0.5)).float())

                if args.write_bvh is not None:
                    for idx, session_id in enumerate(raw_batch["session_id"]):
                        exports.setdefault(session_id, []).append(
                            (
                                int(raw_batch["frame_start"][idx]),
                                pred_pose[idx].cpu().numpy(),
                                pred_trans_world[idx].cpu().numpy(),
                                raw_batch["hierarchy"][idx],
                            )
                        )

        values = metrics.means()
        print(
            "%s MPJPE=%.3fmm PA-MPJPE=%.3fmm MPJRE=%.3fdeg traj_ATE=%.3fmm contact_acc=%.4f"
            % (
                CONFIG_NAMES[config_value],
                values["MPJPE"],
                values["PA-MPJPE"],
                values["MPJRE"],
                values["traj_ATE"],
                values["contact_acc"],
            )
        )
        if args.write_bvh is not None:
            for session_id, windows in exports.items():
                windows.sort(key=lambda item: item[0])
                pose = np.concatenate([item[1] for item in windows], axis=0)
                trans = np.concatenate([item[2] for item in windows], axis=0)
                output_path = args.write_bvh / ("%s_%s.bvh" % (session_id, CONFIG_NAMES[config_value]))
                write_bvh(output_path, windows[0][3], pose_trans_to_motion(pose, trans), 1.0 / FPS)
                print("wrote %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
