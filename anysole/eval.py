"""Evaluate AnySole V1 under VT, V-only, and T-only conditioning.

Besides the motion metrics, every mode also reports tactile aux-head
reconstruction quality (T_mae / T_rmse / T_corr over the 96 cells).  Under
V-only conditioning the tactile input is zeroed, so that row measures V2T
(vision-to-tactile) generation and is mirrored by the top-level "v2t"
field of the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.bvh_io import pose_trans_to_motion, write_bvh
from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import fk_pose6d, positions_to_6d_np, rot6d_to_rotmat, rot6d_to_rotmat_np, rotmat_to_6d, rotmat_to_6d_np
from anysole.losses import soft_contact_from_keypoints
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_NAMES
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    FPS,
    GAIT_ROOT,
    N_JOINTS,
    POSE_DIM,
    anysole_model_dir,
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


def _tactile_corr(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame Pearson correlation over the 96 pressure cells.

    Returns (correlation, valid): frames where either side is constant
    (std ~ 0, e.g. a fully off insole) have no defined correlation and are
    masked out instead of contributing garbage values.
    """
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    denom = (pred_c.square().sum(dim=-1) * target_c.square().sum(dim=-1)).sqrt()
    corr = (pred_c * target_c).sum(dim=-1) / denom.clamp_min(1.0e-8)
    return corr, denom > 1.0e-8


def _rotation_error_deg(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    pred_rot = rot6d_to_rotmat(pred_pose.reshape(*pred_pose.shape[:2], N_JOINTS, 6))
    gt_rot = rot6d_to_rotmat(gt_pose.reshape(*gt_pose.shape[:2], N_JOINTS, 6))
    relative = pred_rot.transpose(-1, -2) @ gt_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


class MetricSums:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {
            name: 0.0
            for name in ("MPJPE", "PA-MPJPE", "MPJRE", "traj_ATE", "contact_acc", "T_mae", "T_mse", "T_corr")
        }
        self.counts: Dict[str, int] = {name: 0 for name in self.sums}

    def add(self, name: str, values: torch.Tensor, scale: float = 1.0) -> None:
        self.sums[name] += float(values.double().sum().item()) * scale
        self.counts[name] += values.numel()

    def means(self) -> Dict[str, float]:
        out = {name: self.sums[name] / max(self.counts[name], 1) for name in self.sums}
        out["T_rmse"] = float(out.pop("T_mse") ** 0.5)
        return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnySole V1 with DDIM sampling.")
    parser.add_argument("--config", type=Path, default=ANYSOLE_ROOT / "configs" / "v1.yaml")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint .pt. Omitted when --modal and --contact-method name a model dir "
        "under results/AnySole (checkpoints/ckpt_last.pt is used).",
    )
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--contact-method",
        default=None,
        help="Test5 contact-label scheme for contact_gt (see results_display/README.md), "
        "and the model-dir suffix when --ckpt is omitted. "
        "Default: the checkpoint's training contact_method, then config, then tactile_abs.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config-id",
        default=None,
        metavar="MODE[,MODE...]",
        help="Evaluation mode(s): VT2M, V2M, T2M; omitted means all three.",
    )
    parser.add_argument(
        "--write-bvh",
        type=Path,
        default=None,
        metavar="DIR",
        help="BVH/traj export dir (default: <model>/predictions/eval_bvh).",
    )
    parser.add_argument("--no-write-bvh", action="store_true", help="Disable the default BVH/traj export.")
    parser.add_argument("--metrics-out", type=Path, default=None, metavar="FILE",
                        help="Write VT2M/V2M/T2M metrics as JSON (default: model results metrics directory).")
    return parser.parse_args(argv)


def _parse_config_values(value: Optional[str]) -> List[int]:
    mode_to_config = dict(zip(CONFIG_MODE_NAMES, (CONFIG_VT, CONFIG_V, CONFIG_T)))
    if value is None:
        return [CONFIG_VT, CONFIG_V, CONFIG_T]
    modes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--config-id must contain at least one mode")
    unknown = [item for item in modes if item not in mode_to_config]
    if unknown:
        raise ValueError("Unknown mode(s) %s; choose from VT2M,V2M,T2M" % ",".join(unknown))
    return list(dict.fromkeys(mode_to_config[item] for item in modes))


def _discover_model_targets(
    modal: Optional[str], contact_method: Optional[str]
) -> List[tuple[str, str, Path]]:
    """Scan results/AnySole for model dirs named <modal>_<contact_method>.

    Returns (modal, contact_method, ckpt) triples sorted by dir name, so a
    bare ``python -m anysole.eval`` re-evaluates every trained model.
    Longest modal names match first (``anysolev1_insole_drift_tactile_abs``
    is the drift modal plus ``tactile_abs``, not ``anysolev1`` plus a long
    contact method), and dirs without ``checkpoints/ckpt_last.pt`` are
    skipped with a warning.
    """
    anysole_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results"))) / "AnySole"
    targets: List[tuple[str, str, Path]] = []
    if not anysole_root.is_dir():
        return targets
    for child in sorted(anysole_root.iterdir()):
        if not child.is_dir():
            continue
        for name in sorted(MODEL_NAMES, key=len, reverse=True):
            prefix = name + "_"
            if not (child.name.startswith(prefix) and len(child.name) > len(prefix)):
                continue
            contact = child.name[len(prefix):]
            if (modal is not None and name != modal) or (
                contact_method is not None and contact != contact_method
            ):
                break
            ckpt = child / "checkpoints" / "ckpt_last.pt"
            if ckpt.is_file():
                targets.append((name, contact, ckpt))
            else:
                print("skip %s: missing %s" % (child.name, ckpt))
            break
    return targets


def _load_model(checkpoint: dict, config: dict, device: torch.device) -> AnySoleModel:
    saved_config = checkpoint.get("config", {})
    d_model = int(saved_config.get("d_model", config["d_model"]))
    # tw always comes from the checkpoint, so evaluating an older model stays
    # correct even after configs/v1.yaml has moved on.
    tw = int(saved_config.get("tw", config["tw"]))
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
    dropout = float(saved_config.get("dropout", config.get("dropout", 0.1)))
    pose_layers = int(saved_config.get("pose_layers", 6))
    tactile_input = str(saved_config.get("tactile_input", "raw108"))
    tactile_direct = bool(saved_config.get("tactile_direct", False))
    no_imu = bool(saved_config.get("no_imu", False))
    model = AnySoleModel(
        d=d_model, tw=tw, modal=modal, use_insole_drift=use_drift, dropout=dropout,
        pose_layers=pose_layers, tactile_input=tactile_input, tactile_direct=tactile_direct,
        no_imu=no_imu,
        **model_kw,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    # E4: sampling init scale must match the checkpoint's training noise scale
    # (train.py saves noise_scaled=True when it scales q_sample noise by
    # pose_std; older checkpoints lack the flag and sample unscaled).
    model.noise_scaled = bool(saved_config.get("noise_scaled", False))
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_values = _parse_config_values(args.config_id)
    config = load_config(args.config)
    if args.modal is not None:
        config["modal"] = args.modal
    if args.ckpt is not None:
        targets: List[tuple[Optional[str], Optional[str], Path]] = [
            (args.modal, args.contact_method, args.ckpt)
        ]
    elif args.modal is not None and args.contact_method is not None:
        targets = [
            (
                args.modal,
                args.contact_method,
                anysole_model_dir(args.modal, args.contact_method) / "checkpoints" / "ckpt_last.pt",
            )
        ]
    else:
        # Bare `python -m anysole.eval`: auto-scan results/AnySole and
        # evaluate every model whose dir is named <modal>_<contact_method>.
        targets = _discover_model_targets(args.modal, args.contact_method)
        if not targets:
            raise ValueError(
                "--ckpt is required unless results/AnySole contains model dirs named "
                "<modal>_<contact_method> (e.g. anysolev1_tactile_abs) with "
                "checkpoints/ckpt_last.pt"
            )
    multi = len(targets) > 1
    if multi and args.metrics_out is not None:
        raise ValueError(
            "--metrics-out is per-model; omit it when evaluating several models "
            "(each model's metrics go to its own metrics/test.json)"
        )
    device = resolve_device(args.device)
    for modal, contact_method, ckpt in targets:
        _evaluate_one(args, config, config_values, modal, contact_method, ckpt, device, multi=multi)
    return 0


def _positions_to_6d_batch(pred_pos: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
    """E6.1 export conversion: root-local positions -> WORLD 6D, per sample.

    positions_to_6d_np recovers rest-frame rotations; pre-multiplying the root
    by the session frame-0 rotation (dataset root_rot_init) yields world
    rotations for FK metrics and BVH export.
    """
    out = []
    for i in range(pred_pos.shape[0]):
        sixd = positions_to_6d_np(
            pred_pos[i].cpu().numpy(),
            batch["offsets"][i].cpu().numpy(),
            batch["parents"][i].cpu().numpy(),
        )
        root_init = batch["root_rot_init"][i].cpu().numpy()  # (3,3)
        root_local = rot6d_to_rotmat_np(sixd.reshape(-1, N_JOINTS, 6)[:, 0:1, :])[:, 0]
        world_root = np.einsum("ij,tjk->tik", root_init, root_local)
        sixd[:, 0:6] = rotmat_to_6d_np(world_root)
        out.append(torch.from_numpy(sixd))
    return torch.stack(out, dim=0).to(device)


def _session_window_groups(dataset: AnySoleDataset) -> dict:
    """{session_id: [dataset indices]} in dataset order (consecutive windows)."""
    groups: dict = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset[i]["session_id"], []).append(i)
    return groups


def _sample_x_t_init(model, bsz, tw, device, prior=None):
    """Sampling init: pure noise, or E6.5 marginal start with a prior."""
    x_T = torch.randn(bsz, tw, model.pose_head.pose_dim, device=device)
    if getattr(model, "noise_scaled", False):
        x_T = x_T * model.pose_head.pose_std.to(device).view(1, 1, -1)
    if prior is not None:
        sqrt_abar = float(np.sqrt(model.diffusion_abar_top))
        x_T = sqrt_abar * prior.to(device) + float(np.sqrt(1.0 - model.diffusion_abar_top)) * x_T
    return x_T


def _evaluate_one(
    args: argparse.Namespace,
    config: dict,
    config_values: List[int],
    modal: Optional[str],
    contact_method: Optional[str],
    ckpt: Path,
    device: torch.device,
    multi: bool = False,
) -> None:
    """Evaluate one checkpoint under each requested conditioning mode."""
    ckpt_parent = ckpt.parent
    model_root = ckpt_parent.parent if ckpt_parent.name == "checkpoints" else ckpt_parent
    print("== evaluating %s (ckpt %s)" % (model_root.name, ckpt))
    bvh_out = None
    if not args.no_write_bvh:
        bvh_out = args.write_bvh or model_root / "predictions" / "eval_bvh"
        if multi and args.write_bvh is not None:
            # Scan mode with an explicit export dir: nest per model so
            # different checkpoints don't clobber each other's BVHs.
            bvh_out = args.write_bvh / model_root.name
    checkpoint = torch.load(ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % ckpt)
    checkpoint_modal = str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1))
    if modal is not None and modal != checkpoint_modal:
        raise ValueError("--modal %s does not match checkpoint modal %s" % (modal, checkpoint_modal))
    ckpt_contact = str(checkpoint.get("config", {}).get("contact_method") or "")
    contact_method = contact_method or ckpt_contact or str(config.get("contact_method", "tactile_abs"))
    model = _load_model(checkpoint, config, device)
    # --no-imu checkpoints: the eval dataset must build the 38-dim T_s2m
    # (IMU channels deleted) to match the checkpoint's encoder input width.
    no_imu = bool(checkpoint.get("config", {}).get("no_imu", False))
    tw = int(checkpoint.get("config", {}).get("tw", config["tw"]))
    diffusion = GaussianDiffusion(n_train_steps=int(checkpoint.get("config", {}).get("diffusion_train_steps", config["diffusion_train_steps"])))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])
    model.diffusion_abar_top = float(diffusion.alphas_cumprod[diffusion.n_train_steps - 1])
    pos_mode = checkpoint_modal == MODEL_ANYSOLEV1_POS
    continuation = bool(config.get("continuation", False))
    warm_start = bool(config.get("warm_start", False))

    session_ids = None
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = load_split_ids(Path(config["split_csv"]), args.split)[: args.limit_sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=tw,
        session_ids=session_ids,
        contact_method=contact_method,
        no_imu=no_imu,
        # E6.4: continuation eval uses overlapping windows (stride = tw/2);
        # without it the stride stays non-overlapping (E3 behavior).
        stride=(tw // 2) if (pos_mode and continuation) else None,
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

    all_values = {}
    for config_value in config_values:
        metrics = MetricSums()
        exports = {}
        with torch.inference_mode():
            if pos_mode:
                # E6.1 position mode: sample positions, convert to world 6D
                # per window (R_init composition), then reuse all 6D metrics.
                if continuation:
                    # E6.4: each session's windows share one DDIM chain.
                    half = tw // 2
                    for session_id, idxs in _session_window_groups(dataset).items():
                        raw = [dataset[i] for i in idxs]
                        batch = move_batch(collate_windows(raw), device)
                        bsz = batch["pose_gt"].shape[0]
                        config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, bsz, tw=tw)
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(bsz, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, bsz, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop_continue(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, half=half)[0]
                        tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                        corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                        for w in range(bsz):
                            sl = slice(0, tw) if w == 0 else slice(half, tw)
                            p6 = pred6d[w : w + 1, sl]
                            pt = pred_trans[w : w + 1, sl]
                            gtt = gt_trans[w : w + 1, sl]
                            kp = fk_pose6d(p6, pt, batch["offsets"][w : w + 1], batch["parents"][w : w + 1])
                            kp_gt = batch["kp_gt"][w : w + 1, sl]
                            metrics.add("MPJPE", torch.linalg.vector_norm(kp - kp_gt, dim=-1), 1000.0)
                            metrics.add("PA-MPJPE", _pa_error(kp, kp_gt), 1000.0)
                            metrics.add("MPJRE", _rotation_error_deg(p6, batch["pose_gt"][w : w + 1, sl]))
                            metrics.add("traj_ATE", torch.linalg.vector_norm(pt - gtt, dim=-1), 1000.0)
                            pred_contact = soft_contact_from_keypoints(kp) > 0.5
                            metrics.add("contact_acc", (pred_contact == (batch["contact_gt"][w : w + 1, sl] > 0.5)).float())
                            metrics.add("T_mae", t_ae[w : w + 1, sl])
                            metrics.add("T_mse", t_ae[w : w + 1, sl].square())
                            if bool(corr_valid[w : w + 1, sl].any()):
                                metrics.add("T_corr", corr[w : w + 1, sl][corr_valid[w : w + 1, sl]])
                            if bvh_out is not None:
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw[w]["frame_start"]) + (0 if w == 0 else half),
                                        p6[0].cpu().numpy(),
                                        pt[0].cpu().numpy(),
                                        gtt[0].cpu().numpy(),
                                        raw[w]["hierarchy"],
                                    )
                                )
                else:
                    for raw_batch in loader:
                        batch = move_batch(raw_batch, device)
                        batch_size = batch["pose_gt"].shape[0]
                        config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, batch_size, tw=tw)
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, batch_size, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, eta=0.0, device=device)
                        tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        pred_kp = fk_pose6d(pred6d, pred_trans, batch["offsets"], batch["parents"])
                        metrics.add("MPJPE", torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1), 1000.0)
                        metrics.add("PA-MPJPE", _pa_error(pred_kp, batch["kp_gt"]), 1000.0)
                        metrics.add("MPJRE", _rotation_error_deg(pred6d, batch["pose_gt"]))
                        metrics.add("traj_ATE", torch.linalg.vector_norm(pred_trans - gt_trans, dim=-1), 1000.0)
                        pred_contact = soft_contact_from_keypoints(pred_kp) > 0.5
                        metrics.add("contact_acc", (pred_contact == (batch["contact_gt"] > 0.5)).float())
                        t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                        metrics.add("T_mae", t_ae)
                        metrics.add("T_mse", t_ae.square())
                        corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                        if bool(corr_valid.any()):
                            metrics.add("T_corr", corr[corr_valid])
                        if bvh_out is not None:
                            for idx, session_id in enumerate(raw_batch["session_id"]):
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw_batch["frame_start"][idx]),
                                        pred6d[idx].cpu().numpy(),
                                        pred_trans[idx].cpu().numpy(),
                                        gt_trans[idx].cpu().numpy(),
                                        raw_batch["hierarchy"][idx],
                                    )
                                )
            else:
                for raw_batch in loader:
                    batch = move_batch(raw_batch, device)
                    batch_size = batch["pose_gt"].shape[0]
                    config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                    batch["config_id"] = config_id
                    assert_batch_shapes(batch, batch_size, tw=tw)
                    v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                    cond = {
                        "V_feat": v_feat,
                        "T_raw": t_raw,
                        "T_phys": t_phys,
                        "T_s2m": t_s2m,
                        "config_id": config_id,
                        "session_id": batch.get("session_id"),
                    }
                    prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                    pred_pose = diffusion.ddim_sample_loop(
                        model,
                        tau_related_kwargs=cond,
                        shape=(batch_size, tw, POSE_DIM),
                        steps=sample_steps,
                        eta=0.0,
                        device=device,
                        prior=prior,
                    )
                    tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                    out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                batch.get("session_id"), T_s2m=t_s2m)
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

                    # Tactile aux-head metrics (normalized 0-1 units).  Under
                    # V-only conditioning the T input is zeroed, so these numbers
                    # measure V2T generation quality (the V2M row in the report).
                    t_ae = (out["pressure_hat"] - batch["T_raw"]).abs()
                    metrics.add("T_mae", t_ae)
                    metrics.add("T_mse", t_ae.square())
                    corr, corr_valid = _tactile_corr(out["pressure_hat"], batch["T_raw"])
                    if bool(corr_valid.any()):
                        metrics.add("T_corr", corr[corr_valid])

                    if bvh_out is not None:
                        for idx, session_id in enumerate(raw_batch["session_id"]):
                            exports.setdefault(session_id, []).append(
                                (
                                    int(raw_batch["frame_start"][idx]),
                                    pred_pose[idx].cpu().numpy(),
                                    pred_trans_world[idx].cpu().numpy(),
                                    gt_trans_world[idx].cpu().numpy(),
                                    raw_batch["hierarchy"][idx],
                                )
                            )

        values = metrics.means()
        all_values[CONFIG_MODE_NAMES[config_value]] = values
        print(
            "%s MPJPE=%.3fmm PA-MPJPE=%.3fmm MPJRE=%.3fdeg traj_ATE=%.3fmm contact_acc=%.4f T_mae=%.3f T_rmse=%.3f T_corr=%.3f"
            % (
                CONFIG_MODE_NAMES[config_value],
                values["MPJPE"],
                values["PA-MPJPE"],
                values["MPJRE"],
                values["traj_ATE"],
                values["contact_acc"],
                values["T_mae"],
                values["T_rmse"],
                values["T_corr"],
            )
        )
        if bvh_out is not None:
            for session_id, windows in exports.items():
                windows.sort(key=lambda item: item[0])
                pose = np.concatenate([item[1] for item in windows], axis=0)
                trans = np.concatenate([item[2] for item in windows], axis=0)
                gt_trans = np.concatenate([item[3] for item in windows], axis=0)
                # E4: windows are non-overlapping (stride = window_length), so
                # the seam between consecutive windows is a hard cut between
                # two independent predictions (measured 213mm jump vs 8.8mm
                # GT frame-to-frame).  Crossfade F frames on both sides of
                # every seam; 6D poses are re-orthonormalized after blending.
                FADE = 0 if pos_mode else 4
                n_windows = len(windows)
                if n_windows > 1 and FADE > 0:
                    for w in range(1, n_windows):
                        seam = w * tw
                        if seam + FADE > pose.shape[0]:
                            break
                        for j in range(FADE):
                            alpha = (j + 1) / (FADE + 1)
                            a, b = seam - FADE + j, seam + j
                            old_a, old_b = pose[a].copy(), pose[b].copy()
                            pose[a] = (1 - alpha) * old_a + alpha * old_b
                            pose[b] = (1 - alpha) * old_b + alpha * old_a
                            old_ta, old_tb = trans[a].copy(), trans[b].copy()
                            trans[a] = (1 - alpha) * old_ta + alpha * old_tb
                            trans[b] = (1 - alpha) * old_tb + alpha * old_ta
                # re-orthonormalize the blended 6D vectors
                # (Gram-Schmidt via rot6d_to_rotmat).  Column-concatenate the
                # first two matrix columns (rotmat_to_6d): the 6D convention
                # is [col0; col1] per joint.  (The old `R[..., :, :2].reshape`
                # interleaved rows [col0.x, col1.x, col0.y, ...], corrupting
                # every exported BVH for rotations away from identity - GT
                # through that line measured 460mm MPJPE.)
                R = rot6d_to_rotmat(torch.from_numpy(pose.reshape(-1, N_JOINTS, 6)).float())
                pose = rotmat_to_6d(R).reshape(-1, POSE_DIM).numpy()
                output_path = bvh_out / ("%s_%s.bvh" % (session_id, CONFIG_MODE_NAMES[config_value]))
                write_bvh(output_path, windows[0][4], pose_trans_to_motion(pose, trans), 1.0 / FPS)
                print("wrote %s" % output_path)
                # Test3 trajectory visualization: export the exact world-space
                # trajectories the traj_ATE metric is computed on (raw BVH frame,
                # meters).  Same concatenation order as the BVH above.
                traj_path = bvh_out / ("%s_%s_traj.npz" % (session_id, CONFIG_MODE_NAMES[config_value]))
                np.savez_compressed(
                    traj_path,
                    pred_trans_world=trans.astype(np.float32),
                    gt_trans_world=gt_trans.astype(np.float32),
                )
                print("wrote %s" % traj_path)
    metrics_out = args.metrics_out
    if metrics_out is None:
        metrics_out = model_root / "metrics" / ("%s.json" % args.split)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "modal": str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1)),
               "contact_method": contact_method,
               "split": args.split, "sample_steps": sample_steps, "metrics": all_values}
    if "V2M" in all_values:
        payload["v2t"] = {name: all_values["V2M"][name] for name in ("T_mae", "T_rmse", "T_corr")}
    metrics_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
