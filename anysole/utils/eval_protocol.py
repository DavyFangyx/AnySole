"""F0a evaluation protocol (fix_plan_v2.md §2): session-level, per-part, global,
temporal, contact, pressure, and robustness metrics.

Complements eval.py: eval.py keeps the frozen per-window metrics JSON
(``metrics/<split>.json``); this module adds ``metrics/<split>_fseries.json``
with the F-series protocol numbers, which every step F0..F9 reports:

- 局部姿态: MPJPE / PA-MPJPE per 9 native SMPL-24 parts
  (root/torso/headneck/l_arm/r_arm/l_leg/r_leg/l_foot/r_foot), plus upper/lower,
  ankle-feet and hands aggregates.  PA-MPJPE uses one per-frame
  similarity Procrustes alignment over all 24 joints (the eval.py 口径) and
  reads the part errors from the aligned points.
- 全局: metrics.py 的 100-frame W-MPJPE/WA-MPJPE、root ATE/RTE
  (root trajectory error normalized by the mean GT per-frame displacement
  length), yaw_abs_deg / yaw_drift_deg (root heading error / accumulated
  drift; native SMPL local +Z is the fixed forward axis).
- 时序: metrics.py 的 acceleration error / jerk，另保留 seam_jump_mm at
  window boundaries (GT frame-to-frame at the same seams as reference).
- 接触: contact F1 (soft_contact_from_keypoints > 0.5 vs contact_gt) and
  contact-period foot slide in mm/帧, both on predicted-contact frames
  (train._evaluate 口径) and GT-contact frames.
- 压力输出 (V2M only): total-force R², per-foot CoP error (grid units,
  cop_from_grid 口径), per-frame 96-cell Pearson correlation.
- T2M 上半身: upper-body acceleration-magnitude distribution error (|mean
  pred - mean gt|), joint-limit violation rate (elbow/knee bone angle < 15°).

Robustness set (VT2M only, F0 起报告): two synthetic degradations per session
— a contiguous 20-40% span of V frames zeroed (``robust_vdrop``) and the same
for the T stream (``robust_tdrop``); the span is deterministic per
(session, seed).

Inference is session-level: eval windows of one session are non-overlapping
(stride == window_length), so their concatenation is the full-session
prediction.  The protocol skips itself when the dataset uses overlapping
windows (E6.4 continuation eval) — those are covered again from F3 on.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from anysole.data.dataset import collate_windows
from anysole.data.smpl_io import (
    pelvis_to_smpl_trans,
    smpl24_pose6d_to_poses,
    smpl_vertices_from_archive_params,
)
from anysole.eval import _sample_x_t_init, _session_window_groups, _tactile_corr
from anysole.utils.geometry import f2_to_world, fk_pose6d, rot6d_to_rotmat
from anysole.utils.losses import soft_contact_from_keypoints
from anysole.utils.metrics import (
    matrix_rotation_error_degrees,
    mean_point_error,
    pa_mpjpe,
    pelvis_align,
    foot_sliding,
    root_trajectory_metrics,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)
from anysole.train import condition_inputs, move_batch
from anysole.types import (
    ANKLE_FOOT_JOINTS,
    CONFIG_MODE_NAMES,
    CONFIG_V,
    CONFIG_VT,
    ELBOW_ANGLE_TRIPLES,
    FPS,
    FOOT_JOINTS,
    HAND_JOINTS,
    KNEE_ANGLE_TRIPLES,
    LOWER_JOINTS,
    N_JOINTS,
    PART_JOINTS,
    PART_NAMES,
    UPPER_JOINTS,
)

# The imported groups remain re-exported for existing probe scripts.  Their
# single source of truth is the named SMPL-24 protocol in anysole.types.
JOINT_LIMIT_RAD = math.radians(15.0)  # bone angle below this is anatomically impossible


def _pa_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame similarity Procrustes alignment over the native 24 joints."""
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
    return aligned.view_as(pred)


def _circ_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _root_rotmats(pose: torch.Tensor) -> torch.Tensor:
    """(T, POSE_DIM) native SMPL pose -> world root rotations."""
    return rot6d_to_rotmat(pose[None].reshape(1, -1, N_JOINTS, 6))[0, :, 0]


def _pick_forward_axis(dataset, device: torch.device) -> int:
    """Return the native SMPL local forward axis (+Z).

    Inferring +X/+Z from locomotion is not a valid protocol operation:
    side-steps need not follow body yaw, and an F2 dataset stores yaw-removed
    root *tilt* in ``pose_gt`` so the inference can inspect the wrong
    representation altogether.  The SMPL convention is fixed and checked by
    ``smoke_f2_roundtrip.py``.  Arguments remain for call-site compatibility.
    """
    del dataset, device
    return 2


def _cop_grid(values: torch.Tensor) -> torch.Tensor:
    """Per-foot CoP in grid units from 96-dim rows (left48, right48) — the
    torch twin of pressure.cop_from_grid (x along the 4 rows / 3, y along the
    12 heel-to-toe cols / 11)."""
    grid = values.reshape(-1, 2, 4, 12).clamp(min=0.0)
    force = grid.sum(dim=(2, 3))
    rows = torch.arange(4, device=values.device, dtype=torch.float32)
    cols = torch.arange(12, device=values.device, dtype=torch.float32)
    cop_x = (grid.sum(dim=3) * rows).sum(dim=2) / force.clamp_min(1e-6) / 3.0
    cop_y = (11.0 - (grid.sum(dim=2) * cols).sum(dim=2) / force.clamp_min(1e-6)) / 11.0
    cop = torch.stack([cop_x, cop_y], dim=-1)
    return torch.where((force > 0).unsqueeze(-1), cop, torch.zeros_like(cop))


def _bone_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Angle at joint b between the bones (a-b) and (b-c), in radians."""
    v1 = F.normalize(a - b, dim=-1)
    v2 = F.normalize(c - b, dim=-1)
    cos = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


class _Accum:
    """Per-name sums/counts; means() returns the protocol dict."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def add(self, name: str, values: torch.Tensor, scale: float = 1.0) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(values.double().sum().item()) * scale
        self.counts[name] = self.counts.get(name, 0.0) + float(values.numel())

    def add_scalar(self, name: str, value: float, weight: float = 1.0) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * weight
        self.counts[name] = self.counts.get(name, 0.0) + weight

    def means(self) -> Dict[str, float]:
        return {name: self.sums[name] / max(self.counts[name], 1e-12) for name in self.sums}


def _session_metrics(seq: dict, tw: int, forward_axis: int, config_value: int,
                     accum: _Accum, contact: Dict[str, int]) -> None:
    """All protocol metrics of one session's full-sequence prediction."""
    T = seq["pred_pose"].shape[0]
    kp_gt = seq["kp_gt"]
    pred_kp = fk_pose6d(
        seq["pred_pose"][None], seq["pred_trans"][None], seq["offsets"], seq["parents"]
    )[0]

    # ---- canonical metrics.py definitions ----
    pred_np = pred_kp.detach().cpu().numpy()
    gt_np = kp_gt.detach().cpu().numpy()
    times_np = np.arange(T, dtype=np.float64) / float(FPS)
    pred_aligned, _ = pelvis_align(pred_np)
    gt_aligned, _ = pelvis_align(gt_np)
    accum.add_scalar("mpjpe_mm", float(mean_point_error(pred_aligned, gt_aligned, scale=1000.0).mean()), T)
    accum.add_scalar("pa_mpjpe_mm", float(pa_mpjpe(pred_np, gt_np).mean()), T)
    w_mpjpe, wa_mpjpe = windowed_world_mpjpe(pred_np, gt_np, window=100)
    accum.add_scalar("w_mpjpe100_mm", w_mpjpe, T)
    accum.add_scalar("wa_mpjpe100_mm", wa_mpjpe, T)
    root = root_trajectory_metrics(pred_np[:, 0], gt_np[:, 0])
    accum.add_scalar("root_ate_mm", root["root_ate_mm"], T)
    accum.add_scalar("root_rte_percent", root["root_rte_percent"], T)
    temporal = temporal_metrics(pred_np, gt_np, times_np)
    for key in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3"):
        accum.add_scalar(key, float(temporal[key]), T)
    pred_rot = rot6d_to_rotmat(seq["pred_pose"].reshape(T, N_JOINTS, 6)).detach().cpu().numpy()
    gt_rot = rot6d_to_rotmat(seq["gt_pose"].reshape(T, N_JOINTS, 6)).detach().cpu().numpy()
    accum.add_scalar("mpjae_deg", float(matrix_rotation_error_degrees(pred_rot, gt_rot).mean()), T)
    betas_np = seq["betas"].detach().cpu().numpy()
    pred_pose6d = seq["pred_pose"].detach().cpu().numpy()
    gt_pose6d = seq["gt_pose"].detach().cpu().numpy()
    pred_poses = smpl24_pose6d_to_poses(pred_pose6d)
    gt_poses = smpl24_pose6d_to_poses(gt_pose6d)
    pred_model_trans = pelvis_to_smpl_trans(
        pred_pose6d, seq["pred_trans"].detach().cpu().numpy(), betas_np
    )
    gt_model_trans = pelvis_to_smpl_trans(
        gt_pose6d, seq["gt_trans"].detach().cpu().numpy(), betas_np
    )
    pred_vertices = smpl_vertices_from_archive_params(pred_poses, pred_model_trans, betas_np)
    gt_vertices = smpl_vertices_from_archive_params(gt_poses, gt_model_trans, betas_np)
    _, pred_vertices_aligned = pelvis_align(pred_np, pred_vertices)
    _, gt_vertices_aligned = pelvis_align(gt_np, gt_vertices)
    accum.add_scalar(
        "pve_mm",
        float(mean_point_error(pred_vertices_aligned, gt_vertices_aligned, scale=1000.0).mean()),
        T,
    )
    accum.add_scalar("shape_vertex_std_mm", shape_vertex_std(pred_vertices), T)
    foot_value, foot_count = foot_sliding(pred_vertices, gt_vertices, times_np)
    if foot_count:
        accum.add_scalar("foot_sliding_mm", foot_value, foot_count)

    # ---- retained AnySole part diagnostics ----
    err = torch.linalg.vector_norm(pred_kp - kp_gt, dim=-1)  # (T, 24)
    for pname, joints in zip(PART_NAMES, PART_JOINTS):
        accum.add("MPJPE_part_%s" % pname, err[:, joints].mean(dim=-1), 1000.0)
    accum.add("MPJPE_upper", err[:, UPPER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_lower", err[:, LOWER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_anklefoot", err[:, ANKLE_FOOT_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_hands", err[:, HAND_JOINTS].mean(dim=-1), 1000.0)

    aligned = _pa_align(pred_kp, kp_gt)
    pa_err = torch.linalg.vector_norm(aligned - kp_gt, dim=-1)
    for pname, joints in zip(PART_NAMES, PART_JOINTS):
        accum.add("PA-MPJPE_part_%s" % pname, pa_err[:, joints].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_upper", pa_err[:, UPPER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_lower", pa_err[:, LOWER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_anklefoot", pa_err[:, ANKLE_FOOT_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_hands", pa_err[:, HAND_JOINTS].mean(dim=-1), 1000.0)

    # ---- retained AnySole orientation diagnostics ----
    R_p = _root_rotmats(seq["pred_pose"])
    R_g = _root_rotmats(seq["gt_pose"])
    yaw_p = torch.atan2(R_p[:, :, forward_axis][:, 0], R_p[:, :, forward_axis][:, 2])
    yaw_g = torch.atan2(R_g[:, :, forward_axis][:, 0], R_g[:, :, forward_axis][:, 2])
    accum.add("yaw_abs_deg", _circ_diff(yaw_p, yaw_g).abs() * 180.0 / math.pi)
    accum.add(
        "yaw_drift_deg",
        _circ_diff(yaw_p - yaw_p[0], yaw_g - yaw_g[0]).abs() * 180.0 / math.pi,
    )

    # ---- retained AnySole temporal diagnostics ----
    if T >= 3:
        a_pred = (pred_kp[2:] - 2.0 * pred_kp[1:-1] + pred_kp[:-2]) * FPS**2
        a_gt = (kp_gt[2:] - 2.0 * kp_gt[1:-1] + kp_gt[:-2]) * FPS**2
        accum.add("accel_mag_ms2", torch.linalg.vector_norm(a_pred, dim=-1))
        accum.add("accel_mag_gt_ms2", torch.linalg.vector_norm(a_gt, dim=-1))
        accum.add("accel_mag_upper_ms2", torch.linalg.vector_norm(a_pred[:, UPPER_JOINTS], dim=-1))
        accum.add("accel_mag_upper_gt_ms2", torch.linalg.vector_norm(a_gt[:, UPPER_JOINTS], dim=-1))
    if T > tw:
        seams = torch.arange(tw, T, tw, device=pred_kp.device)
        accum.add("seam_jump_mm", torch.linalg.vector_norm(pred_kp[seams] - pred_kp[seams - 1], dim=-1), 1000.0)
        accum.add("seam_jump_gt_mm", torch.linalg.vector_norm(kp_gt[seams] - kp_gt[seams - 1], dim=-1), 1000.0)

    # ---- contact ----
    pred_contact = soft_contact_from_keypoints(
        pred_kp[None], seq["floor_y"].reshape(1)
    )[0] > 0.5  # (T, 2)
    gt_contact = seq["contact_gt"] > 0.5
    contact["tp"] += int((pred_contact & gt_contact).sum().item())
    contact["fp"] += int((pred_contact & ~gt_contact).sum().item())
    contact["fn"] += int((~pred_contact & gt_contact).sum().item())
    contact["tn"] += int((~pred_contact & ~gt_contact).sum().item())
    # ---- joint limits (elbow/knee bone angle on the PREDICTED pose: the
    # plausibility check is that generated upper bodies don't fold joints
    # past what anatomy allows) ----
    elbow_angles = [
        _bone_angle(pred_kp[:, shoulder], pred_kp[:, elbow], pred_kp[:, wrist])
        for shoulder, elbow, wrist in ELBOW_ANGLE_TRIPLES
    ]
    viol = torch.stack(
        [angle < JOINT_LIMIT_RAD for angle in elbow_angles], dim=-1
    ).any(dim=-1).float()
    accum.add("joint_limit_viol_elbow", viol)
    knee_angles = [
        _bone_angle(pred_kp[:, hip], pred_kp[:, knee], pred_kp[:, ankle])
        for hip, knee, ankle in KNEE_ANGLE_TRIPLES
    ]
    viol_knee = torch.stack(
        [angle < JOINT_LIMIT_RAD for angle in knee_angles], dim=-1
    ).any(dim=-1).float()
    accum.add("joint_limit_viol_knee", viol_knee)

    # ---- pressure output quality (V2M only: T input is zeroed, so these
    # measure vision-to-tactile generation).  F5 part9 drops the aux heads
    # and skips these rows. ----
    if config_value == CONFIG_V and seq.get("pressure_pred") is not None:
        pressure_pred = seq["pressure_pred"]
        pressure_gt = seq["pressure_gt"]
        diff = pressure_pred - pressure_gt
        accum.add("T_mae", diff.abs())
        accum.add("T_mse", diff.square())
        f_pred_foot = pressure_pred.reshape(-1, 2, 48).sum(dim=-1)
        f_gt_foot = pressure_gt.reshape(-1, 2, 48).sum(dim=-1)
        force_diff = f_pred_foot - f_gt_foot
        accum.add("pressure_force_mae", force_diff.abs())
        accum.add("pressure_force_mse", force_diff.square())
        f_pred = f_pred_foot.sum(dim=-1)
        f_gt = f_gt_foot.sum(dim=-1)
        ss_res = float(((f_pred - f_gt) ** 2).sum().item())
        ss_tot = float(((f_gt - f_gt.mean()) ** 2).sum().item())
        accum.add_scalar("pressure_force_r2", 1.0 - ss_res / max(ss_tot, 1e-12), float(T))
        cop_p = _cop_grid(pressure_pred)
        cop_g = _cop_grid(pressure_gt)
        cop_err = torch.linalg.vector_norm(cop_p - cop_g, dim=-1)
        accum.add("pressure_cop_error_left", cop_err[:, 0])
        accum.add("pressure_cop_error_right", cop_err[:, 1])
        accum.add("pressure_cop_error_mean", cop_err)
        corr, corr_valid = _tactile_corr(pressure_pred, pressure_gt)
        if bool(corr_valid.any()):
            accum.add("T_corr", corr[corr_valid])


def _degrade_inputs(v_feat, t_raw, t_phys, t_s2m, tw: int, kind: str, rng: np.random.RandomState):
    """Zero one contiguous 20-40% span of frames on the V or T stream
    (robustness set).  Windows are non-overlapping, so window w covers session
    frames [w*tw, (w+1)*tw)."""
    n_windows = v_feat.shape[0]
    n_frames = n_windows * tw
    frac = float(rng.uniform(0.2, 0.4))
    start = int(rng.uniform(0.0, max(n_frames - int(frac * n_frames), 1)))
    end = start + int(frac * n_frames)
    frames = torch.arange(n_frames).view(n_windows, tw)
    mask = ((frames >= start) & (frames < end)).to(device=v_feat.device)
    v, tr, tp, ts = v_feat.clone(), t_raw.clone(), t_phys.clone(), t_s2m.clone()
    if kind == "v":
        v[mask] = 0.0
    else:
        tr[mask] = 0.0
        tp[mask] = 0.0
        ts[mask] = 0.0
    return v, tr, tp, ts

def _evaluate_config(
    dataset, model, device, config_value: int, regress_mode: bool, diffusion,
    sample_steps: int, warm_start: bool, tw: int, forward_axis: int,
    degrade: Optional[str] = None, seed: int = 0,
    v2t_out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Full-session inference for one conditioning config, then all protocol
    metrics.  ``degrade`` in ("v", "t") adds the robustness span dropout."""
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel.
    v_hmr_mode = str(getattr(model, "v_input", "hrnet")) == "hmr_gvhmr"
    # F2a: heading/tilt representation.
    f2_repr = bool(getattr(model, "f2_repr", False))
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    groups = _session_window_groups(dataset)
    session_by_id = {s["session_id"]: s for s in dataset.sessions}
    with torch.inference_mode():
        for session_id, idxs in groups.items():
            raw = [dataset[i] for i in idxs]
            batch = move_batch(collate_windows(raw), device)
            bsz = batch["pose_gt"].shape[0]
            config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            if degrade is not None:
                rng = np.random.RandomState((seed * 1000003) ^ zlib.crc32(str(session_id).encode()))
                v_feat, t_raw, t_phys, t_s2m = _degrade_inputs(
                    v_feat, t_raw, t_phys, t_s2m, tw, degrade, rng
                )
            v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
            if regress_mode:
                out = model(v_feat, t_raw, t_phys, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                pred_pose = out["x0_hat"]
            else:
                cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                        "T_s2m": t_s2m, "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                        "config_id": config_id,
                        "session_id": batch.get("session_id")}
                prior = model.pose_head.pose_mean.view(1, 1, -1).expand(bsz, tw, -1) if warm_start else None
                x_T = _sample_x_t_init(model, bsz, tw, device, prior=prior)
                pred_pose = diffusion.ddim_sample_loop(
                    model, x_T=x_T, tau_related_kwargs=cond,
                    steps=sample_steps, eta=0.0, device=device)
                tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
                out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            anchor = batch["trans_anchor"][:, None, :]
            if f2_repr:
                # F2a: world pose/trans recovered from the tilt pose + heading
                # trajectory before the protocol metrics (same FK scope).
                pred_pose_w, pred_trans_w = f2_to_world(
                    pred_pose, out["v_hat"], batch["psi_anchor"], batch["trans_anchor"]
                )
                gt_pose_w, _ = f2_to_world(
                    batch["pose_gt"], batch["traj_gt_f2"], batch["psi_anchor"],
                    batch["trans_anchor"],
                )
                seq = {
                    "pred_pose": pred_pose_w.reshape(-1, pred_pose_w.shape[-1]),
                    "pred_trans": pred_trans_w.reshape(-1, 3),
                    "gt_pose": gt_pose_w.reshape(-1, gt_pose_w.shape[-1]),
                    "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3),
                    "kp_gt": batch["kp_gt"].reshape(-1, N_JOINTS, 3),
                    "contact_gt": batch["contact_gt"].reshape(-1, 2),
                    "floor_y": batch["floor_y"][0],
                    "pressure_gt": batch["T_raw"].reshape(-1, batch["T_raw"].shape[-1]),
                    "pressure_pred": (out["pressure_hat"].reshape(-1, out["pressure_hat"].shape[-1])
                                      if out.get("pressure_hat") is not None else None),
                    "offsets": batch["offsets"][0],
                    "parents": batch["parents"][0],
                    "betas": batch["betas"][0],
                }
            else:
                seq = {
                    "pred_pose": pred_pose.reshape(-1, pred_pose.shape[-1]),
                    "pred_trans": (out["trans_hat"] + anchor).reshape(-1, 3),
                    "gt_pose": batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1]),
                    "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3),
                    "kp_gt": batch["kp_gt"].reshape(-1, N_JOINTS, 3),
                    "contact_gt": batch["contact_gt"].reshape(-1, 2),
                    "floor_y": batch["floor_y"][0],
                    "pressure_gt": batch["T_raw"].reshape(-1, batch["T_raw"].shape[-1]),
                    "pressure_pred": (out["pressure_hat"].reshape(-1, out["pressure_hat"].shape[-1])
                                      if out.get("pressure_hat") is not None else None),
                    "offsets": batch["offsets"][0],
                    "parents": batch["parents"][0],
                    "betas": batch["betas"][0],
                }
            _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
            if (config_value == CONFIG_V and v2t_out_dir is not None
                    and seq.get("pressure_pred") is not None):
                pred_kp = fk_pose6d(
                    seq["pred_pose"][None], seq["pred_trans"][None],
                    seq["offsets"], seq["parents"],
                )[0]
                pred_contact = soft_contact_from_keypoints(
                    pred_kp[None], seq["floor_y"].reshape(1)
                )[0].detach().cpu().numpy() > 0.5
                gt_contact = (seq["contact_gt"].detach().cpu().numpy() > 0.5)
                session = session_by_id[session_id]
                n_frames = int(np.asarray(session["V_feat"]).shape[0])
                pressure_pred = np.zeros((n_frames, seq["pressure_pred"].shape[-1]), dtype=np.float32)
                pressure_gt = np.zeros_like(pressure_pred)
                contact_pred_full = np.zeros((n_frames, 2), dtype=np.uint8)
                contact_gt_full = np.zeros_like(contact_pred_full)
                seen = np.zeros((n_frames,), dtype=bool)
                cursor = 0
                for sample in raw:
                    start = int(sample["frame_start"])
                    length = min(tw, n_frames - start)
                    if length <= 0:
                        cursor += tw
                        continue
                    end = start + length
                    pressure_pred[start:end] = seq["pressure_pred"][cursor:cursor + length].detach().cpu().numpy()
                    pressure_gt[start:end] = seq["pressure_gt"][cursor:cursor + length].detach().cpu().numpy()
                    contact_pred_full[start:end] = pred_contact[cursor:cursor + length].astype(np.uint8)
                    contact_gt_full[start:end] = gt_contact[cursor:cursor + length].astype(np.uint8)
                    seen[start:end] = True
                    cursor += tw
                fake_mask = np.asarray(session.get("fake_mask", np.zeros(n_frames, dtype=np.uint8))).reshape(-1)
                valid = seen & ((fake_mask == 0) if len(fake_mask) == n_frames else True)
                v2t_out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    v2t_out_dir / ("%s_V2T.npz" % session_id),
                    pressure_pred=pressure_pred,
                    pressure_gt=pressure_gt,
                    contact_pred=contact_pred_full,
                    contact_gt=contact_gt_full,
                    frame_indices=np.arange(n_frames, dtype=np.int64),
                    valid_mask=valid.astype(np.uint8),
                    target_fps=np.asarray(FPS, dtype=np.float32),
                    mode=np.asarray("V2T"),
                    pressure_source_grid=np.asarray("4x12_per_foot"),
                    comparison_pressure_grid=np.asarray("31x11_per_foot_linear_resample"),
                )
    result = accum.means()
    p = contact["tp"] / max(contact["tp"] + contact["fp"], 1)
    r = contact["tp"] / max(contact["tp"] + contact["fn"], 1)
    result["contact_f1"] = 2.0 * p * r / max(p + r, 1e-8)
    result["contact_acc"] = (contact["tp"] + contact["tn"]) / max(
        contact["tp"] + contact["fp"] + contact["fn"] + contact["tn"], 1
    )
    result["contact_recall"] = contact["tp"] / max(contact["tp"] + contact["fn"], 1)
    result["air_recall"] = contact["tn"] / max(contact["tn"] + contact["fp"], 1)
    result["contact_balanced_acc"] = 0.5 * (
        result["contact_recall"] + result["air_recall"]
    )
    if "T_mse" in result:
        result["T_rmse"] = math.sqrt(result.pop("T_mse"))
    if "pressure_force_mse" in result:
        result["pressure_force_rmse"] = math.sqrt(result.pop("pressure_force_mse"))
    if "accel_mag_upper_ms2" in result:
        result["accel_dist_err_upper_ms2"] = abs(
            result["accel_mag_upper_ms2"] - result["accel_mag_upper_gt_ms2"]
        )
    return result


def run_protocol(
    checkpoint: dict,
    config: dict,
    model,
    dataset,
    device: torch.device,
    *,
    checkpoint_path: str,
    split: str,
    config_values: List[int],
    regress_mode: bool,
    diffusion,
    sample_steps: int,
    warm_start: bool,
    seed: int,
    robustness: bool,
    contact_method: str,
    tw: int,
    out_path: Path,
    v2t_out_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the full F0a protocol and write metrics/<split>_fseries.json."""
    if dataset.stride != dataset.window_length:
        print("protocol skipped: dataset stride %d != window %d (E6.4 continuation eval)"
              % (dataset.stride, dataset.window_length))
        return {}
    forward_axis = _pick_forward_axis(dataset, device)
    metrics: Dict[str, Dict[str, float]] = {}
    # Deterministic protocol runs.  Sigma-seed for diffusion models comes from
    # three runs with --protocol-seed 0/1/2; regression models have no
    # sampling randomness (dropout is off in eval mode).
    prev_cpu = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        for config_value in config_values:
            name = CONFIG_MODE_NAMES[config_value]
            metrics[name] = _evaluate_config(
                dataset, model, device, config_value, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis,
                v2t_out_dir=v2t_out_dir,
            )
            print("protocol %s: mpjpe=%.3fmm pa_mpjpe=%.3fmm w_mpjpe100=%.3fmm "
                  "contact_f1=%.4f"
                  % (name, metrics[name]["mpjpe_mm"], metrics[name]["pa_mpjpe_mm"],
                     metrics[name].get("w_mpjpe100_mm", float("nan")),
                     metrics[name].get("contact_f1", float("nan"))))
        if robustness and CONFIG_VT in config_values:
            metrics["robust_vdrop"] = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="v", seed=seed,
            )
            metrics["robust_tdrop"] = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="t", seed=seed,
            )
            print("protocol robust_vdrop: mpjpe=%.3fmm pa_mpjpe=%.3fmm contact_f1=%.4f"
                  % (metrics["robust_vdrop"]["mpjpe_mm"], metrics["robust_vdrop"]["pa_mpjpe_mm"],
                     metrics["robust_vdrop"]["contact_f1"]))
            print("protocol robust_tdrop: mpjpe=%.3fmm pa_mpjpe=%.3fmm contact_f1=%.4f"
                  % (metrics["robust_tdrop"]["mpjpe_mm"], metrics["robust_tdrop"]["pa_mpjpe_mm"],
                     metrics["robust_tdrop"]["contact_f1"]))
    finally:
        torch.set_rng_state(prev_cpu)
        if prev_cuda is not None:
            torch.cuda.set_rng_state_all(prev_cuda)
    payload = {
        "checkpoint": checkpoint_path,
        "modal": str(checkpoint.get("config", {}).get("modal", "anysolev1")),
        "contact_method": contact_method,
        "split": split,
        "protocol_seed": seed,
        "tw": tw,
        "sample_steps": sample_steps,
        "robustness": bool(robustness and CONFIG_VT in config_values),
        "forward_axis": "+Z" if forward_axis == 2 else "+X",
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical file (回填表口径) always reflects the latest run; a per-seed
    # copy is archived so the three --protocol-seed 0/1/2 runs can be diffed
    # for sigma-seed without overwriting each other.
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path = out_path.with_name("%s_seed%d.json" % (out_path.stem, seed))
    seed_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s (+ %s)" % (out_path, seed_path.name))
    return metrics
