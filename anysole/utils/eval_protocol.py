"""F0a evaluation protocol (fix_plan_v2.md §2): session-level, per-part, global,
temporal, contact, pressure, and robustness metrics.

eval.py drives this protocol; one run writes two same-source files per split:
``metrics/<split>.json`` (detail: per-part diagnostics plus robustness rows)
and ``metrics/<split>_brief.json`` (whole-body view: every group of the detail
file filtered to whole-body scalars, plus a ``V2T`` group in fourth position
whenever V2M carries tactile rows).  The F-series protocol numbers are what
every step F0..F9 reports:

- 局部姿态: MPJPE / PA-MPJPE per 9 native SMPL-24 parts
  (root/torso/headneck/l_arm/r_arm/l_leg/r_leg/l_foot/r_foot), plus upper/lower,
  ankle-feet and hands aggregates.  PA-MPJPE uses one per-frame
  similarity Procrustes alignment over all 24 joints (the eval.py 口径) and
  reads the part errors from the aligned points.
- 全局: metrics.py 的 100-frame W-MPJPE/WA-MPJPE、root ATE/RTE
  (root trajectory error normalized by the mean GT per-frame displacement
  length), root_orientation_deg / root_orientation_drift_deg (root heading
  error / accumulated drift, computed by the single implementation in
  metrics.py; native SMPL local +Z is the fixed forward axis).
- 时序: metrics.py 的 acceleration error / jerk，另保留 seam_jump_mm at
  window boundaries (GT frame-to-frame at the same seams as reference).
- 接触 (diagnostic only, never a formal metric): contact F1
  (soft_contact_from_keypoints > 0.5 vs contact_gt) on predicted-contact
  frames (train._evaluate 口径) and GT-contact frames.  No trained,
  explicitly exported contact prediction exists, so contact never enters the
  brief or cross-model tables (评估整改任务 01 §4.4).
- 足部滑动: 公共 foot_sliding_mm 为关节版 (SMPL ankle/foot 关节、GT 接触帧),
  由 metrics.py 的唯一实现计算; 顶点版保留为诊断 foot_sliding_vertex_mm。
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
from anysole.utils.geometry import f2_to_world, fk_pose6d, rot6d_to_rotmat
from anysole.utils.losses import soft_contact_from_keypoints
from anysole.utils.metrics import (
    foot_sliding_joints,
    foot_sliding_vertices,
    ground_penetration,
    matrix_rotation_error_degrees,
    mean_point_error,
    pa_mpjpe,
    pelvis_align,
    root_orientation_drift_deg,
    root_orientation_error_deg,
    root_trajectory_metrics,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)
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

# Whole-body brief (metrics/<split>_brief.json 口径): the detail file's groups
# filtered to the approved whole-body candidates.  Contact metrics are
# training/regularization diagnostics only (no trained, explicitly exported
# contact prediction exists) and never enter the brief; the V2T group is
# hierarchical (2026-09-27): brief keys are one representative per level,
# leaf keys are computed and stored in the detail row but excluded from the
# brief.  AnySole's own detail stays pressure-only (grid/force/CoP); the
# contact level (contact_smpl_mse/contact_smpl_bce) exists only in the
# cross-model solver (Baselines/utils/solver.py) for FPP-Net, whose contact
# head output and f6_soft GT support it.
V2T_LEVELS = {
    "grid": ("T_corr", "T_rmse", "T_mse", "T_mae"),
    "force": ("pressure_force_rmse", "pressure_force_r2", "pressure_force_mae"),
    "cop": ("pressure_cop_error_left", "pressure_cop_error_right",
            "pressure_cop_error_mean"),
}
V2T_BRIEF_NAMES = (
    "T_corr", "T_rmse",
    "pressure_force_rmse", "pressure_force_r2",
    "pressure_cop_error_left", "pressure_cop_error_right",
)
V2T_LEAF_NAMES = ("T_mse", "T_mae", "pressure_force_mae", "pressure_cop_error_mean")
V2T_NAMES = V2T_BRIEF_NAMES + V2T_LEAF_NAMES
# Whole-body candidates per motion group (one representative per metric
# family).  AnySole does not generate shape/surface (生成清单裁定 2026-09-27):
# the GT-beta pose-only PVE is a detail diagnostic, not a brief key, so
# pve_mm is absent here.  foot_sliding_mm is the public joint-based
# definition; the vertex-based variant is a detail-file diagnostic
# (foot_sliding_vertex_mm).
# Display (JSON) naming: internal snake_case keys are renamed to the classic
# uppercase style only at serialization time (2026-09-28 裁定：PA-MPJPE 等
# 旧式键名 + 主指标排最前); formulas and internal accumulation stay
# snake_case.  INVERSE_DISPLAY_NAMES lets backfill tools read files that
# already carry display names.
METRIC_DISPLAY_NAMES = {
    "pa_mpjpe_mm": "PA-MPJPE",
    "mpjpe_mm": "MPJPE",
    "mpjae_deg": "MPJRE",
    "w_mpjpe100_mm": "W-MPJPE100",
    "wa_mpjpe100_mm": "WA-MPJPE100",
    "root_ate_mm": "traj_ATE",
    "root_orientation_deg": "RootOrientation",
    "root_orientation_drift_deg": "RootOrientationDrift",
    "accel_error_m_s2": "AccelError",
    "foot_sliding_mm": "FootSliding",
    "jitter_gt_m_s3": "Jitter_gt",
    "jitter_pred_m_s3": "Jitter_pred",
    "global_mpjpe_mm": "GlobalMPJPE",
    "ground_pen_mm": "GroundPen",
    "ground_pen_frac": "GroundPenFrac",
}
INVERSE_DISPLAY_NAMES = {
    display: internal for internal, display in METRIC_DISPLAY_NAMES.items()
}
# Whole-body brief keys in their fixed JSON order — PA-MPJPE first so the
# primary metric is visible at a glance; doubles as the brief whitelist.
BRIEF_MOTION_KEYS = (
    "pa_mpjpe_mm", "mpjpe_mm", "global_mpjpe_mm", "mpjae_deg", "w_mpjpe100_mm",
    "root_ate_mm", "root_orientation_drift_deg", "foot_sliding_mm",
    "ground_pen_mm", "jitter_gt_m_s3", "jitter_pred_m_s3",
)
# Keys the detail file reports but that are diagnostics, not formal metrics
# (kept for training monitoring and history; excluded from brief by the
# whitelist above and declared here explicitly).
DIAGNOSTIC_KEYS = (
    "contact_f1", "contact_acc", "contact_recall", "air_recall",
    "contact_balanced_acc", "foot_sliding_vertex_mm", "shape_vertex_std_mm",
    # GT-beta pose-only PVE: AnySole 不生成 shape，此值仅诊断，不进 brief、
    # 不进正式表。
    "pve_mm",
    # GT ground-penetration reference: the floor estimate is the 5th
    # percentile of the GT joint heights, so GT penetrates slightly by
    # construction — its value contextualizes the prediction rows only.
    "ground_pen_gt_mm",
)
# Fixed group order of the brief file (json.dump sort_keys=False keeps it);
# V2T is always fourth, absent groups are skipped rather than invented.
BRIEF_GROUP_ORDER = ("T2M", "V2M", "VT2M", "V2T", "robust_vdrop", "robust_tdrop")

# Fixed top-level key order of both JSON files; extras (e.g. key_migration
# in backfilled files) follow after "metrics"-order keys if absent above.
PAYLOAD_KEY_ORDER = (
    "checkpoint", "modal", "contact_method", "split", "protocol_seed", "tw",
    "sample_steps", "robustness", "forward_axis",
    "diagnostic_fields", "v2t_leaf_keys", "key_migration", "metrics",
)


def _display_block(block: Dict[str, float]) -> Dict[str, float]:
    """Rename and reorder one metric block for JSON output.

    BRIEF_MOTION_KEYS fixes the leading order (PA-MPJPE first); the remaining
    keys (V2T tactile, per-part aggregates, diagnostics) keep a stable
    alphabetical order.  Internal accumulation stays snake_case.
    """
    ordered: list[tuple[str, float]] = []
    for key in BRIEF_MOTION_KEYS:
        if key in block:
            ordered.append((key, block[key]))
    ordered.extend(
        (key, block[key]) for key in sorted(set(block) - set(BRIEF_MOTION_KEYS))
    )
    return {METRIC_DISPLAY_NAMES.get(key, key): value for key, value in ordered}


def display_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Every group block through the display rename/order; group order fixed."""
    groups: Dict[str, Dict[str, float]] = {}
    for name in BRIEF_GROUP_ORDER:
        if name in metrics:
            groups[name] = _display_block(metrics[name])
    for name, block in metrics.items():
        if name not in groups:
            groups[name] = _display_block(block)
    return groups


def order_payload(payload: Dict) -> Dict:
    """Fixed top-level key order for human-readable JSON output."""
    ordered = {key: payload[key] for key in PAYLOAD_KEY_ORDER if key in payload}
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def summary_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Derive the brief (whole-body) view from the detail metric blocks.

    Groups are copied in BRIEF_GROUP_ORDER; each motion group keeps only
    BRIEF_MOTION_KEYS, renamed via METRIC_DISPLAY_NAMES and ordered with
    PA-MPJPE first; V2T is built from the V2M tactile row and contains
    exactly the brief keys (one representative per level; leaf keys stay
    detail-only) whenever V2M is present.
    """
    groups: Dict[str, Dict[str, float]] = {}
    for name in BRIEF_GROUP_ORDER:
        if name == "V2T":
            if "V2M" in metrics:
                v2t = {key: metrics["V2M"][key] for key in V2T_BRIEF_NAMES if key in metrics["V2M"]}
                if v2t:
                    groups[name] = v2t
            continue
        if name in metrics:
            block = metrics[name]
            groups[name] = {
                METRIC_DISPLAY_NAMES.get(key, key): block[key]
                for key in BRIEF_MOTION_KEYS
                if key in block
            }
    return groups


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
    # Global (unaligned) MPJPE over the protocol world frame — the only
    # definition untouched by any alignment, complementary to PA-MPJPE.
    accum.add_scalar("global_mpjpe_mm", float(mean_point_error(pred_np, gt_np, scale=1000.0).mean()), T)
    w_mpjpe, wa_mpjpe = windowed_world_mpjpe(pred_np, gt_np, window=100)
    accum.add_scalar("w_mpjpe100_mm", w_mpjpe, T)
    accum.add_scalar("wa_mpjpe100_mm", wa_mpjpe, T)
    root = root_trajectory_metrics(pred_np[:, 0], gt_np[:, 0])
    accum.add_scalar("root_ate_mm", root["root_ate_mm"], T)
    accum.add_scalar("root_rte_percent", root["root_rte_percent"], T)
    temporal = temporal_metrics(pred_aligned, gt_aligned, times_np)
    for key in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3"):
        accum.add_scalar(key, float(temporal[key]), T)
    pred_rot = rot6d_to_rotmat(seq["pred_pose"].reshape(T, N_JOINTS, 6)).detach().cpu().numpy()
    gt_rot = rot6d_to_rotmat(seq["gt_pose"].reshape(T, N_JOINTS, 6)).detach().cpu().numpy()
    accum.add_scalar("mpjae_deg", float(matrix_rotation_error_degrees(pred_rot[:, 1:22], gt_rot[:, 1:22]).mean()), T)
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
    # Public joint-based foot sliding (SMPL ankle/foot joints, GT-contact
    # frames); the vertex-based variant is a SMPL-only diagnostic and carries
    # a distinct name so the two definitions can never collide.
    foot_joint_value, foot_joint_count = foot_sliding_joints(
        pred_np[:, ANKLE_FOOT_JOINTS], gt_np[:, ANKLE_FOOT_JOINTS], times_np
    )
    if foot_joint_count:
        accum.add_scalar("foot_sliding_mm", foot_joint_value, foot_joint_count)
    foot_vert_value, foot_vert_count = foot_sliding_vertices(
        pred_vertices, gt_vertices, times_np
    )
    if foot_vert_count:
        accum.add_scalar("foot_sliding_vertex_mm", foot_vert_value, foot_vert_count)

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

    # ---- root orientation (the only implementation lives in metrics.py;
    # keys are the formal root_orientation_* names) ----
    R_p_np = _root_rotmats(seq["pred_pose"]).detach().cpu().numpy()
    R_g_np = _root_rotmats(seq["gt_pose"]).detach().cpu().numpy()
    accum.add(
        "root_orientation_deg",
        torch.from_numpy(
            root_orientation_error_deg(R_p_np, R_g_np, forward_axis=forward_axis)
        ),
    )
    accum.add(
        "root_orientation_drift_deg",
        torch.from_numpy(
            root_orientation_drift_deg(R_p_np, R_g_np, forward_axis=forward_axis)
        ),
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
    # ---- ground penetration (native Y-up; floor = the dataset's GT
    # 5th-percentile estimate).  The GT reference row penetrates slightly by
    # construction and is a diagnostic, not a comparison metric. ----
    floor_y = float(seq["floor_y"].reshape(1)[0].detach().cpu().item())
    accum.add_scalar("ground_pen_mm", ground_penetration(pred_np, floor_y)["ground_pen_mean_mm"], T)
    accum.add_scalar("ground_pen_frac", ground_penetration(pred_np, floor_y)["ground_pen_frac"], T)
    accum.add_scalar(
        "ground_pen_foot_mm",
        ground_penetration(pred_np[:, ANKLE_FOOT_JOINTS], floor_y)["ground_pen_mean_mm"],
        T,
    )
    accum.add_scalar("ground_pen_gt_mm", ground_penetration(gt_np, floor_y)["ground_pen_mean_mm"], T)
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
        from anysole.eval import _tactile_corr
        corr, corr_valid = _tactile_corr(pressure_pred, pressure_gt)
        if bool(corr_valid.any()):
            accum.add("T_corr", corr[corr_valid])


def _window_detail_rows(seq: dict, tw: int) -> dict:
    """Per-window pose errors of one session's full-sequence prediction.

    Windows mirror the eval loop's non-overlapping layout (stride == tw); the
    final window may be shorter when the session length is not a multiple of
    ``tw``.  Overall PA uses the canonical per-frame similarity alignment
    (metrics.py); part/region PA uses the protocol's ``_pa_align`` on the
    window-local frames, matching the pooled per-part diagnostics.  Keys are
    snake_case, matching the formulas.
    """
    T = seq["pred_pose"].shape[0]
    kp_gt = seq["kp_gt"]
    pred_kp = fk_pose6d(
        seq["pred_pose"][None], seq["pred_trans"][None], seq["offsets"], seq["parents"]
    )[0]
    pred_np = pred_kp.detach().cpu().numpy()
    gt_np = kp_gt.detach().cpu().numpy()
    part_keys = ["PA-MPJPE_part_%s" % pname for pname in PART_NAMES]
    region_keys = ["PA-MPJPE_upper", "PA-MPJPE_lower", "PA-MPJPE_anklefoot", "PA-MPJPE_hands"]
    rows: dict = {
        "frame_starts": [],
        "n_frames": [],
        "metrics": {key: [] for key in
                    ["pa_mpjpe_mm", "mpjpe_mm", "global_mpjpe_mm"] + part_keys + region_keys},
    }
    for start in range(0, T, tw):
        end = min(start + tw, T)
        p, g = pred_np[start:end], gt_np[start:end]
        rows["frame_starts"].append(int(start))
        rows["n_frames"].append(int(end - start))
        rows["metrics"]["pa_mpjpe_mm"].append(float(pa_mpjpe(p, g).mean()))
        p_al, _ = pelvis_align(p)
        g_al, _ = pelvis_align(g)
        rows["metrics"]["mpjpe_mm"].append(float(mean_point_error(p_al, g_al, scale=1000.0).mean()))
        rows["metrics"]["global_mpjpe_mm"].append(float(mean_point_error(p, g, scale=1000.0).mean()))
        pa_w = _pa_align(pred_kp[start:end], kp_gt[start:end])
        pa_err_w = torch.linalg.vector_norm(pa_w - kp_gt[start:end], dim=-1)
        for pname, joints in zip(PART_NAMES, PART_JOINTS):
            rows["metrics"]["PA-MPJPE_part_%s" % pname].append(
                float(pa_err_w[:, joints].mean() * 1000.0))
        rows["metrics"]["PA-MPJPE_upper"].append(float(pa_err_w[:, UPPER_JOINTS].mean() * 1000.0))
        rows["metrics"]["PA-MPJPE_lower"].append(float(pa_err_w[:, LOWER_JOINTS].mean() * 1000.0))
        rows["metrics"]["PA-MPJPE_anklefoot"].append(float(pa_err_w[:, ANKLE_FOOT_JOINTS].mean() * 1000.0))
        rows["metrics"]["PA-MPJPE_hands"].append(float(pa_err_w[:, HAND_JOINTS].mean() * 1000.0))
    return rows


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
    dump_detail: bool = False,
) -> Tuple[Dict[str, float], dict, List[dict]]:
    """Full-session inference for one conditioning config, then all protocol
    metrics.  ``degrade`` in ("v", "t") adds the robustness span dropout.

    With ``dump_detail`` the per-session rows and per-window pose errors are
    collected alongside the pooled result (the pooled accumulation sequence is
    unchanged, so the pooled numbers are bit-identical either way).  Returns
    ``(result, window_rows, session_rows)``.
    """
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel.
    v_hmr_mode = str(getattr(model, "v_input", "hrnet")) == "hmr_gvhmr"
    # F2a: heading/tilt representation.
    f2_repr = bool(getattr(model, "f2_repr", False))
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    window_rows = {"session_ids": [], "frame_starts": [], "n_frames": [], "metrics": {}}
    session_rows: List[dict] = []
    from anysole.eval import _sample_x_t_init, _session_window_groups
    from anysole.train import condition_inputs, move_batch
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
            if dump_detail:
                # Per-session accumulation, merged into the pooled accum in the
                # same summation sequence as the direct path, so the pooled
                # result stays bit-identical with or without the dump.
                sess_accum = _Accum()
                sess_contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                _session_metrics(seq, tw, forward_axis, config_value, sess_accum, sess_contact)
                for name, value in sess_accum.sums.items():
                    accum.sums[name] = accum.sums.get(name, 0.0) + value
                    accum.counts[name] = accum.counts.get(name, 0.0) + sess_accum.counts[name]
                for key in contact:
                    contact[key] += sess_contact[key]
                sess_metrics = sess_accum.means()
                p_c = sess_contact["tp"] / max(sess_contact["tp"] + sess_contact["fp"], 1)
                r_c = sess_contact["tp"] / max(sess_contact["tp"] + sess_contact["fn"], 1)
                sess_metrics["contact_f1"] = 2.0 * p_c * r_c / max(p_c + r_c, 1e-8)
                sess_metrics["contact_acc"] = (sess_contact["tp"] + sess_contact["tn"]) / max(
                    sess_contact["tp"] + sess_contact["fp"] + sess_contact["fn"] + sess_contact["tn"], 1)
                sess_metrics["contact_recall"] = sess_contact["tp"] / max(sess_contact["tp"] + sess_contact["fn"], 1)
                sess_metrics["air_recall"] = sess_contact["tn"] / max(sess_contact["tn"] + sess_contact["fp"], 1)
                sess_metrics["contact_balanced_acc"] = 0.5 * (
                    sess_metrics["contact_recall"] + sess_metrics["air_recall"])
                session_rows.append({
                    "session_id": str(session_id),
                    "n_frames": int(seq["pred_pose"].shape[0]),
                    "metrics": sess_metrics,
                })
                w_rows = _window_detail_rows(seq, tw)
                window_rows["session_ids"].extend([str(session_id)] * len(w_rows["frame_starts"]))
                window_rows["frame_starts"].extend(w_rows["frame_starts"])
                window_rows["n_frames"].extend(w_rows["n_frames"])
                for key, values in w_rows["metrics"].items():
                    window_rows["metrics"].setdefault(key, []).extend(values)
            else:
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
        result["T_rmse"] = math.sqrt(result["T_mse"])
    if "pressure_force_mse" in result:
        result["pressure_force_rmse"] = math.sqrt(result.pop("pressure_force_mse"))
    if "accel_mag_upper_ms2" in result:
        result["accel_dist_err_upper_ms2"] = abs(
            result["accel_mag_upper_ms2"] - result["accel_mag_upper_gt_ms2"]
        )
    return result, window_rows, session_rows


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
    brief_out: Optional[Path] = None,
    v2t_out_dir: Optional[Path] = None,
    dump_detail: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run the full F0a protocol and write metrics/<split>.json (detail) and,
    when brief_out is given, metrics/<split>_brief.json (whole-body brief).

    ``dump_detail`` additionally writes the per-window pose errors and
    per-session metric rows next to ``out_path``
    (window_detail_<split>.json / session_detail_<split>.json, snake_case keys);
    the pooled files are bit-identical either way.
    """
    if dataset.stride != dataset.window_length:
        print("protocol skipped: dataset stride %d != window %d (E6.4 continuation eval)"
              % (dataset.stride, dataset.window_length))
        return {}
    forward_axis = _pick_forward_axis(dataset, device)
    metrics: Dict[str, Dict[str, float]] = {}
    window_conditions: Dict[str, dict] = {}
    session_conditions: Dict[str, dict] = {}
    # Deterministic protocol runs.  Sigma-seed for diffusion models comes from
    # three runs with --protocol-seed 0/1/2; regression models have no
    # sampling randomness (dropout is off in eval mode).
    prev_cpu = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        for config_value in config_values:
            name = CONFIG_MODE_NAMES[config_value]
            metrics[name], window_rows, session_rows = _evaluate_config(
                dataset, model, device, config_value, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis,
                v2t_out_dir=v2t_out_dir,
                dump_detail=dump_detail,
            )
            if dump_detail:
                window_conditions[name] = window_rows
                session_conditions[name] = session_rows
            print("protocol %s: mpjpe=%.3fmm pa_mpjpe=%.3fmm w_mpjpe100=%.3fmm"
                  % (name, metrics[name]["mpjpe_mm"], metrics[name]["pa_mpjpe_mm"],
                     metrics[name].get("w_mpjpe100_mm", float("nan"))))
        if robustness and CONFIG_VT in config_values:
            metrics["robust_vdrop"], _, _ = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="v", seed=seed,
            )
            metrics["robust_tdrop"], _, _ = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="t", seed=seed,
            )
            print("protocol robust_vdrop: mpjpe=%.3fmm pa_mpjpe=%.3fmm"
                  % (metrics["robust_vdrop"]["mpjpe_mm"], metrics["robust_vdrop"]["pa_mpjpe_mm"],
                     ))
            print("protocol robust_tdrop: mpjpe=%.3fmm pa_mpjpe=%.3fmm"
                  % (metrics["robust_tdrop"]["mpjpe_mm"], metrics["robust_tdrop"]["pa_mpjpe_mm"],
                     ))
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
        # Detail-file keys that are training/diagnostic outputs, not formal
        # metrics; they must not be consumed as formal comparison candidates.
        "diagnostic_fields": sorted(
            {key for name, block in metrics.items() for key in block if key in DIAGNOSTIC_KEYS}
        ),
        # V2T leaf keys are formal computed metrics excluded from the brief
        # (one representative per level); their values live in the detail
        # V2M tactile row.
        "v2t_leaf_keys": sorted(V2T_LEAF_NAMES),
        "metrics": display_metrics(metrics),
    }
    payload = order_payload(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No sort_keys anywhere: PAYLOAD_KEY_ORDER fixes the top-level order and
    # BRIEF_MOTION_KEYS puts PA-MPJPE first in every motion block, so diffs
    # stay stable.
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if brief_out is not None:
        brief = dict(payload)
        brief["metrics"] = summary_metrics(metrics)
        brief_out.parent.mkdir(parents=True, exist_ok=True)
        brief_out.write_text(
            json.dumps(brief, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("wrote %s (+ %s)" % (out_path, brief_out.name))
    else:
        print("wrote %s" % out_path)
    if dump_detail:
        # Per-window / per-session detail dumps next to the pooled detail
        # file; keys stay snake_case (formula names).  Written compact
        # (no indent) — these files carry hundreds of thousands of values.
        common = {
            "checkpoint": checkpoint_path,
            "split": split,
            "tw": tw,
            "contact_method": contact_method,
            "dump_version": 1,
        }
        window_path = out_path.with_name("window_detail_%s.json" % split)
        window_path.write_text(
            json.dumps({**common, "conditions": window_conditions},
                       sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        session_path = out_path.with_name("session_detail_%s.json" % split)
        session_path.write_text(
            json.dumps({**common, "conditions": session_conditions},
                       sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("wrote detail dumps (+ %s, %s)" % (window_path.name, session_path.name))
    return metrics


# Grouped RGB-only protocol. Legacy diffusion evaluation above is retained.
VISUAL_EVAL_PROTOCOL = "anysole.visual-evaluation.v1"


def visual_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def project_so3(matrix):
    """Uniform chordal SO(3) mean: polar projection with det=+1 correction."""
    matrix = np.asarray(matrix, np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite rotation prediction")
    u, _, vh = np.linalg.svd(matrix)
    correction = np.broadcast_to(np.eye(3), matrix.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vh)
    return u @ correction @ vh


def integrate_visual_velocity(velocity, times, frame_ids, valid, edges):
    """v[i] ends at i; integrate once after aggregation; reset every segment."""
    velocity, times = np.asarray(velocity), np.asarray(times, np.float64)
    frame_ids, valid = np.asarray(frame_ids), np.asarray(valid, bool)
    edges = np.asarray(edges, bool)
    if (velocity.shape != (len(times), 3) or valid.shape != times.shape or
            frame_ids.shape != times.shape or edges.shape != (max(len(times)-1, 0),)):
        raise ValueError("trajectory axis/mask shapes disagree")
    result = np.full((len(times), 3), np.nan, np.float64)
    segment = np.full(len(times), -1, np.int64)
    number = -1
    for i in range(len(times)):
        if not valid[i]:
            continue
        if not np.isfinite(velocity[i]).all():
            raise ValueError("nonfinite valid velocity")
        connected = (i > 0 and valid[i-1] and edges[i-1] and frame_ids[i] == frame_ids[i-1]+1
                     and 0 < times[i]-times[i-1] <= .050000001)
        if connected:
            segment[i] = number
            result[i] = result[i-1] + velocity[i]*(times[i]-times[i-1])
        else:
            number += 1
            segment[i] = number
            result[i] = 0.
    return result, segment


def aggregate_visual_windows(windows):
    """Merge CPU window dictionaries on original (recording,camera,frame).

    Invalid target rows stay on the axis. Labels are optional and never decide
    prediction extent. Duplicate metadata/labels must agree. Source edges are
    reconstructed across window boundaries, with any observed false edge veto.
    """
    groups = {}
    truth_fields = ("motion_valid", "pressure_valid", "gt_joints", "gt_pressure",
                    "gt_pose6d", "gt_root", "gt_betas")
    for window in windows:
        key = (str(window["recording_id"]), int(window["camera_id"]))
        group = groups.setdefault(key, {"rows": {}, "edges": {}, "label_edges": {},
            **{name:window.get(name,"unknown") for name in ('subject','action','date')}})
        ids, times = _array(window["frame_id"]), _array(window["time_s"])
        valid = _array(window["valid"]).astype(bool)
        if times.shape != ids.shape or valid.shape != ids.shape:
            raise ValueError("window target axes disagree")
        for i, frame in enumerate(ids):
            if frame < 0:
                continue
            frame = int(frame)
            row = group["rows"].setdefault(frame, {"time_s": float(times[i]), "n": 0,
                                                   "rotation_sum": np.zeros((24,3,3)),
                                                   "pressure_sum": np.zeros(96), "velocity_sum": np.zeros(3)})
            if not np.isclose(row["time_s"], times[i], atol=1e-7, rtol=0):
                raise ValueError("duplicate target timestamps disagree")
            for name in ("source_frame_id", "source_time_s", "reuse", "time_residual_ms", *truth_fields):
                if name not in window:
                    continue
                value = _array(window[name])
                value = value if name == "gt_betas" and value.ndim == 1 else value[i]
                if name in row and not np.array_equal(row[name], value, equal_nan=True):
                    raise ValueError(f"overlapping windows disagree on {name}")
                row[name] = np.array(value, copy=True)
            if valid[i]:
                for name, shape in (("rotation",(24,3,3)),("pressure",(96,)),("velocity",(3,))):
                    value = _array(window[name])[i]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f"invalid {name} prediction")
                    row[name+"_sum"] += value
                row["n"] += 1
        for i in range(max(len(ids)-1, 0)):
            if ids[i] < 0 or ids[i+1] < 0:
                continue
            pair = (int(ids[i]), int(ids[i+1]))
            edge = bool(_array(window["view_edges"])[i])
            group["edges"][pair] = group["edges"].get(pair, True) and edge
            for name in ("motion_edges", "pressure_edges"):
                if name in window:
                    value = _array(window[name])[i].astype(bool)
                    group["label_edges"][(name,pair)] = group["label_edges"].get((name,pair),True) & value
    result = {}
    for key, group in sorted(groups.items()):
        ids = np.array(sorted(group["rows"]), np.int64)
        rows = [group["rows"][int(i)] for i in ids]
        if not rows:
            continue
        times = np.array([r["time_s"] for r in rows], np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("original target times must increase")
        counts = np.array([r["n"] for r in rows], np.int64)
        valid = counts > 0
        series = {"recording_id": key[0], "camera_id": key[1], "subject": group["subject"],
                  "action":group['action'],"date":group['date'],
                  "frame_id": ids, "time_s": times, "valid": valid, "overlap_count": counts,
                  "rotation": np.full((len(ids),24,3,3), np.nan),
                  "pressure": np.full((len(ids),96), np.nan), "velocity": np.full((len(ids),3), np.nan)}
        for name in ("source_frame_id", "source_time_s", "reuse", "time_residual_ms", *truth_fields):
            if all(name in r for r in rows):
                series[name] = np.stack([r[name] for r in rows])
        if "gt_betas" in series:
            series["gt_betas"] = series["gt_betas"][0]
        for i in np.flatnonzero(valid):
            series["rotation"][i] = project_so3(rows[i]["rotation_sum"]/counts[i])
            for name in ("pressure", "velocity"):
                series[name][i] = rows[i][name+"_sum"]/counts[i]
        dt = np.diff(times)
        edges = valid[:-1] & valid[1:] & (np.diff(ids) == 1) & (dt > 0) & (dt <= .050000001)
        if "source_frame_id" in series:
            edges &= np.diff(series["source_frame_id"]) > 0
        if "source_time_s" in series:
            ds = np.diff(series["source_time_s"])
            edges &= (ds > 0) & (ds <= .050000001)
        else:
            # Missing source provenance cannot establish dynamic continuity.
            edges[:] = False
        if "reuse" in series:
            edges &= ~series["reuse"][1:].astype(bool)
        for i, pair in enumerate(zip(ids[:-1], ids[1:])):
            if pair in group["edges"]:
                edges[i] &= group["edges"][pair]
        series["edges"] = edges
        for name in ("motion_edges", "pressure_edges"):
            shape = (max(len(ids)-1,0),96) if name == "pressure_edges" else (max(len(ids)-1,0),)
            series[name] = np.zeros(shape,bool)
            for i,pair in enumerate(zip(ids[:-1],ids[1:])):
                if (name,pair) in group["label_edges"]:
                    series[name][i] = group["label_edges"][(name,pair)]
        series["trajectory"], series["segment_id"] = integrate_visual_velocity(
            series["velocity"], times, ids, valid, edges)
        result[key] = series
    return result


def beta_zero_visual_offsets(model_path):
    from anysole.data.smpl_io import _rest_joints
    from anysole.types import JOINT_PARENTS
    rest = _rest_joints({"betas": np.zeros(10)}, model_path=Path(model_path))
    offsets = np.zeros((24,3), np.float32)
    for joint, parent in enumerate(JOINT_PARENTS):
        if parent >= 0:
            offsets[joint] = rest[joint]-rest[parent]
    return torch.from_numpy(offsets)


def visual_predicted_joints(series, offsets):
    from anysole.types import JOINT_PARENTS
    valid = series["valid"]
    joints = np.full((len(valid),24,3), np.nan, np.float64)
    rotation = series["rotation"][valid]
    pose = np.concatenate((rotation[...,0], rotation[...,1]), -1).reshape(1,-1,144)
    if valid.any():
        joints[valid] = fk_pose6d(torch.tensor(pose, dtype=torch.float32),
            torch.zeros(1, valid.sum(), 3), torch.as_tensor(offsets, dtype=torch.float32),
            torch.tensor(JOINT_PARENTS))[0].numpy()
    return joints


def visual_selection_metrics(series, offsets, truth_reference=None, prefix="val"):
    """Single shared beta0 root-relative MPJPE/pressure RMSE selector formula."""
    totals, camera_totals, coverage = np.zeros(5), {}, []
    inferred_truth = {}
    for key, item in sorted(series.items()):
        valid = item["valid"]
        mv = np.asarray(item.get("motion_valid", np.zeros(len(valid))), bool)
        pm = np.asarray(item.get("pressure_valid", np.zeros((len(valid),96))), bool)
        for i, frame in enumerate(item["frame_id"]):
            inferred_truth[(*key,int(frame))] = (bool(mv[i]), pm[i])
        pred = visual_predicted_joints(item, offsets)
        take = valid & mv
        err = np.linalg.norm((pred[take]-pred[take,:1])-
                            (item["gt_joints"][take]-item["gt_joints"][take,:1]), axis=-1).mean(-1) if take.any() else np.zeros(0)
        mask = valid[:,None] & pm
        delta = (item["pressure"][mask]-item["gt_pressure"][mask]) if mask.any() else np.zeros(0)
        if not np.isfinite(err).all() or not np.isfinite(delta).all():
            raise ValueError("nonfinite valid evaluation truth/prediction")
        values = np.array([err.sum(), take.sum(), np.square(delta).sum(), np.abs(delta).sum(), mask.sum()])
        totals += values
        camera_totals[key[1]] = camera_totals.get(key[1], np.zeros(5))+values
        for i in np.flatnonzero(valid):
            coverage.append([*key,int(item["frame_id"][i]),bool(mv[i]),pm[i].astype(np.uint8).tolist()])
    truth = inferred_truth if truth_reference is None else truth_reference
    motion_gt = sum(int(m) for m,_ in truth.values())
    pressure_gt = sum(int(np.asarray(p).sum()) for _,p in truth.values())
    motion_sum,motion_n,pressure_sq,pressure_abs,pressure_n = totals
    result = {f"{prefix}/mpjpe": float(motion_sum/motion_n) if motion_n else None,
              f"{prefix}/pressure_rmse": float(np.sqrt(pressure_sq/pressure_n)) if pressure_n else None,
              f"{prefix}/pressure_mae": float(pressure_abs/pressure_n) if pressure_n else None,
              f"{prefix}/motion_frames":int(motion_n),f"{prefix}/pressure_cells":int(pressure_n),
              f"{prefix}/prediction_frames":len(coverage),f"{prefix}/gt_motion_frames":motion_gt,
              f"{prefix}/gt_pressure_cells":pressure_gt,
              f"{prefix}/motion_coverage":motion_n/max(motion_gt,1),
              f"{prefix}/pressure_coverage":pressure_n/max(pressure_gt,1),
              f"{prefix}/coverage_hash":visual_hash({"prediction":coverage,"truth":[
                  [*k,bool(m),np.asarray(p,np.uint8).tolist()] for k,(m,p) in sorted(truth.items())]}),
              f"{prefix}/pred_beta_source":"zero",f"{prefix}/duplicate_policy":"SO3_polar_uniform_pressure_mean",
              f"{prefix}/selection_protocol":"smpl24_rootrelative_m_pressure_normalized.v1"}
    for camera, value in camera_totals.items():
        if value[1]:
            result[f"{prefix}/cam{camera}/mpjpe"] = float(value[0]/value[1])
        if value[4]:
            result[f"{prefix}/cam{camera}/pressure_rmse"] = float(np.sqrt(value[2]/value[4]))
    return result


def collect_visual_windows(model, dataset, device, *, cameras=(1,2,3,4), recording_ids=None):
    """Only V/mask/time reach student. Full GT remains on the evaluator side."""
    from anysole.data.dataset import grouped_collate
    from anysole.train import _move_grouped, capture_training_rng, restore_training_rng
    state, was_training = capture_training_rng(device), model.training
    windows, truth_reference, plot = [], {}, None
    model.eval()
    try:
        with torch.no_grad():
            for index in range(len(dataset)):
                item = dataset[index]
                batch = _move_grouped(grouped_collate([item]), device)
                visual, labels, masks, meta = (batch[k] for k in ("visual","labels","masks","meta"))
                for t, frame in enumerate(_array(meta["target_frame_id"][0])):
                    if frame >= 0:
                        for camera in cameras:
                            truth_reference[(meta["recording_id"][0],camera,int(frame))] = (
                                bool(masks["motion_valid"][0,t]),_array(masks["pressure_valid"][0,t]))
                active = visual["view_valid"].any(-1).nonzero()
                predictions = None
                if len(active):
                    g,c = active.unbind(1)
                    predictions = model.forward_visual(visual["V_feat"][g,c],visual["view_valid"][g,c],visual["target_time_s"][g])
                    rotation = _array(rot6d_to_rotmat(predictions["x0_hat"].float().reshape(len(active),-1,24,6)))
                    flat_index = {int(slot):flat for flat,(_,slot) in enumerate(active.tolist())}
                else:
                    flat_index = {}
                for c,camera in enumerate(_array(meta["camera_id"][0])):
                    if camera < 0 or camera not in cameras:
                        continue
                    length = visual["view_valid"].shape[-1]
                    window = {"recording_id":meta["recording_id"][0],"camera_id":int(camera),
                        "subject":meta.get("subject",["unknown"])[0],
                        "action":meta.get("action",["unknown"])[0],"date":meta.get("date",["unknown"])[0],
                        "frame_id":_array(meta["target_frame_id"][0]),"time_s":_array(visual["target_time_s"][0]),
                        "valid":_array(visual["view_valid"][0,c]),"view_edges":_array(visual["view_edges"][0,c]),
                        "motion_valid":_array(masks["motion_valid"][0]),"pressure_valid":_array(masks["pressure_valid"][0]),
                        "motion_edges":_array(masks["motion_edges"][0]),"pressure_edges":_array(masks["pressure_edges"][0]),
                        "gt_joints":_array(labels["kp_gt"][0]),"gt_pressure":_array(labels["pressure"][0]),
                        "gt_pose6d":_array(labels["pose6d"][0]),"gt_root":_array(labels["root"][0]),
                        "gt_betas":_array(labels["betas"][0])}
                    for name in ("source_frame_id","source_time_s","reuse","time_residual_ms"):
                        if name in meta:
                            window[name] = _array(meta[name][0,c])
                    flat = flat_index.get(c)
                    if flat is None:
                        window.update(rotation=np.zeros((length,24,3,3)),pressure=np.zeros((length,96)),velocity=np.zeros((length,3)))
                    else:
                        window.update(rotation=rotation[flat],pressure=_array(predictions["pressure_hat"][flat].float()),
                                      velocity=_array(predictions["v_hat"][flat].float()))
                    windows.append(window)
                    if plot is None and flat is not None:
                        plot = {"prediction":window["pressure"],"truth":window["gt_pressure"],
                                "mask":window["pressure_valid"] & window["valid"][:,None],
                                "key":meta["sample_key"][0],"split":getattr(dataset,"split","val"),
                                "unit":"normalized_sensor_reading"}
    finally:
        model.train(was_training)
        restore_training_rng(state, device)
    # Manifest axis templates also retain rows absent from every RGB window.
    # This path is evaluation-only and reads labels only on the evaluator side.
    if hasattr(dataset,"records"):
        from anysole.data.workspace_adapter import load_grouped_labels
        for ri,record in enumerate(dataset.records):
            if recording_ids is not None and record['recording_id'] not in recording_ids:
                continue
            labels=dataset._labels.get(ri)
            if labels is None:
                labels=load_grouped_labels(record,dataset.dataset_root,dataset.smpl_model_path)
            n=len(record['target_frame_id'])
            mv=np.asarray(record['motion']['valid'],bool)
            pm=np.repeat(np.asarray(record['pressure']['foot_valid'],bool),48,axis=1)
            for camera in cameras:
                for i,frame in enumerate(record['target_frame_id']):
                    truth_reference[(record['recording_id'],camera,int(frame))]=(bool(mv[i]),pm[i])
                view=next((v for v in record['views'] if v['camera_id']==camera),None)
                template={'recording_id':record['recording_id'],'camera_id':camera,'subject':record['subject'],
                    'action':record['action'],'date':record['date'],
                    'frame_id':record['target_frame_id'],'time_s':record['target_time_s'],'valid':np.zeros(n,bool),
                    'view_edges':np.ones(max(n-1,0),bool),'motion_valid':mv,'pressure_valid':pm,
                    'motion_edges':record['motion_edges'],
                    'pressure_edges':np.repeat(np.asarray(record['pressure_foot_edges'],bool),48,axis=1),
                    'gt_joints':labels['kp_gt'],'gt_pressure':labels['pressure'],'gt_pose6d':labels['pose6d'],
                    'gt_root':labels['root'],'gt_betas':labels['betas']}
                if view:
                    template.update({name:view[name] for name in ('source_frame_id','source_time_s','reuse','time_residual_ms')})
                windows.append(template)
    return windows, truth_reference, plot


def evaluate_visual_dataset(model,dataset,config,device,*,offsets=None):
    if getattr(dataset,"split","val") == "test":
        raise ValueError("training validation cannot use the test role")
    windows,truth,plot = collect_visual_windows(model,dataset,device,cameras=tuple(config["validation_cameras"]))
    if not any(np.asarray(w["valid"]).any() for w in windows):
        raise ValueError("empty validation outputs")
    offsets = offsets if offsets is not None else beta_zero_visual_offsets(config["smpl_model_path"])
    result = visual_selection_metrics(aggregate_visual_windows(windows), offsets, truth)
    if not result["val/motion_frames"] or not result["val/pressure_cells"]:
        raise ValueError("validation has no common motion/pressure supervision")
    return result,plot


def _metric(value, count, unit, *, reason=None, support=None,eligible_support=None,expected_count=None):
    count = int(count)
    expected=count if expected_count is None else int(expected_count)
    missing=max(0,expected-count)
    if missing:
        value=None
        reason=f'incomplete prediction on fixed public metric set: {missing}/{expected} samples unavailable'
    value = float(value) if value is not None and np.isfinite(value) and count else None
    return {"value":value,"count":count,"unit":unit,
            "reason":None if value is not None else (reason or "no valid samples"),
            "support_hash":visual_hash(support or []),
            "eligible_support_hash":visual_hash((support or []) if eligible_support is None else eligible_support),
            "eligible_count":expected,"missing_prediction_count":missing}


def _segment_indices(valid, edges):
    valid, edges = np.asarray(valid,bool),np.asarray(edges,bool)
    result, current = [],[]
    for i in range(len(valid)):
        if not valid[i]:
            if current:
                result.append(np.asarray(current)); current=[]
            continue
        if current and not edges[i-1]:
            result.append(np.asarray(current)); current=[]
        current.append(i)
    if current:
        result.append(np.asarray(current))
    return result


def _time_correlation(first,second,times):
    if len(times) < 3:
        return None
    # Time-weighted Pearson, using trapezoidal quadrature on actual seconds.
    dt=np.diff(times)
    weight=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    weight/=weight.sum()
    a=first-np.sum(weight*first); b=second-np.sum(weight*second)
    denom=np.sqrt(np.sum(weight*a*a)*np.sum(weight*b*b))
    return float(np.sum(weight*a*b)/denom) if denom>1e-12 else None


def visual_pressure_metrics(item, frozen_stats=None):
    """Raw normalized readings; temporal correlation is never spatial T_corr."""
    valid=item['valid']; pm=np.asarray(item.get('pressure_valid',np.zeros((len(valid),96))),bool)
    mask=valid[:,None]&pm
    pred=item['pressure']; truth=item.get('gt_pressure',np.zeros_like(pred))
    if not np.isfinite(truth[mask]).all():
        raise ValueError('nonfinite pressure label marked valid')
    delta=pred[mask]-truth[mask]
    ids=item['frame_id']; keys=lambda take: ids[np.asarray(take)].astype(int).tolist()
    support=np.argwhere(mask).tolist()
    result={
        'pressure_mae':_metric(np.abs(delta).mean() if delta.size else None,delta.size,'normalized_sensor_reading',support=support),
        'pressure_rmse':_metric(np.sqrt(np.square(delta).mean()) if delta.size else None,delta.size,'normalized_sensor_reading',support=support),
        'pressure_out_of_range_fraction':_metric(((pred[valid]<0)|(pred[valid]>1)).mean() if valid.any() else None,
            int(valid.sum())*96,'fraction',support=keys(valid)),
        'pressure_spatial_corr':_metric(None,0,'correlation',reason='no frame with nonconstant paired sensor vectors')}
    spatial=[]; spatial_ids=[];spatial_scored=[]
    for i in np.flatnonzero(valid):
        keep=pm[i]
        a,b=pred[i,keep],truth[i,keep]
        if len(a)>1 and b.std()>1e-12:
            spatial_ids.append(int(ids[i]))
            if a.std()>1e-12:
                spatial.append(np.corrcoef(a,b)[0,1])
                spatial_scored.append(int(ids[i]))
    result['pressure_spatial_corr']=_metric(np.mean(spatial) if spatial else None,len(spatial),'correlation',
        support=spatial_scored,eligible_support=spatial_ids,expected_count=len(spatial_ids),reason='no nonconstant GT sensor vectors')
    foot_pred=pred.reshape(-1,2,48).sum(-1)
    foot_gt=truth.reshape(-1,2,48).sum(-1)
    foot_valid=pm.reshape(-1,2,48).all(-1)&valid[:,None]
    policy=(frozen_stats or {}).get('pressure_eval',{})
    fitted=policy.get('fit_split')=='train' and bool(policy.get('fitted_from_train_sha256'))
    for foot,name in enumerate(('left','right')):
        keep=foot_valid[:,foot]
        d=foot_pred[keep,foot]-foot_gt[keep,foot]
        result[f'{name}_sum_mae']=_metric(np.abs(d).mean() if len(d) else None,len(d),'normalized_reading_sum',support=keys(keep))
        correlations=[]; peak=[]; temporal_support=[];peak_support=[];temporal_scored=[];peak_scored=[];peak_expected=0
        pressure_edges=item.get('pressure_edges',np.zeros((max(len(valid)-1,0),96),bool))
        foot_edges=pressure_edges[:,foot*48:(foot+1)*48].all(-1)
        for ix in _segment_indices(keep,item['edges'] & foot_edges):
            corr=_time_correlation(foot_pred[ix,foot],foot_gt[ix,foot],item['time_s'][ix])
            if len(ix)>=3 and foot_gt[ix,foot].std()>1e-12:
                temporal_support.extend(ids[ix].astype(int).tolist())
            if corr is not None:
                correlations.append((corr,len(ix)))
                temporal_scored.extend(ids[ix].astype(int).tolist())
            # Tied/constant peaks carry no unique physical peak time.
            a,b=foot_pred[ix,foot],foot_gt[ix,foot]
            if len(ix)>=3 and np.count_nonzero(b==b.max())==1:
                peak_support.extend(ids[ix].astype(int).tolist())
                peak_expected+=1
                if np.count_nonzero(a==a.max())==1:
                    peak.append(abs(item['time_s'][ix[np.argmax(a)]]-item['time_s'][ix[np.argmax(b)]])*1000)
                    peak_scored.extend(ids[ix].astype(int).tolist())
        result[f'{name}_temporal_corr']=_metric(
            np.average([v for v,n in correlations],weights=[n for v,n in correlations]) if correlations else None,
            sum(n for _,n in correlations),'correlation',reason='needs >=3 continuous paired frames with nonconstant load',
            support=temporal_scored,eligible_support=temporal_support,expected_count=len(temporal_support))
        result[f'{name}_peak_timing_abs_ms']=_metric(np.mean(peak) if peak else None,len(peak),'ms',
            reason='no continuous segment with unique predicted and GT peaks',support=peak_scored,
            eligible_support=peak_support,expected_count=peak_expected)
        threshold=policy.get('low_load_threshold_normalized') if fitted else None
        if threshold is None:
            result[f'{name}_low_load_fraction']=_metric(None,0,'fraction',reason='missing frozen train-fitted low-load threshold')
        else:
            result[f'{name}_low_load_fraction']=_metric((foot_gt[keep,foot]<=threshold[foot]).mean() if keep.any() else None,
                keep.sum(),'fraction',support=keys(keep))
        threshold=policy.get('contact_threshold_normalized') if fitted else None
        for metric in ('contact_f1','cop_error_grid'):
            result[f'{name}_{metric}']=_metric(None,0,'fraction' if metric=='contact_f1' else 'sensor_grid_units',
                reason='missing frozen train-fitted pressure contact threshold or trusted sensor geometry')
        if threshold is not None and keep.any():
            pp=foot_pred[keep,foot]>threshold[foot]; gg=foot_gt[keep,foot]>threshold[foot]
            tp=np.sum(pp&gg); denominator=pp.sum()+gg.sum()
            result[f'{name}_contact_f1']=_metric(2*tp/denominator if denominator else None,keep.sum(),'fraction',
                reason='no positive contact labels or predictions',support=keys(keep))
            geometry=policy.get('sensor_xy_grid')
            if geometry is not None:
                xy=np.asarray(geometry,float)
                if xy.shape!=(2,48,2) or not np.isfinite(xy).all():
                    raise ValueError('sensor_xy_grid must have trusted [2,48,2] finite geometry')
                sensor=slice(foot*48,(foot+1)*48)
                eligible_cop=keep & (foot_gt[:,foot]>threshold[foot]) & (truth[:,sensor]>=0).all(-1)
                select=eligible_cop & (foot_pred[:,foot]>threshold[foot])
                select &= (pred[:,sensor]>=0).all(-1)&(truth[:,sensor]>=0).all(-1)
                value=None
                if select.any():
                    pc=(pred[select,sensor]@xy[foot])/foot_pred[select,foot,None]
                    gc=(truth[select,sensor]@xy[foot])/foot_gt[select,foot,None]
                    value=np.linalg.norm(pc-gc,axis=-1).mean()
                result[f'{name}_cop_error_grid']=_metric(value,select.sum(),'sensor_grid_units',
                    support=keys(select),eligible_support=keys(eligible_cop),expected_count=eligible_cop.sum())
    both=foot_valid.all(-1)
    pred_total=foot_pred.sum(-1); truth_total=foot_gt.sum(-1)
    eligible=both & (truth_total>1e-8)
    both = eligible & (pred_total>1e-8)
    result['left_ratio_mae']=_metric(np.abs(foot_pred[both,0]/pred_total[both]-foot_gt[both,0]/truth_total[both]).mean()
        if both.any() else None,both.sum(),'fraction',reason='no paired nonzero total loads',support=keys(both),
        eligible_support=keys(eligible),expected_count=eligible.sum())
    return result


def visual_motion_metrics(item, offsets):
    """Canonical SMPL24 formulas, predicted beta0, no absolute root origin."""
    from anysole.utils import metrics as canonical
    from anysole.types import LOWER_JOINTS,ANKLE_FOOT_JOINTS
    valid=item['valid']; motion=np.asarray(item.get('motion_valid',np.zeros(len(valid))),bool)&valid
    pred=visual_predicted_joints(item,offsets); gt=item.get('gt_joints',np.zeros_like(pred))
    keys=item['frame_id'][motion].astype(int).tolist()
    if not np.isfinite(gt[motion]).all():
        raise ValueError('nonfinite motion label marked valid')
    p=pred[motion]-pred[motion,:1]; g=gt[motion]-gt[motion,:1]
    result={}
    for name,joints in (('mpjpe_mm',list(range(24))),('lower_mpjpe_mm',LOWER_JOINTS),('feet_mpjpe_mm',ANKLE_FOOT_JOINTS)):
        result[name]=_metric(canonical.mean_point_error(p[:,joints],g[:,joints],scale=1000).mean() if len(p) else None,
            len(p),'mm',support=keys)
    result['pa_mpjpe_mm']=_metric(canonical.pa_mpjpe(p,g).mean() if len(p) else None,len(p),'mm',support=keys)
    if 'gt_pose6d' in item and motion.any():
        gtrot=rot6d_to_rotmat(torch.tensor(item['gt_pose6d'][motion]).reshape(-1,24,6)).numpy()
        for name,joints in (('local23_angle_deg',list(range(1,24))),('root_angle_deg',[0])):
            err=canonical.matrix_rotation_error_degrees(item['rotation'][motion][:,joints],gtrot[:,joints])
            result[name]=_metric(err.mean(),len(p),'degree',support=keys)
    accum={name:[] for name in ('accel_error_m_s2','jitter_pred_m_s3','jitter_gt_m_s3','foot_sliding_motion_contact_mm',
                               'relative_root_drift_mm','w_mpjpe100_mm')}
    dynamic_support={name:[] for name in accum}
    for ix in _segment_indices(motion,item['edges'] & item.get('motion_edges',np.zeros(max(len(valid)-1,0),bool))):
        if len(ix)>=3:
            pp=pred[ix]-pred[ix,:1]; gg=gt[ix]-gt[ix,:1]
            temporal=canonical.temporal_metrics(pp,gg,item['time_s'][ix])
            for name in ('accel_error_m_s2','jitter_pred_m_s3','jitter_gt_m_s3'):
                count=temporal['acceleration_samples' if name=='accel_error_m_s2' else 'jerk_samples']
                if count:
                    accum[name].append((temporal[name],count))
                    horizon=3 if name=='accel_error_m_s2' else 4
                    dynamic_support[name].extend([item['frame_id'][ix[k:k+horizon]].astype(int).tolist()
                                                 for k in range(len(ix)-horizon+1)])
        if len(ix)>=2 and 'gt_root' in item:
            root=item['gt_root'][ix]-item['gt_root'][ix[0]]
            predroot=item['trajectory'][ix]-item['trajectory'][ix[0]]
            accum['relative_root_drift_mm'].append((np.linalg.norm(predroot-root,axis=-1).mean()*1000,len(ix)))
            dynamic_support['relative_root_drift_mm'].extend(item['frame_id'][ix].astype(int).tolist())
            worldpred=pred[ix]+predroot[:,None]
            worldgt=gt[ix]-gt[ix[0],:1]
            slide,contacts=canonical.foot_sliding_joints(worldpred[:,ANKLE_FOOT_JOINTS],worldgt[:,ANKLE_FOOT_JOINTS],item['time_s'][ix])
            if contacts:
                accum['foot_sliding_motion_contact_mm'].append((slide,contacts))
                speed=np.linalg.norm(np.diff(worldgt[:,ANKLE_FOOT_JOINTS],axis=0),axis=-1)/np.diff(item['time_s'][ix])[:,None]
                dynamic_support['foot_sliding_motion_contact_mm'].extend([
                    [int(item['frame_id'][ix[k]]),int(item['frame_id'][ix[k+1]]),int(foot)] for k,foot in np.argwhere(speed<.3)])
            # Require complete W100; canonical partial-window behavior not used.
            for start in range(0,len(ix)-99,100):
                value,_=canonical.windowed_world_mpjpe(worldpred[start:start+100],worldgt[start:start+100],window=100)
                accum['w_mpjpe100_mm'].append((value,100))
                dynamic_support['w_mpjpe100_mm'].append(item['frame_id'][ix[start:start+100]].astype(int).tolist())
    for name,values in accum.items():
        result[name]=_metric(np.average([v for v,n in values],weights=[n for v,n in values]) if values else None,
            sum(n for _,n in values),'m/s^2' if name=='accel_error_m_s2' else ('m/s^3' if 'jitter' in name else 'mm'),
            reason='no complete continuous source-valid motion segment',support=dynamic_support[name])
    result['root_ate_mm']=_metric(None,0,'mm',reason='absolute world origin unavailable; every predicted segment starts at zero')
    result['world_mpjpe_mm']=_metric(None,0,'mm',reason='absolute world origin unavailable')
    for name in ('pve_mm','pve_t_mm'):
        result[name]=_metric(None,0,'mm',reason='surface evaluation not requested; use --surfaces with authorized current SMPL')
    return result


def visual_session_rows(series,offsets,frozen_stats=None):
    rows=[]
    for (recording,camera),item in sorted(series.items()):
        metrics={**visual_motion_metrics(item,offsets),**visual_pressure_metrics(item,frozen_stats)}
        rows.append({'recording_id':recording,'camera_id':camera,'subject':item['subject'],
            'action':item.get('action','unknown'),'date':item.get('date','unknown'),
            'prediction_frames':int(item['valid'].sum()),'target_frames':len(item['valid']),
            'missing_rgb_frames':int((~item['valid']).sum()),
            'missing_motion_frames':int((~np.asarray(item.get('motion_valid',np.zeros(len(item['valid']))),bool)).sum()),
            'missing_pressure_cells':int((~np.asarray(item.get('pressure_valid',np.zeros((len(item['valid']),96))),bool)).sum()),
            'metrics':metrics})
    return rows


def summarize_visual_rows(rows,*,bootstrap_samples=10000,seed=20261016):
    """Session×camera -> equal session within camera -> equal camera macro.

    Bootstrap samples whole physical recordings and retains every camera row.
    Subjects are separately described; two held-out subjects are not independent
    evidence for a broad population claim.
    """
    names=sorted({name for row in rows for name in row['metrics']})
    result={}
    generator=np.random.default_rng(seed)
    recordings=sorted({r['recording_id'] for r in rows})
    cameras=sorted({r['camera_id'] for r in rows})
    draws_index=generator.integers(0,len(recordings),size=(bootstrap_samples,len(recordings))) if recordings and bootstrap_samples else None
    def macro(subset,name):
        cameras=sorted({r['camera_id'] for r in subset})
        per_camera={}
        for camera in cameras:
            selected=[r['metrics'][name] for r in subset if r['camera_id']==camera and name in r['metrics']]
            if any(m['value'] is None and m.get('eligible_count',0)>0 for m in selected):
                per_camera[str(camera)]=None
                continue
            vals=[m['value'] for m in selected if m['value'] is not None]
            if vals:
                per_camera[str(camera)]=float(np.mean(vals))
        return (float(np.mean(list(per_camera.values()))) if per_camera and all(v is not None for v in per_camera.values()) else None),per_camera
    for name in names:
        value,bycamera=macro(rows,name)
        bysubject={}
        for subject in sorted({r['subject'] for r in rows}):
            bysubject[subject]=macro([r for r in rows if r['subject']==subject],name)[0]
        byaction={action:macro([r for r in rows if r.get('action','unknown')==action],name)[0]
                  for action in sorted({r.get('action','unknown') for r in rows})}
        draws=[]
        if draws_index is not None and value is not None:
            matrix=np.full((len(recordings),len(cameras)),np.nan)
            for ri,recording in enumerate(recordings):
                for ci,camera in enumerate(cameras):
                    values=[r['metrics'][name]['value'] for r in rows if r['recording_id']==recording
                            and r['camera_id']==camera and name in r['metrics'] and r['metrics'][name]['value'] is not None]
                    if values:
                        matrix[ri,ci]=np.mean(values)
            for start in range(0,len(draws_index),512):
                sampled=matrix[draws_index[start:start+512]]
                counts=np.isfinite(sampled).sum(axis=1)
                means=np.divide(np.nansum(sampled,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)
                available=np.isfinite(means).sum(-1)
                macro_values=np.divide(np.nansum(means,axis=-1),available,out=np.full(len(available),np.nan),where=available>0)
                draws.extend(macro_values[np.isfinite(macro_values)].tolist())
        result[name]={'value':value,'camera_macro_inputs':bycamera,'subject_macro':bysubject,'action_macro':byaction,
                      'recording_cluster_ci95':np.percentile(draws,[2.5,97.5]).tolist() if draws else None,
                      'recording_clusters':len(recordings),
                      'valid_session_camera_rows':sum(r['metrics'].get(name,{}).get('value') is not None for r in rows),
                      'missing_reasons':sorted({r['metrics'][name]['reason'] for r in rows if name in r['metrics'] and r['metrics'][name]['reason']})}
    return {'metrics':result,'policy':'session_mean_within_camera_then_equal_camera_macro; physical_recording_cluster_bootstrap',
            'bootstrap_samples':bootstrap_samples,'bootstrap_seed':seed,
            'subject_limit':'held-out S13/S14 only; val and test are reused roles, not independent generalization'}


def load_visual_student(checkpoint_path,device='cpu'):
    """Rebuild exclusively from saved effective config and train-only stats."""
    from anysole.registry import get_builder
    from anysole.train import restore_visual_runtime_state,visual_run_identity
    path=Path(checkpoint_path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('schema_version')!='anysole.visual-training.v1':
        raise ValueError('expected grouped visual-training checkpoint schema')
    config=checkpoint.get('config',{})
    required=('model_name','pressure_head','pressure_time_pe','pressure_time_pe_enabled','time_pe_scale',
              'visual_structure_version','structure_hash','d_model','tw','pose_layers','nhead','dropout','modal','pose_repr')
    if any(key not in config for key in required):
        raise ValueError('checkpoint missing saved effective architecture fields')
    if checkpoint.get('semantic_hash')!=visual_run_identity(config):
        raise ValueError('checkpoint semantic identity mismatch')
    config=get_builder('visual').effective_config(config)
    stats=checkpoint.get('stats',{})
    if 'pose_mean' not in stats or 'pose_std' not in stats:
        raise ValueError('checkpoint missing frozen pose statistics')
    if 'stats_sha256' in stats and visual_hash({k:v for k,v in stats.items() if k!='stats_sha256'})!=stats['stats_sha256']:
        raise ValueError('checkpoint statistics hash mismatch')
    if config.get('stats_hash') is not None and config['stats_hash']!=stats.get('stats_sha256'):
        raise ValueError('checkpoint config/statistics binding mismatch')
    model=get_builder('visual').build(config)
    model.load_state_dict(checkpoint['model'],strict=True)
    restore_visual_runtime_state(model,checkpoint)
    for name in ('pose_mean','pose_std'):
        expected=torch.tensor(stats[name],dtype=torch.float32)
        if name=='pose_std':
            if not (expected>0).all():
                raise ValueError('checkpoint pose_std must be strictly positive')
            expected=expected.clamp_min(.01)
        actual=getattr(model.pose_head,name).detach().cpu()
        if expected.shape!=actual.shape or not torch.isfinite(expected).all() or not torch.equal(expected,actual):
            raise ValueError('checkpoint state/statistics disagree for '+name)
    model.eval().to(device)
    return model,config,stats,checkpoint


def visual_surface_metrics(item,model_path,*,batch_size=32):
    """Canonical root-relative PVE, beta0 predicted and GTbeta reference."""
    import smplx
    from anysole.data.smpl_io import smpl24_pose6d_to_poses
    from anysole.utils import metrics as canonical
    take=item['valid']&np.asarray(item.get('motion_valid',np.zeros(len(item['valid']))),bool)
    if not take.any():
        return {name:_metric(None,0,'mm') for name in ('pve_mm','pve_t_mm')}
    rotation=item['rotation'][take]
    pose=np.concatenate((rotation[...,0],rotation[...,1]),-1).reshape(-1,144)
    predpose=smpl24_pose6d_to_poses(pose)
    gtpose=smpl24_pose6d_to_poses(item['gt_pose6d'][take])
    model=smplx.create(str(Path(model_path)),'smpl',gender='neutral',num_betas=10,batch_size=batch_size)
    beta=np.asarray(item['gt_betas'],np.float32).reshape(-1)[:10]
    def surface(poses,betas):
        vertices,joints=[],[]
        for start in range(0,len(poses),batch_size):
            pp=torch.tensor(poses[start:start+batch_size],dtype=torch.float32).reshape(-1,24,3)
            with torch.no_grad():
                output=model(betas=torch.tensor(np.repeat(betas[None],len(pp),0)),body_pose=pp[:,1:],
                             global_orient=pp[:,:1],transl=torch.zeros(len(pp),3))
            vertices.append(output.vertices.numpy());joints.append(output.joints[:,:24].numpy())
        return np.concatenate(joints),np.concatenate(vertices)
    pj,pv=surface(predpose,np.zeros(10,np.float32)); gj,gv=surface(gtpose,beta)
    pv=pv-pj[:,:1];gv=gv-gj[:,:1]
    pve=canonical.mean_point_error(pv,gv,scale=1000)
    zero=np.zeros((1,72),np.float32)
    ptj,ptv=surface(zero,np.zeros(10,np.float32));gtj,gtv=surface(zero,beta)
    pvet=canonical.pve_t_error(ptj,ptv,gtj,gtv)
    keys=item['frame_id'][take].astype(int).tolist()
    return {'pve_mm':_metric(pve.mean(),len(pve),'mm',support=keys),
            'pve_t_mm':_metric(pvet.mean(),len(pve),'mm',support=keys)}


def export_visual_series(series,output,config,provenance):
    """Prediction-only NPZ; no truth arrays or shape leaked into student export."""
    from anysole.data.smpl_io import _rest_joints,smpl24_pose6d_to_poses
    output=Path(output)
    if not output.resolve().is_relative_to(Path('/private/Tactile')):
        raise ValueError('outputs must remain in /private/Tactile')
    output.mkdir(parents=True,exist_ok=True)
    rest=_rest_joints({'betas':np.zeros(10)},model_path=Path(config['smpl_model_path']))[0]
    paths=[]
    used_names=set()
    for (recording,camera),item in sorted(series.items()):
        # Physical recording IDs are date/session; flatten only that grammar.
        # Keep the original ID in metadata, and reject traversal/extra levels.
        import re
        if re.fullmatch(r'\d{8}/S\d+',recording):
            stem=recording.replace('/','_')
        elif re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',recording):
            stem=recording
        else:
            raise ValueError('unsafe recording identifier')
        name=f'{stem}_cam{camera}.npz'
        if name in used_names:
            raise ValueError('flattened recording filename collision')
        used_names.add(name)
        valid=item['valid']; pose=np.full((len(valid),144),np.nan,np.float32)
        rotation=item['rotation'][valid]
        pose[valid]=np.concatenate((rotation[...,0],rotation[...,1]),-1).reshape(-1,144)
        poses=np.full((len(valid),72),np.nan,np.float32)
        if valid.any():
            poses[valid]=smpl24_pose6d_to_poses(pose[valid])
        meta={**provenance,'protocol':VISUAL_EVAL_PROTOCOL,'recording_id':recording,'camera_id':camera,
              'pred_beta_source':'zero','trajectory_origin':'p0=0 independently per continuous source-valid segment',
              'velocity_convention':'v[i] ends at target i; aggregate windows before integration',
              'pressure_unit':'normalized_sensor_reading; uncalibrated; not Newtons',
              'pressure_main':'unclamped_linear','pressure_display':'clip_0_1_for_display_only',
              'coordinate_system':'native SMPL +Y up, metres','pose_layout':'root world, joints1..23 parent-local',
              'timebase':'actual RGB source timestamps matched to explicit 40Hz target grid',
              'duplicate_policy':'uniform_SO3_chordal_projection_pressure_velocity_mean',
              'smpl_trans':'pelvis_root_translation minus beta0 unrotated rest pelvis',
              'absolute_world_origin':None,'effective_config':config}
        values={name:item[name] for name in ('frame_id','time_s','valid','overlap_count','rotation','pressure','velocity',
                                            'trajectory','segment_id','edges','source_frame_id','source_time_s','reuse','time_residual_ms') if name in item}
        path=output/name
        if path.exists():
            raise FileExistsError('refusing to overwrite existing prediction '+str(path))
        np.savez_compressed(path,**values,pose6d=pose,poses=poses,trans=item['trajectory']-rest,
            betas=np.zeros(10,np.float32),pressure_display=np.clip(item['pressure'],0,1),
            metadata_json=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False)))
        from anysole.data.workspace_adapter import sha256_file
        paths.append({'recording_id':recording,'camera_id':camera,'path':str(path),'sha256':sha256_file(path)})
    return paths


def _compatible_eval_reports(reports,*,same_checkpoint=False):
    if not reports:
        raise ValueError('no evaluation reports')
    for report in reports:
        if report.get('schema_version')!=VISUAL_EVAL_PROTOCOL:
            raise ValueError('evaluation schema mismatch')
    keys=['manifest_sha256','split','protocol_sha256','stats_sha256','smpl_runtime_model_hash']
    if same_checkpoint:
        keys+=['checkpoint_sha256','shard_count','expected_recordings']
    for name in keys:
        if any(report.get(name)!=reports[0].get(name) for report in reports):
            raise ValueError('evaluation reports disagree on '+name)


def merge_visual_shards(reports,*,bootstrap_samples=10000):
    _compatible_eval_reports(reports,same_checkpoint=True)
    count=int(reports[0]['shard_count'])
    if sorted(r['shard_index'] for r in reports)!=list(range(count)):
        raise ValueError('shards missing or duplicated')
    result={k:v for k,v in reports[0].items() if k not in ('rows','summary','predictions','selection_metrics','coverage_hash')}
    rows=[row for report in reports for row in report['rows']]
    keys=[(r['recording_id'],r['camera_id']) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('overlapping session-camera shards')
    expected=sorted({x for report in reports for x in report['expected_recordings']})
    if sorted({r['recording_id'] for r in rows})!=expected:
        raise ValueError('merged shards do not cover expected recordings')
    result.update(rows=sorted(rows,key=lambda r:(r['recording_id'],r['camera_id'])),shard_index=None,
                  predictions=[p for report in reports for p in report['predictions']],expected_recordings=expected)
    result['summary']=summarize_visual_rows(result['rows'],bootstrap_samples=bootstrap_samples)
    result['coverage_hash']=visual_hash([[r['recording_id'],r['camera_id'],{k:v['support_hash'] for k,v in r['metrics'].items()}] for r in result['rows']])
    result['prediction_hash']=visual_hash([entry['sha256'] for entry in result['predictions']])
    result['selected_recordings']=expected
    result['explicit_session_subset']=False
    if all('selection_metrics' in report for report in reports):
        selection={}
        for name in ('motion_frames','pressure_cells','prediction_frames','gt_motion_frames','gt_pressure_cells'):
            selection['eval/'+name]=sum(report['selection_metrics']['eval/'+name] for report in reports)
        for name,denom,squared in (('mpjpe','motion_frames',False),('pressure_mae','pressure_cells',False),('pressure_rmse','pressure_cells',True)):
            n=selection['eval/'+denom]
            numerator=sum((report['selection_metrics']['eval/'+name] or 0)**(2 if squared else 1)*
                          report['selection_metrics']['eval/'+denom] for report in reports)
            selection['eval/'+name]=(np.sqrt(numerator/n) if squared else numerator/n) if n else None
        selection['eval/coverage_hash']=result['coverage_hash']
        selection['eval/coverage_hash_policy']='merged per-session metric supports; unsharded selector uses full mask payload'
        result['selection_metrics']=selection
    return result


def aggregate_visual_reports(reports,*,baseline=None,bootstrap_samples=10000,require_full_matrix=True):
    """Frozen 15-run matrix; main seeds1/2/3 and mechanism seed1 only.

    Reject differing per-metric supports instead of quietly comparing changing
    sets. A new fixed-set reevaluation is required if any run loses coverage.
    """
    _compatible_eval_reports(reports)
    byexperiment={}
    coverage=None
    actual_support={}
    for report in reports:
        if report.get('shard_index') is not None and report.get('shard_count',1)>1:
            raise ValueError('merge all shards before cross-run aggregation')
        if set(r['recording_id'] for r in report['rows'])!=set(report['expected_recordings']):
            raise ValueError('report is a session subset; full expected recordings required')
        support=sorted((r['recording_id'],r['camera_id'],k,v.get('eligible_support_hash',v['support_hash']))
            for r in report['rows'] for k,v in r['metrics'].items() if k not in ('pressure_out_of_range_fraction',))
        if coverage is not None and support!=coverage:
            raise ValueError('metric coverage differs; reevaluate on a fixed common set')
        coverage=support
        for row in report['rows']:
            for name,metric in row['metrics'].items():
                key=(row['recording_id'],row['camera_id'],name)
                previous=actual_support.get(key)
                if previous and metric['value'] is not None and previous['value'] is not None and metric['support_hash']!=previous['support_hash']:
                    raise ValueError('finite metric values use different actual coverage')
                if metric['value'] is not None or previous is None:
                    actual_support[key]=metric
        exp=report['experiment']; seed=int(report['seed'])
        if seed in byexperiment.setdefault(exp,{}):
            raise ValueError('duplicate experiment seed')
        byexperiment[exp][seed]=report
    expected={name:{1,2,3} for name in ('base_legacy','base_time','base_pool','view_consist')}
    expected.update({name:{1} for name in ('pose_pair','press_pair','pair_shuffle')})
    if require_full_matrix and set(byexperiment)!=set(expected):
        raise ValueError('formal aggregate requires all 15 runs: four main groups and three mechanism groups')
    result={'schema_version':VISUAL_EVAL_PROTOCOL+'.aggregate','experiments':{},'paired_differences':{},
            'policy':'main seeds1/2/3 mean/sample-std; mechanism seed1 no std; physical recording paired cluster bootstrap; fixed metric supports',
            'formal_matrix_complete':set(byexperiment)==set(expected),
            'subject_limit':'S13/S14 only; reused val=test, not independent generalization'}
    for exp,runs in byexperiment.items():
        if exp not in expected or set(runs)!=expected[exp]:
            raise ValueError('unexpected or incomplete seeds for '+exp+'; expected '+str(sorted(expected.get(exp,[]))))
        metrics={}
        summaries=[summarize_visual_rows(runs[seed]['rows'],bootstrap_samples=0)['metrics'] for seed in sorted(runs)]
        for name in summaries[0]:
            values=[s[name]['value'] for s in summaries]
            good=[v for v in values if v is not None]
            metrics[name]={'mean':float(np.mean(good)) if len(good)==len(runs) else None,
                           'std':float(np.std(good,ddof=1)) if len(good)==len(runs) and len(runs)>1 else None,
                           'seed_values':values,'seeds':sorted(runs),'seed_count':len(runs),
                           'interpretation':'single-seed mechanism diagnostic' if len(runs)==1 else 'three-seed main comparison'}
        result['experiments'][exp]=metrics
    if baseline:
        if baseline not in byexperiment:
            raise ValueError('unknown paired baseline')
        generator=np.random.default_rng(20261016)
        for exp,runs in byexperiment.items():
            if exp==baseline:
                continue
            differences={}
            for name in result['experiments'][exp]:
                pairs=[]
                common_seeds=sorted(set(runs)&set(byexperiment[baseline]))
                for seed in common_seeds:
                    report=runs[seed]
                    left={(r['recording_id'],r['camera_id']):r for r in report['rows']}
                    for right in byexperiment[baseline][seed]['rows']:
                        key=(right['recording_id'],right['camera_id'])
                        a=left[key]['metrics'][name]['value'];b=right['metrics'][name]['value']
                        eligible=max(left[key]['metrics'][name].get('eligible_count',1),right['metrics'][name].get('eligible_count',1))
                        pairs.append({'recording_id':key[0],'camera_id':key[1],'subject':right['subject'],
                            'metrics':{name:{'value':a-b if a is not None and b is not None else None,
                                'eligible_count':eligible,'reason':None if a is not None and b is not None else 'paired metric incomplete on fixed public set'}}})
                if pairs:
                    summary=summarize_visual_rows(pairs,bootstrap_samples=bootstrap_samples,seed=int(generator.integers(2**31)))
                    differences[name]=summary['metrics'][name]
                    differences[name].update(paired_seeds=common_seeds,seed_count=len(common_seeds),
                        ci_interpretation='physical recordings conditional on these seeds; not a seed-population confidence interval')
            result['paired_differences'][f'{exp}-{baseline}']=differences
    return result


def verify_visual_report_artifacts(report):
    """CLI aggregation verifies hashes and prediction identity before merging."""
    from anysole.data.workspace_adapter import sha256_file
    entries=report.get('predictions',[])
    if report.get('rows') and not entries:
        raise ValueError('evaluation report has no auditable prediction archives')
    keys=[]
    for entry in entries:
        path=Path(entry['path'])
        if sha256_file(path)!=entry['sha256']:
            raise ValueError('prediction archive hash mismatch')
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata_json']))
            if meta['checkpoint_sha256']!=report['checkpoint_sha256']:
                raise ValueError('prediction archive belongs to another checkpoint')
            key=(meta['recording_id'],meta['camera_id']);keys.append(key)
            if key!=(entry['recording_id'],entry['camera_id']):
                raise ValueError('prediction entry identity mismatch')
            if not np.all(np.diff(archive['frame_id'])>0) or not np.all(np.diff(archive['time_s'])>0):
                raise ValueError('prediction target axis is not ordered/unique')
    if sorted(keys)!=sorted((r['recording_id'],r['camera_id']) for r in report.get('rows',[])):
        raise ValueError('report and prediction session-camera coverage disagree')


def visual_gpu_slot(device,resource_plan,identity):
    """Atomically enforce measured eval and total limits in the training ledger."""
    import contextlib
    import os
    @contextlib.contextmanager
    def context():
        if torch.device(device).type!='cuda':
            yield {'device':str(device),'slot':None}
            return
        if resource_plan is None:
            raise ValueError('CUDA evaluation requires --resource-plan with a frozen measured resource envelope')
        plan=json.loads(Path(resource_plan).read_text())
        from configs.tools.runner import SlotLedger,check_visual_gpu_owners,validate_visual_resource_plan
        resources=validate_visual_resource_plan(plan)
        index=torch.device(device).index or 0
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu=int(visible.split(',')[index]) if visible else index
        if gpu not in resources.get('gpus',[]):
            raise ValueError('device absent from frozen resource plan')
        n=resources['runs_per_gpu']
        eval_cap=resources['eval_runs_per_gpu']
        resource_version=resources['version']
        ledger=SlotLedger(Path(plan['base_config']['control_dir'])/'gpu_slots.json')
        check_visual_gpu_owners([gpu],ledger)
        slot=ledger.claim([(gpu,s) for s in range(n)],identity,job_type='eval',
                          job_class='eval',max_job_class_per_gpu=eval_cap,
                          resource_version=resource_version)
        if slot is None:
            raise RuntimeError('evaluation GPU concurrency limit reached or all frozen GPU slots busy; defer evaluation')
        code=1
        try:
            yield {'device':str(device),'physical_gpu':gpu,'slot':slot,'resource_plan':str(resource_plan),
                   'resource_version':resource_version,'eval_runs_per_gpu':eval_cap}
            code=0
        finally:
            ledger.release(slot,identity,code)
    return context()


def visual_gt_shape_diagnostic(item,model_path):
    """Explicit GT-shape oracle; caller stores this outside main metrics."""
    from anysole.data.smpl_io import _rest_joints
    from anysole.types import JOINT_PARENTS
    rest=_rest_joints({'betas':item['gt_betas']},model_path=Path(model_path))
    offsets=np.zeros((24,3),np.float32)
    for joint,parent in enumerate(JOINT_PARENTS):
        if parent>=0:
            offsets[joint]=rest[joint]-rest[parent]
    pred=visual_predicted_joints(item,offsets)
    take=item['valid']&np.asarray(item['motion_valid'],bool)
    gt=item['gt_joints']
    value=np.linalg.norm((pred[take]-pred[take,:1])-(gt[take]-gt[take,:1]),axis=-1).mean()*1000 if take.any() else None
    return {'gt_beta_mpjpe_mm':_metric(value,take.sum(),'mm',support=item['frame_id'][take].astype(int).tolist()),
            'beta_source':'GT label oracle; diagnostic only','excluded_from_main_metrics':True}
