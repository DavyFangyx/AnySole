"""F0a evaluation protocol (fix_plan_v2.md §2): session-level, per-part, global,
temporal, contact, pressure, and robustness metrics.

eval.py drives this protocol; one run writes two same-source files per split:
``metrics/<split>.json`` (detail: per-part diagnostics plus robustness rows)
and ``metrics/<split>_brief.json`` (whole-body view: every group of the detail
file filtered to whole-body scalars, plus a ``V2T`` group in fourth position
whenever V2M carries tactile rows).  The F-series protocol numbers are what
every step F0..F9 reports:

- 局部姿态: MPJPE / PA-MPJPE over the JOINT_HIERARCHY anatomical levels
  (upper/lower partitions, pelvis/spine/head and six limb segments, plus the
  anklefoot/hands subset views).  PA-MPJPE uses one per-frame
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

修改 02（2026-11-03，接缝归因与重叠拼接推理）：eval.py ``--overlap-stride
N`` 声明的重叠窗评估不再跳过，改走重叠组装分支（assemble_overlap_seq：
三角权重混合 + 6D→rotmat→6D 投影；锚点网格保持原非重叠网格），见
assemble_overlap_seq / run_protocol(overlap_assemble=...)。
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
from anysole.utils.constraints import apply_v3e_constraints
from anysole.utils.constraints_g import apply_v3g_constraints
from anysole.utils.constraints_h import apply_v3h
from anysole.models.common.model_v3h import apply_foot_residual
from anysole.models.common.model_v3f import sample_v3f
from anysole.models.common.model_v5_1 import v5_1_z_src
from anysole.utils.geometry import (
    ar2_to_world, compose_rot6d, f2_to_world, fk_pose6d, heading_from_root_np,
    rot6d_to_rotmat, rot6d_to_rotmat_np, rotmat_to_6d, rotmat_to_6d_np,
    rot_root6d, yaw_rotmat, yaw_rotmat_np)
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
    root_trajectory_segmented_error,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)
from anysole.train import condition_inputs, model_observation_kwargs, move_batch
from anysole.types import (
    ANKLE_FOOT_JOINTS,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    ELBOW_ANGLE_TRIPLES,
    FPS,
    FOOT_JOINTS,
    JOINT_HIERARCHY,
    KNEE_ANGLE_TRIPLES,
    KNEE_ANKLE_FOOT_JOINTS,
    N_JOINTS,
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
    "root_ate_seg20_mm": "traj_ATE@0.5s",
    "root_ate_unanchored_mm": "traj_ATE",
    "root_orientation_deg": "RootOrientation",
    "root_orientation_drift_deg": "RootOrientationDrift",
    "accel_error_m_s2": "AccelError",
    "foot_sliding_mm": "FootSliding",
    "jitter_gt_m_s3": "Jitter_gt",
    "jitter_pred_m_s3": "Jitter_pred",
    "global_mpjpe_mm": "GlobalMPJPE",
    "ground_pen_mm": "GroundPen",
    "ground_pen_frac": "GroundPenFrac",
    "traj_rel_ate_mm": "trajRelATE",
}
INVERSE_DISPLAY_NAMES = {
    display: internal for internal, display in METRIC_DISPLAY_NAMES.items()
}
# Whole-body brief keys in their fixed JSON order — PA-MPJPE first so the
# primary metric is visible at a glance; doubles as the brief whitelist.
BRIEF_MOTION_KEYS = (
    "pa_mpjpe_mm", "mpjpe_mm", "global_mpjpe_mm", "mpjae_deg", "w_mpjpe100_mm",
    "root_ate_seg20_mm", "root_ate_unanchored_mm",
    "root_orientation_drift_deg", "traj_rel_ate_mm",
    "foot_sliding_mm",
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


def _apply_pos_corr(pred_kp: torch.Tensor, seq: dict) -> torch.Tensor:
    """V3_Dv1 soft skeleton: replace the six lower-limb joint positions with
    the position head's corrected ones (base FK + pelvis-relative dpos,
    world axes — the probe's target convention).  No-op without the head."""
    corr = seq.get("pos_corr")
    if corr is None:
        return pred_kp
    pred_kp = pred_kp.clone()
    pred_kp[..., KNEE_ANKLE_FOOT_JOINTS, :] = (
        pred_kp[..., KNEE_ANKLE_FOOT_JOINTS, :] + corr.reshape(-1, len(KNEE_ANKLE_FOOT_JOINTS), 3)
    )
    return pred_kp


def _session_metrics(seq: dict, tw: int, forward_axis: int, config_value: int,
                     accum: _Accum, contact: Dict[str, int]) -> None:
    """All protocol metrics of one session's full-sequence prediction."""
    T = seq["pred_pose"].shape[0]
    kp_gt = seq["kp_gt"]
    pred_kp = fk_pose6d(
        seq["pred_pose"][None], seq["pred_trans"][None], seq["offsets"], seq["parents"]
    )[0]
    pred_kp = _apply_pos_corr(pred_kp, seq)

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
    accum.add_scalar("root_rte_percent", root["root_rte_percent"], T)
    accum.add_scalar("root_ate_seg20_mm",
                     root_trajectory_segmented_error(pred_np[:, 0], gt_np[:, 0]), T)
    # 无锚未对齐 ATE（2026-11-01 用户裁定）：链式拼接 trans_hat（减去各自窗口
    # GT 锚）从 GT 起点出发——当前锚协议模型的无锚协议数字（与基线对等口径）。
    if T >= 2:
        gt_anchors = gt_np[:, 0]
        n_windows = (T + tw - 1) // tw
        unanchored = np.zeros_like(pred_np[:, 0])
        for w in range(n_windows):
            start, stop = w * tw, min((w + 1) * tw, T)
            anchor = gt_anchors[max(start - 1, 0)]
            base = unanchored[start - 1] if start > 0 else gt_anchors[0]
            unanchored[start:stop] = base + (pred_np[start:stop, 0] - anchor)
        accum.add_scalar(
            "root_ate_unanchored_mm",
            float(np.linalg.norm(unanchored - gt_anchors, axis=-1).mean() * 1000.0), T)
    # AR2 片段首帧航向系相对轨迹（2026-11-01，V7_AR方案 §4）：p_rel(t) =
    # Y(ψ_gt(t0))ᵀ [p(t) − p(t0)]，pred 与 gt 同变换后比对——自由积分
    # 轨迹的段首对齐口径（与每窗 GT 锚指标语义不同，只作声明过的相对
    # 评分）。航向相对误差 = RootOrientationDrift（自身首帧差分口径）。
    if T >= 2:
        psi0 = heading_from_root_np(
            _root_rotmats(seq["gt_pose"])[:1].cpu().numpy())
        r0_t = yaw_rotmat_np(-psi0)[0]                        # (3,3) R(ψ0)ᵀ
        p_rel_pred = (pred_np[:, 0] - pred_np[0, 0]) @ r0_t.T
        p_rel_gt = (gt_np[:, 0] - gt_np[0, 0]) @ r0_t.T
        accum.add_scalar(
            "traj_rel_ate_mm",
            float(np.linalg.norm(p_rel_pred - p_rel_gt, axis=-1).mean() * 1000.0), T)
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

    # ---- anatomical hierarchy diagnostics (JOINT_HIERARCHY single source) ----
    # Local parts use the same pelvis definition as overall MPJPE. Preserve
    # world-coordinate part errors under an explicit GlobalMPJPE prefix.
    local_err = torch.as_tensor(np.linalg.norm(pred_aligned - gt_aligned, axis=-1), device=pred_kp.device)
    world_err = torch.linalg.vector_norm(pred_kp - kp_gt, dim=-1)
    for name, joints in JOINT_HIERARCHY.items():
        if name == "overall":
            continue
        accum.add("MPJPE_%s" % name, local_err[:, joints].mean(dim=-1), 1000.0)
        accum.add("GlobalMPJPE_%s" % name, world_err[:, joints].mean(dim=-1), 1000.0)

    aligned = _pa_align(pred_kp, kp_gt)
    pa_err = torch.linalg.vector_norm(aligned - kp_gt, dim=-1)
    for name, joints in JOINT_HIERARCHY.items():
        if name == "overall":
            continue
        accum.add("PA-MPJPE_%s" % name, pa_err[:, joints].mean(dim=-1), 1000.0)

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
    if seq.get("contact_pred") is not None:
        # V3_Ev1: the T contact head's own prediction (the constraint layer's
        # input), not the kinematic fallback.
        pred_contact = seq["contact_pred"] > 0.5
    else:
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
        corr, corr_valid = _tactile_corr(pressure_pred, pressure_gt)
        if bool(corr_valid.any()):
            accum.add("T_corr", corr[corr_valid])


def _window_detail_rows(seq: dict, tw: int, frame_starts=None) -> dict:
    """Per-window pose errors of one session's full-sequence prediction.

    Windows mirror the eval loop's non-overlapping layout (stride == tw); the
    final window may be shorter when the session length is not a multiple of
    ``tw``.  Overall PA uses the canonical per-frame similarity alignment
    (metrics.py); part/region PA uses the protocol's ``_pa_align`` on the
    window-local frames, matching the pooled per-part diagnostics.  Keys are
    snake_case, matching the formulas.
    """
    T = seq["pred_pose"].shape[0]
    if frame_starts is not None and len(frame_starts) != (T + tw - 1) // tw:
        raise ValueError("Window frame IDs do not match prediction windows")
    kp_gt = seq["kp_gt"]
    pred_kp = fk_pose6d(
        seq["pred_pose"][None], seq["pred_trans"][None], seq["offsets"], seq["parents"]
    )[0]
    pred_kp = _apply_pos_corr(pred_kp, seq)
    pred_np = pred_kp.detach().cpu().numpy()
    gt_np = kp_gt.detach().cpu().numpy()
    hierarchy_keys = ["PA-MPJPE_%s" % name for name in JOINT_HIERARCHY if name != "overall"]
    rows: dict = {
        "frame_starts": [],
        "n_frames": [],
        "metrics": {key: [] for key in
                    ["pa_mpjpe_mm", "mpjpe_mm", "global_mpjpe_mm"] + hierarchy_keys},
    }
    for start in range(0, T, tw):
        end = min(start + tw, T)
        p, g = pred_np[start:end], gt_np[start:end]
        rows["frame_starts"].append(int(start if frame_starts is None else frame_starts[start // tw]))
        rows["n_frames"].append(int(end - start))
        rows["metrics"]["pa_mpjpe_mm"].append(float(pa_mpjpe(p, g).mean()))
        p_al, _ = pelvis_align(p)
        g_al, _ = pelvis_align(g)
        rows["metrics"]["mpjpe_mm"].append(float(mean_point_error(p_al, g_al, scale=1000.0).mean()))
        rows["metrics"]["global_mpjpe_mm"].append(float(mean_point_error(p, g, scale=1000.0).mean()))
        pa_w = _pa_align(pred_kp[start:end], kp_gt[start:end])
        pa_err_w = torch.linalg.vector_norm(pa_w - kp_gt[start:end], dim=-1)
        for name, joints in JOINT_HIERARCHY.items():
            if name == "overall":
                continue
            rows["metrics"]["PA-MPJPE_%s" % name].append(
                float(pa_err_w[:, joints].mean() * 1000.0))
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
    z_src_override: Optional[str] = None,
    null_t: bool = False,
    checkpoint: Optional[dict] = None,
    gain_floor: float = 0.0,
    overlap_assemble: bool = False,
    init_height: float = 0.0,
) -> Tuple[Dict[str, float], dict, List[dict], int]:
    """Full-session inference for one conditioning config, then all protocol
    metrics.  ``degrade`` in ("v", "t") adds the robustness span dropout.

    ``overlap_assemble``（修改 02，2026-11-03）：重叠 stride 下改走三角
    权重混合组装（assemble_overlap_seq），锚点网格保持原非重叠网格
    （_tile_anchor_index）。返回 ``(result, window_rows, session_rows,
    n_gt_anchors_used)``——n_gt_anchors_used 为预测侧消费的 GT 锚点数
    （标准路径 = 覆盖 tile 数 = 现协议窗数；AR2 自由积分为 0）。

    With ``dump_detail`` the per-session rows and per-window pose errors are
    collected alongside the pooled result (the pooled accumulation sequence is
    unchanged, so the pooled numbers are bit-identical either way).  Returns
    ``(result, window_rows, session_rows)``.
    """
    if overlap_assemble and (getattr(model, "prev_ctx", False)
                             or getattr(model, "ar", False)
                             or getattr(model, "f2_repr", False)):
        raise NotImplementedError(
            "--overlap-stride 目前支持标准路径与 AR2 模型（不支持 ctx/帧级"
            "AR/f2 表示）")
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel.
    v_hmr_mode = str(getattr(model, "v_input", "hrnet")) == "hmr_gvhmr"
    # F2a: heading/tilt representation.
    f2_repr = bool(getattr(model, "f2_repr", False))
    # V3_Ev1: the three conditions are three readouts of one forward.
    v3e_mode = bool(getattr(model, "v3e", False))
    # V3_Gv1: same protocol; the observation layer replaces E's constraints.
    v3g_mode = bool(getattr(model, "v3g", False))
    # V3_Hv1: same protocol; the residual channels replace E's constraints,
    # and the T2M row is the null-V prior (V zeroed upstream) + channels.
    v3h_mode = bool(getattr(model, "v3h", False))
    # V3_Fv1: the three conditions are sampler switches on one prior.
    v3f_mode = bool(getattr(model, "v3f", False))
    # V5_Av1: the three conditions are z-source readouts of one forward
    # (z_src 开关 + T 流三条件恒定); 无推理侧约束层（无锚-几何组件）。
    v5_1_mode = bool(getattr(model, "v5_1", False))
    # V4_Av1: 同一前向三种读出（config_id 被忽略，E/G 同款）; V2M = V
    # 载体（修正不应用）、VT2M = V + pos_corr 软骨架（seq 层应用）、
    # T2M = 先验 + pos_corr_prior。
    v4_1_mode = bool(getattr(model, "v4_1", False))
    v7_mode = bool(getattr(model, "v7", False))
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    window_rows = {"session_ids": [], "frame_starts": [], "n_frames": [], "metrics": {}}
    session_rows: List[dict] = []
    groups = _session_window_groups(dataset)
    session_by_id = {s["session_id"]: s for s in dataset.sessions}
    # v1/v2：三条件同一重叠数据集 → 链式；ctx3：T 行重叠数据集 → 链式，
    # V/VT 行非重叠数据集（stride==window）→ 标准路径（回归 Cv5 口径）。
    v7_ctx = (bool(getattr(model, "prev_ctx", False))
              and dataset.stride != dataset.window_length)
    ar_mode = bool(getattr(model, "ar", False))
    ar2_mode = bool(getattr(model, "ar2", False))
    ar2_regression = (bool(getattr(model, "ar2_regression", False))
                      or bool((checkpoint or {}).get("config", {}).get(
                          "ar2_regression", False)))
    ar2_dyaw_base = str((checkpoint or {}).get("config", {}).get(
        "ar2_dyaw_base", "zero"))
    half = tw // 2
    n_gt_anchors = 0
    with torch.inference_mode():
        for session_id, idxs in groups.items():
            if v7_ctx:
                # 续写协议 §4：逐窗链式生成 + 发射组装（十指标口径不变）。
                n_gt_anchors += len(idxs)   # 每窗 left−1 GT 锚（链式协议）
                seq, seq_frame_ids = _v7_ctx_session_seq(
                    dataset, model, session_id, idxs, config_value, device,
                    half, {"V_hmr": dataset[idxs[0]].get("V_hmr")} if v_hmr_mode else {},
                    hard_prefix=bool((checkpoint or {}).get("config", {}).get("v7_ctx_hardprefix", False)),
                    ctx3=bool(getattr(model, "ctx3", False)),
                    ctx4=bool(getattr(model, "ctx4", False)
                                 or getattr(model, "ctx4b", False)),
                    ctx5=bool(getattr(model, "ctx5", False)),
                    null_t=null_t)
                if dump_detail:
                    _merge_sess_detail(
                        seq, tw, forward_axis, config_value, accum, contact,
                        session_rows, window_rows, session_id=session_id,
                        frame_ids=seq_frame_ids, window_starts=None)
                else:
                    _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
                continue
            if ar2_mode:
                # AR2 自由生成（2026-11-01，V7_AR方案 §6）：逐帧串行 +
                # 外部航向状态/视觉校正 + 自由积分轨迹（无每窗 GT 锚）。
                # ar2_regression（2026-11-02）：窗口并行前向，组装同源。
                # 修改 02：overlap_assemble 时窗口间重叠 → 三角权重混合。
                # 预测侧无 GT 平移锚（自由积分）→ n_gt_anchors 不增。
                seq, ar2_frame_ids = _v7_ar2_session_seq(
                    dataset, model, session_id, idxs, config_value, device, tw,
                    {"V_hmr": dataset[idxs[0]].get("V_hmr")} if v_hmr_mode else {},
                    degrade=degrade, seed=seed, regression=ar2_regression,
                    gain_floor=gain_floor, dyaw_base=ar2_dyaw_base,
                    null_t=null_t, overlap_assemble=overlap_assemble,
                    init_height=init_height)
                if dump_detail:
                    _merge_sess_detail(
                        seq, tw, forward_axis, config_value, accum, contact,
                        session_rows, window_rows, session_id=session_id,
                        frame_ids=(ar2_frame_ids if ar2_frame_ids is not None
                                   else [int(dataset[i]["frame_start"]) + o
                                         for i in idxs for o in range(tw)]),
                        window_starts=None)
                else:
                    _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
                continue
            if ar_mode:
                n_gt_anchors += len(idxs)   # 每窗 left−1 GT 锚（帧级 AR）
                seq = _v7_ar_session_seq(
                    dataset, model, session_id, idxs, config_value, device, tw,
                    {"V_hmr": dataset[idxs[0]].get("V_hmr")} if v_hmr_mode else {},
                    degrade=degrade, seed=seed, null_t=null_t)
                if dump_detail:
                    _merge_sess_detail(
                        seq, tw, forward_axis, config_value, accum, contact,
                        session_rows, window_rows, session_id=session_id,
                        frame_ids=[int(dataset[i]["frame_start"]) + o
                                   for i in idxs for o in range(tw)],
                        window_starts=None)
                else:
                    _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
                continue
            raw = [dataset[i] for i in idxs]
            batch = move_batch(collate_windows(raw), device)
            bsz = batch["pose_gt"].shape[0]
            # 预测侧 GT 平移锚点消费数（登记口径）：非重叠协议每窗一个
            # （left−1）；重叠协议保持 tile 网格 → 覆盖 tile 数，两者逐位
            # 相同（修改 02 §B4）。
            n_gt_anchors += len({int(r["frame_start"]) // tw for r in raw})
            if overlap_assemble:
                # GT 泄漏硬约束（修改 02 §B4）：锚点网格保持原非重叠网格
                # （仅 tile*tw 处取锚），重叠窗继承所属 tile 的锚；
                # stride=tw 时 tile = 窗序号 → 与现协议逐位相同。
                tile_anchors = []
                for r in raw:
                    s = session_by_id[r["session_id"]]
                    tile_anchors.append(np.asarray(s["trans_global"])[
                        _tile_anchor_index(int(r["frame_start"]), tw)])
                batch["trans_anchor"] = torch.from_numpy(
                    np.stack(tile_anchors)).float().to(device)
            config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            if v3f_mode:
                # T stays live under every condition (the contact head reads
                # the real stream); condition_inputs already zeroed V for
                # T2M — exactly the unconditional pass.
                t_raw, t_phys = batch["T_raw"], batch["T_phys"]
            if v5_1_mode or v4_1_mode:
                # V5_Av1/V4_Av1：T 流输入三条件恒定（g_T/接触头/XAttn 恒读
                # 真实压力）。
                t_raw, t_phys = batch["T_raw"], batch["T_phys"]
            if null_t:
                # V4_Av1 判据 ③ null-T 注入（eval --null-t）：T 输入整窗
                # 置零后前向，训练正确的修正分支应输出 ≈0 → VT 行读数
                # 退回 V2M（两次跑配对比较 = 干预实验读数）。
                t_raw = torch.zeros_like(t_raw)
                t_phys = torch.zeros_like(t_phys)
                t_s2m = torch.zeros_like(t_s2m)
            if degrade is not None:
                rng = np.random.RandomState((seed * 1000003) ^ zlib.crc32(str(session_id).encode()))
                v_feat, t_raw, t_phys, t_s2m = _degrade_inputs(
                    v_feat, t_raw, t_phys, t_s2m, tw, degrade, rng
                )
            v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
            if v3f_mode:
                out = sample_v3f(
                    model, v_feat, t_raw, t_phys,
                    anchor=batch["trans_anchor"][:, None, :],
                    floor_y=batch["floor_y"],
                    offsets=batch["offsets"], parents=batch["parents"],
                    use_t=config_value != CONFIG_V,
                    refine=config_value != CONFIG_V,
                    diffusion=diffusion, steps=sample_steps, eta=0.0,
                    device=device)
                pred_pose = out["x0_hat"]
                v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
            elif regress_mode:
                z_kw = {"z_src": v5_1_z_src(config_value, z_src_override)} if v5_1_mode else {}
                out = model(v_feat, t_raw, t_phys, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw, **z_kw)
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
            if not v3f_mode:
                v3e_contact = None
            if v7_mode and config_value != CONFIG_V:
                # V7（2026-10-25 预检项）：接触头指标直读 contact_logits
                # （T 在场行）；V-only 行 T 分支硬关闭、接触头读零输入 →
                # 该行保留几何回退（不声称是接触头读数）。
                v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
            if getattr(model, "ctx4b", False):
                # v4-b（2026-11-01）：V/VT 行出口组合 = 锚估计头 R_a ⊙ 残差
                # （T 行走链式路径；本分支防御：GT 锚 teacher-forced）。
                if config_value == CONFIG_T:
                    R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])
                else:
                    R_a = out["anchor_rot"]
                pred_pose = rot_root6d(R_a[:, None, :, :], out["x0_hat"])
                v_w = torch.einsum("bij,btj->bti", R_a, out["v_hat"])
                pred_trans_w = torch.cumsum(v_w, dim=1) / float(FPS) + anchor
                v4_1_corr = None
            elif v3e_mode:
                # V3_Ev1 condition selection: VT = constraint layer on the
                # full forward, V = raw V pose, T = the T pose prior.
                if config_value == CONFIG_VT:
                    pred_pose, pred_trans_w = apply_v3e_constraints(
                        out["x0_hat"], out["trans_hat"] + anchor,
                        batch["floor_y"],
                        batch["offsets"], batch["parents"])
                    v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                elif config_value == CONFIG_V:
                    pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                else:
                    pred_pose, pred_trans_w = out["x0_prior"], out["trans_prior"] + anchor
            elif v3g_mode:
                # V3_Gv1 condition selection: VT = 分部观测层 on the V stream,
                # V = raw carrier (= E), T = T prior + 观测层.  floor_src
                # stays the registered default "gt" (E v2.1 口径).
                if config_value == CONFIG_VT:
                    pred_pose, pred_trans_w = apply_v3g_constraints(
                        out["x0_hat"], out["trans_hat"] + anchor,
                        out["contact_logits"],
                        batch["offsets"], batch["parents"],
                        floor_y=batch["floor_y"])
                    v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                elif config_value == CONFIG_V:
                    pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                else:
                    pred_pose, pred_trans_w = apply_v3g_constraints(
                        out["x0_prior"], out["trans_prior"] + anchor,
                        out["contact_logits"],
                        batch["offsets"], batch["parents"],
                        floor_y=batch["floor_y"])
            elif v3h_mode:
                # V3_Hv1 condition selection: VT = 残差通道 on the V stream,
                # V = raw V tower, T = null-V 先验 + 残差通道 (H 无 T 塔——
                # 同一 V 塔、V 已由 condition_inputs 零化).  floor_src stays
                # the registered default "gt" (E v2.1 口径).
                # V3_Hv3：VT/T 行先应用一阶足部残差（V2M 行不应用）。
                pose_vt = apply_foot_residual(
                    out["x0_hat"], out["foot_res"],
                    landing=str(getattr(model, "res_landing", "foot"))) if out.get("foot_res") is not None else out["x0_hat"]
                if config_value == CONFIG_VT:
                    pred_pose, pred_trans_w = apply_v3h(
                        bool(getattr(model, "no_anchor", False)),
                        pose_vt, out["trans_hat"] + anchor,
                        out["contact_logits"],
                        batch["offsets"], batch["parents"],
                        floor_y=batch["floor_y"])
                    v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                elif config_value == CONFIG_V:
                    pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                else:
                    pred_pose, pred_trans_w = apply_v3h(
                        bool(getattr(model, "no_anchor", False)),
                        pose_vt, out["trans_hat"] + anchor,
                        out["contact_logits"],
                        batch["offsets"], batch["parents"],
                        floor_y=batch["floor_y"])
            elif v5_1_mode:
                # V5_Av1 condition selection: 三行直接读 z_src 选定的读出
                # （前向已按条件切换；无推理侧约束层——无锚-几何组件，
                # 穿透/Global 行只作报告）。VT 行接触读数 = G 配方接触头。
                if config_value == CONFIG_VT:
                    v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                v4_1_corr = None
            elif v4_1_mode:
                # V4_Av1 condition selection: 三读出 + 修正分支开关。
                # V2M = V 载体（修正不应用，契约 1 守卫）；VT2M = V +
                # pos_corr（seq 层软骨架）；T2M = 先验 + pos_corr_prior。
                if config_value == CONFIG_VT:
                    v3e_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                    pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                    v4_1_corr = out["pos_corr"]
                elif config_value == CONFIG_V:
                    pred_pose, pred_trans_w = out["x0_hat"], out["trans_hat"] + anchor
                    v4_1_corr = None
                else:
                    pred_pose, pred_trans_w = out["x0_prior"], out["trans_prior"] + anchor
                    v4_1_corr = out["pos_corr_prior"]
            else:
                v4_1_corr = None
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
                    "pos_corr": (out["pos_corr"].reshape(-1, out["pos_corr"].shape[-1])
                                 if out.get("pos_corr") is not None else None),
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
                if v4_1_mode:
                    # V4_Av1 契约 1：V2M 行 seq 修正必须为 None（v4_1_corr
                    # 按条件为空）——不得回退到 out["pos_corr"]（V3_Dv1
                    # 遗留口径会污染 V2M 行）。
                    seq_pos_corr = (v4_1_corr.reshape(-1, v4_1_corr.shape[-1])
                                    if v4_1_corr is not None else None)
                else:
                    seq_pos_corr = (out["pos_corr"].reshape(-1, out["pos_corr"].shape[-1])
                                    if out.get("pos_corr") is not None else None)
                seq = {
                    "pred_pose": pred_pose.reshape(-1, pred_pose.shape[-1]),
                    "pred_trans": (pred_trans_w if (v3e_mode or v3g_mode or v3h_mode or v5_1_mode or v4_1_mode)
                                   else out["trans_hat"] + anchor).reshape(-1, 3),
                    "gt_pose": batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1]),
                    "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3),
                    "pos_corr": seq_pos_corr,
                    "contact_pred": (v3e_contact.reshape(-1, v3e_contact.shape[-1])
                                     if v3e_contact is not None else None),
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
            if overlap_assemble:
                # 修改 02 §B3：seq 逐帧键按窗拆回 → 三角权重混合组装
                # （单覆盖帧直接拷贝 → stride=tw 与现协议逐位一致）。
                per_frame = {key: seq[key] for key in (
                    "pred_pose", "pred_trans", "gt_pose", "gt_trans",
                    "pos_corr", "contact_pred", "kp_gt", "contact_gt",
                    "pressure_gt", "pressure_pred")}
                parts = []
                for b in range(bsz):
                    parts.append((int(raw[b]["frame_start"]), {
                        key: (val.view(bsz, tw, *val.shape[1:])[b]
                              if val is not None else None)
                        for key, val in per_frame.items()}))
                seq, seq_frame_ids = assemble_overlap_seq(parts, tw)
                seq.update({"floor_y": batch["floor_y"][0],
                            "offsets": batch["offsets"][0],
                            "parents": batch["parents"][0],
                            "betas": batch["betas"][0]})
            else:
                seq_frame_ids = None
            if dump_detail:
                # Per-session accumulation, merged into the pooled accum in the
                # same summation sequence as the direct path, so the pooled
                # result stays bit-identical with or without the dump.
                _merge_sess_detail(
                    seq, tw, forward_axis, config_value, accum, contact,
                    session_rows, window_rows, session_id=session_id,
                    frame_ids=(seq_frame_ids if seq_frame_ids is not None else
                               [int(sample["frame_start"]) + offset
                                for sample in raw for offset in range(tw)]),
                    window_starts=None if overlap_assemble else
                    [int(sample["frame_start"]) for sample in raw])
            else:
                _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
            if (config_value == CONFIG_V and v2t_out_dir is not None
                    and seq.get("pressure_pred") is not None):
                pred_kp = fk_pose6d(
                    seq["pred_pose"][None], seq["pred_trans"][None],
                    seq["offsets"], seq["parents"],
                )[0]
                pred_kp = _apply_pos_corr(pred_kp, seq)
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
                if overlap_assemble:
                    # 修改 02：组装后的逐帧值按 frame id 散回会话网格
                    # （与逐窗散回等价；重叠帧已混合）。
                    for t, f in enumerate(seq_frame_ids):
                        pressure_pred[f] = seq["pressure_pred"][t].detach().cpu().numpy()
                        pressure_gt[f] = seq["pressure_gt"][t].detach().cpu().numpy()
                        contact_pred_full[f] = pred_contact[t].astype(np.uint8)
                        contact_gt_full[f] = gt_contact[t].astype(np.uint8)
                        seen[f] = True
                else:
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
    return result, window_rows, session_rows, n_gt_anchors


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _triangle_window_weights(tw: int) -> list:
    """修改 02（2026-11-03）三角权重：w(k) = 1 − |k − (tw−1)/2| / (tw/2)。

    标准偶数长三角窗：窗中心 0.95、窗缘 1/tw > 0（窗缘非零保证会话首尾
    单覆盖帧权重合法）；每帧按覆盖窗权重归一化，峰值绝对值不影响结果。
    对称 w(k) = w(tw−1−k)、从窗缘到中心单调增。
    """
    half = (tw - 1) / 2.0
    return [1.0 - abs(k - half) / (tw / 2.0) for k in range(tw)]


def _tile_anchor_index(left: int, tw: int) -> int:
    """GT 泄漏硬约束（修改 02 §B4）：锚点网格保持原非重叠网格——仅 tile
    首帧 tw*t 处取锚（索引 max(tw*t−1, 0)），重叠窗继承所属 tile 的锚。
    stride=tw 时 left = tile*tw → 与现协议 per-window 锚（left−1）逐位
    相同；n_gt_anchors_used = 覆盖 tile 数 = 现协议窗数。"""
    return max((left // tw) * tw - 1, 0)


def prune_overlap_windows(valid_windows, tw: int) -> list:
    """GT 泄漏硬约束（修改 02 §B4，2026-11-03 s5 扫描发现并修正）：只保留
    tile 非重叠窗本身有效的窗口。

    问题：tile 首窗（left = t*tw）因 fake/监督缺口无效时，同 tile 的平移
    窗（left = t*tw+k）可能有效——直接使用会 (a) 引入现协议从未消费的
    GT 锚（tile 锚 trans_global[t*tw−1]），n_gt_anchors_used 不再逐位
    相同；(b) 扩大评分帧集（覆盖基线未评分的帧）。剔除后：覆盖帧集 =
    现协议覆盖帧集、锚点集 = 现协议锚点集，逐位一致。stride=tw 时恒等
    （每个窗即其 tile 窗）。"""
    tile_ok = {(fi, left) for fi, left, _r in valid_windows if left % tw == 0}
    return [w for w in valid_windows
            if (w[0], (w[1] // tw) * tw) in tile_ok]


_FRAME_PROJ6D_KEYS = ("pred_pose",)
_FRAME_LINEAR_KEYS = ("pred_trans", "pos_corr", "pressure_pred")
_FRAME_FIRST_WIN_KEYS = ("gt_pose", "gt_trans", "kp_gt", "contact_gt",
                         "pressure_gt", "contact_pred")


def assemble_overlap_seq(parts, tw):
    """修改 02 §B3：重叠窗组装为会话级 seq（覆盖帧升序）。

    parts = [(start, {key: tensor (tw, ...) 或 None})]，start 升序。
    返回 (seq, frame_ids)：
    - 单覆盖帧直接拷贝（stride=tw 时输出与现协议 reshape 拼接逐位一致
      ——等价性硬门的基础）；
    - 多覆盖帧：pred_pose（6D）按三角权重加权平均后经 6D→rotmat→6D 往返
      投影回 SO(3)（geometry.py 助手，不自写正交化）；pred_trans /
      pos_corr / pressure_pred 线性加权；
    - GT 键与 contact_pred：取首覆盖窗的值（标准路径同帧跨窗逐位一致；
      AR2 的窗口相对积分 GT 按声明取首覆盖窗）。
    meta 键（floor_y/offsets/parents/betas）由调用方补入。
    """
    weights = _triangle_window_weights(tw)
    coverage: Dict[int, list] = {}
    for i, (start, _d) in enumerate(parts):
        for k in range(tw):
            coverage.setdefault(start + k, []).append((i, k))
    frames = sorted(coverage)
    if not frames:
        raise ValueError("assemble_overlap_seq: no covered frames")
    first = parts[0][1]
    out: Dict[str, Optional[torch.Tensor]] = {}
    for key, val0 in first.items():
        if val0 is None:
            out[key] = None
            continue
        tail_shape = val0.shape[1:]
        rows = []
        for f in frames:
            cov = coverage[f]
            i0, k0 = cov[0]
            if len(cov) == 1 or key in _FRAME_FIRST_WIN_KEYS:
                rows.append(parts[i0][1][key][k0])
                continue
            vals = torch.stack([parts[i][1][key][k] for i, k in cov], dim=0)
            w = torch.tensor([weights[k] for _, k in cov],
                             dtype=torch.float64, device=vals.device)
            wv = w.view(-1, *([1] * len(tail_shape)))
            mean = (vals.to(torch.float64) * wv).sum(0) / w.sum()
            if key in _FRAME_PROJ6D_KEYS:
                # 6D 加权平均 → 投影回 SO(3)（rotmat 列正交化 + 取前两列）
                rows.append(rotmat_to_6d(rot6d_to_rotmat(
                    mean.reshape(-1, 6).float())).reshape(tail_shape))
            else:
                rows.append(mean.to(vals.dtype))
        out[key] = torch.stack(rows, dim=0)
    return out, frames


def blend_overlap_windows_np(windows, tw):
    """修改 02：导出侧重叠窗混合（npz 口径 = 协议口径）。

    windows = [(start, pose (tw,144) f32, trans (tw,3) f32, gt_trans (tw,3)
    f32)]，start 升序。返回 (pose (T,144), trans (T,3), frames (T,) int64)
    覆盖帧升序；单覆盖帧直接拷贝（stride=tw 逐位 = 现导出），多覆盖帧
    三角权重线性平均 + pose 6D→rotmat→6D 投影（numpy 助手，与
    geometry.py 同口径）。gt_trans 取首覆盖窗（与协议 seq 一致）。
    """
    weights = _triangle_window_weights(tw)
    coverage: Dict[int, list] = {}
    for i, (start, _pose, _trans, _gt) in enumerate(windows):
        for k in range(tw):
            coverage.setdefault(start + k, []).append((i, k))
    frames = sorted(coverage)
    if not frames:
        raise ValueError("blend_overlap_windows_np: no covered frames")
    pose = np.empty((len(frames), windows[0][1].shape[-1]), dtype=np.float32)
    trans = np.empty((len(frames), 3), dtype=np.float32)
    for t, f in enumerate(frames):
        cov = coverage[f]
        i0, k0 = cov[0]
        if len(cov) == 1:
            pose[t] = windows[i0][1][k0]
            trans[t] = windows[i0][2][k0]
            continue
        w = np.asarray([weights[k] for _, k in cov], dtype=np.float64)
        w = w / w.sum()
        pose_t = (np.stack(
            [windows[i][1][k].astype(np.float64) for i, k in cov])
            * w[:, None]).sum(0)
        trans_t = (np.stack(
            [windows[i][2][k].astype(np.float64) for i, k in cov])
            * w[:, None]).sum(0)
        pose[t] = rotmat_to_6d_np(rot6d_to_rotmat_np(
            pose_t.reshape(-1, 6))).reshape(-1).astype(np.float32)
        trans[t] = trans_t.astype(np.float32)
    return pose, trans, np.asarray(frames, dtype=np.int64)


def ar2_session_chunks(dataset, model, session_id, idxs, config_value,
                       device, tw, v_kw, *, degrade=None, seed=0,
                       regression=False, gain_floor=0.0, dyaw_base="zero",
                       null_t=False, init_height=0.0):
    """AR2 自由生成（2026-11-01，V7_AR方案 §4/§6）：逐窗逐帧串行。

    外部状态：ψ（累计航向）、p（自由积分位置）、vis_ema（视觉航向观测
    的 EMA，0.5 平滑防注入 V 帧级抖动）、prev（148 维相对反馈）。
    V/VT 行：ψ += Δψ̂ 后应用视觉校正 ψ += g·wrap(vis_ema − ψ)（g 只读
    运行时证据）；T 行：纯积分 ψ += Δψ̂，会话首帧 ψ=0（坐标约定）。
    世界组装逐帧：R_root = Y(ψ) @ tilt、v_world = Y(ψ) @ v_head、
    p += v_world/FPS——无每窗 GT 锚（§3.4）。会话开始/跳窗复位。
    返回 (chunks, meta)：chunks 逐窗含 world pred_pose/pred_trans 与
    GT（供评分与导出共用，方案 §6.5 同一序列口径）。

    regression（2026-11-02，V7_Cv5AR2R）：整窗一次并行前向（双向注意
    力、无反馈、无因果掩码），逐帧几何组装与串行完全同源（同一积分
    公式、同一外部状态与复位规则）——差异只在 Δyaw/tilt/v_head 的来
    源：一次并行输出 vs 逐帧依赖上一帧反馈。

    null_t（修改 03，2026-11-03）：T 输入置零 = 当前窗 T 流（t_raw/
    t_phys/t_s2m）+ AR 观测历史（obs_history 的 T_raw/T_phys）全部
    置零（T2M 下 V 已由 condition_inputs 置零 → 无条件先验）。不传
    开关（null_t=False）时行为逐位不变。
    """
    chunks = []
    prev = torch.zeros(148, device=device)
    psi = torch.zeros((), device=device, dtype=torch.float64)
    # 先验初始高度（2026-11-03 修复）：自由积分从原点起步没有高度参照，
    # 水平行走垂直速度均值≈0 → 骨盆恒贴地、足入地 0.8m（GroundPen 855/
    # Global 1271/traj 10.6m 同源）。先验 = train 拟合骨盆高度常数（声明
    # 常量，非逐会话 GT）；默认 0 = 旧口径逐位不变。
    pos = torch.tensor([0.0, float(init_height), 0.0],
                       device=device, dtype=torch.float64)
    vis_ema = None
    prev_start = None
    meta = None
    use_vis = config_value != CONFIG_T
    dropped = set()
    if degrade is not None:
        rng = np.random.RandomState((seed * 1000003) ^ zlib.crc32(str(session_id).encode()))
        frame_ids = [int(dataset[i]["frame_start"]) + t for i in idxs for t in range(tw)]
        span = int(float(rng.uniform(0.2, 0.4)) * len(frame_ids))
        left = int(rng.uniform(0., max(len(frame_ids) - span, 1)))
        dropped = set(frame_ids[left:left + span])
    for i in idxs:
        raw = dataset[i]
        batch = move_batch(collate_windows([raw]), device)
        start = int(raw["frame_start"])
        if prev_start is None or start != prev_start + tw:
            prev = torch.zeros(148, device=device)
            psi = torch.zeros((), device=device, dtype=torch.float64)
            pos = torch.tensor([0.0, float(init_height), 0.0],
                               device=device, dtype=torch.float64)
            vis_ema = None
        config_id = torch.full((1,), config_value, device=device, dtype=torch.long)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
        obs_kw = model_observation_kwargs(batch)
        if null_t:
            # 修改 03（2026-11-03）：null-T 注入（T 内容全部置零）。
            t_raw = torch.zeros_like(t_raw)
            t_phys = torch.zeros_like(t_phys)
            t_s2m = torch.zeros_like(t_s2m)
            if obs_kw:
                history = {key: value.clone()
                           for key, value in batch["obs_history"][0].items()}
                history["T_raw"] = torch.zeros_like(history["T_raw"])
                history["T_phys"] = torch.zeros_like(history["T_phys"])
                obs_kw = {"obs_history": [history]}
        if dropped:
            mask = torch.tensor([start + t in dropped for t in range(tw)], device=device)
            if degrade == "v":
                v_feat = v_feat.clone()
                v_feat[:, mask] = 0
            else:
                t_raw, t_phys, t_s2m = t_raw.clone(), t_phys.clone(), t_s2m.clone()
                t_raw[:, mask], t_phys[:, mask], t_s2m[:, mask] = 0, 0, 0
            if obs_kw:
                history = {key: value.clone() for key, value in batch["obs_history"][0].items()}
                length = history["V_feat"].shape[0]
                past_mask = torch.tensor([start - length + t in dropped for t in range(length)],
                                         dtype=torch.bool)
                for key in (("V_feat",) if degrade == "v" else ("T_raw", "T_phys")):
                    history[key][past_mask] = 0
                obs_kw = {"obs_history": [history]}
        anchor = batch["trans_anchor"][:, None, :]
        ps = torch.zeros(1, tw, 148, device=device)
        ps[0, 0] = prev
        poses, trans = [], []
        if regression:
            # 纯回归（2026-11-02）：一次并行前向，逐帧几何组装与串行
            # 同源（同一积分公式与外部状态），无逐帧模型前向。
            out = model(v_feat, t_raw, t_phys, config_id, [session_id],
                        T_s2m=t_s2m, **v_kw, prev_state=None, causal=False,
                        **obs_kw)
            dyaw_all = out["dyaw_hat"][0, :, 0].detach().double()   # (tw,)
            rel_all = out["x0_hat"][0].detach().float()             # (tw,144)
            v_all = out["v_hat"][0].detach().float()                # (tw,3)
            vis_yaw_all = (out["vis_yaw"][0, :, 0].detach().double()
                           if use_vis else None)
            gain_all = (out["vis_gain"][0, :, 0].detach().double()
                        if use_vis else None)
            # 正面 Δψ 结构（2026-11-02）：Δψ_total = 历史延续基线（帧 0 =
            # 上一窗末帧 Δψ_total、会话首帧 0，帧 t>0 = Δψ_total(t−1)）+
            # 修正项；dyaw_base != "history" 时基线为零（旧 ckpt 语义）。
            d_prev = (prev[0].double() if dyaw_base == "history"
                      else torch.zeros((), dtype=torch.float64, device=device))
            for t in range(tw):
                dyaw_t = dyaw_all[t]
                d_total = dyaw_t + d_prev
                d_prev = d_total
                rel_pose_t = rel_all[t]
                v_head_t = v_all[t]
                psi = psi + d_total
                if use_vis:
                    vis_yaw_t = vis_yaw_all[t]
                    # 增益下限（2026-11-02 诊断钩子）：训练增益中位数仅
                    # 0.005（L_yawcorr 旧口径最优解 = 0），下限压住后量化
                    # 「校正若真生效」的 drift/PA 上限；默认 0 = 关。
                    gain_t = gain_all[t].clamp(min=gain_floor)
                    vis_ema = (vis_yaw_t if vis_ema is None
                               else 0.5 * vis_ema + 0.5 * vis_yaw_t)
                    psi = psi + gain_t * _wrap_angle(vis_ema - psi)
                r_yaw = yaw_rotmat(psi.reshape(1))[0].float()      # (3,3)
                r_root = torch.matmul(r_yaw, rot6d_to_rotmat(rel_pose_t[:6]))
                v_world = torch.matmul(r_yaw, v_head_t)
                pos = pos + v_world.double() / float(FPS)
                poses.append(torch.cat([rotmat_to_6d(r_root), rel_pose_t[6:]]))
                trans.append(pos.float().clone())
        else:
            for t in range(tw):
                out = model(v_feat, t_raw, t_phys, config_id, [session_id],
                            T_s2m=t_s2m, **v_kw, prev_state=ps, causal=True,
                            **obs_kw)
                dyaw_t = out["dyaw_hat"][0, t, 0].detach().double()
                d_total = (dyaw_t + ps[0, t, 0].double()
                           if dyaw_base == "history" else dyaw_t)
                rel_pose_t = out["x0_hat"][0, t].detach().float()
                v_head_t = out["v_hat"][0, t].detach().float()
                psi = psi + d_total
                if use_vis:
                    vis_yaw_t = out["vis_yaw"][0, t, 0].detach().double()
                    gain_t = out["vis_gain"][0, t, 0].detach().double().clamp(
                        min=gain_floor)
                    vis_ema = (vis_yaw_t if vis_ema is None
                               else 0.5 * vis_ema + 0.5 * vis_yaw_t)
                    psi = psi + gain_t * _wrap_angle(vis_ema - psi)
                r_yaw = yaw_rotmat(psi.reshape(1))[0].float()      # (3,3)
                r_root = torch.matmul(r_yaw, rot6d_to_rotmat(rel_pose_t[:6]))
                v_world = torch.matmul(r_yaw, v_head_t)
                pos = pos + v_world.double() / float(FPS)
                poses.append(torch.cat([rotmat_to_6d(r_root), rel_pose_t[6:]]))
                trans.append(pos.float().clone())
                if t + 1 < tw:
                    ps[0, t + 1] = torch.cat([d_total.float().reshape(1),
                                              rel_pose_t, v_head_t])
        pred_pose = torch.stack(poses, dim=0)                  # (tw, 144) 世界
        pred_trans = torch.stack(trans, dim=0)                 # (tw, 3) 自由积分
        world_gt_pose, _ = ar2_to_world(
            batch["pose_gt"], batch["dyaw_gt"], batch["v_head_gt"],
            batch["psi_base"], batch["trans_anchor"])
        chunks.append({
            "pred_pose": pred_pose.cpu(),
            "pred_trans": pred_trans.cpu(),
            "gt_pose": world_gt_pose[0].cpu(),
            "gt_trans": (batch["trans_gt"] + anchor)[0].cpu(),
            "kp_gt": batch["kp_gt"][0].cpu(),
            "contact_gt": batch["contact_gt"][0].cpu(),
            "contact_pred": (torch.sigmoid(out["contact_logits"]) > 0.5)[0].detach().cpu()
                             if config_value != CONFIG_V else None,
            "pressure_gt": batch["T_raw"][0].cpu(),
            "start": start,
        })
        prev = torch.cat([d_total.double().float().reshape(1),
                          rel_pose_t, v_head_t])
        prev_start = start
        meta = (batch["floor_y"][0].cpu(), batch["offsets"][0].cpu(),
                batch["parents"][0].cpu(), batch["betas"][0].cpu())
    return chunks, meta


def _v7_ar2_session_seq(dataset, model, session_id, idxs, config_value,
                        device, tw, v_kw, *, degrade=None, seed=0,
                        regression=False, gain_floor=0.0, dyaw_base="zero",
                        null_t=False, overlap_assemble=False,
                        init_height=0.0):
    """AR2 协议会话序列：ar2_session_chunks 的 seq 组装（评分口径）。

    regression：V7_Cv5AR2R 纯回归格（窗口并行前向，组装公式同源）。
    修改 02：overlap_assemble 时窗口重叠 → 三角权重混合组装
    （assemble_overlap_seq），返回 (seq, frame_ids)。
    null_t（修改 03，2026-11-03）：T 输入（当前窗 + AR 观测历史）整窗
    置零；T2M + null-T = 无条件先验。不传开关行为逐位不变。"""
    chunks, meta = ar2_session_chunks(dataset, model, session_id, idxs,
                                      config_value, device, tw, v_kw,
                                      degrade=degrade, seed=seed,
                                      regression=regression,
                                      gain_floor=gain_floor,
                                      dyaw_base=dyaw_base, null_t=null_t,
                                      init_height=init_height)
    floor_y, offsets, parents, betas = meta
    if overlap_assemble:
        parts = []
        for c in chunks:
            parts.append((int(c["start"]), {
                "pred_pose": c["pred_pose"], "pred_trans": c["pred_trans"],
                "gt_pose": c["gt_pose"], "gt_trans": c["gt_trans"],
                "kp_gt": c["kp_gt"], "contact_gt": c["contact_gt"],
                "contact_pred": c["contact_pred"],
                "pressure_gt": c["pressure_gt"]}))
        seq, frame_ids = assemble_overlap_seq(parts, tw)
        seq.update({"pressure_pred": None, "pos_corr": None,
                    "floor_y": floor_y, "offsets": offsets,
                    "parents": parents, "betas": betas})
        return seq, frame_ids
    seq = {
        "pred_pose": torch.cat([c["pred_pose"] for c in chunks], dim=0),
        "pred_trans": torch.cat([c["pred_trans"] for c in chunks], dim=0),
        "gt_pose": torch.cat([c["gt_pose"] for c in chunks], dim=0),
        "gt_trans": torch.cat([c["gt_trans"] for c in chunks], dim=0),
        "kp_gt": torch.cat([c["kp_gt"] for c in chunks], dim=0),
        "contact_gt": torch.cat([c["contact_gt"] for c in chunks], dim=0),
        "contact_pred": (torch.cat([c["contact_pred"] for c in chunks], dim=0)
                         if config_value != CONFIG_V else None),
        "pressure_gt": torch.cat([c["pressure_gt"] for c in chunks], dim=0),
        "pressure_pred": None,
        "pos_corr": None,
        "floor_y": floor_y, "offsets": offsets, "parents": parents, "betas": betas,
    }
    return seq, None


def _v7_ar_session_seq(dataset, model, session_id, idxs, config_value,
                       device, tw, v_kw, *, degrade=None, seed=0,
                       null_t=False):
    """帧级 AR（2026-11-01，V7 方案 §11）：逐窗逐帧串行生成——反馈 =
    上一帧预测输出 [姿态 144 + 根速度 3]（ps 槽位逐帧填充，因果掩码下
    帧 t 只依赖 ≤t）；窗口间携带上一窗末帧状态；首窗/跳窗起点 = 全零
    （与训练帧 0 一致）。整窗发射（非重叠窗口，无需 emit-half）。

    null_t（修改 03，2026-11-03）：T 输入置零（与标准路径同语义）。
    """
    chunks = []
    prev = torch.zeros(147, device=device)
    prev_start = None
    meta = None
    degraded = None
    if degrade is not None:
        # Sample one session-wide dropout span, as in the non-AR protocol.
        whole = move_batch(collate_windows([dataset[i] for i in idxs]), device)
        cid = torch.full((len(idxs),), config_value, device=device, dtype=torch.long)
        streams = condition_inputs(whole, cid)
        if null_t:
            streams = tuple(torch.zeros_like(s) for s in streams)
        rng = np.random.RandomState((seed * 1000003) ^ zlib.crc32(str(session_id).encode()))
        degraded = _degrade_inputs(*streams, tw, degrade, rng)
    for window_index, i in enumerate(idxs):
        raw = dataset[i]
        batch = move_batch(collate_windows([raw]), device)
        start = int(raw["frame_start"])
        if prev_start is None or start != prev_start + tw:
            prev = torch.zeros(147, device=device)
        config_id = torch.full((1,), config_value, device=device, dtype=torch.long)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
        if null_t:
            # 修改 03（2026-11-03）：null-T 注入（T 内容全部置零）。
            t_raw = torch.zeros_like(t_raw)
            t_phys = torch.zeros_like(t_phys)
            t_s2m = torch.zeros_like(t_s2m)
        if degraded is not None:
            v_feat, t_raw, t_phys, t_s2m = (
                stream[window_index:window_index + 1] for stream in degraded)
        anchor = batch["trans_anchor"][:, None, :]
        ps = torch.zeros(1, tw, 147, device=device)
        ps[0, 0] = prev
        poses, vels = [], []
        for t in range(tw):
            out = model(v_feat, t_raw, t_phys, config_id, [session_id],
                        T_s2m=t_s2m, **v_kw, prev_state=ps, causal=True)
            poses.append(out["x0_hat"][0, t].detach())
            vels.append(out["v_hat"][0, t].detach())
            if t + 1 < tw:
                ps[0, t + 1] = torch.cat([poses[-1], vels[-1]])
        pred_pose = torch.stack(poses, dim=0)              # (tw, 144)
        v_hat = torch.stack(vels, dim=0)                   # (tw, 3)
        pred_trans = torch.cumsum(v_hat, dim=0) / float(FPS) + anchor[0]
        chunks.append({
            "pred_pose": pred_pose.cpu(),
            "pred_trans": pred_trans.cpu(),
            "gt_pose": batch["pose_gt"][0].cpu(),
            "gt_trans": (batch["trans_gt"] + anchor)[0].cpu(),
            "kp_gt": batch["kp_gt"][0].cpu(),
            "contact_gt": batch["contact_gt"][0].cpu(),
            "contact_pred": (torch.sigmoid(out["contact_logits"]) > 0.5)[0].detach().cpu()
                             if config_value != CONFIG_V else None,
            "pressure_gt": batch["T_raw"][0].cpu(),
        })
        prev = torch.cat([poses[-1], vels[-1]])
        prev_start = start
        meta = (batch["floor_y"][0].cpu(), batch["offsets"][0].cpu(),
                batch["parents"][0].cpu(), batch["betas"][0].cpu())
    floor_y, offsets, parents, betas = meta
    seq = {
        "pred_pose": torch.cat([c["pred_pose"] for c in chunks], dim=0),
        "pred_trans": torch.cat([c["pred_trans"] for c in chunks], dim=0),
        "gt_pose": torch.cat([c["gt_pose"] for c in chunks], dim=0),
        "gt_trans": torch.cat([c["gt_trans"] for c in chunks], dim=0),
        "kp_gt": torch.cat([c["kp_gt"] for c in chunks], dim=0),
        "contact_gt": torch.cat([c["contact_gt"] for c in chunks], dim=0),
        "contact_pred": (torch.cat([c["contact_pred"] for c in chunks], dim=0)
                         if config_value != CONFIG_V else None),
        "pressure_gt": torch.cat([c["pressure_gt"] for c in chunks], dim=0),
        "pressure_pred": None,
        "pos_corr": None,
        "floor_y": floor_y, "offsets": offsets, "parents": parents, "betas": betas,
    }
    return seq


def _v7_ctx_session_seq(dataset, model, session_id, idxs, config_value,
                        device, half, v_kw, hard_prefix=False,
                        ctx3=False, ctx4=False, ctx5=False, null_t=False):
    """续写协议 §4：按会话逐窗链式生成（ctx = 上一窗预测在 half 处的根
    状态 [根 6D + 根速度 3 + 有效 1]；首窗/跳窗 → 零 + flag 0），按发射
    规则组装（首窗全 20 帧、后续窗只发新 half；窗口间 fake 帧缺口不在
    评分范围，与现协议 valid_windows 评分帧一致）→ 会话级 seq + frame_ids。

    null_t（修改 03，2026-11-03）：T 输入置零（与标准路径同语义）。
    """
    chunks = []
    frame_ids = []
    ctx = None
    prev_start = None
    meta = None
    prev_pose = None  # G3 硬前缀：上一窗（拼接后）预测
    prev_v = None
    R_prev = None     # v4-a：当前窗锚 R_a 的链式携带（上一窗组合而来）
    for i in idxs:
        raw = dataset[i]
        batch = move_batch(collate_windows([raw]), device)
        start = int(raw["frame_start"])
        config_id = torch.full((1,), config_value, device=device, dtype=torch.long)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
        if null_t:
            # 修改 03（2026-11-03）：null-T 注入（T 内容全部置零）。
            t_raw = torch.zeros_like(t_raw)
            t_phys = torch.zeros_like(t_phys)
            t_s2m = torch.zeros_like(t_s2m)
        reset = (ctx is None or prev_start is None or start != prev_start + half)
        if reset:
            # ctx5：全姿态块 (1, nf, 148)；ctx3/4：根块 (1, nf, 10)；
            # v1/v2：单帧 (1, 10)。
            ctx = (torch.zeros(1, half, 148, device=device) if ctx5
                   else torch.zeros(1, half, 10, device=device) if ctx3
                   else torch.zeros(1, 10, device=device))
        out = model(v_feat, t_raw, t_phys, config_id, [session_id], T_s2m=t_s2m,
                    **v_kw, prev_ctx=ctx)
        anchor = batch["trans_anchor"][:, None, :]
        emit = 0 if reset else half
        if ctx4:
            # v4-a（2026-11-01）：模型输出 = 锚点系残差；组合 R_a 回世界系
            # （首窗/复位 R_a=I，模型在 flag=0 下输出绝对——组合为恒等）。
            # R_a 沿链携带：下一窗锚 = R_a ⊙ 残差根朝向 @ 帧 half。
            R_a = torch.eye(3, device=device) if reset else R_prev
            pose_w = rot_root6d(R_a[None], out["x0_hat"][0])   # (tw,144)
            v_w = torch.einsum("ij,tj->ti", R_a, out["v_hat"][0])
            trans_w = torch.cumsum(v_w, dim=0) / float(FPS) + anchor[0]
            pred_pose_w, pred_trans_w = pose_w, trans_w
            R_prev = torch.matmul(R_a, rot6d_to_rotmat(out["x0_hat"][0, half, :6]))
        elif hard_prefix and prev_pose is not None:
            # G3（续写 v2 实验矩阵）：窗 0:half 逐位覆盖为上一窗预测
            # half: ——接缝按构造消失（Step2Motion inpainting 的回归等价）；
            # trans 由拼接后的 v_hat 重算（模型内 trans_hat 基于旧 v_hat）。
            pose_w = out["x0_hat"][0].clone()
            v_w = out["v_hat"][0].clone()
            pose_w[:half] = prev_pose[half:]
            v_w[:half] = prev_v[half:]
            trans_w = torch.cumsum(v_w, dim=0) / float(FPS) + anchor[0]
            pred_pose_w, pred_trans_w = pose_w, trans_w
        else:
            pred_pose_w = out["x0_hat"][0]
            pred_trans_w = (out["trans_hat"] + anchor)[0]
            pose_w = pred_pose_w
            v_w = out["v_hat"][0]
        chunks.append({
            "pred_pose": pred_pose_w[emit:].detach().cpu(),
            "pred_trans": pred_trans_w[emit:].detach().cpu(),
            "gt_pose": batch["pose_gt"][0, emit:].cpu(),
            "gt_trans": (batch["trans_gt"] + anchor)[0, emit:].cpu(),
            "kp_gt": batch["kp_gt"][0, emit:].cpu(),
            "contact_gt": batch["contact_gt"][0, emit:].cpu(),
            "contact_pred": (torch.sigmoid(out["contact_logits"]) > 0.5)[0, emit:].detach().cpu()
                             if config_value != CONFIG_V else None,
            "pressure_gt": batch["T_raw"][0, emit:].cpu(),
        })
        frame_ids.extend(range(start + emit, start + batch["pose_gt"].shape[1]))
        if ctx3:
            # 硬填块 = 上一窗（拼接后）后 nf 帧逐帧携带；ctx5 携带全姿态
            # [144 维 6D + 根速度 3 + flag]，ctx3/4 携带根状态。
            if ctx5:
                block = torch.cat([pose_w[half:, :144], v_w[half:, :3],
                                   torch.ones(half, 1, device=device)], dim=-1)
            else:
                block = torch.cat([pose_w[half:, :6], v_w[half:, :3],
                                   torch.ones(half, 1, device=device)], dim=-1)
            ctx = block.unsqueeze(0)
        else:
            root6 = pose_w[half, :6].detach()
            vel3 = v_w[half, :3].detach()
            ctx = torch.cat([root6, vel3, torch.ones(1, device=device)], dim=0).unsqueeze(0)
        prev_start = start
        prev_pose = pose_w
        prev_v = v_w
        meta = (batch["floor_y"][0].cpu(), batch["offsets"][0].cpu(),
                batch["parents"][0].cpu(), batch["betas"][0].cpu())
    floor_y, offsets, parents, betas = meta
    seq = {
        "pred_pose": torch.cat([c["pred_pose"] for c in chunks], dim=0),
        "pred_trans": torch.cat([c["pred_trans"] for c in chunks], dim=0),
        "gt_pose": torch.cat([c["gt_pose"] for c in chunks], dim=0),
        "gt_trans": torch.cat([c["gt_trans"] for c in chunks], dim=0),
        "kp_gt": torch.cat([c["kp_gt"] for c in chunks], dim=0),
        "contact_gt": torch.cat([c["contact_gt"] for c in chunks], dim=0),
        "contact_pred": (torch.cat([c["contact_pred"] for c in chunks], dim=0)
                         if config_value != CONFIG_V else None),
        "pressure_gt": torch.cat([c["pressure_gt"] for c in chunks], dim=0),
        "pressure_pred": None,
        "pos_corr": None,
        "floor_y": floor_y, "offsets": offsets, "parents": parents, "betas": betas,
    }
    return seq, frame_ids


def _merge_sess_detail(seq, tw, forward_axis, config_value, accum, contact,
                       session_rows, window_rows, *, session_id, frame_ids,
                       window_starts):
    """dump_detail 逐会话记账（正常路径与续写路径共用）。"""
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
        "frame_ids": list(frame_ids),
        "session_id": str(session_id),
        "n_frames": int(seq["pred_pose"].shape[0]),
        "metrics": sess_metrics,
    })
    if window_starts is not None:
        w_rows = _window_detail_rows(seq, tw, [int(s) for s in window_starts])
        window_rows["session_ids"].extend([str(session_id)] * len(w_rows["frame_starts"]))
        window_rows["frame_starts"].extend(w_rows["frame_starts"])
        window_rows["n_frames"].extend(w_rows["n_frames"])
        for key, values in w_rows["metrics"].items():
            window_rows["metrics"].setdefault(key, []).extend(values)


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
    z_src_override: Optional[str] = None,
    null_t: bool = False,
    dataset_alt=None,
    overlap_assemble: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run the full F0a protocol and write metrics/<split>.json (detail) and,
    when brief_out is given, metrics/<split>_brief.json (whole-body brief).

    ``overlap_assemble``（修改 02，2026-11-03）：eval.py --overlap-stride 声明
    的重叠窗评估改走三角权重混合组装分支（不再跳过）；stride=tw 时新分支
    与现协议逐位一致（等价性硬门）。

    ``dump_detail`` additionally writes the per-window pose errors and
    per-session metric rows next to ``out_path``
    (window_detail_<split>.json / session_detail_<split>.json, snake_case keys);
    the pooled files are bit-identical either way.
    """
    # AR2 视觉校正增益下限（2026-11-02 诊断钩子）：eval 配置 yaml 单键，
    # 默认 0 = 关（不改既有 cell 行为）。
    ar2_gain_floor = float(config.get("ar2_vis_gain_floor", 0.0))
    ar2_init_height = float(config.get("ar2_init_height", 0.0))
    if (dataset.stride != dataset.window_length
            and not getattr(model, "prev_ctx", False)
            and not overlap_assemble):
        # 续写协议（2026-11-01）：V7_Cv5Ctx 的重叠窗评估走 _evaluate_config
        # 的链式分支，不属于 E6.4 扩散续写短路范围；修改 02 的声明口径
        # 重叠评估（--overlap-stride）同样不跳过。
        print("protocol skipped: dataset stride %d != window %d (E6.4 continuation eval)"
              % (dataset.stride, dataset.window_length))
        return {}
    forward_axis = _pick_forward_axis(dataset, device)
    metrics: Dict[str, Dict[str, float]] = {}
    window_conditions: Dict[str, dict] = {}
    session_conditions: Dict[str, dict] = {}
    anchor_counts: Dict[str, int] = {}
    # Deterministic protocol runs.  Sigma-seed for diffusion models comes from
    # three runs with --protocol-seed 0/1/2; regression models have no
    # sampling randomness (dropout is off in eval mode).
    prev_cpu = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        for config_value in config_values:
            # ctx3：T 行用重叠数据集（链式硬填），V/VT 行用非重叠数据集
            # （回归 Cv5 口径）；其他模型 dataset_alt=None → 恒用 dataset。
            ds_for = dataset_alt if (dataset_alt is not None
                                     and (getattr(model, "ctx3", False)
                                          or getattr(model, "ctx_vcond", False))
                                     and config_value != CONFIG_T) else dataset
            name = CONFIG_MODE_NAMES[config_value]
            metrics[name], window_rows, session_rows, anchor_counts[name] = \
                _evaluate_config(
                    ds_for, model, device, config_value, regress_mode, diffusion,
                    sample_steps, warm_start, tw, forward_axis,
                    v2t_out_dir=v2t_out_dir,
                    dump_detail=dump_detail,
                    z_src_override=z_src_override,
                    null_t=null_t,
                    checkpoint=checkpoint,
                    gain_floor=ar2_gain_floor,
                    overlap_assemble=overlap_assemble,
                    init_height=ar2_init_height,
                )
            if dump_detail:
                window_conditions[name] = window_rows
                session_conditions[name] = session_rows
            print("protocol %s: mpjpe=%.3fmm pa_mpjpe=%.3fmm w_mpjpe100=%.3fmm"
                  % (name, metrics[name]["mpjpe_mm"], metrics[name]["pa_mpjpe_mm"],
                     metrics[name].get("w_mpjpe100_mm", float("nan"))))
        if robustness and CONFIG_VT in config_values:
            ds_rob = dataset_alt if (dataset_alt is not None
                                     and getattr(model, "ctx3", False)) else dataset
            metrics["robust_vdrop"], _, _, anchor_counts["robust_vdrop"] = _evaluate_config(
                ds_rob, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="v", seed=seed,
                z_src_override=z_src_override,
                null_t=null_t,
                checkpoint=checkpoint,
                gain_floor=ar2_gain_floor,
                overlap_assemble=overlap_assemble,
                init_height=ar2_init_height,
            )
            metrics["robust_tdrop"], _, _, anchor_counts["robust_tdrop"] = _evaluate_config(
                ds_rob, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="t", seed=seed,
                z_src_override=z_src_override,
                null_t=null_t,
                checkpoint=checkpoint,
                gain_floor=ar2_gain_floor,
                overlap_assemble=overlap_assemble,
                init_height=ar2_init_height,
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
    from anysole.utils.evaluation_contract import evaluation_contract
    contract = evaluation_contract(dataset, checkpoint.get("config", {}))
    payload = {
        "evaluation_contract": contract,
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
        # 修改 02（2026-11-03）：预测侧 GT 平移锚点消费数（每条件）。标准
        # 路径 = 覆盖 tile 数（重叠 stride 下锚点网格保持非重叠网格 → 与
        # 现协议每窗一锚逐位相同）；AR2 自由积分 = 0；ctx/帧级 AR = 窗数
        # （每窗 left−1 锚，仅登记信息）。
        "n_gt_anchors_used": anchor_counts,
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
            "dump_version": 2,
            "evaluation_contract": contract,
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
