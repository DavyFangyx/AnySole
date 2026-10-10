"""Assemble existing AnySole formulas for externally predicted SMPL motion.

Inputs are native SMPL-24 joints, surfaces and local rotations in one world
frame, metres, and the real matched timestamps. No formula is defined here.
Unlike the training evaluator's fixed 40 Hz windows, RGB baseline sampling
is irregular; passing its actual times preserves the canonical solver's
duplicate-time and gap handling. All frames are evaluated.
"""
from __future__ import annotations

import numpy as np

from anysole.types import ANKLE_FOOT_JOINTS
from anysole.utils import metrics as canonical


REQUESTED_METRICS = (
    "mpjpe_mm", "pa_mpjpe_mm", "mpjae_deg", "pve_mm", "pve_t_mm",
    "accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3",
    "w_mpjpe100_mm", "root_ate_mm", "foot_sliding_mm",
)
PVE_T_REASON = "Removed by AnySole e9accee0; no canonical PVE-T definition."


def evaluate_motion(predicted: dict, target: dict, times: np.ndarray) -> dict:
    """Apply the native protocol: 24 rotations, world temporal, joint feet.

    All three baselines predict beta/surface; PVE therefore uses their own
    predicted surfaces, never a surface fabricated with GT beta.
    """
    pj, pv = canonical.pelvis_align(predicted["joints"], predicted["vertices"])
    gj, gv = canonical.pelvis_align(target["joints"], target["vertices"])
    w_error, _ = canonical.windowed_world_mpjpe(predicted["joints"], target["joints"])
    slide, contacts = canonical.foot_sliding_joints(
        predicted["joints"][:, ANKLE_FOOT_JOINTS],
        target["joints"][:, ANKLE_FOOT_JOINTS], times,
    )
    return {
        "mpjpe_mm": float(canonical.mean_point_error(pj, gj, scale=1000).mean()),
        "pa_mpjpe_mm": float(canonical.pa_mpjpe(predicted["joints"], target["joints"]).mean()),
        "mpjae_deg": float(canonical.matrix_rotation_error_degrees(
            predicted["rotations"], target["rotations"]).mean()),
        "pve_mm": float(canonical.mean_point_error(pv, gv, scale=1000).mean()),
        "pve_t_mm": None,
        **canonical.temporal_metrics(predicted["joints"], target["joints"], times),
        "w_mpjpe100_mm": w_error,
        "root_ate_mm": canonical.root_trajectory_metrics(
            predicted["joints"][:, 0], target["joints"][:, 0])["root_ate_mm"],
        "foot_sliding_mm": slide,
        "foot_contact_transitions": contacts,
    }


def evaluate_formula_control(predicted: dict, target: dict, times: np.ndarray,
                             safe_mask: np.ndarray) -> dict:
    """Same historical inputs, evaluated exclusively with AnySole formulas.

    Body-only rotations, pelvis-local unrepaired temporal trajectories and
    vertex feet are *diagnostics*, never the native public protocol above.
    This group isolates formula implementation from protocol/input changes.
    """
    pj, pv = canonical.pelvis_align(predicted["joints"], predicted["vertices"])
    gj, gv = canonical.pelvis_align(target["joints"], target["vertices"])
    w_error, _ = canonical.windowed_world_mpjpe(predicted["joints"], target["joints"])
    slide, contacts = canonical.foot_sliding_vertices(
        predicted["vertices"], target["vertices"], times)
    return {
        "mpjpe_mm": float(canonical.mean_point_error(pj, gj, scale=1000).mean()),
        "pa_mpjpe_mm": float(canonical.pa_mpjpe(pj, gj).mean()),
        "mpjae_deg": float(canonical.rotation_error_degrees(
            predicted["poses"][:, 3:66].reshape(-1, 21, 3),
            target["poses"][:, 3:66].reshape(-1, 21, 3)).mean()),
        "pve_mm": float(canonical.mean_point_error(pv, gv, scale=1000).mean()),
        "pve_t_mm": None,
        **canonical.temporal_metrics(pj[safe_mask], gj[safe_mask], times[safe_mask]),
        "w_mpjpe100_mm": w_error,
        "root_ate_mm": canonical.root_trajectory_metrics(
            predicted["joints"][:, 0], target["joints"][:, 0])["root_ate_mm"],
        "foot_sliding_mm": slide,
        "foot_contact_transitions": contacts,
    }
