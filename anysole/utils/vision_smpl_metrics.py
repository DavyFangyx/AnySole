"""Assemble existing AnySole formulas for externally predicted SMPL motion.

Inputs are native SMPL-24 joints, posed/zero-pose surfaces and SMPL poses in one world
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


def evaluate_motion(predicted: dict, target: dict, times: np.ndarray) -> dict:
    """Body21 rotation, pelvis-relative temporal, all paired frames, joint feet.

    Surfaces and zero-pose surfaces are generated with each prediction's own
    beta, while the reference uses GT beta. PA fits the raw world joints.
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
        "mpjae_deg": float(canonical.rotation_error_degrees(
            predicted["poses"][:, 3:66].reshape(-1, 21, 3),
            target["poses"][:, 3:66].reshape(-1, 21, 3)).mean()),
        "pve_mm": float(canonical.mean_point_error(pv, gv, scale=1000).mean()),
        "pve_t_mm": float(canonical.pve_t_error(
            predicted["tpose_joints"], predicted["tpose_vertices"],
            target["tpose_joints"], target["tpose_vertices"]).mean()),
        **canonical.temporal_metrics(pj, gj, times),
        "w_mpjpe100_mm": w_error,
        "root_ate_mm": canonical.root_trajectory_metrics(
            predicted["joints"][:, 0], target["joints"][:, 0])["root_ate_mm"],
        "foot_sliding_mm": slide,
        "foot_contact_transitions": contacts,
    }
