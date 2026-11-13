"""V3_Bv1_IMU without the pose-head sigma gate (no-gate control).

Identical to ``v3_3b_imu`` except ``gate="none"``: the pose decoder uses the
standard cross-attention path instead of the per-part sigma routing.  This is
the no-gate arm for the IMU fusion comparison.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "foot_conv",
    "f2_repr": False,
    "pose_parts": 9,
    "soft_parts": True,
    "gate": "none",
    "v_input": "hrnet",
    "tactile_input": "raw108_imu",
    "tactile_direct": False,
    "no_imu": False,
}

CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
