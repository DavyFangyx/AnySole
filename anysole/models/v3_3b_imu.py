"""V3_Bv1 + IMU: S2M-style IMU stream concatenated with the tactile encoding.

The pressure path is the unchanged V3_Bv1 raw108 FootConv encoder.  The 12
IMU channels (T_s2m left acc3/gyro3 + right acc3/gyro3) go through a
per-frame MLP, and the resulting embedding is concatenated with the tactile
token on the feature axis, then projected back to ``d`` by
``IMUConcatFusion`` (initialized ``[I | 0]``: zero tax at step 0).

This mirrors Step2Motion's condition embedding order — IMU is its own MLP
group, not mixed into the pressure channel — while keeping the V3_Bv1
tactile representation and the existing ``2*tw`` fusion contract intact.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "foot_conv",
    "f2_repr": False,
    "pose_parts": 9,
    "soft_parts": True,
    "gate": "sigma",
    "v_input": "hrnet",
    "tactile_input": "raw108_imu",
    "tactile_direct": False,
    "no_imu": False,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V3_Bv1_IMU network from the effective train config."""
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
