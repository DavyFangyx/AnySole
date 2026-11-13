"""V3_Bv1 + PhysIMU fusion, method 2.

Three separately encoded streams:

    T_raw (96 grid cells)  -> FootConv (raw-only)                 -> t_raw
    T_phys (12)            -> 3-layer MLP + LayerNorm             -> t_phys
    IMU 12 (T_s2m 16:22 + 41:47) -> 3-layer MLP + LayerNorm       -> imu

then a two-stage fusion:

    physical = Linear(2d -> d)([t_phys, imu])
    t_tok    = Linear(2d -> d)([t_raw, physical])

Both fusion projections are followed by LayerNorm so the fused tactile token
keeps the same scale as the other encoder tokens.  The pressure encoder is the
raw-only FootConv variant: T_phys is *not* consumed inside FootConv here, it is
the separate MLP branch above.
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
    "tactile_input": "raw96_phys_imu",
    "tactile_direct": False,
    "no_imu": False,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V3_Bv1_PhysIMU network from the effective train config."""
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
