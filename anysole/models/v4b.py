"""V4B entry: foot_conv + hard learned partition (--part-json).

The partition (which joints form each part, any K) is structural data —
it is the only per-run input to this entry, passed as ``part_joints`` at
build time from --part-json.  ``pose_parts`` is derived from the partition.

Independent model (2026-09-24 restructure): every training run of V4B starts
from random init.  No warm-start lineage.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "foot_conv",
    "f2_repr": False,
    # pose_parts: derived from the --part-json partition at build time.
    "soft_parts": False,
    "gate": "none",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "tactile_direct": False,
    "no_imu": False,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V4B network from the effective train config."""
    if part_joints is None:
        raise ValueError("V4B requires --part-json (learned hard partition)")
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        pose_parts=len(part_joints),
        part_joints=part_joints,
        **STRUCTURE,
    )
