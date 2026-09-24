"""V3_2 entry: foot_conv + 9-part pose head (hard PART_JOINTS partition).

Independent model (2026-09-24 restructure): the structure is fixed here;
every training run of V3_2 — at any --tw/--stride/… — starts from random
init.  No warm-start lineage.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "foot_conv",
    "f2_repr": False,
    "pose_parts": 9,
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
    """Construct the V3_2 network from the effective train config."""
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
