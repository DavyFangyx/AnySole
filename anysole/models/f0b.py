"""F0b entry: V2 regression base — linear tactile encoder, no f2, 3-part head.

Independent model (2026-09-24 restructure): the structure is fixed here;
every training run of F0b — at any --tw/--stride/… — starts from random
init.  No warm-start lineage.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "linear",
    "f2_repr": False,
    "pose_parts": 3,
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
    """Construct the F0b network from the effective train config."""
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
