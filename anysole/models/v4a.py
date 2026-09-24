"""V4A entry: foot_conv + 24-slot soft assignment (gradient clustering).

The grouping is learned from a near-uniform assignment — ``assign_cluster``
is part of this model's identity (recorded into the ckpt config); the
annealing / loss hyperparameters (--lambda-assign-*, --assign-*) remain
runtime train choices.

Independent model (2026-09-24 restructure): every training run of V4A starts
from random init.  No warm-start lineage.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2

STRUCTURE = {
    "t_encoder": "foot_conv",
    "f2_repr": False,
    "pose_parts": 24,
    "soft_parts": True,
    "gate": "none",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "tactile_direct": False,
    "no_imu": False,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {"assign_cluster": True}


def build(config: dict, part_joints=None):
    """Construct the V4A network from the effective train config."""
    return AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
