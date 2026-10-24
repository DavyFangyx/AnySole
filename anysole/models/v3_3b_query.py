"""V3_Bv2/V3_Bv3 entry: V3_Bv1 + query pressure readout (2026-10-24).

V3_Bv2 = Bv1 + query pressure head (Q-DIRECT direct / Q-NN nonnegative with
         train cell-mean prior init) replacing the linear aux t_rec.
V3_Bv3 = Bv2 + Q-FACT factored readout (total×allocation, zero-init start
         == training prior).

One model structure — ``pressure_head`` is a single config parameter that
selects the output representation (Q-DIRECT / Q-NN / Q-FACT); the per-model
yaml freezes it per registered version.  Independent models (2026-09-24
restructure): everything trains from random init.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.common.pressure_readout import (
    QUERY_KINDS,
    build_query_pressure_head,
)
from anysole.models.v3_3b import STRUCTURE as V3_3B_STRUCTURE

STRUCTURE = {**V3_3B_STRUCTURE}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V3_Bv2/V3_Bv3 network and attach the query pressure head."""
    head = str(config.get("pressure_head", "query_direct"))
    if head not in QUERY_KINDS:
        raise ValueError("unsupported pressure_head %r; expected one of %s"
                         % (head, QUERY_KINDS))
    if head != "query_direct":
        cell_mean = config.get("pressure_cell_mean")
        if (not isinstance(cell_mean, (list, tuple)) or len(cell_mean) != 96
                or any(not isinstance(x, (int, float)) or x <= 0 for x in cell_mean)):
            raise ValueError("Q-NN/Q-FACT require a frozen 96-cell positive train mean prior")
    model = AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
    model.pressure_readout = build_query_pressure_head(config)
    model.pressure_head = head
    return model
