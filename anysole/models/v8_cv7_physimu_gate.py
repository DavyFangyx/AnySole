"""V8_Cv7_PhysIMU + frozen-base four-group independent V/T gate.

Base = V8_Cv7_PhysIMU (no learned gate; PhysIMU tactile; V/T/VT enum pretrain).
A small GateController is attached and trained with the VT motion loss while
the base is frozen.  Gate groups: root / upper / left / right; each has
independent g_V, g_T.  Initial g0 comes from the validation fixed-gate scan
(2026-11-12): root (0.75, 0.0), upper (1.0, 0.5), left (1.0, 1.0),
right (1.0, 1.0).  Boundaries 0/1 are represented by interior values and the
actual deviation is recorded in GateController buffers.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_a import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_external_gate": True,
             "v7_gate_g0": [[1.0, 1.0], [1.0, 1.0],
                            [1.0, 1.0], [1.0, 1.0]]}
CONFIG_EXTRA: dict = {"lambda_tanchor": 0.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=False,
        task_routes=False,
        external_gate=bool(config.get("v7_external_gate", False)),
        gate_g0=config.get("v7_gate_g0"),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=str(config.get("tactile_input", "raw96_phys_imu")),
        gate_gv_min=float(config.get("v7_gate_gv_min", 0.2)),
        gate_smooth=bool(config.get("v7_gate_smooth", False)))
