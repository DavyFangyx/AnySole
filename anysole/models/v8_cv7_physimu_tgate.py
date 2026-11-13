"""V8_Cv7_PhysIMU with the existing V gate frozen and an independent T gate.

The V gate is the validated four-group gate from V8_Cv7_PhysIMU_Gate.  The
new T gate is trained separately: gT = sigmoid(a_window + delta_frame), with
a per-group window-level base and a zero-initialized per-frame correction.
"""
from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_a import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_external_gate": True,
             "v7_t_gate": True,
             "v7_t_gate_g0": 0.9,
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
        external_gate=True,
        gate_g0=config.get("v7_gate_g0"),
        gate_gv_min=float(config.get("v7_gate_gv_min", 0.2)),
        gate_smooth=bool(config.get("v7_gate_smooth", False)),
        root_sd=bool(config.get("v7_root_sd", True)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", True)),
        tactile_input=str(config.get("tactile_input", "raw96_phys_imu")),
        t_gate=True,
        t_gate_g0=float(config.get("v7_t_gate_g0", 0.9)))
