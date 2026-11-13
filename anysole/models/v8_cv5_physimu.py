"""V8_Cv5 + PhysIMU (fusion 2, 2026-11-12).

Same V8_Cv5 structure; ``tactile_input="raw96_phys_imu"`` routes T_raw 96
through a raw-only FootConv, T_phys 12 through its own MLP, and IMU 12 through
its own MLP, then two-stage [T_phys, IMU] -> [T_raw, physical] fusion.  The
fused t_tok keeps the same shape, so the dual T gates and decoder are unchanged.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2sd2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu"}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=STRUCTURE["tactile_input"],
    )
