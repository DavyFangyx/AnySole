"""V8_Cv5 + S2M-style IMU (fusion 1, 2026-11-12).

Same V8_Cv5 structure (dual T gates + SD2 + anchor head); the only change is
``tactile_input="raw108_imu"``: FootConv keeps the raw108 pressure path and
the 12 IMU columns of T_s2m go through their own MLP, concatenated with the
tactile token before the (unchanged) T gates and motion decoder.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2sd2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw108_imu"}
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
