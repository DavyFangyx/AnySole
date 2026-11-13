"""V8_Cv7 (no gate) + S2M-style IMU (fusion 1, 2026-11-12).

V8_Cv7 structure (no T reliability gate) with ``tactile_input="raw108_imu"``:
FootConv raw108 pressure + IMU MLP concat before the shared motion decoder.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_a import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw108_imu",
             "v7_root_sd": True, "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        contact_root=bool(config.get("v7_contact_root", False)),
        reliability_route=False,
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=STRUCTURE["tactile_input"],
    )
