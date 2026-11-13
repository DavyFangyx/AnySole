"""V8_Cv8U_SG: Cv8U with the VT task loss allowed to train K (stop-grad off).

First pilot (V8_Cv8U) calibrated K only on V-only/T-only NLLs and kept
``unc_stopgrad=True``; K settled around 0.45-0.53 and clean VT PA regressed.
This arm lets the VT task loss lower K when the T stream hurts, while the
V-only/T-only NLL still anchors the absolute precision scale.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_unc_router": True,
             "v7_unc_stopgrad": False}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        unc_router=bool(config.get("v7_unc_router", False)),
        unc_stopgrad=bool(config.get("v7_unc_stopgrad", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
