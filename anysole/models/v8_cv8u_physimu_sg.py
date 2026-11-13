"""V8_Cv8U + PhysIMU + SG (2026-11-12).

The gate pilots (V8_Cv8U / V8_Cv8U_SG) use the V8_Cv8 raw108 pressure-only
backbone.  This arm puts the same NLL-calibrated per-slot router on top of the
PhysIMU tactile encoding (``raw96_phys_imu``: T_raw 96 FootConv + T_phys MLP +
IMU MLP, two-stage fusion) and keeps ``unc_stopgrad=False`` so the VT task loss
can close K when T pollutes clean VT.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_root_sd": True,
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
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=str(config.get("tactile_input", "raw96_phys_imu")))
