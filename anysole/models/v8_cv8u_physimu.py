"""V8_Cv8U + PhysIMU (stop-grad on, 2026-11-12).

Same PhysIMU tactile encoding as V8_Cv8U_PhysIMU_SG, but ``unc_stopgrad=True``:
s_V/s_T and the probe are trained only by the V-only/T-only heteroscedastic
NLL; the VT task loss does not train K.  This isolates the SG (task-closes-K)
effect on the PhysIMU base.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_unc_router": True,
             "v7_unc_stopgrad": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        unc_router=bool(config.get("v7_unc_router", False)),
        unc_stopgrad=bool(config.get("v7_unc_stopgrad", True)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=str(config.get("tactile_input", "raw96_phys_imu")))
