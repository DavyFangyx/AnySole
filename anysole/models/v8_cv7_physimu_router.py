"""V8_Cv7_PhysIMU + competitive V/T router gate (fine-tuned, 2026-11-14).

Structure: PhysIMU V8 base (raw96 pressure -> FootConv, separate T_phys MLP,
tactile-side IMU channels) with the four-group V/T gates replaced by the
competitive router k = sigmoid(s_v - s_t): decoder V branch x (1-k), T branch
x k; V-only rows k=0, T-only k=1.  The router, decoder and T encoder were
trained jointly from the frozen base with the enum V/T/VT motion loss.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_a import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "tactile_input": "raw96_phys_imu",
             "v7_reliability_route": True,
             "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_route_router": True,
             "v7_task_routes": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 0.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=True,
        route_router=True,
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        tactile_input=str(config.get("tactile_input", "raw96_phys_imu")))
