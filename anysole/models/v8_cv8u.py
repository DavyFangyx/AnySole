"""V8_Cv8U (2026-11-12): NLL-calibrated per-slot uncertainty router.

Structure = V8_Cv8's V/T router replaced by a precision router:

    q0 -> cross-attn(M_V) -> A_V -> s_V = MLP_V(A_V)
    q0 -> cross-attn(M_T) -> A_T -> s_T = MLP_T([A_T, log‖A_T‖])
    K_p = sigmoid(s_V,p - s_T,p)
    V branch *= (1-K_p), T branch *= K_p   (V-only K=0, T-only K=1)

s_V is trained only by a heteroscedastic NLL on V-only rows, s_T only on
T-only rows; the VT task loss does not train K (``unc_stopgrad=True``).
This is the calibrated version of V8_Cv8's ``k = σ(s_v − s_t)``.
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
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
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
