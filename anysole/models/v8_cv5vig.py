"""V8_Cv5VIG（2026-11-07，修改 15）：Cv5 + V 信任门（T 门保持 Cv5）。

与 V8_Cv5VG 的区别：**T 门不做 2σ**（保持 Cv5 的 sigmoid/clean 抑制），
只加 V 侧镜像门（`g_v = 2σ(route_v_mlp(cat(v_tok,t_tok)))`，init=1，
V-only 行用 `t_present` 把门钉回 1.0）。V 门在阶段 2 训练（冻结主干，
目标 = val 的逐窗 b*），T 门全程冻结。

结构其余逐项 = V8_Cv5（SD2 + 双门 + 枚举 + 锚头）；V8_Cv5 的 T 门权重
原样载入。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_v_gate": True, "v7_gate_2sigma": False,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        v_gate=bool(config.get("v7_v_gate", False)),
        gate_2sigma=bool(config.get("v7_gate_2sigma", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
