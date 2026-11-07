"""V8_Cv5VUG（2026-11-07，修改 15）：Cv5 + 不确定性专家驱动的 V 门。

结构 = Cv5 冻结主干 + ``unc_expert``（MLP：窗均值 v_tok → û，监督量 =
IG 接触可预测性，训练期由冻结接触头 + 接触 GT 构造，运行时零 GT）+
V 门 ``g_v = σ(unc_a·û + unc_b)``（乘 V 交叉注意力输出，值域 (0,1)；
V-only 行由 ``t_present`` 钉回 1.0，U@V 行逐位不变）。T 门保持 Cv5
原样。unc_expert 权重与 (unc_a, unc_b) 逐折拟合后随 ckpt 落盘
（buffer），评估口径零改动。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_v_gate": True, "v7_unc_gate": True,
             "v7_gate_2sigma": False,
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
        unc_gate=bool(config.get("v7_unc_gate", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
