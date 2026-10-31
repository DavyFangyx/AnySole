"""V7_Cv5Q（2026-10-31，V7_C门控修改方案 §6）: Q 条件输入——门读第一层解码上下文.

TLR 的四组逐帧门、MLP 尺寸与六层共享方式保持不变，只替换门的输入：
第一层解码器完成 query 自注意力与 V 交叉注意力后得 h，计算尚未乘门的
T 交叉注意力输出 a_T；按四组在同一帧内分别平均 query，
``u_t,k = cat(mean_query(h_t,k), mean_query(a_T_t,k))`` → ``g_t,k =
sigmoid(MLP_k(u_t,k))``。第一层据此计算的门用于本层，并传后续五层复用；
不增加逐层独立门。梯度正常经 h 与 a_T 回传；缺失模态仍由存在掩码硬关闭。
无新增参数（state dict 与 TLR 逐键一致）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2tlr import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_route_conditional": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        route_temporal=bool(config.get("v7_route_temporal", False)),
        part_routes=bool(config.get("v7_route_parts", False)),
        route_conditional=bool(config.get("v7_route_conditional", False)))
