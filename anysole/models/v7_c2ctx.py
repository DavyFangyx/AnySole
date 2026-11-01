"""V7_Cv5Ctx（2026-11-01）: Cv5 + 上一窗姿态上下文（续写，修复 2 回归版）.

Cv5 双门结构不变；每个训练/推理窗的输入多带一路窗首帧根状态上下文：
[根朝向 6D + 根速度 3 + 有效标志 1]（10 维）→ 经 ctx_emb 加性注入根
速度 query（根朝向经解码器自注意力传递）。训练用 GT 窗首帧根状态
（teacher forcing），推理用模型自己上一窗的预测（评估逐窗串行）；首窗
上下文为零向量 + flag 0。窗口网格、输出头、门控、评估指标全部不变。
零初始化 → init 时上下文贡献为零、逐位复现 Cv5。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_prev_ctx": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx_vcond=bool(config.get("v7_ctx_vcond", False)))
