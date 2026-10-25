"""V7_Cv1 entry（2026-10-25）：候选 C——A + 窗口级可靠性路由。

结构同 V7_Av1；新增 = 窗口级 T 信任门 g∈[0,1]（V/T token 窗口均值 → MLP
→ sigmoid，bias 0 初始化 = 中性），逐窗调制 T 交叉注意力输出。v1 端到端
由主运动损失学习、无显式质量标签（设计思路 §3 候选 C 第一版：质量估计
的显式监督留待 v2，可用预测误差/退化标签接入）。
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v7": True,
    "v7_contact_root": False,
    "v7_reliability_route": True,
}

CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        contact_root=bool(config.get("v7_contact_root", False)),
        reliability_route=bool(config.get("v7_reliability_route", False)),
    )
