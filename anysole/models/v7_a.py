"""V7_Av1 entry（2026-10-25）：候选 A——独立证据 + 共享生成。

z_note/multimodal_fusion/模型设计思路.md §3 候选 A 的最小实现：V/T 各自
时序建模为 M_V/M_T，共享运动 query（根 + 9 部位）分别交叉注意力读取、
缺失模态硬关闭；单一解码器三条件共用；旋转 + 根速度 → FK 统一出位置。
从零训练，无载体、无修正头、无推理侧约束。
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v7": True,
    "v7_contact_root": False,
    "v7_reliability_route": False,
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
