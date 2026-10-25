"""V7_Bv1 entry（2026-10-25）：候选 B——A + 接触参与根运动。

结构同 V7_Av1；新增 = 损失侧足-地一致性：可信支撑窗口内足底世界速度
→ 0，软权重 = 接触概率，经 FK 回传到根运动与下肢生成（compute_losses_v7
的 L_foot 项；不训接触头之外的独立模块，推理不消费 GT 地面）。
"""

from __future__ import annotations

from anysole.models.common.model_v7 import AnySoleModelV7

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v7": True,
    "v7_contact_root": True,
    "v7_reliability_route": False,
}

# train.py injects this into the effective config and every checkpoint.
CONFIG_EXTRA: dict = {"lambda_foot": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        contact_root=bool(config.get("v7_contact_root", False)),
        reliability_route=bool(config.get("v7_reliability_route", False)),
    )
