"""V7_Cv5Ctx4R（2026-11-01）: Cv5 + 续写 v4-a + 接触根运动（B 候选）.

Cv5 双门结构不变；ctx4（v4-a 混合空间版）之上开 v7_contact_root：
L_foot（支撑帧足底世界速度→0，软权重 = 接触概率，经 FK 回传姿态与根）
λ_foot=1.0 预注册（compute_losses_v7 的 B 项，v7_b.py 同款 CONFIG_EXTRA）。
设计主张：滚转的 T 侧唯一物理证据 = 地面接触弱约束——续写链上挂 B
把「脚不滑动」变成残差空间里 roll 增量的物理拉回（2026-11-01 裁定）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx3": True, "v7_ctx4": True,
             "v7_contact_root": True}
CONFIG_EXTRA: dict = {"lambda_foot": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        contact_root=bool(config.get("v7_contact_root", False)),
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx3=bool(config.get("v7_ctx3", False)),
        ctx4=bool(config.get("v7_ctx4", False)))
