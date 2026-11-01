"""V7_Cv5Ctx4B（2026-11-01）: Cv5 + 续写 v4-b（统一残差空间 + V/VT 锚估计头）.

Cv5 双门结构不变。相对 ctx4（v4-a 混合空间版）的核心改动
（V7_续写v4残差分解方案 §10 v4-b）：
  全条件（V/VT/T）目标 = 锚点系残差（锚 = GT 窗首帧根朝向，首窗/复位
  flag=0 → R_a=I 绝对模式）——生成器目标空间唯一（残差），公共分支无
  双空间混淆；pose 统计回到单一残差分布（+ 首窗绝对小分量）。
  V/VT 行锚来源 = 锚估计头（m_v 窗口均值 → 6D 偏移，恒等 6D 起跳，
  L_anchor 测地角监督，λ_anchor=1.0）；T 行锚来源 = ctx 链（同 v4-a）。
  零初始化 → init 锚 = 恒等、逐位 Cv5 契约；头结构不动。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx3": True, "v7_ctx4b": True}
CONFIG_EXTRA: dict = {"lambda_anchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx3=bool(config.get("v7_ctx3", False)),
        ctx4b=bool(config.get("v7_ctx4b", False)))
