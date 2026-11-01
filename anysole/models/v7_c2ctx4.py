"""V7_Cv5Ctx4（2026-11-01）: Cv5 + 续写 v4（锚点系残差分解，v4-a 混合空间版）.

Cv5 双门结构不变。相对 ctx3 的一条核心改动（V7_续写v4残差分解方案 §2）：
  T-only 行的训练目标 = 锚点系残差（R_a = GT 窗首帧根朝向；R_res = R_aᵀ⊙GT，
  训练循环侧变换），注入块同样转残差（帧 0 根残差 = 恒等）——生成器目标
  空间里没有绝对航向，复制锚点的捷径在表示层消失；出口组合 R_a ⊙ R_res
  恢复世界系（评估链携带 R_a）。
  V/VT 行：R_a = I（无变换）+ 门控不注入 → 与 Cv5 完全一致。
首窗/复位：flag=0 → R_a = I，绝对模式先验起跳（offset 由此产生）。
零新增参数（几何变换为确定性算子）；零初始化保留 init 逐位 = Cv5 契约。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx4": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx3=bool(config.get("v7_ctx3", False) or config.get("v7_ctx4", False)),
        ctx4=bool(config.get("v7_ctx4", False)))
