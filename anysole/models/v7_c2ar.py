"""V7_Cv5AR（2026-11-01）: Cv5 + 帧级双模式 AR（V/VT 回归 + T 自回归）.

Cv5 双门结构不变。生成风格按条件分支（V7 方案 §11）：
  全条件（V/VT/T）因果掩码 + 反馈支路（上一帧完整输出 [姿态 144 +
  根速度 3] 经共享投影加性注入 query）——逐帧接续，历史全量注入，ctx
  机制与模式开关全部消失（共享解码器语义唯一，2026-11-01 用户裁定）。
训练全并行（反馈 = GT 上一帧，teacher forcing）；推理全条件逐帧串行。
零初始化保留 init 逐位 = Cv5 契约。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar=bool(config.get("v7_ar", False)))
