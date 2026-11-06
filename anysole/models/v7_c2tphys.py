"""V7_Cv5 + T_phys v2（修改 11 A3，2026-11-05）。

结构逐参数 = V7_Cv5（v7_c2：双独立 T 门 + 枚举三条件）；唯一差异 =
T_phys 12→20（``t_phys_v2=True``：每足 hull 着地几何 3 维 + 双足关系 2 维）。
T_raw 的 96 维口径不动，Cv5 既有 12 维 base 一行不改（R3 零税契约）。

用途：4 折 LSO 测 ``t_phys_v2`` 在 Cv5 + 新骨架上的净兑现量。V5_Av1 老基座 +
默认 split 已测 T2M PA −17.9mm（−15.8%），V2M/VT2M 零税；本变体是去重测
Cv5 + LSO 口径。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "t_phys_v2": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        t_phys_v2=bool(config.get("t_phys_v2", False)))
