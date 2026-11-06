"""V7_Cv5 + T_phys v2（hull_mcol 逐会话重定心）（修改 11 §9.3，2026-11-05）。

结构逐参数 = V7_Cv5_tphys：T_phys 12→20，模型/编码器形状完全一致。
唯一差异在数据侧：``hull_mcol`` 从鞋垫绝对坐标改成相对该会话接触跨度
（活跃帧 p5–p95 线性映射到 [0,1]），只由压力输入逐会话逐足计算，
不使用 GT 标签。其余 6 个新增维（hull_mrow / hull_std / 力比 / CoP 距离）
原样。

用途：4 折 LSO 配对
  * Cv5                 = 无新增维基线；
  * V7_Cv5_tphys        = 8 维原样（raw hull_mcol）；
  * 本变体              = 6 维直迁 + 2 维重定心。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "t_phys_v2": True,
             "t_phys_hull_recenter": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        t_phys_v2=bool(config.get("t_phys_v2", False)))
