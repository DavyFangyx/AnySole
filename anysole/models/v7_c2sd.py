"""V7_Cv5SD（2026-11-04，修改 07）：Cv5 + 根头速率/方向解耦 + T 行方向掩码。

结构 = V7_Cv5（v7_c2 双独立 T 门，枚举三条件、双向无因果、非 AR2）+
``v7_root_sd``：根头输出从世界系 3 维速度改为 [水平速率, 水平单位方向
(x,z), 垂直分量]，损失按通道监督，T 行屏蔽水平方向项（方向不可观测）。
``v_hat`` 的形状与语义不变（仍组装成 (B, tw, 3) 世界系 m/s），因此
trans_hat / 锚消费 / FK / 评估协议均不受影响。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        root_sd=bool(config.get("v7_root_sd", False)))
