"""V8_Cv5T1（2026-11-06）：V8 基座 + T 运动轴 T1。

方向因子换载体：不再让 T 头预测世界系 (x,z) 方向（Δyaw 相关 0.01–0.06、
Drift 45.44° = 机会水平），而是

    d_hat = R(φ) · fwd(pose_t)

其中 ``fwd(pose_t)`` 是同帧根 6D（POSE_DIM 前 6 位，关节 0 = root）经
``rot6d_to_rotmat`` 的第三列（+Z 前向轴）投影到世界 (x,z)；φ 是标量偏角。
根头由 4 通道 [s, dir_x, dir_z, vy] 改为 3 通道 [s, φ, vy]，参数量不增；
φ 末层零初始化 ⇒ init 方向 = 姿态前向轴。

其余逐项 = V8_Cv5（双门 + 枚举 + SD2 + 锚头）；T 行方向掩码保持
（φ 在 T 行不监督，方向由姿态前向轴给出）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True, "v7_root_sd_t1": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        root_sd_t1=bool(config.get("v7_root_sd_t1", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
