"""V7_Cv5T（2026-10-30，V7_C门控修改方案 §4）: T 时间门——逐帧双独立 T 门.

Cv5 的根/姿态双门取消窗口均值池化，复用同两个 ``2d→d→1`` MLP 逐帧计算：
``x_t = cat(v_tok_t, t_tok_t)`` → ``g_vel_t`` / ``g_pose_t`` 各 (B, tw, 1)；
根速度 query 用 g_vel，九姿态 query 用 g_pose。门参数量与 Cv5 相同、六层
共享；首轮无平滑损失（若逐帧门导致预测抖动上升，再按 §4 加 5 帧段门对照）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_route_temporal": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        route_temporal=bool(config.get("v7_route_temporal", False)))
