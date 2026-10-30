"""V7_Cv5LR（2026-10-30，V7_C门控修改方案 §5）: LR 左右门——窗口四组门.

Cv5 双门扩展为四组：g_vel（根速度 query）/ g_other（姿态 root、torso、
headneck、l_arm、r_arm）/ g_left（l_leg、l_foot）/ g_right（r_leg、r_foot）。
窗口均值 ``mean_t(x_t)`` 后四个门在窗内固定；左右门 MLP 结构同原姿态门，
初始化 = 原姿态门深拷贝再独立学习（共享模块随机初始化不受新增参数影响，
LR 的相同门初始化复现 B）。分组严格按 PART_NAMES 映射（model_v7 模块级
常量），展开为 (B, tw, 10, 1) 沿用现有乘法位置，不改变注意力 memory。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_route_parts": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        part_routes=bool(config.get("v7_route_parts", False)))
