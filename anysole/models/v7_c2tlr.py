"""V7_Cv5TLR（2026-10-30，V7_C门控修改方案 §5）: TLR 组合门——逐帧四组门.

LR 的四组门（vel/other/left/right）+ T 的逐帧输入：``x_t`` 不池化，四个
门逐帧输出，展开为 (B, tw, 10, 1) 沿用现有乘法位置；不改变注意力 memory。
左右门初始化 = 原姿态门深拷贝（TLR 的相同门初始化复现 T）。分组严格按
PART_NAMES 映射，不另写关节序号体系。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_route_temporal": True, "v7_route_parts": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        route_temporal=bool(config.get("v7_route_temporal", False)),
        part_routes=bool(config.get("v7_route_parts", False)))
