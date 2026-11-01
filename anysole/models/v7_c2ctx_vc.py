"""V7_Cv5CtxVC（G2）: V 在场（VT/V 条件）不注入 ctx（2026-11-01，续写 v2 实验矩阵）.

底座 = v7_c2ctx（软 ctx 续写 v1）；本入口只加 STRUCTURE 键。G1
（scheduled sampling）是训练配方键 ctx_ss（train_args），不走结构入口。
"""

from anysole.models.v7_c2ctx import build as _build
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx_vcond": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return _build(config)
