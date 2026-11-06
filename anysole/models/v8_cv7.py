"""V8_Cv7（2026-11-05）：V8 消融——无门 + 枚举 + SD2 + 锚头。

结构 = V8_Cv5 去掉 T 信任门的版本：``v7_a`` 的独立证据 + 共享生成，
其余与 V8_Cv5 逐项相同（``v7_root_sd`` + ``v7_root_sd_balance`` +
``v7_trans_anchor_head``）。对照 = V8_Cv5（双门）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_a import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        contact_root=bool(config.get("v7_contact_root", False)),
        reliability_route=False,
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
