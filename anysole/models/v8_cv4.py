"""V8_Cv4（2026-11-05）：V8 消融——单门 + 枚举 + SD2 + 锚头。

结构 = V8_Cv5 的单门版本：``v7_c`` 的窗口级共享门（根/姿态共用），
其余与 V8_Cv5 逐项相同（``v7_root_sd`` + ``v7_root_sd_balance`` +
``v7_trans_anchor_head``）。对照 = V8_Cv5（双门）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=False,
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
