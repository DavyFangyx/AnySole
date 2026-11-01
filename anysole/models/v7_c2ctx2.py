"""V7_Cv5Ctx2（2026-11-01）: Cv5 + 续写 v2（ctx 直连根朝向）.

Cv5 双门结构不变。续写 v1（v7_c2ctx）之上两条直连强化：
  Level 1 —— ctx_vec 同时加性注入根部位（朝向）query 槽位（不再只经
             解码器自注意力间接读取）；
  Level 2 —— 根朝向输出加 ctx 残差（10→6 MLP，零初始化）。
窗首帧根状态 ctx（10 维）、评估链、发射规则同 v1；零初始化保留
init 逐位 = Cv5 契约（V7_续写协议方案 §9）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx_v2": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx_v2=bool(config.get("v7_ctx_v2", False)))
