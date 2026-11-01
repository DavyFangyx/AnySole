"""V7_Cv5Ctx5（2026-11-01）: Cv5 + 续写 v5（块自回归：全姿态硬填）.

Cv5 双门结构不变。相对 ctx3 的一条核心改动（块自回归，用户裁定）：
  ctx 块从根状态摘要（nf×10）扩为完整运动历史（nf × [全 24 关节 6D
  (144) + 根速度 3 + flag 1]），全部 10 个 query 槽位逐部位替换式注入
  （根速度槽位读 vel+flag、各部位槽位读其关节 6D 历史+flag）——上一块
  的后 nf 帧完整硬填，发射区从完整历史出发，身体姿态接缝也构造性钉死
  （Step2Motion 式拷贝的回归最大密度版）。
门控同 ctx3（V 在场/flag=0 不替换）；评估 T 行链式携带 (nf,148) 块、
V/VT 行非重叠协议。零初始化保留 init 逐位 = Cv5 契约。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx5": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx3=bool(config.get("v7_ctx3", False) or config.get("v7_ctx5", False)),
        ctx5=bool(config.get("v7_ctx5", False)))
