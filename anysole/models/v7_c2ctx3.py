"""V7_Cv5Ctx3（2026-11-01）: Cv5 + 续写 v3（V/VT 回归 + T-only 硬填式）.

Cv5 双门结构不变。相对 v1/v2 的两条调整（判别实验钉死的方向）：
  A —— V 在场（VT/V 行）时 ctx 完全不注入：门控使这两行回归 Cv5 行为，
      训练时无 ctx 迫使融合重新长出证据通路（v1 的融合结构性损伤）；
  B —— T-only 用 Step2Motion 式硬填重叠（Level 3）：ctx = 重叠区 nf 帧
      × [根 6D + 根速度 3 + flag]（评估链 = 上一窗后 nf 帧预测，训练 =
      GT 前 nf 帧），替换式注入前 nf 帧的根速度/根朝向 query 槽位——
      重叠区与上一窗构造性一致（inpainting 的回归等价物）。
评估：T2M 走 stride=tw//2 链式（发射规则同 v1）；V/VT 回非重叠协议。
零初始化保留 init 逐位 = Cv5 契约（V7_续写协议方案 §9/§10）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2ctx import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ctx3": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        prev_ctx=bool(config.get("v7_prev_ctx", False)),
        ctx3=bool(config.get("v7_ctx3", False)))
