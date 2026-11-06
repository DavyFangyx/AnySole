"""V8_Cv5PO（2026-11-06）：V8 基座 + 修改 13 D3（按可观测性划定门的作用域）。

门不是坏了，是作用域划得太宽。实测落点：

* 踝足 Δ(VT−V) 在 **occ000 就四折同负**（Cv5 −0.640±0.476），逐受试者
  **6/6 同负**（−0.818±0.660）—— T→足/腿 这条路上没有可压的害。
* 门却对全部 10 个 query 槽一视同仁地衰减（Cv5 pose 门全区间 <0.5），
  于是在腿/足上门只能减收益。

所以把设计主张直接写进结构：**腿/足四部位（l_leg / r_leg / l_foot /
r_foot）的 T 信任门钉成 1.0**，门只留根速度槽与其余姿态槽。`gate_pin_legs`
是常量掩码，**零新增参数**、恰好一处语义改动（真单旗）；其余逐项 =
V8_Cv5（双门 + 枚举 + SD2 + 锚头）。

对照 = V8_Cv5_<fold>_s1（同折同种子）。判据（4/4 同向）：occ000 的
clean Δ踝足PA 更负 + occ080/100 的 Δ(VT−V) PA 更负；零税闸门 =
clean ΔPA 不升超过 +0.5。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_gate_pin_legs": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        gate_pin_legs=bool(config.get("v7_gate_pin_legs", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
