"""V8_Cv5VG（2026-11-06，修改 14 §5）：V8 基座 + 双向信任门。

结构 = V8_Cv5（双门 + 枚举 + SD2 + 锚头）+ 两处镜像改动：

  * ``v7_v_gate``：在 V 交叉注意力输出上加 V 信任门
    ``g_v = 2·sigmoid(route_v_mlp(cat(v_tok, t_tok)))``，末层零初始化
    ⇒ init 时 ``g_v ≡ 1``（逐位 = Cv5 的 V 支路）。
  * ``v7_gate_2sigma``：T 门值域由 (0,1) 改成 (0,2)，末层零初始化
    ⇒ init 时 ``g_t ≡ 1``（T 支路逐位 = 无门 Cv7）。

两旗同开时模型在 init 时等于「无门」的 T 支路 + 全信任的 V 支路；
训练后 g_v→0 让 U@VT 连续插值到 U@T（修改 14 §5.2），g_t 随遮挡上行。

对照 = V8_Cv5（单侧门）与 V8_Cv7（无门）；同折同种子。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_v_gate": True, "v7_gate_2sigma": True,
             "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        v_gate=bool(config.get("v7_v_gate", False)),
        gate_2sigma=bool(config.get("v7_gate_2sigma", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
