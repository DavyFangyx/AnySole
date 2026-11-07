"""V8_Cv8（2026-11-07）：V8 基座 + 竞争路由门（k = σ(s_v − s_t)）。

结构 = V8_Cv5（双门 + 枚举 + SD2 + 锚头）的门控换成竞争路由：

* ``v7_route_router``：k = σ(s_v − s_t)，逐帧逐组（vel/other/left/right
  → 10 槽）。s_v 读 V token；s_t 读 T token + 分组物理/IMU 特征
  （T_phys 的 cop/force/envelope/spatial/temporal + T_s2m 的双足 IMU
  acc/gyro + 接触二值；vel/other 双足 26 维、left/right 单足 13 维，
  LayerNorm 后入 MLP）。
* 解码器 V 支路乘 (1−k)、T 支路乘 k——权重守恒的软路由，V 可被压掉
  （旧门只能衰减 T，occ100 收益被压 +3.66 是结构必然）；V-only 行
  k≡0、T-only 行 k≡1（硬掩码）。
* init：s_v≡0、s_t≡+4 ⇒ k≈0.018——VT 行 init ≈ 纯 V，T 支路须学会
  赢取信任（旧门 = 窗口难度检测器：ρ(g,Δ(VT−V))≈0；本门的目标 =
  价值仲裁器）。

对照 = V8_Cv5（单侧双门）与 V8_Cv7（无门）；同折同种子。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True,
             "v7_trans_anchor_head": True,
             "v7_route_router": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        route_router=bool(config.get("v7_route_router", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
