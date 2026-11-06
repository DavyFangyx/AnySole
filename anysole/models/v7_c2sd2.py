"""V7_Cv5SD2（2026-11-04，修改 07 第二臂 B′）：SD 结构 + 恒等式配平损失。

结构 = V7_Cv5SD（v7_c2sd：Cv5 + 根头三通道读出 + T 行方向掩码）**逐参数
相同**；唯一差异 = 损失配平形式（``v7_root_sd_balance``）：

  * 速度项按水平分量精确恒等式再分配：
    ``‖v−v_gt‖²_h = (s−s_gt)² + 2·s·s_gt·(1−cos)``，三项各乘 1/3；
    全权重 1.0 且不掩码时 ``L_speed+L_vy+L_dir ≡ l_vel``（浮点结合序以内），
    于是 T 行方向掩码成为唯一干预变量。
  * 多尺度位移项对 Δ 向量用同一恒等式再分配（径向/横向/垂直），取代
    现版「向量项 + 标量幅度项」两个独立权重相加。

对照 = V7_Cv5SD_act8020_s1（默认权重臂，同一结构、同 split/seed）与
V7_Cv5_act8020_s1（零税基线）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

# 2026-11-05 用户裁定：平移锚头是模型默认组件，SD2 主线一并带上。
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
        task_routes=bool(config.get("v7_task_routes", False)),
        root_sd=bool(config.get("v7_root_sd", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
