"""V7_Cv5SD2RT（2026-11-05，修改 10 §三 可直接移植 ①）：SD2 + VT 零初始化触觉残差。

结构 = V7_Cv5SD2（v7_c2sd2：Cv5 + 根头三通道 + B′ 恒等式配平损失）+
``vt_t_residual``（AR2RT 移植）：仅在 V&T 同现的 VT 行生效的两个零初始化
适配器（root / pose），V 行 vt_mask=0 不生效、T 行不生效。

契约（实测口径，不写「结构钉死」）：
  * init 时适配器输出严格为零（末层零初始化，RNG fork 不扰动共享参数），
    所以 VT 输出 = SD2 的 VT 主路径输出，残差本身不引入任何初始损伤；
  * VT 主路径仍带 T（VT 与 V 在 init 时并不逐位相同）——「VT−V 的符号」
    由训练决定，不构成数学保证。AR2RT 的历史读数是 +7.0 → +1.8（部分兑现）。
  * 判据 = 无遮挡档 Δ(VT−V) 的符号 + occ010–100 曲线；控制臂 = SD2
    （同结构、仅差 vt_t_residual）。

其余与 SD2 逐位同源（入口 v7_c2sd2 的全部结构与损失）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

# 2026-11-05 用户裁定：平移锚头是模型默认组件，SD2RT 一并带上。
STRUCTURE = {**BASE_STRUCTURE, "v7_root_sd": True,
             "v7_root_sd_balance": True, "vt_t_residual": True,
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
        vt_t_residual=bool(config.get("vt_t_residual", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)))
