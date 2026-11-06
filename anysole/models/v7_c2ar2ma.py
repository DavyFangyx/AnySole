"""V7_Cv5AR2MA（2026-11-04，修改 09）：AR2M + 读出按条件分支（A = absolute V/VT）。

动机（act8020 val 实测）：AR2 的相对表征 + 外部 ψ 让 T 行学会了转向——5 个 180°
折返会话上绝对航向 2–8°、漂移 1–6°，而 Cv5 是 76–117°（冻结在初始朝向）。代价是
V/VT 行 PA 从 28.9 退到 37.7（+8.8mm），且该退化沿运动学链放大（spine +2.2 →
anklefoot +18.3，现状.md §四）——来源是「视觉本来直接观测到绝对航向，却被迫经
Δψ 二次组装」。

本格 = 现状.md §一 表格里写了但代码未实现的那条：
  V / VT 行：目标空间 = 绝对（= Cv5），根 6D 直接是世界朝向，v_hat 是世界系速度；
  T    行：保持 AR2 相对表征（tilt + Δψ + 航向系速度）+ 外部 ψ 累积。
一个 ckpt 同时拿 Cv5 级的 V/VT 与 AR2 级的航向，即本格的判据。

结构差异（相对 V7_Cv5AR2M）唯一 = v7_abs_vrows，贯穿三处：
  1. 模型 V7MotionHead：去航向投影只作用于 T 行（v_present<0.5）；
  2. 损失 _compute_losses_v7_ar2：V/VT 行 pose 目标 = Y(ψ_gt)@tilt、速度目标 =
     vel_gt（世界系），T 行目标不变；yaw_fix 的世界轨迹项不再对已是世界系的
     v_hat 二次旋转；
  3. 评估 ar2_session_chunks：V/VT 行跳过 Y(ψ) 组装，T 行仍自由积分。
默认关 → AR2M 旧格逐位不变。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True, "v7_ar2_fb": False,
             "v7_abs_vrows": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar2=bool(config.get("v7_ar2", False)),
        ar2_fb=bool(config.get("v7_ar2_fb", True)),
        obs_context=int(config.get("obs_context", config["tw"])),
        abs_vrows=bool(config.get("v7_abs_vrows", False)))
