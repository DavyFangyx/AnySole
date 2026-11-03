"""V7_Cv5AR2（2026-11-01）: Cv5 + 帧级 AR 相对运动 + 航向状态分离.

（V7_AR相对运动与航向状态方案）共享 AR 解码器输出相对运动表示
[Δyaw 1 + 去yaw根倾斜 6 + 非根父局部 138 + 航向系根速度 3]，yaw 由
外部状态积分（训练窗口内 ψ = psi_base + cumsum(Δyaw)）；视觉在 V/VT 行
提供航向观测 + 校正增益（vis_yaw/vis_gain，仅评估组装使用，训练经
L_visyaw/L_yawcorr 监督）。反馈 = 上一帧 148 维状态（零初始化保留 init
逐位 = Cv5 契约）；训练同序列两遍式 scheduled sampling，推理逐帧串行
自由生成（eval 侧）。与 V7_Cv5AR 互斥（不同 prev_state 语义）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar2=bool(config.get("v7_ar2", False)),
        obs_context=int(config.get("obs_context", config["tw"])))
