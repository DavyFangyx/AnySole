"""V7_Cv5AR2R（2026-11-02）: Cv5 + AR2 相对读出（纯回归收口）.

回归收口（用户裁定方向）：双向注意力、无反馈、无因果掩码、无 SS/噪声
——编码器/解码器/门控与 Cv5 同构，只换读出表示（Δyaw + 去yaw根倾斜 +
航向系根速度 + 视觉航向观测）与评估链（每窗一次并行前向 + 外部航向
状态跨窗积分，无逐帧串行）。CONFIG_EXTRA 的 ar2_regression 标记驱动
train.py（causal=False、反馈禁用）与 eval 协议（窗口并行组装）。

与 V7_Cv5AR2M 的差别：AR2M = 因果掩码 + 无反馈（串行评估）；本格 =
无掩码 + 无反馈（并行评估）。与 V7_Cv5AR2 的差别：去掉 148 维反馈支路
与同序列两遍式 scheduled sampling。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True, "v7_ar2_fb": False}
CONFIG_EXTRA: dict = {"ar2_regression": True}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar2=bool(config.get("v7_ar2", False)),
        ar2_fb=bool(config.get("v7_ar2_fb", True)),
        ar2_regression=bool(config.get("ar2_regression", False)),
        obs_context=int(config.get("obs_context", config["tw"])))
