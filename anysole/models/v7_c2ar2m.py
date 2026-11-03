"""V7_Cv5AR2M（2026-11-02）: Cv5 + 因果掩码（AR2 对照格，无反馈支路）.

AR2 实验矩阵的「关闭完整姿态反馈」行：ar2=True + ar2_fb=False——因果
掩码全条件生效，148 维反馈支路不建（ar2_fb 不存在、prev_state 不消费）。
与 V7_Cv5AR2 同上下文配对研究反馈效应。相对 Cv5 还改变了运动表示、
航向读出和损失，因此 AR2M − Cv5 不能解释为纯掩码效应。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True, "v7_ar2_fb": False}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar2=bool(config.get("v7_ar2", False)),
        ar2_fb=bool(config.get("v7_ar2_fb", True)),
        obs_context=int(config.get("obs_context", config["tw"])))
