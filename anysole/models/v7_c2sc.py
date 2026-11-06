"""V7_Cv5SC（2026-11-03，Cv5 按模态分窗 Split Context）：Cv5 + V 只看当前窗、T 吃长历史。

结构 = V7_Cv5（v7_c2 双独立 T 门，双向回归，非 AR2）+ v7_vcur：V 流在模型内裁到当前窗 tw，
T 流保留 obs_context 长历史，T 信任门只池化当前窗。目标表示仍是 Cv5 的绝对表示
（dataset 只放行长历史 long_obs，不切换到 AR2 相对目标）。要回答的问题：
T 长窗在非 AR2 架构里能否兑现。对照 = V7_Cv5（V20/T20）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_vcur": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        obs_context=int(config.get("obs_context", config["tw"])),
        v_cur=bool(config.get("v7_vcur", False)))
