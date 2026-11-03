"""V7_Cv5AR2MSC（2026-11-03，按模态分窗 Split Context）：AR2M + V 流只看当前窗。

V 与 T 的时间尺度不同：V 给瞬时姿态（观测上下文 20→80，V2M PA 44.99→48.41 变差），
T 给步态相位（20→80，T2M PA 91.63→82.76 变好）。本结构在模型内把 V 流裁到当前窗 tw，
T 流保留 obs_context 长历史；T 信任门同步只池化当前窗。dataset / eval / 脚本照旧传完整
三路历史，无需改动。相对 V7_Cv5AR2M 的唯一结构差异 = v7_vcur。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True, "v7_ar2_fb": False,
             "v7_vcur": True}
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
        v_cur=bool(config.get("v7_vcur", False)))
