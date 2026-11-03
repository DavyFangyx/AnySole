"""V7_Cv5AR2RL / V7_Cv5AR2RLD（2026-11-03，z_note/修改/04）：AR2 + rollout 损失.

修改 04 的 R / RD 两实验格共用本入口（结构 = AR2：v7_ar2=True、
ar2_fb=True——04 开工裁定：AR2M 无反馈支路（ar2_fb=False、prev_state
不消费），TF≡rollout 按构造，改动 A 在 AR2M 原样上没有作用对象，
反馈支路必须开启）：

  R  = AR2 结构 + 改动 A（多步自由 rollout 损失；train_args
       ar_rollout 族，默认全关 → 逐位复现 AR2 训练）；
  RD = R + 改动 B（反馈降维：v7_ar2_lowdim + v7_ar2_fb_dim，默认关
       → 反馈 = 完整 148 维）。

两处改动各自独立可关（配置键，默认关）；「改动全关」= AR2 原样
（结构/训练逐位一致——契约断言见 tests/test_v7_ar2_rollout.py）。
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
        ar2_fb=bool(config.get("v7_ar2_fb", True)),
        ar2_lowdim=bool(config.get("v7_ar2_lowdim", False)),
        ar2_fb_dim=int(config.get("v7_ar2_fb_dim", 16)),
        obs_context=int(config.get("obs_context", config["tw"])))
