"""V7_Cv5TA（2026-11-04，修改 08）：Cv5 + V/VT 平移锚头（旁路读出）。

结构 = V7_Cv5（v7_c2 双独立 T 门，枚举三条件、双向回归、非 AR2）+
``v7_trans_anchor_head``：m_v 窗口均值 → 逐轴归一化残差 → 绝对平移锚
（米）。锚头只读主干表征（默认 stopgrad），零税契约下主干每一步与 Cv5
数值相同；``trans_hat``、现役 GT 锚消费（n_gt_anchors_used）与现役指标
列一行不动。新列 GlobalMPJPE_self / W-MPJPE100_self / RootAbs_self 与
现役列并列报。T 行锚 = 注册常数（压力读不出绝对位置，损失掩码）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_trans_anchor_head": True}
CONFIG_EXTRA: dict = {"lambda_tanchor": 1.0}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        task_routes=bool(config.get("v7_task_routes", False)),
        trans_anchor_head=bool(config.get("v7_trans_anchor_head", False)),
        trans_anchor_stopgrad=bool(config.get("v7_trans_anchor_stopgrad", True)))
