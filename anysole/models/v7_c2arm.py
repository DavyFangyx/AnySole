"""V7_Cv5ARM（2026-11-01）: Cv5 + 因果掩码（AR 对照格，无反馈支路）.

Cv5 双门结构不变。帧级 AR 的掩码效应对照：ar=True + ar_fb=False——
因果掩码全条件生效，反馈支路不建（ar_fb 不存在、prev_state 不消费）。
与 V7_Cv5AR 配对：反馈效应 = AR − ARM，掩码效应 = ARM − Cv5。
零初始化保留 init 逐位 = Cv5 契约（掩码非参数，init 前向仍逐位一致）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar": True, "v7_ar_fb": False}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar=bool(config.get("v7_ar", False)),
        ar_fb=bool(config.get("v7_ar_fb", True)))
