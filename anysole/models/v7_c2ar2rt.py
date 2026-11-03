"""V7_Cv5AR2RT（2026-11-02）: AR2R + VT 零初始化触觉残差.

论文模型格（VT 融合重构）：VT = V 主路径 + T 残差（zero-init，仅
v_present & t_present 同现行生效）——清洁场景 VT ≈ V（结构保证不亏），
退化场景残差兑现（论文命题「触觉 = 零税鲁棒性模态」的结构载体）。
其余与 AR2R 逐位同源（双向注意力、无反馈、无因果掩码、无 SS/噪声）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar2": True, "v7_ar2_fb": False,
             "vt_t_residual": True}
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
        vt_t_residual=bool(config.get("vt_t_residual", False)),
        obs_context=int(config.get("obs_context", config["tw"])))
