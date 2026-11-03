"""V7_Cv5ARSS（2026-11-01）: AR + 反馈 scheduled sampling + 收缩噪声.

结构同 v7_c2ar（独立入口名 = 独立结果路径——train_args ar_ss/ar_noise
由 z_gen registry 携带；与 AR_s1 共用路径会被 runner 拒绝（同名 ckpt
不完整冲突））。配方键：--ar-ss 线性升至 0.3、--ar-noise 0.1（只扰
姿态分量）。
"""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c2 import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_ar": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=True,
        ar=bool(config.get("v7_ar", False)))
