"""V7_Cv2: V7_Cv1 with separate tactile gates for root and pose queries（2026-10-26 统一编号）."""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_task_routes": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True, task_routes=bool(config.get("v7_task_routes", False)),
        pressure_head=str(config.get("pressure_head", "none")),
        pressure_cell_mean=config.get("pressure_cell_mean"),
        pressure_seed=int(config.get("pressure_seed", config.get("seed", 1))))
