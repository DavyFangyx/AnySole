"""V7_Cv3: V7_Cv1 with private residual adapters active only for T-only inference（2026-10-26 统一编号）."""

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.v7_c import STRUCTURE as BASE_STRUCTURE

STRUCTURE = {**BASE_STRUCTURE, "v7_t_only_adapters": True}
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    return AnySoleModelV7(
        d=int(config["d_model"]), tw=int(config["tw"]),
        dropout=float(config.get("dropout", .1)),
        pose_layers=int(config.get("pose_layers", 6)),
        reliability_route=True,
        t_only_adapters=bool(config.get("v7_t_only_adapters", False)))
