"""Shared RGB-only student for the matched view-consistency experiments.

The pressure head is an explicit checkpoint field: legacy null tokens or
parameter-free time PE before fusion. No tactile observation enters forward.
"""
from __future__ import annotations

import hashlib
import json

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.utils.a2_config import A2_HEADS, is_a2
from anysole.models.v3_3b import STRUCTURE as V3_3B_STRUCTURE


STRUCTURE = {**V3_3B_STRUCTURE, "visual_only": True}
CONFIG_EXTRA = {
    "visual_structure_version": "visual_v1",
    "pressure_time_pe": "before_fusion",
    "time_pe_scale": 40.0,
}


def effective_config(config: dict) -> dict:
    """Validate and complete architecture fields before training or loading.

    Runtime persists this returned config in the checkpoint, alongside model
    state. Operational fields do not affect this architecture-only hash.
    """
    a2 = is_a2(config)
    if config.get("pressure_head") not in (("legacy", "time_pe") if not a2 else A2_HEADS):
        raise ValueError("visual checkpoints require explicit pressure_head='legacy' or 'time_pe'")
    enabled = config["pressure_head"] != "legacy"
    if config.get("pressure_time_pe_enabled", enabled) != enabled:
        raise ValueError("pressure_time_pe_enabled contradicts pressure_head")
    fixed = {**STRUCTURE, **CONFIG_EXTRA, "d_model": 256, "tw": 20, "pose_layers": 6,
             "nhead": 8, "modal": "anysolev2", "pose_repr": "6d"}
    for key, expected in fixed.items():
        if key in config and config[key] != expected:
            raise ValueError("visual structure mismatch for %s: %r != %r" % (key, config[key], expected))
    architecture = {**fixed, "pressure_head": config["pressure_head"],
                    "pressure_time_pe_enabled": enabled,
                    "dropout": float(config.get("dropout", 0.1)), "model_name": "visual"}
    if a2 and config["pressure_head"] not in ("legacy", "time_pe"):
        architecture.update(a2_head_version="anysole.pressure-readout.v1", a2_head_seed=config.get("a2_head_seed"),
                            query_layers=1, query_heads=8, query_hidden=128,
                            query_memory="fusion_visual_slice", query_time="relative_actual_seconds_x40",
                            query_mask="rgb_valid_true", foot_layout="left48_right48",
                            capacity_width=config.get("capacity_width") if config["pressure_head"] == "capacity_time_pe" else None,
                            output_foot_bias=config["pressure_head"] in ("query_nonnegative", "query_factorized"))
    if not 0.0 <= architecture["dropout"] < 1.0:
        raise ValueError("dropout must be in [0,1)")
    digest = hashlib.sha256(json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if config.get("structure_hash") not in (None, digest):
        raise ValueError("visual structure_hash does not match checkpoint architecture")
    if config.get("model_name") not in (None, "visual"):
        raise ValueError("visual builder requires model_name='visual'")
    return {**config, **architecture, "structure_hash": digest}


def build(config: dict, part_joints=None):
    if part_joints is not None:
        raise ValueError("visual uses the fixed nine-part soft assignment")
    effective = effective_config(config)
    model = AnySoleModelV2(
        d=effective["d_model"], tw=effective["tw"], nhead=effective["nhead"],
        dropout=effective["dropout"],
        pose_layers=effective["pose_layers"], pressure_head="time_pe" if effective["pressure_head"] in A2_HEADS else effective["pressure_head"],
        pressure_time_pe=effective["pressure_time_pe"], time_pe_scale=effective["time_pe_scale"],
        **STRUCTURE,
    )
    if effective["pressure_head"] not in ("legacy", "time_pe"):
        import torch.nn as nn
        from anysole.models.common.pressure_a2 import build_pressure_head
        model.pressure_readout = build_pressure_head(effective)
        # Remove the old pressure head: unused trainable capacity is forbidden.
        model.aux_heads.t_rec = nn.Identity()
        model.pressure_head = effective["pressure_head"]
        model.pressure_time_pe_enabled = True
    model.structure_hash = effective["structure_hash"]
    model.effective_config = effective
    return model
