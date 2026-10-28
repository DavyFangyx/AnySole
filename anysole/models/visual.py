"""Shared RGB-only student for the matched view-consistency experiments.

The pressure head is an explicit checkpoint field: legacy null tokens or
parameter-free time PE before fusion. No tactile observation enters forward.
"""
from __future__ import annotations

import hashlib
import json

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.utils.a2_config import A2_HEADS, A3_VERSION, is_a2
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
    b1 = config.get("b1_version") is not None
    if b1 and config["b1_version"] != "anysole.b1.v1":
        raise ValueError("unsupported b1_version")
    if not b1 and "b1_role" in config:
        raise ValueError("b1_role requires explicit b1_version")
    if b1 and (config.get("b1_role") not in ("baseline", "teacher", "student")
               or config.get("a2_version") != A3_VERSION
               or config.get("pressure_head") != "query_factorized"
               or config.get("query_memory") != "fusion_tactile_slice"):
        raise ValueError("B1 requires an explicit role and the A3 T20 Q-FACT architecture")
    if config.get("a2_version") == A3_VERSION and config.get("pressure_head") != "query_factorized":
        raise ValueError("A3 requires Q-FACT")
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
        memory = config.get("query_memory") if config.get("a2_version") == A3_VERSION else "fusion_visual_slice"
        if config.get("a2_version") == A3_VERSION and (config["pressure_head"] != "query_factorized"
                or memory not in ("fusion_tactile_slice", "fusion_all")):
            raise ValueError("A3 requires Q-FACT with tactile-only or full fusion memory")
        if config.get("query_memory", memory) != memory:
            raise ValueError("query_memory disagrees with versioned architecture")
        architecture.update(a2_head_version="anysole.pressure-readout.v1", a2_head_seed=config.get("a2_head_seed"),
                            query_layers=1, query_heads=8, query_hidden=128,
                            query_memory=memory, query_time="relative_actual_seconds_x40",
                            query_mask="rgb_valid_true", foot_layout="left48_right48",
                            capacity_width=config.get("capacity_width") if config["pressure_head"] == "capacity_time_pe" else None,
                            output_foot_bias=config["pressure_head"] in ("query_nonnegative", "query_factorized"))
    if not 0.0 <= architecture["dropout"] < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if b1:
        architecture.update(b1_version=config["b1_version"], b1_role=config["b1_role"])
        if config["b1_role"] == "teacher":
            frozen = {"teacher_reference_camera": 1, "teacher_cross_view_layers": 1,
                      "teacher_refinement_layers": 1, "teacher_geometry_dim": 16}
            for key, expected in frozen.items():
                if config.get(key, expected) != expected:
                    raise ValueError("B1 teacher requires %s=%r" % (key, expected))
            view_dropout = config.get("teacher_view_dropout", {"three_view_probability": .7, "min_views": 2})
            if view_dropout != {"three_view_probability": .7, "min_views": 2}:
                raise ValueError("B1 teacher view dropout is frozen at 70% three / 30% two")
            architecture.update(**frozen, teacher_view_dropout=dict(view_dropout))
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
    model_class = AnySoleModelV2
    extra = {}
    if effective.get("b1_version") and effective.get("b1_role") == "teacher":
        from anysole.models.multiview_teacher import AnySoleMultiviewTeacher
        model_class = AnySoleMultiviewTeacher
        extra["three_view_probability"] = effective["teacher_view_dropout"]["three_view_probability"]
    model = model_class(
        d=effective["d_model"], tw=effective["tw"], nhead=effective["nhead"],
        dropout=effective["dropout"],
        pose_layers=effective["pose_layers"], pressure_head="time_pe" if effective["pressure_head"] in A2_HEADS else effective["pressure_head"],
        pressure_time_pe=effective["pressure_time_pe"], time_pe_scale=effective["time_pe_scale"],
        **STRUCTURE, **extra,
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
