"""V3_3E_denoised entry（2026-10-17 降噪压力数据变体：结构同 V3_3E，tactile_input=denoised108）

V3_3E entry: constraint-fusion model (ruled 2026-10-17).

Fresh architecture — NOT the anysolev2 fusion/gating lineage and NOT the
V3_3D frozen-base lineage.  Independent model (2026-09-24 restructure):
the structure is fixed here; every training run starts from random init.

Structure summary (model_v3e.py):
  V: v_tok -> temporal encoder -> 9-part pose head (pure V, no T) -> 6D pose
     + per-joint log-sigma; traj head (velocity -> cumsum).
  T: t_tok -> contact head (per-foot logits, f6_soft supervision);
     t_tok -> temporal encoder -> 9-part prior head + prior traj head
     (T2M-level fallback prior).
  Inference: anysole/utils/constraints.py applies the inverse-variance
  fallback blend + support-static / ground / anti-penetration constraints
  (zero-parameter v1; the three eval conditions read one forward).
"""

from __future__ import annotations

from anysole.models.common.model_v3e import AnySoleModelV3E

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "denoised108",
    "v3e": True,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V3_3E network from the effective train config."""
    return AnySoleModelV3E(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
    )
