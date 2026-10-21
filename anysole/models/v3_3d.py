"""V3_3D — frozen-V + joint position head (ruled 2026-10-15).

The position-space successor of V3_3C.  V3_3C's residual added its
correction in 144-dim 6D pose space, whose conversion loss eats ~90% of T's
position-level increments (z_note/multimodal_fusion/Appendix/T信息量判定实验.md §3.2); the residual-paradigm
ridge bounds (§3.2 残差范式 ridge 界) additionally showed a FROZEN-V
additive branch is capped at ~−1mm in any output space — ~90% of T's
input-level increment (−10.3/−19.9mm on the six-joint relative position /
foot height) is co-adaptation, i.e. only a JOINT readout of [v_tok; t_tok]
can reach it.

- STRUCTURE is identical to V3_3B plus ``position_head=True`` (the V3_3C
  ``residual_t`` branch is gone — position_head replaces it, the two are
  mutually exclusive).
- build() freezes every base parameter and unfreezes only
  ``position_head``: the V path, the fusion transformer, the pose/traj/aux
  heads never receive a gradient (the V3_3C discipline).
- The frozen base runs in its V-only regime under every condition (model_v2
  feeds the fusion a null tactile stream whenever a correction branch
  exists); the real tactile stream reaches the model only through the
  position head, which reads the PRE-FUSION tokens (the fusion destroys
  T's position increments: vt −10.3mm → F_VT +10.0mm).
- The head's zero-initialized output projections make the corrected outputs
  (FK keypoints of the six lower-limb joints, root trajectory) bit-identical
  to the frozen base at init — the VT == V structural guarantee, now in
  position space.  The head is active under all conditions (the encoders'
  null replacement handles the missing stream), so V-only/T-only outputs
  carry the same readout.
- Soft skeleton: metrics are computed on the corrected keypoints
  (eval_protocol), while the exported pose/mesh stays the base output.
- This fine-tune is a deliberate, sanctioned exception to the 2026-09-24
  independence rule (no warm-start lineage): the base checkpoint is recorded
  in the run manifest (checkpoint_source + sha256).

See z_note/multimodal_fusion/Appendix/T信息量判定实验.md §3.2 for the design bounds.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.v3_3b import STRUCTURE as V3_3B_STRUCTURE

STRUCTURE = {**V3_3B_STRUCTURE, "position_head": True}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct V3_3D from the effective train config, then freeze the base.

    The optimizer (built later in train.py, after --init-from) therefore
    contains only the position-head parameters.
    """
    model = AnySoleModelV2(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        part_joints=part_joints,
        **STRUCTURE,
    )
    model.requires_grad_(False)
    for p in model.position_head.parameters():
        p.requires_grad_(True)
    return model
