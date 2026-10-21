"""V3_3C — 定向修复 (ruled 2026-10-09): frozen-V + zero-init T-residual.

The name was reserved in model_fix_note §3.5 for the "directional repair" of
the fusion deficit (VT worse than V across all 12 architectures).  The ruling
replaced the originally-planned from-scratch gate fix (missing-stream zero
injection + gating differential loss) with a structurally guaranteed
alternative: freeze the V3_3B_vonly specialist wholesale and fine-tune only a
small T-residual branch on VT windows.

- STRUCTURE is identical to V3_3B plus ``residual_t=True``, so the checkpoint
  config records the branch flag and eval/infer structure validation passes.
- build() freezes every base parameter (requires_grad_(False)) and unfreezes
  only ``t_residual``; frozen params keep grad None forever, so although the
  optimizer holds every parameter, only the branch ever updates.
- The frozen base runs in its V-only regime under every condition (model_v2
  feeds the fusion a null tactile stream whenever the branch exists), so the
  real tactile stream reaches the model only through the branch.
- The branch's zero-initialized final layer guarantees VT == V at init (== the
  base's V-only output bitwise); the config_id mask bypasses the branch under
  V-only/T-only, so those outputs stay bit-identical to the frozen base.
- This fine-tune is a deliberate, sanctioned exception to the 2026-09-24
  independence rule (no warm-start lineage): the base checkpoint is recorded
  in the run manifest (checkpoint_source + sha256).

See z_note/multimodal_fusion/Appendix/T信息量判定实验.md §5.1 for the experiment design.
"""

from __future__ import annotations

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.v3_3b import STRUCTURE as V3_3B_STRUCTURE

STRUCTURE = {**V3_3B_STRUCTURE, "residual_t": True}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct V3_3C from the effective train config, then freeze the base.

    The optimizer (built later in train.py, after --init-from) therefore
    contains only the T-residual branch parameters.
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
    for p in model.t_residual.parameters():
        p.requires_grad_(True)
    return model
