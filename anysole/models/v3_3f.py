"""V3_3F entry: diffusion prior + T guidance (ruled 2026-10-17).

Fresh architecture — deliberately independent of the anysolev2
fusion/gating lineage, V3_3D, and V3_3E (no code shared with any of them).
Structure summary (model_v3f.py):
  x = [pose | trans | contact] diffused; V-conditioned DiT denoiser (x0
  prediction); T contact head (f6_soft) + contact/ground guidance at
  sampling; post-sampling differentiable refinement over root + leg chain.

The frozen vonly visual encoder (sanctioned reuse, 方案 §2.1): the entry
loads the per-config V-only specialist's ``encoders.v_enc.*`` projection
and freezes it (V3_3F -> V3_3B_vonly, V3_3F_cam3_occ010 ->
V3_3B_cam3_occ010_vonly).  Embeddings stay fresh: time_pe is sinusoidal by
construction and the modality row trains with the T stream.  Everything
else — the denoiser, the T encoder, the contact head — trains from random
init (2026-09-24 independence rule).
"""

from __future__ import annotations

import torch

from anysole.models.common.model_v3f import AnySoleModelV3F
from anysole.types import anysole_model_dir

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v3f": True,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def _vonly_model_name(model_name: str) -> str:
    """Per-config frozen V specialist: V3_3F[_cam3_occNNN] ->
    V3_3B[_cam3_occNNN]_vonly (the specialist suffix follows the config
    suffix — V3_3B_cam3_occ010_vonly, not V3_3B_vonly_cam3_occ010)."""
    name = str(model_name or "")
    suffix = name[len("V3_3F"):] if name.startswith("V3_3F") else ""
    return "V3_3B" + suffix + "_vonly"


def build(config: dict, part_joints=None):
    """Construct V3_3F, then load + freeze the vonly visual encoder."""
    model = AnySoleModelV3F(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        denoise_layers=int(config.get("denoise_layers", 4)),
    )
    # The vonly specialists are scheduler-derived names, not registry
    # entries — the path is built directly (no registry validation).
    vonly_name = _vonly_model_name(str(config.get("model_name", "")))
    ckpt = anysole_model_dir(vonly_name, "joint_and") / "checkpoints" / "ckpt_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            "V3_3F needs the frozen vonly visual encoder but it is missing: %s\n"
            "Train the V-only specialist first (--model-name %s)."
            % (ckpt, vonly_name))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    src = state.get("model", {}) if isinstance(state, dict) else {}
    dst = model.state_dict()
    loaded = []
    for key, value in src.items():
        if key.startswith("encoders.v_enc.") and key in dst and dst[key].shape == value.shape:
            dst[key] = value.to(dst[key].dtype)
            loaded.append(key)
    if not loaded:
        raise RuntimeError("no encoders.v_enc.* keys copied from %s" % ckpt)
    model.load_state_dict(dst)
    for p in model.encoders.v_enc.parameters():
        p.requires_grad_(False)
    print("v3_3f: frozen vonly V encoder from %s (%d keys)" % (ckpt, len(loaded)))
    return model
