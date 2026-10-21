"""V5_1 entry: frozen E v2.1 carrier + V/T dual z experts + PoE + FiLM
lower-limb injection (2026-10-19, V5_1A方案.md).

Fresh architecture (2026-09-24 independence restructure): the structure is
fixed here; the z experts / FiLM / PoE weights / T prior tower train from
zero.  The V carrier is E v2.1's — loaded from ``carrier_from`` (per-config
E production ckpt_best: cam3 → V3_3E_f6_soft、occ010 →
V3_3E_cam3_occ010_f6_soft), copied key-by-key into the V-stream submodules
and frozen at construction (model_v5_1 freezes them; this entry only
loads).  Provenance (source path + sha256 + key count) is recorded into
CONFIG_EXTRA and lands in every checkpoint's config.

Structure summary (model_v5_1.py):
  V: frozen E carrier — v_enc -> v_temporal -> 9-part pose head + traj
     head (each with a FiLM hook on lower-limb part features / traj
     feature; FiLM zero-init => training start VT2M == V2M == E v2.1).
  z: g_V(v_tok) / g_T(t_tok) experts (contact logits 2 + phase 4 + CoP 4,
     10 channels) -> PoE closed-form synthesis (precision-weighted
     continuous; logit-weighted Bernoulli contact) -> FiLM.
  T: foot_conv -> contact head (f6_soft, air-weighted BCE, G 配方);
     t_temporal -> 9-part prior head + prior traj head (T2M sanity row).
  Inference: zero-parameter — no constraint layer, no GT floor input
     (硬性纪律：无锚-几何组件；穿透/Global 行只作报告，不进机制).
"""

from __future__ import annotations

import hashlib

import torch

from anysole.models.common.model_v5_1 import CARRIER_PREFIXES, AnySoleModelV5_1

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v5_1": True,
}

# Carrier provenance recorded into the ckpt config at build time (one
# process = one config, so the module-level dict is safe here).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V5_1 network; load and freeze the E V-stream carrier."""
    model = AnySoleModelV5_1(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        t_phys_v2=bool(config.get("t_phys_v2", False)),
        d8_heads=bool(config.get("d8_heads", False)),
    )
    carrier = str(config.get("carrier_from") or "")
    if not carrier:
        raise ValueError(
            "V5_1 requires carrier_from (the E v2.1 ckpt_best whose V stream "
            "is the frozen carrier); the registry entries carry it per config")
    ckpt = torch.load(carrier, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    keys = [k for k in state if k.startswith(CARRIER_PREFIXES)]
    if not keys:
        raise ValueError("no V-stream keys found in carrier %r" % carrier)
    # Carrier-scope completeness: every carrier-prefix key of THIS model must
    # be present in the E state (the structure is copied verbatim, so the
    # scope must cover fully).  The T stream / experts / FiLM keys stay
    # unloaded — they train from zero (independence: no warm start).
    carrier_scope = {k for k in model.state_dict() if k.startswith(CARRIER_PREFIXES)}
    uncovered = carrier_scope - set(keys)
    if uncovered:
        raise RuntimeError(
            "carrier keys present in the model but absent from the ckpt: %s"
            % (sorted(uncovered)[:8],))
    missing, unexpected = model.load_state_dict(
        {k: state[k] for k in keys}, strict=False)
    if unexpected:
        raise RuntimeError("unexpected carrier keys: %s" % (unexpected[:8],))
    non_carrier = {k for k in model.state_dict() if not k.startswith(CARRIER_PREFIXES)}
    if not set(missing) <= non_carrier:
        raise RuntimeError("uncovered non-trainable keys: %s" % (sorted(missing)[:8],))
    digest = hashlib.sha256(open(carrier, "rb").read()).hexdigest()
    CONFIG_EXTRA["carrier"] = {
        "source": carrier, "sha256": digest, "n_keys": len(keys),
    }
    return model
