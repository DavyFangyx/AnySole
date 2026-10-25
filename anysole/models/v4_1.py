"""V4_Av1 entry: frozen E v2.1 V-stream carrier + T contact stream +
part-token gated correction branch (2026-10-19).

Fresh architecture (2026-09-24 independence restructure): the structure is
fixed here; the T contact stream, the T prior tower and the correction
branch train from zero.  The V carrier is E v2.1's — loaded from
``carrier_from`` (per-config E production ckpt_best: cam3 = V3_Ev1_f6_soft,
occ010 = V3_Ev1_cam3_occ010_f6_soft), copied key-by-key into the V-stream
submodules and frozen at construction (model_v4_1 freezes them; this entry
only loads).  Provenance (source path + sha256 + key count) is recorded
into CONFIG_EXTRA and lands in every checkpoint's config.

Structure summary (model_v4_1.py):
  V: frozen E carrier — v_enc -> v_temporal -> 9-part pose head + traj head.
  T: foot_conv -> contact head (f6_soft, air-weighted BCE); t_temporal ->
     9-part prior head + prior traj head (T2M row).
  Correction (from zero, position space): XAttn(q = reachable part tokens
     l/r_leg + l/r_foot, kv = same-frame t_tok, kin bias -beta*d_kin) ->
     contact gate a_p (zero-init) -> per-part position heads (zero-init) ->
     six-joint pelvis-relative residual Δx (SG-7 smoothed).  Soft skeleton:
     build_corrected_skeleton (model_v4_1) replaces the six lower-limb
     joints of FK(pose) with FK + Δx — V3_Dv1 同款, metrics only.
"""

from __future__ import annotations

import hashlib

import torch

from anysole.models.common.model_v4_1 import CARRIER_PREFIXES, AnySoleModelV41

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v4_1": True,
}

# Carrier provenance recorded into the ckpt config at build time (one
# process = one config, so the module-level dict is safe here).
CONFIG_EXTRA: dict = {}

# Pre-registered beta sweep (V4_A方案 §2.2): 0 = no-mask ablation (对照,
# not committed), 1.0 = primary, 2.0.  The entry fails loudly outside it.
BETA_SWEEP = (0.0, 1.0, 2.0)


def build(config: dict, part_joints=None):
    """Construct the V4_Av1 network; load and freeze the E V-stream carrier."""
    beta = float(config.get("beta", 1.0))
    if beta not in BETA_SWEEP:
        raise ValueError(
            "V4_Av1 beta must be one of the pre-registered sweep %s, got %r"
            % (BETA_SWEEP, beta))
    # V4_Av2（2026-10-25）：carrier_from="scratch" → V 流从零可训（无 E
    # 载体、不冻结；E 式双流同训）。CONFIG_EXTRA 记录 scratch provenance。
    carrier = str(config.get("carrier_from") or "")
    carrier_scratch = carrier.strip().lower() in ("scratch", "none", "")
    model = AnySoleModelV41(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        beta=beta,
        carrier_scratch=carrier_scratch,
    )
    if carrier_scratch:
        CONFIG_EXTRA["carrier"] = {"source": "scratch", "n_keys": 0}
        return model
    if not carrier:
        raise ValueError(
            "V4_Av1 requires carrier_from (the E v2.1 ckpt_best whose V stream "
            "is the frozen carrier); the registry entries carry it per config")
    ckpt = torch.load(carrier, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    keys = [k for k in state if k.startswith(CARRIER_PREFIXES)]
    if not keys:
        raise ValueError("no V-stream keys found in carrier %r" % carrier)
    # Carrier-scope completeness: every carrier-prefix key of THIS model must
    # be present in the E state (the structure is copied verbatim, so the
    # scope must cover fully).  The T stream / prior tower / correction
    # branch keys stay unloaded — they train from zero (independence: no
    # warm start).
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
    CONFIG_EXTRA["beta"] = beta
    return model
