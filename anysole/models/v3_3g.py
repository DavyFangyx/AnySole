"""V3_Gv1 entry: regression carrier (frozen E V-stream) + part-wise
observation layer (2026-10-17).

Fresh architecture (2026-09-24 independence restructure): the structure is
fixed here; the T contact stream and the T pose prior train from zero.
The V carrier is E v2.1's — loaded from ``carrier_from`` (per-config E
production ckpt_best), copied key-by-key into the V-stream submodules and
frozen at construction (model_v3g freezes them; this entry only loads).
Provenance (source path + sha256 + key count) is recorded into CONFIG_EXTRA
and lands in every checkpoint's config.

Structure summary (model_v3g.py + constraints_g.py):
  V: frozen E carrier — v_enc -> v_temporal -> 9-part pose head + traj head.
  T: foot_conv -> contact head (f6_soft, air-weighted BCE); t_temporal ->
     9-part prior head + prior traj head (T2M row).
  Inference: constraints_g.apply_v3g_constraints — T-derived floor +
     contact-gated two-bone leg lift (pose space) + E-v2.1 residual root lift.
"""

from __future__ import annotations

import hashlib

import torch

from anysole.models.common.model_v3g import CARRIER_PREFIXES, AnySoleModelV3G

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v3g": True,
}

# Carrier provenance recorded into the ckpt config at build time (one
# process = one config, so the module-level dict is safe here).
CONFIG_EXTRA: dict = {}


def build(config: dict, part_joints=None):
    """Construct the V3_Gv1 network; load and freeze the E V-stream carrier."""
    model = AnySoleModelV3G(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
    )
    carrier = str(config.get("carrier_from") or "")
    if not carrier:
        raise ValueError(
            "V3_Gv1 requires carrier_from (the E v2.1 ckpt_best whose V stream "
            "is the frozen carrier); the registry entries carry it per config")
    ckpt = torch.load(carrier, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    keys = [k for k in state if k.startswith(CARRIER_PREFIXES)]
    if not keys:
        raise ValueError("no V-stream keys found in carrier %r" % carrier)
    # Carrier-scope completeness: every carrier-prefix key of THIS model must
    # be present in the E state (the structure is copied verbatim, so the
    # scope must cover fully).  The T stream / prior keys stay unloaded —
    # they train from zero (independence: no warm start).
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
