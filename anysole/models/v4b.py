"""V4_Av3 entry: frozen V3_Bv1 backbone + V4 correction branch (2026-10-25).

Two-stage carrier line (用户裁定，V4_方案 §6 V4_Av3 条目): the stage-1
scaffold is the V3_Bv1 production ckpt_best (its V stream was trained
with T in the shared fusion — the fusion tax lives in its V2M readout;
the stage-2 correction branch is T's antidote + the V-fail helper).
Stage 2 trains ONLY the correction branch + contact head — the backbone
is frozen at construction (model_v4b freezes it; this entry only loads).

Structure summary (model_v4b.py):
  B backbone (frozen): v_enc/foot_conv -> fusion transformer -> 9-part
     pose head (soft A, sigma gate) + traj head + aux heads — state-dict
     identical to V3_Bv1's entry.
  Correction (from zero, position space): PartXAttn(q = slots 5-8 l/r
     leg + l/r foot part tokens, kv = same-frame t_tok, kin bias
     -beta*d_kin) -> contact gate a_p (zero-init) -> per-part position
     heads (zero-init) -> six-joint pelvis-relative residual Δx (SG-7).
     Soft skeleton at metric level (V4_Av1 同款).
  T contact head (from zero): MLP(256->64->2) f6_soft BCE.
  Three conditions: V2M = B V-only row; VT2M = B VT row + pos_corr;
     T2M = B T-only row + pos_corr_prior.
"""

from __future__ import annotations

import hashlib

import torch

from anysole.models.common.model_v4b import AnySoleModelV4B

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
    """Construct the V4_Av3 network; load and freeze the B backbone."""
    beta = float(config.get("beta", 1.0))
    if beta not in BETA_SWEEP:
        raise ValueError(
            "V4_Av3 beta must be one of the pre-registered sweep %s, got %r"
            % (BETA_SWEEP, beta))
    carrier = str(config.get("carrier_from") or "")
    if not carrier:
        raise ValueError(
            "V4_Av3 requires carrier_from (the V3_Bv1 production ckpt_best "
            "whose backbone is the frozen scaffold); the registry entries "
            "carry it per config")
    model = AnySoleModelV4B(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        beta=beta,
    )
    ckpt = torch.load(carrier, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    backbone = {k for k in model.state_dict()
                if not (k.startswith("contact_head.")
                        or k.startswith("corr_attn.")
                        or k.startswith("contact_gate.")
                        or k.startswith("corr_heads.")
                        or k.startswith("sg7."))}
    missing, unexpected = model.load_state_dict(
        {k: state[k] for k in backbone if k in state}, strict=False)
    if unexpected:
        raise RuntimeError("unexpected keys from B ckpt: %s" % (unexpected[:8],))
    # Scaffold completeness: every backbone key must load from the B state
    # (the backbone is constructed verbatim, so the scope must cover fully).
    # The correction branch / contact head keys stay unloaded — they train
    # from zero (independence: no warm start).
    not_loaded = [k for k in backbone if k not in state]
    if not_loaded:
        raise RuntimeError("backbone keys absent from the B ckpt: %s"
                           % (sorted(not_loaded)[:8],))
    if not set(missing) <= {k for k in model.state_dict()
                            if not (k in backbone)}:
        raise RuntimeError("uncovered non-trainable keys: %s" % (sorted(missing)[:8],))
    # 载体策略（2026-10-25）：carrier_lr_mult > 0 → 双阶段精调（骨架可训，
    # 学习率分组 0.1×，train.py 优化器侧生效）；= 0/缺省 → 骨架冻结
    # （V4_Av3 口径）。冻结只对 V4_Av3 生效。
    carrier_lr_mult = float(config.get("carrier_lr_mult", 0.0))
    if carrier_lr_mult <= 0.0:
        for name, param in model.named_parameters():
            if name in backbone:
                param.requires_grad_(False)
    else:
        CONFIG_EXTRA["carrier_lr_mult"] = carrier_lr_mult
    digest = hashlib.sha256(open(carrier, "rb").read()).hexdigest()
    CONFIG_EXTRA["carrier"] = {
        "source": carrier, "sha256": digest, "n_keys": len(backbone),
    }
    CONFIG_EXTRA["beta"] = beta
    return model
