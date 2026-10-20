"""V3_3E (ruled 2026-10-17): the constraint-fusion model.

Fresh architecture, deliberately NOT built on anysolev2 (the cross-modal
fusion transformer + learned sigma gating are the located interference
mechanism, Δ_fusion +13.06) nor on V3_3D (frozen-base position head — ruled
a bad model).  Three streams, zero cross-modal conditioning:

- V 姿态主干（纯 V，T 不进入）: v_tok -> temporal encoder -> 9-part pose
  head (hard PART_JOINTS, single-stream cross-attn, no gates, no soft A) ->
  144-dim 6D pose + per-joint log-sigma (Gaussian-NLL calibrated; the
  inverse-variance fallback weight).  Traj head: per-frame velocity ->
  cumsum (same contract as TrajHead).
- T 物理头: t_tok -> per-foot contact logits (2), supervised on f6_soft
  (lambda_contact, pos-weighted BCE — swing-phase air recall matters for the
  constraint layer).
- T 姿态先验 θ_T: t_tok -> temporal encoder -> the same 9-part head + traj
  head (T2M-level prior, active only through the constraint layer's
  inverse-variance fallback).

约束层 is inference-only in v1 (zero parameters): anysole/utils/constraints.py
applies the fallback blend, support-static root velocity (pose-space foot
velocity — the probe-A lesson), ground contact / anti-penetration, SG-7
smoothing.  Training supervises the three heads only.

The three eval conditions (V2M / VT2M / T2M) are three readouts of ONE
forward — the model ignores ``config_id`` entirely (no null tokens, no
condition sampling; train runs config_probs [1,0,0]).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    CONFIG_VT,
    D_MODEL,
    FPS,
    N_JOINTS,
    PART_JOINTS,
    POSE_DIM,
    TW,
)


def _temporal_encoder(dim: int, n_layers: int, nhead: int = 8,
                      dim_feedforward: int = 1024, dropout: float = 0.1):
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    kwargs = {"num_layers": n_layers}
    try:
        return nn.TransformerEncoder(layer, enable_nested_tensor=False, **kwargs)
    except TypeError:
        return nn.TransformerEncoder(layer, **kwargs)


class V3EPoseHead(nn.Module):
    """Pure single-stream 9-part pose head: part queries decode ONE memory
    stream (no modality views, no gates), per-part unembeds emit the part's
    joints in 6D plus a per-joint log-sigma.  Hard PART_JOINTS partition —
    no soft A.  Sigma semantics: log-sigma, the Gaussian-NLL precision for
    the constraint layer's inverse-variance fallback."""

    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8,
                 dim_feedforward=1024, dropout=0.1, n_layers=6,
                 part_joints=PART_JOINTS, sigma_init=0.0):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_parts = len(part_joints)
        self.part_joints = tuple(tuple(int(j) for j in g) for g in part_joints)
        self.register_buffer("part_place_idx",
                             _part_place_idx(self.part_joints), persistent=True)
        # One query token per part (the part embedding IS the slot identity).
        self.query = nn.Parameter(torch.zeros(1, tw, self.n_parts, dim))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_parts = nn.ModuleList(
            [nn.Linear(dim, 6 * len(joints)) for joints in self.part_joints]
        )
        self.sig_parts = nn.ModuleList(
            [nn.Linear(dim, 6 * len(joints)) for joints in self.part_joints]
        )
        for head in self.sig_parts:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(sigma_init))
        # Pose normalization stats (fitted by train.py for the V head;
        # identity for the T-prior head).
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std", torch.ones(POSE_DIM))

    def set_stats(self, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> None:
        self.pose_mean = pose_mean.to(self.pose_mean.dtype)
        self.pose_std = pose_std.to(self.pose_std.dtype)

    def forward(self, mem):
        """mem: (B, tw, dim).  Returns (pose (B,tw,POSE_DIM), log_sigma)."""
        B = mem.shape[0]
        q = self.query.expand(B, -1, -1, -1).reshape(B, self.tw * self.n_parts, self.dim)
        z = self.decoder(q, mem).reshape(B, self.tw, self.n_parts, self.dim)
        pose_parts = torch.cat(
            [self.out_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        sig_parts = torch.cat(
            [self.sig_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        sig = sig_parts[..., self.part_place_idx]
        return pose, sig


class V3ETrajHead(nn.Module):
    """Single-stream root-trajectory regression (velocity -> cumsum)."""

    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.tw = tw
        self.encoder = _temporal_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.proj = nn.Linear(dim, 3)

    def forward(self, mem):
        h = self.encoder(mem)
        v_hat = self.proj(h)
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        return v_hat, trans_hat


class V3EContactHead(nn.Module):
    """Per-foot contact logits from the tactile tokens (f6_soft supervision)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class AnySoleModelV3E(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # regress machinery in train/eval
        self.v3e = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv",
        )
        self.v_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        self.t_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        # V pose head: sigma starts confident (V dominates the fallback at
        # init); the T prior starts unconfident and is calibrated up by NLL.
        self.pose_head = V3EPoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=-3.0)
        self.prior_head = V3EPoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=3.0)
        self.traj_head = V3ETrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.prior_traj = V3ETrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.contact_head = V3EContactHead(dim=d)
        # v2 ruling (2026-10-17): the inverse-variance fallback is dormant —
        # the sigma heads stay frozen at their init values (V=-3 / prior=+3,
        # the V-dominant regime).  The NLL calibration collapsed both streams
        # to ~0 (prior weight 0.48, PA +3.6mm); a real V-failure trigger is
        # a separate design item.  The heads stay in the state dict so future
        # trigger work can reuse them.
        for head in list(self.pose_head.sig_parts) + list(self.prior_head.sig_parts):
            for param in head.parameters():
                param.requires_grad_(False)

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None):
        """config_id is accepted for the train/eval call contract and
        deliberately IGNORED — V3_3E has no condition sampling: both streams
        always live, and the three eval conditions are three readouts of this
        one forward."""
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        v_tok, t_tok = self.encoders(V_feat, T_tac, cid, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        v_mem = self.v_temporal(v_tok)
        pose_v, sigma_v = self.pose_head(v_mem)
        v_hat, trans_hat = self.traj_head(v_mem)
        t_mem = self.t_temporal(t_tok)
        pose_t, sigma_t = self.prior_head(t_mem)
        v_prior, trans_prior = self.prior_traj(t_mem)
        contact_logits = self.contact_head(t_tok)
        return {
            "x0_hat": pose_v,
            "sigma_v": sigma_v,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "x0_prior": pose_t,
            "sigma_prior": sigma_t,
            "v_prior": v_prior,
            "trans_prior": trans_prior,
            "contact_logits": contact_logits,
            "F": v_mem,          # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
