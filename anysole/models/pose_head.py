"""Pose diffusion head: predict clean 6D pose x0 from x_tau, tau, and fused memory F.

MDM/RoHM-style decoder (structural rewrite of the V1 head). The probes showed the
old head converged to a condition-only solution g(F) that ignored x_tau, because
the timestep signal (additive, after self-attention, ~18% amplitude) was too weak
to learn the noise-level gating. The rewrite fixes that:

- ``[t_emb; x_tokens]`` prepend: the diffusion-step embedding is the first token
  and is visible to every self-attention layer, so each layer can gate how much
  of x_tau it uses by noise level (MDM/RoHM do exactly this).
- Cross-attention to F in every layer (MDM decoder block), instead of one
  post-self-attention cross-attn.
- 6-8 decoder layers instead of 2+1+1.
- Pre-LayerNorm on every residual stream (self-attn / cross-attn / FFN).
- GELU everywhere (the old self-attn encoder used the PyTorch default ReLU).
- Per-dim mean/std normalization of the 6D pose representation (MDM/RoHM
  normalize the motion representation before diffusing); train.py fits the
  stats on the training set and they are saved with the checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import (
    BODY_JOINTS,
    BODY_SLICE,
    D_MODEL,
    LEFT_LEG_JOINTS,
    LEFT_LEG_SLICE,
    N_JOINTS,
    POSE_DIM,
    RIGHT_LEG_JOINTS,
    RIGHT_LEG_SLICE,
    TW,
)


class PoseHead(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024, dropout=0.1, n_layers=6):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_layers = n_layers
        self.n_body = len(BODY_JOINTS)
        self.n_left = len(LEFT_LEG_JOINTS)
        self.n_right = len(RIGHT_LEG_JOINTS)

        self.proj_body = nn.Linear(6, dim)
        self.proj_left = nn.Linear(6, dim)
        self.proj_right = nn.Linear(6, dim)
        self.group_emb = nn.Embedding(3, dim)
        nn.init.normal_(self.group_emb.weight, std=0.02)

        group_ids = torch.zeros(N_JOINTS, dtype=torch.long)
        group_ids[list(LEFT_LEG_JOINTS)] = 1
        group_ids[list(RIGHT_LEG_JOINTS)] = 2
        self.register_buffer("group_ids", group_ids, persistent=True)

        # Learned "position" for the prepended diffusion-step token.
        self.timestep_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.timestep_token, std=0.02)

        # MDM decoder block per layer: pre-LN self-attn -> pre-LN cross-attn(F)
        # -> pre-LN FFN, GELU. Full attention on [t_emb; x_tokens].
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        self.out_body = nn.Linear(dim, 6)
        self.out_left = nn.Linear(dim, 6)
        self.out_right = nn.Linear(dim, 6)

        # Per-dim mean/std of the 6D pose representation over the training set.
        # Defaults are identity so a fresh/unfitted head still runs; train.py
        # calls set_stats() before the first step and the fitted values live in
        # the checkpoint state_dict.
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM), persistent=True)
        self.register_buffer("pose_std", torch.ones(POSE_DIM), persistent=True)

    def set_stats(self, mean, std):
        """Fix the (POSE_DIM,) normalization stats used in forward."""
        mean = torch.as_tensor(mean, dtype=self.pose_mean.dtype, device=self.pose_mean.device)
        std = torch.as_tensor(std, dtype=self.pose_std.dtype, device=self.pose_std.device)
        if mean.shape != (POSE_DIM,) or std.shape != (POSE_DIM,):
            raise ValueError("pose stats must have shape (%d,)" % POSE_DIM)
        self.pose_mean.copy_(mean)
        # Floor keeps near-constant dims (8 of 138 on this dataset) from being
        # amplified by 1e5x; 1e-2 still normalizes every informative dim.
        self.pose_std.copy_(std.clamp(min=1e-2))

    def _embed(self, x_tau):
        batch, tw, _ = x_tau.shape
        body = x_tau[..., BODY_SLICE].reshape(batch, tw, self.n_body, 6)
        left = x_tau[..., LEFT_LEG_SLICE].reshape(batch, tw, self.n_left, 6)
        right = x_tau[..., RIGHT_LEG_SLICE].reshape(batch, tw, self.n_right, 6)
        tokens = torch.cat(
            [self.proj_body(body), self.proj_left(left), self.proj_right(right)],
            dim=2,
        )
        tokens = tokens + self.embeddings.time_pe.to(dtype=tokens.dtype).view(1, tw, 1, self.dim)
        tokens = tokens + self.group_emb(self.group_ids).to(dtype=tokens.dtype).view(
            1, 1, N_JOINTS, self.dim
        )
        return tokens.reshape(batch, tw * N_JOINTS, self.dim)

    def _unembed(self, h):
        batch = h.shape[0]
        tokens = h.view(batch, self.tw, N_JOINTS, self.dim)
        body = self.out_body(tokens[:, :, : self.n_body, :])
        left = self.out_left(tokens[:, :, self.n_body : self.n_body + self.n_left, :])
        right = self.out_right(tokens[:, :, self.n_body + self.n_left :, :])
        return torch.cat([body, left, right], dim=2).reshape(batch, self.tw, POSE_DIM)

    def forward(self, x_tau, tau, F):
        x_tau = (x_tau - self.pose_mean) / self.pose_std
        h = self._embed(x_tau)
        # Prepended timestep token: every decoder layer's self-attention can see
        # the noise level and gate x_tau accordingly.  The timestep embedding is
        # LayerNorm'd (embeddings.py) to sqrt(dim)=16; *0.05 keeps |t_tok| ~0.8,
        # the same scale as the pose tokens, so the t-token neither dominates
        # self-attention nor gets nulled out (tau-blind head, E1/E2 diagnosis).
        t_tok = self.embeddings.timestep(tau).unsqueeze(1) * 0.05 + self.timestep_token
        h = torch.cat([t_tok, h], dim=1)
        h = self.decoder(h, F)
        h = h[:, 1:]  # drop the timestep token before unembedding
        x0 = self._unembed(h)
        return x0 * self.pose_std + self.pose_mean
