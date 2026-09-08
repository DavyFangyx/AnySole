"""Pose diffusion head: predict clean 6D pose x0 from x_tau, tau, and fused memory F."""

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


def _transformer_encoder(dim, n_layers, nhead, dim_feedforward, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    kwargs = {"num_layers": n_layers}
    try:
        return nn.TransformerEncoder(layer, enable_nested_tensor=False, **kwargs)
    except TypeError:
        return nn.TransformerEncoder(layer, **kwargs)


class PoseHead(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_body = len(BODY_JOINTS)
        self.n_left = len(LEFT_LEG_JOINTS)
        self.n_right = len(RIGHT_LEG_JOINTS)

        self.proj_body = nn.Linear(6, dim)
        self.proj_left = nn.Linear(6, dim)
        self.proj_right = nn.Linear(6, dim)
        self.group_emb = nn.Embedding(3, dim)

        group_ids = torch.zeros(N_JOINTS, dtype=torch.long)
        group_ids[list(LEFT_LEG_JOINTS)] = 1
        group_ids[list(RIGHT_LEG_JOINTS)] = 2
        self.register_buffer("group_ids", group_ids, persistent=True)

        self.self_attn = _transformer_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        self.out_body = nn.Linear(dim, 6)
        self.out_left = nn.Linear(dim, 6)
        self.out_right = nn.Linear(dim, 6)

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
        h = self._embed(x_tau)
        h = self.self_attn(h)
        h = h + self.embeddings.timestep(tau).unsqueeze(1)
        h = h + self.cross_attn(h, F, F, need_weights=False)[0]
        h = h + self.ffn(h)
        return self._unembed(h)
