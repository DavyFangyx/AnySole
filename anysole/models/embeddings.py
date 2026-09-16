"""Sinusoidal PEs, modality embeddings, null tokens, and diffusion-step MLP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from anysole.types import D_MODEL, TW


def sinusoidal_pe(length, dim, device=None, dtype=None):
    """pe[t, 2i]=sin(t/10000^{2i/d}), pe[t, 2i+1]=cos(...). Output (length, dim)."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: dim // 2])
    if dtype is not None:
        pe = pe.to(dtype=dtype)
    return pe


class TimestepEmbedding(nn.Module):
    """Sinusoidal encoding of scalar diffusion step tau in [0, 999], then MLP.

    The trailing LayerNorm bounds the embedding magnitude: without it the MLP
    output grew unboundedly during training (|emb| 0.11 at init -> ~37-40,
    ~50x the pose-token scale), and the pose head learned to null the t-token
    in attention, making the head tau-blind and killing the noise-level gating
    (E1/E2 diagnosis 2026-09: see pose_head.forward for the matching 0.05
    scale that keeps |t_tok| ~ |x_tokens|).
    """

    def __init__(self, dim=D_MODEL, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )

    def _sinusoidal(self, tau):
        half = self.dim // 2
        if half == 0:
            return tau.new_zeros((tau.numel(), self.dim), dtype=torch.float32)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, device=tau.device, dtype=torch.float32)
            / half
        )
        args = tau.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

    def forward(self, tau):
        return self.mlp(self._sinusoidal(tau))


class SharedEmbeddings(nn.Module):
    """Time PE, 2-way modality embedding, and one null token per modality."""

    def __init__(self, dim=D_MODEL, tw=TW):
        super().__init__()
        self.dim = dim
        self.tw = tw
        self.modality = nn.Embedding(2, dim)
        self.null_v = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_t = nn.Parameter(torch.zeros(1, 1, dim))
        self.timestep = TimestepEmbedding(dim)
        self.register_buffer("time_pe", sinusoidal_pe(tw, dim), persistent=True)
        nn.init.normal_(self.null_v, std=0.02)
        nn.init.normal_(self.null_t, std=0.02)
        nn.init.normal_(self.modality.weight, std=0.02)
