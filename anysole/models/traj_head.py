"""Root-trajectory regression: fused memory -> per-frame velocity -> cumsum translation."""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.types import D_MODEL, FPS, TW


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


class TrajHead(nn.Module):
    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.tw = tw
        self.encoder = _transformer_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.proj = nn.Linear(dim, 3)

    def forward(self, F):
        batch = F.shape[0]
        pooled = F.reshape(batch, 2, self.tw, -1).mean(dim=1)
        h = self.encoder(pooled)
        v_hat = self.proj(h)
        # v_hat is m/s; integrate the 40 Hz samples back to meters.
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        return v_hat, trans_hat
