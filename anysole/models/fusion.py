"""Fuse V and combined T tokens with a 4-layer Transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.types import D_MODEL, TW


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


class FusionTransformer(nn.Module):
    def __init__(
        self,
        dim=D_MODEL,
        tw=TW,
        n_layers=4,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.tw = tw
        self.encoder = _transformer_encoder(dim, n_layers, nhead, dim_feedforward, dropout)

    def forward(self, v_tok, t_tok):
        tokens = torch.cat([v_tok, t_tok], dim=1)
        return self.encoder(tokens)
