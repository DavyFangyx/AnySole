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
    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024, dropout=0.1,
                 out_dim=3):
        super().__init__()
        self.tw = tw
        self.out_dim = int(out_dim)
        self.encoder = _transformer_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.proj = nn.Linear(dim, self.out_dim)

    def forward(self, F, frame_valid=None, timestamps=None):
        batch = F.shape[0]
        pooled = F.reshape(batch, 2, self.tw, -1).mean(dim=1)
        if frame_valid is None:
            h = self.encoder(pooled)
        else:
            padding = ~frame_valid
            pooled = torch.where(frame_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
            h = self.encoder(pooled, src_key_padding_mask=padding if bool(padding.any()) else None)
        v_hat = self.proj(h)
        if frame_valid is not None:
            v_hat = torch.where(frame_valid.unsqueeze(-1), v_hat, torch.zeros_like(v_hat))
        if timestamps is not None:
            if frame_valid is None:
                raise ValueError("timestamp integration requires frame_valid")
            # v_i describes the interval ending at i. A missing frame or a
            # target gap starts another segment whose first position is zero.
            dt = timestamps[:, 1:].to(torch.float64) - timestamps[:, :-1].to(torch.float64)
            edges = frame_valid[:, 1:] & frame_valid[:, :-1] & (dt > 0) & (dt <= 0.05 + 1e-9)
            dt = dt.to(dtype=v_hat.dtype)
            position = torch.zeros_like(v_hat[:, 0, :3])
            positions = [position]
            for i in range(1, self.tw):
                position = torch.where(
                    edges[:, i - 1, None],
                    position + v_hat[:, i, :3] * dt[:, i - 1, None],
                    torch.zeros_like(position),
                )
                positions.append(position)
            return v_hat, torch.stack(positions, dim=1)
        # F2a (out_dim=4): dims [psi_dot, v_hx, v_hz, h] — only the first
        # three integrate (heading + heading-frame position); h is the direct
        # world height.  V1 (out_dim=3): v_hat is m/s; integrate the 40 Hz
        # samples back to meters.
        trans_hat = torch.cumsum(v_hat[..., :3], dim=1) / float(FPS)
        return v_hat, trans_hat
