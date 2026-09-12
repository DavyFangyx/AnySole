"""Differentiable pressure-derived features for the compensated tactile path."""
from __future__ import annotations
import torch


def _foot(values):
    grid = values.reshape(*values.shape[:-1], 4, 12).clamp_min(0)
    force = grid.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    rows = torch.arange(4, device=grid.device, dtype=grid.dtype)
    cols = torch.arange(12, device=grid.device, dtype=grid.dtype)
    safe = force.clamp_min(1e-6)
    cop_x = (grid.sum(-1) * rows).sum(-1, keepdim=True) / safe / 3.0
    cop_y = 1.0 - (grid.sum(-2) * cols).sum(-1, keepdim=True) / safe / 11.0
    padded = torch.cat((force[:, :1], force, force[:, -1:]), dim=1)
    envelope = torch.maximum(torch.maximum(padded[:, :-2], padded[:, 1:-1]), padded[:, 2:])
    dx = (grid[:, :, 1:] - grid[:, :, :-1]).abs().mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    dy = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    spatial = (dx + dy) / 2.0
    temporal = torch.diff(force, dim=1, prepend=force[:, :1])
    return torch.cat((cop_x, cop_y, force, envelope, spatial, temporal), -1)


def physical_tokens(raw):
    return torch.cat((_foot(raw[..., :48]), _foot(raw[..., 48:])), -1)
