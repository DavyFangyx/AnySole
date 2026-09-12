from __future__ import annotations

import math
import torch
import torch.nn as nn


def make_coords(rows=4, cols=12, sp_ml_mm=26.7, sp_ap_mm=22.7, device=None):
    ml = (torch.arange(rows, dtype=torch.float32) - (rows - 1) / 2) * sp_ml_mm
    ap = (torch.arange(cols, dtype=torch.float32) - (cols - 1) / 2) * sp_ap_mm
    a, b = torch.meshgrid(ml, ap, indexing="ij")
    return torch.stack((a.flatten(), b.flatten()), dim=-1).to(device)


class SE2Warp(nn.Module):
    def __init__(self, sigma_mm=22.0, **grid_kw):
        super().__init__()
        self.register_buffer("coords", make_coords(**grid_kw))
        self.inv2s2 = 1.0 / (2 * sigma_mm ** 2)

    def forward(self, pressure, theta):
        c = self.coords
        tx, ty, angle = theta.unbind(-1)
        ca, sa = torch.cos(angle), torch.sin(angle)
        wx = ca[:, None] * c[:, 0] - sa[:, None] * c[:, 1] + tx[:, None]
        wy = sa[:, None] * c[:, 0] + ca[:, None] * c[:, 1] + ty[:, None]
        slipped = torch.stack((wx, wy), -1)
        diff = c[None, :, None, :] - slipped[:, None, :, :]
        kernel = torch.exp(-diff.square().sum(-1) * self.inv2s2)
        return (kernel * pressure[:, None, :]).sum(-1) / (kernel.sum(-1) + 1e-6)


class _Foot(nn.Module):
    def __init__(self, template, n_iter=2, tx_lim_mm=15.0, ty_lim_mm=8.0, alpha_lim_deg=8.0, **warp_kw):
        super().__init__()
        self.warp = SE2Warp(**warp_kw)
        self.register_buffer("default_template", template / (template.max() + 1e-6))
        self.register_buffer("lim", torch.tensor([tx_lim_mm, ty_lim_mm, math.radians(alpha_lim_deg)]))
        self.n_iter = int(n_iter)
        self.loc = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.loc[-1].weight); nn.init.zeros_(self.loc[-1].bias)

    def forward(self, seq, template):
        b, t, n = seq.shape
        env = seq.amax(1); env = env / (env.amax(1, keepdim=True) + 1e-6)
        theta = torch.tanh(self.loc(env) / self.lim) * self.lim
        # The IC refinement is intentionally lightweight; task loss trains loc directly.
        for _ in range(self.n_iter):
            aligned = self.warp(env, theta)
            residual = aligned - template
            theta = torch.tanh((theta - 0.05 * residual.mean(-1, keepdim=True).expand(-1, 3)) / self.lim) * self.lim
        flat = seq.reshape(b * t, n)
        expanded = theta[:, None].expand(b, t, 3).reshape(b * t, 3)
        return self.warp(flat, expanded).reshape(b, t, n), theta


class InsoleDriftCompensator(nn.Module):
    """Subject-template conditioned, differentiable dual-foot drift correction."""
    def __init__(self, templates, subject_to_index, **foot_kw):
        super().__init__()
        self.subject_to_index = dict(subject_to_index)
        tensor = torch.as_tensor(templates, dtype=torch.float32)
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (2, 48):
            raise ValueError("templates must have shape (subjects, 2, 48)")
        self.register_buffer("templates", tensor)
        self.left = _Foot(self.templates[0, 0], **foot_kw)
        self.right = _Foot(self.templates[0, 1], **foot_kw)

    def forward(self, raw, subject_ids=None):
        if subject_ids is None:
            idx = torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device)
        else:
            idx = torch.tensor([self.subject_to_index.get(str(s), 0) for s in subject_ids], device=raw.device)
        l, tl = self.left(raw[..., :48], self.templates[idx, 0])
        r, tr = self.right(raw[..., 48:], self.templates[idx, 1])
        return torch.cat((l, r), -1), tl, tr
