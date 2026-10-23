"""Single-view A2 pressure readouts; no labels or cross-camera memory inputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from anysole.models.common.embeddings import relative_time_pe


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def split_feet(pressure):
    if pressure.shape[-1] != 96:
        raise ValueError("pressure uses left48 followed by right48")
    return pressure.reshape(*pressure.shape[:-1], 2, 48)


def join_feet(pressure):
    if pressure.shape[-2:] != (2, 48):
        raise ValueError("foot pressure must end in [2,48]")
    return pressure.reshape(*pressure.shape[:-2], 96)


class CapacityPressureHead(nn.Module):
    """All capacity parameters participate in the tactile-memory computation."""
    def __init__(self, dim=256, width=544):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.residual = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, dim))
        self.output = nn.Linear(dim, 96)

    def forward(self, fused, frame_valid, timestamps):
        t = frame_valid.shape[1]
        memory = fused[:, t:2*t]
        h = memory + self.residual(self.norm(memory))
        pressure = self.output(h)
        return {"pressure_hat": torch.where(frame_valid[..., None], pressure, torch.zeros_like(pressure))}


class QueryPressureHead(nn.Module):
    """Foot/time queries read one explicitly selected single-view memory."""
    def __init__(self, kind="query_direct", dim=256, nhead=8, hidden=128,
                 dropout=.1, cell_mean=None, contact_rate=None,
                 memory="fusion_visual_slice"):
        super().__init__()
        if kind not in ("query_direct", "query_nonnegative", "query_factorized"):
            raise ValueError("unsupported query output representation")
        self.kind = kind
        if memory not in ("fusion_visual_slice", "fusion_tactile_slice", "fusion_all"):
            raise ValueError("unsupported query memory")
        self.memory = memory
        self.foot_embedding = nn.Parameter(torch.empty(2, dim))
        nn.init.normal_(self.foot_embedding, std=.02)
        self.attention = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU())
        self.output = nn.Linear(hidden, 48, bias=kind == "query_direct")
        if kind != "query_direct":
            mu = torch.as_tensor(cell_mean, dtype=torch.float32).reshape(2, 48).clamp_min(1e-4)
            rate = torch.as_tensor(contact_rate, dtype=torch.float32).reshape(2)
            if not torch.isfinite(mu).all() or not ((rate > 0) & (rate < 1)).all():
                raise ValueError("query pressure initialization requires finite train means/contact rates")
            self.contact = nn.Linear(hidden, 1, bias=False)
            self.contact_bias = nn.Parameter(torch.logit(rate))
            self.pressure_bias = nn.Parameter(inverse_softplus(mu) if kind == "query_nonnegative"
                                              else (mu/mu.sum(-1, keepdim=True)).log())
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.contact.weight)
            if kind == "query_factorized":
                self.total = nn.Linear(hidden, 1, bias=False)
                self.total_bias = nn.Parameter(inverse_softplus(mu.sum(-1)))
                nn.init.zeros_(self.total.weight)

    def forward(self, fused, frame_valid, timestamps):
        n, t = frame_valid.shape
        if fused.shape[:2] != (n, 2*t):
            raise ValueError("query expects one view's [visual T, tactile T] fusion memory")
        # Do not call attention for all-missing rows: an all-True mask is NaN.
        rows = frame_valid.any(-1)
        h = fused.new_zeros(n, t, 2, self.foot_embedding.shape[-1])
        if bool(rows.any()):
            time = relative_time_pe(timestamps[rows], h.shape[-1], 40., fused.dtype)
            query = time[:, :, None, :] + self.foot_embedding[None, None]
            query = query.reshape(-1, 2*t, h.shape[-1])
            mask = frame_valid[rows]
            memory = fused[rows]
            if self.memory == "fusion_all":
                mask = torch.cat([mask, mask], dim=1)
            else:
                memory = memory[:, :t] if self.memory == "fusion_visual_slice" else memory[:, t:]
            memory = torch.where(mask[..., None], memory, torch.zeros_like(memory))
            attended, _ = self.attention(query, memory, memory,
                                          key_padding_mask=~mask, need_weights=False)
            values = self.norm(query + attended).reshape(-1, t, 2, h.shape[-1])
            h[rows] = values
        z = self.hidden(h)
        output = self.output(z)
        valid = frame_valid[:, :, None, None]
        result = {"pressure_query_valid": frame_valid[:, :, None].expand(-1, -1, 2)}
        if self.kind == "query_direct":
            pressure = output
        else:
            contact = self.contact(z).squeeze(-1) + self.contact_bias
            result["pressure_contact_logits"] = torch.where(frame_valid[:, :, None], contact, torch.zeros_like(contact))
            if self.kind == "query_nonnegative":
                pressure = F.softplus(output + self.pressure_bias)
                result["pressure_total"] = torch.where(frame_valid[:, :, None], pressure.sum(-1), torch.zeros_like(contact))
            else:
                allocation = F.softmax(output + self.pressure_bias, dim=-1)
                total = F.softplus(self.total(z).squeeze(-1) + self.total_bias)
                pressure = total[..., None] * allocation
                result["pressure_allocation"] = torch.where(valid, allocation, torch.zeros_like(allocation))
                result["pressure_total"] = torch.where(frame_valid[:, :, None], total, torch.zeros_like(total))
        pressure = torch.where(valid, pressure, torch.zeros_like(pressure))
        result["pressure_hat"] = join_feet(pressure)
        return result


def build_pressure_head(config):
    # Fork CPU RNG only: modules are built on CPU before .to(device). No CUDA
    # initialization and no construction-order drift in common model weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["a2_head_seed"]))
        if config["pressure_head"] == "capacity_time_pe":
            return CapacityPressureHead(width=int(config["capacity_width"]))
        return QueryPressureHead(config["pressure_head"], dropout=float(config.get("dropout", .1)),
                                 cell_mean=config.get("pressure_cell_mean"),
                                 contact_rate=config.get("pressure_contact_rate"),
                                 memory=config.get("query_memory", "fusion_visual_slice"))
