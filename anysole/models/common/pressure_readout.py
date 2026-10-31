"""Query pressure readout heads (ported from feat/view-consistency, 2026-10-24).

V3_Bv2/Bv3 pressure head: two learnable foot queries plus a per-frame time
query cross-attend over the fusion memory and replace the linear aux t_rec
(same λ_trec supervision position).  Three output representations of the
same mechanism, selected by ``pressure_head``:

  query_direct      Linear(dim→48, bias) per foot (unconstrained baseline)
  query_nonnegative softplus(output + pressure_bias); pressure_bias starts
                    at inverse_softplus of the frozen train cell means, so
                    the zero-initialized head reads out the training prior
  query_factorized  total×allocation: allocation = softmax(output +
                    log-normalized cell-mean bias) over 48 cells, total =
                    softplus(total head + inverse_softplus(mean foot sum));
                    zero-init start == the training prior exactly

No contact head — B keeps the existing contact components (Hv1/E v1).
``memory`` selects the fusion slice: "fusion_tactile_slice" (T half,
default — same position as t_rec), "fusion_visual_slice" (V half) or
"fusion_all" (all 40 slots).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from anysole.models.common.embeddings import relative_time_pe

QUERY_KINDS = ("query_direct", "query_nonnegative", "query_factorized")
QUERY_MEMORY = ("fusion_visual_slice", "fusion_tactile_slice", "fusion_all", "motion_z")


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


class QueryPressureHead(nn.Module):
    """Foot/time queries read a fusion slice or explicit time/part motion z."""

    def __init__(self, kind="query_direct", dim=256, nhead=8, hidden=128,
                 dropout=.1, cell_mean=None, memory="fusion_tactile_slice"):
        super().__init__()
        if kind not in QUERY_KINDS:
            raise ValueError("unsupported query output representation")
        self.kind = kind
        if memory not in QUERY_MEMORY:
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
            if not torch.isfinite(mu).all():
                raise ValueError("query pressure initialization requires finite cell means")
            self.pressure_bias = nn.Parameter(inverse_softplus(mu) if kind == "query_nonnegative"
                                              else (mu / mu.sum(-1, keepdim=True)).log())
            nn.init.zeros_(self.output.weight)
            if kind == "query_factorized":
                self.total = nn.Linear(hidden, 1, bias=False)
                self.total_bias = nn.Parameter(inverse_softplus(mu.sum(-1)))
                nn.init.zeros_(self.total.weight)

    def forward(self, fused, frame_valid=None, timestamps=None, memory_valid=None):
        """``fused`` is the [visual T, tactile T] memory (n, 2t, dim).

        B windows are fully valid at fixed 40 Hz, so both optional inputs
        default to that regime; the masked path ports unchanged for callers
        that carry per-frame validity.
        ``motion_z`` instead accepts [N,T_memory,S,D], time-major motion
        slots, not a [V20,T20] fusion sequence. Query time is explicitly
        timestamps/frame_valid (defaults to T_memory at 40Hz). Memory time
        may differ from query time only with explicit memory_valid[N,Tm]
        or memory_valid[N,Tm*S]. A frame validity bit covers all S slots.
        """
        motion = self.memory == "motion_z"
        if motion:
            if fused.ndim != 4:
                raise ValueError("motion_z memory requires [N,T_memory,S,D]")
            n, tm, slots, dim = fused.shape
            t = (timestamps.shape[-1] if timestamps is not None else
                 frame_valid.shape[-1] if frame_valid is not None else tm)
            fused = fused.reshape(n, tm * slots, dim)
        else:
            if fused.ndim != 3 or fused.shape[1] % 2:
                raise ValueError("fusion memory requires [N,2*T,D]")
            n, t2, dim = fused.shape
            t = t2 // 2
        if frame_valid is None:
            frame_valid = fused.new_ones(n, t, dtype=torch.bool)
        if tuple(frame_valid.shape) != (n, t):
            raise ValueError("query frame_valid must be [N,T_query]")
        frame_valid = frame_valid.to(device=fused.device, dtype=torch.bool)
        if timestamps is None:
            timestamps = torch.arange(t, dtype=fused.dtype, device=fused.device) / 40.0
            timestamps = timestamps[None].expand(n, t)
        if tuple(timestamps.shape) != (n, t):
            raise ValueError("query timestamps must be [N,T_query]")
        timestamps = timestamps.to(device=fused.device)
        if motion:
            if memory_valid is None:
                if tm != t:
                    raise ValueError("different query/memory times require explicit memory_valid")
                memory_valid = frame_valid
            memory_valid = memory_valid.to(device=fused.device, dtype=torch.bool)
            if tuple(memory_valid.shape) == (n, tm):
                memory_valid = memory_valid.repeat_interleave(slots, dim=-1)
            elif tuple(memory_valid.shape) != (n, tm * slots):
                raise ValueError("motion memory_valid must be [N,T_memory] or [N,T_memory*S]")
        # Do not call attention for all-missing rows: an all-True mask is NaN.
        rows = frame_valid.any(-1)
        if motion:
            rows = rows & memory_valid.any(-1)
        h = fused.new_zeros(n, t, 2, self.foot_embedding.shape[-1])
        if bool(rows.any()):
            time = relative_time_pe(timestamps[rows], h.shape[-1], 40., fused.dtype)
            query = time[:, :, None, :] + self.foot_embedding[None, None]
            query = query.reshape(-1, 2 * t, h.shape[-1])
            mask = memory_valid[rows] if motion else frame_valid[rows]
            memory = fused[rows]
            if motion:
                pass
            elif self.memory == "fusion_all":
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
        valid = (frame_valid & rows[:, None])[:, :, None, None]
        if self.kind == "query_direct":
            pressure = output
        elif self.kind == "query_nonnegative":
            pressure = F.softplus(output + self.pressure_bias)
        else:
            allocation = F.softmax(output + self.pressure_bias, dim=-1)
            total = F.softplus(self.total(z).squeeze(-1) + self.total_bias)
            pressure = total[..., None] * allocation
        pressure = torch.where(valid, pressure, torch.zeros_like(pressure))
        return {"pressure_hat": join_feet(pressure)}


def build_query_pressure_head(config):
    """Build the head from the effective config (pressure_head / query_memory /
    pressure_cell_mean).  Module construction is CPU-seeded from the run seed
    so common model weights keep construction-order determinism."""
    kind = str(config.get("pressure_head", "query_direct"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.get("seed", 1)))
        return QueryPressureHead(kind, dropout=float(config.get("dropout", .1)),
                                 cell_mean=config.get("pressure_cell_mean"),
                                 memory=config.get("query_memory", "fusion_tactile_slice"))
