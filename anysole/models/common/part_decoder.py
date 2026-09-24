"""F5: part-query decoder with per-part V/T/empty gating (fix_plan_v2.md §F5).

**ARCHIVED (2026-09-19): v2 §F5 作废，fix_plan_v3.md 取代。本模块无任何调用方，
仅留档。** 四 run 失败根因见 memory/f5-root-cause-tstream-scale-bug.md
（encode_stream 缺 LayerNorm → T token 量级 1400×V → 交叉注意力 one-hot 退化）。
V3 的门控方案（§V3-3 σ 软门控）作用在 fused F 的模态掩码视图上，不复用本模块。

Replaces the fusion transformer + pose head + traj head of AnySole V2 with a
9-part query decoder.  Each frame, each part:

- attends the other parts (motion-tree hop bias, one learnable scalar per
  attention head per hop distance 0-4),
- attends its own history (temporal self-attention within the part),
- cross-attends a LOCAL time window (t-4..t+4) of the V stream and the
  T stream (foot_conv's [left/right/global] tokens) with relative time
  embeddings,
- gates the two cross-attention outputs: g = softmax(l_V, l_T, l_empty),
  l_m = MLP_m(Z) + b_{p,m}; b init: root/legs/feet favor T (+1), torso/
  head/arms favor V (+1); missing modalities get l = -inf (config-level
  and segment-level dropout),
- FFN.

Outputs: per-part 6D rotations (root additionally outputs the trajectory:
3-dim world velocity, or the F2 4-dim heading trajectory per f2_repr);
each foot part outputs a contact logit (unsupervised until F6).  Aux
reconstruction heads (T_rec/V_rec) are dropped in this structure.

``gate_mode="nogate"`` (f5_nogate ablation) drops the gating: a single
cross-attention reads the concatenated local V+T window tokens instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import D_MODEL, FPS, JOINT_PARENTS, N_PARTS, PART_JOINTS, PART_NAMES, TW

LOCAL_WINDOW = 4  # t-4..t+4 cross-attention window
REL_POSITIONS = 2 * LOCAL_WINDOW + 1


def part_hop_matrix() -> torch.Tensor:
    """(9, 9) hop distances between parts on the JOINT_PARENTS tree
    (Floyd-Warshall on the joint-level graph, min over joint pairs)."""
    parents = [int(p) for p in JOINT_PARENTS]
    n = len(parents)
    inf = 99
    dist = [[inf] * n for _ in range(n)]
    for j in range(n):
        dist[j][j] = 0
    for j, par in enumerate(parents):
        if par >= 0:
            dist[j][par] = dist[par][j] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    hops = torch.full((N_PARTS, N_PARTS), inf, dtype=torch.long)
    for a in range(N_PARTS):
        for ja in PART_JOINTS[a]:
            for b in range(N_PARTS):
                for jb in PART_JOINTS[b]:
                    hops[a, b] = min(hops[a, b], dist[ja][jb])
    return hops


class _Layer(nn.Module):
    def __init__(self, dim, nhead, dim_ff, dropout, gate_mode):
        super().__init__()
        self.dim = dim
        self.gate_mode = gate_mode
        self.part_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.cross_v = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.cross_t = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        if gate_mode == "nogate":
            self.cross_join = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        else:
            self.gate_v = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            self.gate_t = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            # b_{p,m} init: root(0)/legs(5,6)/feet(7,8) favor T; torso/head/
            # arms favor V; empty stays 0.
            self.gate_bias = nn.Parameter(torch.zeros(N_PARTS, 3))
            with torch.no_grad():
                for p in (0, 5, 6, 7, 8):
                    self.gate_bias[p, 1] = 1.0
                for p in (1, 2, 3, 4):
                    self.gate_bias[p, 0] = 1.0
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff), nn.GELU(), nn.Linear(dim_ff, dim), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, hop_bias, e_v, e_t, frame_mask_v, frame_mask_t):
        batch, tw, n_parts, dim = z.shape
        # 1) part self-attention (sequence = parts, batch = B*tw).
        z_seq = z.reshape(batch * tw, n_parts, dim)
        nhead = hop_bias.shape[0]
        attn_mask = hop_bias.unsqueeze(0).expand(batch * tw, -1, -1, -1).reshape(
            batch * tw * nhead, n_parts, n_parts)
        h, _ = self.part_attn(
            z_seq, z_seq, z_seq, attn_mask=attn_mask, need_weights=False,
        )
        z = z + self.dropout(h.reshape(batch, tw, n_parts, dim))
        z = self.norm1(z)
        # 2) temporal self-attention per part (sequence = time, batch = B*9).
        z_time = z.permute(0, 2, 1, 3).reshape(batch * n_parts, tw, dim)
        h, _ = self.temporal_attn(z_time, z_time, z_time, need_weights=False)
        z = z + self.dropout(h.reshape(batch, n_parts, tw, dim).permute(0, 2, 1, 3))
        z = self.norm2(z)
        # 3) local cross-attention to the V/T window tokens.  The window
        # tokens are SHARED across the 9 part queries: batch = B*tw frames,
        # 9 queries each — repeating the k/v per part would blow the
        # key/value tensors up 9x (the first F5 OOM at batch 256).
        z_q = z.reshape(batch * tw, n_parts, dim)
        e_v_flat = e_v.reshape(batch * tw, e_v.shape[2], dim)
        e_t_flat = e_t.reshape(batch * tw, e_t.shape[2], dim)
        attn_v, _ = self.cross_v(z_q, e_v_flat, e_v_flat, need_weights=False)
        attn_t, _ = self.cross_t(z_q, e_t_flat, e_t_flat, need_weights=False)
        attn_v = attn_v.reshape(batch, tw, n_parts, dim)
        attn_t = attn_t.reshape(batch, tw, n_parts, dim)
        gates = None
        if self.gate_mode == "nogate":
            joined = torch.cat([e_v_flat, e_t_flat], dim=1)
            h, _ = self.cross_join(z_q, joined, joined, need_weights=False)
            z = z + self.dropout(h.reshape(batch, tw, n_parts, dim))
        else:
            l_v = self.gate_v(z).squeeze(-1) + self.gate_bias[:, 0]
            l_t = self.gate_t(z).squeeze(-1) + self.gate_bias[:, 1]
            l_empty = torch.zeros_like(l_v) + self.gate_bias[:, 2]
            frame_v = frame_mask_v.view(batch, tw, 1).expand(-1, -1, n_parts)
            frame_t = frame_mask_t.view(batch, tw, 1).expand(-1, -1, n_parts)
            l_v = torch.where(frame_v, l_v, torch.full_like(l_v, float("-inf")))
            l_t = torch.where(frame_t, l_t, torch.full_like(l_t, float("-inf")))
            gates = torch.softmax(torch.stack([l_v, l_t, l_empty], dim=-1), dim=-1)
            z = z + gates[..., 0:1] * attn_v + gates[..., 1:2] * attn_t
        z = self.norm3(z)
        # 4) FFN.
        z = z + self.dropout(self.ffn(z))
        z = self.norm4(z)
        return z, gates


class PartDecoder(nn.Module):
    def __init__(self, embeddings=None, dim=D_MODEL, tw=TW, nhead=8, n_layers=6,
                 dim_ff=1024, dropout=0.1, traj_dim=3, gate_mode="gated"):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        if gate_mode not in ("gated", "nogate"):
            raise ValueError("gate_mode must be 'gated' or 'nogate', got %r" % gate_mode)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.traj_dim = int(traj_dim)
        self.gate_mode = gate_mode
        self.part_emb = nn.Parameter(torch.zeros(N_PARTS, dim))
        nn.init.normal_(self.part_emb, std=0.02)
        # Motion-tree hop bias: one learnable scalar per head per hop 0-4.
        self.hop_bias_weight = nn.Parameter(torch.zeros(nhead, 5))
        self.register_buffer("hop_matrix", part_hop_matrix(), persistent=True)
        self.rel_emb = nn.Embedding(REL_POSITIONS, dim)
        nn.init.normal_(self.rel_emb.weight, std=0.02)
        self.layers = nn.ModuleList(
            [_Layer(dim, nhead, dim_ff, dropout, gate_mode) for _ in range(n_layers)]
        )
        # Output heads: root -> 6 + traj; one per remaining part.
        self.root_head = nn.Linear(dim, 6 + self.traj_dim)
        self.part_heads = nn.ModuleList(
            [nn.Linear(dim, len(joints) * 6) for joints in PART_JOINTS[1:]]
        )
        self.contact_l = nn.Linear(dim, 2)
        self.contact_r = nn.Linear(dim, 2)

    def _hop_bias(self, device) -> torch.Tensor:
        idx = torch.minimum(self.hop_matrix.to(device), torch.full_like(self.hop_matrix, 4))
        return torch.stack([
            torch.gather(self.hop_bias_weight[h], 0, idx.reshape(-1)).reshape(N_PARTS, N_PARTS)
            for h in range(self.hop_bias_weight.shape[0])
        ])  # (nhead, 9, 9) added to attention scores

    def _local_windows(self, e, t_dim):
        """(B, tw, t, d) -> (B, tw, (2w+1)*t, d): per-frame local windows with
        relative-time embeddings on the keys/values; edges replicate."""
        batch, tw, t, dim = e.shape
        e_pad = torch.cat(
            [e[:, :1].expand(-1, LOCAL_WINDOW, -1, -1), e,
             e[:, -1:].expand(-1, LOCAL_WINDOW, -1, -1)], dim=1
        )
        windows = torch.stack(
            [e_pad[:, t0:t0 + REL_POSITIONS].reshape(batch, REL_POSITIONS * t, dim)
             for t0 in range(tw)], dim=1
        )
        rel_emb = self.rel_emb(torch.arange(REL_POSITIONS, device=e.device)).repeat_interleave(t, dim=0)
        return windows + rel_emb.view(1, 1, -1, dim)

    def forward(self, E_V, E_T, config_id, frame_mask_v=None, frame_mask_t=None):
        """E_V (B, tw, d); E_T (B, tw, d) or (B, tw, 3, d) foot tokens.

        Returns the standard model output dict plus per-part gates and a
        part-mean memory F (ridge-probe compatible)."""
        batch, tw = E_V.shape[0], E_V.shape[1]
        if E_T.ndim == 3:
            E_T = E_T.unsqueeze(2)
        if frame_mask_v is None:
            frame_mask_v = torch.ones(batch, tw, device=E_V.device, dtype=torch.bool)
        else:
            frame_mask_v = frame_mask_v.bool().to(E_V.device)
            if frame_mask_v.ndim == 2 and frame_mask_v.shape[1] == 1:
                frame_mask_v = frame_mask_v.expand(batch, tw)  # per-row -> per-frame
        if frame_mask_t is None:
            frame_mask_t = torch.ones(batch, tw, device=E_V.device, dtype=torch.bool)
        else:
            frame_mask_t = frame_mask_t.bool().to(E_V.device)
            if frame_mask_t.ndim == 2 and frame_mask_t.shape[1] == 1:
                frame_mask_t = frame_mask_t.expand(batch, tw)
        e_v = self._local_windows(E_V.unsqueeze(2), 1)
        e_t = self._local_windows(E_T, E_T.shape[2])
        # Missing-modality frames: zero the stream contents (the gates also
        # see -inf from the masks).
        e_v = e_v * frame_mask_v.view(batch, tw, 1, 1)
        e_t = e_t * frame_mask_t.view(batch, tw, 1, 1)

        z = self.part_emb.view(1, 1, N_PARTS, self.dim).expand(batch, tw, -1, -1)
        z = z + self.embeddings.time_pe.to(dtype=z.dtype).view(1, tw, 1, self.dim)
        hop_bias = self._hop_bias(E_V.device)
        gate_list = []
        for layer in self.layers:
            z, gates = layer(z, hop_bias, e_v, e_t, frame_mask_v, frame_mask_t)
            if gates is not None:
                gate_list.append(gates)
        gates = torch.stack(gate_list).mean(dim=0) if gate_list else None  # (B, tw, 9, 3)

        x0_hat = torch.cat(
            [self.root_head(z[:, :, 0])[..., :6]]
            + [head(z[:, :, i]) for i, head in enumerate(self.part_heads, start=1)],
            dim=-1,
        )
        traj = self.root_head(z[:, :, 0])[..., 6:]
        trans_hat = torch.cumsum(traj[..., :3], dim=1) / float(FPS)
        contact_logits = torch.stack(
            [self.contact_l(z[:, :, 7]), self.contact_r(z[:, :, 8])], dim=-1
        )  # (B, tw, 2, 2)
        return {
            "x0_hat": x0_hat,
            "v_hat": traj,
            "trans_hat": trans_hat,
            "contact_logits": contact_logits,
            "F": z.mean(dim=2),
            "gates": gates,
        }
