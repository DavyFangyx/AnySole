"""F4a: per-foot spatial-grid tactile encoder on the EXISTING tactile channel.

The tactile input data is unchanged (fix_plan_v2.md F4a items 1-2 — native
rate resampling and per-session normalization — are deliberately NOT applied:
the 40 Hz T_raw/T_phys pipeline is already fully processed).  What F4a adds
is the per-foot structured ENCODING (plan items 3-6):

Input: ``(B, tw, 108)`` — cat([T_raw 96, T_phys 12]), the same tensor the
raw108 path consumes.

Per foot (left = cells 0:48, right = 48:96 of T_raw; T_phys 6 dims per foot):
  - grid: the foot's 48 cells reshaped to 4×12 (rows = medial-lateral,
    cols = heel-toe, pressure.py orientation);
  - handcrafted: T_phys per-foot 6 (CoP 2 + total force + envelope + spatial
    + temporal) plus contact-area ratio and dF/dt computed in-encoder from
    the grid cells (plan item 3; q_T saturated/dead ratios need per-session
    statistics and are deferred — they would require touching the data side).

Both feet share one conv stack; the right foot is mirrored along the
medial-lateral axis (grid rows) so the shared weights see the same canonical
layout.  Per frame the encoder emits three tokens ``[left, right, global]``;
a 4-layer temporal transformer mixes them across the window (frame time-PE +
token-type embedding), and a final linear merge collapses the three tokens
back to one ``(B, tw, d)`` stream — the same shape the fusion transformer
expects from the raw108 path, so everything downstream is untouched
(plan item 6).  The T reconstruction target stays the existing normalized
96-dim T_raw at 40 Hz (plan item 7 = current L_Trec, aux_heads unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import D_MODEL, T_PHYS_DIM, T_RAW_DIM, TW

GRID_ROWS, GRID_COLS = 4, 12
GRID_CELLS = GRID_ROWS * GRID_COLS  # 48 per foot
T_PHYS_PER_FOOT = T_PHYS_DIM // 2  # 6
FEAT_PER_FOOT = T_PHYS_PER_FOOT + 2  # + contact-area ratio, dF/dt
CONTACT_AREA_THRESH = 0.05  # normalized-cell threshold for "in contact"
FPS = 40.0  # dF/dt per second


class FootConvEncoder(nn.Module):
    def __init__(self, embeddings=None, dim=D_MODEL, tw=TW, nhead=8,
                 temporal_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        # Shared spatial conv: 4×12 grid -> 4×6×64, flatten -> 128.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),  # toe axis /2
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        conv_out = 4 * 6 * 64  # 1536
        self.conv_proj = nn.Linear(conv_out, 128)
        self.feat_mlp = nn.Sequential(
            nn.Linear(FEAT_PER_FOOT, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.foot_proj = nn.Linear(256, dim)
        self.foot_emb = nn.Embedding(2, dim)  # left / right identity
        nn.init.normal_(self.foot_emb.weight, std=0.02)
        # 3 token types per frame: [left, right, global].
        self.token_emb = nn.Embedding(3, dim)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        self.global_merge = nn.Linear(2 * dim, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.out_merge = nn.Linear(3 * dim, dim)
        # Match LinearTemporalEncoder's contract: normalized stream + time PE
        # + modality embedding (the fusion input shape stays (B, tw, dim)).
        self.norm = nn.LayerNorm(dim)
        # encode_stream output norm (fix_plan_v3 §3.1 hygiene fix): the F5
        # root cause was this stream lacking LayerNorm (|t_tok| ~1400x the
        # V tokens -> one-hot cross attention).  V3 does not call
        # encode_stream; the norm is so future misuse fails loudly, not
        # silently.  NOTE: adding these keys is tolerated by the quasi-strict
        # loaders (eval.py / infer.py _MISSING_OK) for pre-V3 checkpoints.
        self.stream_norm = nn.LayerNorm(dim)

    def _per_foot_inputs(self, x: torch.Tensor):
        """(B, tw, 108) -> grids (B, tw, 2, 4, 12), feats (B, tw, 2, FEAT_PER_FOOT)."""
        if x.shape[-1] != T_RAW_DIM + T_PHYS_DIM:
            raise ValueError(
                "FootConvEncoder expects the raw108 channel (%d-dim), got %d"
                % (T_RAW_DIM + T_PHYS_DIM, x.shape[-1])
            )
        batch, tw = x.shape[0], x.shape[1]
        t_raw = x[..., :T_RAW_DIM]
        t_phys = x[..., T_RAW_DIM:]
        grids = torch.stack(
            [t_raw[..., :GRID_CELLS], t_raw[..., GRID_CELLS:]],
            dim=2,
        ).reshape(batch, tw, 2, GRID_ROWS, GRID_COLS)
        # In-encoder handcrafted features: contact-area ratio and dF/dt from
        # the grid cells (plan item 3), on top of the T_phys per-foot six.
        area = (grids > CONTACT_AREA_THRESH).float().mean(dim=(-1, -2))  # (B,tw,2)
        force = grids.sum(dim=(-1, -2))  # (B,tw,2)
        dfdt = torch.diff(force, dim=1, prepend=force[:, :1]) * FPS  # (B,tw,2)
        phys = torch.stack(
            [t_phys[..., :T_PHYS_PER_FOOT], t_phys[..., T_PHYS_PER_FOOT:]], dim=2
        )
        feats = torch.cat([phys, area.unsqueeze(-1), dfdt.unsqueeze(-1)], dim=-1)
        return grids, feats

    def _per_foot_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, tw, 108) -> (B, tw, 2, dim): shared conv per foot, right mirrored."""
        batch, tw = x.shape[0], x.shape[1]
        grids, feats = self._per_foot_inputs(x)
        # Right foot (index 1) mirrored along the medial-lateral rows so the
        # shared conv sees the canonical orientation.
        mirrored = torch.flip(grids, dims=[-2])
        tokens = torch.empty(batch, tw, 2, self.dim, device=x.device, dtype=x.dtype)
        for foot in range(2):
            g = (mirrored if foot == 1 else grids)[:, :, foot]  # (B, tw, 4, 12)
            h = self.conv(g.reshape(batch * tw, 1, GRID_ROWS, GRID_COLS))
            h = self.conv_proj(h.reshape(batch * tw, -1))
            f = self.feat_mlp(feats[:, :, foot].reshape(batch * tw, -1))
            h = self.foot_proj(torch.cat([h, f], dim=-1)).reshape(batch, tw, self.dim)
            tokens[:, :, foot] = h
        tokens = tokens + self.foot_emb.weight.to(dtype=tokens.dtype).view(1, 1, 2, self.dim)
        return tokens

    def _stream_raw(self, x: torch.Tensor) -> torch.Tensor:
        """(B, tw, 108) -> (B, tw, 3, dim) RAW token stream after the temporal
        transformer (no output LayerNorm).  ``forward()`` consumes this so the
        F4a/F2p4 forward path is byte-identical to the pre-fix computation."""
        batch, tw = x.shape[0], x.shape[1]
        tokens = self._per_foot_tokens(x)  # (B, tw, 2, dim)
        global_tok = self.global_merge(tokens.reshape(batch, tw, 2 * self.dim))
        seq = torch.cat([tokens, global_tok.unsqueeze(2)], dim=2)  # (B, tw, 3, dim)
        seq = seq.reshape(batch, tw * 3, self.dim)
        # Frame time-PE repeats per frame across its 3 tokens; token-type
        # embedding disambiguates left/right/global.
        pe = self.embeddings.time_pe.to(dtype=seq.dtype).unsqueeze(1)  # (tw,1,dim)
        pe = pe.expand(tw, 3, self.dim).reshape(1, tw * 3, self.dim)
        type_emb = self.token_emb.weight.to(dtype=seq.dtype).repeat(tw, 1).unsqueeze(0)
        seq = self.temporal(seq + pe + type_emb)
        return seq.reshape(batch, tw, 3, self.dim)

    def encode_stream(self, x: torch.Tensor) -> torch.Tensor:
        """(B, tw, 108) -> (B, tw, 3, dim) token stream after the temporal
        transformer (left/right/global).  ARCHIVED alongside part_decoder
        (v2 §F5 作废, fix_plan_v3); the LayerNorm here is the hygiene fix —
        the un-normalized stream was the F5 failure root cause.  ``forward()``
        does NOT route through this normalization (it uses _stream_raw), so
        the F4a/F2p4 forward values are unchanged by the fix."""
        return self.stream_norm(self._stream_raw(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != T_RAW_DIM + T_PHYS_DIM:
            raise ValueError(
                "FootConvEncoder expects (B, tw, %d), got %s"
                % (T_RAW_DIM + T_PHYS_DIM, tuple(x.shape))
            )
        batch, tw = x.shape[0], x.shape[1]
        seq = self._stream_raw(x)
        h = self.out_merge(seq.reshape(batch, tw, 3 * self.dim))
        h = self.norm(h)
        h = h + self.embeddings.time_pe.to(dtype=h.dtype)
        modality = self.embeddings.modality(
            torch.tensor(1, device=x.device, dtype=torch.long)
        )
        return h + modality.to(dtype=h.dtype)
