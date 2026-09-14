"""V / combined T encoders with config-conditioned null-token replacement."""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import CONFIG_T, CONFIG_V, D_MODEL, T_PHYS_DIM, T_RAW_DIM, TW, V_FEAT_DIM


class LinearTemporalEncoder(nn.Module):
    """Linear projection plus time PE and a modality embedding."""

    def __init__(self, in_dim, embeddings, modality_id):
        super().__init__()
        self.proj = nn.Linear(in_dim, embeddings.dim)
        # Normalize the projection before time PE / modality embedding so an
        # input-scale mismatch can never propagate through the residual flow.
        self.norm = nn.LayerNorm(embeddings.dim)
        self.embeddings = embeddings
        self.modality_id = int(modality_id)

    def forward(self, x):
        h = self.norm(self.proj(x))
        h = h + self.embeddings.time_pe.to(dtype=h.dtype)
        modality = self.embeddings.modality(
            torch.tensor(self.modality_id, device=x.device, dtype=torch.long)
        )
        return h + modality.to(dtype=h.dtype)


class ModalEncoders(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.tw = tw
        self.v_enc = LinearTemporalEncoder(V_FEAT_DIM, embeddings, 0)
        self.t_enc = LinearTemporalEncoder(T_RAW_DIM + T_PHYS_DIM, embeddings, 1)

    def forward(self, V_feat, T_raw, T_phys, config_id):
        v_tok = self.v_enc(V_feat)
        t_tok = self.t_enc(torch.cat([T_raw, T_phys], dim=-1))

        config_id = config_id.to(device=v_tok.device, dtype=torch.long).reshape(-1)
        drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
        drop_v = (config_id == CONFIG_T).view(-1, 1, 1)

        null_v = self.embeddings.null_v.to(dtype=v_tok.dtype).expand_as(v_tok)
        null_t = self.embeddings.null_t.to(dtype=t_tok.dtype).expand_as(t_tok)

        v_tok = torch.where(drop_v, null_v, v_tok)
        t_tok = torch.where(drop_t, null_t, t_tok)
        return v_tok, t_tok
