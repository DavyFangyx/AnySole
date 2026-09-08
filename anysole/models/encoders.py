"""V / T_raw / T_phys encoders with config-conditioned null-token replacement."""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    D_MODEL,
    T_PHYS_DIM,
    T_RAW_DIM,
    TW,
    V_FEAT_DIM,
)


class LinearTemporalEncoder(nn.Module):
    """Linear projection plus time PE and a modality embedding."""

    def __init__(self, in_dim, embeddings, modality_id):
        super().__init__()
        self.proj = nn.Linear(in_dim, embeddings.dim)
        self.embeddings = embeddings
        self.modality_id = int(modality_id)

    def forward(self, x):
        h = self.proj(x)
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
        self.traw_enc = LinearTemporalEncoder(T_RAW_DIM, embeddings, 1)
        self.tphys_enc = LinearTemporalEncoder(T_PHYS_DIM, embeddings, 2)

    def forward(self, V_feat, T_raw, T_phys, config_id):
        v_tok = self.v_enc(V_feat)
        traw_tok = self.traw_enc(T_raw)
        tphys_tok = self.tphys_enc(T_phys)

        config_id = config_id.to(device=v_tok.device, dtype=torch.long).reshape(-1)
        drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
        drop_v = (config_id == CONFIG_T).view(-1, 1, 1)

        null_v = self.embeddings.null_v.to(dtype=v_tok.dtype).expand_as(v_tok)
        null_traw = self.embeddings.null_traw.to(dtype=traw_tok.dtype).expand_as(traw_tok)
        null_tphys = self.embeddings.null_tphys.to(dtype=tphys_tok.dtype).expand_as(tphys_tok)

        v_tok = torch.where(drop_v, null_v, v_tok)
        traw_tok = torch.where(drop_t, null_traw, traw_tok)
        tphys_tok = torch.where(drop_t, null_tphys, tphys_tok)
        return v_tok, traw_tok, tphys_tok
