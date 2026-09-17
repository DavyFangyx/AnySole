"""V / combined T encoders with config-conditioned null-token replacement.

E6.6: ``tactile_input="s2m50"`` feeds the Step2Motion-口径 50-dim tactile
channel (T_s2m) instead of ``cat([T_raw, T_phys])`` (108-dim); with
``tactile_direct=True`` the tactile encoder is the per-group TactileEncoder
(E6.6b, Step2Motion-style), otherwise a flat LinearTemporalEncoder (E6.6a).

--no-imu (E6.8): the input becomes the 38-dim T_s2m (IMU channels deleted at
data build time) and the encoder receives 38-dim — for TactileEncoder the two
IMU groups are structurally absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.models.tactile_encoder import TactileEncoder
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    D_MODEL,
    T_PHYS_DIM,
    T_RAW_DIM,
    T_S2M_DIM,
    T_S2M_NOIMU_DIM,
    TW,
    V_FEAT_DIM,
)


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
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, tactile_input="raw108",
                 tactile_direct=False, no_imu=False):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        if tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50', got %r" % tactile_input)
        if tactile_direct and tactile_input != "s2m50":
            raise ValueError("tactile_direct requires tactile_input='s2m50'")
        if no_imu and tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50' (raw108 has no IMU channels)")
        self.embeddings = embeddings
        self.tw = tw
        self.tactile_input = tactile_input
        self.no_imu = bool(no_imu)
        self.v_enc = LinearTemporalEncoder(V_FEAT_DIM, embeddings, 0)
        if tactile_direct:
            # E6.6b: per-group encoding (Step2Motion-style), streams merged.
            # --no-imu: 38-dim input, the two IMU groups are deleted.
            self.t_enc = TactileEncoder(embeddings, dim=dim, tw=tw, no_imu=no_imu)
        else:
            if tactile_input == "s2m50":
                in_dim = T_S2M_NOIMU_DIM if no_imu else T_S2M_DIM
            else:
                in_dim = T_RAW_DIM + T_PHYS_DIM
            self.t_enc = LinearTemporalEncoder(in_dim, embeddings, 1)

    def forward(self, V_feat, T_tac, config_id):
        v_tok = self.v_enc(V_feat)
        t_tok = self.t_enc(T_tac)

        config_id = config_id.to(device=v_tok.device, dtype=torch.long).reshape(-1)
        drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
        drop_v = (config_id == CONFIG_T).view(-1, 1, 1)

        null_v = self.embeddings.null_v.to(dtype=v_tok.dtype).expand_as(v_tok)
        null_t = self.embeddings.null_t.to(dtype=t_tok.dtype).expand_as(t_tok)

        v_tok = torch.where(drop_v, null_v, v_tok)
        t_tok = torch.where(drop_t, null_t, t_tok)
        return v_tok, t_tok
