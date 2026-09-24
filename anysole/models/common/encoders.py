"""V / combined T encoders with config-conditioned null-token replacement.

E6.6: ``tactile_input="s2m50"`` feeds the Step2Motion-口径 50-dim tactile
channel (T_s2m) instead of ``cat([T_raw, T_phys])`` (108-dim); with
``tactile_direct=True`` the tactile encoder is the per-group TactileEncoder
(E6.6b, Step2Motion-style), otherwise a flat LinearTemporalEncoder (E6.6a).

F4a: ``t_encoder="foot_conv"`` (with tactile_input="raw108") replaces the flat
LinearTemporalEncoder with the FootConvEncoder — per-foot spatial-grid conv on
the SAME 108-dim channel (right foot mirrored, [left/right/global] token
stream, temporal transformer).  The input data is untouched; only the
encoding changes (foot_encoder.py).

--no-imu (E6.8): the input becomes the 38-dim T_s2m (IMU channels deleted at
data build time) and the encoder receives 38-dim — for TactileEncoder the two
IMU groups are structurally absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.embeddings import SharedEmbeddings
from anysole.models.foot_encoder import FootConvEncoder
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
    V_HMR_DIM,
    V_HMR_IMG_DIM,
    V_HMR_KP_DIM,
    V_HMR_MISC_DIM,
    V_HMR_ROT_DIM,
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


class VEncHMR(nn.Module):
    """F1: grouped Linear over the GVHMR visual channel, same contract as
    LinearTemporalEncoder.

    V_hmr layout per frame (V_HMR_DIM = 1156):
      rot 76 (body_pose aa 63 + global_orient 3 + betas 10)
      | kp2d 51 (COCO-17 x/y/conf) | img 1024 (HMR2 ViT features)
      | misc 5 (bbox_info 3 + q_V 2)
    One projection per group, summed, then time PE + modality embedding.
    """

    def __init__(self, embeddings, dim=D_MODEL):
        super().__init__()
        self.rot_proj = nn.Linear(V_HMR_ROT_DIM, dim)
        self.kp_proj = nn.Linear(V_HMR_KP_DIM, dim)
        self.img_proj = nn.Linear(V_HMR_IMG_DIM, dim)
        self.misc_proj = nn.Linear(V_HMR_MISC_DIM, dim)
        self.norm = nn.LayerNorm(dim)
        self.embeddings = embeddings

    def forward(self, x):
        if x.shape[-1] != V_HMR_DIM:
            raise ValueError("VEncHMR expects %d-dim input, got %d" % (V_HMR_DIM, x.shape[-1]))
        rot = x[..., :V_HMR_ROT_DIM]
        kp = x[..., V_HMR_ROT_DIM:V_HMR_ROT_DIM + V_HMR_KP_DIM]
        img = x[..., V_HMR_ROT_DIM + V_HMR_KP_DIM:V_HMR_ROT_DIM + V_HMR_KP_DIM + V_HMR_IMG_DIM]
        misc = x[..., -V_HMR_MISC_DIM:]
        h = self.rot_proj(rot) + self.kp_proj(kp) + self.img_proj(img) + self.misc_proj(misc)
        h = self.norm(h)
        h = h + self.embeddings.time_pe.to(dtype=h.dtype)
        modality = self.embeddings.modality(
            torch.tensor(0, device=x.device, dtype=torch.long)
        )
        return h + modality.to(dtype=h.dtype)


class ModalEncoders(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, tactile_input="raw108",
                 tactile_direct=False, no_imu=False, v_input="hrnet",
                 t_encoder="linear"):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        if v_input not in ("hrnet", "hmr_gvhmr"):
            raise ValueError("v_input must be 'hrnet' or 'hmr_gvhmr', got %r" % v_input)
        if tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50', got %r" % tactile_input)
        if t_encoder not in ("linear", "foot_conv"):
            raise ValueError("t_encoder must be 'linear' or 'foot_conv', got %r" % t_encoder)
        if t_encoder == "foot_conv" and tactile_input != "raw108":
            raise ValueError("t_encoder='foot_conv' requires tactile_input='raw108' (F4a)")
        if tactile_direct and tactile_input != "s2m50":
            raise ValueError("tactile_direct requires tactile_input='s2m50'")
        if no_imu and tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50' (raw108 has no IMU channels)")
        self.embeddings = embeddings
        self.tw = tw
        self.tactile_input = tactile_input
        self.t_encoder = t_encoder
        self.v_input = v_input
        self.no_imu = bool(no_imu)
        # F1: hmr_gvhmr replaces the HRNet feature encoder with the grouped
        # GVHMR channel (V_hmr); hrnet keeps the V1 encoder unchanged.
        self.v_enc = VEncHMR(embeddings, dim=dim) if v_input == "hmr_gvhmr" \
            else LinearTemporalEncoder(V_FEAT_DIM, embeddings, 0)
        if t_encoder == "foot_conv":
            # F4a: per-foot grid conv + [left/right/global] temporal tokens
            # on the unchanged 108-dim channel.
            self.t_enc = FootConvEncoder(embeddings, dim=dim, tw=tw)
        elif tactile_direct:
            # E6.6b: per-group encoding (Step2Motion-style), streams merged.
            # --no-imu: 38-dim input, the two IMU groups are deleted.
            self.t_enc = TactileEncoder(embeddings, dim=dim, tw=tw, no_imu=no_imu)
        else:
            if tactile_input == "s2m50":
                in_dim = T_S2M_NOIMU_DIM if no_imu else T_S2M_DIM
            else:
                in_dim = T_RAW_DIM + T_PHYS_DIM
            self.t_enc = LinearTemporalEncoder(in_dim, embeddings, 1)

    def forward(self, V_feat, T_tac, config_id, V_hmr=None, mask_v=None, mask_t=None):
        """``mask_v/mask_t`` (bool, B×tw)：逐帧 null-token mask（missing-rate
        实验 1 / ρ 网格）。True 的位置把该帧 token 换成 null token，叠加在
        config 级替换之后；None = 不 mask（旧路径逐字节一致，零重训）。"""
        if self.v_input == "hmr_gvhmr":
            if V_hmr is None:
                raise ValueError("V_hmr is required when v_input='hmr_gvhmr'")
            v_tok = self.v_enc(V_hmr)
        else:
            v_tok = self.v_enc(V_feat)
        t_tok = self.t_enc(T_tac)

        config_id = config_id.to(device=v_tok.device, dtype=torch.long).reshape(-1)
        drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
        drop_v = (config_id == CONFIG_T).view(-1, 1, 1)

        null_v = self.embeddings.null_v.to(dtype=v_tok.dtype).expand_as(v_tok)
        null_t = self.embeddings.null_t.to(dtype=t_tok.dtype).expand_as(t_tok)

        v_tok = torch.where(drop_v, null_v, v_tok)
        t_tok = torch.where(drop_t, null_t, t_tok)

        def _apply_mask(mask, tok, null):
            k = tok.shape[1] // self.tw
            if tok.shape[1] != k * self.tw:
                raise ValueError("token stream %d is not k*tw=%d; mask must be full-width"
                                 % (tok.shape[1], self.tw))
            m = mask.to(device=tok.device, dtype=torch.bool)
            m = m.view(tok.shape[0], self.tw, 1).repeat_interleave(k, dim=1)
            return torch.where(m, null, tok)

        if mask_v is not None:
            v_tok = _apply_mask(mask_v, v_tok, null_v)
        if mask_t is not None:
            t_tok = _apply_mask(mask_t, t_tok, null_t)
        return v_tok, t_tok
