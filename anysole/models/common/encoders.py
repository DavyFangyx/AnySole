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

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.foot_encoder import FootConvEncoder
from anysole.models.common.tactile_encoder import TactileEncoder
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    D_MODEL,
    T_PHYS_DIM,
    T_RAW_DIM,
    T_S2M_DIM,
    T_S2M_IMU_COLUMNS,
    T_S2M_IMU_DIM,
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
        h = h + self.embeddings.time_pe[-h.shape[1]:].to(dtype=h.dtype)
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
        h = h + self.embeddings.time_pe[-h.shape[1]:].to(dtype=h.dtype)
        modality = self.embeddings.modality(
            torch.tensor(0, device=x.device, dtype=torch.long)
        )
        return h + modality.to(dtype=h.dtype)


class ChannelMLPEncoder(nn.Module):
    """Per-frame low-dim channel embedding: in_dim -> 3-layer MLP -> d.

    Used for the S2M-style IMU embedding and for the separate T_phys stream in
    the PhysIMU variant.  S2M's ``condition_emb_*`` are per-frame MLPs; the
    temporal context is supplied downstream by the fusion transformer, so no
    time PE is added here.  The final LayerNorm matches the token scale of the
    tactile stream.
    """

    def __init__(self, in_dim, dim=D_MODEL):
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError("ChannelMLPEncoder got last dim %d, expected %d"
                             % (x.shape[-1], self.in_dim))
        return self.norm(self.mlp(x))


class IMUEncoder(ChannelMLPEncoder):
    """S2M-style IMU embedding on the 12 ``T_S2M_IMU_COLUMNS`` channels."""

    def __init__(self, in_dim=T_S2M_IMU_DIM, dim=D_MODEL):
        if int(in_dim) != T_S2M_IMU_DIM:
            raise ValueError("IMUEncoder expects %d IMU channels, got %d"
                             % (T_S2M_IMU_DIM, int(in_dim)))
        super().__init__(int(in_dim), dim)


class PhysIMUFusion(nn.Module):
    """Two-stage fusion for the ``raw96_phys_imu`` variant.

    Stage 1: [T_phys embedding, IMU embedding] -> 2d -> d (physical state).
    Stage 2: [T_raw FootConv token, physical state] -> 2d -> d (tactile token).
    Both projections are followed by LayerNorm to keep the fused memory at the
    same scale as the other encoder tokens (the F5 lesson: an un-normalized
    stream can dominate cross-attention).  Unlike ``IMUConcatFusion`` there is
    no bitwise base contract here — the pressure encoder itself changes from
    raw108 to raw96, so the variant is trained from scratch.
    """

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.dim = int(dim)
        self.phys_proj = nn.Linear(2 * self.dim, self.dim)
        self.phys_norm = nn.LayerNorm(self.dim)
        self.out_proj = nn.Linear(2 * self.dim, self.dim)
        self.out_norm = nn.LayerNorm(self.dim)

    def forward(self, t_raw: torch.Tensor, t_phys: torch.Tensor,
                imu: torch.Tensor) -> torch.Tensor:
        if t_raw.shape[:-1] != t_phys.shape[:-1] or t_raw.shape[:-1] != imu.shape[:-1]:
            raise ValueError(
                "t_raw/t_phys/imu token shape mismatch: %s vs %s vs %s"
                % (tuple(t_raw.shape), tuple(t_phys.shape), tuple(imu.shape)))
        phys = self.phys_norm(self.phys_proj(torch.cat([t_phys, imu], dim=-1)))
        return self.out_norm(self.out_proj(torch.cat([t_raw, phys], dim=-1)))


class IMUConcatFusion(nn.Module):
    """Concat [tactile_token, imu_embedding] on the feature axis, project to d.

    The projection is initialized as ``[I | 0]`` with zero bias, so at step 0
    the fused token is *exactly* the tactile token and the IMU path is a
    no-op.  This is the zero-tax contract for the V3_Bv1_IMU variant: a
    freshly built model with the base weights produces bit-identical output to
    V3_Bv1 until the IMU half of the projection moves.  (The right half still
    receives gradient from step 1, so the IMU encoder starts learning
    immediately.)
    """

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Linear(2 * self.dim, self.dim)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :self.dim] = torch.eye(self.dim)
            self.proj.bias.zero_()

    def forward(self, t_tok: torch.Tensor, imu_tok: torch.Tensor) -> torch.Tensor:
        if t_tok.shape[:-1] != imu_tok.shape[:-1]:
            raise ValueError("tactile/IMU token shape mismatch: %s vs %s"
                             % (tuple(t_tok.shape), tuple(imu_tok.shape)))
        return self.proj(torch.cat([t_tok, imu_tok], dim=-1))


class ModalEncoders(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, tactile_input="raw108",
                 tactile_direct=False, no_imu=False, v_input="hrnet",
                 t_encoder="linear", t_phys_v2=False):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        if v_input not in ("hrnet", "hmr_gvhmr"):
            raise ValueError("v_input must be 'hrnet' or 'hmr_gvhmr', got %r" % v_input)
        if tactile_input not in ("raw108", "s2m50", "raw108_imu", "raw96_phys_imu"):
            raise ValueError(
                "tactile_input must be 'raw108', 's2m50', 'raw108_imu' or "
                "'raw96_phys_imu', got %r" % tactile_input)
        if t_encoder not in ("linear", "foot_conv"):
            raise ValueError("t_encoder must be 'linear' or 'foot_conv', got %r" % t_encoder)
        if t_encoder == "foot_conv" and tactile_input not in (
                "raw108", "raw108_imu", "raw96_phys_imu"):
            raise ValueError(
                "t_encoder='foot_conv' requires tactile_input in "
                "'raw108'/'raw108_imu'/'raw96_phys_imu' (F4a)")
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
        raw_only = tactile_input == "raw96_phys_imu"
        if t_encoder == "foot_conv":
            # F4a: per-foot grid conv + [left/right/global] temporal tokens.
            # raw108/raw108_imu keep the 108-dim (pressure+T_phys) path;
            # raw96_phys_imu passes only the 96 pressure cells and encodes
            # T_phys in a separate MLP branch below.
            self.t_enc = FootConvEncoder(embeddings, dim=dim, tw=tw,
                                          t_phys_v2=t_phys_v2,
                                          raw_only=raw_only)
        elif tactile_direct:
            # E6.6b: per-group encoding (Step2Motion-style), streams merged.
            # --no-imu: 38-dim input, the two IMU groups are deleted.
            self.t_enc = TactileEncoder(embeddings, dim=dim, tw=tw, no_imu=no_imu)
        else:
            if tactile_input == "s2m50":
                in_dim = T_S2M_NOIMU_DIM if no_imu else T_S2M_DIM
            elif tactile_input == "raw96_phys_imu":
                in_dim = T_RAW_DIM
            else:
                in_dim = T_RAW_DIM + T_PHYS_DIM
            self.t_enc = LinearTemporalEncoder(in_dim, embeddings, 1)
        # raw108_imu (2026-11-12): S2M-style IMU branch.  Pressure keeps the
        # raw108 encoder above; the 12 IMU columns of T_s2m go through their
        # own MLP and are concatenated with the tactile token at the feature
        # axis.  The concat projection is [I | 0] at init (zero tax).
        self.imu_input = tactile_input == "raw108_imu"
        if self.imu_input:
            self.imu_enc = IMUEncoder(dim=dim)
            self.imu_fuse = IMUConcatFusion(dim=dim)
        else:
            self.imu_enc = None
            self.imu_fuse = None
        # raw96_phys_imu (2026-11-12, fusion 2): T_raw 96 -> FootConv,
        # T_phys 12 -> MLP, IMU 12 -> MLP, then a two-stage 2d->d fusion:
        # [T_phys, IMU] -> physical state, then [T_raw, physical] -> t_tok.
        self.phys_imu = raw_only
        if self.phys_imu:
            t_phys_dim = 20 if t_phys_v2 else T_PHYS_DIM
            self.t_phys_enc = ChannelMLPEncoder(t_phys_dim, dim=dim)
            self.phys_imu_enc = IMUEncoder(dim=dim)
            self.phys_imu_fuse = PhysIMUFusion(dim=dim)
        else:
            self.t_phys_enc = None
            self.phys_imu_enc = None
            self.phys_imu_fuse = None

    @staticmethod
    def _imu_columns(T_s2m: torch.Tensor, mode: str) -> torch.Tensor:
        if T_s2m.shape[-1] != T_S2M_DIM:
            raise ValueError("%s expects %d-dim T_s2m, got %d"
                             % (mode, T_S2M_DIM, T_s2m.shape[-1]))
        return T_s2m[..., list(T_S2M_IMU_COLUMNS)]

    def _add_time_and_modality(self, tok: torch.Tensor) -> torch.Tensor:
        # S2M adds the temporal PE to every condition stream; the low-dim
        # MLP branches get the same frame PE + T-modality tag as t_tok.
        tok = tok + self.embeddings.time_pe[-tok.shape[1]:].to(dtype=tok.dtype)
        tok = tok + self.embeddings.modality(
            torch.tensor(1, device=tok.device, dtype=torch.long)
        ).to(dtype=tok.dtype)
        return tok

    def forward(self, V_feat, T_tac, config_id, V_hmr=None, mask_v=None, mask_t=None,
                T_s2m=None, T_phys=None):
        """``mask_v/mask_t`` (bool, B×tw)：逐帧 null-token mask（missing-rate
        实验 1 / ρ 网格）。True 的位置把该帧 token 换成 null token，叠加在
        config 级替换之后；None = 不 mask（旧路径逐字节一致，零重训）。

        ``T_s2m`` is consumed when ``tactile_input`` is 'raw108_imu' (IMU
        columns only) or 'raw96_phys_imu' (IMU columns + separate T_phys
        branch).  ``T_phys`` is required for 'raw96_phys_imu' and ignored by
        the other modes (their T_phys is already inside T_tac).
        """
        if self.v_input == "hmr_gvhmr":
            if V_hmr is None:
                raise ValueError("V_hmr is required when v_input='hmr_gvhmr'")
            v_tok = self.v_enc(V_hmr)
        else:
            v_tok = self.v_enc(V_feat)
        t_tok = self.t_enc(T_tac)
        if self.phys_imu:
            if T_s2m is None or T_phys is None:
                raise ValueError(
                    "T_s2m and T_phys are required when tactile_input='raw96_phys_imu'")
            imu = self._imu_columns(T_s2m, "raw96_phys_imu")
            t_phys_tok = self._add_time_and_modality(self.t_phys_enc(T_phys))
            imu_tok = self._add_time_and_modality(self.phys_imu_enc(imu))
            t_tok = self.phys_imu_fuse(t_tok, t_phys_tok, imu_tok)
        elif self.imu_input:
            if T_s2m is None:
                raise ValueError("T_s2m is required when tactile_input='raw108_imu'")
            imu = self._imu_columns(T_s2m, "raw108_imu")
            imu_tok = self._add_time_and_modality(self.imu_enc(imu))
            t_tok = self.imu_fuse(t_tok, imu_tok)

        config_id = config_id.to(device=v_tok.device, dtype=torch.long).reshape(-1)
        drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
        drop_v = (config_id == CONFIG_T).view(-1, 1, 1)

        null_v = self.embeddings.null_v.to(dtype=v_tok.dtype).expand_as(v_tok)
        null_t = self.embeddings.null_t.to(dtype=t_tok.dtype).expand_as(t_tok)

        v_tok = torch.where(drop_v, null_v, v_tok)
        t_tok = torch.where(drop_t, null_t, t_tok)

        def _apply_mask(mask, tok, null):
            frames = mask.shape[1]
            k = tok.shape[1] // frames
            if tok.shape[1] != k * frames:
                raise ValueError("token stream %d is not k*tw=%d; mask must be full-width"
                                 % (tok.shape[1], self.tw))
            m = mask.to(device=tok.device, dtype=torch.bool)
            m = m.view(tok.shape[0], frames, 1).repeat_interleave(k, dim=1)
            return torch.where(m, null, tok)

        if mask_v is not None:
            v_tok = _apply_mask(mask_v, v_tok, null_v)
        if mask_t is not None:
            t_tok = _apply_mask(mask_t, t_tok, null_t)
        return v_tok, t_tok
