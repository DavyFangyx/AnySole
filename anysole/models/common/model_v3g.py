"""V3_3G (2026-10-17): regression carrier + part-wise observation layer.

Fresh architecture.  V3_3G = E 的回归载体（冻结继承）+ 从零训练的 T 接触流
+ 推理侧分部观测层（constraints_g.py）。三条件 = 同一前向的三种读出
（config_id 被忽略，同 V3_3E 协议）。

- V 载体（冻结）: v_enc -> v_temporal -> 9-part pose head（144 维 6D）+
  traj head（速度 -> cumsum）。权重来自 E v2.1 生产 ckpt_best（入口按
  carrier_from 装载）；构建时冻结 embeddings + v_enc + v_temporal +
  pose_head + traj_head——载体层对标的结构保证（V2M 行 = E 的 40.0）。
  sigma 头保留冻结状态（与 E 载体 state-dict 键兼容），观测层不读。
- T 接触流（从零）: foot_conv -> 接触头（每脚 logits，f6_soft，air 加权
  BCE——摆动相 air recall 是观测层的前置条件）。
- T 先验流（从零）: t_temporal -> 9-part prior head + prior traj head
  （T2M 行 = 先验 + 观测层，~90mm 健全性检查）。

V3EPoseHead / V3ETrajHead / V3EContactHead / _temporal_encoder 逐字复制自
model_v3e（载体即 E 的 V 流——复制而非 import 以保持独立模型代码边界；
state-dict 键结构一致使载体装载零映射）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    D_MODEL,
    FPS,
    PART_JOINTS,
    POSE_DIM,
    TW,
)

# V-stream parameter prefixes frozen at construction (carrier identity).
# SharedEmbeddings appears at six paths in the state dict (top level,
# ModalEncoders, both encoder heads, both pose heads); every path is frozen
# so the carrier load is complete and the T stream shares the frozen token
# space (the trained v_enc and its embeddings are coupled).
CARRIER_PREFIXES = (
    "embeddings.",
    "encoders.embeddings.",
    "encoders.t_enc.embeddings.",
    "encoders.v_enc.",
    "v_temporal.",
    "pose_head.",
    "prior_head.embeddings.",
    "traj_head.",
)


def _temporal_encoder(dim: int, n_layers: int, nhead: int = 8,
                      dim_feedforward: int = 1024, dropout: float = 0.1):
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    kwargs = {"num_layers": n_layers}
    try:
        return nn.TransformerEncoder(layer, enable_nested_tensor=False, **kwargs)
    except TypeError:
        return nn.TransformerEncoder(layer, **kwargs)


class V3EPoseHead(nn.Module):
    """[copied from model_v3e, 2026-10-17 — the carrier head]
    Pure single-stream 9-part pose head: part queries decode ONE memory
    stream (no modality views, no gates), per-part unembeds emit the part's
    joints in 6D plus a per-joint log-sigma.  Hard PART_JOINTS partition.
    Sigma heads are frozen at init in V3_3G (E carrier state, not read)."""

    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8,
                 dim_feedforward=1024, dropout=0.1, n_layers=6,
                 part_joints=PART_JOINTS, sigma_init=0.0):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_parts = len(part_joints)
        self.part_joints = tuple(tuple(int(j) for j in g) for g in part_joints)
        self.register_buffer("part_place_idx",
                             _part_place_idx(self.part_joints), persistent=True)
        self.query = nn.Parameter(torch.zeros(1, tw, self.n_parts, dim))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_parts = nn.ModuleList(
            [nn.Linear(dim, 6 * len(joints)) for joints in self.part_joints]
        )
        self.sig_parts = nn.ModuleList(
            [nn.Linear(dim, 6 * len(joints)) for joints in self.part_joints]
        )
        for head in self.sig_parts:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(sigma_init))
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std", torch.ones(POSE_DIM))

    def set_stats(self, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> None:
        self.pose_mean = pose_mean.to(self.pose_mean.dtype)
        self.pose_std = pose_std.to(self.pose_std.dtype)

    def forward(self, mem):
        """mem: (B, tw, dim).  Returns (pose (B,tw,POSE_DIM), log_sigma)."""
        B = mem.shape[0]
        q = self.query.expand(B, -1, -1, -1).reshape(B, self.tw * self.n_parts, self.dim)
        z = self.decoder(q, mem).reshape(B, self.tw, self.n_parts, self.dim)
        pose_parts = torch.cat(
            [self.out_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        sig_parts = torch.cat(
            [self.sig_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        sig = sig_parts[..., self.part_place_idx]
        return pose, sig


class V3ETrajHead(nn.Module):
    """[copied from model_v3e] Single-stream root-trajectory regression
    (velocity -> cumsum)."""

    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.tw = tw
        self.encoder = _temporal_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.proj = nn.Linear(dim, 3)

    def forward(self, mem):
        h = self.encoder(mem)
        v_hat = self.proj(h)
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        return v_hat, trans_hat


class V3EContactHead(nn.Module):
    """[copied from model_v3e] Per-foot contact logits from the tactile
    tokens (f6_soft supervision, air-weighted in V3_3G training)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class AnySoleModelV3G(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # regress machinery in train/eval
        self.v3g = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv",
        )
        self.v_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        self.t_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        # sigma inits match the E carrier (V=-3 / prior=+3); the heads are
        # frozen and never read — state-dict compatibility only.
        self.pose_head = V3EPoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=-3.0)
        self.prior_head = V3EPoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=3.0)
        self.traj_head = V3ETrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.prior_traj = V3ETrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.contact_head = V3EContactHead(dim=d)
        # Carrier rule: the V stream is frozen at construction.  The E
        # ckpt_best weights are loaded by the entry (anysole/models/v3_3g.py);
        # freezing here makes the carrier identity load-independent.
        for name, param in self.named_parameters():
            if name.startswith(CARRIER_PREFIXES):
                param.requires_grad_(False)
        for head in list(self.pose_head.sig_parts) + list(self.prior_head.sig_parts):
            for param in head.parameters():
                param.requires_grad_(False)

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None):
        """config_id is accepted for the train/eval call contract and
        deliberately IGNORED — both streams always live, and the three eval
        conditions are three readouts of this one forward (V3_3E protocol)."""
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        v_tok, t_tok = self.encoders(V_feat, T_tac, cid, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        v_mem = self.v_temporal(v_tok)
        pose_v, sigma_v = self.pose_head(v_mem)
        v_hat, trans_hat = self.traj_head(v_mem)
        t_mem = self.t_temporal(t_tok)
        pose_t, sigma_t = self.prior_head(t_mem)
        v_prior, trans_prior = self.prior_traj(t_mem)
        contact_logits = self.contact_head(t_tok)
        return {
            "x0_hat": pose_v,
            "sigma_v": sigma_v,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "x0_prior": pose_t,
            "sigma_prior": sigma_t,
            "v_prior": v_prior,
            "trans_prior": trans_prior,
            "contact_logits": contact_logits,
            "F": v_mem,          # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
