"""V3_3H (2026-10-17): single-generator regression + T physical state +
residual channels.

Fresh architecture.  V3_3H = 纯 V 回归主干（与 E 同构的已证机器，从零训练）
+ T 接触流（从零）+ 推理侧残差通道（constraints_h.py）。三条件 = 同一前向
的三种读出（config_id 被忽略，同 V3_3E 协议）。

- V 主干: v_enc（冻结 vonly 专才编码器，入口装载）-> v_temporal ->
  9-part pose head（144 维 6D，硬划分、无门控、无 soft A）+ traj head
  （速度 -> cumsum）。V 塔是唯一生成者——H 与 E 的结构性区别：无 T 姿态
  塔、无先验头，T 侧不出内容（「完整流形」只有一个生成者）。
- T 侧（证据，不出内容）: foot_conv -> 接触头（每脚 logits，f6_soft，
  air 加权 BCE——接触置信帧是残差通道的闸门）。
- 训练条件：VT 90% / V 零化 10%（T2M 协议代价——null-V 先验只能靠输入
  零化训练，无学习参数 null token；V3_3H方案.md §3）。

V3HPoseHead / V3HTrajHead / V3HContactHead / _temporal_encoder 与 model_v3e
的对应部件同构（回归主干 = 已证机器），独立实现：无 sigma 头（H 无
NLL/回退，sigma 分支整体不携带——state dict 与 E 不共享，无映射要求）。

V3_3H_v3 = v2 + 一阶足部残差头（foot_res_head：t_tok → 每足 6D 残差，
接触软门控、输出饱和）——命题 2 的兑现通道（V3_3H方案 §3 对照实验）。
残差只写足关节（10/11）的 6D 槽位，上身/对侧结构上不可被 T 改动；V2M
行不应用残差（零税），VT/T 行在残差层之前应用。
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
    T_RAW_DIM,
    TW,
)

# 残差头落点槽位：足关节（10/11）为叶节点，改自身旋转不移动任何关键点
# （V3_3H设计错误裁定.md §1）；踝关节（7/8）旋转经 127 mm 杠杆 ≈ 平移足
# 关键点。V3_3H_v6 = 落点改踝（错误一修复），v1–v5 保持足落点。
RES_SLOTS = {
    "foot": (slice(10 * 6, 11 * 6), slice(11 * 6, 12 * 6)),
    "ankle": (slice(7 * 6, 8 * 6), slice(8 * 6, 9 * 6)),
}
FOOT_RES_MAX = 0.05   # per-foot 6D 残差范数饱和上限
FOOT_RES_DIM = 12     # 2 足 × 6D


def apply_foot_residual(pose: torch.Tensor, foot_res: torch.Tensor,
                        landing: str = "foot") -> torch.Tensor:
    """零税应用：残差只写落点关节的 6D 槽位，其余部位逐位不变。

    pose: (B, tw, 144) 原始 6D 姿态；foot_res: (B, tw, 12)。
    landing="foot" = 足关节 10/11（v1–v5）；"ankle" = 踝关节 7/8（V3_3H_v6）。
    """
    out = pose.clone()
    s0, s1 = RES_SLOTS[landing]
    out[:, :, s0] = out[:, :, s0] + foot_res[:, :, 0:6]
    out[:, :, s1] = out[:, :, s1] + foot_res[:, :, 6:12]
    return out


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


class V3HPoseHead(nn.Module):
    """Pure single-stream 9-part pose head: part queries decode ONE memory
    stream (no modality views, no gates), per-part unembeds emit the part's
    joints in 6D.  Hard PART_JOINTS partition.  E-machinery minus the sigma
    branch (H has no NLL/fallback — the sigma head is not carried at all)."""

    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8,
                 dim_feedforward=1024, dropout=0.1, n_layers=6,
                 part_joints=PART_JOINTS):
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
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std", torch.ones(POSE_DIM))

    def set_stats(self, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> None:
        self.pose_mean = pose_mean.to(self.pose_mean.dtype)
        self.pose_std = pose_std.to(self.pose_std.dtype)

    def forward(self, mem):
        """mem: (B, tw, dim).  Returns the 6D pose (B, tw, POSE_DIM)."""
        B = mem.shape[0]
        q = self.query.expand(B, -1, -1, -1).reshape(B, self.tw * self.n_parts, self.dim)
        z = self.decoder(q, mem).reshape(B, self.tw, self.n_parts, self.dim)
        pose_parts = torch.cat(
            [self.out_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        return pose


class V3HTrajHead(nn.Module):
    """Single-stream root-trajectory regression (velocity -> cumsum)."""

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


class V3HContactHead(nn.Module):
    """Per-foot contact logits from the tactile tokens (f6_soft supervision,
    air-weighted in V3_3H training — contact confidence gates the residual
    channels)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class AnySoleModelV3H(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6,
                 t_rec: bool = False, foot_res: bool = False,
                 no_anchor: bool = False, res_landing: str = "foot"):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # regress machinery in train/eval
        self.v3h = True
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
        self.pose_head = V3HPoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers)
        self.traj_head = V3HTrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.contact_head = V3HContactHead(dim=d)
        # V3_3H_v2: aux pressure reconstruction on t_tok (B 的已验证配方，
        # λ_trec=0.1) — 命题 2 载体依赖修复：T 编码器监督从「仅接触」恢复
        # 为「接触 + 重建」，让 t_tok 保留足部姿态信息。v1（t_rec=False）
        # 无此头，state dict 与 v1 逐键兼容。
        self.t_rec_head = nn.Linear(d, T_RAW_DIM) if t_rec else None
        # V3_3H_v3: 一阶足部残差头（接触软门控 + 输出饱和）——命题 2 的
        # 兑现通道；v1/v2 无此头，state dict 逐键兼容。
        self.foot_res_head = nn.Linear(d, FOOT_RES_DIM) if foot_res else None
        self.foot_res = bool(foot_res)
        # V3_3H_v5：无锚版——地面/穿透通道整体删除（T 不提供绝对高度，
        # 模型不消费任何 floor 输入）；推理层只跑支撑通道。
        self.no_anchor = bool(no_anchor)
        # V3_3H_v6：残差头落点改踝（"ankle"，错误一修复）；v1–v5 = "foot"。
        self.res_landing = str(res_landing)
        # The frozen vonly v_enc is loaded and frozen by the entry
        # (anysole/models/v3_3h.py, F-line precedent) — no freezing here:
        # the load is the only sanctioned reuse, everything else trains.

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None):
        """config_id is accepted for the train/eval call contract and
        deliberately IGNORED — the three eval conditions are three readouts
        of this one forward (V3_3E protocol); the T2M row runs it with V
        zeroed upstream.  T stays live under every condition (the contact
        head always reads the real stream)."""
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        v_tok, t_tok = self.encoders(V_feat, T_tac, cid, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        v_mem = self.v_temporal(v_tok)
        pose_v = self.pose_head(v_mem)
        v_hat, trans_hat = self.traj_head(v_mem)
        contact_logits = self.contact_head(t_tok)
        pressure_hat = self.t_rec_head(t_tok) if self.t_rec_head is not None else None
        foot_res = None
        if self.foot_res_head is not None:
            # 接触软门控 × 输出饱和：残差范数上限 FOOT_RES_MAX，门控 = 接触
            # 概率（平滑、有界——学习式门控禁用的规则内）。
            c = torch.sigmoid(contact_logits)                    # (B,tw,2)
            r = self.foot_res_head(t_tok).reshape(-1, self.tw, 2, 6)
            rn = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            r = r / rn * torch.minimum(
                rn, torch.full_like(rn, FOOT_RES_MAX))
            foot_res = (r * c.unsqueeze(-1)).reshape(-1, self.tw, FOOT_RES_DIM)
        return {
            "x0_hat": pose_v,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "contact_logits": contact_logits,
            "F": v_mem,          # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            # call-site contract: H carries no T pose tower and no sigma
            # branch — these stay None so the shared loss/metric paths skip.
            "x0_prior": None,
            "sigma_v": None,
            "v_prior": None,
            "trans_prior": None,
            "pressure_hat": pressure_hat,
            "vfeat_hat": None,
            "foot_res": foot_res,
            "res_landing": self.res_landing,
        }
