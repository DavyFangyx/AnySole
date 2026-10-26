"""V7 共同骨架（2026-10-25）：独立证据 + 共享生成。

z_note/multimodal_fusion/模型设计思路.md 候选 A/B/C 的实现载体：
  A = 本骨架本身（独立证据流、共享运动解码、三条件硬关闭分支）；
  B = A + 接触参与根运动（contact_root：损失侧足-地一致性，经 FK 回传）；
  C = A + 窗口级可靠性路由（reliability_route：T 信任门调制 T 交叉
      注意力输出，v1 端到端学习、无显式质量标签）。

结构（数据流）：
  V：HRNet 2051/帧 → v_enc → v_temporal(4L) → M_V (B, tw, d)
  T：raw108 → foot_conv → t_temporal(4L) → M_T (B, tw, d)
  运动 query = 根 query(1) + 9 部位 query（learned + 时间 PE）→ 共享
    解码器 N 层：每层 = query 自注意力 + xattn(M_V) + xattn(M_T) + FFN；
    缺失模态 = 硬关闭（该分支输出乘 per-sample 存在掩码 0——同一决定
    驱动输入置零、注意力掩码与损失掩码，模型设计思路 §4）。
  读出：部位线性头 → 6D 旋转（POSE_DIM，hard PART_JOINTS 拼装，pose
    统计归一化与既有头同款）；根头 → 3D 根速度（cumsum → 轨迹）。
    关节位置一律由 FK 从「旋转 + 根运动」统一计算（eval 协议侧），
    模型不独立修关节点（设计思路 §2 运动学一致要求）。
  T 侧：接触头（f6_soft BCE，T 在场窗口监督）。

三条件 = 同一前向的三种输入组合（config_id ∈ {VT, V, T} → 存在掩码）；
T-only 的上身经「共享运动先验」回退（部位 query 只 attend 到 M_T，
上身无证据 → 学到的条件先验；不强制蒸馏视觉教师的上肢具体动作）。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.model_v4_1 import (
    V41ContactHead,
    _temporal_encoder,
)
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    D_MODEL,
    FPS,
    PART_JOINTS,
    POSE_DIM,
    TW,
)

__all__ = ["AnySoleModelV7"]


class _V7DecoderLayer(nn.Module):
    """一层共享运动解码：query 自注意力 + 分别读取 M_V / M_T + FFN。

    缺失模态硬关闭：该分支输出乘 per-sample 存在掩码（B,1,1）——贡献
    严格为零（不是软掩码）。C 开关：T 分支输出再乘窗口级信任门
    g (B, tw, 1)。
    """

    def __init__(self, dim, nhead, dim_feedforward, dropout, route=False):
        super().__init__()
        self.route = route
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.xattn_v = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.xattn_t = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, dim))
        self.ln_q = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)
        self.ln_t = nn.LayerNorm(dim)
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, q, m_v, m_t, v_present, t_present, tw, gate=None):
        # q / m_v / m_t: (B, seq, d)；v_present/t_present: (B,) float01
        h = self.ln_q(q + self.self_attn(q, q, q, need_weights=False)[0])
        v_out = self.xattn_v(h, m_v, m_v, need_weights=False)[0]
        h = self.ln_v(h + v_out * v_present.view(-1, 1, 1))
        t_out = self.xattn_t(h, m_t, m_t, need_weights=False)[0]
        if gate is not None:
            batch, queries, dim = t_out.shape
            nq = queries // tw
            t_out = t_out.view(batch, tw, nq, dim)
            if gate.shape == (batch, tw, 1):
                gate = gate.unsqueeze(2)
            elif gate.shape != (batch, tw, nq, 1):
                raise ValueError("T gate must be window-level or match the motion queries")
            t_out = t_out * gate
            t_out = t_out.view(q.shape[0], q.shape[1], q.shape[2])
        h = self.ln_t(h + t_out * t_present.view(-1, 1, 1))
        return self.ln_f(h + self.ffn(h))


class V7MotionHead(nn.Module):
    """运动 query 解码器：根 + 9 部位 query → 6D 旋转 + 根速度。

    set_stats 接口与既有 pose head 同款（train.py 统一拟合姿态统计）。"""

    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8,
                 dim_feedforward=1024, dropout=0.1, n_layers=6, route=False):
        super().__init__()
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_parts = len(PART_JOINTS)
        self.part_joints = tuple(tuple(int(j) for j in g) for g in PART_JOINTS)
        self.register_buffer("part_place_idx",
                             _part_place_idx(self.part_joints), persistent=True)
        self.root_query = nn.Parameter(torch.zeros(1, tw, 1, dim))
        self.part_queries = nn.Parameter(torch.zeros(1, tw, self.n_parts, dim))
        nn.init.normal_(self.root_query, std=0.02)
        nn.init.normal_(self.part_queries, std=0.02)
        self.layers = nn.ModuleList([
            _V7DecoderLayer(dim, nhead, dim_feedforward, dropout, route=route)
            for _ in range(n_layers)
        ])
        self.out_parts = nn.ModuleList(
            [nn.Linear(dim, 6 * len(joints)) for joints in self.part_joints])
        self.root_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std", torch.ones(POSE_DIM))

    def set_stats(self, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> None:
        self.pose_mean = pose_mean.to(self.pose_mean.dtype)
        self.pose_std = pose_std.to(self.pose_std.dtype)

    def forward(self, m_v, m_t, v_present, t_present, gate=None,
                t_only_mask=None, t_only_adapters=None, z_override=None):
        """→ (pose (B,tw,POSE_DIM), v_hat (B,tw,3), z_root, z_parts)

        ``z_override``（TTA 注入点，2026-10-27）：给定 (B, tw*(1+n_parts), d)
        时替换 decoder 输出（out_parts/root_head 前的部位表示，论文 token_out
        的对应位置；梯度只经冻结的 out_parts/root_head，不回传 decoder/
        temporal/编码器）；默认 None = 原行为不变。
        """
        B = m_v.shape[0]
        q_root = self.root_query.expand(B, -1, -1, -1) \
            + self.embeddings.time_pe.to(self.root_query.dtype).view(1, self.tw, 1, self.dim)
        q_parts = self.part_queries.expand(B, -1, -1, -1) \
            + self.embeddings.time_pe.to(self.part_queries.dtype).view(1, self.tw, 1, self.dim)
        q = torch.cat([q_root, q_parts], dim=2)          # (B, tw, 1+9, d)
        q = q.reshape(B, self.tw * (1 + self.n_parts), self.dim)
        for layer in self.layers:
            q = layer(q, m_v, m_t, v_present, t_present, self.tw, gate=gate)
        if z_override is not None:
            q = z_override  # TTA 注入点（decoder 输出 / out_parts·root_head 前）
        q = q.reshape(B, self.tw, 1 + self.n_parts, self.dim)
        z_root = q[:, :, 0:1, :]
        z_parts = q[:, :, 1:, :]
        if t_only_adapters is not None:
            if t_only_mask is None:
                raise ValueError("T-only adapters require an explicit availability mask")
            mask = t_only_mask.to(q.dtype).view(B, 1, 1, 1)
            z_root = z_root + mask * t_only_adapters[0](z_root)
            z_parts = z_parts + mask * t_only_adapters[1](z_parts)
        v_hat = self.root_head(z_root)[:, :, 0, :]       # (B, tw, 3)
        pose_parts = torch.cat(
            [self.out_parts[p](z_parts[..., p, :]) for p in range(self.n_parts)],
            dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        return pose, v_hat, z_root, z_parts


class AnySoleModelV7(nn.Module):
    """独立证据 + 共享生成（候选 A/B/C 载体）。"""

    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6,
                 contact_root: bool = False, reliability_route: bool = False,
                 task_routes: bool = False, t_only_adapters: bool = False):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"
        self.v7 = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.contact_root = bool(contact_root)
        self.reliability_route = bool(reliability_route)
        self.task_routes = bool(task_routes)
        self.t_only_adapters = bool(t_only_adapters)
        if self.task_routes and not self.reliability_route:
            raise ValueError("Task routes require tactile reliability routing")
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv")
        self.v_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        self.t_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        # 运动解码（挂名为 pose_head：train.py 的姿态统计拟合按名调用）。
        self.pose_head = V7MotionHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, route=reliability_route)
        self.contact_head = V41ContactHead(dim=d)
        if reliability_route:
            # C：窗口级 T 信任门（端到端学习，v1 无显式质量标签——设计
            # 思路 §3 候选 C 的第一版）。输入 = V/T 编码 token 的窗口均值。
            self.route_mlp = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        # Add private parameters after shared initialization. V2 initially
        # reproduces C's single gate, then learns independent root routing.
        if self.task_routes:
            self.route_root_mlp = copy.deepcopy(self.route_mlp)
        if self.t_only_adapters:
            with torch.random.fork_rng(devices=[]):
                self.t_only_root_adapter = self._make_t_only_adapter(d)
                self.t_only_pose_adapter = self._make_t_only_adapter(d)

    @staticmethod
    def _make_t_only_adapter(dim):
        adapter = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, max(dim // 4, 1)), nn.GELU(),
            nn.Linear(max(dim // 4, 1), dim))
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def _route_gate(self, v_tok, t_tok):
        # 窗口级池化（2026-10-25 修订）：两路 token 的窗口均值 → 每训练
        # 窗口一个标量门（设计思路 §3 候选 C 的「窗口级」原义）。
        x = torch.cat([v_tok, t_tok], dim=-1).mean(dim=1)   # (B, 2d)
        g = torch.sigmoid(self.route_mlp(x))                # (B, 1)
        if self.task_routes:
            root = torch.sigmoid(self.route_root_mlp(x))
            gates = torch.cat([root, g.expand(-1, self.pose_head.n_parts)], dim=1)
            return gates[:, None, :, None].expand(-1, self.tw, -1, -1)
        return g.view(-1, 1, 1).expand(-1, self.tw, 1)      # (B, tw, 1)

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None,
                v_present=None, t_present=None, z_override=None):
        """三条件 = 同一前向的三种输入组合。存在掩码默认由 config_id 决定
        （与输入置零同一决定，设计思路 §4）；v_present/t_present 允许调用
        侧覆盖（诊断/干预用）。"""
        if config_id is None:
            config_id = torch.zeros(V_feat.shape[0], dtype=torch.long,
                                    device=V_feat.device)
        if v_present is None:
            v_present = (config_id != CONFIG_T).to(V_feat.dtype)
        if t_present is None:
            t_present = (config_id != CONFIG_V).to(V_feat.dtype)
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        m_v = self.v_temporal(v_tok)
        m_t = self.t_temporal(t_tok)
        gate = self._route_gate(v_tok, t_tok) if self.reliability_route else None
        adapters = ((self.t_only_root_adapter, self.t_only_pose_adapter)
                    if self.t_only_adapters else None)
        t_only = (~v_present.bool()) & t_present.bool()
        pose, v_hat, z_root, z_parts = self.pose_head(
            m_v, m_t, v_present, t_present, gate=gate,
            t_only_mask=t_only, t_only_adapters=adapters,
            z_override=z_override)
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        contact_logits = self.contact_head(t_tok)
        return {
            "x0_hat": pose,                  # 单一生成器的输出（当前条件）
            "v_hat": v_hat,                  # 根速度
            "trans_hat": trans_hat,          # 根轨迹（cumsum）
            "contact_logits": contact_logits,
            "route_gate": gate,              # C：窗口级 T 信任门（监控/损失）
            "z": torch.cat([z_root, z_parts], dim=2),  # (B,tw,1+n_parts,d) TTA 叶子
            "F": m_v + m_t,                  # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "m_v": m_v,
            "m_t": m_t,
            "z_root": z_root,
            "z_parts": z_parts,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
