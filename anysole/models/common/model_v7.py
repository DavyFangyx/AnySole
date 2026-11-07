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
from anysole.utils.geometry import (
    compose_rot6d, deyaw_rotmat, rot6d_to_rotmat, rotmat_to_6d)
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    D_MODEL,
    FPS,
    PART_JOINTS,
    PART_NAMES,
    POSE_DIM,
    TW,
)

__all__ = ["AnySoleModelV7"]

# V7 C 门控修改方案（2026-10-30，z_note/multimodal_fusion/V7_方案/
# V7_C门控修改方案.md §5）：左右门分组严格按 PART_NAMES 映射，不另写
# 关节序号体系。门组序 = vel（根速度 query）/ other（其余姿态）/
# left（l_leg、l_foot）/ right（r_leg、r_foot）；query 轴序 = [根速度]
# + PART_NAMES（9 部位）。
_V7_GATE_GROUPS = ("vel", "other", "left", "right")
_V7_PART_GATE_GROUP = {
    "root": "other", "torso": "other", "headneck": "other",
    "l_arm": "other", "r_arm": "other",
    "l_leg": "left", "r_leg": "right",
    "l_foot": "left", "r_foot": "right",
}
if set(_V7_PART_GATE_GROUP) != set(PART_NAMES):
    raise AssertionError("part gate groups must cover PART_NAMES exactly")
_V7_QUERY_GATE_IDX = (0,) + tuple(
    _V7_GATE_GROUPS.index(_V7_PART_GATE_GROUP[name]) for name in PART_NAMES)

# 修改 13 D3（2026-11-06）：按可观测性钉门的 query 槽 —— 腿/足四部位。
# query 轴序 = [根速度] + PART_NAMES ⇒ 槽 6/7/8/9。
_V7_LEG_FOOT_PARTS = ("l_leg", "r_leg", "l_foot", "r_foot")
_V7_GATE_PIN_LEGS = (0.0,) + tuple(
    1.0 if name in _V7_LEG_FOOT_PARTS else 0.0 for name in PART_NAMES)

# 竞争路由门（2026-11-07，V8_Cv8）分组特征维度：vel/other = 双足 phys 12
# + IMU 12 + 接触二值 2 = 26；left/right = 单足 phys 6 + IMU 6 + 接触 1。
# IMU 列 = T_s2m 的 16:22（左）/ 41:47（右，tactile_s2m.S2M_IMU_COLUMNS）。
_ROUTER_FEAT_DIMS = (26, 26, 13, 13)


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

    def forward(self, q, m_v, m_t, v_present, t_present, tw, gate=None,
                gate_fn=None, attn_mask=None, gate_v=None):
        # q / m_v / m_t: (B, seq, d)；v_present/t_present: (B,) float01。
        # attn_mask：帧级 AR（2026-11-01）的因果掩码（仅 T 行）。
        # gate_fn（Q，§6）：第一层以「视觉更新后的 query h 与未乘门的 T
        # 注意力输出 a_T」为条件计算门——本层使用，并返回供后续层复用。
        # gate_v（修改 14 §5）：V 侧信任门，乘在 V 交叉注意力输出上；
        # 与 T 门同形状契约（(B,tw,1,1) 或 (B,tw,nq,1)）。
        h = self.ln_q(q + self.self_attn(q, q, q, need_weights=False,
                                         attn_mask=attn_mask)[0])
        v_out = self.xattn_v(h, m_v, m_v, need_weights=False)[0]
        if gate_v is not None:
            batch, queries, dim = v_out.shape
            nq = queries // tw
            gv = gate_v
            if gv.dim() == 3:
                gv = gv.unsqueeze(2)
            if gv.dim() != 4 or gv.shape[0] != batch or gv.shape[1] != tw:
                raise ValueError("V gate must be window-level or match the motion queries")
            if gv.shape[2] == 1:
                gv = gv.expand(-1, -1, nq, -1)
            elif gv.shape[2] != nq:
                raise ValueError("V gate query axis must be 1 or nq")
            v_out = (v_out.view(batch, tw, nq, dim) * gv).view(batch, queries, dim)
        h = self.ln_v(h + v_out * v_present.view(-1, 1, 1))
        t_out = self.xattn_t(h, m_t, m_t, need_weights=False)[0]
        if gate_fn is not None:
            gate = gate_fn(h, t_out)
        if gate is not None:
            batch, queries, dim = t_out.shape
            nq = queries // tw
            t_out = t_out.view(batch, tw, nq, dim)
            g = gate  # 局部变形，不改变返回给后续层复用的门形状
            if g.shape == (batch, tw, 1):
                g = g.unsqueeze(2)
            elif g.shape != (batch, tw, nq, 1):
                raise ValueError("T gate must be window-level or match the motion queries")
            t_out = t_out * g
            t_out = t_out.view(q.shape[0], q.shape[1], q.shape[2])
        h = self.ln_t(h + t_out * t_present.view(-1, 1, 1))
        return self.ln_f(h + self.ffn(h)), gate


class V7MotionHead(nn.Module):
    """运动 query 解码器：根 + 9 部位 query → 6D 旋转 + 根速度。

    set_stats 接口与既有 pose head 同款（train.py 统一拟合姿态统计）。"""

    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8,
                 dim_feedforward=1024, dropout=0.1, n_layers=6, route=False,
                 ctx_v2=False, ctx5=False, ar=False, ar_fb=True, ar2=False,
                 ar2_fb=True, ar2_lowdim=False, ar2_fb_dim=16, root_sd=False,
                 root_sd_t1=False, abs_vrows=False):
        super().__init__()
        # 读出按条件分支（2026-11-04，修改 09）：abs_vrows=True 时只有 T 行
        # （v_present<0.5）做去航向投影 → V/VT 行根 6D 保持绝对世界朝向
        # （= Cv5 目标空间），T 行仍是 tilt + 外部 ψ 累积。默认 False =
        # AR2M 旧行为逐位不变（全条件去航向）。
        self.abs_vrows = bool(abs_vrows)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        # 根头速率/方向解耦（2026-11-04，修改 07 改动 A）：True 时根头
        # 输出 [水平速率, 水平单位方向 (x,z), 垂直分量] 四通道；False 时
        # 与 Cv5 逐位一致（3 通道、同一构造顺序 → 同一 RNG 消耗）。
        self.root_sd = bool(root_sd)
        self.root_sd_t1 = bool(root_sd_t1)
        self.ctx_v2 = bool(ctx_v2)
        self.ctx5 = bool(ctx5)
        self.ar = bool(ar)
        self.ar2 = bool(ar2)
        self.ar_fb = bool(ar_fb)
        self.ar2_fb = bool(ar2_fb)
        self.ar2_lowdim = bool(ar2_lowdim)
        self.ar2_fb_dim = int(ar2_fb_dim)
        if self.ar2_lowdim and not (self.ar2 and self.ar2_fb):
            raise ValueError("ar2_lowdim requires ar2 with feedback")
        if self.ar2_lowdim and not 1 <= self.ar2_fb_dim <= 22:
            # 低维状态 ≤ 32 = Δyaw 1 + 根倾斜 6 + 投影 k + 航向系速度 3
            # → k ≤ 22（修改 04 改动 B 维度上限）。
            raise ValueError("ar2_fb_dim must be in [1, 22] "
                             "(low-dim feedback total ≤ 32)")
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
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 4 if (self.root_sd and not self.root_sd_t1) else 3))
        if self.root_sd_t1:
            # T1（2026-11-06）：根头 3 通道 [s, φ, vy]；φ 末层零初始化
            # ⇒ init 时方向 = 同帧根 6D 前向轴（不消耗 RNG，其他通道不变）。
            with torch.no_grad():
                self.root_head[-1].weight[1].zero_()
                self.root_head[-1].bias[1].zero_()
        if self.ar or self.ar2:
            # 帧级 AR（2026-11-01，V7 方案 §11）：反馈 = 上一帧完整输出，
            # 共享投影加性注入；零初始化保留 init 逐位 = Cv5 契约（RNG
            # 存/复构造）。ar（V7_Cv5AR）= [姿态 144 + 根速度 3]；ar2
            # （V7_Cv5AR2）= [Δyaw 1 + tilt/body 144 + v_head 3]。ar_fb
            # 仅 ar 用（ARM 对照格：只留因果掩码、无反馈支路）。
            _rng = torch.get_rng_state()
            if ar2 and ar2_fb:
                # 修改 04 改动 B（2026-11-03）：ar2_lowdim 时反馈 = 低维
                # 相对状态 [Δyaw 1 + 根倾斜 6 + body 投影 k + v_head 3]
                # （总维 ≤ 32，k = ar2_fb_dim ≤ 22）；body 138 → k 的投影
                # 零初始化（init 贡献严格为零 → 保留 AR2 零初始化契约）。
                # 外部接口仍 148 维（dataset/eval 协议不感知降维）。
                fb_in = (10 + int(ar2_fb_dim)) if ar2_lowdim else 148
                self.ar2_fb = nn.Linear(fb_in, dim)
                nn.init.zeros_(self.ar2_fb.weight)
                nn.init.zeros_(self.ar2_fb.bias)
                if ar2_lowdim:
                    self.ar2_body_proj = nn.Linear(138, int(ar2_fb_dim))
                    nn.init.zeros_(self.ar2_body_proj.weight)
                    nn.init.zeros_(self.ar2_body_proj.bias)
            elif ar and ar_fb:
                self.ar_fb = nn.Linear(147, dim)
                nn.init.zeros_(self.ar_fb.weight)
                nn.init.zeros_(self.ar_fb.bias)
            if ar2:
                # Δyaw 头：首层正常初始化、末层零初始化（init 输出 0 →
                # 走直，探针口径「转向不可观测，预期 ≈0」；双层全零会
                # 死梯度——末层 W=0 截断首层梯度链，头退化成常数偏置）。
                self.dyaw_head = nn.Sequential(
                    nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
                nn.init.zeros_(self.dyaw_head[2].weight)
                nn.init.zeros_(self.dyaw_head[2].bias)
            torch.set_rng_state(_rng)
            # 因果掩码：(tw*10) 序列按槽 t*10+p 展平，位置 i 只 attend
            # t_j <= t_i（同帧 10 槽互见）。float 加性掩码语义：允许=0、
            # 屏蔽=-inf（PyTorch MHA 将 attn_mask 加到分数上）。
            _t = torch.arange(tw).repeat_interleave(1 + self.n_parts)
            # (i,j) = t_i >= t_j（query i 只 attend 过去与同帧）。
            self.register_buffer("causal_mask",
                                 torch.where(_t[:, None] >= _t[None, :],
                                             torch.zeros(()),
                                             torch.tensor(float("-inf"))),
                                 persistent=True)
        if self.ctx5:
            # 块自回归（2026-11-01）：全姿态硬填——根速度槽位读 [vel3+flag]、
            # 各部位槽位读其关节的 6D 历史（+flag）。零初始化保留
            # init 逐位 = Cv5 契约（贡献严格为零）；RNG 存/复构造（nn.Linear
            # 默认初始化消耗 RNG——不存复会扰动后续共享参数的初始化序列）。
            _rng = torch.get_rng_state()
            self.ctx_root_emb = nn.Linear(4, dim)
            nn.init.zeros_(self.ctx_root_emb.weight)
            nn.init.zeros_(self.ctx_root_emb.bias)
            self.ctx_part_embs = nn.ModuleList()
            for _j in self.part_joints:
                _m = nn.Linear(6 * len(_j) + 1, dim)
                nn.init.zeros_(_m.weight)
                nn.init.zeros_(_m.bias)
                self.ctx_part_embs.append(_m)
            # pose 按关节序：关节 j 的 6D 在 [6j:6j+6]。
            self.ctx5_part_idx = [torch.tensor(
                [6 * j + k for j in _j for k in range(6)], dtype=torch.long)
                for _j in self.part_joints]
            torch.set_rng_state(_rng)
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std", torch.ones(POSE_DIM))

    def set_stats(self, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> None:
        self.pose_mean = pose_mean.to(self.pose_mean.dtype)
        self.pose_std = pose_std.to(self.pose_std.dtype)

    def forward(self, m_v, m_t, v_present, t_present, gate=None,
                gate_v=None,
                t_only_mask=None, t_only_adapters=None, vt_mask=None,
                vt_adapters=None, z_override=None,
                gate_fn=None, ctx_vec=None, ctx_pose=None, ctx_block=None,
                ctx_emb=None, prev_state=None, causal=False, max_frames=None):
        """→ (pose (B,tw,POSE_DIM), v_hat (B,tw,3), z_root, z_parts)

        ``z_override``（TTA 注入点，2026-10-27）：给定 (B, tw*(1+n_parts), d)
        时替换 decoder 输出（out_parts/root_head 前的部位表示，论文 token_out
        的对应位置；梯度只经冻结的 out_parts/root_head，不回传 decoder/
        temporal/编码器）；默认 None = 原行为不变。

        ``max_frames``（修改 04 改动 A，2026-11-03）：非 None 时只解码
        前 max_frames 帧的前缀（query/时间 PE/因果掩码/门逐帧切片——
        因果掩码保证前缀输出与全窗解码逐位一致），供训练期闭环 rollout
        逐帧解码省算力；默认 None = 原行为不变。
        """
        B = m_v.shape[0]
        frames = self.tw if max_frames is None else int(max_frames)
        if frames > self.tw:
            raise ValueError("max_frames must be ≤ tw")
        q_root = self.root_query[:, :frames].expand(B, -1, -1, -1) \
            + self.embeddings.time_pe[-self.tw:][:frames].to(self.root_query.dtype).view(1, frames, 1, self.dim)
        q_parts = self.part_queries[:, :frames].expand(B, -1, -1, -1) \
            + self.embeddings.time_pe[-self.tw:][:frames].to(self.part_queries.dtype).view(1, frames, 1, self.dim)
        if ctx_block is not None and max_frames is not None:
            raise ValueError("prefix decoding (max_frames) is unsupported "
                             "with ctx_block (ar2 rollout never carries ctx)")
        if ctx_block is not None:
            # v3 硬填式（Level 3）：ctx = 重叠区 nf 帧 × [根 6D + 根速度 3
            # + flag]（评估链 = 上一窗后 nf 帧预测，训练 = GT 前 nf 帧）——
            # 替换式注入前 nf 帧的根速度 query 与根朝向 query 槽位（含时间
            # PE），重叠区与上一窗构造性一致（Step2Motion inpainting 的
            # 回归等价物）。V 在场或 flag=0 的样本不替换（保留学习 query，
            # V/VT 行回归 Cv5 行为）。
            nf = ctx_block.shape[1]
            has = ctx_block[..., -1].mean(dim=1) > 0.5        # (B,)
            allow = has & (v_present < 0.5)
            if allow.any():
                tpe = self.embeddings.time_pe[:nf]
                if self.ctx5:
                    # 块自回归：全姿态硬填——ctx_block = [全 24 关节 6D
                    # (144) + 根速度 3 + flag 1] × nf 帧，全部 10 个 query
                    # 槽位逐部位注入（发射区从完整运动历史出发）。
                    root_feat = ctx_block[..., 144:148]       # vel3 + flag
                    q_root[allow, :nf, 0, :] = (
                        self.ctx_root_emb(root_feat)[allow] + tpe.to(root_feat.dtype))
                    for p_ in range(self.n_parts):
                        idx = self.ctx5_part_idx[p_].to(ctx_block.device)
                        feat = torch.cat([ctx_block[..., idx], ctx_block[..., -1:]],
                                         dim=-1)
                        q_parts[allow, :nf, p_, :] = (
                            self.ctx_part_embs[p_](feat)[allow] + tpe.to(feat.dtype))
                else:
                    emb = ctx_emb(ctx_block)                  # (B, nf, d)
                    q_root[allow, :nf, 0, :] = emb[allow] + tpe.to(emb.dtype)
                    q_parts[allow, :nf, 0, :] = emb[allow] + tpe.to(emb.dtype)
        if ctx_vec is not None:
            # 续写上下文加性注入（(B,d) → 广播到全部帧）：v1 = 根速度
            # query；v2（Level 1）= 根部位（朝向）query 槽位同步注入，
            # 朝向不再只经自注意力间接读取。
            ctx_bias = ctx_vec.to(q_root.dtype).view(B, 1, 1, self.dim)
            q_root = q_root + ctx_bias
            if self.ctx_v2:
                q_parts[:, :, 0:1, :] = (q_parts[:, :, 0:1, :]
                                         + ctx_bias.to(q_parts.dtype))
        if self.ar2 and self.ar2_fb and prev_state is not None:
            # AR2 相对运动反馈（2026-11-01，V7_AR方案 §2）：prev_state =
            # 上一帧 [Δyaw + tilt/body + v_head]，共享投影加性注入（与
            # ar_fb 同模式，零初始化保留 init 逐位 = Cv5 契约）。
            ps = prev_state[:, :frames] if max_frames is not None else prev_state
            if self.ar2_lowdim:
                # 修改 04 改动 B（2026-11-03）：低维相对状态——Δyaw 1 +
                # 根倾斜 6 + 身体低维投影 k（ar2_body_proj 零初始化）+
                # 航向系速度 3；148 维外部布局不变（dataset/eval 不感知）。
                body = self.ar2_body_proj(ps[..., 7:145])      # (B, frames, k)
                ps = torch.cat([ps[..., :7], body, ps[..., 145:]], dim=-1)
            fb = self.ar2_fb(ps.to(q_root.dtype))   # (B, frames, d)
            q_root = q_root + fb.view(B, frames, 1, self.dim)
            q_parts = q_parts + fb.view(B, frames, 1, self.dim)
        if self.ar and self.ar_fb and prev_state is not None:
            # 帧级 AR 反馈（2026-11-01 用户裁定：全条件注入——V/VT/T 都走
            # 因果 + 反馈，共享解码器语义唯一）。q += proj(上一帧输出)。
            fb = self.ar_fb(prev_state.to(q_root.dtype))      # (B, tw, d)
            q_root = q_root + fb.view(B, frames, 1, self.dim)
            q_parts = q_parts + fb.view(B, frames, 1, self.dim)
        q = torch.cat([q_root, q_parts], dim=2)          # (B, frames, 1+9, d)
        q = q.reshape(B, frames * (1 + self.n_parts), self.dim)
        _mask = self.causal_mask if (causal and (self.ar or self.ar2)) else None
        if _mask is not None and frames < self.tw:
            _mask = _mask[:frames * (1 + self.n_parts),
                          :frames * (1 + self.n_parts)]
        if gate is not None and max_frames is not None:
            gate = gate[:, :frames]
        if gate_v is not None and max_frames is not None:
            gate_v = gate_v[:, :frames]
        if gate_fn is not None and max_frames is not None:
            raise ValueError("prefix decoding (max_frames) is unsupported "
                             "with route_conditional gate_fn")
        for layer in self.layers:
            q, gate = layer(q, m_v, m_t, v_present, t_present, frames,
                            gate=gate, gate_fn=gate_fn, attn_mask=_mask,
                            gate_v=gate_v)
            gate_fn = None  # Q：第一层计算的同一组门供后续五层复用（§6）
        if z_override is not None:
            q = z_override  # TTA 注入点（decoder 输出 / out_parts·root_head 前）
        q = q.reshape(B, frames, 1 + self.n_parts, self.dim)
        z_root = q[:, :, 0:1, :]
        z_parts = q[:, :, 1:, :]
        if t_only_adapters is not None:
            if t_only_mask is None:
                raise ValueError("T-only adapters require an explicit availability mask")
            mask = t_only_mask.to(q.dtype).view(B, 1, 1, 1)
            z_root = z_root + mask * t_only_adapters[0](z_root)
            z_parts = z_parts + mask * t_only_adapters[1](z_parts)
        if vt_adapters is not None:
            if vt_mask is None:
                raise ValueError("VT adapters require an explicit availability mask")
            mask = vt_mask.to(q.dtype).view(B, 1, 1, 1)
            z_root = z_root + mask * vt_adapters[0](z_root)
            z_parts = z_parts + mask * vt_adapters[1](z_parts)
        root_out = self.root_head(z_root)[:, :, 0, :]    # (B, tw, 3|4)
        sd_out = None
        if self.root_sd:
            # 修改 07 改动 A（2026-11-04）：三通道组装。轴约定 y-up、
            # forward_axis=+Z → 水平面 = (x, z) = 下标 [0, 2]。softplus
            # 保证速率非负；normalize 的 eps 让静止帧方向模长趋零成为
            # 合法状态（不加前向报错，原始模长另行登记进诊断）。
            s_hat = F.softplus(root_out[..., 0])               # (B, tw)
            if self.root_sd_t1:
                # T1（2026-11-06，T 运动轴 T1）：方向载体 = 同帧根 6D 前向轴
                # + 标量偏角 φ。根头 3 通道 [s, φ, vy]；φ 零初始化 ⇒ init
                # 方向 = 姿态前向轴。前向轴取 POSE_DIM 前 6 位（关节 0 =
                # root）经 rot6d_to_rotmat 的第三列（+Z），投影到 (x,z)。
                phi = root_out[..., 1]                         # (B, tw) rad
                vy_hat = root_out[..., 2]
                root6 = (self.out_parts[0](z_parts[..., 0, :])[..., :6]
                         * self.pose_std[:6] + self.pose_mean[:6])
                fwd = rot6d_to_rotmat(root6)[..., :, 2]        # (B,tw,3) +Z
                fwd_xz = F.normalize(fwd[..., [0, 2]], dim=-1, eps=1e-6)
                c, s = torch.cos(phi), torch.sin(phi)
                d_x = c * fwd_xz[..., 0] - s * fwd_xz[..., 1]
                d_z = s * fwd_xz[..., 0] + c * fwd_xz[..., 1]
                d_hat = torch.stack([d_x, d_z], dim=-1)        # (B,tw,2)
                sd_dir_norm = torch.ones_like(phi)             # 诊断占位
            else:
                dir_raw = root_out[..., 1:3]
                d_hat = F.normalize(dir_raw, dim=-1, eps=1e-6)  # (B, tw, 2)
                vy_hat = root_out[..., 3]                      # (B, tw)
                sd_dir_norm = torch.linalg.vector_norm(dir_raw, dim=-1)
            # v_hat 契约不变：(B, tw, 3) 世界系 m/s（下游 cumsum/锚/FK 全不动）
            v_hat = torch.stack([s_hat * d_hat[..., 0], vy_hat,
                                 s_hat * d_hat[..., 1]], dim=-1)
            sd_out = (s_hat, d_hat, vy_hat, sd_dir_norm)
        else:
            v_hat = root_out                                   # (B, tw, 3)
        dyaw = None
        if self.ar2:
            dyaw = self.dyaw_head(z_root)[:, :, 0, :]    # (B, tw, 1) rad/帧
        pose_parts = torch.cat(
            [self.out_parts[p](z_parts[..., p, :]) for p in range(self.n_parts)],
            dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        if ctx_pose is not None:
            # v2（Level 2）：根朝向输出加 ctx 残差（10→6 MLP，零初始化，
            # init 贡献严格为零——保留 init 逐位 = Cv5 契约）。根 6D 位于
            # POSE_DIM 前 6 位（关节 0 = root，part_place_idx 正典序）。
            pose[..., :6] = pose[..., :6] + ctx_pose.to(pose.dtype).view(B, 1, 6)
        if self.ar2:
            # 根读出强制去航向（方案 §3.3）：输出头无法经 tilt 通道表达航向
            # ——反归一化后的物理旋转上投影（§3.1），前向退化报错不静默。
            root6 = pose[..., :6]
            r_root = rot6d_to_rotmat(root6)
            deyawed = rotmat_to_6d(deyaw_rotmat(r_root))
            if self.abs_vrows:
                # 修改 09：只 T 行去航向；V/VT 行保留绝对朝向（视觉直接观测
                # 到世界航向，无需经 Δψ 二次组装——避免沿运动学链放大）。
                t_rows = (v_present < 0.5).view(-1, 1, 1).to(deyawed.dtype)
                root_out6 = t_rows * deyawed + (1.0 - t_rows) * root6
            else:
                root_out6 = deyawed
            pose = torch.cat([root_out6, pose[..., 6:]], dim=-1)
        # root_sd=True 时在第 6 位追加 sd_out；False 时保持 5 元组，既有
        # 调用方（model.forward / rollout_forward / tests）逐位不变。
        if self.root_sd:
            return pose, v_hat, dyaw, z_root, z_parts, sd_out
        return pose, v_hat, dyaw, z_root, z_parts


class AnySoleModelV7(nn.Module):
    """独立证据 + 共享生成（候选 A/B/C 载体）。

    C 类门控的额外开关（V7_C门控修改方案，2026-10-30）：
      ``route_temporal``：逐帧门（T/TLR）——x_t 不池化，MLP 逐帧作用于
        末维（门参数量与 Cv5 相同，仍六层共享）。
      ``part_routes``：四组门（LR/TLR）——vel/other/left/right，左右门
        = 姿态门深拷贝（初始化等价契约 §5：共享模块随机初始化不受新增
        参数影响；LR 的相同门初始化复现 B，TLR 复现 T）。
      ``gate_pin_legs``：PO（修改 13 D3）——腿/足四部位 query 的门钉成
        1.0（门只作用在 vel/其余姿态上）。零新增参数，只缩门的作用域。
      ``route_conditional``：Q（§6）——保持 TLR 四组逐帧门与 MLP 尺寸，
        门输入改为第一层解码上下文（视觉更新后的 query 均值 + 未乘门的
        T 注意力输出均值），第一层计算、六层复用；无新增参数。
      ``route_router``：竞争路由门（2026-11-07，V8_Cv8）——k = σ(s_v − s_t)
        逐帧逐组（vel/other/left/right → 10 槽），s_v 读 V token、s_t 读
        T token + 分组物理/IMU 特征；解码器 V 支路乘 (1−k)、T 支路乘 k，
        V-only 行 k≡0、T-only 行 k≡1。取代 T 门与 V 门（与 v_gate 互斥）。
    """

    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6,
                 contact_root: bool = False, reliability_route: bool = False,
                 task_routes: bool = False, t_only_adapters: bool = False,
                 route_temporal: bool = False, part_routes: bool = False,
                 route_conditional: bool = False,
                 gate_pin_legs: bool = False,
                 v_gate: bool = False, gate_2sigma: bool = False,
                 unc_gate: bool = False,
                 route_router: bool = False,
                 prev_ctx: bool = False,
                 ctx_v2: bool = False, ctx_vcond: bool = False,
                 ctx3: bool = False, ctx4: bool = False,
                 ctx4b: bool = False, ctx5: bool = False,
                 ar: bool = False, ar_fb: bool = True, ar2: bool = False,
                 ar2_fb: bool = True, ar2_lowdim: bool = False,
                 ar2_fb_dim: int = 16, ar2_regression: bool = False,
                 abs_vrows: bool = False,
                 vt_t_residual: bool = False, root_sd: bool = False,
                 root_sd_t1: bool = False,
                 trans_anchor_head: bool = True,
                 trans_anchor_stopgrad: bool = True,
                 obs_context=None, v_cur: bool = False,
                 t_phys_v2: bool = False):
        super().__init__()
        self.d = d
        self.tw = tw
        self.obs_context = int(obs_context if obs_context is not None else tw)
        # 按模态分窗（2026-11-03，V7_Cv5AR2MSC）：V 流在模型内裁到当前窗 tw，
        # T 流保留 obs_context；T 信任门只池化当前窗。默认关 = 旧格逐字节不变。
        self.v_cur = bool(v_cur)
        if self.obs_context < tw or self.obs_context % tw:
            raise ValueError("obs_context must be a multiple of tw and at least as large")
        if self.obs_context > tw and not (ar2 or self.v_cur):
            raise ValueError("extended observation context requires ar2 or v_cur")
        self.modal = "anysolev2"
        self.v7 = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        # 维度 2（2026-11-05，修改 11 A3）：T_phys 12→20 只改物理量那一段，
        # raw108 的 96 维 T_raw 口径不动；默认 False = Cv5/SD2 逐位不变。
        self.t_phys_v2 = bool(t_phys_v2)
        self.f2_repr = False
        self.contact_root = bool(contact_root)
        self.reliability_route = bool(reliability_route)
        self.task_routes = bool(task_routes)
        self.t_only_adapters = bool(t_only_adapters)
        self.route_temporal = bool(route_temporal)
        self.part_routes = bool(part_routes)
        self.route_conditional = bool(route_conditional)
        # 按可观测性划定门的作用域（修改 13 D3，2026-11-06）：腿/足 query
        # 的 T 信任门钉成 1.0 —— 实测踝足 Δ(VT−V) 在 occ000 就四折同负、
        # 逐受试者 6/6 同负，T→足/腿 这条路上没有可压的害。零新增参数，
        # 只改门的作用域（单旗）；默认关 = Cv5 逐位不变。
        self.gate_pin_legs = bool(gate_pin_legs)
        # 双向信任门（修改 14 §5，2026-11-06）：V 侧镜像门 + T 门 2σ。
        #   v_gate      —— 在 V 交叉注意力输出上加连续门 g_v（2σ，init 1）；
        #   gate_2sigma —— T 门值域由 (0,1) 改成 (0,2)，末层零初始化
        #                  ⇒ init 时 g_t ≡ 1（= 无门 Cv7 的 T 支路）。
        # 两旗默认关 = Cv5 逐位不变；V8_Cv5VG 两旗同开。
        self.v_gate = bool(v_gate)
        self.gate_2sigma = bool(gate_2sigma)
        self.prev_ctx = bool(prev_ctx)
        self.ctx_v2 = bool(ctx_v2)
        self.ctx_vcond = bool(ctx_vcond)
        self.ctx3 = bool(ctx3)
        self.ctx4 = bool(ctx4)
        self.ctx4b = bool(ctx4b)
        self.ctx5 = bool(ctx5)
        self.ar = bool(ar)
        self.ar2 = bool(ar2)
        self.ar_fb = bool(ar_fb)
        self.ar2_fb = bool(ar2_fb)
        self.ar2_lowdim = bool(ar2_lowdim)
        self.ar2_fb_dim = int(ar2_fb_dim)
        # 纯回归收口（2026-11-02，v7_c2ar2r）：训练/评估两侧读本属性
        # （模型属性 = 真值，checkpoint config 只作回退）。
        self.ar2_regression = bool(ar2_regression)
        # 读出按条件分支（修改 09，2026-11-04）：V/VT 行绝对根朝向 + 绝对
        # 世界速度（= Cv5 目标空间），T 行保持 AR2 相对表征 + 外部 ψ。
        # 训练/评估两侧读本属性（模型属性 = 真值，ckpt config 只作回退）。
        self.abs_vrows = bool(abs_vrows)
        if self.abs_vrows and not self.ar2:
            raise ValueError("abs_vrows requires ar2 (它只改 AR2 的条件读出)")
        if self.gate_pin_legs:
            if not self.task_routes:
                raise ValueError("gate_pin_legs requires task_routes "
                                 "(它钉的是 10 槽 query 门里的腿/足槽)")
            if self.route_conditional:
                raise ValueError("gate_pin_legs is incompatible with "
                                 "route_conditional (Q 门走独立出口)")
            # (10,) 常量掩码：1 = 该 query 槽的门钉成 1.0（无新增参数）。
            self.register_buffer("gate_pin",
                                 torch.tensor(_V7_GATE_PIN_LEGS), persistent=False)
        if self.v_gate and not self.reliability_route:
            raise ValueError("v_gate requires tactile reliability routing")
        self.unc_gate = bool(unc_gate)
        if self.unc_gate and not self.v_gate:
            raise ValueError("unc_gate requires v_gate (V 支路门由不确定性专家驱动)")
        if self.v_gate and self.route_conditional:
            raise ValueError("v_gate is incompatible with route_conditional (Q)")
        if self.gate_2sigma and not self.reliability_route:
            raise ValueError("gate_2sigma requires tactile reliability routing")
        self.route_router = bool(route_router)
        if self.route_router:
            if not (self.reliability_route and self.task_routes):
                raise ValueError("route_router requires reliability_route "
                                 "and task_routes (10 槽 query 轴)")
            if self.v_gate:
                raise ValueError("route_router replaces the V gate (k 本身即 V 门)")
            if self.part_routes or self.route_conditional:
                raise ValueError("route_router is incompatible with "
                                 "part_routes / route_conditional")
            if self.gate_pin_legs:
                raise ValueError("route_router is incompatible with gate_pin_legs")
            if self.t_phys_v2:
                raise ValueError("route_router assumes T_phys 12 维（t_phys_v2 未定义分组特征）")
        self.vt_t_residual = bool(vt_t_residual)
        self.root_sd_t1 = bool(root_sd_t1)
        # 根头速率/方向解耦（修改 07 改动 A）：只定义世界系根的 Cv5 线；
        # AR/AR2 的根目标在航向系（v_head_gt/dyaw），语义不同，不允许混用。
        self.root_sd = bool(root_sd)
        # 平移锚头（修改 08，2026-11-04）：V/VT 行旁路读出窗外那个常数
        # 平移（评估仍吃 GT 锚；新头只喂并列自锚列）。默认关 = 旧格不变。
        self.trans_anchor_head = bool(trans_anchor_head)
        self.trans_anchor_stopgrad = bool(trans_anchor_stopgrad)
        if self.root_sd and (self.ar or self.ar2):
            raise ValueError("root_sd is a world-frame root head; "
                             "ar/ar2 use their own root targets")
        if self.ar2 and self.ar:
            raise ValueError("ar2 and ar are mutually exclusive")
        if self.ar2 and (self.prev_ctx or self.ctx3 or self.ctx4
                         or self.ctx4b or self.ctx5):
            raise ValueError("ar2 replaces ctx: prev_ctx/ctx3/4/5 must be off")
        if self.task_routes and not self.reliability_route:
            raise ValueError("Task routes require tactile reliability routing")
        if self.route_temporal and not self.reliability_route:
            raise ValueError("Temporal routing requires tactile reliability routing")
        if self.part_routes and not self.task_routes:
            raise ValueError("Part routing requires task routes")
        if self.route_conditional and not (self.reliability_route
                                           and self.task_routes
                                           and self.route_temporal
                                           and self.part_routes):
            raise ValueError("Conditional routing (Q) requires TLR's full gate structure")
        if self.ctx_v2 and not self.prev_ctx:
            raise ValueError("ctx_v2 requires prev_ctx")
        if self.ctx3 and not self.prev_ctx:
            raise ValueError("ctx3 requires prev_ctx")
        if self.ctx4 and not self.ctx3:
            raise ValueError("ctx4 requires ctx3")
        if self.ctx4b and not self.ctx3:
            raise ValueError("ctx4b requires ctx3")
        if self.ctx4b and self.ctx4:
            raise ValueError("ctx4b (uniform residual) replaces ctx4 "
                             "(dual-space); mutually exclusive")
        if self.ctx5 and not self.ctx3:
            raise ValueError("ctx5 requires ctx3")
        if (self.ctx4 or self.ctx4b) and self.ctx5:
            raise ValueError("ctx4/ctx4b and ctx5 are mutually exclusive "
                             "(block layouts differ)")
        if self.ar and (self.prev_ctx or self.ctx3):
            raise ValueError("ar mode replaces ctx: prev_ctx/ctx3 must be off")
        self.embeddings = SharedEmbeddings(
            dim=d, tw=self.obs_context, time_offset=tw - self.obs_context)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv", t_phys_v2=self.t_phys_v2)
        self.v_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        self.t_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        # 运动解码（挂名为 pose_head：train.py 的姿态统计拟合按名调用）。
        self.pose_head = V7MotionHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, route=reliability_route, ctx_v2=bool(ctx_v2),
            ctx5=bool(ctx5), ar=bool(ar), ar_fb=bool(ar_fb), ar2=bool(ar2),
            ar2_fb=bool(ar2_fb), ar2_lowdim=bool(ar2_lowdim),
            ar2_fb_dim=int(ar2_fb_dim), root_sd=bool(root_sd),
            root_sd_t1=bool(root_sd_t1), abs_vrows=bool(abs_vrows))
        self.contact_head = V41ContactHead(dim=d)
        if ar2:
            # 视觉航向观测分支（方案 §2/§4，仅 V/VT 生效；T 行 v_present=0
            # 不消费）：vis_yaw 零初始化（0 rad）；vis_gain 零权重 + 负偏置
            # 初始化 → 校正 ≈ 0，训练后经 L_yawcorr 学到可靠性（推理只读
            # 运行时证据，不读 GT）。RNG 存/复：新组件不扰动共享参数序列。
            _rng = torch.get_rng_state()
            self.vis_yaw_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.vis_yaw_head[2].weight)
            nn.init.zeros_(self.vis_yaw_head[2].bias)
            self.vis_gain_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.vis_gain_head[2].weight)
            nn.init.constant_(self.vis_gain_head[2].bias, -4.0)
            torch.set_rng_state(_rng)
        if reliability_route:
            # C：T 信任门（端到端学习，v1 无显式质量标签——设计思路 §3
            # 候选 C 的第一版）。输入 = V/T 编码 token（窗口均值或逐帧，
            # 由 route_temporal 决定）。task_routes 下本 MLP = 姿态门。
            self.route_mlp = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            if self.gate_2sigma:
                # 修改 14 §5.2：T 门值域 (0,2)，末层零初始化 ⇒ g_t ≡ 1
                # （init 逐位 = 无门 Cv7 的 T 支路）。零初始化不消耗 RNG。
                nn.init.zeros_(self.route_mlp[-1].weight)
                nn.init.zeros_(self.route_mlp[-1].bias)
            if self.v_gate:
                # 修改 14 §5：V 信任门——与 T 门同输入、同粒度、同展开；
                # 末层零初始化 ⇒ g_v = 2σ(0) = 1（init 逐位 = Cv5）。
                # fork_rng：新模块的构造不扰动共享参数的 RNG 序列。
                with torch.random.fork_rng(devices=[]):
                    self.route_v_mlp = nn.Sequential(
                        nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
                    nn.init.zeros_(self.route_v_mlp[-1].weight)
                    nn.init.zeros_(self.route_v_mlp[-1].bias)
                if self.unc_gate:
                    # 修改 15（2026-11-07）：不确定性专家驱动 V 门。
                    # unc_expert = MLP(窗均值 v_tok → û)，û 的监督量 =
                    # IG（接触可预测性，训练期由冻结接触头 + 接触 GT 算），
                    # 运行时零 GT；g_v = σ(unc_a·û + unc_b)，unc_a/unc_b
                    # 为逐折拟合的缓冲（随 ckpt 落盘）。与 route_v_mlp
                    # 并存（后者闲置，保持结构兼容）。
                    with torch.random.fork_rng(devices=[]):
                        self.unc_expert = nn.Sequential(
                            nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
                    self.register_buffer("unc_a", torch.tensor(32.0))
                    self.register_buffer("unc_b", torch.tensor(0.0))
            if self.route_router:
                # 竞争路由门（2026-11-07，V8_Cv8）：k = σ(s_v − s_t)，逐帧
                # 逐组（vel/other/left/right → 10 槽）。s_v 读 V token；
                # s_t 读 T token + 分组物理/IMU 特征（vel/other 26 维、
                # left/right 13 维，过 LayerNorm）。init：s_v≡0、s_t≡+4
                # ⇒ k≈0.018（VT 行 init ≈ 纯 V，T 支路须赢取信任；
                # T-only 行由掩码强制 k≡1）。fork_rng 同上。
                with torch.random.fork_rng(devices=[]):
                    self.router_v_mlps = nn.ModuleList([
                        nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                      nn.Linear(d, 1))
                        for _ in range(4)])
                    self.router_t_mlps = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(d + _ROUTER_FEAT_DIMS[g], d),
                            nn.GELU(), nn.Linear(d, 1))
                        for g in range(4)])
                    self.router_feat_norms = nn.ModuleList([
                        nn.LayerNorm(_ROUTER_FEAT_DIMS[g]) for g in range(4)])
                    for _mlp in self.router_v_mlps:
                        nn.init.zeros_(_mlp[-1].weight)
                        nn.init.zeros_(_mlp[-1].bias)
                    for _mlp in self.router_t_mlps:
                        nn.init.zeros_(_mlp[-1].weight)
                        nn.init.constant_(_mlp[-1].bias, 4.0)
        if self.prev_ctx:
            # 续写上下文（2026-11-01，V7_续写协议方案 §1-§2）：窗首帧根状态
            # [根朝向 6D + 根速度 3 + 有效标志 1]（评估链 = 上一窗预测在其
            # 内部帧 half 处）→ 加性注入根速度 query（q_root += ctx_vec；
            # 根朝向 query 经解码器自注意力读取）。零初始化保证 init 时
            # prev_ctx 贡献为零、逐位复现 Cv5。
            self.ctx_emb = nn.Sequential(
                nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.ctx_emb[-1].weight)
            nn.init.zeros_(self.ctx_emb[-1].bias)
            if self.ctx_v2:
                self.ctx_pose_emb = nn.Sequential(
                    nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 6))
                nn.init.zeros_(self.ctx_pose_emb[-1].weight)
                nn.init.zeros_(self.ctx_pose_emb[-1].bias)
        if self.ctx4b:
            # v4-b（2026-11-01）：V/VT 行锚估计头——m_v 窗口均值 → 6D 偏移，
            # 加恒等 6D 后 GS → R_a（出口组合用）。零初始化 → init 锚 = 恒等
            # （逐位 Cv5 契约）；RNG 存/复构造。
            _rng = torch.get_rng_state()
            self.anchor_off = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))
            nn.init.zeros_(self.anchor_off[-1].weight)
            nn.init.zeros_(self.anchor_off[-1].bias)
            torch.set_rng_state(_rng)
            self.register_buffer(
                "identity_6d", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        if self.trans_anchor_head:
            # 修改 08：V/VT 行平移锚头（模型改动 A）。m_v 窗口均值 → 归一化
            # 残差 → 绝对锚（米）。零初始化 ⇒ init 锚 = anchor_mean（最佳常数
            # 地板，与 ctx4b「恒等 6D 起跳」同角色）；RNG 存/复构造保证
            # trans_anchor_head=False 时与 Cv5 共享参数 init 逐位相同。
            _rng = torch.get_rng_state()
            self.anchor_trans_off = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
            nn.init.zeros_(self.anchor_trans_off[-1].weight)
            nn.init.zeros_(self.anchor_trans_off[-1].bias)
            torch.set_rng_state(_rng)
            # 逐轴统计（三轴量程差 5 倍，不归一化则等权 MSE 被 z 独占）。
            self.register_buffer("anchor_mean", torch.zeros(3))
            self.register_buffer("anchor_scale", torch.ones(3))
            # provenance：联合训练 vs 事后拟合（修改 10 改动 A）。
            self.trans_anchor_source = "joint"
        # Add private parameters after shared initialization. V2 initially
        # reproduces C's single gate, then learns independent root routing.
        if self.task_routes:
            self.route_root_mlp = copy.deepcopy(self.route_mlp)
        if self.part_routes:
            # 左右门 = 姿态门深拷贝再独立学习（初始化等价契约）。
            self.route_left_mlp = copy.deepcopy(self.route_mlp)
            self.route_right_mlp = copy.deepcopy(self.route_mlp)
        if self.t_only_adapters:
            with torch.random.fork_rng(devices=[]):
                self.t_only_root_adapter = self._make_t_only_adapter(d)
                self.t_only_pose_adapter = self._make_t_only_adapter(d)
        if self.vt_t_residual:
            # VT 零初始化触觉残差（2026-11-02，V7_Cv5AR2RT）：仅 V&T 同现行
            # 生效（vt_mask）；zero-init 保证 init 时 VT ≡ V 主路径输出
            # （RNG fork 不扰动共享参数序列）。
            with torch.random.fork_rng(devices=[]):
                self.vt_root_adapter = self._make_t_only_adapter(d)
                self.vt_pose_adapter = self._make_t_only_adapter(d)

    def attach_trans_anchor_head(self, state_dict, mean, scale,
                                 source: str = "post_hoc") -> None:
        """修改 10 改动 A：把事后拟合的平移锚头挂到冻结 ckpt 上。

        与联合训练版同构（Linear(d,d)+GELU+Linear(d,3)），输入仍是
        ``m_v.mean(1).detach()``；差别只在 ``trans_anchor_source`` 的
        provenance（`post_hoc` vs `joint`）。评估侧由 eval.py 在
        ckpt 旁发现 ``anchor_head.pt`` 时调用。
        """
        device = next(self.parameters()).device
        head = nn.Sequential(
            nn.Linear(self.d, self.d), nn.GELU(), nn.Linear(self.d, 3))
        head.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        self.anchor_trans_off = head.to(device)
        if "anchor_mean" not in dict(self.named_buffers()):
            self.register_buffer("anchor_mean", torch.zeros(3))
            self.register_buffer("anchor_scale", torch.ones(3))
        self.anchor_mean = mean.to(device=device, dtype=self.anchor_mean.dtype)
        self.anchor_scale = scale.to(device=device, dtype=self.anchor_scale.dtype)
        self.trans_anchor_head = True
        self.trans_anchor_stopgrad = True
        self.trans_anchor_source = str(source)

    def set_anchor_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """修改 08 改动 A2：平移锚的逐轴统计（与 set_stats 同款签名）。

        ``train.py`` 在首步前用一遍训练 loader 拟合 ``batch["trans_anchor"]``
        的均值/标准差（std 下限 1e-2 m）后冻结进 ckpt。
        """
        self.anchor_mean = mean.to(self.anchor_mean.dtype)
        self.anchor_scale = scale.to(self.anchor_scale.dtype)

    @staticmethod
    def _make_t_only_adapter(dim):
        adapter = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, max(dim // 4, 1)), nn.GELU(),
            nn.Linear(max(dim // 4, 1), dim))
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def _gate_input(self, v_tok, t_tok):
        """T 门与 V 门共用的输入（修改 14 §5：同输入、同粒度）。"""
        if self.v_cur:
            t_tok = t_tok[:, -v_tok.shape[1]:]   # 门只池化当前窗（V 已裁到 tw）
        x = torch.cat([v_tok, t_tok], dim=-1)               # (B, tw, 2d)
        if not self.route_temporal:
            x = x.mean(dim=1)                               # (B, 2d)
        return x

    def _gate_act(self, z):
        # 修改 14 §5.2：T 门值域 (0,2)（gate_2sigma）或 Cv5 的 (0,1)。
        out = torch.sigmoid(z)
        return 2.0 * out if self.gate_2sigma else out

    def _route_gate(self, v_tok, t_tok):
        # C 类 T 信任门（V7_C门控修改方案 §2/§4/§5）：
        #   x_t = cat(v_tok_t, t_tok_t)。窗口级（B 基线/LR）= 时间均值池化
        #   后每训练窗口一组门；逐帧（T/TLR）= 每帧一组门（MLP 作用于末维，
        #   门参数量与 Cv5 相同）。组 = vel（根速度 query）/ other（其余
        #   姿态）/ left（l_leg、l_foot）/ right（r_leg、r_foot）；无
        #   part_routes 时合为单一姿态门（T）。
        x = self._gate_input(v_tok, t_tok)
        if self.part_routes:
            g_vel = self._gate_act(self.route_root_mlp(x))   # (B, tw|1, 1)
            g_other = self._gate_act(self.route_mlp(x))
            g_left = self._gate_act(self.route_left_mlp(x))
            g_right = self._gate_act(self.route_right_mlp(x))
            gates = torch.cat([g_vel, g_other, g_left, g_right], dim=-1)
            gates = gates[..., _V7_QUERY_GATE_IDX]          # (B, tw|1, 10)
            return self._expand_gate(gates)                 # (B, tw, 10, 1)
        if self.task_routes:
            root = self._gate_act(self.route_root_mlp(x))
            pose = self._gate_act(self.route_mlp(x))
            if self.route_temporal:
                gates = torch.cat(
                    [root, pose.expand(-1, -1, self.pose_head.n_parts)], dim=-1)
            else:
                gates = torch.cat(
                    [root, pose.expand(-1, self.pose_head.n_parts)], dim=-1)
            return self._expand_gate(gates)
        g = self._gate_act(self.route_mlp(x))
        if self.route_temporal:
            return g.unsqueeze(-1)                          # (B, tw, 1)
        return g.view(-1, 1, 1).expand(-1, self.tw, 1)      # (B, tw, 1)

    def _route_v_gate(self, v_tok, t_tok, t_present=None):
        """修改 14 §5：V 侧信任门（单标量，值域 (0,1]，init 逐位 = 1）。

        与 T 门同输入 / 同粒度；V 只有一条支路，所以不分组。门乘在
        `v_out` 上：g_v → 0 让 U@VT 连续插值到 U@T。

        ``t_present``（修改 15）：V 门只应在双模态在场时仲裁；V-only 行
        用 ``1 + (g_v−1)·t_present`` 把门钉回 1.0，U@V 行逐位不变。

        修改 15 校准（2026-11-07）：值域改为 (0,1]（cap=1）。g>1 的
        「过信放大」区是模型训练分布之外（放大 V 特征无监督证据），
        margin 训练直接把健康窗顶到 cap；1 = 完全信任（不动作）。
        """
        if self.unc_gate:
            u = self.unc_expert(v_tok.mean(dim=1))
            g_v = torch.sigmoid(self.unc_a * u + self.unc_b)
        else:
            x = self._gate_input(v_tok, t_tok)
            g_v = (2.0 * torch.sigmoid(self.route_v_mlp(x))).clamp(max=1.0)
        if t_present is not None:
            g_v = 1.0 + (g_v - 1.0) * t_present.view(-1, 1)
        return self._expand_gate(g_v)

    def _route_router_gate(self, v_tok, t_tok, t_phys, t_s2m,
                           v_present, t_present):
        """竞争路由门（2026-11-07，V8_Cv8）：k = σ(s_v − s_t)，逐帧逐组。

        s_v_g = MLP_v_g(v_tok)——V 信任分（V-only 内容自评）；s_t_g =
        MLP_t_g(cat(t_tok, feat_g))——T 信任分，feat_g 分组物理特征：
        vel/other = 双足 T_phys 12（cop/force/envelope/spatial/temporal）
        + T_s2m IMU 12（左 16:22 / 右 41:47，世界系去重力 acc + 局部系
        gyro）+ 双足接触二值；left/right = 单足对应 13 维。特征过
        LayerNorm（混合尺度）。硬掩码：V-only 行 k≡0、T-only 行 k≡1、
        VT 行 k（逐样本）。返回 (B, tw, 10, 1)，解码器 V 支路乘 (1−k)、
        T 支路乘 k。t_phys/t_s2m 为 None 时对应特征置零兜底。
        """
        b, tw, d = v_tok.shape
        if t_phys is None:
            t_phys = v_tok.new_zeros(b, tw, 12)
        if t_s2m is None:
            t_s2m = v_tok.new_zeros(b, tw, 50)
        phys_l, phys_r = t_phys[..., :6], t_phys[..., 6:12]
        imu_l, imu_r = t_s2m[..., 16:22], t_s2m[..., 41:47]
        contact_l = (phys_l[..., 2:3] > 0.05).to(v_tok.dtype)
        contact_r = (phys_r[..., 2:3] > 0.05).to(v_tok.dtype)
        both = torch.cat([phys_l, phys_r, imu_l, imu_r,
                          contact_l, contact_r], dim=-1)
        feats = (both, both,
                 torch.cat([phys_l, imu_l, contact_l], dim=-1),
                 torch.cat([phys_r, imu_r, contact_r], dim=-1))
        ks = []
        for g in range(4):
            s_v = self.router_v_mlps[g](v_tok)                    # (B,tw,1)
            s_t = self.router_t_mlps[g](
                torch.cat([t_tok, self.router_feat_norms[g](feats[g])],
                          dim=-1))                                # (B,tw,1)
            ks.append(torch.sigmoid(s_v - s_t))
        k = torch.cat(ks, dim=-1)                                 # (B,tw,4)
        k = k[..., _V7_QUERY_GATE_IDX]                            # (B,tw,10)
        k = (k * v_present.view(-1, 1, 1)
             + (1.0 - v_present.view(-1, 1, 1))) * t_present.view(-1, 1, 1)
        return k.unsqueeze(-1)                                    # (B,tw,10,1)

    def _expand_gate(self, gates):
        # (B, tw, 10)（逐帧）或 (B, 10)（窗口）→ (B, tw, 10, 1)。
        if self.gate_pin_legs and gates.shape[-1] == len(_V7_GATE_PIN_LEGS):
            # 修改 13 D3：腿/足槽的门钉成 1.0（T→足/腿 不设门），
            # 其余槽照常由 MLP 给。pin 是常量掩码，梯度只流未钉的槽。
            # V 门是单标量，槽数 != 10，不走这条（修改 14 §5）。
            pin = self.gate_pin.to(dtype=gates.dtype, device=gates.device)
            gates = gates * (1.0 - pin) + pin
        if gates.dim() == 3:
            return gates.unsqueeze(-1)
        return gates[:, None, :, None].expand(-1, self.tw, -1, -1)

    def _route_gate_conditional(self, h, a_t):
        # Q 条件输入（V7_C门控修改方案 §6）：h = 第一层 query 自注意力 +
        # V 交叉注意力后的表示；a_t = 尚未乘门的 T 交叉注意力输出，均为
        # (B, tw*(1+n_parts), d)。按四组在同一帧内平均 query：
        #   u_t,k = cat(mean_query(h_t,k), mean_query(a_T_t,k))  (B, tw, 2d)
        #   g_t,k = sigmoid(MLP_k(u_t,k)) → 展开 (B, tw, 10, 1)
        # 梯度正常经 h / a_T 回传（不 detach）；缺失模态仍由存在掩码硬关闭。
        b = h.shape[0]
        nq = 1 + self.pose_head.n_parts
        h = h.view(b, self.tw, nq, self.d)
        a = a_t.view(b, self.tw, nq, self.d)
        idx = h.new_tensor(_V7_QUERY_GATE_IDX, dtype=torch.long)
        onehot = F.one_hot(idx, num_classes=len(_V7_GATE_GROUPS)).to(h.dtype)
        counts = onehot.sum(0).clamp(min=1)
        mean_h = torch.einsum("btqd,qk->btkd", h, onehot) / counts.view(1, 1, -1, 1)
        mean_a = torch.einsum("btqd,qk->btkd", a, onehot) / counts.view(1, 1, -1, 1)
        u = torch.cat([mean_h, mean_a], dim=-1)          # (B, tw, 4, 2d)
        g_vel = self._gate_act(self.route_root_mlp(u[..., 0, :]))
        g_other = self._gate_act(self.route_mlp(u[..., 1, :]))
        g_left = self._gate_act(self.route_left_mlp(u[..., 2, :]))
        g_right = self._gate_act(self.route_right_mlp(u[..., 3, :]))
        gates = torch.cat([g_vel, g_other, g_left, g_right], dim=-1)  # (B, tw, 4)
        gates = gates[..., _V7_QUERY_GATE_IDX]           # (B, tw, 10)
        return gates.unsqueeze(-1)                       # (B, tw, 10, 1)

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None,
                v_present=None, t_present=None, z_override=None,
                gate_override=None, prev_ctx=None, prev_state=None,
                causal=False, obs_history=None):
        """三条件 = 同一前向的三种输入组合。存在掩码默认由 config_id 决定
        （与输入置零同一决定，设计思路 §4）；v_present/t_present 允许调用
        侧覆盖（诊断/干预用）；gate_override（§7 干预诊断钩子）非 None 时
        替换解码器门（仅诊断脚本用，训练/推理不读 GT）。"""
        if config_id is None:
            config_id = torch.zeros(V_feat.shape[0], dtype=torch.long,
                                    device=V_feat.device)
        if v_present is None:
            v_present = (config_id != CONFIG_T).to(V_feat.dtype)
        if t_present is None:
            t_present = (config_id != CONFIG_V).to(V_feat.dtype)
        if obs_history is not None:
            if not (self.ar2 or self.v_cur) or len(obs_history) != V_feat.shape[0]:
                raise ValueError("obs_history requires one AR2 or split-window (v_cur) history per sample")
            groups = {}
            for i, history in enumerate(obs_history):
                if set(history) != {"V_feat", "T_raw", "T_phys"}:
                    raise ValueError("observation history must contain only V_feat, T_raw and T_phys")
                lengths = {value.shape[0] for value in history.values()}
                if len(lengths) != 1:
                    raise ValueError("observation history channels have different lengths")
                length = lengths.pop()
                if length + self.tw > self.obs_context:
                    raise ValueError("observation history exceeds obs_context")
                groups.setdefault(length, []).append(i)
            outputs, order = [], []
            for length, indices in groups.items():
                idx = torch.tensor(indices, device=V_feat.device)

                def select(value):
                    return value.index_select(0, idx) if value is not None else None

                streams = []
                for key, current in (("V_feat", V_feat), ("T_raw", T_raw),
                                     ("T_phys", T_phys)):
                    past = torch.stack([obs_history[i][key] for i in indices]).to(current)
                    streams.append(torch.cat([past, select(current)], dim=1))

                def extend_mask(value):
                    value = select(value)
                    if value is None:
                        return None
                    return torch.cat([value.new_zeros(len(indices), length), value], dim=1)

                outputs.append(self.forward(
                    *streams, config_id=select(config_id),
                    subject_ids=([subject_ids[i] for i in indices] if subject_ids is not None else None),
                    T_s2m=select(T_s2m), V_hmr=select(V_hmr),
                    mask_v=extend_mask(mask_v), mask_t=extend_mask(mask_t),
                    v_present=select(v_present), t_present=select(t_present),
                    z_override=select(z_override), gate_override=select(gate_override),
                    prev_ctx=select(prev_ctx), prev_state=select(prev_state), causal=causal))
                order.extend(indices)
            restore = torch.tensor(order, device=V_feat.device).argsort()
            return {key: (torch.cat([out[key] for out in outputs], dim=0).index_select(0, restore)
                          if outputs[0][key] is not None else None)
                    for key in outputs[0]}
        if not self.tw <= V_feat.shape[1] <= self.obs_context:
            raise ValueError("observation length is outside [tw, obs_context]")
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        if self.v_cur:
            V_feat = V_feat[:, -self.tw:]
            mask_v = mask_v[:, -self.tw:] if mask_v is not None else None
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        m_v = self.v_temporal(v_tok)
        m_t = self.t_temporal(t_tok)
        anchor_rot = None
        if self.ctx4b:
            # v4-b：锚估计头（V/VT 行出口组合用；T 行锚 = 链/恒等，头输入
            # 为 V 零化常数——L_anchor 对 T 行掩码，不消费）。
            off = self.anchor_off(m_v.mean(dim=1))                # (B, 6)
            anchor_rot = rot6d_to_rotmat(
                self.identity_6d.to(dtype=m_v.dtype, device=m_v.device) + off)
        anchor_trans = None
        anchor_scale = None
        if self.trans_anchor_head:
            # 修改 08 改动 A3：m_v 窗口均值 → 归一化残差 → 绝对锚（米）。
            # 零税契约：stopgrad=True 时 feat.detach() ⇒
            # ∂L_tanchor/∂θ_主干 ≡ 0（主干每一步与 Cv5 数值相同）。
            # 旁路读出：不进 cumsum，trans_hat 形状/语义一字不动。
            feat = m_v.mean(dim=1)
            if self.trans_anchor_stopgrad:
                feat = feat.detach()
            _a_mean = self.anchor_mean.to(dtype=m_v.dtype, device=m_v.device)
            _a_scale = self.anchor_scale.to(dtype=m_v.dtype, device=m_v.device)
            anchor_trans = _a_mean + _a_scale * self.anchor_trans_off(feat)  # (B,3)
            # 损失侧归一化尺度（逐样本展开，随 out 透传；obs_history 合并
            # 沿 dim 0 拼接，故必须带样本维）。
            anchor_scale = _a_scale.view(1, 3).expand(m_v.shape[0], 3)
        ctx_vec = None
        ctx_pose = None
        ctx_block = None
        if self.prev_ctx and prev_ctx is not None:
            if self.ctx3:
                # 硬填式：prev_ctx = (B, nf, 10) 块，透传（槽位替换在
                # pose_head 内做，含 V 在场门控）。ctx4：块先转锚点系残差
                # （锚 = 块帧 0 根朝向，帧 0 根残差 = 恒等）——模型全程
                # 接触不到绝对航向（续写 v4 方案 §2.3）。
                ctx_block = prev_ctx.to(m_v.dtype)
                if self.ctx4 or self.ctx4b:
                    # 锚 = 块帧 0 根朝向；首窗/复位（flag=0）锚 = 恒等
                    # （与评估 R_a=I 同口径；不依赖零输入 rot6d 的退化行为）。
                    has = ctx_block[..., -1].mean(dim=1) > 0.5        # (B,)
                    R_a = torch.eye(3, device=ctx_block.device,
                                    dtype=ctx_block.dtype).expand(
                        ctx_block.shape[0], 3, 3).clone()
                    R_a[has] = rot6d_to_rotmat(ctx_block[has, 0, :6])
                    RaT = R_a.transpose(-1, -2)
                    res_root = compose_rot6d(RaT, ctx_block[..., :6])   # (B,nf,6)
                    res_vel = torch.einsum("bij,bgj->bgi", RaT,
                                           ctx_block[..., 6:9])         # (B,nf,3)
                    ctx_block = torch.cat([res_root, res_vel,
                                           ctx_block[..., 9:]], dim=-1)
            else:
                if self.ctx_vcond and config_id is not None:
                    # G2（续写 v2 实验矩阵）：V 在场（VT/V 条件）ctx 强制置零
                    # ——训练与评估一致（config_id 两侧都已知），打破「ctx
                    # 压过自身证据」的 oracle 契约在 V 行的表现。
                    keep = (config_id == CONFIG_T).to(prev_ctx.dtype).view(-1, 1)
                    prev_ctx = prev_ctx * keep
                ctx_vec = self.ctx_emb(prev_ctx.to(m_v.dtype))  # (B, d)
                if self.ctx_v2:
                    ctx_pose = self.ctx_pose_emb(prev_ctx.to(m_v.dtype))  # (B, 6)
        if self.route_router:
            gate = (gate_override if gate_override is not None
                    else self._route_router_gate(v_tok, t_tok, T_phys, T_s2m,
                                                 v_present, t_present))
            gate_v = 1.0 - gate
        else:
            if gate_override is not None:
                gate = gate_override   # 干预/诊断钩子（§7 单组关门；推理不读 GT）
            elif self.route_conditional:
                gate = None            # Q：门由第一层解码上下文计算（gate_fn）
            else:
                gate = self._route_gate(v_tok, t_tok) if self.reliability_route else None
            gate_v = (self._route_v_gate(v_tok, t_tok, t_present)
                      if self.v_gate else None)
        if self.route_conditional and gate_override is None:
            captured = {}

            def gate_fn(h, a_t):
                captured["gate"] = self._route_gate_conditional(h, a_t)
                return captured["gate"]
        else:
            gate_fn = None
        adapters = ((self.t_only_root_adapter, self.t_only_pose_adapter)
                    if self.t_only_adapters else None)
        t_only = (~v_present.bool()) & t_present.bool()
        vt_adapters = ((self.vt_root_adapter, self.vt_pose_adapter)
                       if self.vt_t_residual else None)
        vt_both = v_present.bool() & t_present.bool()
        head_out = self.pose_head(
            m_v, m_t, v_present, t_present, gate=gate, gate_v=gate_v,
            gate_fn=gate_fn,
            t_only_mask=t_only, t_only_adapters=adapters,
            vt_mask=vt_both, vt_adapters=vt_adapters,
            z_override=z_override, ctx_vec=ctx_vec, ctx_pose=ctx_pose,
            ctx_block=ctx_block, ctx_emb=(self.ctx_emb if self.ctx3 else None),
            prev_state=prev_state,
            causal=bool(causal) and (self.ar or self.ar2))
        if self.root_sd:
            pose, v_hat, dyaw_hat, z_root, z_parts, sd_out = head_out
            s_hat, d_hat, vy_hat, sd_dir_norm = sd_out
        else:
            pose, v_hat, dyaw_hat, z_root, z_parts = head_out
            s_hat = d_hat = vy_hat = sd_dir_norm = None
        vis_yaw = None
        vis_gain = None
        if self.ar2:
            vis_yaw = self.vis_yaw_head(v_tok[:, -self.tw:])
            vis_gain = torch.sigmoid(self.vis_gain_head(v_tok[:, -self.tw:]))
        if self.route_conditional and gate_override is None:
            gate = captured.get("gate")  # 监控/损失输出用（Q 的门在解码器内计算）
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        contact_logits = self.contact_head(t_tok[:, -self.tw:])
        return {
            "x0_hat": pose,                  # 单一生成器的输出（当前条件）
            "v_hat": v_hat,                  # 根速度
            "trans_hat": trans_hat,          # 根轨迹（cumsum）
            # 修改 07 改动 A：三通道读出（仅 root_sd；旧模型为 None）。
            # v_hat 契约不变，损失侧按 s_hat 是否存在分流。
            "s_hat": s_hat,                  # 水平速率 (B,tw) m/s
            "d_hat": d_hat,                  # 水平单位方向 (B,tw,2) (x,z)
            "vy_hat": vy_hat,                # 垂直速度 (B,tw) m/s
            "sd_dir_norm": sd_dir_norm,      # 方向头原始模长（诊断，无梯度）
            "contact_logits": contact_logits,
            "route_gate": gate,              # C：窗口级 T 信任门（监控/损失）
            "route_v_gate": gate_v,          # 修改 14：V 信任门（无门模型为 None）
            "z": torch.cat([z_root, z_parts], dim=2),  # (B,tw,1+n_parts,d) TTA 叶子
            "F": m_v[:, -self.tw:] + m_t[:, -self.tw:],
            "v_tok": v_tok[:, -self.tw:],
            "t_tok": t_tok[:, -self.tw:],
            "m_v": m_v[:, -self.tw:],
            "m_t": m_t[:, -self.tw:],
            "anchor_rot": anchor_rot,
            # 修改 08 改动 A4：平移锚旁路读出（B,3）+ 损失归一化尺度（B,3）。
            "anchor_trans": anchor_trans,
            "anchor_scale": anchor_scale,
            "dyaw_hat": dyaw_hat,          # AR2：逐帧航向增量 (B,tw,1) rad/帧
            "vis_yaw": vis_yaw,            # AR2：视觉航向观测（V/VT 行）
            "vis_gain": vis_gain,          # AR2：视觉校正增益 ∈ (0,1)
            "z_root": z_root,
            "z_parts": z_parts,
            "pressure_hat": None,
            "vfeat_hat": None,
        }

    def rollout_forward(self, V_feat, T_raw, T_phys, config_id=None,
                        subject_ids=None, T_s2m=None, prev_state=None,
                        rollout_len=1, detach=True, dyaw_base="zero"):
        """修改 04 改动 A（2026-11-03）：训练期闭环多步 rollout 前向。

        帧 0 与 >rollout_len 的尾帧 = GT 反馈 TF 解码（迭代 0 的完整前
        向，与 forward 逐位同源）；帧 1..rollout_len = 逐帧闭环解码：
        第 t 帧反馈 = 第 t−1 帧模型输出组装 [Δyaw_total + tilt/body +
        v_head]（与 eval ar2_session_chunks 的逐帧更新式同构；dyaw_base
        与训练损失侧同一规则）。因果掩码保证第 t 帧输出只依赖 ≤t 的
        反馈槽，前缀解码（max_frames=t+1）与全窗解码逐位一致。返回与
        forward 同构的 out dict，其中 x0_hat/v_hat/dyaw_hat 在帧
        1..rollout_len 为闭环输出、其余为 TF 输出，trans_hat 由合并后
        v_hat 重积分；contact/vis 头与 forward 同源（编码器侧）。

        detach=True（默认）：反馈值截断梯度——只影响优化、不改反馈
        来源（每个闭环解码的图独立，逐帧截断）。

        与 --ar-ss（scheduled sampling）的区别：SS 是逐帧按概率替换
        单帧反馈（两遍式缓存，反馈来源 = GT 条件下的输出）；本改动是
        连续 L 帧真实闭环 rollout（反馈来源 = 自预测条件下的输出，
        误差在段内复合——修改 03 测量靶心 0.53→0.05 的对应项）。
        """
        if config_id is None:
            config_id = torch.zeros(V_feat.shape[0], dtype=torch.long,
                                    device=V_feat.device)
        if prev_state is None:
            raise ValueError("rollout requires batch-carried prev_state")
        if not (self.ar2 and self.ar2_fb):
            raise ValueError("rollout requires ar2 with feedback (ar2_fb=True)")
        if not 1 <= int(rollout_len) < self.tw:
            raise ValueError("rollout_len must be in [1, tw-1]")
        if not self.tw <= V_feat.shape[1] <= self.obs_context:
            raise ValueError("observation length is outside [tw, obs_context]")
        # 编码段与 forward 同源（复制面最小：ar2 前向无 ctx/适配器路径；
        # 两段同步维护纪律见 forward 注释）。
        v_present = (config_id != CONFIG_T).to(V_feat.dtype)
        t_present = (config_id != CONFIG_V).to(V_feat.dtype)
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        if self.v_cur:
            V_feat = V_feat[:, -self.tw:]
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id)
        m_v = self.v_temporal(v_tok)
        m_t = self.t_temporal(t_tok)
        if self.route_router:
            gate = self._route_router_gate(v_tok, t_tok, T_phys, T_s2m,
                                           v_present, t_present)
            gate_v = 1.0 - gate
        else:
            gate = self._route_gate(v_tok, t_tok) if self.reliability_route else None
            gate_v = (self._route_v_gate(v_tok, t_tok, t_present)
                      if self.v_gate else None)
        t_only = (~v_present.bool()) & t_present.bool()
        vt_both = v_present.bool() & t_present.bool()
        adapters = ((self.t_only_root_adapter, self.t_only_pose_adapter)
                    if self.t_only_adapters else None)
        vt_adapters = ((self.vt_root_adapter, self.vt_pose_adapter)
                       if self.vt_t_residual else None)

        def decode(pstate, frames):
            return self.pose_head(
                m_v, m_t, v_present, t_present, gate=gate, gate_v=gate_v,
                t_only_mask=t_only, t_only_adapters=adapters,
                vt_mask=vt_both, vt_adapters=vt_adapters,
                prev_state=pstate, causal=True, max_frames=frames)

        # 迭代 0：GT 反馈 TF 解码（帧 0 与尾帧的损失来源；合并 out 骨架）。
        pose0, v0, dyaw0, z_root, z_parts = decode(prev_state, self.tw)
        history_base = (dyaw_base == "history")
        # 行 1 反馈 = 帧 0 状态（帧 0 输出只依赖行 0 的 batch 携带反馈 →
        # 该状态与 eval 的闭环帧 0 状态同构；行 1 的 GT 携带必须被模型
        # 自身帧 0 状态替换——闭环链起点）。
        x0_fb, v0_fb, d0_fb = pose0[:, 0], v0[:, 0], dyaw0[:, 0, 0]
        if detach:
            x0_fb, v0_fb, d0_fb = x0_fb.detach(), v0_fb.detach(), d0_fb.detach()
        d_total0 = d0_fb + (prev_state[:, 0, 0] if history_base else 0.0)
        prev_filled = torch.cat([
            prev_state[:, :1],
            torch.cat([d_total0[:, None], x0_fb, v0_fb], dim=-1)[:, None, :],
            prev_state[:, 2:]], dim=1)
        roll_pose, roll_v, roll_dyaw = [], [], []
        for t in range(1, int(rollout_len) + 1):
            # 前缀解码帧 0..t（因果掩码 → 与全窗解码在 ≤t 帧逐位一致）；
            # 行 1..t 在进入本迭代前已全部替换为自预测（闭环）。
            pose_t, v_t, dyaw_t, _, _ = decode(prev_filled, t + 1)
            x_t, v_t_, d_t = pose_t[:, t], v_t[:, t], dyaw_t[:, t, 0]
            if detach:
                x_t, v_t_, d_t = x_t.detach(), v_t_.detach(), d_t.detach()
            # 行 t+1 反馈 = 帧 t 状态（eval 更新式：d_total = Δψ̂ + 历史
            # 延续基线；history 基线 = 反馈槽自身携带的 Δyaw_total(t−1)）。
            d_total = d_t + (prev_filled[:, t, 0] if history_base else 0.0)
            if t < int(rollout_len):
                fb_t = torch.cat([d_total[:, None], x_t, v_t_],
                                 dim=-1)[:, None, :]
                prev_filled = torch.cat(
                    [prev_filled[:, :t + 1], fb_t, prev_filled[:, t + 2:]],
                    dim=1)
            roll_pose.append(x_t)
            roll_v.append(v_t_)
            roll_dyaw.append(d_t)
        # 合并：帧 1..rollout_len = 闭环输出，其余 = TF 输出。
        pose = pose0.clone()
        v_hat = v0.clone()
        dyaw_hat = dyaw0.clone()
        if roll_pose:
            pose[:, 1:rollout_len + 1] = torch.stack(roll_pose, dim=1)
            v_hat[:, 1:rollout_len + 1] = torch.stack(roll_v, dim=1)
            dyaw_hat[:, 1:rollout_len + 1, 0] = torch.stack(roll_dyaw, dim=1)
        vis_yaw = self.vis_yaw_head(v_tok[:, -self.tw:])
        vis_gain = torch.sigmoid(self.vis_gain_head(v_tok[:, -self.tw:]))
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        contact_logits = self.contact_head(t_tok[:, -self.tw:])
        return {
            "x0_hat": pose,                  # 帧 1..L 闭环、其余 TF
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            # root_sd 与 ar2 互斥（__init__ 守卫）→ 此路径恒为 None；键集
            # 与 forward 对齐，读取方无需按路径分支。
            "s_hat": None, "d_hat": None, "vy_hat": None, "sd_dir_norm": None,
            "contact_logits": contact_logits,
            "route_gate": gate,
            "route_v_gate": gate_v,
            "z": torch.cat([z_root, z_parts], dim=2),
            "F": m_v[:, -self.tw:] + m_t[:, -self.tw:],
            "v_tok": v_tok[:, -self.tw:],
            "t_tok": t_tok[:, -self.tw:],
            "m_v": m_v[:, -self.tw:],
            "m_t": m_t[:, -self.tw:],
            "anchor_rot": None,
            # 修改 08：rollout 路径不含锚头（锚头只作用于 forward）；键集
            # 与 forward 对齐，读取方无需按路径分支。
            "anchor_trans": None,
            "anchor_scale": None,
            "dyaw_hat": dyaw_hat,
            "vis_yaw": vis_yaw,
            "vis_gain": vis_gain,
            "z_root": z_root,
            "z_parts": z_parts,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
