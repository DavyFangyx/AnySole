"""V4_1A (2026-10-19): frozen E v2.1 carrier + T contact stream + part-token
gated correction branch (position space).

Fresh architecture.  z_note/multimodal_fusion/V4_1方案/V4_1A方案.md.  The one
generator is the frozen E V-stream (286 keys); T mounts only on the four
reachable part-token slots (l/r_foot d_kin=0, l/r_leg d_kin=2 — the upper
parts are structurally deleted, no T path) through a zero-initialized
correction branch that writes ONLY the six lower-limb joint positions
(soft skeleton); the pose output / mesh export stays pure V.

- V 载体（冻结）: v_enc -> v_temporal -> 9-part pose head（144 维 6D）+
  traj head（速度 -> cumsum）。权重来自 E v2.1 生产 ckpt_best（入口按
  carrier_from 装载）；构建时冻结 embeddings + v_enc + v_temporal +
  pose_head + traj_head + prior_head.embeddings（V 流 286 键，与 E
  state-dict 键逐位一致——装载零映射）。sigma 头保留冻结状态（与 E 载体
  键兼容），不读出。
- T 接触流（从零）: foot_conv -> t_tok (B, tw, 256)；接触头 = MLP(256->64->2)
  每脚 logits（f6_soft pos-weighted BCE，air 加权为训练开关）。
- T 先验流（从零）: t_temporal（4 层）-> 9 部位 prior head + prior traj head
  （与载体同构；T2M 行 = 先验 + 修正分支读先验塔部位 token）。
- 修正分支（从零，位置空间）: XAttn 联合读出（1 层多头注意力 nhead=8，
  q = 可达部位 token z_p (B, tw, 4, dim)，kv = 同帧 t_tok + 零参考键，
  注意力 logits 加固定偏置 bias_p = -beta*d_kin(p)）-> 接触门控 a_p
  （MLP(c_T (B,tw,2) -> (B,tw,4)，输出层零初始化；Δz = tanh(a_p) ⊙ XAttn_p）
  -> z'_p = z_p + Δz_p -> 每部位线性位置修正头（零初始化）-> 六关节骨盆
  相对位置残差 Δx（18 维，KNEE_ANKLE_FOOT_JOINTS 顺序）-> SG-7 平滑。
  软骨架评估: x_corr = FK(pose_v) 在六关节处替换为 FK + Δx（V3_3D 同款，
  build_corrected_skeleton）。
- 三条件 = 同一前向的三种读出（config_id 被忽略，E/G 同款）: V2M = V 流
  （不应用修正）、VT2M = V 流 + pos_corr、T2M = 先验流 + pos_corr_prior。
- 契约 1（训练起点）: a_p 头 + 位置修正头零初始化 -> x_corr ≡ FK(pose_v)
  -> VT2M ≡ V2M 逐位；上肢部位结构免疫 = 删边永久保证。

V41PoseHead / V41TrajHead / V41ContactHead / _temporal_encoder 逐字复制自
model_v3e（载体即 E 的 V 流——复制而非 import 以保持独立模型代码边界；
state-dict 键结构一致使 286 键装载零映射）。V41PoseHead.forward 增加
``return_z`` 开关以暴露 decoder 部位 token z（(B, tw, 9, dim)，out_parts
之前槽位 = 本方案挂接点）——forward 签名扩展不影响 state dict。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    D_MODEL,
    FPS,
    KNEE_ANKLE_FOOT_JOINTS,
    PART_JOINTS,
    PART_NAMES,
    POSE_DIM,
    TW,
)

# V-stream parameter prefixes frozen at construction and copied from the E
# v2.1 ckpt_best by the entry (anysole/models/v4_1.py).  The prior tower is
# T 流: only its SharedEmbeddings path (prior_head.embeddings. — the same
# params as embeddings.) is part of the carrier; prior_head.{query,decoder,
# out_parts} + prior_traj + t_temporal + foot_conv + contact head + the
# correction branch all train from zero.  Verified against the E production
# ckpt_best: this prefix set selects exactly 286 state-dict keys.
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

# Reachable part-token slots (PART_NAMES order) + part-level kinematic hops
# to the same-side foot (V4_1A方案 §2.2 d_kin 表).  root/torso/headneck/
# l_arm/r_arm are structurally deleted — they never feed the correction
# branch (no T path by construction, no parameters).
REACH_PART_NAMES = ("l_leg", "r_leg", "l_foot", "r_foot")
REACH_PARTS = tuple(PART_NAMES.index(n) for n in REACH_PART_NAMES)  # (5,6,7,8)
REACH_D_KIN = (2, 2, 0, 0)  # per REACH_PARTS slot


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


class V41PoseHead(nn.Module):
    """[copied from model_v3e, 2026-10-19 — the carrier head]
    Pure single-stream 9-part pose head: part queries decode ONE memory
    stream (no modality views, no gates), per-part unembeds emit the part's
    joints in 6D plus a per-joint log-sigma.  Hard PART_JOINTS partition.
    Sigma heads are frozen at init in V4_1 (E carrier state, not read).

    V4_1 addition: ``forward(mem, return_z=True)`` also returns the decoder
    part tokens z (B, tw, n_parts, dim) — the slot BEFORE out_parts, i.e.
    the correction branch's mount point.  Signature-only change; the state
    dict is key-identical to model_v3e's V3EPoseHead."""

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

    def forward(self, mem, return_z: bool = False):
        """mem: (B, tw, dim).  Returns (pose (B,tw,POSE_DIM), log_sigma), or
        (pose, log_sigma, z (B,tw,n_parts,dim)) with return_z — the z slots
        are the part tokens the V4_1 correction branch mounts on."""
        B = mem.shape[0]
        q = self.query.expand(B, -1, -1, -1).reshape(B, self.tw * self.n_parts, self.dim)
        z = self.decoder(q, mem).reshape(B, self.tw, self.n_parts, self.dim)
        pose_parts = torch.cat(
            [self.out_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        sig_parts = torch.cat(
            [self.sig_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        sig = sig_parts[..., self.part_place_idx]
        if return_z:
            return pose, sig, z
        return pose, sig


class V41TrajHead(nn.Module):
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


class V41ContactHead(nn.Module):
    """[copied from model_v3e] Per-foot contact logits from the tactile
    tokens (f6_soft supervision, air-weighted BCE in training)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class PartXAttn(nn.Module):
    """XAttn 联合读出（V4_1A方案 §2.2 修正分支 1/2）.

    One multi-head attention layer (nhead=8): q = the reachable part tokens
    z_p (B, tw, 4, dim) — the pose-head decoder slots of l_leg/r_leg/
    l_foot/r_foot — kv = the SAME-FRAME tactile token t_tok.  Frame-local
    (masked): a part token attends only to its own frame, so t and v part
    tokens interact inside one layer without a cross-frame path and without
    the frozen-V additive-residual paradigm (V4_1A方案 联合读出规则).

    Kinematic soft mask: the attention logits get the FIXED per-part bias
    bias_p = -beta * d_kin(p) (foot d=0, leg d=2).  With one tactile key per
    frame a per-query-row-constant bias would be softmax shift-invariant (a
    no-op), so each frame carries a ZERO REFERENCE KEY (fixed logit 0, value
    0) and the bias is added to the REAL key's logits:

        out_p = sigmoid(l_p - beta * d_kin(p)) * V

    beta=0 -> no mask (ablation); beta=1 -> leg suppressed ~e^-2; beta=2 ->
    ~e^-4; beta -> inf limits to foot-only mounting.  Zero parameters beyond
    the MHA itself; the bias + frame mask are fixed buffers (persistent=False,
    rebuilt from tw/beta).  The zero reference key is a mechanism realization
    note for review — it is what makes the per-query bias a real magnitude
    gate; 零参数、无 GT 依赖（无锚-几何组件）.
    """

    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dropout=0.1, beta=1.0):
        super().__init__()
        self.dim = dim
        self.tw = tw
        self.n_reach = len(REACH_PARTS)
        self.beta = float(beta)
        self.mha = nn.MultiheadAttention(dim, nhead, dropout=dropout,
                                         batch_first=True)
        d_kin = torch.tensor(REACH_D_KIN, dtype=torch.float32)
        bias = -self.beta * d_kin  # (n_reach,)
        # attn mask (tw*n_reach, 2*tw): rows = q (t, p), cols = [real t_tok_t |
        # zero ref_t].  Same-frame only; bias_p lands on the real-key column.
        mask = torch.full((tw * self.n_reach, 2 * tw), float("-inf"))
        for t in range(tw):
            rows = slice(t * self.n_reach, (t + 1) * self.n_reach)
            mask[rows, 2 * t] = 0.0
            mask[rows, 2 * t + 1] = 0.0
        real_cols = torch.zeros(2 * tw, dtype=torch.bool)
        real_cols[0::2] = True
        # rows are t*4+p (interleaved) — bias.repeat(tw) puts bias[p] on row p
        mask = mask + torch.where(
            real_cols, bias.repeat(tw).unsqueeze(1), 0.0)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z_p, t_tok):
        """z_p: (B, tw, n_reach, dim); t_tok: (B, tw, dim) -> (B, tw, n_reach, dim)."""
        B, tw, n_reach, dim = z_p.shape
        q = z_p.reshape(B, tw * n_reach, dim)
        kv = torch.cat([t_tok, torch.zeros_like(t_tok)], dim=1)  # (B, 2tw, dim)
        out, _ = self.mha(q, kv, kv, attn_mask=self.attn_mask, need_weights=False)
        return out.reshape(B, tw, n_reach, dim)


class ContactGate(nn.Module):
    """接触门控 a_p（V4_1A方案 §2.2 修正分支 2）.

    MLP(contact logits c_T (B,tw,2) -> per-reachable-part scalar (B,tw,4));
    the output layer is zero-initialized, so the gate is exactly 0 at init
    (contract 1).  The tanh is applied by the caller:
    Δz_p = tanh(a_p(c_T)) ⊙ XAttn_p.  Contact evidence gates magnitude only;
    part selection is the kinematic mask + structural deletion (门控规则)."""

    def __init__(self, n_reach: int = 4, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, n_reach))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c_t):
        return self.mlp(c_t)  # (B, tw, n_reach)


class PositionCorrHeads(nn.Module):
    """位置修正头（V4_1A方案 §2.2 修正分支 3）.

    Per-reachable-part linear heads on z'_p -> six-joint pelvis-relative
    position residual Δx (18 dims), in KNEE_ANKLE_FOOT_JOINTS order
    (lk, rk, la, ra, lf, rf — the V3_3D pos_corr convention):
      l_leg slot -> left knee (3); r_leg slot -> right knee (3);
      l_foot slot -> left ankle + left foot (6); r_foot slot -> right (6).
    Weight AND bias zero-initialized: the whole correction branch outputs
    exactly 0 at init -> x_corr ≡ FK(pose_v) (contract 1, V3_3C/D 同款
    结构保证先例)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.knee_l = nn.Linear(dim, 3)
        self.knee_r = nn.Linear(dim, 3)
        self.foot_l = nn.Linear(dim, 6)
        self.foot_r = nn.Linear(dim, 6)
        for m in (self.knee_l, self.knee_r, self.foot_l, self.foot_r):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, zc):
        """zc: (B, tw, 4, dim) in REACH_PARTS order -> (B, tw, 18)."""
        B, tw, _, _ = zc.shape
        out = torch.zeros(B, tw, 6, 3, device=zc.device, dtype=zc.dtype)
        out[:, :, 0] = self.knee_l(zc[:, :, 0])
        out[:, :, 1] = self.knee_r(zc[:, :, 1])
        out[:, :, 2:4] = self.foot_l(zc[:, :, 2]).reshape(B, tw, 2, 3)
        out[:, :, 4:6] = self.foot_r(zc[:, :, 3]).reshape(B, tw, 2, 3)
        return out.reshape(B, tw, 18)


class SG7Smooth(nn.Module):
    """SG-7 平滑（V4_1A方案 §2.2 修正分支 3，G/H 同款防抖动税）.

    Fixed Savitzky-Golay (window 7, order 2) along time applied to the
    correction output.  Coefficient-equivalent to constraints._sg_smooth's
    savgol_filter(7, 2), implemented as a FIXED 1-D conv (zero parameters)
    so the gradient flows through smoothing in training; edge handling is
    reflect padding (savgol's interp default differs on the two edge frames
    only).  Zero-in -> zero-out (contract 1 preserved exactly)."""

    KERNEL = torch.tensor([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", self.KERNEL.view(1, 1, 7),
                             persistent=False)

    def forward(self, x):
        """x: (B, tw, D) -> (B, tw, D); no-op for tw < 5."""
        if x.shape[1] < 5:
            return x
        # reflect pad (3, 3) along time then the fixed kernel (padding=0),
        # applied per output channel (reshape (B*D, 1, tw)); padding_mode
        # kwarg is unavailable on the env's torch build.
        B, T, D = x.shape
        y = F.pad(x.transpose(1, 2), (3, 3), mode="reflect")  # (B, D, T+6)
        y = F.conv1d(y.reshape(B * D, 1, T + 6), self.kernel).reshape(B, D, T)
        return y.transpose(1, 2)


def build_corrected_skeleton(pose, trans, offsets, parents, pos_corr):
    """软骨架（V4_1A方案 §2.2 修正分支 4）: FK(pose) with the six
    KNEE_ANKLE_FOOT_JOINTS positions replaced by FK + Δx (V3_3D 同款).

    pose: (B, tw, 144); trans: (B, tw, 3); offsets: (B, 24, 3) or (24, 3);
    parents: (B, 24) or (24,); pos_corr: (B, tw, 18).
    Returns (B, tw, 24, 3).  The pose output / mesh export stays pure V —
    the corrected skeleton feeds metrics only."""
    from anysole.utils.geometry import fk_pose6d

    kp = fk_pose6d(pose, trans, offsets, parents)
    kp = kp.clone()
    kp[..., KNEE_ANKLE_FOOT_JOINTS, :] = (
        kp[..., KNEE_ANKLE_FOOT_JOINTS, :]
        + pos_corr.reshape(*pos_corr.shape[:2], 6, 3))
    return kp


class AnySoleModelV41(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6,
                 beta=1.0):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # regress machinery in train/eval
        self.v4_1 = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.beta = float(beta)
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
        self.pose_head = V41PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=-3.0)
        self.prior_head = V41PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=3.0)
        self.traj_head = V41TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.prior_traj = V41TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.contact_head = V41ContactHead(dim=d)
        # Correction branch (from zero).  ONE branch instance, two readouts:
        # VT2M reads the pose-head part tokens, T2M reads the prior-head part
        # tokens (V4_1A方案 §2.3) — the same mechanism, different source.
        self.corr_attn = PartXAttn(dim=d, tw=tw, nhead=nhead, dropout=dropout,
                                   beta=beta)
        self.contact_gate = ContactGate(n_reach=len(REACH_PARTS))
        self.corr_heads = PositionCorrHeads(dim=d)
        self.sg7 = SG7Smooth()
        # Carrier rule: the V stream is frozen at construction.  The E
        # ckpt_best weights are loaded by the entry (anysole/models/v4_1.py);
        # freezing here makes the carrier identity load-independent.  The T
        # stream and the correction branch stay trainable.
        for name, param in self.named_parameters():
            if name.startswith(CARRIER_PREFIXES):
                param.requires_grad_(False)
        for head in list(self.pose_head.sig_parts) + list(self.prior_head.sig_parts):
            for param in head.parameters():
                param.requires_grad_(False)

    def _correction(self, z, t_tok, contact_logits):
        """Correction branch on one source's part tokens -> Δx (B, tw, 18).

        z: (B, tw, 9, dim) decoder part tokens (pose head or prior head);
        t_tok: (B, tw, dim); contact_logits: (B, tw, 2).  Structural
        deletion: only the REACH_PARTS slots are sliced out — the upper
        parts have no T path by construction (no parameters, no read)."""
        z_r = z[:, :, REACH_PARTS, :]                 # (B, tw, 4, dim)
        attn = self.corr_attn(z_r, t_tok)             # XAttn joint readout
        a_p = self.contact_gate(contact_logits)       # (B, tw, 4), zero-init
        dz = torch.tanh(a_p).unsqueeze(-1) * attn     # contact-gated
        zc = z_r + dz                                 # z'_p = z_p + Δz_p
        return self.sg7(self.corr_heads(zc))          # (B, tw, 18), SG-7

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None):
        """config_id is accepted for the train/eval call contract and
        deliberately IGNORED — both streams always live, and the three eval
        conditions are three readouts of this one forward (V3_3E protocol;
        the switch acts on the correction branch only).  T stays live under
        every condition (contact head / XAttn always read the real stream).
        """
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        v_tok, t_tok = self.encoders(V_feat, T_tac, cid, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        v_mem = self.v_temporal(v_tok)
        pose_v, sigma_v, z_v = self.pose_head(v_mem, return_z=True)
        v_hat, trans_hat = self.traj_head(v_mem)
        t_mem = self.t_temporal(t_tok)
        pose_t, sigma_t, z_t = self.prior_head(t_mem, return_z=True)
        v_prior, trans_prior = self.prior_traj(t_mem)
        contact_logits = self.contact_head(t_tok)
        pos_corr = self._correction(z_v, t_tok, contact_logits)       # VT2M row
        pos_corr_prior = self._correction(z_t, t_tok, contact_logits)  # T2M row
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
            "pos_corr": pos_corr,
            "pos_corr_prior": pos_corr_prior,
            "F": v_mem,          # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
