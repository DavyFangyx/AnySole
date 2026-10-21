"""V5_1A (2026-10-19): frozen E v2.1 carrier + V/T dual z experts + PoE
synthesis + FiLM lower-limb injection.

Fresh architecture (V5_1A方案.md = the authoritative design).  V5_1A =
E 的回归载体（冻结继承，286 键装载自 E v2.1 ckpt_best）+ 从零训练的
g_V/g_T 双 z 专家（各读出 接触 logit 2 + 相位 (cos,sin) 4 + CoP 4，
全通道 10 维；相位保留 = 用户裁定 2026-10-19，探针未过闸结论留存备查）
+ PoE 闭式合成（连续分量对角高斯精度加权 μ_PoE=Σpμ/Σp；接触分量
伯努利 logit 加权 logit_PoE=Σw·logit，w 可学习非负；p(z) 先验不参数化）
+ FiLM（γ init 1 / β init 0 零初始化，注入 = 9 部位 pose head 下肢四部位
输出特征（部位 Linear 前）+ 根轨迹头输出特征 + T 先验塔同部位；其余部位
直通 = 结构免疫）+ G 配方接触头（air 加权 λ_air=2.0）+ T 先验流
（t_temporal + 9 部位 prior head + prior traj head，T2M 健全性行 ~90mm
量级，G 前科 315.5 防回归）。

三条件 = 同一前向的三种 z 来源读出（z_src 开关；config_id 被忽略，
同 V3_3E 协议）：
  V2M  = Dec(x_V; FiLM(z_V))          z_V   = g_V(V)
  VT2M = Dec(x_V; FiLM(z_PoE))        z_PoE = PoE(g_V(V), g_T(T))
  T2M  = Dec(x_T; FiLM(z_T))          z_T   = g_T(T)（T 先验塔 = null-V
                                            前向，H 口径：T2M 行无 V）

契约 1（结构保证）：FiLM 零初始化 → 训练起点 VT2M ≡ V2M ≡ E v2.1 逐位；
上肢无 FiLM 部件恒等。

V3EPoseHead / V3ETrajHead / V3EContactHead / _temporal_encoder 逐字复制自
model_v3e（载体即 E 的 V 流——复制而非 import 以保持独立模型代码边界；
state-dict 键结构一致使 286 键载体装载零映射）。唯一结构性增补：pose/
traj head 的 forward 接受可选 film 钩子（None = 恒等，E 逐位行为；
钩子不注册参数——头部 state dict 与 E 逐键相同）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.pose_head import _part_place_idx
from anysole.types import (
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    D_MODEL,
    FPS,
    PART_JOINTS,
    POSE_DIM,
    TW,
)

# z 通道结构（V5_1A §2.2，通道证据分级）：
#   接触（每足 1）伯努利 logit | 相位（每足 2）(cos 2πφ, sin 2πφ) 高斯
#   | CoP（每足 2）0-1 网格坐标高斯。全通道 10 维。
# 相位保留裁定（2026-10-19）：探针未过闸但先看完整形态效果，相位不进
# 停用——v1 z 保持全通道 10 维，判据 ③ 干预实验与专家分歧审计负向读。
Z_CONT = 2        # 接触 logits（每足 1）
Z_GAUSS = 8       # 相位 4 + CoP 4（连续分量 μ）
Z_DIM = Z_CONT + Z_GAUSS            # FiLM z 维度 = 10
EXPERT_OUT = Z_CONT + 2 * Z_GAUSS   # 专家输出 = logits 2 + (μ 8, log σ 8) = 18

# FiLM 注入 = 9 部位 pose head 的下肢四部位（部位 Linear 前）。部位索引
# 按 types.PART_NAMES 顺序：root, torso, headneck, l_arm, r_arm,
# l_leg, r_leg, l_foot, r_foot —— 下肢四部位 = 5..8。
LOWER_LIMB_PARTS = (5, 6, 7, 8)

LOG_SIG_CLAMP = 3.0   # 连续专家 log σ 钳位 [-3, 3]：p(z) 不参数化，
                      # σ 只经 PoE 精度合成受训，钳位防精度退化
                      # （σ→0 精度爆炸、单专家接管合成）。

# V-stream parameter prefixes frozen at construction (carrier identity).
# SharedEmbeddings appears at six paths in the state dict (top level,
# ModalEncoders, both encoder heads, both pose heads); every path is frozen
# so the carrier load is complete and the T stream shares the frozen token
# space (the trained v_enc and its embeddings are coupled).  Same scope as
# V3_3G (2026-10-17).
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


def v5_1_z_src(config_value: int, override: str | None = None) -> str:
    """Eval 三条件 → z 来源开关；``override`` = 判据 ③ T 参与干预
    （--z-src V 把 VT 行的 z_PoE 换成 z_V = 禁 T 专家；T = 反向对照）。"""
    if override and config_value == CONFIG_VT:
        return override
    return {CONFIG_VT: "VT", CONFIG_V: "V", CONFIG_T: "T"}[config_value]


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


class V5_1PoseHead(nn.Module):
    """[copied from model_v3e, 2026-10-19 — the carrier head, plus one
    structural hook] Pure single-stream 9-part pose head: part queries decode
    ONE memory stream, per-part unembeds emit the part's joints in 6D plus a
    per-joint log-sigma.  Hard PART_JOINTS partition.

    The only addition over the E verbatim class: ``forward(mem, film=None)``
    — an optional callable applied to the part features BEFORE the part
    Linear (FiLM injection point, V5_1A §2.2).  ``film=None`` reproduces the
    E forward bit-exactly, and the hook registers no parameters, so the
    state dict is key-identical to E (zero-mapping carrier load).

    Sigma heads are frozen at init in V5_1 (E carrier state, not read)."""

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

    def forward(self, mem, film=None):
        """mem: (B, tw, dim).  Returns (pose (B,tw,POSE_DIM), log_sigma).
        ``film``: optional callable (B,tw,n_parts,dim) -> same — FiLM 注入
        （部位 Linear 前；None = E 逐位行为）。"""
        B = mem.shape[0]
        q = self.query.expand(B, -1, -1, -1).reshape(B, self.tw * self.n_parts, self.dim)
        z = self.decoder(q, mem).reshape(B, self.tw, self.n_parts, self.dim)
        if film is not None:
            z = film(z)
        pose_parts = torch.cat(
            [self.out_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        sig_parts = torch.cat(
            [self.sig_parts[p](z[..., p, :]) for p in range(self.n_parts)], dim=-1)
        pose = pose_parts[..., self.part_place_idx] * self.pose_std + self.pose_mean
        sig = sig_parts[..., self.part_place_idx]
        return pose, sig


class V5_1TrajHead(nn.Module):
    """[copied from model_v3e + the same film hook] Single-stream
    root-trajectory regression (velocity -> cumsum).  ``film`` applies to the
    encoder output feature (proj 前，V5_1A 注入点：根轨迹头输出特征)."""

    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.tw = tw
        self.encoder = _temporal_encoder(dim, 2, nhead, dim_feedforward, dropout)
        self.proj = nn.Linear(dim, 3)

    def forward(self, mem, film=None):
        h = self.encoder(mem)
        if film is not None:
            h = film(h)
        v_hat = self.proj(h)
        trans_hat = torch.cumsum(v_hat, dim=1) / float(FPS)
        return v_hat, trans_hat


class V3EContactHead(nn.Module):
    """[copied from model_v3e] Per-foot contact logits from the tactile
    tokens (f6_soft supervision, air-weighted in V5_1 training — G 配方，
    λ_air=2.0：G 针对 air recall 前科的开关)。"""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class ZExpert(nn.Module):
    """z 专家（V5_1A §2.2，从零）：MLP(tok → z 参数)。

    输出 (B, tw, 18) = 接触 logits 2 | 连续 μ 8 | 连续 log σ 8。
    每分量被各自 GT 监督校准（监督规则 §3：接触三读数、相位/CoP 只监督
    各专家 μ——监督化 PoE，p(z) 先验不参数化）。"""

    def __init__(self, dim: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, EXPERT_OUT))

    def forward(self, tok):
        out = self.mlp(tok)                      # (B, tw, 18)
        logits = out[..., :Z_CONT]
        mu = out[..., Z_CONT:Z_CONT + Z_GAUSS]
        log_sig = out[..., Z_CONT + Z_GAUSS:].clamp(-LOG_SIG_CLAMP, LOG_SIG_CLAMP)
        return logits, mu, log_sig


class FiLMLayer(nn.Module):
    """h' = h + (γ(z)−1) ⊙ h + β(z)，z → (γ, β) 轻量 MLP（V5_1A §2.2）。

    零初始化：weight=0、bias=[1…1, 0…0] → 任意输入下 γ≡1、β≡0，训练起点
    逐位恒等（契约 1 的结构保证，与输入无关）。加法形式（而非 γ⊙h+β）
    的关键：∂h'/∂γ = h ≠ 0——训练第一步起 FiLM 即可被 L_pose/L_prior_pose
    驱动（γ⊙h+β 直取写法在恒等分支上切断 γ/β 梯度，FiLM 永远锁死恒等）。
    一个模块服务全部注入点（姿态头下肢四部位特征 + 根轨迹头特征 + 先验塔
    同部位）——z 语义（接触概率/相位/CoP）对全部注入点共享一套调制。"""

    def __init__(self, z_dim: int, dim: int):
        super().__init__()
        self.z_dim = int(z_dim)
        self.dim = int(dim)
        self.mlp = nn.Linear(self.z_dim, 2 * self.dim)
        nn.init.zeros_(self.mlp.weight)
        with torch.no_grad():
            self.mlp.bias[: self.dim] = 1.0
            self.mlp.bias[self.dim:] = 0.0

    def forward(self, z):
        return self.mlp(z)  # (B, tw, 2*dim): [γ | β]


def poe_synthesize(params_v, params_t, logit_w):
    """PoE 闭式合成（V5_1A §2.2，零参数 + 接触温度权重）。

    params_v / params_t: (logits (B,tw,2), mu (B,tw,8), log_sig (B,tw,8))。
    logit_w: (2,) 可学习非负标量（softplus 参数化；等价于常先验
    Bernoulli PoE 的温度推广）。
    - 连续分量：对角高斯精度加权 μ_PoE = Σ p_i μ_i / Σ p_i、
      σ²_PoE = 1/Σ p_i（p_i = 1/σ_i²）。
    - 接触分量：logit_PoE = Σ w_i logit_i。
    返回 dict：三读数接触 logits、各专家 μ/log_sig、μ_PoE/log_sig_PoE、
    以及入 FiLM 的 z 向量（接触概率 sigmoid(logit) ‖ 连续 μ，10 维）。
    """
    logit_v, mu_v, ls_v = params_v
    logit_t, mu_t, ls_t = params_t
    p_v = (-2.0 * ls_v).exp()
    p_t = (-2.0 * ls_t).exp()
    p_sum = p_v + p_t
    mu_poe = (p_v * mu_v + p_t * mu_t) / p_sum.clamp(min=1e-8)
    log_sig_poe = -0.5 * torch.log(p_sum.clamp(min=1e-8))
    w = F.softplus(logit_w)                     # (2,) 非负
    logit_poe = (w[0] * logit_v + w[1] * logit_t) / w.sum().clamp(min=1e-8)

    def z_for(logits, mu):
        return torch.cat([torch.sigmoid(logits), mu], dim=-1)   # (B, tw, 10)

    return {
        "logit_gV": logit_v, "logit_gT": logit_t, "logit_poe": logit_poe,
        "mu_gV": mu_v, "mu_gT": mu_t,
        "log_sig_gV": ls_v, "log_sig_gT": ls_t,
        "mu_poe": mu_poe, "log_sig_poe": log_sig_poe,
        "z_v": z_for(logit_v, mu_v),
        "z_t": z_for(logit_t, mu_t),
        "z_poe": z_for(logit_poe, mu_poe),
    }


class AnySoleModelV5_1(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, pose_layers=6,
                 t_phys_v2: bool = False, d8_heads: bool = False):
        super().__init__()
        self.t_phys_v2 = bool(t_phys_v2)
        self.d8_heads = bool(d8_heads)
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # regress machinery in train/eval
        self.v5_1 = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv", t_phys_v2=t_phys_v2,
        )
        self.v_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        self.t_temporal = _temporal_encoder(d, 4, nhead, dropout=dropout)
        # 维度 3 足-地监督三头（2026-10-21，T增强三组实验方案.md）：
        # tcop = 触觉 CoP 自重建（目标 T_phys cop 通道，力>0 掩码）；
        # strike = 着地模式三分类（T 侧落地事件标签，仅落地帧计损失）；
        # gtcop = GT CoP 回归（v5_1_gt 数据件，接触帧掩码）。
        if d8_heads:
            self.tcop_head = nn.Linear(d, 4)
            self.strike_head = nn.Linear(d, 6)
            self.gtcop_head = nn.Linear(d, 4)
        # sigma inits match the E carrier (V=-3 / prior=+3); the sigma heads
        # are frozen and never read — state-dict compatibility only (G 口径).
        self.pose_head = V5_1PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=-3.0)
        self.prior_head = V5_1PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, sigma_init=3.0)
        self.traj_head = V5_1TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.prior_traj = V5_1TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.contact_head = V3EContactHead(dim=d)
        # z 专家（从零）：g_V 读冻结 v_tok（冻结图上的轻量读出），g_T 读
        # 从零 foot_conv 的 t_tok。
        self.g_V = ZExpert(dim=d)
        self.g_T = ZExpert(dim=d)
        # 接触分量 PoE 温度权重（可学习非负，softplus 参数化；init 0 →
        # w≈0.693 等权，logit_PoE = 两专家 logit 均值起步）。
        self.logit_w = nn.Parameter(torch.zeros(2))
        # FiLM（从零，z 进解码的唯一通道）。
        self.film_layer = FiLMLayer(z_dim=Z_DIM, dim=d)
        # Carrier rule: the V stream is frozen at construction.  The E
        # ckpt_best weights are loaded by the entry (anysole/models/v5_1.py);
        # freezing here makes the carrier identity load-independent.
        # 冻结范围（V5_1A §3）：V 流 286 键全冻结；训练 = foot_conv +
        # g_V + g_T + FiLM + PoE 权重 + t_temporal/prior 塔/接触头。
        for name, param in self.named_parameters():
            if name.startswith(CARRIER_PREFIXES):
                param.requires_grad_(False)
        for head in list(self.pose_head.sig_parts) + list(self.prior_head.sig_parts):
            for param in head.parameters():
                param.requires_grad_(False)

    def _film_parts(self, h, z):
        """FiLM 注入（部位特征，部位 Linear 前）：只作用下肢四部位
        （l_leg/r_leg/l_foot/r_foot），其余部位直通（结构免疫）。

        零税契约 1 的保证（h' = h + (γ−1)⊙h + β）：γ≡1、β≡0（零初始化）
        时 h' = h + 0 + 0，与 E 逐位一致（零符号位除外——+0.0 的符号
        翻转不改变任何后续算术值）；关键在梯度路径：∂h'/∂γ = h ≠ 0，
        训练第一步起 FiLM 即可被 L_pose/L_prior_pose 驱动（torch.where
        直取原特征的写法会把 γ/β 的梯度在恒等分支上切断，FiLM 永远
        锁死在恒等）。"""
        out = self.film_layer(z)                  # (B, tw, 2d)
        gamma = out[..., : self.d]
        beta = out[..., self.d:]
        h = h.clone()
        parts = h[..., LOWER_LIMB_PARTS, :]
        h[..., LOWER_LIMB_PARTS, :] = (
            parts + (gamma - 1.0).unsqueeze(-2) * parts + beta.unsqueeze(-2))
        return h

    def _film_vec(self, h, z):
        """FiLM 注入（根轨迹头输出特征，proj 前）。同零税形式 + 梯度路径。"""
        out = self.film_layer(z)
        gamma = out[..., : self.d]
        beta = out[..., self.d:]
        return h + (gamma - 1.0) * h + beta

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None, z_src="all"):
        """config_id is accepted for the train/eval call contract and
        deliberately IGNORED — the three eval conditions are three readouts
        of one forward selected by ``z_src`` (V5_1A §2.3；T 流输入三条件
        恒定——g_T 始终读真实压力).

        z_src: "all"（训练，一次前向出三读数——x0_hat = VT2M 行）|
        "V" | "VT" | "T"。"""
        if z_src not in ("all", "V", "VT", "T"):
            raise ValueError("z_src must be 'all'|'V'|'VT'|'T', got %r" % z_src)
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        v_tok, t_tok = self.encoders(V_feat, T_tac, cid, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        v_mem = self.v_temporal(v_tok)
        t_mem = self.t_temporal(t_tok)
        params_v = self.g_V(v_tok)
        params_t = self.g_T(t_tok)
        zs = poe_synthesize(params_v, params_t, self.logit_w)
        contact_logits = self.contact_head(t_tok)
        tcop = self.tcop_head(t_tok) if self.d8_heads else None
        strike_logits = self.strike_head(t_tok) if self.d8_heads else None
        gtcop = self.gtcop_head(t_tok) if self.d8_heads else None
        # T 先验塔（G 风格，从零）：T2M 行 = 无条件先验 + FiLM(z_T) 的
        # 健全性检查（~90mm 量级参照；G 前科 315.5 防回归——先验塔单独
        # 回归到 315.5，FiLM(z_T) 与 z 多通道监督是防回归结构）。
        pose_t, sigma_t = self.prior_head(t_mem, film=lambda h: self._film_parts(h, zs["z_t"]))
        v_prior, trans_prior = self.prior_traj(t_mem, film=lambda h: self._film_vec(h, zs["z_t"]))

        if z_src == "all":
            pose_v, sigma_v = self.pose_head(v_mem, film=lambda h: self._film_parts(h, zs["z_v"]))
            pose_vt, _ = self.pose_head(v_mem, film=lambda h: self._film_parts(h, zs["z_poe"]))
            v_hat, trans_hat = self.traj_head(v_mem, film=lambda h: self._film_vec(h, zs["z_poe"]))
            v_hat_v, trans_v = self.traj_head(v_mem, film=lambda h: self._film_vec(h, zs["z_v"]))
            x0_hat, x0_v, x0_prior = pose_vt, pose_v, pose_t
        elif z_src == "V":
            pose_v, sigma_v = self.pose_head(v_mem, film=lambda h: self._film_parts(h, zs["z_v"]))
            v_hat, trans_hat = self.traj_head(v_mem, film=lambda h: self._film_vec(h, zs["z_v"]))
            x0_hat, x0_v, x0_prior = pose_v, None, pose_t
        elif z_src == "VT":
            pose_vt, sigma_v = self.pose_head(v_mem, film=lambda h: self._film_parts(h, zs["z_poe"]))
            v_hat, trans_hat = self.traj_head(v_mem, film=lambda h: self._film_vec(h, zs["z_poe"]))
            x0_hat, x0_v, x0_prior = pose_vt, None, pose_t
        else:  # "T" — null-V 前向（先验塔不消费 V；调用侧 V 已零化）
            sigma_v = None
            x0_hat, x0_v, x0_prior = pose_t, None, pose_t
            v_hat, trans_hat = v_prior, trans_prior
        return {
            "x0_hat": x0_hat,            # 主读出（z_src 选定；"all" = VT2M）
            "x0_v": x0_v,                # V2M 行（"all" 训练损失用）
            "x0_prior": x0_prior,        # T2M 行（先验塔 + FiLM(z_T)）
            "sigma_v": sigma_v,
            "sigma_prior": sigma_t,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "v_hat_v": v_hat_v if z_src == "all" else None,   # V2M 行轨迹（"all" 训练读出）
            "trans_v": trans_v if z_src == "all" else None,
            "v_prior": v_prior,
            "trans_prior": trans_prior,
            "contact_logits": contact_logits,
            "tcop": tcop,
            "strike_logits": strike_logits,
            "gtcop": gtcop,
            # z 通道读数（损失三读数监督 + 判据 ③ 专家分歧审计）
            "z": zs,
            "logit_w": self.logit_w,
            "F": v_mem,          # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "pressure_hat": None,
            "vfeat_hat": None,
        }
