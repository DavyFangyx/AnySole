"""V3_Hv1 residual layer (v1, inference-only, zero parameters).

设计规则（V3_H方案.md §1/§2.3）——「完整流形，残差归零」：
- 反馈规则：T 先定义物理状态（接触、地面），修正由 V 假设与该状态的
  物理残差驱动；残差为零则修正为零（零税双保险）。
- 通道规则：每条通道 = (坐标, 残差, 方向, 增益, 饱和)，单遍应用、
  无反馈迭代、应用层零可训参数；本层只读模型读出
  （pose/trans/contact_logits），不进训练。
- 不动根规则：支撑通道只写踝关节（E v1 支撑静态 0.476 vs 0.314、
  贴地推根 ATE+7.3 两个失败形式的反面）。

链路（apply_v3h_constraints）：
0. floor 来源（floor_src 开关）：主判据 = "gt"（数据集会话 floor，E v2.1
   同口径——干净隔离出「残差通道」这一唯一变量）。"t" = T 派生 floor
   （接触帧足高 25% 分位）——诊断开关：接触只给相对约束、不给绝对高度
   （gauge 自由度），系统性姿态偏置下自锚定 floor 继承同量偏置 → gap≈0
   → 修正不触发（G 线冒烟实测同源退化）；「T floor 替代 GT floor」不作为
   主机制，作为来源税报告项预注册（两口径差值）。
1. 穿透：E v2.1 形式（dy = 0.5·max(0, floor−h_foot) 双足平均、仅升不降、
   SG-7、加至根 Y）。原始姿态的足高进 E 同款公式。
2. 支撑：接触置信帧（c>TAU_C）足部帧间水平位移 Δ 的踝-偏航阻尼——绕
   世界 Y 轴旋转 γ = clamp(−α_s·(Δ·e)/r)（e = 足绕踝水平切向、
   r = 踝-足水平臂长），SG-7。只写踝、不动根、不改足高（纯水平）。
   滑步无 T 线性证据（探针 +2.15*）——本通道是接触消费的几何正则
   （额度 E −0.5 水平）。GT 接触帧 Δ≈0 → γ≈0（残差为零修正为零）。

足部通道（方案 §2.3 的第三通道）不进 v1：冒烟实测（2026-10-17）证伪了
零阶几何形式——本骨架足关节在趾部（无跟关节），踝→足向量随姿态相位
滚动（GT 接触帧与竖直方向夹角均值偏差 0.50 rad、足关节局部"足底法向"
均值偏差 0.20 rad、p95 0.44 rad），不存在稳定的平足几何参照；任何固定
参照的朝向对齐都会在 GT 正确姿态上产生偏置。足部姿态兑现整体移入一阶
对照实验（T 特征残差头，V3_H方案.md §3）。
"""

from __future__ import annotations

import torch

from anysole.types import FOOT_JOINTS, N_JOINTS
from anysole.utils.geometry import rot6d_to_rotmat, rotmat_to_6d

TAU_C = 0.8        # contact-probability gate (precision-oriented)
LIFT_WEIGHT = 0.5  # E v2.1 residual root-lift halving
GAIN_SUPPORT = 0.5  # ankle-yaw damping proportional gain
GAMMA_MAX = 0.1    # per-frame yaw cap (rad)
FLOOR_Q = 0.25     # quantile of contact-gated foot heights for the floor
MIN_GATED = 3      # fallback below this many gated frames per window

# SMPL-24 leg chain: (hip, knee, ankle, foot) = (1,4,7,10) / (2,5,8,11);
# the support channel writes ONLY the ankle joints (7/8, parents 4/5).
ANKLE_JOINTS = (7, 8)
FOOT_OF = {7: 10, 8: 11}
KNEE_OF = {7: 4, 8: 5}

_Y = torch.tensor([0.0, 1.0, 0.0])


def _sg_smooth(x: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Savitzky-Golay(order 2) along the time axis; no-op for short windows."""
    from scipy.signal import savgol_filter

    B, T, D = x.shape
    if T < 5:
        return x
    window = min(window, T if T % 2 == 1 else T - 1)
    if window < 5:
        return x
    return torch.from_numpy(
        savgol_filter(x.detach().cpu().numpy(), window, 2, axis=1)
    ).to(x)


def _fk_world(pose_6d: torch.Tensor, trans: torch.Tensor, offsets: torch.Tensor,
              parents: torch.Tensor):
    """FK returning world positions (B,tw,24,3) AND world rotations
    (B,tw,24,3,3) — the ankle channel writes local deltas, so the world
    rotations are required."""
    batch, time, _ = pose_6d.shape
    rotmats = rot6d_to_rotmat(pose_6d.reshape(batch, time, N_JOINTS, 6))
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch, -1, -1)
    parents_list = (parents.detach().cpu().tolist() if parents.ndim == 1
                    else parents[0].detach().cpu().tolist())
    g_rot, g_pos = [], []
    for j in range(N_JOINTS):
        p = int(parents_list[j])
        if p < 0:
            jr = rotmats[:, :, j]
            jp = offsets[:, j].unsqueeze(1).expand(-1, time, -1)
        else:
            jr = torch.matmul(g_rot[p], rotmats[:, :, j])
            jp = g_pos[p] + torch.matmul(
                g_rot[p], offsets[:, j].unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        g_rot.append(jr)
        g_pos.append(jp)
    return torch.stack(g_pos, dim=2) + trans.unsqueeze(2), torch.stack(g_rot, dim=2)


def _rotmat_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: unit axis (...,3), angle (...,) -> rotation matrices
    (...,3,3).  R v = v + sin(θ)(a×v) + (1−cos θ)(a×(a×v))."""
    c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return c * eye + (1.0 - c) * torch.matmul(K, K) + s * K


def _derive_floor(contact_prob: torch.Tensor, h_foot: torch.Tensor) -> torch.Tensor:
    """T-derived floor per window (B,): 25th percentile of foot heights on
    high-confidence contact frames; falls back to the ungated per-frame
    min-foot-height percentile when fewer than MIN_GATED frames qualify."""
    batch = h_foot.shape[0]
    gated = torch.where(
        contact_prob > TAU_C, h_foot,
        torch.full_like(h_foot, float("inf")))
    vals = gated.reshape(batch, -1)
    sorted_vals, _ = torch.sort(vals, dim=-1)          # infs at the end
    cnt = torch.isfinite(vals).sum(dim=-1)
    idx = ((cnt - 1).float() * FLOOR_Q).long().clamp(min=0)
    idx = torch.clamp(idx, max=vals.shape[-1] - 1)
    floor_gated = torch.gather(sorted_vals, 1, idx.unsqueeze(-1)).squeeze(-1)
    hmin = h_foot.min(dim=-1).values
    hmin_sorted, _ = torch.sort(hmin, dim=-1)
    idx_fb = int((hmin.shape[-1] - 1) * FLOOR_Q)
    floor_fallback = hmin_sorted[:, idx_fb]
    return torch.where(cnt >= MIN_GATED, floor_gated, floor_fallback)


def _ankle_yaw_damp(pose_6d: torch.Tensor, kp: torch.Tensor, g_rot: torch.Tensor,
                    contact_prob: torch.Tensor) -> torch.Tensor:
    """Support channel: ankle-yaw damping of the contact foot's horizontal
    drift.  The frame-to-frame displacement Δ is damped by an ankle yaw about
    the world vertical — the induced tangential displacement opposes Δ; pure
    horizontal (no foot height change), the root is never touched.

    kp/g_rot: the FK state BEFORE correction.  Writes ONLY the ankle joints'
    local 6D; the foot follows as a rigid segment.  GT contact frames have
    Δ≈0 → γ≈0 (zero residual, zero correction).
    """
    pose_out = pose_6d.clone()
    y = _Y.to(pose_6d)
    for ank in ANKLE_JOINTS:
        foot = FOOT_OF[ank]
        gate = contact_prob[..., 0 if ank == 7 else 1] > TAU_C     # (B,tw)
        delta = kp[:, :, foot, [0, 2]] - torch.cat(
            [kp[:, :1, foot, [0, 2]], kp[:, :-1, foot, [0, 2]]], dim=1)
        arm_xy = (kp[:, :, foot] - kp[:, :, ank])[..., [0, 2]]
        r = arm_xy.norm(dim=-1).clamp(min=1e-3)
        e = torch.zeros_like(arm_xy)                               # yaw x arm tangent
        e[..., 0] = arm_xy[..., 1] / r
        e[..., 1] = -arm_xy[..., 0] / r
        proj = (delta * e).sum(dim=-1)                             # drift along tangent
        gamma = torch.where(
            gate, (-GAIN_SUPPORT * proj / r).clamp(-GAMMA_MAX, GAMMA_MAX),
            torch.zeros_like(proj))
        gamma = _sg_smooth(gamma.unsqueeze(-1)).squeeze(-1)
        R_yaw = _rotmat_axis_angle(
            y.expand(arm_xy.shape[0], arm_xy.shape[1], 3), gamma)
        R_ankle_new = torch.matmul(R_yaw, g_rot[:, :, ank])
        R_local = torch.matmul(
            g_rot[:, :, KNEE_OF[ank]].transpose(-1, -2), R_ankle_new)
        d6 = rotmat_to_6d(R_local)
        s6 = ank * 6
        pose_out[:, :, s6:s6 + 6] = torch.where(
            gate.unsqueeze(-1), d6, pose_out[:, :, s6:s6 + 6])
    return pose_out


def apply_v3h_noanchor(
    pose: torch.Tensor,
    trans: torch.Tensor,
    contact_logits: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V3_Hv5 无锚版：只留 gauge-free 的支撑通道（接触门控踝-偏航阻尼）。

    地面/穿透通道整体删除——T 提供不了绝对高度（gauge 自由度），模型
    不再消费任何 floor 输入；穿透/全局/ATE 成为纯观测值（底盘误差全额
    暴露），模型的 T 消费 = 接触标量（门控）+ T 特征（足部残差头，v5 保留）。
    trans 原样返回（不动根、不抬升）。
    """
    kp, g_rot = _fk_world(pose, trans, offsets, parents)
    c = torch.sigmoid(contact_logits)
    pose_out = _ankle_yaw_damp(pose, kp, g_rot, c)
    return pose_out, trans


def apply_v3h(no_anchor: bool, pose: torch.Tensor, trans: torch.Tensor,
              contact_logits: torch.Tensor, offsets: torch.Tensor,
              parents: torch.Tensor, floor_y: torch.Tensor = None):
    """V3_Hv1 推理层分派：no_anchor=True 只跑支撑通道（不消费 floor）；
    否则穿透 + 支撑（gt/t floor 开关）。"""
    if no_anchor:
        return apply_v3h_noanchor(pose, trans, contact_logits, offsets, parents)
    return apply_v3h_constraints(pose, trans, contact_logits, offsets, parents,
                                 floor_y=floor_y)


def apply_v3h_constraints(
    pose: torch.Tensor,
    trans: torch.Tensor,
    contact_logits: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
    floor_y: torch.Tensor = None,
    floor_src: str = "gt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pose, trans) in WORLD meters after the v1 residual chain.

    pose: (B, tw, 144) — the V stream under any condition (for the T2M row
    V was already zeroed upstream); trans: (B, tw, 3) world (anchor added);
    contact_logits: (B, tw, 2); floor_y: (B,) dataset session floor
    (floor_src="gt", the main claim — E v2.1 口径, clean isolation).
    floor_src="t" derives the floor from contact-gated foot heights
    (diagnostic switch, the 来源税 report item).
    """
    batch = pose.shape[0]
    kp, g_rot = _fk_world(pose, trans, offsets, parents)
    c = torch.sigmoid(contact_logits)
    h_foot = kp[:, :, FOOT_JOINTS, 1]                     # (B, tw, 2)
    if floor_src == "t":
        floor_hat = _derive_floor(c, h_foot)              # (B,)
    else:
        if floor_y is None:
            raise ValueError("floor_src='gt' requires floor_y "
                             "(dataset session floor, E v2.1 口径)")
        floor_hat = floor_y.view(batch)
    # 1. penetration: E v2.1 residual root lift on the raw pose (halved,
    # lift-only, SG-7) — the trained root is otherwise untouched.
    dy = LIFT_WEIGHT * (floor_hat.view(batch, 1, 1) - h_foot).clamp(min=0.0).mean(-1)
    dy = _sg_smooth(dy.unsqueeze(-1)).squeeze(-1)
    trans_out = trans + dy.unsqueeze(-1) * _Y.to(trans)
    # 2. support: ankle-yaw damping (ankle-only, contact-gated)
    pose_out = _ankle_yaw_damp(pose, kp, g_rot, c)
    return pose_out, trans_out
