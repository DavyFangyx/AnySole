"""V3_Gv1 分部观测层 (v1.1, inference-only, zero parameters).

设计规则（V3_G方案_GT燃料.md §1/§2.4）。

v1.1 裁定（2026-10-18，生产模型穿透帧诊断）：两骨 IK 腿链修正退场——
诊断数字：穿透帧上 GT 髋高于预测 12.1mm、GT 足高于预测 11.8mm（全身
均匀低），膝角差仅 1.9°、踝角差 −1.2°。穿透 = 整体平移误差，不是腿形
误差；弯曲一根基本正确的膝盖去补偿平移 → val PA +4.5、踝足 PA +4.1、
MPJRE +2.1（零税契约被破坏）。正确机制 = 平移（E v2.1 口径）；接触证据
的作用从「部位选择」改为「平移幅度门控」。

链路（apply_v3g_constraints，v1.1）：
0. floor 来源（floor_src 开关）：主判据 = "gt"（数据集会话 floor，E v2.1
   同口径）；"t" = T 派生 floor（接触帧足高 25% 分位），诊断开关——穿透
   是系统性偏置，预测自锚定的 floor 继承同量偏置 → gap≈0，不作为主机制
   （来源税报告项预注册）。
1. 接触门控平移提升：gap = max(0, floor − h_foot) 双足平均；λ = 0.5 +
   0.5·c̄（c̄ = 接触头概率双足平均，SG-7 平滑）——摆动帧 λ≈0.5（E 半量，
   ATE 安全），高置信接触帧 λ→1.0（足在证据上贴地，全额修正）。纯平移，
   PA 结构上不变（零税契约的机制保证）。λ=0.5 常数时与 E v2.1 逐位一致。
2. 腿链 IK（_leg_lift，v1 退场保留）：两骨 IK + 踝摆补偿的完整实现保留
   备查——若未来误差结构改变（诊断显示腿形误差主导）可直接复用。

滑步/支撑项不进 v1（E v1 支撑静态 0.476 vs 0.314 前科；F 精修 +14 前科；
探针无 T 滑步线性信息）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anysole.types import FOOT_JOINTS, N_JOINTS
from anysole.utils.constraints import _sg_smooth
from anysole.utils.geometry import rot6d_to_rotmat, rotmat_to_6d

TAU_C = 0.5      # contact-probability gate (the head's calibrated operating
# point; 0.8 was unreachable — the pos-weighted head's sigmoid hovers ~0.5,
# so the precision-oriented gate never fired on the production model
# (2026-10-18 val debug: c>0.8 = 0/80 frames).  Mislabeled swing frames that
# pass 0.5 are neutralized by the penetration condition + SG-7 + G_MAX.)
EPS_PEN = 0.005  # minimum penetration before the leg lift engages (m)
G_MAX = 0.04     # per-frame leg-lift cap (m)
LIFT_WEIGHT = 0.5  # E v2.1 residual root-lift halving
FLOOR_Q = 0.25   # quantile of contact-gated foot heights for the floor
MIN_GATED = 3    # fallback below this many gated frames per window

_Y = torch.tensor([0.0, 1.0, 0.0])
_Z = torch.tensor([0.0, 0.0, 1.0])


def _fk_world(pose_6d, trans, offsets, parents):
    """FK returning world positions (B,tw,24,3) AND world rotations
    (B,tw,24,3,3) — the leg lift needs the rotations to write local deltas."""
    batch, time, _ = pose_6d.shape
    rotmats = rot6d_to_rotmat(pose_6d.reshape(batch, time, N_JOINTS, 6))
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch, -1, -1)
    parents_list = parents.detach().cpu().tolist() if parents.ndim == 1 else parents[0].detach().cpu().tolist()
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


def _from_to(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Shortest-arc rotation mapping normalized v1 -> v2; (..., 3) -> (..., 3, 3).

    Antiparallel pairs fall back to a perpendicular axis (same convention as
    geometry._from_to_np).  Parallel pairs yield the identity.
    """
    y = _Y.to(v1)
    z = _Z.to(v1)
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    cross = torch.linalg.cross(v1, v2, dim=-1)
    s = cross.norm(dim=-1)
    axis = cross / s.clamp(min=1e-8).unsqueeze(-1)
    angle = torch.atan2(s, dot)
    # antiparallel: any perpendicular axis, pi rotation
    anti = dot < -1.0 + 1e-6
    perp = torch.linalg.cross(v1, y.expand_as(v1), dim=-1)
    perp_alt = torch.linalg.cross(v1, z.expand_as(v1), dim=-1)
    perp = torch.where(perp.norm(dim=-1, keepdim=True) < 1e-4, perp_alt, perp)
    perp = F.normalize(perp, dim=-1)
    axis = torch.where(anti.unsqueeze(-1), perp, axis)
    angle = torch.where(anti, torch.full_like(angle, 3.141592653589793), angle)
    # parallel (including exact zeros): identity
    small = s < 1e-8
    angle = torch.where(small, torch.zeros_like(angle), angle)
    rotvec = axis * angle.unsqueeze(-1)
    return _rodrigues(rotvec)


def _rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    ang = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / ang.clamp(min=1e-8)
    c = torch.cos(ang).unsqueeze(-1)      # (..., 1, 1)
    sn = torch.sin(ang).unsqueeze(-1)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    # skew matrix [a]x
    a1, a2, a3 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(a1)
    skew = torch.stack([
        torch.stack([zero, -a3, a2], dim=-1),
        torch.stack([a3, zero, -a1], dim=-1),
        torch.stack([-a2, a1, zero], dim=-1),
    ], dim=-2)
    outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)
    rot = c * eye + (1.0 - c) * outer + sn * skew
    # zero-angle rows land exactly on I (axis ~ 0 there)
    return rot


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


def _leg_lift(pose_6d, kp, g_rot, offsets, gap, mask):
    """Two-bone IK lift per foot: hip fixed, ankle raised vertically by gap,
    knee takes the solution nearest its current pose; hip/knee/ankle local 6D
    updated, the ankle swing preserves the FOOT JOINT's horizontal position
    exactly (no sliding introduced).  Only masked frames are written back.

    kp/g_rot: the FK state BEFORE correction; gap: (B,tw,2) smoothed lifts;
    mask: (B,tw,2) bool.
    """
    pose_out = pose_6d.clone()
    y = _Y.to(pose_6d)
    for hip, knee, ankle, foot in ((1, 4, 7, 10), (2, 5, 8, 11)):
        g = gap[..., (0 if hip == 1 else 1)]
        m = mask[..., (0 if hip == 1 else 1)]
        if not bool(m.any()):
            continue
        P = kp[:, :, hip]
        K = kp[:, :, knee]
        A = kp[:, :, ankle]
        o_knee = offsets[:, knee]
        o_ankle = offsets[:, ankle]
        o_foot = offsets[:, foot]
        L1 = o_knee.norm(dim=-1, keepdim=True).unsqueeze(1)   # (B,1,1)
        L2 = o_ankle.norm(dim=-1, keepdim=True).unsqueeze(1)
        A2 = A + g.unsqueeze(-1) * y
        d = (A2 - P).norm(dim=-1)
        d = d.clamp(min=(L1 - L2).abs().squeeze(-1) + 1e-4,
                    max=(L1 + L2).squeeze(-1) - 1e-4)
        a = (L1.squeeze(-1) ** 2 - L2.squeeze(-1) ** 2 + d ** 2) / (2.0 * d)
        h = (L1.squeeze(-1) ** 2 - a ** 2).clamp(min=0.0).sqrt()
        u = (A2 - P) / d.unsqueeze(-1)
        w = torch.linalg.cross(u, y.expand_as(u), dim=-1)
        w_alt = torch.linalg.cross(u, _Z.to(pose_6d).expand_as(u), dim=-1)
        w = torch.where(w.norm(dim=-1, keepdim=True) > 1e-6, w, w_alt)
        w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        K1 = P + a.unsqueeze(-1) * u + h.unsqueeze(-1) * w
        K2 = P + a.unsqueeze(-1) * u - h.unsqueeze(-1) * w
        pick = ((K1 - K).norm(dim=-1) <= (K2 - K).norm(dim=-1)).unsqueeze(-1)
        Kp = torch.where(pick, K1, K2)
        R_hip_w = g_rot[:, :, hip]
        R_knee_w = g_rot[:, :, knee]
        R_ankle_w = g_rot[:, :, ankle]
        Rh = _from_to(K - P, Kp - P)
        Rk = _from_to(
            torch.einsum("btij,bj->bti", Rh @ R_knee_w, o_ankle),  # A''-K'
            A2 - Kp)
        Rka = Rk @ Rh @ R_ankle_w
        f_cur = torch.einsum("btij,bj->bti", Rka, o_foot)
        f_tgt = kp[:, :, foot] + g.unsqueeze(-1) * y - A2
        Ra = _from_to(f_cur, f_tgt)
        R0_w = g_rot[:, :, 0]
        hip_new = R0_w.transpose(-1, -2) @ Rh @ R_hip_w
        knee_new = (R_hip_w.transpose(-1, -2) @ Rh.transpose(-1, -2)
                    @ Rk @ Rh @ R_knee_w)
        ankle_new = (R_knee_w.transpose(-1, -2) @ Rh.transpose(-1, -2)
                     @ Rk.transpose(-1, -2) @ Ra @ Rk @ Rh @ R_knee_w)
        for joint, new_rot in ((hip, hip_new), (knee, knee_new), (ankle, ankle_new)):
            s6 = joint * 6
            d6 = rotmat_to_6d(new_rot)
            pose_out[:, :, s6:s6 + 6] = torch.where(
                m.unsqueeze(-1), d6, pose_out[:, :, s6:s6 + 6])
    return pose_out


def apply_v3g_constraints(
    pose: torch.Tensor,
    trans: torch.Tensor,
    contact_logits: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
    floor_y: torch.Tensor = None,
    floor_src: str = "gt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pose, trans) in WORLD meters after the v1 observation chain.

    pose: (B, tw, 144) the V stream (or the T prior for the T2M row);
    trans: (B, tw, 3) world (anchor added); contact_logits: (B, tw, 2);
    floor_y: (B,) dataset session floor (floor_src="gt", the main claim —
    E v2.1 口径, clean isolation).  floor_src="t" derives the floor from
    contact-gated foot heights (diagnostic switch, the 来源税 report item).
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
    # 1. contact-gated translation lift (v1.1): pure translation, PA-invariant
    # by construction.  lam = 0.5 + 0.5 * cbar — E's halving at swing frames,
    # up to full correction where the contact head is confident (the foot is
    # on the ground by evidence, so the gap is correction, not noise).
    gap = (floor_hat.view(batch, 1, 1) - h_foot).clamp(min=0.0).mean(-1)  # (B, tw)
    cbar = _sg_smooth(c.mean(-1, keepdim=True)).squeeze(-1)              # (B, tw)
    lam = LIFT_WEIGHT + 0.5 * cbar                                       # 0.5..1.0
    dy = (lam * gap).unsqueeze(-1)
    dy = _sg_smooth(dy).squeeze(-1)
    trans_out = trans + dy.unsqueeze(-1) * _Y.to(trans)
    return pose, trans_out
