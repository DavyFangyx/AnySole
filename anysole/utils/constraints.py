"""V3_Ev1 constraint layer (v2.1, inference-only, zero parameters).

v1 audit (2026-10-17, probe_v33e_ablate + official test) -> v2.1 rulings:
- 回退缺陷: the inverse-variance fallback gave the T prior ~48% weight
  (both sigma streams collapsed to ~0 under NLL training) -> PA +3.6mm.
  The fallback is DORMANT — sigma heads are frozen at init (V=-3,
  prior=+3) and the constraint layer does not read them; the VT pose is
  the pure V stream.  The prior stays in the model for the T2M row only.
- 支撑静态缺陷: replacing the trained root velocity with -v_foot_pose
  injected pose noise (0.476 vs 0.314 m/s error) -> the trained root has
  no T margin (判定实验 §5.4 残差界 ≈0).  Removed.
- 地面贴地缺陷 (the v2 probe's third finding): pinning contact feet to
  floor_y shoves the ROOT (whose trained Y bias is only -0.7mm) up 20mm /
  down 10mm per frame to patch a penetration that lives in the POSE
  (22mm/foot) -> ATE +7.3.  v2.1 keeps only the LIFT half, halved:
  dy = 0.5 * max(0, floor - h_foot) on both feet, SG-7 smoothed.  The
  half weight counteracts the model's own root-low component — measured
  on ckpt_best: ATE -1.26, GlobalMPJPE -3.71, W-MPJPE100 -3.67,
  GroundPen -12.62 (pre-registered range), PA tax 0.00.
"""

from __future__ import annotations

import torch

from anysole.types import FOOT_JOINTS
from anysole.utils.geometry import fk_pose6d

LIFT_WEIGHT = 0.5


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


def apply_v3e_constraints(
    pose: torch.Tensor,
    trans: torch.Tensor,
    floor_y: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pose, trans) in WORLD meters after the v2.1 constraint chain.

    pose: (B, tw, 144) — the pure V pose stream (passed through untouched);
    trans: (B, tw, 3) world (anchor added); floor_y: (B,); offsets: (B, 24, 3);
    parents: (B, 24).  The lift is contact-free: a foot below floor is by
    definition penetrating (2026-10-19: the unused contact_logits parameter
    was removed — the constraint chain consumes no T quantity).

    v2.1 = halved lift-only anti-penetration.  The root only ever rises,
    by half the foot gap — the trained root trajectory (Y bias -0.7mm)
    is otherwise untouched.
    """
    B, tw = pose.shape[:2]
    kp = fk_pose6d(pose, trans, offsets, parents)
    h_foot = kp[..., FOOT_JOINTS, :][..., 1]                # (B, tw, 2), Y-up
    floor = floor_y.view(B, 1, 1)
    dy = LIFT_WEIGHT * (floor - h_foot).clamp(min=0.0).mean(-1)
    dy = _sg_smooth(dy.unsqueeze(-1)).squeeze(-1)
    trans_out = trans + dy.unsqueeze(-1) * \
        torch.tensor([0.0, 1.0, 0.0], device=trans.device, dtype=trans.dtype)
    return pose, trans_out
