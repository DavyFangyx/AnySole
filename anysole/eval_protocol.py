"""F0a evaluation protocol (fix_plan_v2.md §2): session-level, per-part, global,
temporal, contact, pressure, and robustness metrics.

Complements eval.py: eval.py keeps the frozen per-window metrics JSON
(``metrics/<split>.json``); this module adds ``metrics/<split>_fseries.json``
with the F-series protocol numbers, which every step F0..F9 reports:

- 局部姿态: MPJPE / PA-MPJPE per 9 parts (root/torso/headneck/l_arm/r_arm/
  l_leg/r_leg/l_foot/r_foot), plus upper (1-14) / lower (15-22) / ankle-feet
  (17,18,21,22) / hands (10,14) aggregates.  PA-MPJPE uses one per-frame
  similarity Procrustes alignment over all 23 joints (the eval.py 口径) and
  reads the part errors from the aligned points.
- 全局: W-MPJPE (160-frame = 4 s segments, first-frame aligned), RTE_norm
  (root trajectory error normalized by the mean GT per-frame displacement
  length), yaw_abs_deg / yaw_drift_deg (root heading error / accumulated
  drift; the local forward axis +Z or +X is picked from the data, see
  _pick_forward_axis).
- 时序: jitter (probe_jitter 口径: per-joint frame-to-frame displacement,
  mm/帧 @40Hz), accel_err_ms2 (second difference * FPS^2), seam_jump_mm at
  window boundaries (GT frame-to-frame at the same seams as reference).
- 接触: contact F1 (soft_contact_from_keypoints > 0.5 vs contact_gt) and
  contact-period foot slide in mm/帧, both on predicted-contact frames
  (train._evaluate 口径) and GT-contact frames.
- 压力输出 (V2M only): total-force R², per-foot CoP error (grid units,
  cop_from_grid 口径), per-frame 96-cell Pearson correlation.
- T2M 上半身: upper-body acceleration-magnitude distribution error (|mean
  pred - mean gt|), joint-limit violation rate (elbow/knee bone angle < 15°).

Robustness set (VT2M only, F0 起报告): two synthetic degradations per session
— a contiguous 20-40% span of V frames zeroed (``robust_vdrop``) and the same
for the T stream (``robust_tdrop``); the span is deterministic per
(session, seed).

Inference is session-level: eval windows of one session are non-overlapping
(stride == window_length), so their concatenation is the full-session
prediction.  The protocol skips itself when the dataset uses overlapping
windows (E6.4 continuation eval) — those are covered again from F3 on.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from anysole.data.dataset import collate_windows
from anysole.eval import _sample_x_t_init, _session_window_groups, _tactile_corr
from anysole.geometry import f2_to_world, fk_pose6d, rot6d_to_rotmat
from anysole.losses import soft_contact_from_keypoints
from anysole.train import condition_inputs, move_batch
from anysole.types import (
    CONFIG_MODE_NAMES,
    CONFIG_V,
    CONFIG_VT,
    FPS,
    FOOT_JOINTS,
    N_JOINTS,
    PART_JOINTS,
    PART_NAMES,
)

# Re-exported for probe scripts that import the 9-part grouping from here
# (single source of truth = anysole.types, fix_plan_v3 §V3-2).
UPPER_JOINTS = tuple(range(1, 15))  # 1-14
LOWER_JOINTS = tuple(range(15, 23))  # 15-22
ANKLE_FOOT_JOINTS = (17, 18, 21, 22)
HAND_JOINTS = (10, 14)
W_SEGMENT = 160  # W-MPJPE segment: 4 s at 40 Hz
JOINT_LIMIT_RAD = math.radians(15.0)  # bone angle below this is anatomically impossible


def _pa_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame similarity Procrustes alignment (the exact math of
    eval._pa_error, applied to all 23 joints) -> aligned pred (..., J, 3)."""
    n_joints = pred.shape[-2]
    x = pred.reshape(-1, n_joints, 3)
    y = target.reshape(-1, n_joints, 3)
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x0 = x - x_mean
    y0 = y - y_mean
    covariance = x0.transpose(1, 2) @ y0
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[:, -1] = torch.where(
        torch.det(u @ vh) < 0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    rotation = u @ torch.diag_embed(correction) @ vh
    variance = x0.square().sum(dim=(1, 2)).clamp_min(1.0e-8)
    scale = (singular * correction).sum(dim=1) / variance
    aligned = scale[:, None, None] * (x0 @ rotation) + y_mean
    return aligned.view_as(pred)


def _circ_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _root_rotmats(pose: torch.Tensor) -> torch.Tensor:
    """(T, 138) 6D pose -> world root rotation matrices (T, 3, 3)."""
    return rot6d_to_rotmat(pose[None].reshape(1, -1, N_JOINTS, 6))[0, :, 0]


def _pick_forward_axis(dataset, device: torch.device) -> int:
    """Choose the local forward axis, +Z (2) or +X (0), that best explains the
    GT: the winner is the axis whose GT root yaw tracks the GT horizontal
    velocity heading with the smaller mean circular error.  This keeps the yaw
    metrics independent of the skeleton's facing convention; F2a settles the
    axis with a unit test and the protocol stays data-driven until then."""
    scores = {0: 0.0, 2: 0.0}
    weights = 0.0
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample = dataset[i]
            pose = sample["pose_gt"].to(device)
            trans_gt = sample["trans_gt"].to(device) + sample["trans_anchor"].to(device)
            R = _root_rotmats(pose)
            vel = trans_gt[1:] - trans_gt[:-1]
            if vel.shape[0] == 0:
                continue
            head = torch.atan2(vel[:, 0], vel[:, 2])
            for axis in (0, 2):
                yaw = torch.atan2(R[1:, :, axis][:, 0], R[1:, :, axis][:, 2])
                scores[axis] += float(_circ_diff(yaw, head).abs().mean().item()) * vel.shape[0]
            weights += float(vel.shape[0])
    if weights <= 0.0:
        return 2  # no motion anywhere: default to +Z
    return 2 if scores[2] <= scores[0] else 0


def _cop_grid(values: torch.Tensor) -> torch.Tensor:
    """Per-foot CoP in grid units from 96-dim rows (left48, right48) — the
    torch twin of pressure.cop_from_grid (x along the 4 rows / 3, y along the
    12 heel-to-toe cols / 11)."""
    grid = values.reshape(-1, 2, 4, 12).clamp(min=0.0)
    force = grid.sum(dim=(2, 3))
    rows = torch.arange(4, device=values.device, dtype=torch.float32)
    cols = torch.arange(12, device=values.device, dtype=torch.float32)
    cop_x = (grid.sum(dim=3) * rows).sum(dim=2) / force.clamp_min(1e-6) / 3.0
    cop_y = (11.0 - (grid.sum(dim=2) * cols).sum(dim=2) / force.clamp_min(1e-6)) / 11.0
    cop = torch.stack([cop_x, cop_y], dim=-1)
    return torch.where((force > 0).unsqueeze(-1), cop, torch.zeros_like(cop))


def _bone_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Angle at joint b between the bones (a-b) and (b-c), in radians."""
    v1 = F.normalize(a - b, dim=-1)
    v2 = F.normalize(c - b, dim=-1)
    cos = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


class _Accum:
    """Per-name sums/counts; means() returns the protocol dict."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def add(self, name: str, values: torch.Tensor, scale: float = 1.0) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(values.double().sum().item()) * scale
        self.counts[name] = self.counts.get(name, 0.0) + float(values.numel())

    def add_scalar(self, name: str, value: float, weight: float = 1.0) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * weight
        self.counts[name] = self.counts.get(name, 0.0) + weight

    def means(self) -> Dict[str, float]:
        return {name: self.sums[name] / max(self.counts[name], 1e-12) for name in self.sums}


def _session_metrics(seq: dict, tw: int, forward_axis: int, config_value: int,
                     accum: _Accum, contact: Dict[str, int]) -> None:
    """All protocol metrics of one session's full-sequence prediction."""
    T = seq["pred_pose"].shape[0]
    kp_gt = seq["kp_gt"]
    pred_kp = fk_pose6d(
        seq["pred_pose"][None], seq["pred_trans"][None], seq["offsets"], seq["parents"]
    )[0]

    # ---- local pose: MPJPE / PA-MPJPE, per part + aggregates ----
    err = torch.linalg.vector_norm(pred_kp - kp_gt, dim=-1)  # (T, 23)
    accum.add("MPJPE", err.mean(dim=-1), 1000.0)
    for pname, joints in zip(PART_NAMES, PART_JOINTS):
        accum.add("MPJPE_part_%s" % pname, err[:, joints].mean(dim=-1), 1000.0)
    accum.add("MPJPE_upper", err[:, UPPER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_lower", err[:, LOWER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_anklefoot", err[:, ANKLE_FOOT_JOINTS].mean(dim=-1), 1000.0)
    accum.add("MPJPE_hands", err[:, HAND_JOINTS].mean(dim=-1), 1000.0)

    aligned = _pa_align(pred_kp, kp_gt)
    pa_err = torch.linalg.vector_norm(aligned - kp_gt, dim=-1)
    accum.add("PA-MPJPE", pa_err.mean(dim=-1), 1000.0)
    for pname, joints in zip(PART_NAMES, PART_JOINTS):
        accum.add("PA-MPJPE_part_%s" % pname, pa_err[:, joints].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_upper", pa_err[:, UPPER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_lower", pa_err[:, LOWER_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_anklefoot", pa_err[:, ANKLE_FOOT_JOINTS].mean(dim=-1), 1000.0)
    accum.add("PA-MPJPE_hands", pa_err[:, HAND_JOINTS].mean(dim=-1), 1000.0)

    # ---- global ----
    if T >= W_SEGMENT:
        n_seg = T // W_SEGMENT
        seg_p = pred_kp[: n_seg * W_SEGMENT].reshape(n_seg, W_SEGMENT, N_JOINTS, 3)
        seg_g = kp_gt[: n_seg * W_SEGMENT].reshape(n_seg, W_SEGMENT, N_JOINTS, 3)
        w_err = torch.linalg.vector_norm(
            (seg_p - seg_p[:, :1]) - (seg_g - seg_g[:, :1]), dim=-1
        )
        accum.add("W-MPJPE", w_err, 1000.0)
    pred_rel = seq["pred_trans"] - seq["pred_trans"][:1]
    gt_rel = seq["gt_trans"] - seq["gt_trans"][:1]
    step = torch.linalg.vector_norm(seq["gt_trans"][1:] - seq["gt_trans"][:-1], dim=-1)
    if step.numel() > 0:
        rte = torch.linalg.vector_norm(pred_rel - gt_rel, dim=-1).mean() / step.mean().clamp_min(1e-3)
        accum.add_scalar("RTE_norm", float(rte.item()), float(T))
    R_p = _root_rotmats(seq["pred_pose"])
    R_g = _root_rotmats(seq["gt_pose"])
    yaw_p = torch.atan2(R_p[:, :, forward_axis][:, 0], R_p[:, :, forward_axis][:, 2])
    yaw_g = torch.atan2(R_g[:, :, forward_axis][:, 0], R_g[:, :, forward_axis][:, 2])
    accum.add("yaw_abs_deg", _circ_diff(yaw_p, yaw_g).abs() * 180.0 / math.pi)
    accum.add(
        "yaw_drift_deg",
        _circ_diff(yaw_p - yaw_p[0], yaw_g - yaw_g[0]).abs() * 180.0 / math.pi,
    )

    # ---- temporal ----
    d_pred = torch.linalg.vector_norm(pred_kp[1:] - pred_kp[:-1], dim=-1).mean(dim=-1)
    d_gt = torch.linalg.vector_norm(kp_gt[1:] - kp_gt[:-1], dim=-1).mean(dim=-1)
    accum.add("jitter_mm", d_pred, 1000.0)
    accum.add("jitter_gt_mm", d_gt, 1000.0)
    if T >= 3:
        a_pred = (pred_kp[2:] - 2.0 * pred_kp[1:-1] + pred_kp[:-2]) * FPS**2
        a_gt = (kp_gt[2:] - 2.0 * kp_gt[1:-1] + kp_gt[:-2]) * FPS**2
        accum.add("accel_err_ms2", torch.linalg.vector_norm(a_pred - a_gt, dim=-1))
        accum.add("accel_mag_ms2", torch.linalg.vector_norm(a_pred, dim=-1))
        accum.add("accel_mag_gt_ms2", torch.linalg.vector_norm(a_gt, dim=-1))
        accum.add("accel_mag_upper_ms2", torch.linalg.vector_norm(a_pred[:, UPPER_JOINTS], dim=-1))
        accum.add("accel_mag_upper_gt_ms2", torch.linalg.vector_norm(a_gt[:, UPPER_JOINTS], dim=-1))
    if T > tw:
        seams = torch.arange(tw, T, tw, device=pred_kp.device)
        accum.add("seam_jump_mm", torch.linalg.vector_norm(pred_kp[seams] - pred_kp[seams - 1], dim=-1), 1000.0)
        accum.add("seam_jump_gt_mm", torch.linalg.vector_norm(kp_gt[seams] - kp_gt[seams - 1], dim=-1), 1000.0)

    # ---- contact ----
    pred_contact = soft_contact_from_keypoints(pred_kp[None])[0] > 0.5  # (T, 2)
    gt_contact = seq["contact_gt"] > 0.5
    contact["tp"] += int((pred_contact & gt_contact).sum().item())
    contact["fp"] += int((pred_contact & ~gt_contact).sum().item())
    contact["fn"] += int((~pred_contact & gt_contact).sum().item())
    contact["tn"] += int((~pred_contact & ~gt_contact).sum().item())
    foot = pred_kp[:, FOOT_JOINTS][..., [0, 2]]  # world-horizontal XY
    speed = torch.linalg.vector_norm(foot[1:] - foot[:-1], dim=-1).mean(dim=-1) * 1000.0  # mm/帧
    pred_stance = pred_contact[1:].any(dim=-1)
    gt_stance = gt_contact[1:].any(dim=-1)
    if bool(pred_stance.any()):
        accum.add("foot_slide_mm", speed[pred_stance])
    if bool(gt_stance.any()):
        accum.add("foot_slide_gt_mm", speed[gt_stance])

    # ---- joint limits (elbow/knee bone angle on the PREDICTED pose: the
    # plausibility check is that generated upper bodies don't fold joints
    # past what anatomy allows) ----
    viol = (
        (_bone_angle(pred_kp[:, 7], pred_kp[:, 8], pred_kp[:, 9]) < JOINT_LIMIT_RAD)
        | (_bone_angle(pred_kp[:, 11], pred_kp[:, 12], pred_kp[:, 13]) < JOINT_LIMIT_RAD)
    ).float()
    accum.add("joint_limit_viol_elbow", viol)
    viol_knee = (
        (_bone_angle(pred_kp[:, 15], pred_kp[:, 16], pred_kp[:, 17]) < JOINT_LIMIT_RAD)
        | (_bone_angle(pred_kp[:, 19], pred_kp[:, 20], pred_kp[:, 21]) < JOINT_LIMIT_RAD)
    ).float()
    accum.add("joint_limit_viol_knee", viol_knee)

    # ---- pressure output quality (V2M only: T input is zeroed, so these
    # measure vision-to-tactile generation).  F5 part9 drops the aux heads
    # and skips these rows. ----
    if config_value == CONFIG_V and seq.get("pressure_pred") is not None:
        f_pred = seq["pressure_pred"].sum(dim=-1)
        f_gt = seq["pressure_gt"].sum(dim=-1)
        ss_res = float(((f_pred - f_gt) ** 2).sum().item())
        ss_tot = float(((f_gt - f_gt.mean()) ** 2).sum().item())
        accum.add_scalar("pressure_force_r2", 1.0 - ss_res / max(ss_tot, 1e-12), float(T))
        cop_p = _cop_grid(seq["pressure_pred"])
        cop_g = _cop_grid(seq["pressure_gt"])
        accum.add("pressure_cop_err", torch.linalg.vector_norm(cop_p - cop_g, dim=-1))
        corr, corr_valid = _tactile_corr(seq["pressure_pred"], seq["pressure_gt"])
        if bool(corr_valid.any()):
            accum.add("pressure_pearson", corr[corr_valid])


def _degrade_inputs(v_feat, t_raw, t_phys, t_s2m, tw: int, kind: str, rng: np.random.RandomState):
    """Zero one contiguous 20-40% span of frames on the V or T stream
    (robustness set).  Windows are non-overlapping, so window w covers session
    frames [w*tw, (w+1)*tw)."""
    n_windows = v_feat.shape[0]
    n_frames = n_windows * tw
    frac = float(rng.uniform(0.2, 0.4))
    start = int(rng.uniform(0.0, max(n_frames - int(frac * n_frames), 1)))
    end = start + int(frac * n_frames)
    frames = torch.arange(n_frames).view(n_windows, tw)
    mask = ((frames >= start) & (frames < end)).to(device=v_feat.device)
    v, tr, tp, ts = v_feat.clone(), t_raw.clone(), t_phys.clone(), t_s2m.clone()
    if kind == "v":
        v[mask] = 0.0
    else:
        tr[mask] = 0.0
        tp[mask] = 0.0
        ts[mask] = 0.0
    return v, tr, tp, ts

def _evaluate_config(
    dataset, model, device, config_value: int, regress_mode: bool, diffusion,
    sample_steps: int, warm_start: bool, tw: int, forward_axis: int,
    degrade: Optional[str] = None, seed: int = 0,
) -> Dict[str, float]:
    """Full-session inference for one conditioning config, then all protocol
    metrics.  ``degrade`` in ("v", "t") adds the robustness span dropout."""
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel.
    v_hmr_mode = str(getattr(model, "v_input", "hrnet")) == "hmr_gvhmr"
    # F2a: heading/tilt representation.
    f2_repr = bool(getattr(model, "f2_repr", False))
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    groups = _session_window_groups(dataset)
    with torch.inference_mode():
        for session_id, idxs in groups.items():
            raw = [dataset[i] for i in idxs]
            batch = move_batch(collate_windows(raw), device)
            bsz = batch["pose_gt"].shape[0]
            config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            if degrade is not None:
                rng = np.random.RandomState((seed * 1000003) ^ zlib.crc32(str(session_id).encode()))
                v_feat, t_raw, t_phys, t_s2m = _degrade_inputs(
                    v_feat, t_raw, t_phys, t_s2m, tw, degrade, rng
                )
            v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
            if regress_mode:
                out = model(v_feat, t_raw, t_phys, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                pred_pose = out["x0_hat"]
            else:
                cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                        "T_s2m": t_s2m, "config_id": config_id,
                        "session_id": batch.get("session_id")}
                prior = model.pose_head.pose_mean.view(1, 1, -1).expand(bsz, tw, -1) if warm_start else None
                x_T = _sample_x_t_init(model, bsz, tw, device, prior=prior)
                pred_pose = diffusion.ddim_sample_loop(
                    model, x_T=x_T, tau_related_kwargs=cond,
                    steps=sample_steps, eta=0.0, device=device)
                tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
                out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            anchor = batch["trans_anchor"][:, None, :]
            if f2_repr:
                # F2a: world pose/trans recovered from the tilt pose + heading
                # trajectory before the protocol metrics (same FK scope).
                pred_pose_w, pred_trans_w = f2_to_world(
                    pred_pose, out["v_hat"], batch["psi_anchor"], batch["trans_anchor"]
                )
                gt_pose_w, _ = f2_to_world(
                    batch["pose_gt"], batch["traj_gt_f2"], batch["psi_anchor"],
                    batch["trans_anchor"],
                )
                seq = {
                    "pred_pose": pred_pose_w.reshape(-1, pred_pose_w.shape[-1]),
                    "pred_trans": pred_trans_w.reshape(-1, 3),
                    "gt_pose": gt_pose_w.reshape(-1, gt_pose_w.shape[-1]),
                    "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3),
                    "kp_gt": batch["kp_gt"].reshape(-1, N_JOINTS, 3),
                    "contact_gt": batch["contact_gt"].reshape(-1, 2),
                    "pressure_gt": batch["T_raw"].reshape(-1, batch["T_raw"].shape[-1]),
                    "pressure_pred": (out["pressure_hat"].reshape(-1, out["pressure_hat"].shape[-1])
                                      if out.get("pressure_hat") is not None else None),
                    "offsets": batch["offsets"][0],
                    "parents": batch["parents"][0],
                }
            else:
                seq = {
                    "pred_pose": pred_pose.reshape(-1, pred_pose.shape[-1]),
                    "pred_trans": (out["trans_hat"] + anchor).reshape(-1, 3),
                    "gt_pose": batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1]),
                    "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3),
                    "kp_gt": batch["kp_gt"].reshape(-1, N_JOINTS, 3),
                    "contact_gt": batch["contact_gt"].reshape(-1, 2),
                    "pressure_gt": batch["T_raw"].reshape(-1, batch["T_raw"].shape[-1]),
                    "pressure_pred": (out["pressure_hat"].reshape(-1, out["pressure_hat"].shape[-1])
                                      if out.get("pressure_hat") is not None else None),
                    "offsets": batch["offsets"][0],
                    "parents": batch["parents"][0],
                }
            _session_metrics(seq, tw, forward_axis, config_value, accum, contact)
    result = accum.means()
    p = contact["tp"] / max(contact["tp"] + contact["fp"], 1)
    r = contact["tp"] / max(contact["tp"] + contact["fn"], 1)
    result["contact_f1"] = 2.0 * p * r / max(p + r, 1e-8)
    result["contact_acc"] = (contact["tp"] + contact["tn"]) / max(
        contact["tp"] + contact["fp"] + contact["fn"] + contact["tn"], 1
    )
    if "accel_mag_upper_ms2" in result:
        result["accel_dist_err_upper_ms2"] = abs(
            result["accel_mag_upper_ms2"] - result["accel_mag_upper_gt_ms2"]
        )
    return result


def run_protocol(
    checkpoint: dict,
    config: dict,
    model,
    dataset,
    device: torch.device,
    *,
    checkpoint_path: str,
    split: str,
    config_values: List[int],
    regress_mode: bool,
    diffusion,
    sample_steps: int,
    warm_start: bool,
    seed: int,
    robustness: bool,
    contact_method: str,
    tw: int,
    out_path: Path,
) -> None:
    """Run the full F0a protocol and write metrics/<split>_fseries.json."""
    if dataset.stride != dataset.window_length:
        print("protocol skipped: dataset stride %d != window %d (E6.4 continuation eval)"
              % (dataset.stride, dataset.window_length))
        return
    forward_axis = _pick_forward_axis(dataset, device)
    metrics: Dict[str, Dict[str, float]] = {}
    # Deterministic protocol runs.  Sigma-seed for diffusion models comes from
    # three runs with --protocol-seed 0/1/2; regression models have no
    # sampling randomness (dropout is off in eval mode).
    prev_cpu = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        for config_value in config_values:
            name = CONFIG_MODE_NAMES[config_value]
            metrics[name] = _evaluate_config(
                dataset, model, device, config_value, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis,
            )
            print("protocol %s: MPJPE=%.3fmm PA-MPJPE=%.3fmm W-MPJPE=%.3fmm "
                  "contact_f1=%.4f foot_slide=%.2fmm/帧"
                  % (name, metrics[name]["MPJPE"], metrics[name]["PA-MPJPE"],
                     metrics[name].get("W-MPJPE", float("nan")),
                     metrics[name].get("contact_f1", float("nan")),
                     metrics[name].get("foot_slide_mm", float("nan"))))
        if robustness and CONFIG_VT in config_values:
            metrics["robust_vdrop"] = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="v", seed=seed,
            )
            metrics["robust_tdrop"] = _evaluate_config(
                dataset, model, device, CONFIG_VT, regress_mode, diffusion,
                sample_steps, warm_start, tw, forward_axis, degrade="t", seed=seed,
            )
            print("protocol robust_vdrop: MPJPE=%.3fmm PA-MPJPE=%.3fmm contact_f1=%.4f"
                  % (metrics["robust_vdrop"]["MPJPE"], metrics["robust_vdrop"]["PA-MPJPE"],
                     metrics["robust_vdrop"]["contact_f1"]))
            print("protocol robust_tdrop: MPJPE=%.3fmm PA-MPJPE=%.3fmm contact_f1=%.4f"
                  % (metrics["robust_tdrop"]["MPJPE"], metrics["robust_tdrop"]["PA-MPJPE"],
                     metrics["robust_tdrop"]["contact_f1"]))
    finally:
        torch.set_rng_state(prev_cpu)
        if prev_cuda is not None:
            torch.cuda.set_rng_state_all(prev_cuda)
    payload = {
        "checkpoint": checkpoint_path,
        "modal": str(checkpoint.get("config", {}).get("modal", "anysolev1")),
        "contact_method": contact_method,
        "split": split,
        "protocol_seed": seed,
        "tw": tw,
        "sample_steps": sample_steps,
        "robustness": bool(robustness and CONFIG_VT in config_values),
        "forward_axis": "+Z" if forward_axis == 2 else "+X",
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical file (回填表口径) always reflects the latest run; a per-seed
    # copy is archived so the three --protocol-seed 0/1/2 runs can be diffed
    # for sigma-seed without overwriting each other.
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path = out_path.with_name("%s_seed%d.json" % (out_path.stem, seed))
    seed_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s (+ %s)" % (out_path, seed_path.name))
