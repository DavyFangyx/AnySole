"""Training losses for AnySole V1."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from anysole.geometry import f2_to_world, fk_pose6d
from anysole.types import (
    CONFIG_T,
    FPS,
    JOINT_PARENTS,
    LEFT_FOOT_JOINT,
    LEFT_TOE_JOINT,
    N_JOINTS,
    RIGHT_FOOT_JOINT,
    RIGHT_TOE_JOINT,
)


DEFAULT_TRAJ_DELTAS = (2, 4, 8, 19)


def soft_contact_from_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """Compute differentiable left/right contact probabilities from world-space joints."""
    foot_pairs = (
        (LEFT_FOOT_JOINT, LEFT_TOE_JOINT),
        (RIGHT_FOOT_JOINT, RIGHT_TOE_JOINT),
    )
    contacts = []
    for joint_ids in foot_pairs:
        foot = keypoints[:, :, joint_ids, :]
        height = foot[..., 1].amin(dim=2)
        if foot.shape[1] > 1:
            speed_rest = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1).mean(dim=2)
            speed = torch.cat([speed_rest[:, :1], speed_rest], dim=1) * FPS
        else:
            speed = height.new_zeros(height.shape)
        height_prob = torch.sigmoid((0.05 - height) / 0.02)
        speed_prob = torch.sigmoid((0.20 - speed) / 0.05)
        contacts.append(height_prob * speed_prob)
    return torch.stack(contacts, dim=-1)


def soft_contact_from_pose(
    pose_6d: torch.Tensor,
    trans: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    keypoints = fk_pose6d(pose_6d, trans, offsets, parents)
    return soft_contact_from_keypoints(keypoints)


def _weight(weights: Mapping[str, float], name: str, default: float) -> float:
    value = weights.get(name, default)
    return float(value)


def _trajectory_losses(out, batch, weights):
    """Return velocity and multi-scale relative-displacement losses.

    ``v_hat`` and ``vel_gt`` are m/s.  ``trans_hat``/``trans_gt`` are the
    corresponding relative positions in meters, so each displacement term is
    translation-invariant and compares a distinct temporal scale.
    """
    v_hat = out["v_hat"]
    vel_gt = batch["vel_gt"]
    x_hat = out["trans_hat"]
    # Reconstruct the target from the velocity definition so the two targets
    # cannot silently drift apart if a caller builds a batch manually.
    x_gt = torch.cumsum(vel_gt, dim=1) / float(FPS)

    l_vel = F.mse_loss(v_hat, vel_gt)
    deltas = weights.get("traj_deltas", DEFAULT_TRAJ_DELTAS)
    if isinstance(deltas, str):
        deltas = tuple(int(value.strip()) for value in deltas.split(",") if value.strip())
    else:
        deltas = tuple(int(value) for value in deltas)
    valid = tuple(delta for delta in deltas if 1 <= delta < x_hat.shape[1])
    if not valid:
        raise ValueError("traj_deltas must contain an integer in [1, Tw-1]")
    power = float(weights.get("traj_delta_weight_power", 1.0))
    delta_terms = []
    for delta in valid:
        pred_delta = x_hat[:, delta:] - x_hat[:, :-delta]
        gt_delta = x_gt[:, delta:] - x_gt[:, :-delta]
        scale_weight = 1.0 / (float(delta) ** power)
        delta_terms.append(scale_weight * F.mse_loss(pred_delta, gt_delta))
    l_delta = torch.stack(delta_terms).mean()
    return (
        _weight(weights, "traj_velocity_w", 1.0) * l_vel
        + _weight(weights, "traj_delta_w", 1.0) * l_delta,
        l_vel,
        l_delta,
    )


def _f2_trajectory_losses(out, batch, weights):
    """F2a trajectory losses on the 4-dim heading-frame target.

    Velocity MSE over all 4 dims (normalized by the fitted per-dim stats so
    psi_dot / v_h / h share one scale); multi-scale displacement on the
    INTEGRATED heading-frame trajectory (first 3 dims), as per fix_plan_v2.md
    §F2a item 6.
    """
    v_hat = out["v_hat"]
    target = batch["traj_gt_f2"]
    stats = weights.get("traj_f2_stats")
    if isinstance(stats, dict):
        mean = torch.as_tensor(stats["mean"], dtype=v_hat.dtype, device=v_hat.device).view(1, 1, 4)
        std = torch.as_tensor(stats["std"], dtype=v_hat.dtype, device=v_hat.device).view(1, 1, 4)
    else:
        mean = v_hat.new_zeros(1, 1, 4)
        std = v_hat.new_ones(1, 1, 4)
    v_hat_n = (v_hat - mean) / std
    target_n = (target - mean) / std
    l_vel = F.mse_loss(v_hat_n, target_n)
    x_hat = torch.cumsum(v_hat_n[..., :3], dim=1) / float(FPS)
    x_gt = torch.cumsum(target_n[..., :3], dim=1) / float(FPS)
    deltas = weights.get("traj_deltas", DEFAULT_TRAJ_DELTAS)
    if isinstance(deltas, str):
        deltas = tuple(int(value.strip()) for value in deltas.split(",") if value.strip())
    else:
        deltas = tuple(int(value) for value in deltas)
    valid = tuple(delta for delta in deltas if 1 <= delta < x_hat.shape[1])
    if not valid:
        raise ValueError("traj_deltas must contain an integer in [1, Tw-1]")
    power = float(weights.get("traj_delta_weight_power", 1.0))
    delta_terms = []
    for delta in valid:
        pred_delta = x_hat[:, delta:] - x_hat[:, :-delta]
        gt_delta = x_gt[:, delta:] - x_gt[:, :-delta]
        scale_weight = 1.0 / (float(delta) ** power)
        delta_terms.append(scale_weight * F.mse_loss(pred_delta, gt_delta))
    l_delta = torch.stack(delta_terms).mean()
    return (
        _weight(weights, "traj_velocity_w", 1.0) * l_vel
        + _weight(weights, "traj_delta_w", 1.0) * l_delta,
        l_vel,
        l_delta,
    )


def compute_losses(out, batch, config_id, weights) -> dict:
    """Return all frozen V1 loss terms as scalar tensors."""
    pose_gt = batch["pose_gt"]
    trans_gt = batch["trans_gt"]
    # E6.1: pose_repr "pos" switches the pose target to root-local positions.
    # 6D is the default and keeps the original MSE loss.
    pos_mode = str(weights.get("pose_repr", "6d")) == "pos"
    # F2a: heading/tilt representation — the pose target's root 6D is the
    # tilt, and the trajectory loss + FK supervision use f2_to_world.
    f2_mode = bool(weights.get("f2_repr", False)) and not pos_mode

    if pos_mode:
        l_pose = F.l1_loss(out["x0_hat"], batch["pose_gt_pos"])
        target = batch["pose_gt_pos"]
    else:
        l_pose = F.mse_loss(out["x0_hat"], pose_gt)
        target = pose_gt
    if f2_mode:
        l_traj, l_traj_vel, l_traj_delta = _f2_trajectory_losses(out, batch, weights)
    else:
        l_traj, l_traj_vel, l_traj_delta = _trajectory_losses(out, batch, weights)
    l_trec = (
        F.mse_loss(out["pressure_hat"], batch["T_raw"])
        if out.get("pressure_hat") is not None
        else l_pose.new_zeros(())
    )

    # E6.2: frame-to-frame smoothness of the diffused target (works for both
    # representations; default weight 0 keeps the E5 behavior).
    diff_pred = out["x0_hat"][:, 1:] - out["x0_hat"][:, :-1]
    diff_gt = target[:, 1:] - target[:, :-1]
    l_pose_vel = F.mse_loss(diff_pred, diff_gt)

    # E6.2: bone-length rigidity (pos mode only - positions are the output
    # space, so rigid skeleton lengths are a direct constraint; default 0).
    l_bone = l_pose.new_zeros(())
    if pos_mode:
        pos3 = out["x0_hat"].reshape(*out["x0_hat"].shape[:2], 22, 3)
        par_ids = [int(JOINT_PARENTS[j]) - 1 for j in range(1, N_JOINTS)]
        ppos = torch.zeros_like(pos3)
        for k, p in enumerate(par_ids):
            if p >= 0:
                ppos[..., k, :] = pos3[..., p, :]
        len_pred = torch.linalg.vector_norm(pos3 - ppos, dim=-1)
        len_gt = torch.linalg.vector_norm(batch["offsets"][:, 1:, :], dim=-1)
        l_bone = F.mse_loss(len_pred, len_gt.unsqueeze(1).expand_as(len_pred))

    vrec_per_sample = (
        (out["vfeat_hat"] - batch["V_feat"]).square().flatten(1).mean(dim=1)
        if out.get("vfeat_hat") is not None
        else l_pose.new_zeros((batch["V_feat"].shape[0],))
    )
    vrec_mask = config_id.to(device=vrec_per_sample.device).reshape(-1) == CONFIG_T
    if bool(vrec_mask.any()):
        l_vrec = vrec_per_sample[vrec_mask].mean()
    else:
        l_vrec = vrec_per_sample.new_zeros(())

    # Geometry/contact remain in world coordinates.  The anchor is excluded
    # from L_traj but restores the world frame for these auxiliary terms.
    anchor = batch.get("trans_anchor")
    if anchor is None:
        anchor = trans_gt.new_zeros((trans_gt.shape[0], 3))
    trans_world = trans_gt + anchor[:, None, :]
    if pos_mode:
        # The E6 series keeps the FK keypoint supervision of the 6D lineage
        # (lambda_kp): world kp = R_init @ local + trans, with the session
        # frame-0 rotation from the dataset.  Contact loss stays disabled
        # (lambda_con=0, E3/E5 convention) in pos experiments.
        world = torch.einsum(
            "bij,btpj->btpi", batch["root_rot_init"], pos3
        ) + trans_world.unsqueeze(2)
        pred_kp = torch.cat([trans_world.unsqueeze(2), world], dim=2)
        l_kp = F.mse_loss(pred_kp, batch["kp_gt"])
        l_con = l_pose.new_zeros(())
    else:
        if f2_mode:
            # World pose/trans recovered from the tilt pose + heading-frame
            # trajectory (geometry.f2_to_world); FK supervision then compares
            # the same world keypoints as the V1 path.
            world_pose, world_trans = f2_to_world(
                out["x0_hat"], out["v_hat"], batch["psi_anchor"], anchor
            )
            pred_kp = fk_pose6d(world_pose, world_trans, batch["offsets"], batch["parents"])
        else:
            pred_kp = fk_pose6d(out["x0_hat"], trans_world, batch["offsets"], batch["parents"])
        l_kp = F.mse_loss(pred_kp, batch["kp_gt"])
        # Contact: computed only when its weight is nonzero — a zero-weighted
        # BCE is pure cost AND a NaN risk (cold-start outputs produce extreme
        # poses; inf-inf NaN in the soft-contact speed can then feed the BCE).
        # The clamp is belt-and-suspenders for the enabled case.
        if _weight(weights, "lambda_con", 0.1) > 0.0:
            soft_contact = soft_contact_from_keypoints(pred_kp)
            soft_contact = soft_contact.nan_to_num(0.0).clamp(1e-7, 1.0 - 1e-7)
            l_con = F.binary_cross_entropy(soft_contact, batch["contact_gt"])
        else:
            l_con = l_pose.new_zeros(())

    total = (
        _weight(weights, "lambda_pose", 1.0) * l_pose
        + _weight(weights, "lambda_traj", 1.0) * l_traj
        + _weight(weights, "lambda_trec", 0.1) * l_trec
        + _weight(weights, "lambda_vrec", 0.1) * l_vrec
        + _weight(weights, "lambda_kp", 1.0) * l_kp
        + _weight(weights, "lambda_con", 0.1) * l_con
        + _weight(weights, "lambda_pose_vel", 0.0) * l_pose_vel
        + _weight(weights, "lambda_bone", 0.0) * l_bone
    )
    return {
        "L_pose": l_pose,
        "L_traj": l_traj,
        "L_traj_vel": l_traj_vel,
        "L_traj_delta": l_traj_delta,
        "L_Trec": l_trec,
        "L_Vrec": l_vrec,
        "L_kp": l_kp,
        "L_con": l_con,
        "L_pose_vel": l_pose_vel,
        "L_bone": l_bone,
        "loss": total,
    }
