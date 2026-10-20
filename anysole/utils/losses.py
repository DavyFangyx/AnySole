"""Training losses for AnySole V1."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from anysole.utils.geometry import f2_to_world, fk_pose6d
from anysole.types import (
    CONFIG_T,
    FPS,
    JOINT_PARENTS,
    KNEE_ANKLE_FOOT_JOINTS,
    LEFT_FOOT_JOINT,
    LEFT_TOE_JOINT,
    N_JOINTS,
    RIGHT_FOOT_JOINT,
    RIGHT_TOE_JOINT,
)


DEFAULT_TRAJ_DELTAS = (2, 4, 8, 19)


def soft_contact_from_keypoints(
    keypoints: torch.Tensor, floor_y: torch.Tensor | None = None
) -> torch.Tensor:
    """Differentiable contact probability using height above session floor.

    ``keypoints`` is ``(B,T,J,3)`` in native SMPL Y-up coordinates.  SMPL
    foot joints are above the sole and world Y=0 is not a guaranteed floor,
    so callers using real data must pass the GT-derived per-session
    ``floor_y``.  ``None`` remains available for isolated synthetic probes.
    """
    foot_pairs = (
        (LEFT_FOOT_JOINT, LEFT_TOE_JOINT),
        (RIGHT_FOOT_JOINT, RIGHT_TOE_JOINT),
    )
    contacts = []
    for joint_ids in foot_pairs:
        foot = keypoints[:, :, joint_ids, :]
        height = foot[..., 1].amin(dim=2)
        if floor_y is not None:
            floor = torch.as_tensor(floor_y, device=height.device, dtype=height.dtype)
            if floor.ndim == 0:
                floor = floor.view(1, 1)
            elif floor.ndim == 1:
                floor = floor[:, None]
            elif floor.ndim != 2:
                raise ValueError("floor_y must be scalar, (B,), or (B,1)")
            height = height - floor
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
    floor_y: torch.Tensor | None = None,
) -> torch.Tensor:
    keypoints = fk_pose6d(pose_6d, trans, offsets, parents)
    return soft_contact_from_keypoints(keypoints, floor_y=floor_y)


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


def _cluster_assign_terms(assign, anneal, dead_beta):
    """V4A cluster L_assign terms from a (N_PARTS, N_JOINTS) assignment A:
    annealed per-joint entropy (soft early so the grouping can reorganize),
    shifted concentration reward N^2 - sum(mass^2) (>= 0; K emerges from the
    recon-loss tension), and the dead-slot tax sum(mass * exp(-beta*mass))
    whose gradient rounds crumb loads (0 < mass < 1/beta) down to exactly 0.
    Pure functions so smoke_v4a_cluster24.py can verify the signs directly."""
    mass = assign.sum(dim=1)                          # (N_PARTS,) slot loads
    l_ent = anneal * -(assign * (assign + 1e-12).log()).sum(dim=0).mean()
    l_conc = float(assign.shape[1]) ** 2 - (mass * mass).sum()
    l_dead = (mass * torch.exp(-dead_beta * mass)).sum()
    return l_ent, l_conc, l_dead


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
        pos3 = out["x0_hat"].reshape(*out["x0_hat"].shape[:2], N_JOINTS - 1, 3)
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
        l_pos = l_pose.new_zeros(())  # pos-mode models have no position head
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
        # V3_3D: L_pos — the position head's corrected lower-limb keypoints
        # (soft skeleton).  World-frame MSE is equivalent to the pelvis-
        # relative probe target for the head's gradient (the pelvis term is
        # constant w.r.t. pos_corr) and needs no pelvis bookkeeping.  Zero
        # for every model without the head (bit-identical total elsewhere).
        if out.get("pos_corr") is not None:
            six = KNEE_ANKLE_FOOT_JOINTS
            l_pos = F.mse_loss(
                pred_kp[..., six, :]
                + out["pos_corr"].reshape(*out["pos_corr"].shape[:2], len(six), 3),
                batch["kp_gt"][..., six, :],
            )
        else:
            l_pos = l_pose.new_zeros(())
        # Contact: computed only when its weight is nonzero — a zero-weighted
        # BCE is pure cost AND a NaN risk (cold-start outputs produce extreme
        # poses; inf-inf NaN in the soft-contact speed can then feed the BCE).
        # The clamp is belt-and-suspenders for the enabled case.
        if _weight(weights, "lambda_con", 0.1) > 0.0:
            soft_contact = soft_contact_from_keypoints(pred_kp, batch["floor_y"])
            soft_contact = soft_contact.nan_to_num(0.0).clamp(1e-7, 1.0 - 1e-7)
            l_con = F.binary_cross_entropy(soft_contact, batch["contact_gt"])
        else:
            l_con = l_pose.new_zeros(())

    # V3-3 soft assignment regularization (fix_plan_v3.md §V3-3/§8) + V4A
    # clustering terms.  Common input: the assignment A actually used by the
    # forward (temperature-scaled softmax; model_v2 exports it detached as
    # part_assignment).  Zero for every model without soft_parts.
    if bool(weights.get("soft_parts", False)) and out.get("part_assignment") is not None:
        assign = out["part_assignment"]                       # (N_PARTS, N_JOINTS)
        if bool(weights.get("assign_cluster", False)):
            # V4A (方案 A): discover the grouping from a near-uniform init.
            # The V3-3 pair (per-joint entropy + load balance) hardens onto
            # the init and locks it there (2026-09-21 audit of V3_3A/B), so
            # here the entropy is ANNEALED (assign_anneal: 1 -> 0 over the
            # anneal phase, kept soft early so the assignment can reorganize),
            # balance is REPLACED by a concentration reward + dead-slot tax
            # (K then emerges from the recon-loss tension instead of being
            # forced to n_parts), and A gets its own optimizer lr in train.py.
            l_ent, l_conc, l_dead = _cluster_assign_terms(
                assign,
                float(weights.get("assign_anneal", 1.0)),
                float(weights.get("assign_dead_beta", 2.0)),
            )
            l_assign = (
                float(weights.get("lambda_assign_ent", 0.05)) * l_ent
                + float(weights.get("lambda_assign_conc", 0.01)) * l_conc
                + float(weights.get("lambda_assign_dead", 0.02)) * l_dead
            )
        else:
            # V3-3 original: A must stay sharp (each joint owned by one slot)
            # and balanced (no slot owns everything).  Both terms are
            # entropies of the softmax'd part_logits.  The balance term is
            # -H(mass), whose floor is -log(N); +log(N) shifts it to be
            # non-negative so the total loss stays an error measure
            # (2026-09-21 fix: the raw term made loss_total negative once the
            # warm-started MSE terms converged — a constant shift leaves the
            # gradient untouched).
            log_assign = F.log_softmax(out["part_logits"], dim=0)
            assign_init = log_assign.exp()
            l_ent = -(assign_init * log_assign).sum(dim=0).mean()  # per-joint entropy -> minimize
            mass = assign_init.mean(dim=1)
            l_balance = (mass * (mass + 1e-8).log()).sum() + math.log(mass.numel())
            l_assign = l_ent + l_balance
    else:
        l_assign = l_pose.new_zeros(())

    # V3-3 beta-NLL sigma supervision (plan §3, implemented 2026-09-21 after
    # the V3-3B "static gate" audit): calibrates the per-slot sigma logits to
    # the slot's normalized-space pose error through the gate-mixture variance
    # v = g_V*exp(-sigma_V) + g_T*exp(-sigma_T) — the TRUST convention: high
    # sigma = high trust = low claimed variance, so the gate weight and the
    # calibration share one semantics (equilibrium sigma ~ -log r2: the more
    # accurate a slot, the more trust; "perfect" parts get maximal weight).
    # Two deviations from the plan's letter: (1) the letter supervises both
    # sigma_V and sigma_T with the SAME r_p, which degenerates to sigma_V =
    # sigma_T (no V/T differentiation) — the mixture form attributes the error
    # by the gate's own weights instead; (2) the letter's variance convention
    # (sigma = log variance) inverts the gate semantics for accurate parts.
    # The error target (per-slot hypotheses), the attribution weights g and
    # the assignment A are stopgrad'd — sigma is the only live parameter.
    # beta = 0.5, sigma clipped to [-3, 6].  Skipped while sigma is frozen
    # (plan: sigma fixed at 1 for the first sigma_freeze_frac of steps) and
    # whenever lambda_sigma = 0.
    l_sigma = l_pose.new_zeros(())
    if (
        out.get("gate_sigma") is not None
        and not bool(out.get("sigma_frozen", False))
        and _weight(weights, "lambda_sigma", 0.0) > 0.0
    ):
        beta = float(weights.get("sigma_beta", 0.5))
        sigma_hat = out["gate_sigma"].clamp(-3.0, 6.0)      # (B, tw, N_PARTS, 2)
        g_streams = out["gate_g"][..., :2]                   # (B, tw, N_PARTS, 2), detach
        var = (g_streams * (-sigma_hat).exp()).sum(dim=-1)   # (B, tw, N_PARTS)
        hypo = out["part_hypotheses"]                        # (N_PARTS, B, tw, N_JOINTS, 6), detach
        assign = out["part_assignment"]                      # (N_PARTS, N_JOINTS)
        target = (pose_gt - out["pose_mean"]) / out["pose_std"]
        target = target.reshape(*target.shape[:2], N_JOINTS, 6)
        err = (hypo - target).square().mean(dim=-1)          # (N_PARTS, B, tw, N_JOINTS)
        # (N_PARTS, 1, 1) — trailing-dim broadcast against (N_PARTS, B, tw).
        mass = assign.sum(dim=-1, keepdim=True).clamp(min=1e-6).unsqueeze(-1)
        r2 = (assign.unsqueeze(1).unsqueeze(2) * err).sum(dim=-1) / mass  # (N_PARTS, B, tw)
        r2 = r2.permute(1, 2, 0)                             # (B, tw, N_PARTS)
        nll = r2 / var.clamp(min=1e-4) + var.clamp(min=1e-4).log()
        l_sigma = beta * nll.mean()

    # ---- V3_3E terms (2026-10-17) ----
    # Contact head: per-foot logits vs f6_soft targets.  Pos-weighted BCE —
    # the contact rate is ~94%, and the constraint layer needs swing-phase
    # air recall (locking swing feet is probe A's failure mode), so the
    # per-batch pos_weight lifts the negatives.
    # V3_3G air weighting (2026-10-17): swing/air frames additionally carry
    # (1 + lambda_air * (1 - gt)) — F's contact head (air_recall 0.51) showed
    # the plain recipe undershoots air recall, and G's observation layer
    # gates on the head, so air recall is its registered precondition.
    l_contact = l_pose.new_zeros(())
    if out.get("contact_logits") is not None and _weight(weights, "lambda_contact", 0.0) > 0.0:
        gt = batch["contact_gt"]
        pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
        pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
        air_w = _weight(weights, "lambda_air", 0.0)
        if air_w > 0.0:
            elem_w = 1.0 + air_w * (1.0 - gt.float())
            l_contact = F.binary_cross_entropy_with_logits(
                out["contact_logits"], gt, pos_weight=pos_weight, weight=elem_w)
        else:
            l_contact = F.binary_cross_entropy_with_logits(
                out["contact_logits"], gt, pos_weight=pos_weight)

    # Gaussian-NLL sigma calibration (same beta-NLL convention as V3-3's
    # l_sigma): the error target is detached, log-sigma is the only live
    # parameter, clamped to [-6, 6].  sigma feeds the constraint layer's
    # inverse-variance fallback weights (w = exp(-2·sigma)).
    def _gauss_nll(mu, log_sig, target):
        s = log_sig.clamp(-6.0, 6.0)
        err = (mu.detach() - target).square()
        return float(weights.get("sigma_beta", 0.5)) * \
            (err * (-2.0 * s).exp() + 2.0 * s).mean()

    l_nll = l_pose.new_zeros(())
    if out.get("sigma_v") is not None and _weight(weights, "lambda_nll", 0.0) > 0.0:
        l_nll = _gauss_nll(out["x0_hat"], out["sigma_v"], pose_gt)
    l_prior_pose = l_pose.new_zeros(())
    l_prior_nll = l_pose.new_zeros(())
    l_prior_traj = l_pose.new_zeros(())
    if out.get("x0_prior") is not None:
        l_prior_pose = F.mse_loss(out["x0_prior"], pose_gt)
        if _weight(weights, "lambda_prior_nll", 0.0) > 0.0:
            l_prior_nll = _gauss_nll(out["x0_prior"], out["sigma_prior"], pose_gt)
        if _weight(weights, "lambda_prior_traj", 0.0) > 0.0:
            l_prior_traj, _, _ = _trajectory_losses(
                {"v_hat": out["v_prior"], "trans_hat": out["trans_prior"]},
                batch, weights)

    # ---- V3_3H_v3 一阶足部残差（2026-10-19）----
    # 残差头的唯一监督：修正后足关节 6D vs GT。基础部分（V 塔的足 6D）
    # detach——V 塔不被 T 残差损失触碰（零税的训练侧保证）；梯度只进
    # foot_res_head 与 T 编码器。
    # 落点槽位：v1–v5 足关节（60:72）；V3_3H_v6 踝关节（42:54，7/8 相邻）
    # ——损失只监督「修正后」落点关节 6D 且基础部分 detach（零税训练侧
    # 保证；V3_3H设计错误裁定.md §1 错误一修复）。
    l_foot_res = l_pose.new_zeros(())
    if (out.get("foot_res") is not None
            and _weight(weights, "lambda_foot_res", 0.0) > 0.0):
        slots = (slice(42, 54) if str(out.get("res_landing", "foot")) == "ankle"
                 else slice(60, 72))
        base = out["x0_hat"][..., slots].detach()
        l_foot_res = F.mse_loss(base + out["foot_res"], pose_gt[..., slots])

    total = (
        _weight(weights, "lambda_pose", 1.0) * l_pose
        + _weight(weights, "lambda_traj", 1.0) * l_traj
        + _weight(weights, "lambda_trec", 0.1) * l_trec
        + _weight(weights, "lambda_vrec", 0.1) * l_vrec
        + _weight(weights, "lambda_kp", 1.0) * l_kp
        + _weight(weights, "lambda_pos", 0.0) * l_pos
        + _weight(weights, "lambda_con", 0.1) * l_con
        + _weight(weights, "lambda_pose_vel", 0.0) * l_pose_vel
        + _weight(weights, "lambda_bone", 0.0) * l_bone
        + _weight(weights, "lambda_assign", 0.05) * l_assign
        + _weight(weights, "lambda_sigma", 0.0) * l_sigma
        + _weight(weights, "lambda_contact", 0.0) * l_contact
        + _weight(weights, "lambda_nll", 0.0) * l_nll
        + _weight(weights, "lambda_prior_pose", 0.0) * l_prior_pose
        + _weight(weights, "lambda_prior_nll", 0.0) * l_prior_nll
        + _weight(weights, "lambda_prior_traj", 0.0) * l_prior_traj
        + _weight(weights, "lambda_foot_res", 0.0) * l_foot_res
    )
    return {
        "L_pose": l_pose,
        "L_traj": l_traj,
        "L_traj_vel": l_traj_vel,
        "L_traj_delta": l_traj_delta,
        "L_Trec": l_trec,
        "L_Vrec": l_vrec,
        "L_kp": l_kp,
        "L_pos": l_pos,
        "L_con": l_con,
        "L_pose_vel": l_pose_vel,
        "L_bone": l_bone,
        "L_assign": l_assign,
        "L_sigma": l_sigma,
        "L_contact": l_contact,
        "L_nll": l_nll,
        "L_prior_pose": l_prior_pose,
        "L_prior_nll": l_prior_nll,
        "L_prior_traj": l_prior_traj,
        "L_foot_res": l_foot_res,
        "L_phase": l_pose.new_zeros(()),   # V5_1 only — uniform log panel
        "L_cop": l_pose.new_zeros(()),
        "loss": total,
    }


def compute_losses_v3f(out, x0_norm, batch, weights) -> dict:
    """V3_3F training losses (2026-10-17).

    L_pose = the denoiser's normalized full state MSE; L_kp = world-frame FK
    keypoint MSE on the native-space x0 (lambda_kp > 0, the pose-space
    bottleneck experiment 2026-10-17: the world/position-space supervision
    the v1 batch lacked — probe 支柱 1 says pose space cannot carry position
    increments, and every structure that delivered ground-quantity gains
    (V3_3B L_kp/traj head, V3_3D position head) supervised in position
    space); L_contact = the pos-weighted f6_soft BCE (swing-phase air recall
    matters for the guidance/refinement layers).  All other keys exist as
    zeros so the train.py logging loop sees one uniform loss dict.
    """
    l_pose = F.mse_loss(out["x0_hat"], x0_norm)
    gt = batch["contact_gt"]
    pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
    pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
    l_contact = F.binary_cross_entropy_with_logits(
        out["contact_logits"], gt, pos_weight=pos_weight)
    zero = l_pose.new_zeros(())
    l_kp = zero
    if _weight(weights, "lambda_kp", 0.0) > 0.0:
        anchor = batch.get("trans_anchor")
        if anchor is None:
            anchor = batch["trans_gt"].new_zeros((batch["trans_gt"].shape[0], 3))
        pred_kp = fk_pose6d(out["x0_pose"], out["x0_trans"] + anchor[:, None, :],
                            batch["offsets"], batch["parents"])
        l_kp = F.mse_loss(pred_kp, batch["kp_gt"])
    total = (
        _weight(weights, "lambda_pose", 3.0) * l_pose
        + _weight(weights, "lambda_kp", 1.0) * l_kp
        + _weight(weights, "lambda_contact", 1.0) * l_contact
    )
    return {
        "L_pose": l_pose,
        "L_traj": zero,
        "L_traj_vel": zero,
        "L_traj_delta": zero,
        "L_Trec": zero,
        "L_Vrec": zero,
        "L_kp": l_kp,
        "L_pos": zero,
        "L_con": zero,
        "L_pose_vel": zero,
        "L_bone": zero,
        "L_assign": zero,
        "L_sigma": zero,
        "L_contact": l_contact,
        "L_nll": zero,
        "L_prior_pose": zero,
        "L_prior_nll": zero,
        "L_prior_traj": zero,
        "L_foot_res": zero,
        "L_phase": zero,                   # V5_1 only — uniform log panel
        "L_cop": zero,
        "loss": total,
    }


def _masked_mean(loss_per_sample: torch.Tensor, mask: torch.Tensor):
    """masked mean over a per-sample loss vector; 0 when the mask is empty."""
    mask = mask.to(loss_per_sample.device).bool()
    if bool(mask.any()):
        return loss_per_sample[mask].mean()
    return loss_per_sample.new_zeros(())


def _v5_1_traj_losses(v_hat, trans_hat, vel_gt, mask, weights):
    """V5_1A L_traj（VT 行，live 样本掩码）：velocity + multi-scale
    displacement 的逐样本版本（_trajectory_losses 的口径，逐样本聚合后
    按掩码取均值——零化 V 样本的 VT 行无意义，不进损失）。"""
    x_gt = torch.cumsum(vel_gt, dim=1) / float(FPS)
    per = ((v_hat - vel_gt).square().mean(dim=-1).mean(dim=-1))
    deltas = weights.get("traj_deltas", DEFAULT_TRAJ_DELTAS)
    if isinstance(deltas, str):
        deltas = tuple(int(value.strip()) for value in deltas.split(",") if value.strip())
    else:
        deltas = tuple(int(value) for value in deltas)
    valid = tuple(delta for delta in deltas if 1 <= delta < trans_hat.shape[1])
    if not valid:
        raise ValueError("traj_deltas must contain an integer in [1, Tw-1]")
    power = float(weights.get("traj_delta_weight_power", 1.0))
    l_vel = _masked_mean(per, mask)
    l_delta = per.new_zeros(())
    for delta in valid:
        pred_delta = trans_hat[:, delta:] - trans_hat[:, :-delta]
        gt_delta = x_gt[:, delta:] - x_gt[:, :-delta]
        per_delta = (pred_delta - gt_delta).square().mean(dim=(-1, -2))
        l_delta = l_delta + (1.0 / (float(delta) ** power)) * _masked_mean(per_delta, mask)
    l_delta = l_delta / len(valid)
    return (
        _weight(weights, "traj_velocity_w", 1.0) * l_vel
        + _weight(weights, "traj_delta_w", 1.0) * l_delta,
        l_vel,
        l_delta,
    )


def _v5_1_contact_bce(logits, gt, live, air_w, pos_weight):
    """G 配方 air 加权 pos-weighted BCE（逐元素权重 = (1 + air_w·(1-gt))
    × live 样本掩码；和值归一 → 掩码外样本零贡献）。"""
    elem = F.binary_cross_entropy_with_logits(
        logits, gt, pos_weight=pos_weight, reduction="none")
    w = (1.0 + air_w * (1.0 - gt.float())) * live.to(logits.device).float().view(-1, 1, 1)
    return (elem * w).sum() / w.sum().clamp(min=1.0)


def _v5_1_gauss_mu_mse(mu, gt, gt_mask, live):
    """专家 μ 监督（V5_1A §3：相位/CoP 只监督各专家 μ，掩码跳过）。
    mu: (B, tw, 4) [L c/s, R c/s]；gt: (B, tw, 2, 2)（无效帧 NaN）；
    gt_mask: (B, tw, 2)；live: (B,) 样本掩码。掩码全空（静止会话）→ 0。"""
    mu = mu.reshape(*mu.shape[:2], 2, 2)
    gt_safe = torch.nan_to_num(gt, nan=0.0)
    err = (mu - gt_safe).square().sum(dim=-1)                     # (B, tw, 2)
    valid = gt_mask.to(err.device).float() * live.to(err.device).float().view(-1, 1, 1)
    return (err * valid).sum() / valid.sum().clamp(min=1.0)


def compute_losses_v5_1(out, batch, config_id, weights) -> dict:
    """V5_1A training losses (2026-10-19, V5_1A方案.md §3).

    单点预注册权重（λ 不设网格，z_gen/model_registry.py train_args 携带）：
      L_pose 3.0（6D MSE，V2M 行 + VT2M 行，live 样本）
      L_traj 1.0（VT 行，live 样本）
      L_contact 1.0（f6_soft pos-weighted BCE + air 加权 λ_air=2.0；z 接触
        分量三读数均监督——g_V、g_T、PoE 合成 + G 配方接触头）
      L_phase 1.0 / L_cop 1.0（只监督各专家 μ，掩码跳过；CoP 只在接触帧
        ——掩码即接触帧且和值>0；PoE 合成由闭式决定不额外监督）
      L_prior_pose 1.0（T2M 行健全性，不参与承诺）

    batch["v_zero"] (B,) = V 零化样本（VT 90% / V 零化 10%，H 口径）：
    零化样本只监督 T2M 行 + g_T 读出（g_V 读零化 V 无意义，PoE 混入
    垃圾 g_V 同样掩掉）。其余键与 compute_losses 同构（统一日志面板）。
    """
    pose_gt = batch["pose_gt"]
    live = ~batch.get("v_zero", torch.zeros(pose_gt.shape[0], dtype=torch.bool,
                                            device=pose_gt.device)).bool()
    live = live.to(pose_gt.device)

    def per_pose_mse(pred):
        return (pred - pose_gt).square().mean(dim=(-1, -2))

    # L_pose：V2M 行 + VT2M 行同批监督（live 样本）。
    l_pose = _masked_mean(per_pose_mse(out["x0_hat"]) + per_pose_mse(out["x0_v"]),
                          live)
    # L_traj：VT 行（live 样本）。
    l_traj, l_traj_vel, l_traj_delta = _v5_1_traj_losses(
        out["v_hat"], out["trans_hat"], batch["vel_gt"], live, weights)

    # L_contact：z 接触分量三读数 + 接触头（air 加权，G 配方）。
    gt = batch["contact_gt"]
    pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
    pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
    air_w = _weight(weights, "lambda_air", 0.0)
    zs = out["z"]
    l_contact = _v5_1_contact_bce(out["contact_logits"], gt, torch.ones_like(live),
                                  air_w, pos_weight) \
        + _v5_1_contact_bce(zs["logit_gT"], gt, torch.ones_like(live),
                            air_w, pos_weight) \
        + _v5_1_contact_bce(zs["logit_gV"], gt, live, air_w, pos_weight) \
        + _v5_1_contact_bce(zs["logit_poe"], gt, live, air_w, pos_weight)

    # L_phase / L_cop：各专家 μ 监督（g_V 仅 live 样本；g_T 恒监督——
    # T 流输入三条件恒定）。掩码跳过（静止会话掩码全空 → 0）。
    l_phase = _v5_1_gauss_mu_mse(zs["mu_gV"][..., 0:4], batch["phase_gt"],
                                 batch["phase_mask"], live) \
        + _v5_1_gauss_mu_mse(zs["mu_gT"][..., 0:4], batch["phase_gt"],
                             batch["phase_mask"], torch.ones_like(live))
    l_cop = _v5_1_gauss_mu_mse(zs["mu_gV"][..., 4:8], batch["cop_gt"],
                               batch["cop_mask"], live) \
        + _v5_1_gauss_mu_mse(zs["mu_gT"][..., 4:8], batch["cop_gt"],
                             batch["cop_mask"], torch.ones_like(live))

    # T2M 行 L_prior_pose 健全性（全样本；不参与承诺——约 90mm 量级参照，
    # G 前科 315.5 防回归）。
    l_prior_pose = F.mse_loss(out["x0_prior"], pose_gt)

    total = (
        _weight(weights, "lambda_pose", 3.0) * l_pose
        + _weight(weights, "lambda_traj", 1.0) * l_traj
        + _weight(weights, "lambda_contact", 0.0) * l_contact
        + _weight(weights, "lambda_phase", 0.0) * l_phase
        + _weight(weights, "lambda_cop", 0.0) * l_cop
        + _weight(weights, "lambda_prior_pose", 0.0) * l_prior_pose
    )
    zero = l_pose.new_zeros(())
    return {
        "L_pose": l_pose,
        "L_traj": l_traj,
        "L_traj_vel": l_traj_vel,
        "L_traj_delta": l_traj_delta,
        "L_Trec": zero,
        "L_Vrec": zero,
        "L_kp": zero,
        "L_pos": zero,
        "L_con": zero,
        "L_pose_vel": zero,
        "L_bone": zero,
        "L_assign": zero,
        "L_sigma": zero,
        "L_contact": l_contact,
        "L_nll": zero,
        "L_prior_pose": l_prior_pose,
        "L_prior_nll": zero,
        "L_prior_traj": zero,
        "L_foot_res": zero,
        "L_phase": l_phase,
        "L_cop": l_cop,
        "loss": total,
    }


def per_sample_contact_bce(logits, gt, pos_weight, air_w):
    """Per-sample contact BCE (pos-weighted, air-weighted, G 配方口径) —
    V6_1A A1 监控键的 per-sample 值（train.py 侧重算，reduction="none"
    同公式；不进入总损失）。"""
    elem = F.binary_cross_entropy_with_logits(
        logits, gt, pos_weight=pos_weight, reduction="none")
    w = 1.0 + air_w * (1.0 - gt.float())
    return (elem * w).mean(dim=(-1, -2))  # (B,)


def compute_losses_v4_1(out, batch, config_id, weights) -> dict:
    """V4_1A training losses (2026-10-19, V4_1A方案.md §3).

    单点预注册权重（λ 不设网格，z_gen/model_registry.py train_args 携带）：
      L_pose 3.0（V 流 x0_hat，无梯度常数——载体冻结后该项恒 0，G 同款
        口径；V 塔不被任何可训梯度触碰）
      L_contact 1.0（f6_soft pos-weighted BCE + air 加权 λ_air=2.0，G 配方）
      L_prior_pose 3.0 + L_prior_traj 1.0（T 先验流，T2M 行健全性）
      L_corr 3.0（lambda_pos 复用键）：out["pos_corr"] 六关节骨盆相对
        位置校正 MSE，目标 = GT 六关节骨盆相对位置 − FK(pose_v) 六关节
        骨盆相对位置（残差监督 x_gt−x_V 在位置空间，V4_1A方案 §3）
      L_corr_prior（out["pos_corr_prior"] 同式，λ 同为 3.0 单点注册——
        T2M 行 = 先验 + 修正分支，同式监督）
    sigma NLL 恒 0（继承冻结）。其余键与 compute_losses 同构（统一日志
    面板）。"""
    pose_gt = batch["pose_gt"]
    l_pose = F.mse_loss(out["x0_hat"], pose_gt)

    # L_contact: G 配方 pos-weighted BCE + air 加权。
    gt = batch["contact_gt"]
    pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
    pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
    air_w = _weight(weights, "lambda_air", 0.0)
    if air_w > 0.0:
        elem_w = 1.0 + air_w * (1.0 - gt.float())
        l_contact = F.binary_cross_entropy_with_logits(
            out["contact_logits"], gt, pos_weight=pos_weight, weight=elem_w)
    else:
        l_contact = F.binary_cross_entropy_with_logits(
            out["contact_logits"], gt, pos_weight=pos_weight)

    # T 先验流（T2M 行）。
    l_prior_pose = F.mse_loss(out["x0_prior"], pose_gt)
    l_prior_traj, _, _ = _trajectory_losses(
        {"v_hat": out["v_prior"], "trans_hat": out["trans_prior"]},
        batch, weights)

    # L_corr：位置空间残差监督。骨盆相对（anchor 在相对差中抵消）；
    # pose_v 来自冻结载体，梯度只经 pos_corr 进修正分支与 T 流。
    anchor = batch.get("trans_anchor")
    if anchor is None:
        anchor = pose_gt.new_zeros((pose_gt.shape[0], 3))
    six = KNEE_ANKLE_FOOT_JOINTS
    fk_v = fk_pose6d(out["x0_hat"], out["trans_hat"] + anchor[:, None, :],
                     batch["offsets"], batch["parents"])
    gt_rel = batch["kp_gt"][..., six, :] - batch["kp_gt"][..., 0:1, :]
    v_rel = fk_v[..., six, :] - fk_v[..., 0:1, :]
    l_corr = F.mse_loss(
        out["pos_corr"].reshape(*out["pos_corr"].shape[:2], len(six), 3),
        gt_rel - v_rel)
    # T2M 行同式（修正分支读先验塔部位 token，V4_1A方案 §2.3）。
    fk_prior = fk_pose6d(out["x0_prior"], out["trans_prior"] + anchor[:, None, :],
                         batch["offsets"], batch["parents"])
    prior_rel = fk_prior[..., six, :] - fk_prior[..., 0:1, :]
    l_corr_prior = F.mse_loss(
        out["pos_corr_prior"].reshape(*out["pos_corr_prior"].shape[:2], len(six), 3),
        gt_rel - prior_rel)

    total = (
        _weight(weights, "lambda_pose", 3.0) * l_pose
        + _weight(weights, "lambda_contact", 0.0) * l_contact
        + _weight(weights, "lambda_prior_pose", 0.0) * l_prior_pose
        + _weight(weights, "lambda_prior_traj", 0.0) * l_prior_traj
        + _weight(weights, "lambda_pos", 0.0) * (l_corr + l_corr_prior)
    )
    zero = l_pose.new_zeros(())
    return {
        "L_pose": l_pose,
        "L_traj": zero,                 # V 流 traj 不单列（方案 §3 五条款）
        "L_traj_vel": zero,
        "L_traj_delta": zero,
        "L_Trec": zero,
        "L_Vrec": zero,
        "L_kp": zero,
        "L_pos": l_corr,                # lambda_pos 复用键（VT 行修正）
        "L_corr_prior": l_corr_prior,   # T2M 行修正（同 λ）
        "L_con": zero,
        "L_pose_vel": zero,
        "L_bone": zero,
        "L_assign": zero,
        "L_sigma": zero,
        "L_contact": l_contact,
        "L_nll": zero,
        "L_prior_pose": l_prior_pose,
        "L_prior_nll": zero,
        "L_prior_traj": l_prior_traj,
        "L_foot_res": zero,
        "L_phase": zero,                # V5_1 only — uniform log panel
        "L_cop": zero,
        "loss": total,
    }
