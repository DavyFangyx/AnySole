"""Training losses for AnySole V1."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from anysole.utils.geometry import f2_to_world, fk_pose6d, rot6d_to_rotmat
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
    # the assignment A are stopgrad'd. Raw sigma logits stay live, including
    # their upstream visual/decoder path; this is not a sigma-only update.
    # beta = 0.5, sigma clipped to [-3, 6].  Skipped while sigma is frozen
    # (sigma logits fixed at 0, variance 1, during sigma_freeze_frac) and
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

    total = (
        _weight(weights, "lambda_pose", 1.0) * l_pose
        + _weight(weights, "lambda_traj", 1.0) * l_traj
        + _weight(weights, "lambda_trec", 0.1) * l_trec
        + _weight(weights, "lambda_vrec", 0.1) * l_vrec
        + _weight(weights, "lambda_kp", 1.0) * l_kp
        + _weight(weights, "lambda_con", 0.1) * l_con
        + _weight(weights, "lambda_pose_vel", 0.0) * l_pose_vel
        + _weight(weights, "lambda_bone", 0.0) * l_bone
        + _weight(weights, "lambda_assign", 0.05) * l_assign
        + _weight(weights, "lambda_sigma", 0.0) * l_sigma
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
        "L_assign": l_assign,
        "L_sigma": l_sigma,
        "loss": total,
    }


# Strict RGB student reductions. Legacy compute_losses above remains the
# compatibility entry; grouped training consumes these sums and counts so
# complete-group microbatches can share one denominator per optimizer update.
def _visual_deltas(config, tw):
    values = config.get("traj_deltas", DEFAULT_TRAJ_DELTAS)
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]
    values = tuple(int(v) for v in values)
    if len(set(values)) != len(values):
        raise ValueError("traj_deltas must not contain duplicates")
    return tuple(v for v in values if 1 <= v < tw)


def loss_term_weights(config, tw, lambda_mv=0.0):
    """Exact coefficients for independently reduced grouped loss terms."""
    deltas = _visual_deltas(config, tw)
    weights = {
        "pose": _weight(config, "lambda_pose", 3.0),
        "traj_vel": _weight(config, "lambda_traj", 1.0) * _weight(config, "traj_velocity_w", 1.0),
        "pressure": _weight(config, "lambda_trec", .1),
        "kp": _weight(config, "lambda_kp", 1.0),
        "sigma": _weight(config, "lambda_sigma", .01),
        "assignment": _weight(config, "lambda_assign", .05),
        "local23_pair": float(lambda_mv) * _weight(config, "mv_pose_weight", 1.0),
        "pressure_pair": float(lambda_mv) * _weight(config, "mv_pressure_weight", 1.0),
    }
    power = float(config.get("traj_delta_weight_power", 1.0))
    for delta in deltas:
        weights[f"traj_delta_{delta}"] = (
            _weight(config, "lambda_traj", 1.0) * _weight(config, "traj_delta_w", 1.0)
            / (len(deltas) * float(delta) ** power)
        )
    if config.get("a2_version") is not None:
        weights.update(pressure_contact=float(config.get("lambda_contact", 0.)),
                       pressure_total=float(config.get("lambda_pressure_aux", 0.)),
                       pressure_diff=float(config.get("lambda_pressure_aux", 0.)))
    return weights


def _loss_float(value):
    return value if value.dtype == torch.float64 else value.float()


def _connected_zero(out):
    return sum((_loss_float(out[key]).reshape(-1)[:0].sum()
                for key in ("x0_hat", "pressure_hat", "v_hat") if out.get(key) is not None))


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"valid {name} contains NaN or Inf")
    return value


def _checked_labels(batch):
    labels, masks = batch["labels"], batch["masks"]
    motion = masks["motion_valid"]
    for key in ("pose6d", "root", "kp_gt"):
        _finite(labels[key][motion], f"{key} labels")
    _finite(labels["velocity"][masks["velocity_valid"]], "velocity labels")
    _finite(labels["pressure"][masks["pressure_valid"]], "pressure labels")
    _finite(labels["offsets"][motion.any(dim=1)], "FK offsets")
    expected = torch.tensor(JOINT_PARENTS, device=labels["parents"].device)
    if not bool((labels["parents"] == expected).all()):
        raise ValueError("FK parents must use the canonical SMPL-24 tree")


def _view_masks(batch, view_index):
    visual, masks = batch["visual"], batch["masks"]
    vv = visual["view_valid"]
    if vv.ndim != 3 or vv.dtype != torch.bool:
        raise ValueError("view_valid must be bool [B,C,T]")
    if view_index.ndim != 2 or view_index.shape[1] != 2 or view_index.dtype != torch.long:
        raise ValueError("view_index must be long [N,2] containing group and camera slot")
    index = view_index.to(device=vv.device)
    if index.numel():
        if (bool((index < 0).any()) or bool((index[:, 0] >= vv.shape[0]).any())
                or bool((index[:, 1] >= vv.shape[1]).any())
                or len(torch.unique(index, dim=0)) != len(index)):
            raise ValueError("view_index has duplicate or out-of-range rows")
    group, camera = index.unbind(dim=1)
    frame = vv[group, camera]
    if len(index) and not bool(frame.any(dim=1).all()):
        raise ValueError("view_index must exclude all-invalid views")
    times = visual["target_time_s"][group].to(torch.float64)
    if (not bool(torch.isfinite(times).all())
            or not bool((times[:, 1:] > times[:, :-1]).all())):
        raise ValueError("target timestamps must be finite and strictly increasing")
    motion = frame & masks["motion_valid"][group]
    pressure = frame.unsqueeze(-1) & masks["pressure_valid"][group]
    dt = times[:, 1:] - times[:, :-1]
    edges = (visual["view_edges"][group, camera] & masks["motion_edges"][group]
             & motion[:, :-1] & motion[:, 1:] & masks["velocity_valid"][group, 1:]
             & (dt > 0) & (dt <= .05 + 1e-9))
    velocity = torch.cat([torch.zeros_like(frame[:, :1]), edges], dim=1)
    return group, camera, frame, motion, pressure, velocity, edges, dt


def compute_visual_loss_counts(batch, view_index, config, *, include_assignment=True,
                               sigma_frozen=False):
    """Masks-only per-term denominators, before any model graph is built.

    Call with include_assignment=True only once per optimizer accumulation;
    sigma_frozen must reflect that update's schedule. Count units match the
    raw sums returned by compute_visual_loss_terms exactly.
    """
    _, _, frame, motion, pressure, velocity, edges, _ = _view_masks(batch, view_index)
    motion_count = int(motion.sum().item())
    counts = {"pose": motion_count * 144, "traj_vel": int(velocity.sum().item()) * 3,
              "pressure": int(pressure.sum().item()), "kp": motion_count * N_JOINTS * 3}
    sigma_active = (not sigma_frozen and _weight(config, "lambda_sigma", .01) > 0)
    counts["sigma"] = motion_count * 9 if sigma_active else 0
    supervised = counts["pose"] > 0 or counts["pressure"] > 0
    counts["assignment"] = int(include_assignment and supervised and
                               _weight(config, "lambda_assign", .05) > 0)
    for delta in _visual_deltas(config, frame.shape[1]):
        counts[f"traj_delta_{delta}"] = int(edges.unfold(1, delta, 1).all(dim=-1).sum().item()) * 3
    if config.get("a2_version") is not None:
        from anysole.utils.a2_pressure_losses import a2_pressure_counts
        counts.update(a2_pressure_counts(batch, view_index, config))
    return counts


def _mse_sum(pred, target, valid, name):
    pred = _loss_float(_finite(pred[valid], f"{name} prediction"))
    target = _loss_float(_finite(target[valid], f"{name} labels"))
    return (pred - target).square().sum()


def _term(raw_sum, count, unit):
    return {"raw_sum": raw_sum, "valid_count": int(count), "unit": unit}


def compute_visual_loss_terms(out, batch, view_index, config, *, include_assignment=True):
    """Raw supervised losses for flat independent RGB predictions.

    Pose is already denormalized. FK uses the GT root/beta-derived offsets
    only as train-only pose supervision, never the predicted trajectory.
    Displacements integrate v_i over actual dt on complete valid source
    edges; the model's window trans_hat is deliberately not consumed.
    """
    _checked_labels(batch)
    group, _, frame, motion, pressure, velocity, edges, dt = _view_masks(batch, view_index)
    if out["x0_hat"].shape != (*frame.shape, 144) or out["pressure_hat"].shape != (*frame.shape, 96):
        raise ValueError("flat predictions and view_index shapes do not match")
    if out["v_hat"].shape != (*frame.shape, 3):
        raise ValueError("v_hat must be [N,T,3]")
    counts = compute_visual_loss_counts(batch, view_index, config, include_assignment=include_assignment,
                                        sigma_frozen=bool(out.get("sigma_frozen", False)))
    zero = _connected_zero(out)
    terms = {name: _term(zero, count, "dimensionless") for name, count in counts.items()}
    labels = batch["labels"]
    pose = labels["pose6d"][group]
    terms["pose"] = _term(_mse_sum(out["x0_hat"], pose, motion.unsqueeze(-1).expand_as(pose), "pose"),
                           counts["pose"], "raw_6d_squared")
    terms["pressure"] = _term(_mse_sum(out["pressure_hat"], labels["pressure"][group], pressure, "pressure"),
                               counts["pressure"], "normalized_reading_squared")
    target_v = labels["velocity"][group]
    terms["traj_vel"] = _term(_mse_sum(out["v_hat"], target_v,
                                      velocity.unsqueeze(-1).expand_as(target_v), "velocity"),
                               counts["traj_vel"], "m2_s2")
    # Fill before multiplication/cumsum: a masked NaN must never enter the
    # graph, even when only a long-span displacement term is enabled.
    safe_pred = torch.where(velocity.unsqueeze(-1), out["v_hat"], torch.zeros_like(out["v_hat"]))
    safe_gt = torch.where(velocity.unsqueeze(-1), target_v, torch.zeros_like(target_v))
    safe_pred, safe_gt = _loss_float(safe_pred), _loss_float(safe_gt)
    pred_x = torch.cat([safe_pred[:, :1] * 0,
                        torch.cumsum(safe_pred[:, 1:] * dt.to(safe_pred.dtype).unsqueeze(-1), dim=1)], dim=1)
    gt_x = torch.cat([safe_gt[:, :1] * 0,
                      torch.cumsum(safe_gt[:, 1:] * dt.to(safe_gt.dtype).unsqueeze(-1), dim=1)], dim=1)
    for delta in _visual_deltas(config, frame.shape[1]):
        key = f"traj_delta_{delta}"
        valid = edges.unfold(1, delta, 1).all(dim=-1).unsqueeze(-1)
        pred_delta, gt_delta = pred_x[:, delta:] - pred_x[:, :-delta], gt_x[:, delta:] - gt_x[:, :-delta]
        terms[key] = _term(_mse_sum(pred_delta, gt_delta, valid.expand_as(pred_delta), key), counts[key], "m2")
    if counts["kp"]:
        rows = group[:, None].expand_as(frame)[motion]
        with torch.autocast(device_type=out["x0_hat"].device.type, enabled=False):
            pred_kp = fk_pose6d(_loss_float(out["x0_hat"][motion])[:, None],
                                _loss_float(labels["root"][group][motion])[:, None],
                                _loss_float(labels["offsets"][rows]), labels["parents"][rows])[:, 0]
        target_kp = _loss_float(labels["kp_gt"][group][motion])
        terms["kp"] = _term((_finite(pred_kp, "FK prediction") - target_kp).square().sum(), counts["kp"], "m2")
    else:
        terms["kp"]["unit"] = "m2"
    if counts["assignment"]:
        logits = out["part_logits"]
        log_assign = F.log_softmax(logits, dim=0)
        assignment = log_assign.exp()
        entropy = -(assignment * log_assign).sum(dim=0).mean()
        mass = assignment.mean(dim=1)
        balance = (mass * (mass + 1e-8).log()).sum() + math.log(mass.numel())
        terms["assignment"] = _term(entropy + balance, 1, "dimensionless")
    if counts["sigma"]:
        sigma = _finite(out["gate_sigma"][motion], "sigma logits").clamp(-3, 6)
        gate = _finite(out["gate_g"][motion][..., :2].detach(), "sigma gate target")
        hypotheses = _finite(out["part_hypotheses"].permute(1, 2, 0, 3, 4)[motion].detach(),
                             "sigma hypotheses target")
        assignment = _finite(out["part_assignment"].detach(), "sigma assignment target")
        mean, std = out["pose_mean"].detach(), out["pose_std"].detach()
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("pose stats must be finite with positive std")
        target = ((_loss_float(pose[motion]) - mean) / std).reshape(-1, N_JOINTS, 6)
        err = (_loss_float(hypotheses) - target[:, None]).square().mean(dim=-1)
        r2 = (assignment[None] * err).sum(dim=-1) / assignment.sum(dim=-1).clamp(min=1e-6)[None]
        variance = (_loss_float(gate) * (-_loss_float(sigma)).exp()).sum(dim=-1).clamp(min=1e-4)
        nll = float(config.get("sigma_beta", .5)) * (r2 / variance + variance.log())
        if nll.numel() != counts["sigma"]:
            raise ValueError("visual sigma requires nine parts")
        terms["sigma"] = _term(nll.sum(), counts["sigma"], "dimensionless")
    if config.get("a2_version") is not None:
        from anysole.utils.a2_pressure_losses import modify_pressure_terms
        modify_pressure_terms(terms, out, batch, view_index, config)
    return terms


def _pair_layout(batch, view_index, pair_group_index):
    group, camera, frame, _, _, _, _, _ = _view_masks(batch, view_index)
    bsz = batch["visual"]["view_valid"].shape[0]
    camera_ids = batch["meta"]["camera_id"].detach().cpu()
    mapping = list(range(bsz))
    if pair_group_index is not None:
        permutation = torch.as_tensor(pair_group_index)
        if permutation.ndim != 1 or permutation.dtype not in (torch.int32, torch.int64):
            raise ValueError("pair_group_index must be an integer [B] derangement")
        mapping = permutation.detach().cpu().tolist()
        if (bsz < 2 or len(mapping) != bsz or sorted(mapping) != list(range(bsz))
                or any(i == other for i, other in enumerate(mapping))):
            raise ValueError("pair_group_index must be a cross-group derangement")
        keys = batch["meta"].get("sample_key")
        if keys is None or any(keys[i] == keys[other] for i, other in enumerate(mapping)):
            raise ValueError("shuffle requires different sample_key at both endpoints")
    by_group = [{} for _ in range(bsz)]
    for row, (g, c) in enumerate(view_index.detach().cpu().tolist()):
        cam = int(camera_ids[g, c])
        if cam < 1 or cam in by_group[g]:
            raise ValueError("pair routing requires unique real camera_id per group")
        by_group[g][cam] = row
    pairs = []
    for g, views in enumerate(by_group):
        cameras = sorted(views)
        for c, left_camera in enumerate(cameras):
            for right_camera in cameras[c+1:]:
                peer = by_group[mapping[g]].get(right_camera)
                if peer is not None:
                    pairs.append((views[left_camera], peer))
    if not pairs:
        empty = group.new_empty((0,))
        return empty, empty, frame.new_empty((0, frame.shape[1])), frame.new_empty((0, frame.shape[1], 96))
    pair_index = torch.tensor(pairs, dtype=torch.long, device=frame.device)
    left, right = pair_index.unbind(dim=1)
    left_group, right_group = group[left], group[right]
    valid = frame[left] & frame[right]
    masks = batch["masks"]
    motion = valid & masks["motion_valid"][left_group] & masks["motion_valid"][right_group]
    pressure = valid.unsqueeze(-1) & masks["pressure_valid"][left_group] & masks["pressure_valid"][right_group]
    return left, right, motion, pressure


def compute_pair_loss_counts(batch, view_index, config, pair_group_index=None):
    """Masks-only pair denominators using the same camera routing as losses."""
    _, _, motion, pressure = _pair_layout(batch, view_index, pair_group_index)
    return {"local23_pair": int(motion.sum().item()) * 23,
            "pressure_pair": int(pressure.sum().item())}


def compute_pair_loss_terms(out, batch, view_index, config, pair_group_index=None):
    """Unordered local23 squared-chordal and normalized pressure pair sums.

    pair_group_index optionally reroutes only each right endpoint to the
    same actual camera in another group. The permutation is supplied by the
    caller's independent RNG; these functions never draw random numbers.
    Ground-truth supervision is untouched, and both predictions stay live.
    """
    _checked_labels(batch)
    left, right, motion, pressure = _pair_layout(batch, view_index, pair_group_index)
    counts = {"local23_pair": int(motion.sum().item()) * 23,
              "pressure_pair": int(pressure.sum().item())}
    zero = _connected_zero(out)
    pose_sum = pressure_sum = zero
    if counts["local23_pair"]:
        a = _finite(out["x0_hat"][left][motion], "pair left pose").reshape(-1, 24, 6)[:, 1:]
        b = _finite(out["x0_hat"][right][motion], "pair right pose").reshape(-1, 24, 6)[:, 1:]
        rotations_a, rotations_b = rot6d_to_rotmat(_loss_float(a)), rot6d_to_rotmat(_loss_float(b))
        pose_sum = (rotations_a - rotations_b).square().sum()
    if counts["pressure_pair"]:
        a = _loss_float(_finite(out["pressure_hat"][left][pressure], "pair left pressure"))
        b = _loss_float(_finite(out["pressure_hat"][right][pressure], "pair right pressure"))
        pressure_sum = (a - b).square().sum()
    return {"local23_pair": _term(pose_sum, counts["local23_pair"], "squared_frobenius_per_rotation"),
            "pressure_pair": _term(pressure_sum, counts["pressure_pair"], "normalized_reading_squared")}
