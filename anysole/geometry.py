"""YXZ euler, 6D rotation, and BVH forward kinematics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from anysole.types import FPS, N_JOINTS

# F2a: the BVH Hips local forward axis (verified 2026-09-18 by
# z_note/smoke_f2_roundtrip.py: heading-frame lateral/forward velocity ratio
# and speed correlation favor +Z on every informative session).
FORWARD_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _as_rotmat_np(eulers_deg: np.ndarray) -> np.ndarray:
    flat = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    mats = SciRotation.from_euler("YXZ", flat, degrees=True).as_matrix()
    return mats.reshape(*eulers_deg.shape[:-1], 3, 3)


def euler_yxz_to_rotmat(eulers_deg: np.ndarray) -> np.ndarray:
    return _as_rotmat_np(eulers_deg).astype(np.float64)


def rotmat_to_euler_yxz(rotmats: np.ndarray) -> np.ndarray:
    flat = np.asarray(rotmats, dtype=np.float64).reshape(-1, 3, 3)
    eulers = SciRotation.from_matrix(flat).as_euler("YXZ", degrees=True)
    return eulers.reshape(*rotmats.shape[:-2], 3).astype(np.float64)


def rotmat_to_6d_np(rotmats: np.ndarray) -> np.ndarray:
    cols = np.concatenate([rotmats[..., :, 0], rotmats[..., :, 1]], axis=-1)
    return cols.astype(np.float32)


def rot6d_to_rotmat_np(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def euler_yxz_to_6d(eulers_deg: np.ndarray) -> np.ndarray:
    return rotmat_to_6d_np(euler_yxz_to_rotmat(eulers_deg))


def rot6d_to_euler_yxz(d6: np.ndarray) -> np.ndarray:
    return rotmat_to_euler_yxz(rot6d_to_rotmat_np(d6))


def rotmat_to_6d(rotmats: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotmats[..., :, 0], rotmats[..., :, 1]], dim=-1)


def rot6d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def slerp_eulers(src_t: np.ndarray, eulers_deg: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """SLERP YXZ euler sequences. `eulers_deg` is (T, J, 3)."""
    n_src, n_joints, _ = eulers_deg.shape
    out = np.empty((query_t.size, n_joints, 3), dtype=np.float64)
    clipped = np.clip(query_t, src_t[0], src_t[-1])
    if n_src == 1:
        out[:] = eulers_deg[0]
        return out
    for joint_i in range(n_joints):
        key_rots = SciRotation.from_euler("YXZ", eulers_deg[:, joint_i], degrees=True)
        slerp = Slerp(src_t, key_rots)
        out[:, joint_i] = slerp(clipped).as_euler("YXZ", degrees=True)
    return out


def resample_bvh_motion(motion: np.ndarray, frame_time: float, query_t: np.ndarray) -> np.ndarray:
    """Resample 72-channel BVH motion: linear XYZ, SLERP all 23 joints."""
    src_t = np.arange(motion.shape[0], dtype=np.float64) * frame_time
    out = np.empty((query_t.size, motion.shape[1]), dtype=np.float64)
    for col in range(3):
        out[:, col] = np.interp(query_t, src_t, motion[:, col])
    eulers = motion[:, 3:].reshape(motion.shape[0], N_JOINTS, 3)
    out[:, 3:] = slerp_eulers(src_t, eulers, query_t).reshape(query_t.size, N_JOINTS * 3)
    return out


def fk_local_np(local_rotmats: np.ndarray, offsets: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_joints, _, _ = local_rotmats.shape
    global_rot = np.zeros_like(local_rotmats)
    global_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    for j in range(n_joints):
        parent = int(parents[j])
        if parent < 0:
            global_rot[:, j] = local_rotmats[:, j]
            global_pos[:, j] = offsets[j]
        else:
            global_rot[:, j] = global_rot[:, parent] @ local_rotmats[:, j]
            offset = offsets[j][None, :, None]
            global_pos[:, j] = global_pos[:, parent] + (global_rot[:, parent] @ offset)[..., 0]
    return global_pos, global_rot


def fk_pose6d_np(
    pose_6d: np.ndarray,
    trans: np.ndarray,
    offsets: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    rotmats = rot6d_to_rotmat_np(pose_6d.reshape(pose_6d.shape[0], N_JOINTS, 6))
    pos, _ = fk_local_np(rotmats, offsets, parents)
    return pos + trans[:, None, :]


def fk_pose6d(
    pose_6d: torch.Tensor,
    trans: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """Batched FK. pose_6d (B, T, 138), trans (B, T, 3), offsets (B, 23, 3) or (23, 3)."""
    batch, time, _ = pose_6d.shape
    rotmats = rot6d_to_rotmat(pose_6d.reshape(batch, time, N_JOINTS, 6))
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch, -1, -1)
    n_joints = rotmats.shape[2]
    global_rot = []
    global_pos = []
    parents_list = parents.detach().cpu().tolist() if parents.ndim == 1 else parents[0].detach().cpu().tolist()
    for j in range(n_joints):
        parent = int(parents_list[j])
        if parent < 0:
            joint_rot = rotmats[:, :, j]
            joint_pos = offsets[:, j].unsqueeze(1).expand(-1, time, -1)
        else:
            joint_rot = torch.matmul(global_rot[parent], rotmats[:, :, j])
            offset = offsets[:, j].unsqueeze(1).unsqueeze(-1)
            joint_pos = global_pos[parent] + torch.matmul(global_rot[parent], offset).squeeze(-1)
        global_rot.append(joint_rot)
        global_pos.append(joint_pos)
    return torch.stack(global_pos, dim=2) + trans.unsqueeze(2)


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def _from_to_np(v1: np.ndarray, v2: np.ndarray) -> SciRotation:
    """Shortest-arc rotation mapping normalized v1 -> v2 (Step2Motion from_to)."""
    v1 = _normalize_np(v1)
    v2 = _normalize_np(v2)
    dot = np.clip(np.einsum("...i,...i->...", v1, v2), -1.0, 1.0)
    cross = np.cross(v1, v2)
    s = np.linalg.norm(cross, axis=-1)
    axis = cross / np.clip(s, 1e-8, None)[..., None]
    angle = np.arctan2(s, dot)
    parallel = dot > 1.0 - 1e-6
    antiparallel = dot < -1.0 + 1e-6
    angle = np.where(parallel | antiparallel, 0.0, angle)
    if bool(antiparallel.any()):
        perp = np.cross(v1, np.array([1.0, 0.0, 0.0]))
        fallback = np.where(
            np.linalg.norm(perp, axis=-1, keepdims=True) < 1e-4,
            np.cross(v1, np.array([0.0, 1.0, 0.0])),
            perp,
        )
        axis = np.where(antiparallel[..., None], _normalize_np(fallback), axis)
        angle = np.where(antiparallel, np.pi, angle)
    return SciRotation.from_rotvec(axis * angle[..., None])


def _from_to_axis_np(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> SciRotation:
    """Rotation about `axis` mapping v1's perpendicular component to v2's."""
    axis = _normalize_np(axis)
    u = _normalize_np(v1 - np.einsum("...i,...i->...", v1, axis)[..., None] * axis)
    w = _normalize_np(v2 - np.einsum("...i,...i->...", v2, axis)[..., None] * axis)
    angle = np.arctan2(
        np.einsum("...i,...i->...", axis, np.cross(u, w)),
        np.einsum("...i,...i->...", u, w),
    )
    return SciRotation.from_rotvec(axis * angle[..., None])


def positions_to_6d_np(
    positions: np.ndarray,
    offsets: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Root-local positions (..., 66) -> 6D rotations (..., 138), rest-frame.

    E6.1 inverse FK (Step2Motion skeleton_pos_to_rot in matrix form): each
    joint's rotation comes from its children's directions - the first child
    fixes the direction, the second fixes the roll about it (R = rot @ roll,
    roll about the REST first-child axis with target rot^-1 @ pred; verified
    roundtrip 0.00mm / root rotation 0.00 deg on GT).  Multiply the recovered
    root by R_init (dataset root_rot_init) to get world rotations for export.
    """
    lead = positions.shape[:-1]
    n_frames = int(np.prod(lead)) if lead else 1
    pos = np.concatenate(
        [np.zeros(lead + (1, 3), dtype=np.float64), positions.reshape(*lead, 22, 3)],
        axis=-2,
    ).reshape(n_frames, 23, 3)
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = [int(p) for p in np.asarray(parents).reshape(-1)]
    children: list = [[] for _ in range(23)]
    for i, parent in enumerate(parents):
        if i > 0:
            children[parent].append(i)

    rotmats = np.tile(np.eye(3), (n_frames, 23, 1, 1))
    for j in range(23):
        if not children[j]:
            continue
        gpos, grot = fk_local_np(rotmats, offsets, np.asarray(parents))
        rj = np.transpose(grot[:, j], (0, 2, 1))  # current local frame of j
        c0 = children[j][0]
        rest_dir = np.einsum("tij,tj->ti", rj, gpos[:, c0] - gpos[:, j])
        pred_dir = np.einsum("tij,tj->ti", rj, pos[:, c0] - pos[:, j])
        rot = _from_to_np(rest_dir, pred_dir)
        rotmats[:, j] = rot.as_matrix()
        for gc in children[j][1:]:
            rest_gc = np.einsum("tij,tj->ti", rj, gpos[:, gc] - gpos[:, j])
            pred_gc = np.einsum("tij,tj->ti", rj, pos[:, gc] - pos[:, j])
            rot_inv = np.transpose(rot.as_matrix(), (0, 2, 1))
            target = np.einsum("tij,tj->ti", rot_inv, pred_gc)
            roll = _from_to_axis_np(rest_gc, target, _normalize_np(rest_dir))
            rotmats[:, j] = (rot * roll).as_matrix()
    out = rotmat_to_6d_np(rotmats)
    return out.reshape(*lead, N_JOINTS * 6)


# ---------------------------------------------------------------------------
# F2a: heading/tilt representation (fix_plan_v2.md §F2a).
#
# pose_f2 (..., 138): joints 1-22 keep the parent-local 6D rotations; the root
#   6D is the TILT = R_yaw(psi)^T @ R_root_world — the world root rotation
#   with the ground heading removed.
# traj_f2 (..., 4): [psi_dot (rad/s), v_hx, v_hz (heading-frame horizontal
#   velocity, m/s), h (world root height, m)].
# Anchors: psi_anchor (heading at the frame before the window) and
#   trans_anchor (world root position at that frame).
# ---------------------------------------------------------------------------


def yaw_rotmat_np(psi: np.ndarray) -> np.ndarray:
    """(...,) heading angles -> (..., 3, 3) rotation about the world Y axis.

    World (x, y, z), y up; a yaw of psi rotates the heading frame's +x axis
    to world direction (cos psi, 0, sin psi).
    """
    psi = np.asarray(psi, dtype=np.float64)
    c, s = np.cos(psi), np.sin(psi)
    out = np.zeros(psi.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def yaw_rotmat(psi: torch.Tensor) -> torch.Tensor:
    """Torch twin of yaw_rotmat_np."""
    c, s = torch.cos(psi), torch.sin(psi)
    out = torch.zeros(psi.shape + (3, 3), device=psi.device, dtype=psi.dtype)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def heading_from_root_np(root_rotmats: np.ndarray) -> np.ndarray:
    """(T, 3, 3) world root rotations -> ground heading psi (T,), unwrapped.

    psi = atan2(fz, fx) of the world FORWARD_AXIS; np.unwrap removes the
    +/-pi jumps so frame differences are physical.
    """
    forward = np.einsum("tij,j->ti", root_rotmats, FORWARD_AXIS)
    psi = np.arctan2(forward[:, 2], forward[:, 0])
    return np.unwrap(psi)


def f2_to_world_np(pose_f2, traj_f2, psi_anchor, trans_anchor):
    """(T, 138) tilt pose + (T, 4) traj -> (world_pose_6d (T, 138), world_trans (T, 3)).

    Integrates psi_dot/v_h from the window anchors (heading frame at
    psi_anchor, position at trans_anchor); world height is traj_f2[:, 3]
    directly.  Joints 1-22 are untouched (yaw-invariant parent-local 6D).
    """
    pose_f2 = np.asarray(pose_f2, dtype=np.float64).reshape(-1, N_JOINTS, 6)
    traj_f2 = np.asarray(traj_f2, dtype=np.float64).reshape(-1, 4)
    psi_anchor = float(psi_anchor)
    trans_anchor = np.asarray(trans_anchor, dtype=np.float64).reshape(3)
    psi_rel = np.cumsum(traj_f2[:, 0], axis=0) / float(FPS)
    psi = psi_anchor + psi_rel
    r_yaw = yaw_rotmat_np(psi)
    r_tilt = rot6d_to_rotmat_np(pose_f2[:, 0])
    r_root_world = np.einsum("tij,tjk->tik", r_yaw, r_tilt)
    v_h = traj_f2[:, 1:3]
    v_h3 = np.stack([v_h[:, 0], np.zeros_like(v_h[:, 0]), v_h[:, 1]], axis=1)
    v_world_xy = np.einsum("tij,tj->ti", r_yaw, v_h3)[:, [0, 2]]
    pos_rel = np.cumsum(v_world_xy, axis=0) / float(FPS)
    world_trans = trans_anchor[None, :] + np.stack(
        [pos_rel[:, 0], traj_f2[:, 3] - trans_anchor[1], pos_rel[:, 1]], axis=1
    )
    world_pose = pose_f2.copy()
    world_pose[:, 0] = rotmat_to_6d_np(r_root_world)
    return world_pose.reshape(-1, N_JOINTS * 6).astype(np.float32), world_trans.astype(np.float32)


def f2_to_world(pose_f2: torch.Tensor, traj_f2: torch.Tensor,
                psi_anchor: torch.Tensor, trans_anchor: torch.Tensor):
    """Torch twin of f2_to_world_np; batch-aware.

    pose_f2 (B, T, 138), traj_f2 (B, T, 4), psi_anchor (B,), trans_anchor (B, 3).
    Returns (world_pose (B, T, 138), world_trans (B, T, 3)).
    """
    batch, time = pose_f2.shape[0], pose_f2.shape[1]
    pose = pose_f2.reshape(batch, time, N_JOINTS, 6)
    psi_rel = torch.cumsum(traj_f2[:, :, 0], dim=1) / float(FPS)
    psi = psi_anchor.view(batch, 1) + psi_rel
    r_yaw = yaw_rotmat(psi)  # (B, T, 3, 3)
    r_tilt = rot6d_to_rotmat(pose[:, :, 0])
    r_root_world = torch.matmul(r_yaw, r_tilt)
    v_h = traj_f2[:, :, 1:3]
    v_h3 = torch.stack(
        [v_h[..., 0], torch.zeros_like(v_h[..., 0]), v_h[..., 1]], dim=-1
    )
    v_world_xy = torch.einsum("btij,btj->bti", r_yaw, v_h3)[..., [0, 2]]
    pos_rel = torch.cumsum(v_world_xy, dim=1) / float(FPS)
    world_trans = torch.stack(
        [trans_anchor[:, 0:1] + pos_rel[:, :, 0],
         traj_f2[:, :, 3],
         trans_anchor[:, 2:3] + pos_rel[:, :, 1]],
        dim=-1,
    )
    world_pose = pose.clone()
    world_pose[:, :, 0] = rotmat_to_6d(r_root_world)
    return world_pose.reshape(batch, time, N_JOINTS * 6), world_trans
