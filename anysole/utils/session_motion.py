"""Shared motion postprocessing for evaluation metrics and saved predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from anysole.types import FPS, JOINT_NAMES, N_JOINTS, POSE_DIM
from anysole.utils.geometry import fk_pose6d, rot6d_to_rotmat, rotmat_to_6d


PIPELINE_VERSION = "session_frames_v2"


def pack_motion_windows(session: dict, windows: list[tuple], *, fade: int = 4) -> dict:
    """Restore original frames; blend adjacent ranges without mutating inputs."""
    n = len(session["V_feat"])
    identity = np.tile([1., 0., 0., 0., 1., 0.], N_JOINTS).astype(np.float32)
    pose = np.repeat(identity[None], n, axis=0)
    trans = np.zeros((n, 3), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    ordered = [(int(start), np.asarray(p, dtype=np.float32).copy(),
                np.asarray(t, dtype=np.float32).copy())
               for start, p, t, *_ in sorted(windows, key=lambda w: int(w[0]))]
    for start, p, t in ordered:
        if start < 0 or p.ndim != 2 or p.shape[1] != POSE_DIM or t.shape != (len(p), 3):
            raise ValueError("motion window must have a nonnegative start, [T,144] pose and [T,3] translation")
    if fade:
        for previous, current in zip(ordered, ordered[1:]):
            ps, pp, pt = previous
            cs, cp, ct = current
            if cs != ps + len(pp) or min(len(pp), len(cp)) < fade:
                continue
            for j in range(fade):
                alpha = (j + 1) / (fade + 1)
                a, b = len(pp) - fade + j, j
                pa, pb, ta, tb = pp[a].copy(), cp[b].copy(), pt[a].copy(), ct[b].copy()
                pp[a], cp[b] = (1 - alpha) * pa + alpha * pb, (1 - alpha) * pb + alpha * pa
                pt[a], ct[b] = (1 - alpha) * ta + alpha * tb, (1 - alpha) * tb + alpha * ta
    for start, p, t in ordered:
        count = min(len(p), max(n - start, 0))
        if not count:
            continue
        if seen[start:start + count].any():
            raise ValueError("motion windows must not overlap")
        pose[start:start + count], trans[start:start + count] = p[:count], t[:count]
        seen[start:start + count] = True
    fake = np.asarray(session.get("fake_mask", np.zeros(n)), dtype=bool)
    if fake.shape != (n,):
        raise ValueError("session fake mask does not match the original frame grid")
    # Metrics and archives consume exactly the same orthonormalized rotations.
    pose_tensor = torch.from_numpy(pose.reshape(-1, N_JOINTS, 6))
    pose = rotmat_to_6d(rot6d_to_rotmat(pose_tensor)).reshape(-1, POSE_DIM).numpy()
    times = np.asarray(session.get("frame_times_s", np.arange(n) / FPS), dtype=np.float64)
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("session times must be finite and strictly increasing on the original frame grid")
    return {"pose": pose, "trans": trans,
            "gt_trans": np.asarray(session["trans_global"], dtype=np.float32),
            "valid": seen & ~fake, "frame_indices": np.arange(n, dtype=np.int64),
            "times": times}


def prepare_session_sequence(seq: dict, raw: list[dict], session: dict) -> tuple[dict, dict]:
    """Use one postprocessed prediction for metrics and motion serialization."""
    tw = raw[0]["pose_gt"].shape[0]
    starts = [int(sample["frame_start"]) for sample in raw]
    poses = seq["pred_pose"].detach().cpu().numpy().reshape(len(raw), tw, POSE_DIM)
    trans = seq["pred_trans"].detach().cpu().numpy().reshape(len(raw), tw, 3)
    packed = pack_motion_windows(session, list(zip(starts, poses, trans)))
    source_ids = np.concatenate([np.arange(start, start + tw) for start in starts])
    if np.any(np.diff(source_ids) <= 0):
        raise ValueError("evaluation windows must be ordered and non-overlapping")
    ids = np.flatnonzero(packed["valid"])
    positions = np.searchsorted(source_ids, ids)
    result = dict(seq)
    temporal_fields = ("gt_pose", "gt_trans", "kp_gt", "contact_gt", "pressure_gt", "pressure_pred")
    for key in temporal_fields:
        if seq.get(key) is not None:
            result[key] = seq[key][torch.as_tensor(positions, device=seq[key].device)]
    for key, field in (("pred_pose", "pose"), ("pred_trans", "trans")):
        result[key] = torch.as_tensor(packed[field][ids], device=seq[key].device, dtype=seq[key].dtype)
    result["frame_indices"] = ids
    result["times"] = packed["times"][ids]
    return result, packed


def write_motion_archive(path: Path, session: dict, packed: dict) -> None:
    """Save the same processed poses used by the session metric evaluator."""
    from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata

    pose, trans = packed["pose"], packed["trans"]
    betas = np.asarray(session.get("betas", np.zeros(10)), dtype=np.float32)
    joints = fk_pose6d(torch.from_numpy(pose)[None], torch.from_numpy(trans)[None],
                      torch.as_tensor(session["offsets"]).float(),
                      torch.as_tensor(session["parents"]).long())[0].numpy()
    joints = np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)
    poses = smpl24_pose6d_to_poses(pose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, poses=poses, trans=pelvis_to_smpl_trans(pose, trans, betas),
        pred_pelvis_trans=trans, gt_pelvis_trans=packed["gt_trans"], gt_trans=packed["gt_trans"],
        betas=betas, betas_source=np.asarray("ground_truth_session"),
        joint_xyz_world=joints, joint_names=np.asarray(JOINT_NAMES),
        valid_mask=packed["valid"], frame_indices=packed["frame_indices"],
        joint_coordinate_system=np.asarray("world_z_up"), session_id=np.asarray(session["session_id"]),
        root_orient=poses[:, :3], pose_body=poses[:, 3:],
        mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
        source_frame_times_s=packed["times"], evaluation_pipeline=np.asarray(PIPELINE_VERSION),
        **smpl_archive_metadata())
