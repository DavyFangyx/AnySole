"""SAM3DB package predictions -> unified contract payload (decision archive 11, W2).

Mirrors ``eval_rgb_smpl.predicted_geometry`` for SAM3DB: origin-frame neutral
SMPL forward, then the per-frame similarity transform (rotation/scale/
translation plus the weak-perspective camera translation), then the shared
camera -> checkerboard -> mocap-world chain.  The FK root is the
similarity-transformed pelvis (``camera_translation`` is not the root).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from anysole.data.rgb_smpl import coordinates as _coords
from anysole.data.rgb_smpl import io as _io
from . import common

BASELINE = "SAM3DB"


def export_session(row: dict, args) -> dict:
    session = row["session_id"]
    cam = int(args.cam)
    date, subject = common.session_parts(row)
    package = Path(args.package_root)
    entry = common.load_unit(package, BASELINE, date, session, cam,
                                  manifest_override=getattr(args, 'unit_manifest', None) or None)
    unit = common.unit_object(entry)
    result = package / "reference_results" / "3_Result" / f"{BASELINE}_baseline" / unit.short_id / "result.pkl"
    raw = _io.load_sam(result, unit)
    native_poses = np.zeros((unit.frame_numbers.size, 72), dtype=np.float32)
    native_poses[:, :3], native_poses[:, 3:66] = raw["global_orient"], raw["body_pose"]
    visual = common.canonical_grid(row, cam)
    epoch, epoch_stats = common.align_unit_to_canonical(unit, visual)
    # Interpolate the similarity parameters BEFORE the forward so the surface
    # and the parameters stay on the same path (decision archive 11, W2).
    similarity_rotvec = Rotation.from_matrix(raw["similarity_rotation"]).as_rotvec()
    resampled, resample_valid, resample_stats = common.resample_on_canonical(
        {"pose": native_poses.reshape(native_poses.shape[0], 24, 3),
         "betas": raw["betas"],
         "similarity_rotation": similarity_rotvec,
         "similarity_scale": raw["similarity_scale"],
         "similarity_translation": raw["similarity_translation"],
         "camera_translation": raw["camera_translation"]},
        unit.physical_times, visual, epoch)
    poses = resampled["pose"].reshape(visual.size, 72).astype(np.float32)
    betas = resampled["betas"]
    vertices, joints = common.smpl_forward(poses, betas)
    rotation = Rotation.from_rotvec(resampled["similarity_rotation"]).as_matrix()
    scale = resampled["similarity_scale"]
    translation = resampled["similarity_translation"] + resampled["camera_translation"]
    # eval_rgb_smpl.py:112-113 row-vector similarity application, kept verbatim.
    vertices = scale[:, None, None] * np.einsum("tvi,tij->tvj", vertices, rotation) + translation[:, None]
    joints = scale[:, None, None] * np.einsum("tji,tik->tjk", joints, rotation) + translation[:, None]
    calibration, calib_path, calib_sha = common.load_calibration(date)
    camera = calibration["cameras"][f"cam{unit.camera_index}"]
    vertices_world = common.camera_chain(vertices, camera, calibration)
    joints_world = common.camera_chain(joints, camera, calibration)
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(
        visual.size, 24, 3, 3)
    rotations[:, 0] = np.einsum("tji,tjk->tik", rotation, rotations[:, 0])
    rotations[:, 0] = _coords.camera_root_rotation_to_checkerboard(rotations[:, 0], camera)
    rotations[:, 0] = common.checkerboard_to_gt_root_rotation(rotations[:, 0], calibration)
    from anysole.data.smpl_io import load_smpl

    gt_motion = load_smpl(common.resolve_uri(row["smpl_path"]))
    # FK root = similarity-transformed pelvis (translation is not the root).
    fk_joints, poses_aa = common.protocol_assembly(rotations, joints_world[:, 0], gt_motion)
    return {
        "joints": fk_joints.astype(np.float32),
        "vertices": vertices_world.astype(np.float32),
        "joint_rotations": rotations.astype(np.float32),
        "poses": poses_aa.astype(np.float32),
        "valid": resample_valid,
        "source": str(result),
        "extra_meta": common.export_meta(unit, result, calibration, calib_path, calib_sha,
                                         epoch_stats, resample_stats, replay=0.0),
    }
