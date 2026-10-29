"""GVHMR package predictions -> unified contract payload (decision archive 11, W3).

Mirrors ``eval_rgb_smpl.predicted_geometry`` for GVHMR: the camera-space branch
(``pose/betas/trans``), native joint replay gate below 0.001 mm RMS against
``joints_camera``, resample to the canonical grid, then the shared
camera -> checkerboard -> mocap-world chain.  The global branch
(``pose_global/trans_global``) stays unverified and unused (archive 11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from anysole.data.rgb_smpl import coordinates as _coords
from . import common

BASELINE = "GVHMR"


def export_session(row: dict, args) -> dict:
    session = row["session_id"]
    cam = int(args.cam)
    date, subject = common.session_parts(row)
    package = Path(args.package_root)
    entry = common.load_unit(package, BASELINE, date, session, cam,
                                  manifest_override=getattr(args, 'unit_manifest', None) or None)
    unit = common.unit_object(entry)
    result = package / "reference_results" / "3_Result" / f"{BASELINE}_baseline" / unit.short_id / "result.npz"
    with np.load(result, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in
               ("pose", "betas", "trans", "source_frame_numbers", "physical_times_s", "joints_camera")}
    if not np.array_equal(raw["source_frame_numbers"], unit.frame_numbers):
        raise ValueError(f"{session}: GVHMR source frame mismatch")
    if not np.array_equal(raw["physical_times_s"], unit.physical_times):
        raise ValueError(f"{session}: GVHMR timestamp mismatch")
    calibration, calib_path, calib_sha = common.load_calibration(date)
    camera = calibration["cameras"][f"cam{unit.camera_index}"]
    visual = common.canonical_grid(row, cam)
    # Wall-clock epoch from the recording meta.json (no shift search, archive 11).
    recording_start = common.recording_start_day_s(row)
    epoch, epoch_stats = common.anchor_unit_to_canonical(unit, visual, recording_start)
    # Native forward + replay gate (eval_rgb_smpl.py:124-127 same criterion).
    _, joints_native = common.smpl_forward(raw["pose"], raw["betas"], raw["trans"])
    replay = float(np.sqrt(np.mean((joints_native - raw["joints_camera"]) ** 2)) * 1000)
    if not np.isfinite(replay) or replay > 0.001:
        raise ValueError(f"{session}: GVHMR joint replay RMS {replay} mm")
    resampled, resample_valid, resample_stats = common.resample_on_canonical(
        {"pose": raw["pose"].reshape(raw["pose"].shape[0], 24, 3),
         "betas": raw["betas"], "trans": raw["trans"]},
        unit.physical_times, visual, epoch)
    poses = resampled["pose"].reshape(visual.size, 72).astype(np.float32)
    betas = resampled["betas"]
    trans = resampled["trans"]
    verts, _ = common.smpl_forward(poses, betas, trans)
    calibration, calib_path, calib_sha = common.load_calibration(date)
    camera = calibration["cameras"][f"cam{unit.camera_index}"]
    vertices_world = common.camera_chain(verts, camera, calibration)
    transl_world = common.camera_chain(trans, camera, calibration)
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(
        visual.size, 24, 3, 3)
    rotations[:, 0] = _coords.camera_root_rotation_to_checkerboard(rotations[:, 0], camera)
    rotations[:, 0] = common.checkerboard_to_gt_root_rotation(rotations[:, 0], calibration)
    from anysole.data.smpl_io import load_smpl

    gt_motion = load_smpl(common.resolve_uri(row["smpl_path"]))
    fk_joints, poses_aa = common.protocol_assembly(rotations, transl_world, gt_motion)
    return {
        "joints": fk_joints.astype(np.float32),
        "vertices": vertices_world.astype(np.float32),
        "joint_rotations": rotations.astype(np.float32),
        "poses": poses_aa.astype(np.float32),
        "valid": resample_valid,
        "source": str(result),
        "extra_meta": common.export_meta(unit, result, calibration, calib_path, calib_sha,
                                         epoch_stats, resample_stats, replay),
    }
