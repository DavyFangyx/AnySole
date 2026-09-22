"""Unified motion reader for results_display.

AnySole motion artifacts use the standard SMPL NPZ contract.  BVH remains
supported for Step2Motion and legacy result folders; callers do not need a
format flag because the archive suffix/keys are inspected automatically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ``motion_io`` is shared by scripts that are commonly launched as files, for
# example ``python results_display/script/r_test1_visualize_anysole.py``.  In that
# launch mode Python adds only ``results_display/script`` to ``sys.path``, not
# the repository root that contains the ``anysole`` package.  Bootstrap both
# locations here, before importing AnySole, so every visualization entry point
# gets the same format adapter without requiring callers to set PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # utils/ -> script/ -> gait repo root
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from anysole.data.smpl_io import load_smpl, resolve_smpl_path
from anysole.utils.geometry import fk_pose6d_np
from anysole.types import JOINT_NAMES, SMPL_ROOTS
try:  # direct script execution
    from bvh_aligner_pose import parse_bvh_aligner
except ModuleNotFoundError:  # package import (tests/tools)
    from .bvh_aligner_pose import parse_bvh_aligner


# Historical Skeleton3 order used only by Step2Motion/legacy result files.
# Keeping it in the display adapter prevents the AnySole model package from
# acquiring a second motion protocol again.
LEGACY_BVH_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)


def joints_to_meters(joints):
    pts = np.asarray(joints, dtype=np.float32)
    return pts * 0.01 if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0 else pts


def smpl_yup_to_display(joints):
    """Native SMPL (x,y-up,z) -> renderer/common z-up (x,-z,y)."""
    pts = np.asarray(joints, dtype=np.float64)
    return np.stack((pts[..., 0], -pts[..., 2], pts[..., 1]), axis=-1)


def interp_joints(joints, frame_time, query_t):
    src_t = np.arange(joints.shape[0], dtype=np.float64) * float(frame_time)
    clipped = np.clip(np.asarray(query_t, dtype=np.float64), src_t[0], src_t[-1])
    out = np.empty((len(clipped),) + joints.shape[1:], dtype=np.float64)
    for joint_i in range(joints.shape[1]):
        for axis_i in range(3):
            out[:, joint_i, axis_i] = np.interp(clipped, src_t, joints[:, joint_i, axis_i])
    return out.astype(np.float32)


def detect_motion_format(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".bvh":
        return "bvh"
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported motion file: {path}")
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
    if {"poses", "trans"}.issubset(keys) or {"root_orient", "pose_body", "trans"}.issubset(keys):
        return "smpl"
    raise ValueError(f"Cannot identify motion NPZ format from keys {sorted(keys)}: {path}")


def load_motion(path: Path, query_t: Optional[np.ndarray] = None) -> dict:
    """Load one motion file and return ``joints`` (T,J,3), ``parents``, format."""
    path = Path(path)
    fmt = detect_motion_format(path)
    if fmt == "bvh":
        parsed = parse_bvh_aligner(path, trim_leading_seconds=0.0)
        joints = joints_to_meters(parsed["joints"])
        if query_t is not None:
            joints = joints_to_meters(interp_joints(parsed["joints"], parsed["frame_time"], query_t))
        return {
            "joints": joints,
            "parents": np.asarray(parsed["parents"], dtype=np.int64),
            "names": tuple(parsed["names"]),
            "format": fmt,
        }
    if fmt == "smpl":
        motion = load_smpl(path, query_t=query_t)
        joints = fk_pose6d_np(
            motion["pose_6d"], motion["trans_m"],
            motion["offsets_m"], motion["parents"],
        )
        return {
            "joints": smpl_yup_to_display(joints),
            "parents": motion["parents"],
            "names": tuple(JOINT_NAMES),
            "format": fmt,
        }
    raise AssertionError(f"unhandled motion format: {fmt}")


def load_session_gt(seq_dir: Path, n_frames: int, fps: float = 40.0) -> dict:
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    try:
        smpl = resolve_smpl_path(meta, tuple(SMPL_ROOTS))
        result = load_motion(smpl, query_t=t_mocap)
        result["source_path"] = str(smpl)
    except FileNotFoundError:
        # Legacy/Step2Motion sessions may only have BVH.  This fallback is
        # confined to the display reader; AnySole training/inference never
        # takes it.
        bvh = Path(str(meta.get("bvh_path", ""))).expanduser()
        if not bvh.is_file():
            raise
        result = load_motion(bvh, query_t=t_mocap)
        result["source_path"] = str(bvh)
    # Extra keys let mesh renderers reuse the exact same time grid/archive:
    # ``format`` (smpl/bvh) already comes from load_motion.
    result["t_mocap"] = t_mocap
    return result
