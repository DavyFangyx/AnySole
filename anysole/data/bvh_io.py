"""Parse, resample, and write 23-joint Skeleton3 BVH files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from anysole.geometry import euler_yxz_to_6d, resample_bvh_motion, rot6d_to_euler_yxz
from anysole.types import JOINT_NAMES, JOINT_PARENTS, N_JOINTS


class BVHData:
    def __init__(
        self,
        hierarchy: str,
        names: List[str],
        parents: np.ndarray,
        offsets_cm: np.ndarray,
        motion: np.ndarray,
        frame_time: float,
    ):
        self.hierarchy = hierarchy
        self.names = names
        self.parents = parents.astype(np.int64)
        self.offsets_cm = offsets_cm.astype(np.float64)
        self.motion = motion.astype(np.float64)
        self.frame_time = float(frame_time)

    @property
    def offsets_m(self) -> np.ndarray:
        return self.offsets_cm * 0.01


def _parse_hierarchy(text: List[str]) -> tuple[str, List[str], np.ndarray, np.ndarray, int]:
    motion_idx = next(i for i, line in enumerate(text) if line.startswith("MOTION"))
    hierarchy = "\n".join(text[:motion_idx]).rstrip() + "\n"
    names: List[str] = []
    parents: List[int] = []
    offsets: List[List[float]] = []
    stack: List[int] = []
    for line in text[:motion_idx]:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            name = stripped.split()[1]
            parent = stack[-1] if stack else -1
            names.append(name)
            parents.append(parent)
            stack.append(len(names) - 1)
        elif stripped.startswith("End Site"):
            stack.append(-999)
        elif stripped.startswith("OFFSET") and stack and stack[-1] >= 0 and len(offsets) < len(names):
            offsets.append([float(v) for v in stripped.split()[1:4]])
        elif stripped == "}":
            if stack:
                stack.pop()
    if names != list(JOINT_NAMES):
        raise ValueError("Unexpected BVH joint names: %s" % names)
    if tuple(parents) != JOINT_PARENTS:
        raise ValueError("Unexpected BVH parents: %s" % parents)
    if len(offsets) != N_JOINTS:
        raise ValueError("Unexpected BVH joint count %d" % len(offsets))
    return hierarchy, names, np.asarray(parents, dtype=np.int64), np.asarray(offsets, dtype=np.float64), motion_idx


def load_bvh(path: Path) -> BVHData:
    text = Path(path).read_text(errors="replace").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    hierarchy, names, parents, offsets, motion_idx = _parse_hierarchy(text)
    n_frames = int(text[motion_idx + 1].split(":")[1])
    frame_time = float(text[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in text[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape[1] != 72:
        raise ValueError("Unexpected BVH channel count %d in %s" % (motion.shape[1], path))
    return BVHData(hierarchy, names, parents, offsets, motion, frame_time)


def write_bvh(
    path: Path,
    hierarchy: str,
    motion: np.ndarray,
    frame_time: float,
) -> None:
    if motion.ndim != 2 or motion.shape[1] != 72:
        raise ValueError("BVH motion must be (T, 72), got %s" % (motion.shape,))
    lines = [hierarchy.rstrip("\n"), "MOTION"]
    lines.append("Frames: %d" % motion.shape[0])
    lines.append("Frame Time: %.8f" % float(frame_time))
    for row in motion:
        lines.append(" ".join("%.6f" % float(v) for v in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def pose_trans_to_motion(pose_6d: np.ndarray, trans_m: np.ndarray) -> np.ndarray:
    """Convert (T, 138) 6D + (T, 3) meters into 72-channel BVH motion."""
    n = pose_6d.shape[0]
    eulers = rot6d_to_euler_yxz(pose_6d.reshape(n, N_JOINTS, 6)).reshape(n, N_JOINTS * 3)
    motion = np.empty((n, 72), dtype=np.float64)
    motion[:, :3] = np.asarray(trans_m, dtype=np.float64) * 100.0
    motion[:, 3:] = eulers
    return motion


def motion_to_pose_trans(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_6d = euler_yxz_to_6d(motion[:, 3:].reshape(motion.shape[0], N_JOINTS, 3)).reshape(motion.shape[0], N_JOINTS * 6)
    trans_m = motion[:, :3] * 0.01
    return pose_6d.astype(np.float32), trans_m.astype(np.float32)


def resample_session_bvh(bvh_path: Path, t_mocap: np.ndarray, hierarchy_cache: Optional[Dict[str, BVHData]] = None) -> dict:
    key = str(bvh_path)
    if hierarchy_cache is not None and key in hierarchy_cache:
        bvh = hierarchy_cache[key]
    else:
        bvh = load_bvh(bvh_path)
        if hierarchy_cache is not None:
            hierarchy_cache[key] = bvh
    sampled = resample_bvh_motion(bvh.motion, bvh.frame_time, t_mocap)
    pose_6d, trans_m = motion_to_pose_trans(sampled)
    return {
        "pose_6d": pose_6d,
        "trans_m": trans_m,
        "offsets_m": bvh.offsets_m.astype(np.float32),
        "parents": bvh.parents.copy(),
        "hierarchy": bvh.hierarchy,
        "names": list(bvh.names),
    }
