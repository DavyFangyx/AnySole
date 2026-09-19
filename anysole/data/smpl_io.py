"""SMPL-NPZ loading and conversion to AnySole's canonical 23-joint protocol.

The motion archive produced by the Mocap SMPL pipeline stores axis-angle
rotations for the standard SMPL-24 kinematic tree and translations in meters.
AnySole historically consumed a Skeleton3 BVH (23 joints), so this module is
the deliberately small compatibility boundary: the model keeps its frozen
23x6D target while the source of truth can now be SMPL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation, Slerp



def _rotmat_to_6d_np(rotmats: np.ndarray) -> np.ndarray:
    return np.concatenate([rotmats[..., :, 0], rotmats[..., :, 1]], axis=-1).astype(np.float32)

SMPL24_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)

# Canonical AnySole joint -> SMPL joint.  ToeBase is synthesized as a short
# terminal joint because SMPL has no toe joint in its 24-joint skeleton.  The
# Skeleton3 tree has one extra torso bridge compared with SMPL: canonical
# ``Spine3`` receives SMPL neck, canonical ``Neck`` is an identity bridge, and
# ``Head`` receives SMPL head, preserving the original parent topology.
_DIRECT = (0, 3, 6, 9, 12, None, 15, None, 18, 20, 22, None, 19, 21, 23, 1, 4, 7, None, 2, 5, 8, None)
_COLLAR = {7: (13, 16), 11: (14, 17)}
_SMPL_PARENTS = (-1, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 9, 9, 12, 12, 12, 16, 17, 16, 17, 18, 19)


def _scalar(value, default):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else default


def _read_bvh_offsets(path: Optional[Path]) -> Dict[str, np.ndarray]:
    """Read SMPL-24 rest offsets when the paired archival BVH is available."""
    if path is None or not Path(path).is_file():
        return {}
    names, offsets, stack = [], [], []
    lines = Path(path).read_text(errors="replace").replace("\r", "").splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            names.append(s.split()[1]); stack.append(len(names) - 1)
        elif s.startswith("OFFSET") and stack and stack[-1] >= 0:
            offsets.append(np.asarray([float(v) for v in s.split()[1:4]], dtype=np.float32))
        elif s == "End Site":
            stack.append(-999)
        elif s == "}" and stack:
            stack.pop()
        elif s == "MOTION":
            break
    return {n: offsets[i] for i, n in enumerate(names) if i < len(offsets)}


def resolve_smpl_path(meta: dict, roots: tuple[Path, ...], cache: Optional[dict] = None) -> Path:
    """Resolve a session's NPZ from metadata or the supplied archive roots."""
    recorded = str(meta.get("smpl_path", ""))
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    sid = str(meta.get("session_id", ""))
    if cache is not None and sid in cache:
        return cache[sid]
    for root in roots:
        root = Path(root)
        candidates = sorted(root.glob(f"**/{sid}/motion_neutral_smpl.npz"))
        if candidates:
            if cache is not None:
                cache[sid] = candidates[0]
            return candidates[0]
    raise FileNotFoundError(f"SMPL NPZ not found for {sid}; searched {roots}")


def _pose_arrays(data: dict) -> np.ndarray:
    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float64)
    else:
        root = np.asarray(data["root_orient"], dtype=np.float64)
        body = np.asarray(data["pose_body"], dtype=np.float64)
        poses = np.concatenate([root, body], axis=-1)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"SMPL poses must have shape (T,72), got {poses.shape}")
    return poses.reshape(-1, 24, 3)


def load_smpl(path: Path, query_t: Optional[np.ndarray] = None, paired_bvh: Optional[Path] = None) -> dict:
    """Load/resample one SMPL archive and expose AnySole-compatible fields."""
    data = dict(np.load(Path(path), allow_pickle=True))
    poses = _pose_arrays(data)
    trans = np.asarray(data["trans"], dtype=np.float64)
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"SMPL trans must have shape ({poses.shape[0]},3), got {trans.shape}")
    fps = _scalar(data.get("mocap_frame_rate", 120.0), 120.0)
    src_t = np.asarray(data.get("source_frame_times_s", np.arange(len(poses)) / fps), dtype=np.float64).reshape(-1)
    if src_t.size != len(poses):
        src_t = np.arange(len(poses), dtype=np.float64) / fps
    if query_t is not None:
        q = np.asarray(query_t, dtype=np.float64)
        clipped = np.clip(q, src_t[0], src_t[-1])
        if len(src_t) == 1:
            poses = np.repeat(poses[:1], len(q), axis=0)
            trans = np.repeat(trans[:1], len(q), axis=0)
        else:
            sampled = np.empty((len(q), 24, 3), dtype=np.float64)
            for joint in range(24):
                key = SciRotation.from_rotvec(poses[:, joint])
                sampled[:, joint] = Slerp(src_t, key)(clipped).as_rotvec()
            poses = sampled
            trans = np.stack([np.interp(clipped, src_t, trans[:, i]) for i in range(3)], axis=1)

    smpl_R = SciRotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    # Compose collar+shoulder because the canonical Skeleton3 tree has no
    # collar joints.  All other joints retain their SMPL local rotation.
    canon_R = np.tile(np.eye(3, dtype=np.float64), (len(poses), 23, 1, 1))
    for j, src in enumerate(_DIRECT):
        if src is not None:
            canon_R[:, j] = smpl_R[:, src]
    for j, (collar, shoulder) in _COLLAR.items():
        canon_R[:, j] = smpl_R[:, collar] @ smpl_R[:, shoulder]

    offsets_smpl = _read_bvh_offsets(paired_bvh)
    def off(name):
        return offsets_smpl.get(name, np.zeros(3, dtype=np.float32)).astype(np.float64)
    offsets = np.zeros((23, 3), dtype=np.float64)
    offsets[1] = off("spine1")
    offsets[2] = off("spine2")
    offsets[3] = off("spine3")
    offsets[4] = off("neck")
    offsets[5] = np.zeros(3, dtype=np.float64)
    offsets[6] = off("head")
    offsets[7] = off("left_collar") + off("left_shoulder")
    offsets[8:11] = [off("left_elbow"), off("left_wrist"), off("left_hand")]
    offsets[11] = off("right_collar") + off("right_shoulder")
    offsets[12:15] = [off("right_elbow"), off("right_wrist"), off("right_hand")]
    offsets[15:19] = [off("left_hip"), off("left_knee"), off("left_ankle"), off("left_foot")]
    offsets[18] = np.array([0.0, 0.0, 0.05])
    offsets[19:23] = [off("right_hip"), off("right_knee"), off("right_ankle"), off("right_foot")]
    pose6d = _rotmat_to_6d_np(canon_R).reshape(len(poses), 23 * 6).astype(np.float32)
    return {
        "pose_6d": pose6d,
        "trans_m": trans.astype(np.float32),
        "offsets_m": offsets.astype(np.float32),
        "parents": np.asarray((-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21), dtype=np.int64),
        "fps": fps,
        "source_path": str(path),
    }
