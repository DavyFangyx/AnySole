"""Step2Motion adapter: shared facts + canonical split -> native gait inputs.

Rebuilds the Step2Motion ``gait`` / ``gait_noimu`` datasets from the frozen
public workspace contract only:

* split membership: ``protocol://splits/default/splits.csv`` (read-only, no
  self-split logic);
* per-frame facts: ``shared/facts/sessions/cam3/<date>/<subject>/<sid>/``
  (``frames.npz`` frame id / mocap time / valid / fake,
  ``pressure_48.npz`` left/right 4x12 grids);
* BVH source: the shared session metadata ``source_files.bvh``.

Semantics preserved verbatim from the native pipeline: BVH-23 joint order,
Y-X-Z euler parsing, root translation, synthesized foot IMU, the D_Test4
accepted 48->16 pooling (heel[0:8] + toe[8:16]) and the MotionDatasetState
clip layout (first frame of every clip discarded).  The time axis is the
shared ``mocap_time_s``; no visual/pressure offset is constructed here.
Every kept input frame carries its shared ``frame_id`` (``clip_frame_ids``);
fake frames never enter clips.

Writes (all under ``model_inputs/Step2Motion/<adapter_version>/``):

    gait/{gait_train,gait_val,gait_test}.pt + normalizer_gait.pth + split_sessions.json
    gait_noimu/... + normalizer_gait_noimu.pth + split_sessions.json
    artifact.json, pressure_ranges.csv

Run (touch_gait env, repo root):

    python3 -m AnysoleWorkspace.tool.adapters.Step2Motion.build_gait
    python3 -m AnysoleWorkspace.tool.adapters.Step2Motion.build_gait --sessions S10101 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
ADAPTER_VERSION = "adapter_v1"
INPUT_ROOT = WORKSPACE / "model_inputs/Step2Motion" / ADAPTER_VERSION
SPLIT_CSV = WORKSPACE / "protocol/splits/default/splits.csv"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
STEP2MOTION_SRC = ROOT / "Baselines/Step2Motion/src"

TARGET_HZ = 40.0
G = 9.81
# input_T=100 plus the first frame MotionDataset discards
MIN_FRAMES = 101
FOOT_INDICES = [21, 22, 17, 18]  # right foot, right toe, left foot, left toe

# BVH-23 semantic order and hierarchy, preserved from the native pipeline.
SRC_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]
SRC_PARENTS = np.array(
    [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21],
    dtype=np.int64,
)
JOINT_NAMES = SRC_JOINTS
JOINT_PARENTS = SRC_PARENTS
N_JOINTS = len(JOINT_NAMES)
LEFT_FOOT = JOINT_NAMES.index("LeftFoot")
RIGHT_FOOT = JOINT_NAMES.index("RightFoot")
LEFT_TOE = JOINT_NAMES.index("LeftToeBase")
RIGHT_TOE = JOINT_NAMES.index("RightToeBase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step2Motion gait inputs from shared facts.")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs; default: every canonical split session")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--imu-only", action="store_true", help="Build only the 50-dim gait datasets.")
    parser.add_argument("--no-imu", action="store_true", help="Build only the 38-dim gait_noimu datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Process without writing .pt files.")
    parser.add_argument("--self-test", action="store_true", help="Run unit checks and exit.")
    return parser.parse_args()


def read_split(path: Path = SPLIT_CSV) -> dict[str, list[str]]:
    """Canonical split membership in CSV row order; no split logic here."""
    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or not {"train", "val", "test"} <= set(rows.fieldnames):
            raise ValueError(f"{path} is not a canonical split table")
        for row in rows:
            for key in groups:
                value = (row.get(key) or "").strip()
                if value:
                    if value in groups[key]:
                        raise ValueError(f"{path}: duplicate session in {key}: {value}")
                    groups[key].append(value)
    if not any(groups.values()):
        raise ValueError(f"canonical split {path} is empty")
    return groups


def shared_session_dir(session_id: str) -> Path:
    matches = sorted(
        path.parent for path in FACTS_ROOT.glob(f"*/*/{session_id}/session.json") if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"shared session {session_id!r} not found under {FACTS_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"session id is ambiguous in shared facts: {session_id}: {matches}")
    return matches[0]


def load_shared_session(session_id: str) -> dict:
    session_dir = shared_session_dir(session_id)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = dict(np.load(session_dir / "frames.npz", allow_pickle=True))
    pressure = dict(np.load(session_dir / "pressure_48.npz", allow_pickle=True))
    required_frames = {"frame_id", "mocap_time_s", "valid", "fake"}
    required_pressure = {"frame_id", "left48", "right48", "valid", "fake"}
    if not required_frames <= frames.keys():
        raise ValueError(f"{session_dir}: missing frame keys {required_frames - frames.keys()}")
    if not required_pressure <= pressure.keys():
        raise ValueError(f"{session_dir}: missing pressure keys {required_pressure - pressure.keys()}")
    frame_id = np.asarray(frames["frame_id"])
    if frame_id.ndim != 1 or not np.array_equal(frame_id, np.arange(len(frame_id))):
        raise ValueError(f"{session_dir}: frame_id must be unique and monotonic from zero")
    n = len(frame_id)
    for key in ("mocap_time_s", "valid", "fake"):
        if np.asarray(frames[key]).shape != (n,):
            raise ValueError(f"{session_dir}: frames.{key} length does not match frame_id")
    if np.any(np.asarray(frames["fake"], dtype=bool) & np.asarray(frames["valid"], dtype=bool)):
        raise ValueError(f"{session_dir}: fake and valid masks overlap")
    if not np.array_equal(np.asarray(pressure["frame_id"]), frame_id):
        raise ValueError(f"{session_dir}: pressure frame_id differs from frames.npz")
    for key in ("left48", "right48"):
        if np.asarray(pressure[key]).shape != (n, 48):
            raise ValueError(f"{session_dir}: pressure.{key} shape != {(n, 48)}")
    if str(meta.get("session_id")) != str(session_id):
        raise ValueError(f"{session_dir}: session.json identity mismatch")
    return {
        "session_id": str(session_id),
        "dir": session_dir,
        "meta": meta,
        "frames": frames,
        "pressure": pressure,
        "source_artifact": session_dir / "artifact.json",
    }


def shared_bvh_path(session: dict) -> Path:
    value = session["meta"].get("source_files", {}).get("bvh", "")
    if not value:
        raise FileNotFoundError(f"{session['session_id']}: shared metadata has no BVH source")
    return resolve_uri(value, must_exist=True)


# ---------------------------------------------------------------------------
# Native conversion core (moved verbatim from the removed src/process_gait.py).
# ---------------------------------------------------------------------------

def pool_48_to_16(values48: np.ndarray) -> np.ndarray:
    """Map 4x12 sensors onto Moticon 16 channels.

    Grid is (row=width, col=length). Columns 0-5 are toe, 6-11 are heel.
    Each 4x6 half is pooled into 8 channels as 4 rows x 2 groups of 3 columns.
    Official order is heel[8] then toe[8].
    """
    grid = values48.reshape(-1, 4, 12)
    toe = grid[:, :, :6].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
    heel = grid[:, :, 6:].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
    return np.concatenate([heel, toe], axis=1).astype(np.float32)


def cop_from_grid(values48: np.ndarray) -> np.ndarray:
    grid = np.clip(values48.reshape(-1, 4, 12), 0.0, None)
    force = grid.sum(axis=(1, 2))
    rows = np.arange(4, dtype=np.float32)
    cols = np.arange(12, dtype=np.float32)
    cop_x = (grid.sum(axis=2) * rows).sum(axis=1) / np.maximum(force, 1e-6) / 3.0
    heel_to_toe = 11.0 - (grid.sum(axis=1) * cols).sum(axis=1) / np.maximum(force, 1e-6)
    cop_y = heel_to_toe / 11.0
    cop = np.stack([cop_x, cop_y], axis=1).astype(np.float32)
    cop[force <= 0] = 0.0
    return cop


def parse_bvh(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    text = path.read_text(errors="replace").splitlines()
    offsets = []
    joint_i = -1
    for line in text:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            joint_i += 1
        elif stripped.startswith("OFFSET") and joint_i >= 0 and len(offsets) == joint_i:
            offsets.append([float(v) for v in stripped.split()[1:4]])
        if stripped.startswith("MOTION"):
            break
    motion_idx = next(i for i, line in enumerate(text) if line.startswith("MOTION"))
    n_frames = int(text[motion_idx + 1].split(":")[1])
    frame_time = float(text[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in text[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape[1] != 72:
        raise ValueError(f"Unexpected BVH channel count {motion.shape[1]} in {path}")
    if len(offsets) != 23:
        raise ValueError(f"Unexpected BVH joint count {len(offsets)} in {path}")
    return motion, frame_time, np.asarray(offsets, dtype=np.float64)


def interp_motion(motion: np.ndarray, frame_time: float, query_t: np.ndarray) -> np.ndarray:
    src_t = np.arange(motion.shape[0], dtype=np.float64) * frame_time
    out = np.empty((query_t.size, motion.shape[1]), dtype=np.float64)
    for col in range(3):
        out[:, col] = np.interp(query_t, src_t, motion[:, col])
    clipped = np.clip(query_t, src_t[0], src_t[-1])
    eulers = motion[:, 3:].reshape(motion.shape[0], 23, 3)
    if motion.shape[0] == 1:
        out[:, 3:] = motion[0, 3:]
        return out
    for joint_i in range(23):
        key_rots = SciRotation.from_euler("YXZ", eulers[:, joint_i], degrees=True)
        slerp = Slerp(src_t, key_rots)
        out[:, 3 + 3 * joint_i : 6 + 3 * joint_i] = slerp(clipped).as_euler("YXZ", degrees=True)
    return out


def euler_yxz_to_rotmat(eulers_deg: np.ndarray) -> np.ndarray:
    flat = eulers_deg.reshape(-1, 3)
    mats = SciRotation.from_euler("YXZ", flat, degrees=True).as_matrix().astype(np.float64)
    return mats.reshape(*eulers_deg.shape[:-1], 3, 3)


def rotmat_to_quat_wxyz(rotmats: np.ndarray) -> np.ndarray:
    xyza = SciRotation.from_matrix(rotmats.reshape(-1, 3, 3)).as_quat().reshape(*rotmats.shape[:-2], 4)
    return np.concatenate([xyza[..., 3:], xyza[..., :3]], axis=-1).astype(np.float32)


def quat_inverse(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def fk_local(local_rotmats: np.ndarray, offsets: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_joints, _, _ = local_rotmats.shape
    global_rot = np.zeros_like(local_rotmats)
    global_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    for j in range(n_joints):
        parent = int(parents[j])
        if parent < 0:
            global_rot[:, j] = local_rotmats[:, j]
            global_pos[:, j] = 0.0
        else:
            global_rot[:, j] = global_rot[:, parent] @ local_rotmats[:, j]
            offset = offsets[j][None, :, None]
            global_pos[:, j] = global_pos[:, parent] + (global_rot[:, parent] @ offset)[..., 0]
    return global_pos, global_rot


def remap_skeleton(
    src_offsets_cm: np.ndarray,
    src_local_rotmats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_offsets_m = src_offsets_cm * 0.01
    return src_offsets_m, src_local_rotmats


def rest_foot_imu_axes(offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Match MotionDataset.compute_axes in rest pose."""
    identity = np.broadcast_to(np.eye(3), (1, N_JOINTS, 3, 3)).copy()
    pos, _ = fk_local(identity, offsets, JOINT_PARENTS)

    def compute_axes(toes_index: int, foot_index: int, opposite_foot_index: int, invert_up: bool) -> np.ndarray:
        right = pos[0, toes_index] - pos[0, foot_index]
        right = right / np.linalg.norm(right)
        up = pos[0, opposite_foot_index] - pos[0, foot_index]
        if invert_up:
            up = -up
        up = up / np.linalg.norm(up)
        forward = np.cross(right, up)
        forward = forward / np.linalg.norm(forward)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        return np.stack([right, up, forward], axis=-1)

    left = compute_axes(LEFT_TOE, LEFT_FOOT, RIGHT_FOOT, invert_up=True)
    right = compute_axes(RIGHT_TOE, RIGHT_FOOT, LEFT_FOOT, invert_up=False)
    return {"left": left, "right": right}


def angular_velocity_from_rotmats(rotmats: np.ndarray, dt: float) -> np.ndarray:
    q = rotmat_to_quat_wxyz(rotmats)
    dots = np.sum(q[1:] * q[:-1], axis=-1)
    signs = np.cumprod(np.concatenate([[1.0], np.where(dots < 0, -1.0, 1.0)]))
    q = q * signs[:, None]
    q_prev = np.concatenate([q[:1], q[:-1]], axis=0)
    q_rel = quat_mul(q, quat_inverse(q_prev))
    q_rel = np.where(q_rel[..., :1] < 0, -q_rel, q_rel)
    xyz = q_rel[..., 1:]
    w = np.clip(q_rel[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    axis = xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8)
    omega = (axis * angle[..., None]) / dt
    small = np.linalg.norm(xyz, axis=-1) < 1e-8
    omega[small] = 0.0
    if len(omega) > 1:
        omega[0] = omega[1]
    return omega.astype(np.float32)


def synthesize_imu(
    global_pos: np.ndarray,
    global_rot: np.ndarray,
    rest_axes: np.ndarray,
    dt: float,
    flip_local_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    acc_world = np.zeros_like(global_pos)
    acc_world[1:-1] = (global_pos[2:] - 2.0 * global_pos[1:-1] + global_pos[:-2]) / (dt * dt)
    if len(acc_world) > 2:
        acc_world[0] = acc_world[1]
        acc_world[-1] = acc_world[-2]
    acc_world[..., 1] += G
    imu_rot = global_rot @ rest_axes[None, ...]
    acc_local = np.einsum("...ji,...j->...i", imu_rot, acc_world) / G
    if flip_local_y:
        acc_local[..., 1] *= -1.0
    gyro_world = angular_velocity_from_rotmats(imu_rot, dt)
    gyro_local = np.einsum("...ji,...j->...i", imu_rot, gyro_world)
    return acc_local.astype(np.float32), gyro_local.astype(np.float32)


def foot_features(values48: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    pressure16 = pool_48_to_16(values48)
    force = values48.sum(axis=1, keepdims=True).astype(np.float32)
    cop = cop_from_grid(values48)
    return np.concatenate([pressure16, acc, gyro, force, cop], axis=1)


def foot_features_noimu(values48: np.ndarray) -> np.ndarray:
    """--no-imu: 19-dim per-foot channel = pressure16 + force1 + cop2.

    Identical to ``foot_features`` with the acc/gyro block removed; the IMU
    values are never computed (synthesize_imu is not called on this path).
    """
    pressure16 = pool_48_to_16(values48)
    force = values48.sum(axis=1, keepdims=True).astype(np.float32)
    cop = cop_from_grid(values48)
    return np.concatenate([pressure16, force, cop], axis=1)


def split_fake_clips(fake_mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    clips = []
    n = len(fake_mask)
    i = 0
    while i < n:
        if fake_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not fake_mask[j]:
            j += 1
        if j - i >= min_frames:
            clips.append((i, j))
        i = j
    return clips


def convert_session(session: dict, min_frames: int, no_imu: bool = False) -> list[dict] | None:
    """One shared session -> native clips.  Source: shared facts only."""
    sid = session["session_id"]
    frames = session["frames"]
    pressure = session["pressure"]
    n = int(session["meta"]["frame_count"])
    fake_mask = np.asarray(frames["fake"], dtype=bool)
    valid_mask = np.asarray(frames["valid"], dtype=bool)
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    # Shared canonical time axis; no visual/pressure offset is constructed.
    mocap_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    fps = float(session["meta"].get("fps", TARGET_HZ))
    dt = 1.0 / fps

    clips = split_fake_clips(fake_mask, min_frames)
    if not clips:
        return None

    # Pressure is consumed per shared frame (already on the shared grid).
    left48 = np.asarray(pressure["left48"], dtype=np.float32)
    right48 = np.asarray(pressure["right48"], dtype=np.float32)

    bvh = shared_bvh_path(session)
    motion, frame_time, src_offsets = parse_bvh(bvh)
    sampled = interp_motion(motion, frame_time, mocap_t)
    root_pos = sampled[:, :3] * 0.01
    src_local = np.zeros((n, 23, 3, 3), dtype=np.float64)
    src_local[:, 0] = euler_yxz_to_rotmat(sampled[:, 3:6])
    src_local[:, 1:] = euler_yxz_to_rotmat(sampled[:, 6:].reshape(n, N_JOINTS - 1, 3))
    dst_offsets, dst_local = remap_skeleton(src_offsets, src_local)
    dst_pos, dst_global_rot = fk_local(dst_local, dst_offsets, JOINT_PARENTS)
    dst_pos = dst_pos + root_pos[:, None, :]

    if no_imu:
        # --no-imu: the IMU channels are structurally absent. synthesize_imu
        # is never called, so no IMU value is ever computed or stored.
        insole = np.concatenate(
            [foot_features_noimu(left48), foot_features_noimu(right48)],
            axis=1,
        )
    else:
        rest_axes = rest_foot_imu_axes(dst_offsets)
        left_acc, left_gyro = synthesize_imu(
            dst_pos[:, LEFT_FOOT], dst_global_rot[:, LEFT_FOOT], rest_axes["left"], dt, True
        )
        right_acc, right_gyro = synthesize_imu(
            dst_pos[:, RIGHT_FOOT], dst_global_rot[:, RIGHT_FOOT], rest_axes["right"], dt, False
        )
        insole = np.concatenate(
            [foot_features(left48, left_acc, left_gyro), foot_features(right48, right_acc, right_gyro)],
            axis=1,
        )

    local_pos_root = dst_pos - root_pos[:, None, :]
    pose = local_pos_root[:, 1:, :].reshape(n, -1).astype(np.float32)
    quats = rotmat_to_quat_wxyz(dst_global_rot)
    displacements = np.zeros((n, 3), dtype=np.float32)
    displacements[1:] = (root_pos[1:] - root_pos[:-1]).astype(np.float32)
    root_quats = quats[:, 0]
    out = []
    for start, end in clips:
        # Native raw time window: first kept frame is start+1 (MotionDataset
        # discards the first frame of each clip); the end boundary is the
        # exclusive end index, computed on the same shared axis.
        t_of = lambda k: mocap_t[0] + k * dt  # noqa: E731
        out.append(
            {
                "insole": insole[start:end],
                "pose": pose[start:end],
                "quats": quats[start:end],
                "displacements": displacements[start:end],
                "global_rots": root_quats[start:end],
                "parents": JOINT_PARENTS.copy(),
                "offsets": dst_offsets.astype(np.float32),
                "joint_names": list(JOINT_NAMES),
                "session_id": sid,
                "clip_index": int(len(out)),
                "n_session_clips": int(len(clips)),
                # Every input frame keeps its shared frame id / validity;
                # these cover the clip rows before the native first-frame
                # discard and are trimmed together with the data.
                "frame_ids": frame_ids[start:end],
                "valid": valid_mask[start:end],
                "raw_bvh_path": str(bvh),
                "raw_start_time": float(t_of(start + 1)),
                "raw_end_time": float(t_of(end)),
            }
        )
    return out


class MotionDatasetState:
    """Picklable stand-in for official MotionDataset when pymotion is unavailable."""


def quat_mul_vec(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    rotmats = quat_wxyz_to_rotmat(quats)
    if vecs.ndim == quats.ndim:
        return np.einsum("...ij,...j->...i", rotmats, vecs)
    return np.einsum("...ij,...j->...i", rotmats, np.broadcast_to(vecs, quats.shape[:-1] + (3,)))


def quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    xyza = np.concatenate([quats[..., 1:], quats[..., :1]], axis=-1)
    return SciRotation.from_quat(xyza.reshape(-1, 4)).as_matrix().reshape(*quats.shape[:-1], 3, 3)


def world_imu_quat(foot_quat: np.ndarray, rest_axes: np.ndarray) -> np.ndarray:
    # rest_axes columns are right, up, forward in rest pose / foot local space.
    rotmats = quat_wxyz_to_rotmat(foot_quat) @ rest_axes[None, ...]
    return rotmat_to_quat_wxyz(rotmats)


def build_motion_dataset_state(clips_data: list[dict], no_imu: bool = False) -> object:
    import torch

    insole = np.concatenate([d["insole"] for d in clips_data], axis=0).astype(np.float32)
    pose = np.concatenate([d["pose"] for d in clips_data], axis=0).astype(np.float32)
    quats = np.concatenate([d["quats"] for d in clips_data], axis=0).astype(np.float32)
    displacements = np.concatenate([d["displacements"] for d in clips_data], axis=0).astype(np.float32)
    global_rots = np.concatenate([d["global_rots"] for d in clips_data], axis=0).astype(np.float32)
    offsets = np.stack([d["offsets"] for d in clips_data], axis=0).astype(np.float32)
    parents = clips_data[0]["parents"].astype(np.int64)
    clips = [0]
    for item in clips_data[:-1]:
        clips.append(clips[-1] + item["pose"].shape[0])

    poses_full = np.concatenate([displacements, pose], axis=-1)
    distances = np.linalg.norm(offsets[:, 1:], axis=-1).astype(np.float32)
    if no_imu:
        # --no-imu: 38-dim insole (pressure16 + force1 + cop2 per foot), no
        # acc/gyro columns exist, so the world-frame conversion is skipped
        # entirely and the IMU-only dataset fields stay None.
        left_acc_local = None
        right_acc_local = None
        quat_lIMU = None
        quat_rIMU = None
    else:
        rest_axes = rest_foot_imu_axes(offsets[0])
        left_acc_local = insole[:, 16:19].copy()
        right_acc_local = insole[:, 41:44].copy()
        quat_lIMU = world_imu_quat(quats[:, LEFT_FOOT], rest_axes["left"])
        quat_rIMU = world_imu_quat(quats[:, RIGHT_FOOT], rest_axes["right"])
        left_acc = left_acc_local.copy()
        left_acc[:, 1] *= -1.0
        left_acc_world = quat_mul_vec(quat_lIMU, left_acc)
        left_acc_world[:, 1] -= 1.0
        right_acc_world = quat_mul_vec(quat_rIMU, right_acc_local)
        right_acc_world[:, 1] -= 1.0
        insole[:, 16:19] = left_acc_world
        insole[:, 41:44] = right_acc_world

    identity = np.broadcast_to(np.eye(3), (1, N_JOINTS, 3, 3)).copy()
    rest_pos, _ = fk_local(identity, offsets[0].astype(np.float64), JOINT_PARENTS)

    def compute_axes(toes_index: int, foot_index: int, opposite_foot_index: int, invert_up: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right = rest_pos[0, toes_index] - rest_pos[0, foot_index]
        right = right / np.linalg.norm(right)
        up = rest_pos[0, opposite_foot_index] - rest_pos[0, foot_index]
        if invert_up:
            up = -up
        up = up / np.linalg.norm(up)
        forward = np.cross(right, up)
        forward = forward / np.linalg.norm(forward)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)

    right_lfoot_local, up_lfoot_local, forward_lfoot_local = compute_axes(LEFT_TOE, LEFT_FOOT, RIGHT_FOOT, True)
    right_rfoot_local, up_rfoot_local, forward_rfoot_local = compute_axes(RIGHT_TOE, RIGHT_FOOT, LEFT_FOOT, False)

    poses_list, insole_list, quats_list = [], [], []
    left_acc_list, right_acc_list, quat_l_list, quat_r_list = [], [], [], []
    frame_ids_list, valid_list = [], []
    clips_out = clips + [len(poses_full)]
    n_poses = 0
    for clip_idx in range(len(clips_out) - 1):
        start_idx = clips_out[clip_idx]
        end_idx = clips_out[clip_idx + 1]
        poses_list.append(torch.from_numpy(poses_full[start_idx + 1 : end_idx]).float())
        insole_list.append(torch.from_numpy(insole[start_idx + 1 : end_idx]).float())
        quats_list.append(torch.from_numpy(quats[start_idx + 1 : end_idx]).float())
        frame_ids_list.append(np.asarray(clips_data[clip_idx]["frame_ids"][1:], dtype=np.int64))
        valid_list.append(np.asarray(clips_data[clip_idx]["valid"][1:], dtype=np.uint8))
        if not no_imu:
            left_acc_list.append(torch.from_numpy(left_acc_local[start_idx + 1 : end_idx]).float())
            right_acc_list.append(torch.from_numpy(right_acc_local[start_idx + 1 : end_idx]).float())
            quat_l_list.append(torch.from_numpy(quat_lIMU[start_idx + 1 : end_idx]).float())
            quat_r_list.append(torch.from_numpy(quat_rIMU[start_idx + 1 : end_idx]).float())
        n_poses += poses_list[-1].shape[0]
        clips_out[clip_idx] -= clip_idx
    clips_out[-1] = n_poses

    dataset = MotionDatasetState()
    dataset.temporality = 1
    dataset.stride = 1
    dataset.poses = poses_list
    dataset.insole = insole_list
    dataset.parents = torch.from_numpy(parents).long()
    dataset.offsets = torch.from_numpy(offsets).float()
    dataset.quats = quats_list
    dataset.is_acceleration_world = True
    dataset.quat_lIMU = quat_l_list if not no_imu else None
    dataset.quat_rIMU = quat_r_list if not no_imu else None
    dataset.initial_global_rot = torch.from_numpy(global_rots[0]).float()
    dataset.sample_rate = TARGET_HZ
    dataset.delta_time = 1.0 / TARGET_HZ
    dataset.distances = torch.from_numpy(distances).float()
    # --no-imu: 38-dim layout (pressure16 + force1 + cop2 per foot). The
    # acc/gyro index fields are None, so consumers (data_augmentation,
    # cs_to_json) can detect the missing channels instead of mis-slicing.
    dataset.has_imu = not no_imu
    if no_imu:
        dataset.l_pressure_idx = (0, 16)
        dataset.l_acceleration_idx = None
        dataset.l_angular_velocity_idx = None
        dataset.l_total_force_idx = (16, 17)
        dataset.l_center_of_pressure_idx = (17, 19)
        dataset.r_pressure_idx = (19, 35)
        dataset.r_acceleration_idx = None
        dataset.r_angular_velocity_idx = None
        dataset.r_total_force_idx = (35, 36)
        dataset.r_center_of_pressure_idx = (36, 38)
    else:
        dataset.l_pressure_idx = (0, 16)
        dataset.l_acceleration_idx = (16, 19)
        dataset.l_angular_velocity_idx = (19, 22)
        dataset.l_total_force_idx = (22, 23)
        dataset.l_center_of_pressure_idx = (23, 25)
        dataset.r_pressure_idx = (25, 41)
        dataset.r_acceleration_idx = (41, 44)
        dataset.r_angular_velocity_idx = (44, 47)
        dataset.r_total_force_idx = (47, 48)
        dataset.r_center_of_pressure_idx = (48, 50)
    dataset.rfoot_index = FOOT_INDICES[0]
    dataset.rtoes_index = FOOT_INDICES[1]
    dataset.lfoot_index = FOOT_INDICES[2]
    dataset.ltoes_index = FOOT_INDICES[3]
    dataset.right_lfoot_local = torch.from_numpy(right_lfoot_local).float()
    dataset.up_lfoot_local = torch.from_numpy(up_lfoot_local).float()
    dataset.forward_lfoot_local = torch.from_numpy(forward_lfoot_local).float()
    dataset.right_rfoot_local = torch.from_numpy(right_rfoot_local).float()
    dataset.up_rfoot_local = torch.from_numpy(up_rfoot_local).float()
    dataset.forward_rfoot_local = torch.from_numpy(forward_rfoot_local).float()
    dataset.left_acceleration_local = left_acc_list if not no_imu else None
    dataset.right_acceleration_local = right_acc_list if not no_imu else None
    dataset.n_poses = n_poses
    dataset.clips = clips_out
    dataset.session_ids = [str(item.get("session_id", f"clip{i}")) for i, item in enumerate(clips_data)]
    dataset.clip_indices = [int(item.get("clip_index", 0)) for item in clips_data]
    dataset.n_session_clips = [int(item.get("n_session_clips", 1)) for item in clips_data]
    dataset.joint_names = list(clips_data[0].get("joint_names", JOINT_NAMES))
    dataset.raw_bvh_paths = [str(item["raw_bvh_path"]) for item in clips_data]
    dataset.raw_start_times = [float(item["raw_start_time"]) for item in clips_data]
    dataset.raw_end_times = [float(item["raw_end_time"]) for item in clips_data]
    dataset.clip_frame_ids = frame_ids_list
    dataset.clip_valid = valid_list
    return dataset


def _prefer_env_site() -> None:
    """Drop ``~/.local`` user site-packages so the conda-env pymotion wins.

    The ``~/.local`` pymotion copy uses ``tuple[T, T]`` annotations that crash
    Python 3.8 imports of ``pymotion.ops.skeleton_torch``.
    """
    kept = []
    for path in sys.path:
        normalized = path.replace("\\", "/")
        if "/.local/lib/python" in normalized and normalized.rstrip("/").endswith("site-packages"):
            continue
        kept.append(path)
    sys.path[:] = kept
    for name in list(sys.modules):
        if name == "pymotion" or name.startswith("pymotion."):
            del sys.modules[name]


def _dataset_shim() -> object:
    """Class-identity stand-in for the pristine ``dataset.MotionDataset``.

    The native .pt contract is a pickled ``MotionDataset`` instance whose
    pickle records the class reference ``dataset.MotionDataset``. The pristine
    ``dataset.py`` cannot be imported on Python 3.8 (``list[int]`` annotations
    without ``from __future__ import annotations``), so the producer builds the
    instance on a synthetic module named ``dataset``; the consumer unpickles it
    against the real class and restores ``__dict__`` through ``__new__``.
    """
    import types

    import torch

    mod = sys.modules.get("dataset")
    if mod is None:
        mod = types.ModuleType("dataset")

        class MotionDataset:
            """Identity stand-in; behaviour comes from the real class at unpickle time."""

            @staticmethod
            def load(data_path: str, device: torch.device):
                return torch.load(data_path).to(device)

            def to(self, device: torch.device):
                # Mirrors the real MotionDataset.to: move tensor attributes
                # (also inside lists/tuples) so the producer-side load works.
                for key, value in list(self.__dict__.items()):
                    if isinstance(value, torch.Tensor):
                        setattr(self, key, value.to(device))
                    elif isinstance(value, (list, tuple)):
                        moved = type(value)(
                            item.to(device) if isinstance(item, torch.Tensor) else item
                            for item in value
                        )
                        setattr(self, key, moved)
                return self

        MotionDataset.__module__ = "dataset"
        MotionDataset.__qualname__ = "MotionDataset"
        mod.MotionDataset = MotionDataset
        sys.modules["dataset"] = mod
    return mod.MotionDataset


def create_dataset(clips_data: list[dict], out_path: Path, dry_run: bool, no_imu: bool = False):
    if not clips_data:
        return None
    if dry_run:
        return clips_data
    import torch

    state = build_motion_dataset_state(clips_data, no_imu=no_imu)
    # Native serialization: a pickled MotionDataset instance (upstream
    # torch.save(dataset) contract). __init__ must not run here — the
    # displacements are already folded into poses, and __init__ would
    # concatenate them a second time.
    shim = _dataset_shim()
    dataset = shim.__new__(shim)
    dataset.__dict__.update(state.__dict__)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    return dataset


def run_self_test() -> None:
    values = np.zeros((1, 48), dtype=np.float32)
    grid = values.reshape(1, 4, 12)
    grid[0, 0, 0] = 10.0  # toe
    grid[0, 3, 11] = 20.0  # heel
    pooled = pool_48_to_16(grid.reshape(1, 48))
    assert pooled.shape == (1, 16)
    assert pooled[0, 8] > 0  # first toe channel
    assert pooled[0, 7] > 0  # last heel channel
    assert pooled[0, :8].sum() > pooled[0, 8:].sum()

    zeros = pool_48_to_16(np.zeros((2, 48), dtype=np.float32))
    assert np.all(zeros == 0)
    cop = cop_from_grid(np.zeros((2, 48), dtype=np.float32))
    assert np.all(cop == 0)

    heel_only = np.zeros((1, 4, 12), dtype=np.float32)
    heel_only[0, 1, 11] = 8.0
    cop_h = cop_from_grid(heel_only.reshape(1, 48))
    assert cop_h[0, 1] < 0.1
    toe_only = np.zeros((1, 4, 12), dtype=np.float32)
    toe_only[0, 1, 0] = 8.0
    cop_t = cop_from_grid(toe_only.reshape(1, 48))
    assert cop_t[0, 1] > 0.9

    mask = np.zeros(20, dtype=bool)
    mask[5:8] = True
    assert split_fake_clips(mask, 4) == [(0, 5), (8, 20)]
    assert split_fake_clips(mask, 6) == [(8, 20)]

    dt = 1.0 / TARGET_HZ
    t = np.arange(20, dtype=np.float64) * dt
    pos = np.stack([0.5 * 2.0 * t * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    rot = np.broadcast_to(np.eye(3), (20, 3, 3)).copy()
    rest = np.eye(3)
    acc, gyro = synthesize_imu(pos, rot, rest, dt, False)
    assert abs(float(acc[10, 0]) - (2.0 / G)) < 0.05
    assert np.allclose(acc[10, 1], 1.0, atol=0.05)
    assert np.allclose(gyro[5:], 0.0, atol=1e-5)

    acc_flip, _ = synthesize_imu(pos, rot, rest, dt, True)
    assert np.allclose(acc_flip[10, 1], -1.0, atol=0.05)

    src_offsets_cm = np.zeros((23, 3), dtype=np.float64)
    src_offsets_cm[1] = [0.0, 8.0, 0.0]  # Spine
    src_offsets_cm[2] = [0.0, 7.0, 0.0]  # Spine1
    src_offsets_cm[3] = [0.0, 7.0, 0.0]  # Spine2
    src_offsets_cm[4] = [0.0, 7.0, 0.0]  # Spine3
    identity = np.broadcast_to(np.eye(3), (2, 23, 3, 3)).copy()
    dst_offsets, dst_local = remap_skeleton(src_offsets_cm, identity)
    # Offsets are local BVH bone lengths; 7 cm is 0.07 m, not the
    # accumulated Spine->Spine3 chain length.
    assert np.allclose(dst_offsets[JOINT_NAMES.index("Spine3")], [0.0, 0.07, 0.0])
    assert np.allclose(dst_local, np.broadcast_to(np.eye(3), (2, N_JOINTS, 3, 3)))

    motion = np.zeros((2, 72), dtype=np.float64)
    motion[1, 3] = 90.0
    sampled = interp_motion(motion, 0.1, np.array([0.05]))
    assert abs(sampled[0, 3] - 45.0) < 1e-5

    print("self-test ok")


# ---------------------------------------------------------------------------
# Normalizer (gait native contract, built from the canonical train .pt only).
# ---------------------------------------------------------------------------

def build_normalizer(dataset_path: Path, out_path: Path, name: str, *, dry_run: bool = False) -> dict:
    """Mean/std normalizer from one dataset, pickled as ``normalizer.Normalizer``.

    The class identity must be the module object named ``normalizer`` (the
    upstream training/test code unpickles it from the Step2Motion src dir).
    The pristine ``normalizer.Normalizer.__init__`` has the upstream NameError
    (``prior_db``/``insole_db`` undefined, registry Step2Motion #6), so the
    same stats are computed here and attached to a bare pristine-class
    instance — the pickled class identity stays ``normalizer.Normalizer``.
    """
    _prefer_env_site()
    _dataset_shim()
    if not str(STEP2MOTION_SRC) in sys.path:
        sys.path.insert(0, str(STEP2MOTION_SRC))
    import importlib

    normalizer_mod = importlib.import_module("normalizer")
    import torch

    db = normalizer_mod.MotionDataset.load(str(dataset_path), torch.device("cpu"))

    def compute_mean_std(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = data.mean(dim=0).float()
        std = data.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std).float()
        return mean, std

    normalizer = normalizer_mod.Normalizer.__new__(normalizer_mod.Normalizer)
    normalizer.poses_mean, normalizer.poses_std = compute_mean_std(torch.cat(db.poses, dim=0))
    normalizer.distances_mean, normalizer.distances_std = compute_mean_std(db.distances)
    normalizer.insoles_mean, normalizer.insoles_std = compute_mean_std(torch.cat(db.insole, dim=0))
    normalizer.id = hash(
        normalizer.poses_mean.mean().item()
        + normalizer.poses_std.mean().item()
        + normalizer.distances_mean.mean().item()
        + normalizer.distances_std.mean().item()
        + normalizer.insoles_mean.mean().item()
        + normalizer.insoles_std.mean().item()
    )
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(normalizer, out_path)
    return {
        "name": name,
        "source": canonical_uri(dataset_path),
        "source_sha256": sha256_file(dataset_path),
        "output": canonical_uri(out_path),
        "output_sha256": sha256_file(out_path) if out_path.is_file() else "",
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def split_of(session_id: str, splits: dict[str, list[str]]) -> str:
    for name, ids in splits.items():
        if session_id in ids:
            return name
    raise KeyError(session_id)


def write_split_sessions(out_dir: Path, *, splits: dict[str, list[str]], details: dict,
                         skipped: dict, split_csv: Path, manifest: Path, variant: str) -> None:
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "variant": variant,
        "split_csv": str(split_csv),
        "split_sha256": sha256_file(split_csv),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "counts": {name: len(ids) for name, ids in splits.items()},
        "sessions": splits,
        "sessions_detail": {
            sid: {
                "split": info["split"],
                "n_clips": info["n_clips"],
                "n_frames": info["n_frames"],
                "clips": [
                    {"start_frame_id": int(c[0]), "end_frame_id": int(c[1]),
                     "n_frames": int(c[1] - c[0] + 1)}
                    for c in info["clip_ranges"]
                ],
            }
            for sid, info in sorted(details.items())
        },
        "empty_or_skipped": dict(sorted(skipped.items())),
        "n_clips": {name: sum(details[sid]["n_clips"] for sid in ids if sid in details)
                    for name, ids in splits.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_sessions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_clips(session_ids: list[str], splits: dict[str, list[str]], min_frames: int,
                  no_imu: bool, cache: dict, sessions_cache: dict,
                  errors: dict, details: dict) -> list[dict]:
    converted: list[dict] = []
    for sid in session_ids:
        if sid in errors:
            continue
        key = (no_imu, sid)
        if key not in cache:
            try:
                session = sessions_cache.setdefault(sid, load_shared_session(sid))
                cache[key] = convert_session(session, min_frames, no_imu=no_imu)
            except Exception as exc:  # noqa: BLE001
                errors[sid] = f"{type(exc).__name__}: {exc}"
                print(f"skip {sid}: {errors[sid]}")
                continue
        clips = cache[key]
        if not clips:
            session = sessions_cache[sid]
            fake_all = bool(np.asarray(session["frames"]["fake"], dtype=bool).all())
            errors[sid] = "all_frames_fake" if fake_all else f"too_short_or_fake(min_frames={min_frames})"
            continue
        converted.extend(clips)
        # Kept input frames are clip rows after the native first-frame
        # discard: frame_ids[1:] (and valid[1:]) is what the .pt stores.
        details[sid] = {
            "split": split_of(sid, splits),
            "n_clips": len(clips),
            "n_frames": sum(int(len(c["frame_ids"][1:])) for c in clips),
            "clip_ranges": [(int(c["frame_ids"][1]), int(c["frame_ids"][-1])) for c in clips],
        }
        print(f"{details[sid]['split']} {sid}: {len(clips)} clip(s), kept_frames={sum(int(c['pose'].shape[0]) - 1 for c in clips)}")
    return converted


def build_one(split_name: str, session_ids: list[str], splits: dict[str, list[str]], *,
              variant: str, min_frames: int, dry_run: bool,
              cache: dict, sessions_cache: dict, errors: dict, details: dict) -> tuple[Path, dict]:
    out_dir = INPUT_ROOT / variant
    out_path = out_dir / f"gait_{split_name}.pt"
    clips = collect_clips(session_ids, splits, min_frames, variant == "gait_noimu",
                          cache, sessions_cache, errors, details)
    n_clips = len(clips)
    n_frames = sum(int(c["pose"].shape[0]) for c in clips)
    insole_dim = int(clips[0]["insole"].shape[-1]) if clips else 0
    print(f"{variant}/{split_name}: clips={n_clips} frames={n_frames} insole={insole_dim}")
    dataset = create_dataset(clips, out_path, dry_run, no_imu=variant == "gait_noimu")
    if dataset is None:
        print(f"{variant}/{split_name}: empty")
    elif not dry_run:
        print(f"wrote {out_path}")
    return out_path, details


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    splits = read_split(SPLIT_CSV)
    if args.sessions:
        wanted = {value.strip() for value in args.sessions.split(",") if value.strip()}
        unknown = wanted - set().union(*map(set, splits.values()))
        if unknown:
            raise SystemExit(f"sessions not in canonical split: {sorted(unknown)}")
        splits = {name: [sid for sid in ids if sid in wanted] for name, ids in splits.items()}

    variants = []
    if not args.no_imu:
        variants.append("gait")
    if not args.imu_only:
        variants.append("gait_noimu")

    cache: dict = {}
    sessions_cache: dict = {}
    errors: dict = {}
    details: dict = {}
    normalizers: dict = {}
    for variant in variants:
        for split_name in ("train", "val", "test"):
            out_path, _ = build_one(split_name, splits[split_name], splits,
                                    variant=variant, min_frames=args.min_frames,
                                    dry_run=args.dry_run,
                                    cache=cache, sessions_cache=sessions_cache,
                                    errors=errors, details=details)
            if split_name == "train" and not args.dry_run and out_path.is_file():
                name = "gait_noimu" if variant == "gait_noimu" else "gait"
                normalizers[variant] = build_normalizer(
                    out_path, INPUT_ROOT / variant / f"normalizer_{name}.pth",
                    name, dry_run=False,
                )
        if not args.dry_run:
            write_split_sessions(INPUT_ROOT / variant, splits=splits, details=details,
                                 skipped=errors, split_csv=SPLIT_CSV, manifest=MANIFEST,
                                 variant=variant)

    if not args.dry_run:
        write_artifact(
            INPUT_ROOT,
            schema_version="model_input.step2motion.v1",
            producer="AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py",
            repository_root=ROOT,
            parameters={
                "adapter_version": ADAPTER_VERSION,
                "variants": variants,
                "min_frames": args.min_frames,
                "target_hz": TARGET_HZ,
                "input_dims": {"gait": 50, "gait_noimu": 38},
                "output_dim": 69,
                "skeleton": "BVH-23 Skeleton3 (Y-X-Z euler, root translation)",
                "tactile_source_chain": (
                    "raw pressure_left/right.csv (48/foot) -> shared/facts "
                    "pressure_48.npz (2x4x12) -> Step2Motion pool_48_to_16 "
                    "(D_Test4 accepted) -> left16/right16 (heel[0:8]+toe[8:16]) "
                    "-> Step2Motion pressure/force/CoP/(IMU) native input"
                ),
                "time_axis": "shared frames.npz mocap_time_s (no visual/pressure offset constructed)",
                "clip_policy": "fake frames never enter clips; every kept input frame carries its shared frame_id (clip_frame_ids)",
                "pooling": "pool_48_to_16 unchanged from the D_Test4 accepted implementation",
                "normalizers": normalizers,
                "normalizer_policy": "built from the canonical train .pt only",
                "pressure_gain": "per-session value ranges/dates in pressure_ranges.csv; no cross-session calibration applied",
            },
            source_artifacts=[canonical_uri(SPLIT_CSV), canonical_uri(MANIFEST), canonical_uri(FACTS_ROOT)],
            source_hashes={"split_csv": sha256_file(SPLIT_CSV), "manifest": sha256_file(MANIFEST)},
            consumers=["Baselines/Step2Motion"],
        )
        write_pressure_ranges(splits, errors)

    print("empty_or_skipped:", dict(sorted(errors.items())))
    unexpected = {sid: reason for sid, reason in errors.items() if reason != "all_frames_fake"}
    if unexpected:
        print(f"ERROR: {len(unexpected)} unexpected skips: {unexpected}", file=sys.stderr)
        return 1
    return 0


def write_pressure_ranges(splits: dict[str, list[str]], errors: dict) -> None:
    rows = []
    for name in ("train", "val", "test"):
        for sid in splits[name]:
            if sid in errors:
                continue
            session = load_shared_session(sid)
            pressure = session["pressure"]
            rows.append({
                "session_id": sid,
                "date": session["meta"].get("date", ""),
                "split": name,
                "value_min": float(pressure["value_min"]),
                "value_max": float(pressure["value_max"]),
                "unit": str(pressure["unit"]),
                "source_left": str(pressure["source_file_left"]),
                "source_right": str(pressure["source_file_right"]),
            })
    path = INPUT_ROOT / "pressure_ranges.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
