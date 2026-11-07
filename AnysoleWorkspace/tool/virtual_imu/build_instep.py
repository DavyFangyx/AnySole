#!/usr/bin/env python3
"""Virtual instep IMU facts producer.

Implements ``虚拟IMU脚面传感器生成方案_20261102.md`` (facts-level producer,
recommended defaults): a per-frame virtual IMU mounted on the dorsal midfoot
surface, driven by the shared facts time axis and the session SMPL fit.

Inputs (read-only, the linked AnysoleWorkspace):
    AnysoleWorkspace/shared/facts/sessions/<cam>/<date>/<subject>/<sid>/frames.npz
    AnysoleWorkspace/shared/facts/sessions/<cam>/<date>/<subject>/<sid>/session.json
    the session ``source_files.smpl`` npz (poses / joints / betas)
    AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl

Output (AnysoleWorkspace shared facts):
    AnysoleWorkspace/shared/facts/virtual_imu/<date>/<sid>/imu_instep.npz        # 原始测量契约
    AnysoleWorkspace/shared/facts/virtual_imu/<date>/<sid>/imu_instep_state.npz  # 生成状态/派生量/元数据
    AnysoleWorkspace/shared/facts/virtual_imu/<date>/<sid>/artifact.json

File split (2026-11-06): ``imu_instep.npz`` carries ONLY what a real sensor
outputs -- the raw per-frame measurements ``{left,right}_acc_g`` (g, sensor
frame) and ``{left,right}_gyro_rad_s`` (rad/s, sensor frame) -- plus the frame
alignment keys (``frame_id`` / ``mocap_time_s`` / ``target_hz``).  Everything
else (world acceleration, sensor quaternions, noiseless variants, instep world
position, rest axes, mount matrix, noise state, betas, rest joints) lives in
``imu_instep_state.npz`` so downstream consumers cannot mistake oracle state
for measurements.

Model (see the plan for the rationale):

* install point: per-session betas -> shaped mesh -> nearest dorsal midfoot
  vertex whose dominant skin weight is the ankle joint;
  ``offset_rest = vertex_rest - ankle_rest``.
* trajectory: ``p(t) = ankle_world(t) + R_ankle_world(t) @ offset_rest`` on the
  shared 40 Hz grid (positions linear, rotations SLERP from the 120 Hz fit);
  central second difference for acceleration, then +g.
* sensor frame: ``S(t) = R_ankle_world(t) @ rest_axes @ M`` with the legacy
  rest axes (x = toe direction, y = medial/lateral, z = dorsal) and a constant
  mount rotation ``M``.
* measurement: local frame white noise, turn-on bias, bias random walk, scale
  factor, axis misalignment, then 16-bit quantisation.  Every layer is
  individually switchable; the RNG seed is stable per (session, foot).

The left-foot y flip of the legacy Step2Motion channel convention is *not*
applied here: facts store the physical sensor-frame acceleration.  adapter_v3
applies the flip exactly like ``build_gait.synthesize_imu`` does.

Run (gait repo root, touch_gait env):

    python3 workspace/virtual_imu/build_instep.py
    python3 workspace/virtual_imu/build_instep.py --sessions S10103 --dry-run
    python3 workspace/virtual_imu/build_instep.py --no-noise
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

REPO_ROOT = Path(__file__).resolve().parents[3]   # .../gait (AnysoleWorkspace 所在仓库根)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import WORKSPACE as EXTERNAL_WORKSPACE  # noqa: E402
from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri  # noqa: E402

# Inputs: the linked AnysoleWorkspace (read-only, outside this project).
WORKSPACE = EXTERNAL_WORKSPACE
FACTS_ROOT = WORKSPACE / "shared/facts/sessions"
SPLIT_CSV = WORKSPACE / "protocol/splits/default/splits.csv"
SMPL_MODEL = WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl"
# Outputs: this project's own workspace only.
OUT_ROOT = WORKSPACE / "shared/facts/virtual_imu"

G = 9.81
TARGET_HZ = 40.0
PRODUCER = "tool/virtual_imu/build_instep.py"

# SMPL-24 npz joint order (same as Step2Motion / BVH24 export).
ANKLE_JOINT = {"left": 7, "right": 8}
TOE_JOINT = {"left": 10, "right": 11}
OPPOSITE_ANKLE = {"left": 8, "right": 7}
SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

# Plan section 4.1 install target.
INSTALL_FRAC = 0.6
INSTALL_HEIGHT_M = 0.04

# Plan section 4.3 noise defaults (all layers on, recommended).
NOISE_DEFAULTS = {
    "white_acc_g": 1e-3,                 # 1 mg
    "white_gyro_rad_s": 0.03 * np.pi / 180.0,  # 0.03 deg/s
    "bias_acc_g": 0.010,                 # +/- 10 mg
    "bias_gyro_rad_s": 1.0 * np.pi / 180.0,    # +/- 1 deg/s
    "walk_acc_g": 1e-6,                  # per-sample random-walk step
    "walk_gyro_rad_s": 1e-7,
    "scale_acc": 0.01,                   # +/- 1 %
    "scale_gyro": 0.02,                  # +/- 2 %
    "misalign_deg": 1.0,                 # per-axis, default 0.5-2 deg
    "mount_deg": 2.0,                    # constant installation rotation
    "acc_lsb_g": 32.0 / 65536.0,         # +/-16 g over 16 bit
    "gyro_lsb_rad_s": 4000.0 * np.pi / 180.0 / 65536.0,  # +/-2000 deg/s over 16 bit
}

ALL_LAYERS = ("white", "bias", "walk", "scale", "misalign", "mount", "quantize")


# ---------------------------------------------------------------------------
# Small numeric helpers (copied from build_gait.py; that file stays untouched).
# ---------------------------------------------------------------------------

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


def angular_velocity_from_rotmats(rotmats: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a rotation sequence (build_gait:270)."""
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
    return omega.astype(np.float64)


def central_acceleration(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central second difference, endpoint copy (build_gait.synthesize_imu)."""
    acc = np.zeros_like(pos)
    if len(pos) > 2:
        acc[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) / (dt * dt)
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc


# ---------------------------------------------------------------------------
# SMPL model / rest geometry.
# ---------------------------------------------------------------------------

def load_smpl_model(path: Path = SMPL_MODEL) -> dict:
    with path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    return {
        "v_template": np.asarray(model["v_template"], dtype=np.float64),
        "shapedirs": shapedirs,
        "J_regressor": model["J_regressor"],
        "weights": np.asarray(model["weights"], dtype=np.float64),
        "faces": np.asarray(model["f"], dtype=np.int64),
    }


def rest_geometry(model: dict, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest mesh and rest joints for one session's shape coefficients."""
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    if betas.size < 10:
        raise ValueError(f"expected >=10 betas, got {betas.size}")
    v_shaped = model["v_template"] + model["shapedirs"][..., :10] @ betas[:10]
    joints = np.asarray(model["J_regressor"].dot(v_shaped), dtype=np.float64)
    return v_shaped, joints


def rest_foot_axes(joints_rest: np.ndarray) -> dict[str, np.ndarray]:
    """Legacy Step2Motion rest axes (columns: toe dir, lateral, dorsal)."""

    def compute_axes(toes_index: int, foot_index: int, opposite_foot_index: int, invert_up: bool) -> np.ndarray:
        right = joints_rest[toes_index] - joints_rest[foot_index]
        right = right / np.linalg.norm(right)
        up = joints_rest[opposite_foot_index] - joints_rest[foot_index]
        if invert_up:
            up = -up
        up = up / np.linalg.norm(up)
        forward = np.cross(right, up)
        forward = forward / np.linalg.norm(forward)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        return np.stack([right, up, forward], axis=-1)

    return {
        "left": compute_axes(TOE_JOINT["left"], ANKLE_JOINT["left"], OPPOSITE_ANKLE["left"], True),
        "right": compute_axes(TOE_JOINT["right"], ANKLE_JOINT["right"], OPPOSITE_ANKLE["right"], False),
    }


def vertex_normals(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = v[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(v)
    for corner in range(3):
        np.add.at(out, faces[:, corner], normals)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def install_point(
    model: dict,
    betas: np.ndarray,
    *,
    frac: float = INSTALL_FRAC,
    height: float = INSTALL_HEIGHT_M,
) -> dict[str, dict]:
    """Nearest dorsal midfoot mesh vertex per foot.

    Restriction to vertices whose dominant skin weight is the ankle joint keeps
    the tracked point on the rigid ankle segment (plan section 3.1).  The
    dorsal side is the z column of the legacy rest axes.
    """
    v_shaped, joints = rest_geometry(model, betas)
    axes = rest_foot_axes(joints)
    weights = model["weights"]
    vn = vertex_normals(v_shaped, np.asarray(model["faces"], dtype=np.int64))
    dominant = weights.argmax(axis=1)

    out: dict[str, dict] = {}
    for side in ("left", "right"):
        ankle = ANKLE_JOINT[side]
        toe = TOE_JOINT[side]
        dorsal = axes[side][:, 2]
        target = joints[ankle] + frac * (joints[toe] - joints[ankle]) + height * dorsal
        base = dominant == ankle
        candidates = np.where(base & (vn @ dorsal > 0.4))[0]
        if candidates.size == 0:
            candidates = np.where(base & (vn @ dorsal > 0.0))[0]
        if candidates.size == 0:
            candidates = np.where(base)[0]
        if candidates.size == 0:
            raise RuntimeError(f"no mesh vertex bound to the {side} ankle joint")
        distances = np.linalg.norm(v_shaped[candidates] - target, axis=1)
        index = int(candidates[int(np.argmin(distances))])
        out[side] = {
            "offset_rest": (v_shaped[index] - joints[ankle]).astype(np.float64),
            "vertex_index": index,
            "vertex_rest": v_shaped[index].astype(np.float64),
            "target": target.astype(np.float64),
            "ankle_rest": joints[ankle].astype(np.float64),
            "rest_distance_m": float(np.linalg.norm(v_shaped[index] - joints[ankle])),
            "weight_ankle": float(weights[index, ankle]),
            "normal_dorsal_dot": float(vn[index] @ dorsal),
            "rest_axes": axes[side].astype(np.float64),
        }
    return out


# ---------------------------------------------------------------------------
# SMPL fit -> shared grid.
# ---------------------------------------------------------------------------

def load_smpl_fit(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = ("poses", "trans", "betas", "joints", "joints_rest", "source_frame_times_s")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing SMPL keys {missing}")
    poses = np.asarray(data["poses"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"{path}: poses must be (T,72), got {poses.shape}")
    times = np.asarray(data["source_frame_times_s"], dtype=np.float64)
    if times.ndim != 1 or len(times) != len(poses):
        raise ValueError(f"{path}: source_frame_times_s does not match poses")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError(f"{path}: source_frame_times_s must be strictly increasing")
    parents = np.asarray(data["joint_parents"], dtype=np.int64) if "joint_parents" in data.files else None
    return {
        "poses": poses,
        "betas": np.asarray(data["betas"], dtype=np.float64),
        "joints": np.asarray(data["joints"], dtype=np.float64),
        "joints_rest": np.asarray(data["joints_rest"], dtype=np.float64),
        "times": times,
        "parents": parents,
        "frame_rate": float(np.asarray(data["mocap_frame_rate"])) if "mocap_frame_rate" in data.files else float("nan"),
    }


def global_rotations(poses: np.ndarray, parents: np.ndarray = SMPL_PARENTS) -> np.ndarray:
    """SMPL-24 global rotations (rest-relative) from axis-angle poses."""
    n_frames = len(poses)
    local = SciRotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(n_frames, 24, 3, 3)
    out = np.empty_like(local)
    for joint in range(24):
        parent = int(parents[joint])
        out[:, joint] = local[:, joint] if parent < 0 else out[:, parent] @ local[:, joint]
    return out


def resample_to_grid(fit: dict, query_t: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Linear positions + SLERP rotations on the shared fact time axis."""
    times = fit["times"]
    clipped = np.clip(np.asarray(query_t, dtype=np.float64), times[0], times[-1])
    joints = np.stack(
        [np.interp(clipped, times, fit["joints"][:, j, a]) for j in range(24) for a in range(3)],
        axis=1,
    ).reshape(len(clipped), 24, 3)
    rotations = global_rotations(fit["poses"])
    foot_rot: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        joint = ANKLE_JOINT[side]
        if len(times) == 1:
            foot_rot[side] = np.broadcast_to(rotations[0, joint], (len(clipped), 3, 3)).copy()
        else:
            slerp = Slerp(times, SciRotation.from_matrix(rotations[:, joint]))
            foot_rot[side] = slerp(clipped).as_matrix()
    return joints, foot_rot


# ---------------------------------------------------------------------------
# Sensor model.
# ---------------------------------------------------------------------------

def stable_seed(session_id: str, foot: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{session_id}|{foot}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(salt)) & 0xFFFFFFFFFFFFFFFF


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_mount_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    if max_deg <= 0:
        return np.eye(3)
    rotvec = np.radians(rng.uniform(-max_deg, max_deg, 3))
    return SciRotation.from_rotvec(rotvec).as_matrix()


def sample_noise(
    true_local: np.ndarray,
    rng: np.random.Generator,
    *,
    kind: str,
    params: dict,
    layers: set[str],
) -> tuple[np.ndarray, dict]:
    """MEMS measurement model in the sensor frame; returns measured + state."""
    n = len(true_local)
    if kind == "acc":
        white = params["white_acc_g"]
        bias_max = params["bias_acc_g"]
        walk = params["walk_acc_g"]
        scale_max = params["scale_acc"]
        lsb = params["acc_lsb_g"]
    elif kind == "gyro":
        white = params["white_gyro_rad_s"]
        bias_max = params["bias_gyro_rad_s"]
        walk = params["walk_gyro_rad_s"]
        scale_max = params["scale_gyro"]
        lsb = params["gyro_lsb_rad_s"]
    else:
        raise ValueError(kind)

    measured = np.asarray(true_local, dtype=np.float64).copy()
    state: dict = {"lsb": float(lsb)}

    scale = np.zeros(3)
    if "scale" in layers:
        scale = rng.uniform(-scale_max, scale_max, 3)
        measured = measured * (1.0 + scale)
    state["scale"] = scale

    misalign = np.eye(3)
    if "misalign" in layers:
        rotvec = np.radians(rng.uniform(-params["misalign_deg"], params["misalign_deg"], 3))
        misalign = SciRotation.from_rotvec(rotvec).as_matrix()
        measured = measured @ misalign.T
    state["misalign"] = misalign

    bias = np.zeros(3)
    if "bias" in layers:
        bias = rng.uniform(-bias_max, bias_max, 3)
        measured = measured + bias
    state["bias"] = bias

    walk_values = np.zeros_like(measured)
    if "walk" in layers:
        walk_values = np.cumsum(rng.normal(0.0, walk, size=(n, 3)), axis=0)
        measured = measured + walk_values
    state["random_walk"] = walk_values

    if "white" in layers:
        measured = measured + rng.normal(0.0, white, size=(n, 3))

    if "quantize" in layers:
        measured = np.round(measured / lsb) * lsb
    return measured, state


def build_session(
    session: dict,
    model: dict,
    *,
    params: dict,
    layers: set[str],
    install_frac: float,
    install_height: float,
    seed_salt: int,
) -> tuple[dict[str, np.ndarray], dict]:
    sid = session["session_id"]
    frames = np.load(session["dir"] / "frames.npz", allow_pickle=False)
    frame_id = np.asarray(frames["frame_id"], dtype=np.int64)
    mocap_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    n = len(frame_id)
    if n == 0:
        raise ValueError(f"{sid}: empty fact timeline")
    if len(mocap_t) != n:
        raise ValueError(f"{sid}: mocap_time_s length mismatch")
    fps = float(session["meta"].get("fps", TARGET_HZ))
    if not np.isfinite(fps) or fps <= 0:
        fps = TARGET_HZ
    dt = 1.0 / fps

    fit_path = resolve_uri(session["meta"]["source_files"]["smpl"], must_exist=True)
    fit = load_smpl_fit(fit_path)
    if fit["parents"] is not None and not np.array_equal(fit["parents"], SMPL_PARENTS):
        raise ValueError(f"{sid}: unexpected SMPL joint parents {fit['parents']}")
    if np.abs(fit["joints_rest"] - rest_geometry(model, fit["betas"])[1]).max() > 1e-3:
        raise ValueError(f"{sid}: SMPL model rest joints do not match the fit npz")

    joints, foot_rot = resample_to_grid(fit, mocap_t)
    install = install_point(model, fit["betas"], frac=install_frac, height=install_height)

    raw: dict[str, np.ndarray] = {
        "frame_id": frame_id,
        "mocap_time_s": mocap_t.astype(np.float64),
        "target_hz": np.asarray(fps, dtype=np.float64),
    }
    state: dict[str, np.ndarray] = {}
    summary = {
        "session_id": sid,
        "subject": session["meta"].get("subject", ""),
        "date": session["meta"].get("date", ""),
        "frame_count": int(n),
        "source_fps": float(fit["frame_rate"]) if np.isfinite(fit["frame_rate"]) else None,
        "smpl": str(session["meta"]["source_files"]["smpl"]),
        "install": {},
        "feet": {},
    }

    for side in ("left", "right"):
        joint = ANKLE_JOINT[side]
        offset = install[side]["offset_rest"]
        p_instep = joints[:, joint] + np.einsum("nij,j->ni", foot_rot[side], offset)
        acc = central_acceleration(p_instep, dt)
        acc[..., 1] += G

        # Sensor frame: legacy rest axes + constant mount rotation.
        rng = make_rng(stable_seed(sid, side, seed_salt))
        mount = sample_mount_rotation(rng, params["mount_deg"]) if "mount" in layers else np.eye(3)
        sensor_rot = np.einsum("nij,jk,kl->nil", foot_rot[side], install[side]["rest_axes"], mount)
        sensor_quat = rotmat_to_quat_wxyz(sensor_rot)

        acc_local_true = np.einsum("nji,nj->ni", sensor_rot, acc) / G
        gyro_world = angular_velocity_from_rotmats(sensor_rot, dt)
        gyro_local_true = np.einsum("nji,nj->ni", sensor_rot, gyro_world)

        acc_meas, acc_state = sample_noise(acc_local_true, rng, kind="acc", params=params, layers=layers)
        gyro_meas, gyro_state = sample_noise(gyro_local_true, rng, kind="gyro", params=params, layers=layers)

        # World acceleration in the .pt contract: the *measured* local value
        # (noise, bias and quantisation included, already in g units) rotated
        # back to the world with gravity removed.  This is what
        # build_gait/build_motion_dataset_state stores for adapter_v2.
        acc_world_meas = np.einsum("nij,nj->ni", sensor_rot, acc_meas)
        acc_world_meas[..., 1] -= 1.0
        acc_world_true = np.einsum("nij,nj->ni", sensor_rot, acc_local_true)
        acc_world_true[..., 1] -= 1.0

        raw[f"{side}_acc_g"] = acc_meas.astype(np.float32)
        raw[f"{side}_gyro_rad_s"] = gyro_meas.astype(np.float32)
        state[f"{side}_acc_noiseless_g"] = acc_local_true.astype(np.float32)
        state[f"{side}_gyro_noiseless_rad_s"] = gyro_local_true.astype(np.float32)
        state[f"{side}_acc_world_g"] = acc_world_meas.astype(np.float32)
        state[f"{side}_acc_world_noiseless_g"] = acc_world_true.astype(np.float32)
        state[f"{side}_sensor_quat_wxyz"] = sensor_quat.astype(np.float32)
        state[f"{side}_sensor_rot"] = sensor_rot.astype(np.float32)
        state[f"{side}_instep_world"] = p_instep.astype(np.float32)
        state[f"{side}_offset_rest"] = offset.astype(np.float64)
        state[f"{side}_rest_axes"] = install[side]["rest_axes"].astype(np.float64)
        state[f"{side}_mount_matrix"] = mount.astype(np.float64)
        state[f"{side}_vertex_index"] = np.asarray(install[side]["vertex_index"], dtype=np.int64)
        state[f"{side}_seed"] = np.asarray(stable_seed(sid, side, seed_salt), dtype=np.uint64)
        for key, value in acc_state.items():
            state[f"{side}_acc_{key}"] = np.asarray(value)
        for key, value in gyro_state.items():
            state[f"{side}_gyro_{key}"] = np.asarray(value)

        summary["install"][side] = {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in install[side].items()
            if key in ("vertex_index", "rest_distance_m", "weight_ankle", "normal_dorsal_dot")
        }
        summary["install"][side]["offset_rest"] = install[side]["offset_rest"].tolist()
        summary["install"][side]["vertex_rest"] = install[side]["vertex_rest"].tolist()
        summary["feet"][side] = {
            "seed": int(stable_seed(sid, side, seed_salt)),
            "mount_rotvec_deg": (
                SciRotation.from_matrix(mount).as_rotvec() * 180.0 / np.pi
            ).tolist(),
            "acc_bias_g": acc_state["bias"].tolist(),
            "gyro_bias_rad_s": gyro_state["bias"].tolist(),
            "acc_scale": acc_state["scale"].tolist(),
            "gyro_scale": gyro_state["scale"].tolist(),
            "acc_misalign_deg": (
                SciRotation.from_matrix(acc_state["misalign"]).as_rotvec() * 180.0 / np.pi
            ).tolist(),
            "gyro_misalign_deg": (
                SciRotation.from_matrix(gyro_state["misalign"]).as_rotvec() * 180.0 / np.pi
            ).tolist(),
            "acc_max_abs_g": float(np.abs(acc_meas).max()),
            "gyro_max_abs_rad_s": float(np.abs(gyro_meas).max()),
        }

    state["betas"] = fit["betas"].astype(np.float64)
    state["joints_rest"] = fit["joints_rest"].astype(np.float64)
    state["noise_config_json"] = np.asarray(
        json.dumps(
            {
                "layers": sorted(layers),
                "params": {key: float(value) for key, value in params.items()},
                "install_frac": float(install_frac),
                "install_height_m": float(install_height),
                "seed_salt": int(seed_salt),
                "phase": "physical sensor local frame; left y flip NOT applied",
            },
            sort_keys=True,
        )
    )
    return raw, state, summary


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------

def read_split_sessions(path: Path = SPLIT_CSV) -> list[str]:
    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in groups:
                value = (row.get(key) or "").strip()
                if value:
                    groups[key].append(value)
    ids = groups["train"] + groups["val"] + groups["test"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate session ids in canonical split")
    return ids


def find_session_dir(session_id: str, camera: str) -> Path:
    matches = sorted(
        path.parent
        for path in (FACTS_ROOT / camera).glob(f"*/*/{session_id}/session.json")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"shared session {session_id!r} not found under {FACTS_ROOT / camera}")
    if len(matches) > 1:
        raise ValueError(f"session id is ambiguous: {session_id}: {matches}")
    return matches[0]


def load_session(session_id: str, camera: str) -> dict:
    session_dir = find_session_dir(session_id, camera)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if str(meta.get("session_id")) != str(session_id):
        raise ValueError(f"{session_dir}: session.json identity mismatch")
    if not meta.get("source_files", {}).get("smpl"):
        raise FileNotFoundError(f"{session_dir}: no source_files.smpl")
    return {"session_id": str(session_id), "dir": session_dir, "meta": meta}


def list_all_sessions(camera: str) -> list[str]:
    ids = sorted({path.parent.name for path in (FACTS_ROOT / camera).glob("*/*/*/session.json") if path.is_file()})
    return ids


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    # Rest axes match the Step2Motion construction on the SMPL rest skeleton.
    model = load_smpl_model()
    _, joints = rest_geometry(model, np.zeros(10))
    axes = rest_foot_axes(joints)
    assert np.allclose(axes["left"].T @ axes["left"], np.eye(3), atol=1e-9)
    assert np.allclose(axes["right"].T @ axes["right"], np.eye(3), atol=1e-9)

    # Install point is bound to the ankle joint, dorsal and ~10-15 cm from it.
    install = install_point(model, np.zeros(10))
    for side in ("left", "right"):
        info = install[side]
        assert 0.05 < info["rest_distance_m"] < 0.20, info
        assert info["weight_ankle"] > 0.3, info
        assert info["normal_dorsal_dot"] > 0.4, info

    # Rigid tracker: p = ankle + R offset under a known rotation.
    offset = install["left"]["offset_rest"]
    angles = np.linspace(0.0, 0.5, 8)
    rots = SciRotation.from_rotvec(np.stack([angles, np.zeros_like(angles), np.zeros_like(angles)], axis=1)).as_matrix()
    ankle = np.tile(install["left"]["ankle_rest"], (len(angles), 1))
    p = ankle + np.einsum("nij,j->ni", rots, offset)
    expected = ankle + SciRotation.from_rotvec(
        np.stack([angles, np.zeros_like(angles), np.zeros_like(angles)], axis=1)
    ).apply(offset[None, :])
    assert np.allclose(p, expected, atol=1e-12)

    # Static gravity check: a level sensor still reads 1 g and the dorsal
    # (z) axis carries the dominant share.
    acc_world = np.tile(np.array([0.0, G, 0.0]), (4, 1))
    rot = np.tile(axes["left"], (4, 1, 1))
    acc_local = np.einsum("nji,nj->ni", rot, acc_world) / G
    assert np.allclose(np.linalg.norm(acc_local, axis=1), 1.0, atol=1e-9), acc_local
    assert np.all(acc_local[:, 2] > 0.8), acc_local

    # Noise model: residual is exactly zero when all layers are off.
    rng = make_rng(0)
    true = np.zeros((20, 3))
    measured, _ = sample_noise(true, rng, kind="acc", params=NOISE_DEFAULTS, layers=set())
    assert np.allclose(measured, 0.0)

    # Quantisation: outputs are multiples of the LSB.
    measured, state = sample_noise(true, make_rng(1), kind="acc", params=NOISE_DEFAULTS,
                                   layers={"white", "bias", "walk", "scale", "misalign", "quantize"})
    lsb = state["lsb"]
    assert np.allclose(np.mod(measured / lsb, 1.0), 0.0, atol=1e-6)

    # Seeded RNG is stable across processes.
    assert stable_seed("S10103", "left") == stable_seed("S10103", "left")
    assert stable_seed("S10103", "left") != stable_seed("S10103", "right")

    print("self-test ok")


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build virtual instep IMU facts.")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs; default: canonical split")
    parser.add_argument("--all-sessions", action="store_true", help="process every session under --camera")
    parser.add_argument("--camera", default="cam3", help="shared facts camera (default cam3, the Step2Motion axis)")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--smpl-model", default=str(SMPL_MODEL))
    parser.add_argument("--noise-layers", default=",".join(ALL_LAYERS),
                        help=f"comma-separated subset of {','.join(ALL_LAYERS)}")
    parser.add_argument("--no-noise", action="store_true", help="disable every noise layer")
    parser.add_argument("--mount-deg", type=float, default=NOISE_DEFAULTS["mount_deg"])
    parser.add_argument("--install-frac", type=float, default=INSTALL_FRAC)
    parser.add_argument("--install-height", type=float, default=INSTALL_HEIGHT_M)
    parser.add_argument("--seed-salt", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="compute without writing files")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing sessions")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def session_ids_from_args(args: argparse.Namespace) -> list[str]:
    if args.sessions:
        ids = [value.strip() for value in args.sessions.split(",") if value.strip()]
        if args.all_sessions:
            raise SystemExit("--sessions and --all-sessions are mutually exclusive")
        return ids
    if args.all_sessions:
        return list_all_sessions(args.camera)
    return read_split_sessions()


def write_session_artifact(out_dir: Path, session: dict, summary: dict,
                           params: dict, layers: set[str], args: argparse.Namespace,
                           smpl_hash: str, frames_hash: str) -> None:
    write_artifact(
        out_dir,
        schema_version="shared.virtual_imu.v1",
        producer=PRODUCER,
        repository_root=PROJECT_ROOT,
        parameters={
            "camera": args.camera,
            "target_hz": TARGET_HZ,
            "noise_layers": sorted(layers),
            "noise_params": {key: float(value) for key, value in params.items()},
            "install_frac": float(args.install_frac),
            "install_height_m": float(args.install_height),
            "seed_salt": int(args.seed_salt),
            "seed_policy": "sha256('<session_id>|<foot>') ^ seed_salt",
            "left_y_flip": "not applied in facts; adapter_v3 applies the legacy flip",
            "install": summary["install"],
            "feet": summary["feet"],
            "smpl_model": str(Path(args.smpl_model).resolve()),
            "summary": summary,
        },
        source_artifacts=[
            canonical_uri(session["dir"]),
            str(session["meta"]["source_files"]["smpl"]),
        ],
        source_hashes={"smpl": smpl_hash, "frames": frames_hash},
        frame_id_min=0,
        frame_id_max=int(summary["frame_count"]) - 1,
        frame_count=int(summary["frame_count"]),
        consumers=["Baselines/Step2Motion (adapter_v3)"],
    )


def write_root_artifact(out_root: Path, *, args: argparse.Namespace, params: dict,
                        layers: set[str], count: int, skipped: dict) -> None:
    present = len(list(out_root.glob("*/*/imu_instep.npz")))
    write_artifact(
        out_root,
        schema_version="shared.virtual_imu.index.v1",
        producer=PRODUCER,
        repository_root=PROJECT_ROOT,
        parameters={
            "camera": args.camera,
            "target_hz": TARGET_HZ,
            "noise_layers": sorted(layers),
            "noise_params": {key: float(value) for key, value in params.items()},
            "install_frac": float(args.install_frac),
            "install_height_m": float(args.install_height),
            "seed_salt": int(args.seed_salt),
            "smpl_model": str(Path(args.smpl_model).resolve()),
            "sessions_present": int(present),
            "sessions_written_this_run": int(count),
            "sessions_skipped_this_run": dict(sorted(skipped.items())),
            "layout": "<date>/<session_id>/{imu_instep.npz,imu_instep_state.npz,artifact.json}",
        },
        source_artifacts=[canonical_uri(SPLIT_CSV)],
        consumers=["Baselines/Step2Motion (adapter_v3)"],
    )


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    if args.no_noise:
        layers: set[str] = set()
    else:
        layers = {value.strip() for value in args.noise_layers.split(",") if value.strip()}
        unknown = layers - set(ALL_LAYERS)
        if unknown:
            raise SystemExit(f"unknown noise layers: {sorted(unknown)}")
    params = dict(NOISE_DEFAULTS)
    params["mount_deg"] = float(args.mount_deg)

    model = load_smpl_model(Path(args.smpl_model))

    out_root = Path(args.out_root)
    session_ids = session_ids_from_args(args)
    errors: dict[str, str] = {}
    written = 0
    smpl_hashes: dict[str, str] = {}

    for sid in session_ids:
        try:
            session = load_session(sid, args.camera)
            date = str(session["meta"].get("date", ""))
            if not date:
                raise ValueError("session.json has no date")
            out_dir = out_root / date / sid
            npz_path = out_dir / "imu_instep.npz"
            if npz_path.is_file() and not args.overwrite:
                print(f"skip {sid}: exists (use --overwrite)")
                continue
            raw, state_data, summary = build_session(
                session, model,
                params=params,
                layers=layers,
                install_frac=float(args.install_frac),
                install_height=float(args.install_height),
                seed_salt=int(args.seed_salt),
            )
            if args.dry_run:
                print(f"dry-run {sid}: frames={summary['frame_count']} "
                      f"left_offset={np.round(state_data['left_offset_rest'], 4)}")
                written += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            state_path = out_dir / "imu_instep_state.npz"
            np.savez_compressed(state_path, **state_data)
            np.savez_compressed(npz_path, **raw)
            smpl_path = resolve_uri(session["meta"]["source_files"]["smpl"], must_exist=True)
            smpl_hash = smpl_hashes.setdefault(str(smpl_path), sha256_file(smpl_path))
            frames_path = session["dir"] / "frames.npz"
            frames_hash = sha256_file(frames_path)
            write_session_artifact(out_dir, session, summary, params, layers,
                                   args, smpl_hash, frames_hash)
            written += 1
            print(f"wrote {npz_path} + imu_instep_state.npz frames={summary['frame_count']} "
                  f"L={np.round(state_data['left_offset_rest'], 4)} R={np.round(state_data['right_offset_rest'], 4)}")
        except Exception as exc:  # noqa: BLE001
            errors[sid] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR {sid}: {errors[sid]}", file=sys.stderr)

    if not args.dry_run:
        write_root_artifact(out_root, args=args, params=params, layers=layers,
                            count=written, skipped=errors)
    print(f"done: {written} session(s), {len(errors)} error(s)")
    if errors:
        print(json.dumps(errors, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
