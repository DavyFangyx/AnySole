"""E6.6a: Step2Motion-口径触觉通道（T_s2m，50 维/帧）。

Ports from ``Baselines/Step2Motion/src/process_gait.py`` (pool/cop/IMU synthesis)
and ``src/dataset.py`` (world-frame conversion + gravity removal), copied verbatim
in structure so the channels match the Step2Motion gait export:

    每脚 25 维 = pressure16（pool_48_to_16：heel[8] toes[8]）+ acc3 + gyro3 + force1 + cop2

IMU 通道只由 **直接导出的 GT BVH-23 脚部运动学**合成（脚部世界位置
二阶差分 + 脚部全局旋转差分）。这里冻结 Step2Motion ``process_gait.py`` 的
23 关节/ToeBase 语义，不使用也不接受 AnySole 的 SMPL-24 target。它含 GT BVH
派生信息，只是历史实验输入；真实部署时应由鞋垫硬件 IMU 替代。

两段式管线（与 Step2Motion 同构，此处用 rotmat 等价实现其 quat 往返）：
1. 合成（synthesize_imu）：a_world = 位置二阶差分，+g 后转到局部系再 /G（模拟真实
   IMU 读数，左脚 Y 翻转）；gyro = 脚部全局旋转差分的局部系角速度。
2. 世界系转换（S2M dataset.py:136-148）：左脚 Y 翻回、转回世界系、Y 减 1（去重力）。
   旋转往返严格抵消，最终 acc 通道 = a_world/G（g 单位）；gyro 保持局部系（rad/s）。

尺度约定：pressure16/force/cop 在本实现中基于 anysole 的**归一化**压力
（clip/PRESSURE_CLIP 后除以 PRESSURE_CLIP），而非 S2M 的原始值+z-score——结构同口径，
尺度由编码器的 LayerNorm 吸收（anysole 编码器自带 LayerNorm，S2M 依赖 Normalizer）。

--no-imu：``build_t_s2m(..., no_imu=True)`` 输出 38 维/帧（每脚 pressure16 +
force1 + cop2），acc/gyro 通道被真删：不调用 synthesize_imu、不产生任何 IMU
数值，编码器侧对应分组也一并删除（见 tactile_encoder.py）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from anysole.geometry import fk_local_np
from anysole.types import (
    FPS,
    T_S2M_DIM,
    T_S2M_NOIMU_DIM,
    WORKSPACE_ROOT,
)

G = 9.81

# --no-imu：T_s2m 里被真删的 IMU 列（左脚 acc/gyro、右脚 acc/gyro），
# 删除后每脚 19 维 = pressure16 + force1 + cop2，共 38 维。
S2M_IMU_COLUMNS = tuple(range(16, 22)) + tuple(range(41, 47))

# Frozen verbatim from Baselines/Step2Motion/src/process_gait.py.  These are
# deliberately local: importing AnySole's SMPL-24 JOINT_* constants here would
# silently change the meaning of the historical synthetic IMU channel.
LEGACY_BVH_JOINT_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)
LEGACY_BVH_PARENTS = np.asarray(
    [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21],
    dtype=np.int64,
)
LEGACY_BVH_N_JOINTS = len(LEGACY_BVH_JOINT_NAMES)
LEGACY_LEFT_FOOT = LEGACY_BVH_JOINT_NAMES.index("LeftFoot")
LEGACY_LEFT_TOE = LEGACY_BVH_JOINT_NAMES.index("LeftToeBase")
LEGACY_RIGHT_FOOT = LEGACY_BVH_JOINT_NAMES.index("RightFoot")
LEGACY_RIGHT_TOE = LEGACY_BVH_JOINT_NAMES.index("RightToeBase")


def resolve_legacy_bvh_path(meta: dict) -> Path:
    """Resolve the directly exported BVH using Step2Motion's original rule."""
    recorded = Path(meta["bvh_path"])
    if recorded.is_file():
        return recorded
    session_id = str(meta["session_id"])
    date = str(meta["date"])
    root = WORKSPACE_ROOT / "sources" / "raw" / date / "mocap_ori_bvh" / session_id
    candidates = sorted(root.glob("*.bvh"))
    if not candidates:
        raise FileNotFoundError(
            "BVH not found for %s: recorded=%s, workspace raw=%s"
            % (session_id, recorded, root)
        )
    return candidates[0]


def _parse_legacy_bvh(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """The BVH parser used by Step2Motion's process_gait.py."""
    text = Path(path).read_text(errors="replace").splitlines()
    offsets = []
    names = []
    joint_i = -1
    for line in text:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            names.append(stripped.split()[1])
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
    if motion.shape != (n_frames, 72):
        raise ValueError("Unexpected BVH motion shape %s in %s" % (motion.shape, path))
    if tuple(names) != LEGACY_BVH_JOINT_NAMES:
        raise ValueError("Unexpected BVH joint names in %s: %s" % (path, names))
    if len(offsets) != LEGACY_BVH_N_JOINTS:
        raise ValueError("Unexpected BVH joint count %d in %s" % (len(offsets), path))
    return motion, frame_time, np.asarray(offsets, dtype=np.float64)


def _interp_legacy_bvh(motion: np.ndarray, frame_time: float, query_t: np.ndarray) -> np.ndarray:
    """Step2Motion translation interpolation + per-joint YXZ rotation SLERP."""
    query_t = np.asarray(query_t, dtype=np.float64)
    src_t = np.arange(motion.shape[0], dtype=np.float64) * float(frame_time)
    out = np.empty((query_t.size, motion.shape[1]), dtype=np.float64)
    for col in range(3):
        out[:, col] = np.interp(query_t, src_t, motion[:, col])
    clipped = np.clip(query_t, src_t[0], src_t[-1])
    eulers = motion[:, 3:].reshape(motion.shape[0], LEGACY_BVH_N_JOINTS, 3)
    if motion.shape[0] == 1:
        out[:, 3:] = motion[0, 3:]
        return out
    for joint_i in range(LEGACY_BVH_N_JOINTS):
        key_rots = SciRotation.from_euler("YXZ", eulers[:, joint_i], degrees=True)
        slerp = Slerp(src_t, key_rots)
        out[:, 3 + 3 * joint_i : 6 + 3 * joint_i] = slerp(clipped).as_euler("YXZ", degrees=True)
    return out


def _legacy_bvh_kinematics(path: Path, query_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return BVH foot-ready world positions, global rotations and offsets."""
    motion, frame_time, offsets_cm = _parse_legacy_bvh(path)
    sampled = _interp_legacy_bvh(motion, frame_time, query_t)
    root_pos = sampled[:, :3] * 0.01
    local_rot = SciRotation.from_euler(
        "YXZ",
        sampled[:, 3:].reshape(-1, 3),
        degrees=True,
    ).as_matrix().reshape(-1, LEGACY_BVH_N_JOINTS, 3, 3)
    offsets_m = offsets_cm * 0.01
    pos, global_rot = fk_local_np(local_rot, offsets_m, LEGACY_BVH_PARENTS)
    return pos + root_pos[:, None, :], global_rot, offsets_m


def pool_48_to_16(values48: np.ndarray) -> np.ndarray:
    """Map 4x12 sensors onto Moticon 16 channels (verbatim S2M process_gait.py).

    Grid is (row=width, col=length). Columns 0-5 are toe, 6-11 are heel.
    Each 4x6 half is pooled into 8 channels as 4 rows x 2 groups of 3 columns.
    Official order is heel[8] then toe[8].
    """
    grid = values48.reshape(-1, 4, 12)
    toe = grid[:, :, :6].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
    heel = grid[:, :, 6:].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
    return np.concatenate([heel, toe], axis=1).astype(np.float32)


def cop_from_grid(values48: np.ndarray) -> np.ndarray:
    """Center of pressure per frame (verbatim S2M process_gait.py)."""
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


def rest_foot_imu_axes(offsets: np.ndarray) -> dict:
    """BVH-23 foot IMU basis (verbatim S2M rest_foot_imu_axes)."""
    identity = np.broadcast_to(np.eye(3), (1, LEGACY_BVH_N_JOINTS, 3, 3)).copy()
    pos, _ = fk_local_np(identity, offsets, LEGACY_BVH_PARENTS)

    def compute_axes(toes_index, foot_index, opposite_foot_index, invert_up):
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

    left = compute_axes(LEGACY_LEFT_TOE, LEGACY_LEFT_FOOT, LEGACY_RIGHT_FOOT, invert_up=True)
    right = compute_axes(LEGACY_RIGHT_TOE, LEGACY_RIGHT_FOOT, LEGACY_LEFT_FOOT, invert_up=False)
    return {"left": left, "right": right}


def rotmat_to_quat_wxyz(rotmats: np.ndarray) -> np.ndarray:
    """Shepperd rotmat->quat [w, x, y, z] (S2M uses scipy; pure-numpy port).

    The branch divisions evaluate on all elements (np.where), so unused
    branches divide by s=0 — suppressed via errstate; the results of those
    lanes are discarded by the where().
    """
    m00 = rotmats[..., 0, 0]; m01 = rotmats[..., 0, 1]; m02 = rotmats[..., 0, 2]
    m10 = rotmats[..., 1, 0]; m11 = rotmats[..., 1, 1]; m12 = rotmats[..., 1, 2]
    m20 = rotmats[..., 2, 0]; m21 = rotmats[..., 2, 1]; m22 = rotmats[..., 2, 2]
    shape = rotmats.shape[:-2]
    w = np.zeros(shape, dtype=np.float64)
    x = np.zeros(shape, dtype=np.float64)
    y = np.zeros(shape, dtype=np.float64)
    z = np.zeros(shape, dtype=np.float64)
    tr = m00 + m11 + m22

    with np.errstate(divide="ignore", invalid="ignore"):
        c0 = tr > 0
        s = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2.0
        w = np.where(c0, s * 0.25, w)
        x = np.where(c0, (m21 - m12) / s, x)
        y = np.where(c0, (m02 - m20) / s, y)
        z = np.where(c0, (m10 - m01) / s, z)

        c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
        s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        w = np.where(c1, (m21 - m12) / s, w)
        x = np.where(c1, s * 0.25, x)
        y = np.where(c1, (m01 + m10) / s, y)
        z = np.where(c1, (m02 + m20) / s, z)

        c2 = (~c0) & (~c1) & (m11 >= m22)
        s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0)) * 2.0
        w = np.where(c2, (m02 - m20) / s, w)
        x = np.where(c2, (m01 + m10) / s, x)
        y = np.where(c2, s * 0.25, y)
        z = np.where(c2, (m12 + m21) / s, z)

        c3 = ~(c0 | c1 | c2)
        s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0)) * 2.0
        w = np.where(c3, (m10 - m01) / s, w)
        x = np.where(c3, (m02 + m20) / s, x)
        y = np.where(c3, (m12 + m21) / s, y)
        z = np.where(c3, s * 0.25, z)

    q = np.stack([w, x, y, z], axis=-1)
    return (q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


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
    """Local angular velocity from rotation matrices (verbatim S2M)."""
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


def synthesize_imu(global_pos, global_rot, rest_axes, dt, flip_local_y):
    """Foot IMU channels (verbatim S2M synthesize_imu, then the world-frame
    round trip of S2M dataset.py:136-148 in rotmat form).

    The round trip cancels exactly (acc_local = R^T(a+g)/G -> R·acc_local
    minus 1 on Y = a/G), so the final channel is world-frame gravity-free
    acceleration in g units; gyro stays in the foot-local frame (rad/s).
    """
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
    # World-frame conversion (S2M dataset.py:136-148): undo the left-foot Y
    # flip, rotate back to world, subtract gravity on Y.
    if flip_local_y:
        acc_local[..., 1] *= -1.0
    acc_world_rec = np.einsum("...ij,...j->...i", imu_rot, acc_local)
    acc_world_rec[..., 1] -= 1.0
    return acc_world_rec.astype(np.float32), gyro_local.astype(np.float32)


def _foot_features(values48, foot_pos, foot_rot, rest_axes, dt, flip_local_y):
    """25-dim per-foot channel (verbatim S2M foot_features ordering)."""
    pressure16 = pool_48_to_16(values48)
    acc, gyro = synthesize_imu(foot_pos, foot_rot, rest_axes, dt, flip_local_y)
    force = values48.sum(axis=1, keepdims=True).astype(np.float32)
    cop = cop_from_grid(values48)
    return np.concatenate([pressure16, acc, gyro, force, cop], axis=1).astype(np.float32)


def _foot_features_noimu(values48):
    """--no-imu: 19-dim per-foot channel = pressure16 + force1 + cop2.

    Identical to ``_foot_features`` with the acc/gyro block removed — the IMU
    values are never computed (synthesize_imu is not called on this path).
    """
    pressure16 = pool_48_to_16(values48)
    force = values48.sum(axis=1, keepdims=True).astype(np.float32)
    cop = cop_from_grid(values48)
    return np.concatenate([pressure16, force, cop], axis=1).astype(np.float32)


def drop_imu_columns(t_s2m50: np.ndarray) -> np.ndarray:
    """Delete the IMU columns from a 50-dim T_s2m (verification helper).

    The 38-dim --no-imu build must equal this drop exactly; it documents the
    deletion mapping ``[16:22] + [41:47]`` and is used by the no-IMU probes.
    """
    if t_s2m50.shape[-1] != T_S2M_DIM:
        raise ValueError("drop_imu_columns expects 50-dim input, got %d" % t_s2m50.shape[-1])
    keep = [i for i in range(T_S2M_DIM) if i not in S2M_IMU_COLUMNS]
    return t_s2m50[..., keep].astype(np.float32)


def build_t_s2m(
    t_raw_norm: np.ndarray,
    *,
    bvh_path: Path | None = None,
    query_t: np.ndarray | None = None,
    fps: float = FPS,
    no_imu: bool = False,
) -> np.ndarray:
    """Build the Step2Motion-口径 tactile channel for one session.

    ``t_raw_norm`` is the anysole-normalized 96-dim pressure (clip/PRESSURE_CLIP
    then divide).  With IMU enabled, ``bvh_path`` and ``query_t`` are required;
    the foot IMU is synthesized exclusively from the directly exported BVH-23
    motion using Step2Motion's Left/RightFoot + ToeBase semantics.

    ``no_imu=True`` (--no-imu): the IMU channels are structurally absent —
    neither BVH parsing nor ``synthesize_imu`` is called and the output is
    38-dim per frame (pressure16 + force1 + cop2 per foot).
    """
    t_raw_norm = np.asarray(t_raw_norm, dtype=np.float32)
    if t_raw_norm.ndim != 2 or t_raw_norm.shape[1] != 96:
        raise ValueError("t_raw_norm must have shape (T, 96), got %s" % (t_raw_norm.shape,))
    left48 = t_raw_norm[:, :48]
    right48 = t_raw_norm[:, 48:]
    if no_imu:
        left = _foot_features_noimu(left48)
        right = _foot_features_noimu(right48)
    else:
        if bvh_path is None or query_t is None:
            raise ValueError("BVH-derived s2m50 IMU requires both bvh_path and query_t")
        query_t = np.asarray(query_t, dtype=np.float64)
        if query_t.shape != (t_raw_norm.shape[0],):
            raise ValueError(
                "query_t shape %s != (%d,)" % (query_t.shape, t_raw_norm.shape[0])
            )
        world_pos, global_rot, offsets_m = _legacy_bvh_kinematics(Path(bvh_path), query_t)
        rest_axes = rest_foot_imu_axes(offsets_m)
        dt = 1.0 / float(fps)
        left = _foot_features(
            left48, world_pos[:, LEGACY_LEFT_FOOT], global_rot[:, LEGACY_LEFT_FOOT],
            rest_axes["left"], dt, flip_local_y=True,
        )
        right = _foot_features(
            right48, world_pos[:, LEGACY_RIGHT_FOOT], global_rot[:, LEGACY_RIGHT_FOOT],
            rest_axes["right"], dt, flip_local_y=False,
        )
    out = np.concatenate([left, right], axis=1).astype(np.float32)
    expected = T_S2M_NOIMU_DIM if no_imu else T_S2M_DIM
    if out.shape[1] != expected:
        raise ValueError("T_s2m dim %d != %d (no_imu=%s)" % (out.shape[1], expected, no_imu))
    return out
