"""SMPL-NPZ I/O for AnySole's native SMPL-24 protocol.

No Skeleton3/BVH conversion occurs here.  AnySole receives the standard 24
SMPL local rotations as 6D values and uses the standard SMPL parent tree.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
import hashlib
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation, Slerp

from anysole.types import (
    JOINT_NAMES,
    JOINT_PARENTS,
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    SMPL_MODEL_PATH,
)


SMPL_COORDINATE_SYSTEM = (
    "SMPL right-handed +X left, +Y up, +Z forward; world motion preserved"
)
SMPL_MODEL_UNITS = "m"


@lru_cache(maxsize=4)
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smpl_archive_metadata() -> dict:
    """Metadata written on every AnySole-predicted native SMPL archive."""
    model_path = Path(os.environ.get("ANYSOLE_SMPL_MODEL", str(SMPL_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL_NEUTRAL.pkl not found: {model_path}")
    return {
        "surface_model_type": np.asarray("smpl"),
        "gender": np.asarray("neutral"),
        "model_units": np.asarray(SMPL_MODEL_UNITS),
        "coordinate_system": np.asarray(SMPL_COORDINATE_SYSTEM),
        "surface_model_sha256": np.asarray(_sha256(str(model_path))),
        "motion_protocol": np.asarray(MOTION_PROTOCOL),
        "joint_protocol_checksum": np.asarray(JOINT_PROTOCOL_CHECKSUM),
    }


def _rotmat_to_6d_np(rotmats: np.ndarray) -> np.ndarray:
    return np.concatenate([rotmats[..., :, 0], rotmats[..., :, 1]], axis=-1).astype(np.float32)


def _scalar(value, default):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else default


def _metadata_scalar(data: dict, key: str, default: str = "") -> str:
    value = np.asarray(data.get(key, default))
    return str(value.reshape(-1)[0]) if value.size else default


def _validate_archive(data: dict, path: Path) -> None:
    """Reject SMPL archives whose coordinate/model contract is incompatible.

    Shape checks alone are not sufficient here: a 72-D pose in a Z-up or
    centimetre-valued archive is still 72-D, but would silently corrupt every
    trajectory, floor and contact computation downstream. Older archives may
    omit provenance metadata, so absence is tolerated; contradictory metadata
    is not.
    """
    model_type = _metadata_scalar(data, "surface_model_type").lower()
    gender = _metadata_scalar(data, "gender").lower()
    units = _metadata_scalar(data, "model_units").lower()
    coordinates = _metadata_scalar(data, "coordinate_system")
    protocol = _metadata_scalar(data, "motion_protocol")
    checksum = _metadata_scalar(data, "joint_protocol_checksum")
    if model_type and model_type != "smpl":
        raise ValueError(f"{path}: surface_model_type={model_type!r}, expected 'smpl'")
    if gender and gender != "neutral":
        raise ValueError(
            f"{path}: gender={gender!r}, but AnySole currently uses the neutral SMPL model"
        )
    if units and units != SMPL_MODEL_UNITS:
        raise ValueError(f"{path}: model_units={units!r}, expected metres ('m')")
    if coordinates and coordinates != SMPL_COORDINATE_SYSTEM:
        raise ValueError(
            f"{path}: incompatible coordinate_system={coordinates!r}; "
            f"expected {SMPL_COORDINATE_SYSTEM!r}"
        )
    # Raw MoSh archives predate AnySole and legitimately omit both fields.
    # Once an archive declares itself to be an AnySole SMPL-24 artifact, the
    # joint-tree checksum is mandatory: early migration outputs were also
    # 24/72D but were generated/evaluated with the wrong collar parents.
    if protocol:
        if protocol != MOTION_PROTOCOL or checksum != JOINT_PROTOCOL_CHECKSUM:
            raise ValueError(
                f"{path}: incompatible generated-motion protocol "
                f"(motion_protocol={protocol!r}, joint checksum={checksum or 'missing'}); "
                "the file may be an early SMPL-24 artifact using the incorrect "
                "collar-parent tree"
            )


@lru_cache(maxsize=4)
def _smpl_model(path: str):
    with open(path, "rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    reg = model["J_regressor"]
    if hasattr(reg, "toarray"):
        reg = reg.toarray()
    reg = np.asarray(reg, dtype=np.float64)
    template = np.asarray(model["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    if reg.shape != (N_JOINTS, 6890) or template.shape != (6890, 3):
        raise ValueError("unexpected SMPL model shapes: %s %s" % (reg.shape, template.shape))
    tree = np.asarray(model.get("kintree_table"))
    if tree.shape != (2, N_JOINTS):
        raise ValueError("unexpected SMPL kintree_table shape: %s" % (tree.shape,))
    ids = [int(value) for value in tree[1]]
    id_to_index = {joint_id: index for index, joint_id in enumerate(ids)}
    parents = [-1]
    for joint in range(1, N_JOINTS):
        parent_id = int(tree[0, joint])
        parents.append(id_to_index.get(parent_id, -2))
    if tuple(parents) != tuple(JOINT_PARENTS):
        raise ValueError(
            "local SMPL model kinematic tree does not match AnySole SMPL-24: %s" % parents
        )
    return reg, template, shapedirs


def _rest_joints(data: dict) -> np.ndarray:
    model_path = Path(os.environ.get("ANYSOLE_SMPL_MODEL", str(SMPL_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError("SMPL_NEUTRAL.pkl not found: %s" % model_path)
    reg, template, shapedirs = _smpl_model(str(model_path))
    betas = np.asarray(data.get("betas", np.zeros(10)), dtype=np.float64)
    if not np.isfinite(betas).all():
        raise ValueError("SMPL betas contain NaN or Inf")
    if betas.ndim > 1:
        betas = betas.reshape(-1, betas.shape[-1])[0]
    n_shape = shapedirs.shape[-1]
    beta = np.zeros(n_shape, dtype=np.float64)
    beta[: min(n_shape, betas.size)] = betas[:n_shape]
    vertices = template + np.einsum("vkc,c->vk", shapedirs, beta)
    return reg.dot(vertices)


def resolve_smpl_path(meta: dict, roots: tuple[Path, ...], cache: Optional[dict] = None) -> Path:
    recorded = str(meta.get("smpl_path", ""))
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    sid = str(meta.get("session_id", ""))
    if cache is not None and sid in cache:
        return cache[sid]
    for root in roots:
        candidates = sorted(Path(root).glob(f"**/{sid}/motion_neutral_smpl.npz"))
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
        # Some MoSh archives expose only 21 articulated body joints (63D):
        # standard SMPL's two terminal hand rotations are omitted because they
        # do not move any of the 24 joints. Their native SMPL value is zero.
        if body.ndim == 2 and body.shape[1] == 63:
            body = np.concatenate([body, np.zeros((body.shape[0], 6))], axis=-1)
        poses = np.concatenate([root, body], axis=-1)
    if poses.ndim != 2 or poses.shape[1] != N_JOINTS * 3:
        raise ValueError(f"SMPL poses must have shape (T,{N_JOINTS * 3}), got {poses.shape}")
    return poses.reshape(-1, N_JOINTS, 3)


def load_smpl(path: Path, query_t: Optional[np.ndarray] = None) -> dict:
    data = dict(np.load(Path(path), allow_pickle=True))
    _validate_archive(data, Path(path))
    poses = _pose_arrays(data)
    model_trans = np.asarray(data["trans"], dtype=np.float64)
    if model_trans.shape != (poses.shape[0], 3):
        raise ValueError(f"SMPL trans must have shape ({poses.shape[0]},3), got {model_trans.shape}")
    fps = _scalar(data.get("mocap_frame_rate", 120.0), 120.0)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: invalid mocap_frame_rate={fps!r}")
    src_t = np.asarray(data.get("source_frame_times_s", np.arange(len(poses)) / fps), dtype=np.float64).reshape(-1)
    if src_t.size != len(poses):
        src_t = np.arange(len(poses), dtype=np.float64) / fps
    if not np.isfinite(src_t).all() or (len(src_t) > 1 and not np.all(np.diff(src_t) > 0.0)):
        raise ValueError(f"{path}: source_frame_times_s must be finite and strictly increasing")
    if not np.isfinite(poses).all() or not np.isfinite(model_trans).all():
        raise ValueError(f"{path}: poses/trans contain NaN or Inf")
    # Redundant fields are independent guards against a wrong joint order or
    # a truncated pose_body array.
    if "root_orient" in data and not np.allclose(
        poses[:, 0], np.asarray(data["root_orient"]), atol=1e-8
    ):
        raise ValueError(f"{path}: poses root does not match root_orient")
    if "pose_body" in data:
        body = np.asarray(data["pose_body"], dtype=np.float64)
        if body.ndim != 2 or body.shape[0] != len(poses) or body.shape[1] not in (63, 69):
            raise ValueError(f"{path}: pose_body must be (T,63) or (T,69), got {body.shape}")
        if not np.allclose(
            poses[:, 1 : 1 + body.shape[1] // 3].reshape(len(poses), -1),
            body,
            atol=1e-8,
        ):
            raise ValueError(f"{path}: poses body does not match pose_body")
    if query_t is not None:
        q = np.asarray(query_t, dtype=np.float64).reshape(-1)
        clipped = np.clip(q, src_t[0], src_t[-1])
        if len(src_t) == 1:
            poses = np.repeat(poses[:1], len(q), axis=0)
            model_trans = np.repeat(model_trans[:1], len(q), axis=0)
        else:
            sampled = np.empty((len(q), N_JOINTS, 3), dtype=np.float64)
            for joint in range(N_JOINTS):
                sampled[:, joint] = Slerp(src_t, SciRotation.from_rotvec(poses[:, joint]))(clipped).as_rotvec()
            poses = sampled
            model_trans = np.stack([np.interp(clipped, src_t, model_trans[:, i]) for i in range(3)], axis=1)

    rest_joints = _rest_joints(data)
    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for joint, parent in enumerate(JOINT_PARENTS):
        if parent >= 0:
            offsets[joint] = rest_joints[joint] - rest_joints[parent]
    rotmats = SciRotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, N_JOINTS, 3, 3)
    # SMPL's ``trans`` locates the model origin.  In the standard SMPL
    # batch-rigid-transform, the root transform has translation J_root and
    # rotation R_root, so the root joint itself is ``trans + J_root`` -- the
    # rest pelvis offset is *not* rotated by the root orientation.  AnySole's
    # FK translation is the pelvis position, hence this constant shape-
    # dependent offset must be added without R_root.
    pelvis_world = model_trans + rest_joints[0]
    return {
        "pose_6d": _rotmat_to_6d_np(rotmats).reshape(len(poses), N_JOINTS * 6),
        "trans_m": pelvis_world.astype(np.float32),
        "model_trans_m": model_trans.astype(np.float32),
        "offsets_m": offsets.astype(np.float32),
        "parents": np.asarray(JOINT_PARENTS, dtype=np.int64),
        "joint_rest_m": rest_joints.astype(np.float32),
        "joint_names": tuple(JOINT_NAMES),
        "betas": np.asarray(data.get("betas", np.zeros(10))).reshape(-1)[:10].astype(np.float32),
        "fps": fps,
        "source_path": str(path),
        "source_model_sha256": _metadata_scalar(data, "surface_model_sha256"),
        "coordinate_system": _metadata_scalar(data, "coordinate_system"),
    }


def estimate_floor_y(joints: np.ndarray) -> np.float32:
    """Shared native-SMPL floor reference used by labels, loss and metrics.

    SMPL joints are joint centres rather than sole vertices, so this is a
    session-relative reference, not a claim that archive world Y=0 is exactly
    the floor. Keeping this helper shared prevents the label builder and
    Dataset from silently using different definitions.
    """
    points = np.asarray(joints)
    if points.ndim != 3 or points.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"SMPL joints must be (T,{N_JOINTS},3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("SMPL joints contain NaN or Inf")
    return np.float32(np.percentile(points[:, :, 1], 5))


def smpl24_pose6d_to_poses(pose_6d: np.ndarray) -> np.ndarray:
    """Serialize native SMPL-24 6D rotations to 72D axis-angle."""
    pose = np.asarray(pose_6d, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != N_JOINTS * 6:
        raise ValueError(f"pose_6d must have shape (T,{N_JOINTS * 6}), got {pose.shape}")
    c = pose.reshape(-1, N_JOINTS, 6)
    c0 = c[..., :3] / np.maximum(np.linalg.norm(c[..., :3], axis=-1, keepdims=True), 1e-8)
    c1 = c[..., 3:] - np.sum(c0 * c[..., 3:], axis=-1, keepdims=True) * c0
    c1 = c1 / np.maximum(np.linalg.norm(c1, axis=-1, keepdims=True), 1e-8)
    mats = np.stack((c0, c1, np.cross(c0, c1)), axis=-1)
    return SciRotation.from_matrix(mats.reshape(-1, 3, 3)).as_rotvec().reshape(-1, N_JOINTS * 3).astype(np.float32)


def pelvis_to_smpl_trans(pose_6d: np.ndarray, pelvis_trans: np.ndarray, betas=None) -> np.ndarray:
    """Convert AnySole's pelvis-root translation back to SMPL ``trans``.

    SMPL archives store the model-origin translation, while AnySole FK stores
    the pelvis position.  Exporters must subtract the (unrotated)
    shape-dependent pelvis rest joint or a subsequent load would add that
    offset twice.
    """
    pose = np.asarray(pose_6d, dtype=np.float64).reshape(-1, N_JOINTS, 6)
    pelvis = np.asarray(pelvis_trans, dtype=np.float64).reshape(-1, 3)
    if pose.shape[0] != pelvis.shape[0]:
        raise ValueError("pose and pelvis translation frame counts differ")
    rest = _rest_joints({"betas": np.zeros(10) if betas is None else betas})[0]
    return (pelvis - rest).astype(np.float32)
