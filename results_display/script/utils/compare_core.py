"""Shared protocol/metric core for the R_Test1/2/3 comparison layer.

Extracted from ``r_test2_compare.py`` so the mode-aware comparison tables
(R_Test2 ``--by-mode``) and the side-by-side comparison views (R_Test1
``r_test1_compare.py`` / R_Test3 ``--compare``) evaluate against the exact
same machinery: same common19 semantic joint mapping, same native-protocol GT
loading, same metric formulas.  Any number that appears in one comparison
output is by construction identical to the same row elsewhere.

Protocol rule: a prediction is always evaluated against GT expressed in the
*prediction's own native protocol* (SMPL-24 or BVH-23); protocols are never
aligned by numeric joint index.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # script/ (for utils.*)
from utils import cli_common  # noqa: E402

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anysole.types import JOINT_NAMES  # noqa: E402
from anysole.utils.metrics import (  # noqa: E402
    MOTION_METRIC_NAMES,
    foot_sliding,
    mean_point_error,
    pa_mpjpe,
    pelvis_align,
    root_trajectory_metrics,
    matrix_rotation_error_degrees,
    rotation_error_degrees,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)
from utils.motion_io import LEGACY_BVH_NAMES, load_motion, load_rotations, load_surface  # noqa: E402

CONTACT_METRICS = ("contact_f1", "contact_acc", "contact_recall", "air_recall")
V2T_METRICS = (
    "T_mae", "T_rmse", "T_corr", "pressure_force_mae", "pressure_force_rmse",
    "pressure_force_r2", "pressure_cop_error_left", "pressure_cop_error_right",
    "pressure_cop_error_mean",
)
PRESSURE_GRID_SHAPE = (31, 11)
METRICS = MOTION_METRIC_NAMES + CONTACT_METRICS + V2T_METRICS
MODE_METRICS = {
    "VT2M": MOTION_METRIC_NAMES + CONTACT_METRICS,
    "V2M": MOTION_METRIC_NAMES + CONTACT_METRICS,
    "T2M": MOTION_METRIC_NAMES + CONTACT_METRICS,
    "V2T": V2T_METRICS + CONTACT_METRICS,
}

# Generation modes.  Rows from different modes are never compared in one
# table block: T2M vs V2M is a different task, not a different model.
MODES = ("VT2M", "V2M", "T2M", "V2T")
MOTION_MODES = ("VT2M", "V2M", "T2M")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "models_modes.yaml"

# Skeleton3 has one more intermediate spine joint and no SMPL terminal hand
# joints.  The intersection below deliberately excludes ambiguous spine
# levels and the unmatched SMPL hands.  Ordering starts with pelvis because
# metrics() uses joint 0 as the root.
COMMON_JOINTS = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot", "neck", "head",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
SMPL_COMMON_NAMES = {
    name: name for name in COMMON_JOINTS
}
BVH_COMMON_NAMES = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg", "right_knee": "RightLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_foot": "LeftToeBase", "right_foot": "RightToeBase",
    "neck": "Neck", "head": "Head",
    "left_collar": "LeftShoulder", "right_collar": "RightShoulder",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}

# Stick edges over COMMON_JOINTS (pelvis-rooted display convention).
COMMON_EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
    ("pelvis", "neck"), ("neck", "head"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]

# Panel title colors for the comparison views: main model keeps the standard
# prediction blue, GT stays orange, baselines get one hue each (registry).
MAIN_COLOR = (80, 200, 255)
GT_COLOR = (255, 170, 80)
MISSING_COLOR = (120, 120, 128)
FALLBACK_BASELINE_COLORS = ((138, 92, 230), (31, 157, 138), (194, 91, 31), (91, 122, 194))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    try:
        out = json.loads(str(value))
        return [int(x) for x in out]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


def load_split_ids(split_csv: Path, split: str) -> list[str]:
    """Read the project-owned canonical split table in row order."""
    ids: list[str] = []
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or split not in rows.fieldnames:
            raise ValueError(f"{split_csv} missing split column {split!r}")
        for row in rows:
            value = (row.get(split) or "").strip()
            if value:
                ids.append(value)
    if not ids:
        raise ValueError(f"canonical split {split!r} is empty in {split_csv}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"canonical split {split!r} contains duplicate session ids")
    return ids


def read_manifest(path: Path, split: str, split_csv: Path | None = None,
                  evaluable_only: bool = True) -> list[dict[str, str]]:
    """Read manifest metadata using the canonical split table when supplied.

    The manifest remains the source of per-session metadata, but it is not a
    second authority for train/val/test membership.  This prevents a stale
    ``split_iid``/``split_ood`` tag from silently changing a comparison set.
    """
    opener = path.open(encoding="utf-8-sig")
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in opener if line.strip()]
    else:
        rows = list(csv.DictReader(opener))
    by_id = {str(row["session_id"]): row for row in rows if row.get("session_id")}
    if split_csv is not None:
        ids = load_split_ids(split_csv, split)
        missing = [sid for sid in ids if sid not in by_id]
        if missing:
            raise ValueError(
                f"canonical split {split!r} contains sessions absent from {path}: {missing}"
            )
        selected = [by_id[sid] for sid in ids]
        if evaluable_only:
            selected = [
                row for row in selected
                if parse_list(row.get("valid_frame_indices"))
                or not ("valid_frame_indices" in row)
            ]
        return selected
    selected = []
    for row in rows:
        tags = {x.strip() for x in str(row.get("split_iid", "")).split(",")}
        if split not in tags and str(row.get("split_ood", "")).strip() != split:
            continue
        selected.append(row)
    return selected


def resolve_repo_path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return p
    marker = "/mocap_ori_bvh/"
    normalized = p.as_posix()
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        matches = sorted((ROOT / "AnysoleWorkspace/sources/raw").glob("*/mocap_ori_bvh/" + suffix))
        if matches:
            return matches[0]
    for candidate in (base / p, ROOT / p):
        if candidate.exists():
            return candidate
    return p if p.is_absolute() else ROOT / p


def bvh_joints(path: Path, row: dict[str, str] | None = None) -> np.ndarray:
    """World joints from a BVH; with a manifest row, resampled onto the session grid.

    Without ``row`` the raw BVH is returned as-is.  With ``row`` the motion is
    queried at ``t_mocap = visual_start_s + i/target_fps - offset_s``, the
    mocap<->tactile/video bias compensation used by MotionPRO
    ``prepare_sequences.py`` and ``anysole/data/dataset.py``.  Predictions are
    exported on the session grid, so comparing them frame-by-frame against the
    raw BVH would misalign GT by ``mocap_start_s`` (median ~0.08 s, max ~0.9 s).
    """
    query_t = None
    if row is not None and "visual_start_s" in row and "offset_s" in row:
        n = int(float(row.get("n_frames") or 0))
        if n <= 0:
            raise ValueError("manifest n_frames must be positive for aligned BVH loading")
        fps = float(row.get("target_fps") or 40.0)
        t_grid = float(row["visual_start_s"]) + np.arange(n, dtype=np.float64) / fps
        query_t = t_grid - float(row["offset_s"])
    loaded = load_motion(path, query_t=query_t)
    if loaded["format"] != "bvh":
        raise ValueError(f"expected legacy BVH, got {loaded['format']}: {path}")
    return np.asarray(loaded["joints"], dtype=np.float32)


def _decode_names(values) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v)
                 for v in np.asarray(values).reshape(-1).tolist())


def _infer_protocol(arr: np.ndarray, names: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]]:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"joint array must be (T,J,3), got {arr.shape}")
    if names is not None:
        if len(names) != arr.shape[1]:
            raise ValueError(f"joint_names has {len(names)} entries for J={arr.shape[1]}")
        if set(JOINT_NAMES).issubset(names):
            return "smpl24", names
        if set(LEGACY_BVH_NAMES).issubset(names):
            return "bvh23", names
        raise ValueError(f"unrecognized joint_names protocol: {names}")
    if arr.shape[1] == len(JOINT_NAMES):
        return "smpl24", tuple(JOINT_NAMES)
    if arr.shape[1] == len(LEGACY_BVH_NAMES):
        return "bvh23", tuple(LEGACY_BVH_NAMES)
    raise ValueError(
        f"cannot infer motion protocol from J={arr.shape[1]}; include joint_names"
    )


def array_from_file(path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...], str]:
    """Load (T,J,3) world joints + optional valid mask + joint names + protocol.

    NPZ files follow the unified ``eval_motion`` contract
    (``joint_xyz_world`` / ``joint_names`` / ``valid_mask``) or the SMPL NPZ
    motion contract (``poses``/``trans``).  ``*.bvh`` files are parsed as
    BVH-23 (Skeleton3).  Returned joints are in the z-up display frame.
    """
    if path.suffix.lower() == ".bvh":
        return bvh_joints(path), None, tuple(LEGACY_BVH_NAMES), "bvh23"
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            # The explicit unified contract wins over native SMPL parameter
            # fields that may be carried in the same archive for trajectory
            # and mesh readers.  AnySole archives contain both forms.
            if "joint_xyz_world" in data:
                arr = np.asarray(data["joint_xyz_world"])
                mask = np.asarray(data["valid_mask"]).reshape(-1) if "valid_mask" in data else None
                names = _decode_names(data["joint_names"]) if "joint_names" in data else None
                protocol, names = _infer_protocol(arr, names)
                return arr, mask, names, protocol
            if ({"poses", "trans"}.issubset(data.files)
                    or {"root_orient", "pose_body", "trans"}.issubset(data.files)):
                motion = load_motion(path)
                return motion["joints"], None, tuple(motion["names"]), "smpl24"
            keys = list(data.keys())
            candidates = ("joints_world", "pred_joints", "joints", "xyz", "pose")
            key = next((k for k in candidates if k in data), None)
            if key is None:
                raise ValueError(f"no joint array in {path}; keys={keys}")
            arr = np.asarray(data[key])
            mask = np.asarray(data["valid_mask"]).reshape(-1) if "valid_mask" in data else None
            names = _decode_names(data["joint_names"]) if "joint_names" in data else None
            protocol, names = _infer_protocol(arr, names)
            return arr, mask, names, protocol
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        protocol, names = _infer_protocol(obj, None)
        return obj, None, names, protocol
    if isinstance(obj, dict):
        for key in ("joint_xyz_world", "joints_world", "pred_joints", "joints", "xyz"):
            if key in obj:
                arr = np.asarray(obj[key])
                names_in = _decode_names(obj["joint_names"]) if "joint_names" in obj else None
                protocol, names = _infer_protocol(arr, names_in)
                mask = np.asarray(obj.get("valid_mask")).reshape(-1) if "valid_mask" in obj else None
                return arr, mask, names, protocol
    raise ValueError(f"unsupported prediction object: {path}")


def select_common_joints(joints: np.ndarray, names: tuple[str, ...], protocol: str) -> np.ndarray:
    """Select the fixed semantic intersection; never align protocols by index."""
    mapping = SMPL_COMMON_NAMES if protocol == "smpl24" else BVH_COMMON_NAMES
    index = {name: i for i, name in enumerate(names)}
    missing = [mapping[key] for key in COMMON_JOINTS if mapping[key] not in index]
    if missing:
        raise ValueError(f"{protocol} missing common joints: {missing}")
    return np.asarray(joints)[:, [index[mapping[key]] for key in COMMON_JOINTS], :]


def select_common_rotations(rotations: np.ndarray, names: tuple[str, ...], protocol: str) -> np.ndarray:
    mapping = SMPL_COMMON_NAMES if protocol == "smpl24" else BVH_COMMON_NAMES
    index = {name: i for i, name in enumerate(names)}
    return np.asarray(rotations)[:, [index[mapping[key]] for key in COMMON_JOINTS]]


def protocol_gt(row: dict[str, str], protocol: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if protocol == "bvh23":
        path = resolve_repo_path(row["bvh_path"], ROOT)
        return bvh_joints(path, row), tuple(LEGACY_BVH_NAMES)
    if protocol == "smpl24":
        path = resolve_repo_path(row["smpl_path"], ROOT)
        n = int(float(row["n_frames"]))
        fps = float(row.get("target_fps") or 40.0)
        query_t = (float(row["visual_start_s"])
                   + np.arange(n, dtype=np.float64) / fps
                   - float(row["offset_s"]))
        motion = load_motion(path, query_t=query_t)
        return motion["joints"], tuple(motion["names"])
    raise ValueError(f"unsupported protocol {protocol!r}")


def protocol_gt_surface(row: dict[str, str], protocol: str) -> dict | None:
    if protocol != "smpl24":
        return None
    path = resolve_repo_path(row["smpl_path"], ROOT)
    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    query_t = (float(row["visual_start_s"])
               + np.arange(n, dtype=np.float64) / fps
               - float(row["offset_s"]))
    return load_surface(path, query_t=query_t)


def protocol_gt_rotations(row: dict[str, str], protocol: str) -> dict:
    if protocol == "bvh23":
        path = resolve_repo_path(row["bvh_path"], ROOT)
        n = int(float(row.get("n_frames") or 0))
        fps = float(row.get("target_fps") or 40.0)
        query_t = (float(row["visual_start_s"])
                   + np.arange(n, dtype=np.float64) / fps
                   - float(row["offset_s"]))
        return load_rotations(path, query_t=query_t)
    path = resolve_repo_path(row["smpl_path"], ROOT)
    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    query_t = (float(row["visual_start_s"])
               + np.arange(n, dtype=np.float64) / fps
               - float(row["offset_s"]))
    return load_rotations(path, query_t=query_t)


def find_prediction(root: Path, session_id: str, pattern: str | None = None, config_id: str | None = None) -> Path | None:
    if pattern:
        try:
            candidate = root / pattern.format(session_id=session_id, config_id=config_id)
        except (KeyError, IndexError):
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate
    for ext in (".npz", ".npy", ".bvh"):
        exact = root / f"{session_id}{ext}"
        if exact.is_file():
            return exact
    matches = sorted(p for p in root.rglob(f"*{session_id}*") if p.suffix.lower() in (".npz", ".npy", ".bvh"))
    return matches[0] if matches else None


def valid_mask(row: dict[str, str], n: int) -> np.ndarray:
    if row.get("valid_frame_indices"):
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray([i for i in parse_list(row["valid_frame_indices"]) if 0 <= i < n], dtype=int)] = True
        return keep
    mask = np.ones(n, dtype=bool)
    for i in parse_list(row.get("fake_frame_indices")):
        if 0 <= i < n:
            mask[i] = False
    return mask


def load_contact_gt(row: dict[str, str], n: int, method: str = "joint_and") -> np.ndarray | None:
    pressure_path = resolve_repo_path(row["pressure_path"], ROOT)
    path = pressure_path.parent / ("contact.npy" if method == "tactile_abs" else f"contact_{method}.npy")
    if not path.is_file():
        return None
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 8:
        raise ValueError(f"invalid contact labels {values.shape}: {path}")
    return values[:n, 6:8] > 0.5


def contact_from_joints(joints: np.ndarray, names: tuple[str, ...], floor: float,
                        fps: float) -> np.ndarray:
    """Current AnySole soft-contact definition in the shared z-up frame."""
    index = {name: i for i, name in enumerate(names)}
    candidates = (("left_foot", "LeftToeBase"), ("right_foot", "RightToeBase"))
    ids = []
    for choices in candidates:
        found = next((index[name] for name in choices if name in index), None)
        if found is None:
            raise ValueError(f"cannot find foot joint in protocol names: {choices}")
        ids.append(found)
    foot = np.asarray(joints, dtype=np.float64)[:, ids]
    height = foot[..., 2] - float(floor)
    speed = np.zeros(height.shape, dtype=np.float64)
    if len(foot) > 1:
        delta = np.linalg.norm(np.diff(foot, axis=0), axis=-1) * float(fps)
        speed[0] = delta[0]
        speed[1:] = delta
    with np.errstate(over="ignore", under="ignore"):
        height_prob = 1.0 / (1.0 + np.exp(-(0.05 - height) / 0.02))
        speed_prob = 1.0 / (1.0 + np.exp(-(0.20 - speed) / 0.05))
    return height_prob * speed_prob > 0.5


def canonical_pressure_feet(values: np.ndarray) -> np.ndarray:
    """Return normalized pressure as ``(T,2,31,11)`` for V2T metrics."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 96:
        grids = values.reshape(-1, 2, 4, 12)
    elif values.ndim == 3 and values.shape[1:] == (31, 22):
        grids = np.stack([values[:, :, :11], values[:, :, 11:]], axis=1)
    elif values.ndim == 4 and values.shape[1] == 2:
        grids = values
    else:
        raise ValueError(f"unsupported V2T pressure array shape {values.shape}")
    if tuple(grids.shape[-2:]) == (4, 12):
        grids = zoom(
            grids,
            (1, 1, PRESSURE_GRID_SHAPE[0] / 4, PRESSURE_GRID_SHAPE[1] / 12),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    if tuple(grids.shape[-2:]) != PRESSURE_GRID_SHAPE:
        raise ValueError(
            f"pressure grids must resample to {PRESSURE_GRID_SHAPE}, got {grids.shape[-2:]}"
        )
    return grids


def pressure_metrics(pred: np.ndarray, target: np.ndarray,
                     valid: np.ndarray | None = None) -> dict[str, float]:
    """Pressure reconstruction measures on one common 31x11 foot grid.

    AnySole stores 4x12 cells per foot while FPP-Net stores 31x11 cells per
    foot.  Comparing the flattened native arrays would be dimensionally
    invalid.  The former is bilinearly resampled to 31x11; force and CoP are
    then computed from the same normalized grid for both methods.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    p = canonical_pressure_feet(pred)
    g = canonical_pressure_feet(target)
    n = min(len(p), len(g))
    p, g = p[:n], g[:n]
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    p, g = p[keep], g[keep]
    if not len(p):
        return {key: float("nan") for key in V2T_METRICS}
    diff = p - g
    frame_corr = []
    for p_frame, g_frame in zip(p, g):
        pv, gv = p_frame.reshape(-1), g_frame.reshape(-1)
        if np.std(pv) > 1e-8 and np.std(gv) > 1e-8:
            frame_corr.append(float(np.corrcoef(pv, gv)[0, 1]))
    corr = float(np.mean(frame_corr)) if frame_corr else float("nan")
    force_p, force_g = p.sum(axis=(-1, -2)), g.sum(axis=(-1, -2))
    force_diff = force_p - force_g
    total_p, total_g = force_p.sum(axis=-1), force_g.sum(axis=-1)
    ss_res = float(np.square(total_p - total_g).sum())
    ss_tot = float(np.square(total_g - total_g.mean()).sum())

    def cop(grid):
        # Both supported layouts are canonicalized to [T,foot,width,length].
        width, length = grid.shape[-2:]
        mass = np.clip(grid, 0.0, None)
        force = mass.sum(axis=(-1, -2))
        x = np.arange(width, dtype=np.float64).reshape(1, 1, width, 1)
        y = np.arange(length, dtype=np.float64).reshape(1, 1, 1, length)
        cx = (mass * x).sum(axis=(-1, -2)) / np.maximum(force, 1e-8) / max(width - 1, 1)
        cy = 1.0 - (mass * y).sum(axis=(-1, -2)) / np.maximum(force, 1e-8) / max(length - 1, 1)
        return np.stack([cx, cy], axis=-1)

    cop_diff = np.linalg.norm(cop(p) - cop(g), axis=-1)
    return {
        "T_mae": float(np.abs(diff).mean()),
        "T_rmse": float(np.sqrt(np.square(diff).mean())),
        "T_corr": corr,
        "pressure_force_mae": float(np.abs(force_diff).mean()),
        "pressure_force_rmse": float(np.sqrt(np.square(force_diff).mean())),
        "pressure_force_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "pressure_cop_error_left": float(cop_diff[:, 0].mean()),
        "pressure_cop_error_right": float(cop_diff[:, 1].mean()),
        "pressure_cop_error_mean": float(cop_diff.mean()),
    }


def v2t_metrics(pred_pressure: np.ndarray, gt_pressure: np.ndarray,
                pred_contact: np.ndarray, gt_contact: np.ndarray,
                valid: np.ndarray | None = None) -> dict[str, float]:
    """Canonical V2T metrics.

    The contact arguments are retained for archive compatibility, but the
    cross-model contact score is derived from the same canonical pressure
    grid (any cell > 0.5 means that foot is in contact).
    """
    out = {key: float("nan") for key in METRICS}
    out.update(pressure_metrics(pred_pressure, gt_pressure, valid))
    pc = canonical_pressure_feet(pred_pressure).max(axis=(-1, -2)) > 0.5
    gc = canonical_pressure_feet(gt_pressure).max(axis=(-1, -2)) > 0.5
    n = min(len(pc), len(gc))
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    pc, gc = pc[:n][keep], gc[:n][keep]
    tp = int((pc & gc).sum()); fp = int((pc & ~gc).sum())
    fn = int((~pc & gc).sum()); tn = int((~pc & ~gc).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out["contact_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-8)
    out["contact_acc"] = (tp + tn) / max(tp + fp + fn + tn, 1)
    out["contact_recall"] = recall
    out["air_recall"] = tn / max(tn + fp, 1)
    out["n_valid_frames"] = int(keep.sum())
    return out


def load_v2t_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pressure_pred", "pressure_gt", "contact_pred", "contact_gt", "valid_mask"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"V2T archive missing keys {sorted(missing)}: {path}")
        return {key: np.asarray(data[key]) for key in required}


def metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    keep: np.ndarray,
    fps: float,
    *,
    pred_rotations: np.ndarray | None = None,
    gt_rotations: np.ndarray | None = None,
    pred_vertices: np.ndarray | None = None,
    gt_vertices: np.ndarray | None = None,
    times: np.ndarray | None = None,
    pred_contact: np.ndarray | None = None,
    gt_contact: np.ndarray | None = None,
) -> dict[str, float]:
    """Canonical metrics migrated from ``metrics.py``; no local redefinitions."""
    n = min(len(pred), len(gt), len(keep))
    pred = np.asarray(pred[:n], dtype=np.float64)
    gt = np.asarray(gt[:n], dtype=np.float64)
    keep = np.asarray(keep[:n], dtype=bool)
    out = {key: float("nan") for key in METRICS}
    out["n_valid_frames"] = int(keep.sum())
    if not keep.any():
        return out
    if times is None:
        times = np.arange(n, dtype=np.float64) / float(fps)
    else:
        times = np.asarray(times[:n], dtype=np.float64)
    p, g, t = pred[keep], gt[keep], times[keep]
    p_aligned, _ = pelvis_align(p)
    g_aligned, _ = pelvis_align(g)
    out["mpjpe_mm"] = float(mean_point_error(p_aligned, g_aligned, scale=1000.0).mean())
    try:
        out["pa_mpjpe_mm"] = float(pa_mpjpe(p, g).mean())
    except ValueError:
        pass
    out.update(root_trajectory_metrics(p[:, 0], g[:, 0]))
    out.pop("root_path_length_m", None)
    try:
        out["w_mpjpe100_mm"], out["wa_mpjpe100_mm"] = windowed_world_mpjpe(p, g, window=100)
    except ValueError:
        pass
    temporal = temporal_metrics(p, g, t)
    for key in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3"):
        out[key] = float(temporal[key])
    if pred_rotations is not None and gt_rotations is not None:
        pr = np.asarray(pred_rotations[:n])[keep]
        gr = np.asarray(gt_rotations[:n])[keep]
        if pr.shape[-2:] == (3, 3):
            out["mpjae_deg"] = float(matrix_rotation_error_degrees(pr, gr).mean())
        else:
            out["mpjae_deg"] = float(rotation_error_degrees(pr, gr).mean())
    if pred_vertices is not None and gt_vertices is not None:
        pv = np.asarray(pred_vertices[:n])[keep]
        gv = np.asarray(gt_vertices[:n])[keep]
        _, pv_aligned = pelvis_align(p, pv)
        _, gv_aligned = pelvis_align(g, gv)
        out["pve_mm"] = float(mean_point_error(pv_aligned, gv_aligned, scale=1000.0).mean())
        out["shape_vertex_std_mm"] = shape_vertex_std(pv)
        if pv.shape[1] > 6787:
            out["foot_sliding_mm"], _ = foot_sliding(pv, gv, t)
    if pred_contact is not None and gt_contact is not None:
        pc = np.asarray(pred_contact[:n], dtype=bool)[keep]
        gc = np.asarray(gt_contact[:n], dtype=bool)[keep]
        tp = int((pc & gc).sum())
        fp = int((pc & ~gc).sum())
        fn = int((~pc & gc).sum())
        tn = int((~pc & ~gc).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        out["contact_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-8)
        out["contact_acc"] = (tp + tn) / max(tp + fp + fn + tn, 1)
        out["contact_recall"] = recall
        out["air_recall"] = tn / max(tn + fp, 1)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = sum(int(r["n_valid_frames"]) for r in rows)
    for k in METRICS:
        vals = [(float(r[k]), int(r["n_valid_frames"])) for r in rows if math.isfinite(float(r[k]))]
        if vals: out[k] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    return out


# ---------------------------------------------------------------------------
# Mode registry: the single declaration point of which mode a model belongs
# to.  Rows whose mode cannot be resolved are refused by the comparison
# layer instead of being guessed.
# ---------------------------------------------------------------------------

def load_mode_registry(path: Path | None = None) -> dict[str, list[dict[str, str]]]:
    """Load ``models_modes.yaml`` -> ``{mode: [entry, ...]}``."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY
    text = registry_path.read_text(encoding="utf-8")
    try:
        import yaml
        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"mode registry must be a mapping: {registry_path}")
    unknown = sorted(set(raw) - set(MODES))
    if unknown:
        raise ValueError(f"mode registry has unknown modes {unknown} (allowed: {MODES}): {registry_path}")
    for mode, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"registry[{mode}] must be a list of model entries")
    return {mode: list(raw[mode]) for mode in MODES if mode in raw}


def resolve_mode_for(model: dict[str, str], registry: dict[str, list[dict[str, str]]]) -> str:
    """Mode of one comparison row; AnySole rows inherit it from config_id.

    Returns "" when the row's mode cannot be resolved (never a guess).
    """
    config_id = str(model.get("config_id") or "")
    if config_id in MODES:
        return config_id
    name = str(model.get("name") or "")
    for mode in MODES:
        for entry in registry.get(mode, []):
            if str(entry.get("name") or "") == name:
                return mode
    return ""


def common_edges_for(names: tuple[str, ...]) -> list[tuple[int, int]]:
    """COMMON_EDGES as index pairs over a common19-ordered joint list."""
    index = {name: i for i, name in enumerate(names)}
    return [(index[a], index[b]) for a, b in COMMON_EDGES if a in index and b in index]


def frame_mpjpe_mm(pred_common: np.ndarray, gt_common: np.ndarray) -> np.ndarray:
    """Per-frame pelvis-relative common19 MPJPE in mm (same formula as METRICS)."""
    pred_common = np.asarray(pred_common, dtype=np.float64)
    gt_common = np.asarray(gt_common, dtype=np.float64)
    n = min(pred_common.shape[0], gt_common.shape[0])
    ppa = pred_common[:n] - pred_common[:n, :1]
    gpa = gt_common[:n] - gt_common[:n, :1]
    return np.linalg.norm(ppa - gpa, axis=-1).mean(axis=-1) * 1000.0
