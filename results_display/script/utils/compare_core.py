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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # script/ (for utils.*)
from utils import cli_common  # noqa: E402

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anysole.types import JOINT_NAMES  # noqa: E402
from utils.motion_io import LEGACY_BVH_NAMES, load_motion  # noqa: E402

METRICS = ("MPJPE_mm", "PA_MPJPE_mm", "WMPJPE_mm", "WAMPJPE_mm", "RTE_mm", "Accel_mps2", "Jitter_1e-3_mps2")

# Generation modes.  Rows from different modes are never compared in one
# table block: T2M vs V2M is a different task, not a different model.
MODES = ("VT2M", "V2M", "T2M")
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


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    opener = path.open(encoding="utf-8-sig")
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in opener if line.strip()]
    else:
        rows = list(csv.DictReader(opener))
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
        with np.load(path, allow_pickle=True) as probe:
            if ({"poses", "trans"}.issubset(probe.files)
                    or {"root_orient", "pose_body", "trans"}.issubset(probe.files)):
                motion = load_motion(path)
                return motion["joints"], None, tuple(motion["names"]), "smpl24"
        data = np.load(path, allow_pickle=False)
        keys = list(data.keys())
        candidates = ("joint_xyz_world", "joints_world", "pred_joints", "joints", "xyz", "pose")
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


def procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    xs, ys = src.mean(0), dst.mean(0)
    a, b = src - xs, dst - ys
    u, s, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = s.sum() / max((a * a).sum(), 1e-12)
    return scale * (src - xs) @ r + ys


def metrics(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray, fps: float) -> dict[str, float]:
    n = min(len(pred), len(gt), len(keep)); pred, gt, keep = pred[:n], gt[:n], keep[:n]
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = int(keep.sum())
    if not keep.any(): return out
    ppa, gpa = pred - pred[:, :1], gt - gt[:, :1]
    out["MPJPE_mm"] = float(np.linalg.norm(ppa[keep] - gpa[keep], axis=-1).mean() * 1000)
    pa = [np.linalg.norm(procrustes(pred[t], gt[t]) - gt[t], axis=-1).mean() for t in np.flatnonzero(keep)]
    out["PA_MPJPE_mm"] = float(np.mean(pa) * 1000)
    ids = np.flatnonzero(keep)
    if len(ids) >= 2:
        ref = ids[:2]; aligned = pred.copy(); aligned = pred - pred[ref].mean(0) + gt[ref].mean(0)
        err = np.linalg.norm(aligned[keep] - gt[keep], axis=-1).mean()
        out["WMPJPE_mm"] = float(err * 1000)
    if len(ids) >= 1:
        aligned = pred - pred[keep].mean((0, 1)) + gt[keep].mean((0, 1))
        out["WAMPJPE_mm"] = float(np.linalg.norm(aligned[keep] - gt[keep], axis=-1).mean() * 1000)
        out["RTE_mm"] = float(np.linalg.norm((aligned[ids[-1], 0] - gt[ids[-1], 0])) * 1000)
    if n >= 3:
        triple = keep[:-2] & keep[1:-1] & keep[2:]
        ap = (pred[:-2] - 2 * pred[1:-1] + pred[2:]) * fps ** 2
        ag = (gt[:-2] - 2 * gt[1:-1] + gt[2:]) * fps ** 2
        if triple.any():
            out["Accel_mps2"] = float(np.linalg.norm(ap[triple] - ag[triple], axis=-1).mean())
            out["Jitter_1e-3_mps2"] = float(np.linalg.norm(ap[triple], axis=-1).mean() * 1000)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = sum(int(r["n_valid_frames"]) for r in rows)
    for k in METRICS:
        vals = [(float(r[k]), int(r["n_valid_frames"])) for r in rows if math.isfinite(float(r[k]))]
        if vals: out[k] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    vals = [float(r["RTE_mm"]) for r in rows if math.isfinite(float(r["RTE_mm"]))]
    out["RTE_mm"] = float(np.mean(vals)) if vals else float("nan")
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
