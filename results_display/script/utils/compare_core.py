"""Display-layer comparison core for the R_Test1/2/3 comparison views.

The evaluation core — data protocols, GT loading, capability gating and the
protocol-aware metric assembly — lives in ``Baselines/utils`` (2026-09-26
结构裁定: 基线侧数据/评估协议统一放入 Baselines/utils; 公式唯一实现在
``anysole/utils/metrics.py``).  This module is the display-side consumer
surface: it re-exports the core interfaces so R_Test scripts keep their import
paths, and keeps only display-specific machinery here (mode registry loading,
panel colors, prediction discovery, diagnostic contact helpers).

results_display consumes baseline metrics produced by the baseline evaluation
entry (``python -m Baselines.utils.evaluate``) — it is not the place where
baseline metrics are produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # script/ (for utils.*)
from utils import cli_common  # noqa: E402

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- evaluation core re-exports (canonical homes in Baselines/utils) ----
from anysole.utils.metrics import MOTION_METRIC_NAMES  # noqa: E402
from Baselines.utils.capabilities import (  # noqa: E402
    CAPABILITY_ALIASES,
    CAPABILITY_FIELDS,
    DENIED_SOURCES,
    METRIC_REQUIRES,
    applicable_for,
    applicable_metrics,
    denied_capabilities,
    mask_metrics,
    read_capabilities,
)
from Baselines.utils.gt_loading import (  # noqa: E402
    array_from_file,
    archive_meta,
    bvh_joints,
    find_prediction,
    load_split_ids,
    parse_list,
    protocol_gt,
    protocol_gt_rotations,
    protocol_gt_surface,
    read_manifest,
    resolve_repo_path,
    sha256,
    valid_mask,
)
from Baselines.utils.motion_io import LEGACY_BVH_NAMES, load_motion, load_rotations, load_surface  # noqa: E402
from Baselines.utils.protocols import (  # noqa: E402
    PROTOCOL_FORWARD_AXIS,
    PROTOCOL_FOOT_NAMES,
    PROTOCOL_LABELS,
    PROTOCOL_PELVIS,
    PROTOCOLS,
    native_edges,
    native_frame_mpjpe_mm,
    normalize_protocol,
)
from Baselines.utils.solver import (  # noqa: E402
    COMPARISON_SCHEMA_V2,
    LEGACY_SCHEMA_V1,
    METRICS,
    PRESSURE_GRID_SHAPE,
    V2T_BRIEF_KEYS,
    V2T_LEAF_KEYS,
    V2T_LEVELS,
    V2T_METRICS,
    aggregate,
    canonical_pressure_feet,
    contact_smpl_metrics,
    load_v2t_archive,
    metrics,
    pressure_metrics,
    v2t_metrics,
)

# ---- display-specific remainder ----

# Generation modes.  Rows from different modes are never compared in one
# table block: T2M vs V2M is a different task, not a different model.
MODES = ("VT2M", "V2M", "T2M", "V2T")
MOTION_MODES = ("VT2M", "V2M", "T2M")
MODE_METRICS = {
    "VT2M": MOTION_METRIC_NAMES,
    "V2M": MOTION_METRIC_NAMES,
    "T2M": MOTION_METRIC_NAMES,
    # V2T tables show the brief keys only; leaf keys stay in the detail
    # rows / comparison JSON (see Baselines.utils.solver.V2T_LEAF_KEYS).
    "V2T": V2T_BRIEF_KEYS,
}
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "models_modes.yaml"

# Panel title colors for the comparison views: main model keeps the standard
# prediction blue, GT stays orange, baselines get one hue each (registry).
MAIN_COLOR = (80, 200, 255)
GT_COLOR = (255, 170, 80)
MISSING_COLOR = (120, 120, 128)
FALLBACK_BASELINE_COLORS = ((138, 92, 230), (31, 157, 138), (194, 91, 31), (91, 122, 194))


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


def load_contact_gt(row: dict[str, str], n: int, method: str = "joint_and") -> np.ndarray | None:
    """Load GT contact labels — diagnostic input only.

    Contact metrics are not formal (评估整改任务 01 §4.4): no motion model
    trains and exports an explicit contact prediction, and heuristically
    derived contact can never feed a formal ``contact_f1``.  This loader and
    :func:`contact_from_joints` remain for visualization/diagnosis only.
    """
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
    """AnySole soft-contact heuristic in the shared z-up frame — diagnostic only.

    Never a source of formal contact metrics (see :func:`load_contact_gt`).
    """
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
