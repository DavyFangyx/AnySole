"""Resolve centralized Anysole workspace paths."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = Path(os.environ.get("ANYSOLE_WORKSPACE", REPO_ROOT / "AnysoleWorkspace")).expanduser()
RESULTS_ROOT = Path(os.environ.get("ANYSOLE_RESULTS", REPO_ROOT / "results")).expanduser()
DISPLAY_ROOT = Path(os.environ.get("ANYSOLE_RESULTSDISPLAY", REPO_ROOT / "results_display")).expanduser()

PREFIXES = {
    "workspace://": WORKSPACE_ROOT,
    "results://": RESULTS_ROOT,
    "display://": DISPLAY_ROOT,
}

LEGACY_PREFIXES = {
    "data/exp/checkpoint": RESULTS_ROOT / "MotionPRO/checkpoints",
    "data/exp/result": RESULTS_ROOT / "MotionPRO/metrics",
    "data/exp/viz_compare": DISPLAY_ROOT / "MotionPRO",
    "data/tensorboard": RESULTS_ROOT / "MotionPRO/tensorboard",
    "data/sequences": WORKSPACE_ROOT / "derived/MotionPRO/sequences",
    "data/splits": WORKSPACE_ROOT / "splits/default",
    "data/smpl": WORKSPACE_ROOT / "dependencies/smpl",
    "data/cliff_ckpt": WORKSPACE_ROOT / "dependencies/MotionPRO/cliff_ckpt",
    "data/mmdetection": WORKSPACE_ROOT / "dependencies/MotionPRO/mmdetection",
}


def _resolve_raw_bvh(path: Path) -> Path:
    """Resolve BVHs recorded before the workspace was reorganized."""
    marker = "/mocap_ori_bvh/"
    normalized = path.as_posix()
    if marker not in normalized:
        return path
    suffix = normalized.split(marker, 1)[1]
    matches = sorted((WORKSPACE_ROOT / "sources/raw").glob("*/mocap_ori_bvh/" + suffix))
    return matches[0] if matches else path


def resolve_path(value, base_dir=None) -> str:
    text = str(value or "")
    for prefix, root in PREFIXES.items():
        if text.startswith(prefix):
            return str(root / text[len(prefix):])
    normalized = text.replace("\\", "/").rstrip("/")
    for prefix, root in LEGACY_PREFIXES.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return str(root / normalized[len(prefix):].lstrip("/"))
    path = Path(os.path.expandvars(text)).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(_resolve_raw_bvh(path))
    return str(_resolve_raw_bvh(Path(base_dir) / path))


def sequence_root(cam_id=3) -> Path:
    return WORKSPACE_ROOT / "derived/MotionPRO/sequences" / f"cam{int(cam_id)}"
