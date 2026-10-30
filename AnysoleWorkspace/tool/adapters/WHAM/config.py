"""Stable workspace defaults for the WHAM baseline integration."""

from __future__ import annotations

import os
from pathlib import Path


# 决策 11（2026-10-30 收口）：项目布局为默认运行布局（包删除后照常）；
# VB_WORKSPACE_ROOT 保留源侧包布局兼容。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if os.environ.get("VB_WORKSPACE_ROOT"):
    WORKSPACE_ROOT = Path(os.environ["VB_WORKSPACE_ROOT"])
    DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "1_Dataset"
    DEFAULT_WHAM_ROOT = WORKSPACE_ROOT / "third_party" / "WHAM"
    DEFAULT_RESULT_ROOT = WORKSPACE_ROOT / "3_Result" / "WHAM"
else:
    WORKSPACE_ROOT = _PROJECT_ROOT
    DEFAULT_DATASET_ROOT = Path("/data/lizhe/projects/Tactile")
    DEFAULT_WHAM_ROOT = _PROJECT_ROOT / "Baselines" / "WHAM"
    DEFAULT_RESULT_ROOT = _PROJECT_ROOT / "AnysoleWorkspace" / "work" / "visual_baselines"
DEFAULT_WHAM_PYTHON = Path(os.environ.get("VB_WHAM_PYTHON", "python"))
DEFAULT_GPU_IDS = tuple(range(7))
DEFAULT_WORKERS_PER_GPU = 2


def runtime_paths() -> dict[str, Path]:
    """Return the operator-visible paths without creating or mutating them."""

    return {
        "workspace_root": WORKSPACE_ROOT,
        "dataset_root": DEFAULT_DATASET_ROOT,
        "wham_root": DEFAULT_WHAM_ROOT,
        "result_root": DEFAULT_RESULT_ROOT,
        "wham_python": DEFAULT_WHAM_PYTHON,
    }
