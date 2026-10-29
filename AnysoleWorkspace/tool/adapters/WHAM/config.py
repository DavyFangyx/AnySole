"""Stable workspace defaults for the WHAM baseline integration."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("VB_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "1_Dataset"
DEFAULT_WHAM_ROOT = WORKSPACE_ROOT / "third_party" / "WHAM"
DEFAULT_RESULT_ROOT = WORKSPACE_ROOT / "3_Result" / "WHAM"
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
