"""Local integration layer for the pinned WHAM research baseline."""

from .config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_GPU_IDS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_WHAM_PYTHON,
    DEFAULT_WHAM_ROOT,
    DEFAULT_WORKERS_PER_GPU,
    WORKSPACE_ROOT,
)

__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_GPU_IDS",
    "DEFAULT_RESULT_ROOT",
    "DEFAULT_WHAM_PYTHON",
    "DEFAULT_WHAM_ROOT",
    "DEFAULT_WORKERS_PER_GPU",
    "WORKSPACE_ROOT",
]

__version__ = "0.1.0"
