"""Import/path setup for running WHAM code from the shared repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WHAM_ROOT = ROOT / "Baselines" / "WHAM"


def activate() -> Path:
    """Make WHAM's top-level ``configs``/``lib`` packages win over repo ones."""
    import AnysoleWorkspace.tool.workspace  # noqa: F401  (preload before root removal)

    root = str(ROOT.resolve())
    sys.path[:] = [
        entry for entry in sys.path
        if not entry or Path(entry).expanduser().resolve() != ROOT.resolve()
    ]
    module = sys.modules.get("configs")
    if module is not None and not str(getattr(module, "__file__", "")).startswith(
        str(WHAM_ROOT)
    ):
        del sys.modules["configs"]
    if str(WHAM_ROOT) in sys.path:
        sys.path.remove(str(WHAM_ROOT))
    sys.path.insert(0, str(WHAM_ROOT))
    os.chdir(WHAM_ROOT)
    return WHAM_ROOT
