#!/usr/bin/env python3
"""Compatibility entry point for building and checking AnysoleWorkspace."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from manage_anysole_workspace import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
