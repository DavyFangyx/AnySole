"""Re-export shim: the canonical unified motion reader lives in
``Baselines/utils/motion_io.py`` (2026-09-26 结构裁定：基线侧数据协议
统一放入 Baselines/utils)。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # utils/ -> script/ -> results_display/ -> root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Baselines.utils.motion_io import *  # noqa: F401,F403
from Baselines.utils.motion_io import LEGACY_BVH_NAMES  # noqa: F401
