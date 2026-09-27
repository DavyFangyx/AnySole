"""Re-export shim: the canonical BVH pose conversion lives in
``Baselines/utils/bvh_aligner.py`` (2026-09-26 结构裁定：基线侧数据协议
统一放入 Baselines/utils)。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # utils/ -> script/ -> results_display/ -> root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Baselines.utils.bvh_aligner import parse_bvh_aligner  # noqa: F401
