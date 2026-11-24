"""项目路径符号（仓库相对，无机器硬编码）。"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def facts_session_dir(sid: str, cam: int) -> Path:
    """按 session_id 全局检索 facts 会话目录（{date}/{subject} 由树内定位）。"""
    for sub in sorted((PROJECT_ROOT / "AnysoleWorkspace/shared/facts/sessions").glob(f"cam{cam}/*/*/{sid}")):
        return sub
    raise FileNotFoundError(f"facts session not found: {sid} cam{cam}")
