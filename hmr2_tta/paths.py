"""HMR2.0 实验线的路径与 URI 解析（只读复用 AnysoleWorkspace 的 URI 契约）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/data/fangyuxuan/projects/gait")
HMR2_ROOT = PROJECT_ROOT / "z_note/HMR2.0"
UPSTREAM_ROOT = HMR2_ROOT / "4D-Humans"          # 官方实现（只读，仅 2 处必要改动）
PACKAGE_ROOT = PROJECT_ROOT / "hmr2_tta"           # 本实验线新代码
RESULTS_ROOT = PROJECT_ROOT / "results/baselines/HMR2.0"
CACHE_4DHUMANS = Path.home() / ".cache/4DHumans"
SMPL_MODEL_PATH = PROJECT_ROOT / "AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl"

# 上游导入（hmr2 包、anysole 数据读取共用）
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_uri(value: str) -> Path:
    """raw:// work:// shared:// 等 URI → 绝对路径（复用 AnysoleWorkspace 实现）。"""
    from AnysoleWorkspace.tool.workspace import resolve_uri as _resolve
    return _resolve(value, for_write=False, must_exist=True)


def facts_session_dir(sid: str, cam: int) -> Path:
    """按 session_id 全局检索 facts 会话目录（{date}/{subject} 由树内定位）。"""
    for sub in sorted((PROJECT_ROOT / "AnysoleWorkspace/shared/facts/sessions").glob(f"cam{cam}/*/*/{sid}")):
        return sub
    raise FileNotFoundError(f"facts session not found: {sid} cam{cam}")


def bbox_path(sid: str, cam: int) -> Path:
    root = PROJECT_ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet"
    sub = "bbox" if cam == 3 else f"bbox_cam{cam}"
    return root / sub / f"{sid}.npy"


def calibration_json(date: str) -> Path:
    return PROJECT_ROOT / "AnysoleWorkspace/protocol/calibration" / f"{date}.json"
