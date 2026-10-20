"""Handoff pressure variant (2026-10-17 Tactile_pressure_handoff v1).

``tactile_input="denoised108"`` 读取交接包的 baseline_step_corrected 值
（``values48`` = processed48）并映射到共享帧网格。映射 = build_shared 的
绝对最近行规则原样套在交接包行钟上：

- frame i wall = ``video_start_day_s + i/fps``，其中 video_start_day_s =
  源录像 cam{N} 目录首张 jpeg 文件名的日秒（build_shared.build_one 同口径）；
- row wall = ``wall_time_day_s``（= started_at_iso 日秒 + 原生接收钟，
  与 build_shared 的 pressure_epoch + raw_t_us/1e6 同一约定，S2 §三 证据 C）。

最近行、无插值、边缘钳位（build_shared.resample_pressure 同语义）。
验证（S13011，2026-10-17）：与 final_fake_marked CSV 的 raw_t_us 映射
对照，双脚 97.5%+ 帧选中行时间差 ≤ 30ms（≈ 一行间隔）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# v1 = 四项处理（孤立突变/悬空底值/持续阶跃/自适应平滑）；
# v2 = 前三项（无平滑）；v3 = v2 三项 + 故障格突发清洗（基线之前，
# raw 行上，2026-10-20）。各包 pressure_io.load_pressure 签名一致，
# 默认 version='baseline_step_corrected' → values48 = processed48。
HANDOFF_PACKAGES = {
    "v1": REPO / "z_note" / "data_pressure_fixpipe"
    / "Tactile_pressure_handoff_20261017_v1",
    "v2": REPO / "z_note" / "data_pressure_fixpipe"
    / "Tactile_pressure_handoff_20261017_v2_no_smoothing",
    "v3": REPO / "z_note" / "data_pressure_fixpipe"
    / "Tactile_pressure_handoff_20261020_v3",
}
HANDOFF_PACKAGE = HANDOFF_PACKAGES["v1"]
# pressure_io 的 root = data/（其下 file_manifest.csv + dataset/…npz）。
HANDOFF_ROOT = HANDOFF_PACKAGE / "data"

_ios = {}


def handoff_root(package: str = "v1") -> Path:
    """交接包 data/ 目录（pressure_io.load_pressure 的 root）。"""
    return HANDOFF_PACKAGES[package] / "data"


def _pressure_io(package: str = "v1"):
    if package not in HANDOFF_PACKAGES:
        raise ValueError("unknown handoff package %r" % package)
    if package not in _ios:
        spec = importlib.util.spec_from_file_location(
            "handoff_pressure_io_" + package,
            HANDOFF_PACKAGES[package] / "pipeline" / "pressure_io.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ios[package] = mod
    return _ios[package]


def day_seconds_from_jpeg_name(name: str) -> float:
    """Local day-seconds from a frame name ``<cam>_HHMMSS.mmm.jpg``."""
    core = name.split("_")[-1][:-4]  # strip ".jpg"
    return int(core[0:2]) * 3600 + int(core[2:4]) * 60 + float(core[4:])


def video_start_for(session_dir: Path) -> float:
    """源录像 cam{N} 目录首张 jpeg 的日秒（build_shared 同口径）。"""
    from anysole.data.workspace_adapter import resolve_uri

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    recording = resolve_uri(meta["source_files"]["rgb"], must_exist=True)
    cam = str(meta.get("camera", "cam3")).lstrip("cam") or "3"
    images = sorted((recording / cam).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"{session_dir}: raw RGB cam{cam} has no extracted frames")
    return day_seconds_from_jpeg_name(images[0].name)


def resample_handoff(session_id: str, date: str, subject: str,
                     video_start: float, n_frames: int, *,
                     fps: float = 40.0,
                     package: str = "v1",
                     root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame-aligned handoff pressure (left48, right48), shape (n_frames, 48).

    ``package`` 选 v1 / v2 / v3 交接包；``root`` 缺省时按 package 取对应包
    data/（调用方建议显式传 root，避免依赖定义时绑定的旧默认值）。
    """
    io = _pressure_io(package)
    if root is None:
        root = handoff_root(package)
    target = video_start + np.arange(n_frames, dtype=np.float64) / fps
    out = {}
    for side in ("left", "right"):
        p = io.load_pressure(str(root), session_id, side)
        wall = np.asarray(p["wall_time_day_s"], dtype=np.float64)
        idx = np.clip(np.searchsorted(wall, target), 1, len(wall) - 1)
        idx = np.where(
            np.abs(wall[idx - 1] - target) <= np.abs(wall[idx] - target),
            idx - 1, idx,
        )
        out[side] = np.asarray(p["values48"], dtype=np.float32)[idx]
    return out["left"], out["right"]
