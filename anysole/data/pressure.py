"""Load 48-cell insoles and derive the frozen 12-D physical tokens."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from anysole.types import CONTACT_SUM_THRESH, FAKE_MARKED_ROOT, PRESSURE_CLIP, T_PHYS_DIM, T_RAW_DIM


def load_pressure_csv(path: Path) -> dict:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Empty pressure CSV: %s" % path)
        value_cols = [name for name in reader.fieldnames if name.isdigit()]
        t_us, fake, valid, values = [], [], [], []
        for row in reader:
            t_us.append(float(row["t_us"]))
            fake.append(int(float(row.get("fake", 0))))
            valid.append(int(float(row.get("valid_mask", 1))))
            values.append([float(row[col]) for col in value_cols])
    values_np = np.asarray(values, dtype=np.float32) if values else np.zeros((0, 48), dtype=np.float32)
    if values_np.ndim != 2 or values_np.shape[1] != 48:
        raise ValueError("Expected 48 pressure cells in %s, got %s" % (path, values_np.shape))
    return {
        "t_us": np.asarray(t_us, dtype=np.float64),
        "fake": np.asarray(fake, dtype=np.uint8),
        "valid": np.asarray(valid, dtype=np.uint8),
        "values": values_np,
    }


def interpolate_values(source_t: np.ndarray, values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((query_t.size, 48), dtype=np.float32)
    out = np.empty((query_t.size, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(query_t, source_t, values[:, col])
    return out


def nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0:
        raise ValueError("Empty source timeline")
    idx = np.clip(np.searchsorted(source_t, query_t), 1, source_t.size - 1)
    left = idx - 1
    pick_right = np.abs(source_t[idx] - query_t) < np.abs(source_t[left] - query_t)
    return np.where(pick_right, idx, left)


def cop_from_grid(values48: np.ndarray) -> np.ndarray:
    grid = np.clip(values48.reshape(-1, 4, 12), 0.0, None)
    force = grid.sum(axis=(1, 2))
    rows = np.arange(4, dtype=np.float32)
    cols = np.arange(12, dtype=np.float32)
    cop_x = (grid.sum(axis=2) * rows).sum(axis=1) / np.maximum(force, 1e-6) / 3.0
    heel_to_toe = 11.0 - (grid.sum(axis=1) * cols).sum(axis=1) / np.maximum(force, 1e-6)
    cop_y = heel_to_toe / 11.0
    cop = np.stack([cop_x, cop_y], axis=1).astype(np.float32)
    cop[force <= 0] = 0.0
    return cop


def _foot_phys(values48: np.ndarray) -> np.ndarray:
    force = np.clip(values48, 0.0, None).sum(axis=1, keepdims=True)
    cop = cop_from_grid(values48)
    padded = np.pad(force[:, 0], (1, 1), mode="edge")
    envelope = np.maximum.reduce([padded[:-2], padded[1:-1], padded[2:]]).astype(np.float32)[:, None]
    grid = np.clip(values48.reshape(-1, 4, 12), 0.0, None)
    dx = np.abs(np.diff(grid, axis=1, prepend=grid[:, :1, :])).mean(axis=(1, 2))[:, None]
    dy = np.abs(np.diff(grid, axis=2, prepend=grid[:, :, :1])).mean(axis=(1, 2))[:, None]
    spatial = (dx + dy) / 2.0
    temporal = np.diff(force[:, 0], prepend=force[0, 0])[:, None].astype(np.float32)
    return np.concatenate([cop, force.astype(np.float32), envelope, spatial.astype(np.float32), temporal], axis=1)


HULL_THRESH = 0.05   # 与 foot_encoder.CONTACT_AREA_THRESH 同阈值


def _hull_feats(values48: np.ndarray) -> np.ndarray:
    """每足 3 维着地几何（维度 2，2026-10-21）：激活格的沿足长位置
    （0=跟、1=尖——脚尖先/脚跟先着地的显式编码）、内外侧位置、沿足长
    散布（平铺 vs 点接触）。零载荷帧全零。"""
    grid = np.clip(values48.reshape(-1, 4, 12), 0.0, None)
    active = grid > HULL_THRESH
    cnt = active.sum(axis=(1, 2))
    rows = np.arange(4, dtype=np.float32)[None, :, None]
    cols = np.arange(12, dtype=np.float32)[None, None, :]
    mean_row = (active * rows).sum(axis=(1, 2)) / np.maximum(cnt, 1.0)
    mean_col = (active * cols).sum(axis=(1, 2)) / np.maximum(cnt, 1.0)
    var_col = (active * (cols - mean_col[:, None, None]) ** 2).sum(
        axis=(1, 2)) / np.maximum(cnt, 1.0)
    out = np.stack([mean_col / 11.0, mean_row / 3.0,
                    np.sqrt(var_col) / 11.0], axis=1)
    out[cnt == 0] = 0.0
    return out.astype(np.float32)


def _foot_phys_v2(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    """维度 2 显式物理特征（2026-10-21）：基础 12 维 + 每足着地几何 3 维
    + 双足关系 2 维（力比、CoP 距离）= 20 维。"""
    base = np.concatenate([_foot_phys(left48), _foot_phys(right48)], axis=1)
    lh = _hull_feats(left48)
    rh = _hull_feats(right48)
    lf = np.clip(left48, 0.0, None).sum(axis=1)
    rf = np.clip(right48, 0.0, None).sum(axis=1)
    ratio = lf / np.maximum(lf + rf, 1e-6)
    cl = cop_from_grid(left48)
    cr = cop_from_grid(right48)
    dist = np.linalg.norm(cl - cr, axis=1)
    return np.concatenate([base, lh, rh, ratio[:, None], dist[:, None]],
                          axis=1).astype(np.float32)


def physical_tokens(left48: np.ndarray, right48: np.ndarray,
                    t_phys_v2: bool = False) -> np.ndarray:
    if t_phys_v2:
        return _foot_phys_v2(left48, right48)
    phys = np.concatenate([_foot_phys(left48), _foot_phys(right48)], axis=1)
    if phys.shape[1] != T_PHYS_DIM:
        raise ValueError("T_phys dim %d != %d" % (phys.shape[1], T_PHYS_DIM))
    return phys.astype(np.float32)


def load_session_pressure(meta: dict, t_grid: np.ndarray,
                          t_phys_v2: bool = False) -> dict:
    rec_dir = FAKE_MARKED_ROOT / meta["date"] / meta["subject"] / meta["rec_name"]
    left = load_pressure_csv(rec_dir / "pressure_left.csv")
    right = load_pressure_csv(rec_dir / "pressure_right.csv")
    t_grid_us = t_grid * 1e6
    left48 = interpolate_values(left["t_us"], left["values"], t_grid_us)
    right48 = interpolate_values(right["t_us"], right["values"], t_grid_us)
    t_raw = np.concatenate([left48, right48], axis=1)
    if t_raw.shape[1] != T_RAW_DIM:
        raise ValueError("T_raw dim %d != %d" % (t_raw.shape[1], T_RAW_DIM))
    # T_phys must share T_raw's scale: derive physical tokens from the
    # normalized pressures (same formula as normalize_raw: clip to
    # PRESSURE_CLIP, then divide by PRESSURE_CLIP).
    norm_left = np.clip(left48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
    norm_right = np.clip(right48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
    return {
        "T_raw": t_raw.astype(np.float32),
        "T_phys": physical_tokens(norm_left, norm_right, t_phys_v2=t_phys_v2),
        "left48": left48,
        "right48": right48,
    }


def normalize_raw(t_raw: np.ndarray) -> np.ndarray:
    return np.clip(t_raw, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
