"""Load 48-cell insoles and derive the frozen 12-D physical tokens."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from anysole.types import FAKE_MARKED_ROOT, PRESSURE_CLIP, T_PHYS_DIM, T_RAW_DIM


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
    dx = np.abs(np.diff(grid, axis=1, prepend=grid[:, :1, :])).mean(axis=(1, 2), keepdims=True)
    dy = np.abs(np.diff(grid, axis=2, prepend=grid[:, :, :1])).mean(axis=(1, 2), keepdims=True)
    spatial = (dx + dy) / 2.0
    temporal = np.diff(force[:, 0], prepend=force[0, 0])[:, None].astype(np.float32)
    return np.concatenate([cop, force.astype(np.float32), envelope, spatial.astype(np.float32), temporal], axis=1)


def physical_tokens(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    phys = np.concatenate([_foot_phys(left48), _foot_phys(right48)], axis=1)
    if phys.shape[1] != T_PHYS_DIM:
        raise ValueError("T_phys dim %d != %d" % (phys.shape[1], T_PHYS_DIM))
    return phys.astype(np.float32)


def load_session_pressure(meta: dict, t_grid: np.ndarray) -> dict:
    rec_dir = FAKE_MARKED_ROOT / meta["date"] / meta["subject"] / meta["rec_name"]
    left = load_pressure_csv(rec_dir / "pressure_left.csv")
    right = load_pressure_csv(rec_dir / "pressure_right.csv")
    t_grid_us = t_grid * 1e6
    left48 = interpolate_values(left["t_us"], left["values"], t_grid_us)
    right48 = interpolate_values(right["t_us"], right["values"], t_grid_us)
    t_raw = np.concatenate([left48, right48], axis=1)
    if t_raw.shape[1] != T_RAW_DIM:
        raise ValueError("T_raw dim %d != %d" % (t_raw.shape[1], T_RAW_DIM))
    return {
        "T_raw": t_raw.astype(np.float32),
        "T_phys": physical_tokens(left48, right48),
        "left48": left48,
        "right48": right48,
    }


def normalize_raw(t_raw: np.ndarray) -> np.ndarray:
    return np.clip(t_raw, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
