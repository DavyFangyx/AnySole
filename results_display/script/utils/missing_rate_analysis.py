"""Shared data definitions for the missing-rate/complementarity analyses.

The task book compares a small, fixed set of motion components.  This module
keeps their metric aliases and the error-direction conversion in one place so
the C1/C2/C3 and M1/M2 front ends cannot silently use different definitions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# component, display label, metric aliases, higher_is_better
# 别名按新名优先、旧名兜底排序：eval 明细文件与 rho_grid 网格的指标词汇不同源
# （eval 用 RootOrientation/foot_sliding_vertex_mm，grid 保留 root_orientation_deg/
# foot_sliding_mm），resolve_metric 取第一个存在的键，保证两边解析到同一指标。
COMPONENTS = {
    "upper": ("Upper body", ("PA-MPJPE_upper",), False),
    "hands": ("Hands", ("PA-MPJPE_hands",), False),
    "global_yaw": ("Global yaw", ("RootOrientation", "root_orientation_deg"), False),
    "root_traj": ("Root trajectory", ("root_rte_percent",), False),
    "contact": ("Contact timing", ("contact_mcc", "contact_f1", "contact_acc"), True),
    "support": ("Support foot stability", ("foot_sliding_vertex_mm", "foot_sliding_mm"), False),
    "foot_ground": ("Foot-ground relation", ("seam_jump_mm", "foot_sliding_vertex_mm"), False),
}

V_COMPONENTS = ("upper", "hands", "global_yaw", "root_traj")
T_COMPONENTS = ("contact", "support", "foot_ground")


def load_fseries(path: Path) -> dict[str, Any]:  # noqa: N802 — 读明细文件 metrics/<split>.json（旧称 fseries）
    if not path.is_file():
        raise FileNotFoundError("明细文件缺失：%s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError("不是明细文件（缺 metrics）：%s" % path)
    return data


def load_grid_prior(path: Path, cell: str = "rhoV0_rhoT0") -> dict[str, float]:
    """Read one rho-grid cell and aggregate seed values into a metric dict."""
    if not path.is_file():
        raise FileNotFoundError("rho 网格产物缺失：%s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", {})
    values = []
    for seed_cells in cells.values():
        metrics = seed_cells.get(cell)
        if metrics:
            values.append(metrics)
    if not values:
        raise ValueError("rho 网格中没有单元 %s：%s" % (cell, path))
    keys = set().union(*(item.keys() for item in values))
    return {key: float(np.mean([item[key] for item in values if key in item]))
            for key in keys if any(key in item for item in values)}


def resolve_metric(metrics: dict[str, Any], component: str) -> tuple[str, float, bool]:
    """Return (metric key, raw value, higher_is_better) for a component."""
    _label, aliases, higher = COMPONENTS[component]
    for key in aliases:
        if key in metrics and metrics[key] is not None:
            return key, float(metrics[key]), higher
    raise KeyError("分量 %s 缺少指标（候选：%s）" % (component, ", ".join(aliases)))


def error_value(metrics: dict[str, Any], component: str) -> tuple[str, float, float]:
    """Return metric key, raw value, and lower-is-better error value."""
    key, raw, higher = resolve_metric(metrics, component)
    return key, raw, -raw if higher else raw


def component_label(component: str) -> str:
    return COMPONENTS[component][0]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("没有可写出的分析行：%s" % path)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
