"""Shared data definitions for the missing-rate/complementarity analyses.

The task book compares a fixed set of body-region error rows (2026-10-07 起
按区域分组：上半身/下半身/全身/足-地，替代原 7 分量表).  This module keeps
their metric aliases and the error-direction conversion in one place so the
C1/C2/C3 and M1/M2 front ends cannot silently use different definitions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# region, region_label, rows: (row_id, label, metric aliases, higher_is_better)
# 别名按新名优先、旧名兜底排序：eval 明细文件与 rho_grid 网格的指标词汇不同源
# （eval 用显示名 RootOrientation/AccelError/PA-MPJPE，grid 用 snake_case
# root_orientation_deg/accel_error_m_s2/pa_mpjpe_mm；区域/部位键两端同名），
# resolve_metric 取第一个存在的键，保证两边解析到同一指标。
REGION_GROUPS: tuple = (
    ("upper", "Upper body (PA-MPJPE mm)", (
        ("upper", "Upper body", ("PA-MPJPE_upper",), False),
        ("hands", "Hands", ("PA-MPJPE_hands",), False),
        ("headneck", "Head+neck", ("PA-MPJPE_part_headneck",), False),
        ("torso", "Torso", ("PA-MPJPE_part_torso",), False),
        ("l_arm", "L arm", ("PA-MPJPE_part_l_arm",), False),
        ("r_arm", "R arm", ("PA-MPJPE_part_r_arm",), False),
    )),
    ("lower", "Lower body (PA-MPJPE mm)", (
        ("lower", "Lower body", ("PA-MPJPE_lower",), False),
        ("anklefoot", "Ankle+foot", ("PA-MPJPE_anklefoot",), False),
        ("l_leg", "L leg", ("PA-MPJPE_part_l_leg",), False),
        ("r_leg", "R leg", ("PA-MPJPE_part_r_leg",), False),
        ("l_foot", "L foot", ("PA-MPJPE_part_l_foot",), False),
        ("r_foot", "R foot", ("PA-MPJPE_part_r_foot",), False),
    )),
    ("full", "Full body (mixed units)", (
        ("overall", "Overall PA-MPJPE (mm)", ("PA-MPJPE", "pa_mpjpe_mm"), False),
        ("accel", "AccelError (m/s²)", ("AccelError", "accel_error_m_s2"), False),
        ("root_traj", "Root trajectory (%)", ("root_rte_percent",), False),
        ("root_orient", "Root orientation (°)", ("RootOrientation", "root_orientation_deg"), False),
    )),
    ("foot", "Foot-ground (mixed units)", (
        ("contact", "Contact (higher better)", ("contact_f1", "contact_mcc", "contact_acc"), True),
        ("sliding", "Foot sliding (mm)", ("foot_sliding_vertex_mm", "foot_sliding_mm"), False),
        ("seam", "Seam jump (mm)", ("seam_jump_mm",), False),
    )),
)

# V 域 = 上半身 + 全身（对应任务书旧 V 负责：上肢/手/朝向/根轨迹）
# T 域 = 下半身 + 足-地（旧 T 负责：接触/支撑/足-地，扩展至下肢）
V_DOMAIN_ROWS = frozenset(
    row_id for region, _, rows in REGION_GROUPS if region in ("upper", "full")
    for row_id, _, _, _ in rows
)
T_DOMAIN_ROWS = frozenset(
    row_id for region, _, rows in REGION_GROUPS if region in ("lower", "foot")
    for row_id, _, _, _ in rows
)


def row_specs(domains: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Flat row spec list (all rows, or filtered to region/domain names)."""
    out: list[dict[str, Any]] = []
    for region, region_label, rows in REGION_GROUPS:
        if domains and region not in domains:
            continue
        for row_id, label, aliases, higher in rows:
            out.append({
                "row_id": row_id, "region": region, "region_label": region_label,
                "label": label, "aliases": aliases, "higher": higher,
            })
    return out


def row_specs_in_domain(domain_rows: frozenset[str]) -> list[dict[str, Any]]:
    return [row for row in row_specs() if row["row_id"] in domain_rows]


def load_fseries(path: Path) -> dict[str, Any]:  # noqa: N802 — 读明细文件 metrics/<split>.json（旧称 fseries）
    if not path.is_file():
        raise FileNotFoundError("明细文件缺失：%s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError("明细文件缺 metrics：%s" % path)
    return data


def load_grid_prior(path: Path, cell: str = "rhoV0_rhoT0") -> dict[str, Any]:
    """ρ 网格空输入先验：指定 cell 跨种子取平均（指标同明细文件口径）。"""
    if not path.is_file():
        raise FileNotFoundError("ρ 网格产物缺失：%s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", {})
    values: dict[str, list[float]] = {}
    for metrics in cells.values():
        entry = metrics.get(cell, {})
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(value)
    return {key: float(np.mean(vals)) for key, vals in values.items()}


def resolve_metric(metrics: dict[str, Any], aliases: tuple[str, ...], higher_is_better: bool) -> tuple[str, float, bool]:
    for alias in aliases:
        if alias in metrics:
            return alias, float(metrics[alias]), higher_is_better
    raise KeyError("行缺少指标（候选：%s）" % ", ".join(aliases))


def error_value(metrics: dict[str, Any], aliases: tuple[str, ...], higher_is_better: bool) -> tuple[str, float, float]:
    """(metric key, raw value, error-direction value)；越大越好的指标反向成误差。"""
    key, raw, higher = resolve_metric(metrics, aliases, higher_is_better)
    return key, raw, -raw if higher else raw


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
