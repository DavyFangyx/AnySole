#!/usr/bin/env python3
"""Test6 · V-degradation difficulty-axis display (V削弱实验报告.md).

Two suites × 8 configs (cam3 baseline / cam1 / cam2 / cam4 / occ010–050):

  V3_Bv1  融合主线 -- V = main model V2M condition, VT = main VT2M
                    (sources: singlemodal_eval[_cams|_occl])
  V3_Dv1  干净通道 -- same two conditions of the position-head model
                    (sources: position_test / position_test_vdegrad)

每行只报 Δ = VT − V（负 = 加触觉更优）。指标面板 = PA / 踝足 PA / 全局
MPJPE / 地面穿透 / 根轨迹 ATE + 协议指标行胜率；V3_Dv1 另出全配置 val
decile 分层表（oracle 上限 / VT 胜率 / d10·d1 Δ，与 oracle_analysis.py
同一计算）。

输出（results_display/Test6_VdegradTest/<suite>/）：
  delta_table.csv / physical_delta.csv / win_rates.csv / delta_panels.png
  decile_table.csv / decile_panels.png（仅 V3_Dv1）/ analysis_summary.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import oracle_analysis
from utils.mpl_fonts import setup_cjk_fonts

setup_cjk_fonts()
# 与 component_analysis.py 同一裁定：图内文字全英文、DejaVu Sans 粗体。
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.weight"] = "bold"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
VT_BETTER = "#1baf7a"   # Δ < 0（VT 更优）
VT_WORSE = "#c0504d"    # Δ > 0
DPI = 200

CONFIG_ORDER = ("cam3", "cam1", "cam2", "cam4",
                "occ010", "occ020", "occ030", "occ050")

# config id -> (experiment id, model id).  cam3 基准行 = 既有主线/position
# 实验，其余 = 该套件自训自测模型。
SUITE_CONFIGS: dict[str, dict[str, tuple[str, str]]] = {
    "V3_Bv1": {
        "cam3": ("singlemodal_eval", "V3_Bv1"),
        "cam1": ("singlemodal_eval_cams", "V3_Bv1_cam1"),
        "cam2": ("singlemodal_eval_cams", "V3_Bv1_cam2"),
        "cam4": ("singlemodal_eval_cams", "V3_Bv1_cam4"),
        "occ010": ("singlemodal_eval_occl", "V3_Bv1_cam3_occ010"),
        "occ020": ("singlemodal_eval_occl", "V3_Bv1_cam3_occ020"),
        "occ030": ("singlemodal_eval_occl", "V3_Bv1_cam3_occ030"),
        "occ050": ("singlemodal_eval_occl", "V3_Bv1_cam3_occ050"),
    },
    "V3_Dv1": {
        "cam3": ("position_test", "V3_Dv1"),
        "cam1": ("position_test_vdegrad", "V3_Dv1_cam1"),
        "cam2": ("position_test_vdegrad", "V3_Dv1_cam2"),
        "cam4": ("position_test_vdegrad", "V3_Dv1_cam4"),
        "occ010": ("position_test_vdegrad", "V3_Dv1_cam3_occ010"),
        "occ020": ("position_test_vdegrad", "V3_Dv1_cam3_occ020"),
        "occ030": ("position_test_vdegrad", "V3_Dv1_cam3_occ030"),
        "occ050": ("position_test_vdegrad", "V3_Dv1_cam3_occ050"),
    },
}

# (metric key, display label) — 面板定义（V削弱实验报告.md §1）
METRIC_ROWS = (
    ("PA-MPJPE", "PA"),
    ("PA-MPJPE_anklefoot", "Ankle-foot PA"),
    ("GlobalMPJPE", "Global MPJPE"),
    ("GroundPen", "Ground penetration"),
    ("traj_ATE", "Root traj ATE"),
)
SUPPLEMENTARY = (
    ("contact_f1", "Contact F1"),
    ("FootSliding", "Foot sliding"),
)
HIGHER_BETTER = frozenset(
    ("air_recall", "contact_acc", "contact_balanced_acc",
     "contact_f1", "contact_recall"),
)


def _load_metrics(experiment_id: str, model_id: str, split: str) -> dict[str, Any]:
    path = (REPO / "results" / "experiments" / experiment_id
            / model_id / "metrics" / ("%s.json" % split))
    if not path.is_file():
        raise FileNotFoundError("metrics missing: %s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError("metrics file lacks metrics: %s" % path)
    return data["metrics"]


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _delta(v: float | None, vt: float | None) -> float | None:
    if v is None or vt is None:
        return None
    return vt - v


def _win_rate(v: dict[str, Any], vt: dict[str, Any]) -> tuple[float, int, int]:
    """VT-beats-V share over the common protocol rows; ties count half."""
    keys = sorted(set(v) & set(vt))
    wins = 0.0
    ties = 0
    for key in keys:
        a, b = _numeric(v[key]), _numeric(vt[key])
        if a is None or b is None:
            continue
        if a == b:
            wins += 0.5
            ties += 1
        elif (b > a) if key in HIGHER_BETTER else (b < a):
            wins += 1.0
    return wins / len(keys), len(keys), ties


def _csv(rows: list[dict[str, Any]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("%.4f" % v) if isinstance(v, float) else v
                         for k, v in row.items()})
    return handle.getvalue()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def _suite_rows(suite: str, split: str) -> list[dict[str, Any]]:
    """Per config: V/VT values + Δ for the metric panel, win rate, decile head."""
    rows: list[dict[str, Any]] = []
    for config in CONFIG_ORDER:
        experiment_id, model_id = SUITE_CONFIGS[suite][config]
        metrics = _load_metrics(experiment_id, model_id, split)
        if "V2M" not in metrics or "VT2M" not in metrics:
            raise ValueError("%s/%s: no V2M/VT2M conditions"
                             % (experiment_id, model_id))
        v, vt = metrics["V2M"], metrics["VT2M"]
        row: dict[str, Any] = {"config": config}
        for key, _label in METRIC_ROWS:
            v_val = _numeric(v.get(key))
            vt_val = _numeric(vt.get(key))
            row["v_%s" % key] = v_val
            row["vt_%s" % key] = vt_val
            row["d_%s" % key] = _delta(v_val, vt_val)
        for key, _label in SUPPLEMENTARY:
            v_val = _numeric(v.get(key))
            vt_val = _numeric(vt.get(key))
            row["v_%s" % key] = v_val
            row["vt_%s" % key] = vt_val
            row["d_%s" % key] = _delta(v_val, vt_val)
        rate, n_rows, n_ties = _win_rate(v, vt)
        row["win_rate"] = rate
        row["win_n_rows"] = n_rows
        row["win_n_ties"] = n_ties
        rows.append(row)
    return rows


def _draw_delta_panels(rows: list[dict[str, Any]], suite: str, path: Path) -> None:
    configs = [row["config"] for row in rows]
    fig, axes = plt.subplots(len(METRIC_ROWS), 1, figsize=(11.0, 10.5), dpi=DPI,
                             sharex=True)
    x = np.arange(len(configs))
    for ax, (key, label) in zip(axes, METRIC_ROWS):
        values = [row["d_%s" % key] for row in rows]
        colors = [VT_BETTER if value is not None and value < 0 else VT_WORSE
                  if value is not None else MUTED
                  for value in values]
        heights = [value if value is not None else 0.0 for value in values]
        ax.bar(x, heights, color=colors, width=0.62, zorder=3)
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, 0.0, "n/a", ha="center", va="bottom",
                        fontsize=8, color=MUTED)
                continue
            sign = "+" if value >= 0 else ""
            ax.text(i, heights[i], "%s%.2f" % (sign, value),
                    ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=8.5, color=INK)
        ax.axhline(0.0, color=INK, linewidth=1.2, zorder=2)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.tick_params(colors=INK2)
        ax.set_ylabel("%s Δ" % label, color=INK)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(configs, color=INK)
    axes[-1].set_xlabel("Config (negative Δ = V+T better)", color=INK)
    fig.suptitle("%s suite · Δ = VT − V across configs" % suite,
                 color=INK, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _decile_head(suite: str, config: str) -> dict[str, Any] | None:
    """V3_Dv1 per-config val decile summary (oracle_analysis.py 同计算)."""
    experiment_id, model_id = SUITE_CONFIGS[suite][config]
    detail = (REPO / "results" / "experiments" / experiment_id
              / model_id / "metrics" / "window_detail_val.json")
    if not detail.is_file():
        return None
    v_cond = oracle_analysis._load_condition(detail, "V2M")
    vt_cond = oracle_analysis._load_condition(detail, "VT2M")
    paired = oracle_analysis._paired_windows(v_cond, vt_cond)
    overall = oracle_analysis._region_stats(
        *oracle_analysis._win_series(paired, "pa_mpjpe_mm"))
    deciles = oracle_analysis._deciles(paired)
    d1, d10 = deciles[0], deciles[-1]

    def d_of(dec: dict[str, Any], label: str) -> float:
        return float(dec["%s_vt" % label]) - float(dec["%s_v" % label])

    return {
        "config": config,
        "oracle_gain": overall["oracle_gain"],
        "vt_win_frac": overall["vt_win_frac"],
        "d10_delta": d_of(d10, "Overall"),
        "d10_delta_anklefoot": d_of(d10, "Ankle-foot"),
        "d10_delta_upper": d_of(d10, "Upper"),
        "d1_delta": d_of(d1, "Overall"),
        "n_windows": len(paired["keys"]),
        # decile curves for the figure
        "decile_curve": [
            {"decile": row["decile"], "delta": d_of(row, "Overall")}
            for row in deciles
        ],
    }


def _draw_decile_panels(heads: list[dict[str, Any]], suite: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), dpi=DPI)
    for ax, head in zip(axes.ravel(), heads):
        curve = head["decile_curve"]
        deciles = [row["decile"] for row in curve]
        deltas = [row["delta"] for row in curve]
        colors = [VT_BETTER if value < 0 else VT_WORSE for value in deltas]
        ax.bar(deciles, deltas, color=colors, width=0.6, zorder=3)
        ax.axhline(0.0, color=INK, linewidth=1.2, zorder=2)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.tick_params(colors=INK2)
        ax.set_xticks(deciles)
        ax.set_title("%s  (d10 Δ %+.2f)" % (head["config"], head["d10_delta"]),
                     color=INK, fontsize=10.5)
        if ax in axes[:, 0]:
            ax.set_ylabel("Δ (mm)", color=INK)
    fig.suptitle("%s suite · val decile Δ = VT − V per config" % suite,
                 color=INK, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _summary(rows: list[dict[str, Any]], suite: str, split: str,
             decile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def d(row: dict[str, Any], key: str) -> float | None:
        return row["d_%s" % key]

    pa_flips = [r["config"] for r in rows
                if d(r, "PA-MPJPE") is not None and d(r, "PA-MPJPE") < 0]
    pa_deltas = [d(r, "PA-MPJPE") for r in rows if d(r, "PA-MPJPE") is not None]
    summary: dict[str, Any] = {
        "suite": suite,
        "split": split,
        "n_configs": len(rows),
        "pa": {
            "delta_range": [min(pa_deltas), max(pa_deltas)],
            "sign_flip_configs": pa_flips,
        },
        "physical_direction": {},
        "supplementary": {},
        "win_rate": {
            "values": {r["config"]: r["win_rate"] for r in rows},
            "n_rows": {r["config"]: r["win_n_rows"] for r in rows},
        },
    }
    for key, label in (("GlobalMPJPE", "Global MPJPE"),
                       ("GroundPen", "Ground penetration"),
                       ("traj_ATE", "Root traj ATE")):
        deltas = [d(r, key) for r in rows if d(r, key) is not None]
        summary["physical_direction"][label] = {
            "n": len(deltas),
            "n_negative": sum(1 for value in deltas if value < 0),
            "delta_range": [min(deltas), max(deltas)] if deltas else None,
        }
    for key, label in SUPPLEMENTARY:
        deltas = [d(r, key) for r in rows if d(r, key) is not None]
        summary["supplementary"][label] = {
            "n": len(deltas),
            "n_negative": sum(1 for value in deltas if value < 0),
            "delta_range": [min(deltas), max(deltas)] if deltas else None,
        }
    if decile_rows:
        summary["decile"] = {
            "oracle_gain_range": [min(r["oracle_gain"] for r in decile_rows),
                                  max(r["oracle_gain"] for r in decile_rows)],
            "d10_delta_range": [min(r["d10_delta"] for r in decile_rows),
                                max(r["d10_delta"] for r in decile_rows)],
        }
    return summary


def run_suite(suite: str, split: str, out_root: Path) -> None:
    rows = _suite_rows(suite, split)
    out = out_root / suite
    out.mkdir(parents=True, exist_ok=True)

    delta_cols = (["config"]
                  + ["%s_%s" % (side, key)
                     for key, _ in METRIC_ROWS for side in ("v", "vt")]
                  + ["d_%s" % key for key, _ in METRIC_ROWS])
    (out / "delta_table.csv").write_text(_csv(
        [{k: row[k] for k in delta_cols} for row in rows]), encoding="utf-8")
    (out / "physical_delta.csv").write_text(_csv([
        {"config": row["config"],
         "d_GlobalMPJPE": row["d_GlobalMPJPE"],
         "d_GroundPen": row["d_GroundPen"],
         "d_traj_ATE": row["d_traj_ATE"]} for row in rows]), encoding="utf-8")
    (out / "win_rates.csv").write_text(_csv([
        {"config": row["config"], "win_rate": row["win_rate"],
         "n_rows": row["win_n_rows"], "n_ties": row["win_n_ties"]}
        for row in rows]), encoding="utf-8")
    _draw_delta_panels(rows, suite, out / "delta_panels.png")

    decile_rows: list[dict[str, Any]] = []
    if suite == "V3_Dv1":
        for config in CONFIG_ORDER:
            head = _decile_head(suite, config)
            if head is None:
                raise FileNotFoundError(
                    "window_detail_val.json missing for %s/%s"
                    % (SUITE_CONFIGS[suite][config][0],
                       SUITE_CONFIGS[suite][config][1]))
            decile_rows.append(head)
        table = [{k: head[k] for k in
                  ("config", "oracle_gain", "vt_win_frac", "d10_delta",
                   "d10_delta_anklefoot", "d10_delta_upper", "d1_delta",
                   "n_windows")} for head in decile_rows]
        (out / "decile_table.csv").write_text(_csv(table), encoding="utf-8")
        _draw_decile_panels(decile_rows, suite, out / "decile_panels.png")

    _write_json(out / "analysis_summary.json",
                _summary(rows, suite, split, decile_rows))
    print("[%s] wrote %s (%d configs)"
          % (suite, out, len(rows)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="V3_Bv1,V3_Dv1",
                        help="comma-separated suite IDs (default: both)")
    parser.add_argument("--split", choices=("val", "test"), default="test",
                        help="split for the Δ/win-rate tables (decile is val)")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test6_VdegradTest")
    args = parser.parse_args(argv)
    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    unknown = [s for s in suites if s not in SUITE_CONFIGS]
    if unknown:
        raise SystemExit("unknown suite IDs: %s (registered: %s)"
                         % (",".join(unknown), ",".join(SUITE_CONFIGS)))
    for suite in suites:
        run_suite(suite, args.split, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
