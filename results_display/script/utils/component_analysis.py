#!/usr/bin/env python3
"""C1/C2/C3/M1/M2 shared component analysis for the missing-rate task book.

The legacy probe and t-SNE views are intentionally not part of the main
experiment anymore.  This module implements the contrasts used by the
revised task book (the canonical analysis entry points import ``main``):

  --c1  V/T specialist division
  --c2  main VT versus the best V/T specialist (fusion completeness)
  --c3  main V-only versus main T-only (missing-condition division)
  --m1  main V versus V specialist and prior on T-owned components
  --m2  main T versus T specialist and prior on V-owned components

``--all`` runs all shared component analyses.  ``--bar`` is retained as a
compatibility alias for ``--c2``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1]  # results_display/script
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.missing_rate_analysis import (
    COMPONENTS,
    T_COMPONENTS,
    V_COMPONENTS,
    component_label,
    error_value,
    load_fseries,
    load_grid_prior,
    write_rows,
)
from utils.mpl_fonts import setup_cjk_fonts

setup_cjk_fonts()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
V_COLOR = "#2a78d6"
T_COLOR = "#eb6834"
GOOD = "#1baf7a"
BAD = "#e34948"
FLAT = "#8c8b84"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1-C3/M1-M2 missing-rate component analysis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--c1", action="store_true", help="C1: V/T specialist division")
    mode.add_argument("--all", action="store_true", help="run all shared component analyses")
    mode.add_argument("--c2", action="store_true", help="C2: main VT versus best specialist")
    mode.add_argument("--c3", action="store_true", help="C3: main V-only versus main T-only")
    mode.add_argument("--m1", action="store_true", help="M1: main V on T-owned components")
    mode.add_argument("--m2", action="store_true", help="M2: main T on V-owned components")
    mode.add_argument("--bar", action="store_true", help="deprecated alias for --c2")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--fs-main", type=Path,
                        default=REPO / "results" / "experiments" / "singlemodal_eval" / "V3_3B" / "metrics" / "val.json")
    parser.add_argument("--fs-vspecialist", type=Path, default=None,
                        help="V 专才明细文件 (metrics/<split>.json)；C2/M1 必需")
    parser.add_argument("--fs-tspecialist", type=Path, default=None,
                        help="T 专才明细文件 (metrics/<split>.json)；C2/M2 必需")
    parser.add_argument("--prior-grid", type=Path,
                        default=REPO / "results" / "experiments" / "rho_grid_eval" / "V3_3B" / "grid_metrics.json",
                        help="主线 rhoV0_rhoT0 先验单元，M1/M2 必需")
    parser.add_argument("--prior-cell", default="rhoV0_rhoT0")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test1_ComplementTest" / "C2_fusion")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="误差差值分类容差；混合单位时默认 0，只按方向分类")
    return parser.parse_args(argv)


def metric_block(data: dict[str, Any], config: str) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    if config not in metrics:
        raise ValueError("明细文件缺配置 %s；已有：%s" % (config, ", ".join(metrics)))
    return metrics[config]


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    main = load_fseries(args.fs_main)
    out = {"main": main}
    if args.fs_vspecialist:
        out["v_specialist"] = load_fseries(args.fs_vspecialist)
    if args.fs_tspecialist:
        out["t_specialist"] = load_fseries(args.fs_tspecialist)
    return out


def style_ax(ax) -> None:
    """Minimal academic axis: only the bottom spine, dashed x grid."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)


def _forest_color(value: float, positive_is_better: bool) -> str:
    good = value > 0 if positive_is_better else value < 0
    bad = value < 0 if positive_is_better else value > 0
    return GOOD if good else BAD if bad else FLAT


def forest(path: Path, rows: list[dict[str, Any]], value_key: str, title: str,
           zero_text: str = "0 = no difference", positive_is_better: bool = False) -> None:
    if not rows:
        return
    rows = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(7.6, max(2.9, 0.44 * len(rows) + 1.4)), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    y = np.arange(len(rows))
    vals = np.array([float(row[value_key]) for row in rows])
    colors = [_forest_color(value, positive_is_better) for value in vals]
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
    ax.hlines(y, 0, vals, colors=colors, linewidth=2.6, zorder=3)
    ax.scatter(vals, y, s=26, c=colors, edgecolors="white", linewidths=0.6, zorder=4)
    span = float(np.max(np.abs(vals))) if len(vals) else 0.0
    dx = max(span * 0.035, 0.01)
    for yi, value in zip(y, vals):
        ax.text(value + (dx if value >= 0 else -dx), yi, "%+.3g" % value,
                ha="left" if value >= 0 else "right", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([row["component"] for row in rows], fontsize=9.5)
    direction = "positive" if positive_is_better else "negative"
    ax.set_xlabel("Error difference (%s = favorable direction)" % direction,
                  color=INK2, fontsize=10, labelpad=8)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    ax.text(0.99, -0.14, zero_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run_c1(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C1 需要 --fs-vspecialist 与 --fs-tspecialist")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for component in (*V_COMPONENTS, *T_COMPONENTS):
        v_key, v_raw, v_err = error_value(v, component)
        t_key, t_raw, t_err = error_value(t, component)
        delta = t_err - v_err
        rows.append({
            "component_id": component, "component": component_label(component),
            "owner": "V" if component in V_COMPONENTS else "T",
            "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "V_specialist": round(v_raw, 6), "T_specialist": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6),
            "winner": "V specialist" if delta > args.tol else "T specialist" if delta < -args.tol else "tie",
        })
    out = args.out
    write_rows(out / "specialist_table.csv", rows)
    forest(out / "specialist_forest.png", rows, "delta_error_T_minus_V",
           "C1 specialist division: T error minus V error",
           positive_is_better=True)
    summary = {"V_wins": [r["component_id"] for r in rows if r["winner"] == "V specialist"],
               "T_wins": [r["component_id"] for r in rows if r["winner"] == "T specialist"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c2(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C2 需要 --fs-vspecialist 与 --fs-tspecialist")
    main = metric_block(inputs["main"], "VT2M")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for component in (*V_COMPONENTS, *T_COMPONENTS):
        main_key, main_raw, main_err = error_value(main, component)
        v_key, v_raw, v_err = error_value(v, component)
        t_key, t_raw, t_err = error_value(t, component)
        best = min(v_err, t_err)
        best_raw = v_raw if v_err <= t_err else t_raw
        delta = main_err - best
        category = "synergy" if delta < -args.tol else "interference" if delta > args.tol else "keep or select"
        rows.append({
            "component_id": component, "component": component_label(component),
            "owner": "V" if component in V_COMPONENTS else "T", "metric": main_key,
            "main_VT": round(main_raw, 6), "V_specialist": round(v_raw, 6),
            "T_specialist": round(t_raw, 6), "best_specialist": round(best_raw, 6),
            "delta_error_VT_minus_best": round(delta, 6), "category": category,
        })
    out = args.out
    write_rows(out / "fusion_table.csv", rows)
    forest(out / "fusion_forest.png", rows, "delta_error_VT_minus_best",
           "C2 fusion: main VT minus best specialist")
    summary = {"synergy": sum(r["category"] == "synergy" for r in rows),
               "selection": sum(r["category"] == "keep or select" for r in rows),
               "interference": sum(r["category"] == "interference" for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c3(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    v = metric_block(inputs["main"], "V2M")
    t = metric_block(inputs["main"], "T2M")
    rows = []
    for component in (*V_COMPONENTS, *T_COMPONENTS):
        v_key, v_raw, v_err = error_value(v, component)
        t_key, t_raw, t_err = error_value(t, component)
        delta = t_err - v_err
        winner = "V-only" if delta > args.tol else "T-only" if delta < -args.tol else "tie"
        rows.append({
            "component_id": component, "component": component_label(component),
            "owner": "V" if component in V_COMPONENTS else "T", "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "main_V": round(v_raw, 6), "main_T": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6), "winner": winner,
            "expected_winner": "V-only" if component in V_COMPONENTS else "T-only",
            "matches_expected": winner == ("V-only" if component in V_COMPONENTS else "T-only"),
        })
    out = args.out
    write_rows(out / "main_branch_table.csv", rows)
    forest(out / "main_branch_forest.png", rows, "delta_error_T_minus_V",
           "C3 main branches: T-only error minus V-only error",
           positive_is_better=True)
    summary = {"matches_expected": sum(bool(r["matches_expected"]) for r in rows), "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def branch_table(args: argparse.Namespace, inputs: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch == "m1":
        specialist_name, specialist_cfg, main_cfg, components = "v_specialist", "V2M", "V2M", T_COMPONENTS
        title = "M1: main V on T-owned components"
    else:
        specialist_name, specialist_cfg, main_cfg, components = "t_specialist", "T2M", "T2M", V_COMPONENTS
        title = "M2: main T on V-owned components"
    if specialist_name not in inputs:
        raise SystemExit("%s 需要对应专才明细文件" % branch.upper())
    main = metric_block(inputs["main"], main_cfg)
    specialist = metric_block(inputs[specialist_name], specialist_cfg)
    prior = load_grid_prior(args.prior_grid, args.prior_cell)
    rows = []
    for component in components:
        m_key, m_raw, m_err = error_value(main, component)
        s_key, s_raw, s_err = error_value(specialist, component)
        p_key, p_raw, p_err = error_value(prior, component)
        d_spec = m_err - s_err
        d_prior = m_err - p_err
        rows.append({
            "component_id": component, "component": component_label(component), "metric": m_key,
            "prior": round(p_raw, 6), "specialist": round(s_raw, 6), "main": round(m_raw, 6),
            "delta_error_main_minus_specialist": round(d_spec, 6),
            "delta_error_main_minus_prior": round(d_prior, 6),
            "beats_specialist": d_spec < -args.tol,
            "beats_prior": d_prior < -args.tol,
            "holds": d_spec < -args.tol and d_prior < -args.tol,
        })
    out = args.out
    write_rows(out / "branch_table.csv", rows)
    _dual_forest(out / "branch_forest.png", rows, title)
    summary = {"holds": [r["component_id"] for r in rows if r["holds"]],
               "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def _dual_forest(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    rows = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(7.6, max(2.9, 0.5 * len(rows) + 1.5)), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    y = np.arange(len(rows))
    d1 = np.array([r["delta_error_main_minus_specialist"] for r in rows])
    d2 = np.array([r["delta_error_main_minus_prior"] for r in rows])
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
    ax.hlines(y + 0.1, 0, d1, colors=V_COLOR, linewidth=2.2, zorder=3)
    ax.hlines(y - 0.1, 0, d2, colors=T_COLOR, linewidth=2.2, zorder=3)
    ax.scatter(d1, y + 0.1, s=22, color=V_COLOR, edgecolors="white", linewidths=0.5,
               label="main − specialist", zorder=4)
    ax.scatter(d2, y - 0.1, s=22, color=T_COLOR, edgecolors="white", linewidths=0.5,
               label="main − prior", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r["component"] for r in rows], fontsize=9.5)
    ax.set_xlabel("Error difference (negative = main is better)", color=INK2,
                  fontsize=10, labelpad=8)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args)
    if args.bar:
        args.c2 = True
    modes = {"c1": args.c1, "c2": args.c2, "c3": args.c3, "m1": args.m1, "m2": args.m2}
    if args.all:
        modes = {key: True for key in modes}
    elif not any(modes.values()):
        modes = {key: False for key in modes}
        modes["c2"] = True
    results = {}
    for mode_key in ("c1", "c2", "c3", "m1", "m2"):
        if not modes[mode_key]:
            continue
        # --all 一次跑多模式时按模式分目录，避免去前缀后的同名文件互相覆盖
        mode_out = args.out / mode_key if args.all else args.out
        mode_out.mkdir(parents=True, exist_ok=True)
        mode_args = argparse.Namespace(**{**vars(args), "out": mode_out})
        if mode_key == "c1":
            results[mode_key] = run_c1(mode_args, inputs)
        elif mode_key == "c2":
            results[mode_key] = run_c2(mode_args, inputs)
        elif mode_key == "c3":
            results[mode_key] = run_c3(mode_args, inputs)
        elif mode_key == "m1":
            results[mode_key] = branch_table(mode_args, inputs, "m1")
        elif mode_key == "m2":
            results[mode_key] = branch_table(mode_args, inputs, "m2")
    (args.out / "analysis_summary.json").write_text(
        json.dumps({key: value["summary"] for key, value in results.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("完成：%s（%s）" % (args.out, ", ".join(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
