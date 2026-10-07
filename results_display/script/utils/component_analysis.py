#!/usr/bin/env python3
"""C1/C2/C3/M1/M2 shared component analysis for the missing-rate task book.

The legacy probe and t-SNE views are intentionally not part of the main
experiment anymore.  This module implements the contrasts used by the
revised task book (the canonical analysis entry points import ``main``):

  --c1  V/T specialist division
  --c2  main VT versus the best V/T specialist (fusion completeness)
  --c3  main V-only versus main T-only (missing-condition division)
  --m1  main V versus V specialist and prior on T-domain rows
  --m2  main T versus T specialist and prior on V-domain rows

2026-10-07 起对比行集由 7 分量表改为区域分组（上半身/下半身/全身/足-地，
见 utils/missing_rate_analysis.py REGION_GROUPS）。表现结构按 2026-10-07
裁定：C1/C3 = 1×2 面板（左 V 右 T）、C2 = 最佳专才 vs 主线 1×2 + 三色合图、
M1/M2 = 三色合图 + 1×3 面板（M1 行 = T 域 9、M2 行 = V 域 10）；
构图风格为经典森林图的细线端点（细横线 + 小端点 + 数值标注），每行一个
迷你轴（混合单位各轴独立）。

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
    T_DOMAIN_ROWS,
    V_DOMAIN_ROWS,
    error_value,
    load_fseries,
    load_grid_prior,
    row_specs,
    row_specs_in_domain,
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
MAIN_COLOR = "#1baf7a"
SPEC_COLOR = "#8c8b84"
PRIOR_COLOR = "#b9b7ae"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1-C3/M1-M2 missing-rate component analysis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--c1", action="store_true", help="C1: V/T specialist division")
    mode.add_argument("--all", action="store_true", help="run all shared component analyses")
    mode.add_argument("--c2", action="store_true", help="C2: main VT versus best specialist")
    mode.add_argument("--c3", action="store_true", help="C3: main V-only versus main T-only")
    mode.add_argument("--m1", action="store_true", help="M1: main V on T-domain rows")
    mode.add_argument("--m2", action="store_true", help="M2: main T on V-domain rows")
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


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CSV 行（去掉渲染用的 values/aliases/higher 键）。"""
    return [{k: v for k, v in row.items() if k not in ("values", "aliases", "higher")} for row in rows]


def domain_label_colors(rows: list[dict[str, Any]]) -> dict[str, str]:
    """V 域行标签蓝色、T 域行标签橙色。"""
    return {row["row_id"]: V_COLOR if row["row_id"] in V_DOMAIN_ROWS else T_COLOR
            for row in rows}


def _fmt(row: dict[str, Any], value: float) -> str:
    if row["row_id"] == "contact":
        return "%.3f" % value
    if row["row_id"] == "accel":
        return "%.2f" % value
    if row["row_id"] == "root_traj":
        return "%.1f%%" % value
    if row["row_id"] == "root_orient":
        return "%.1f°" % value
    return "%.1f" % value


def region_bars(path: Path, rows: list[dict[str, Any]], series: list[str],
                colors: dict[str, str], title: str, layout: str = "columns",
                label_colors: dict[str, str] | None = None) -> None:
    """按行面板图，细线端点风格（构图学自经典森林图，非粗柱）。

    每行一个迷你横向轴（混合单位各轴独立），行内画细横线 + 小端点 +
    数值标注。layout="columns"：每个 series 一列面板（1×N）；layout=
    "grouped"：单列，每行并排各 series 的细线端点。组标题标在每组第一行
    上方；行标签画在左缘，可经 label_colors 按域着色。
    """
    if not rows:
        return
    label_colors = label_colors or {}
    n_cols = len(series) if layout == "columns" else 1
    row_h = 0.42
    fig = plt.figure(figsize=(3.0 * n_cols + 0.8, row_h * len(rows) + 1.0), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.985, title, fontsize=12, color=INK, ha="left", va="top")
    grid = fig.add_gridspec(len(rows), n_cols, hspace=0.30, wspace=0.50,
                            left=0.26 if layout == "columns" else 0.34,
                            right=0.985, top=0.925, bottom=0.02)
    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = fig.add_subplot(grid[i, j])
            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            if layout == "columns":
                sname = series[j]
                val = row["values"].get(sname)
                if val is None:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            color=MUTED, fontsize=9, transform=ax.transAxes)
                else:
                    color = colors.get(sname, INK2)
                    if row.get("source_color") and j == 0:
                        color = row["source_color"]
                    ax.hlines(0.0, 0, val, color=color, linewidth=2.2, zorder=3)
                    ax.scatter([val], [0.0], s=20, color=color, edgecolors="white",
                               linewidths=0.5, zorder=4)
                    ax.text(val, 0.16, _fmt(row, val), ha="left", va="bottom",
                            fontsize=8, color=INK2)
                    ax.set_xlim(0, max(val * 1.28, 1e-6))
                if j == 0:
                    ax.text(-0.06, 0.5, row["label"], ha="right", va="center",
                            transform=ax.transAxes, fontsize=8.5,
                            color=label_colors.get(row["row_id"], INK))
            else:
                vals = [row["values"].get(s) for s in series]
                known = [v for v in vals if v is not None]
                if not known:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            color=MUTED, fontsize=9, transform=ax.transAxes)
                else:
                    ax.set_xlim(0, max(known) * 1.30 or 1e-6)
                    n = len(series)
                    for k, sname in enumerate(series):
                        v = vals[k]
                        if v is None:
                            continue
                        y = 0.5 - 0.16 * (k - (n - 1) / 2)
                        ax.hlines(y, 0, v, color=colors.get(sname, INK2),
                                  linewidth=1.8, zorder=3)
                        ax.scatter([v], [y], s=16, color=colors.get(sname, INK2),
                                   edgecolors="white", linewidths=0.5, zorder=4)
                        ax.text(v, y + 0.075, _fmt(row, v), ha="left", va="bottom",
                                fontsize=7, color=INK2)
                ax.text(-0.06, 0.5, row["label"], ha="right", va="center",
                        transform=ax.transAxes, fontsize=8.5,
                        color=label_colors.get(row["row_id"], INK))
            if j == 0 and (i == 0 or rows[i - 1]["region"] != row["region"]):
                ax.set_title(row["region_label"], loc="left", fontsize=9,
                             color=MUTED, pad=16)
    if layout == "columns":
        for j, sname in enumerate(series):
            x0 = grid[0, j].get_position(fig).x0
            x1 = grid[-1, j].get_position(fig).x1
            fig.text((x0 + x1) / 2, 0.965, sname, ha="center", va="bottom",
                     fontsize=11, color=INK)
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(s, INK2)) for s in series]
        fig.legend(handles, series, loc="upper right", frameon=False, fontsize=8.5)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_c1(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C1 需要 --fs-vspecialist 与 --fs-tspecialist")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for spec in row_specs():
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        delta = t_err - v_err
        winner = "V specialist" if delta > args.tol else "T specialist" if delta < -args.tol else "tie"
        rows.append({
            **spec,
            "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "V_specialist": round(v_raw, 6), "T_specialist": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6), "winner": winner,
            "values": {"V specialist": v_raw, "T specialist": t_raw},
        })
    out = args.out
    write_rows(out / "specialist_table.csv", _csv_rows(rows))
    region_bars(out / "specialist_panels.png", rows,
                ["V specialist", "T specialist"],
                {"V specialist": V_COLOR, "T specialist": T_COLOR},
                "C1 specialist division by region", layout="columns",
                label_colors=domain_label_colors(rows))
    summary = {"V_wins": [r["row_id"] for r in rows if r["winner"] == "V specialist"],
               "T_wins": [r["row_id"] for r in rows if r["winner"] == "T specialist"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c2(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C2 需要 --fs-vspecialist 与 --fs-tspecialist")
    main = metric_block(inputs["main"], "VT2M")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for spec in row_specs():
        main_key, main_raw, main_err = error_value(main, spec["aliases"], spec["higher"])
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        best_err = min(v_err, t_err)
        best_raw = v_raw if v_err <= t_err else t_raw
        best_src = "V specialist" if v_err <= t_err else "T specialist"
        delta = main_err - best_err
        category = "synergy" if delta < -args.tol else "interference" if delta > args.tol else "keep or select"
        rows.append({
            **spec,
            "metric": main_key,
            "main_VT": round(main_raw, 6), "V_specialist": round(v_raw, 6),
            "T_specialist": round(t_raw, 6), "best_specialist": round(best_raw, 6),
            "best_source": best_src,
            "delta_error_VT_minus_best": round(delta, 6), "category": category,
            "values": {"V specialist": v_raw, "T specialist": t_raw, "Main VT": main_raw,
                       "Best specialist": best_raw},
            "source_color": V_COLOR if v_err <= t_err else T_COLOR,
        })
    out = args.out
    write_rows(out / "fusion_table.csv", _csv_rows(rows))
    region_bars(out / "fusion_best_panels.png", rows,
                ["Best specialist", "Main VT"],
                {"Best specialist": SPEC_COLOR, "Main VT": MAIN_COLOR},
                "C2 fusion: best specialist vs main VT", layout="columns")
    region_bars(out / "fusion_series.png", rows,
                ["V specialist", "T specialist", "Main VT"],
                {"V specialist": V_COLOR, "T specialist": T_COLOR, "Main VT": MAIN_COLOR},
                "C2 fusion: specialists vs main VT", layout="grouped")
    summary = {"synergy": sum(r["category"] == "synergy" for r in rows),
               "selection": sum(r["category"] == "keep or select" for r in rows),
               "interference": sum(r["category"] == "interference" for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c3(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    v = metric_block(inputs["main"], "V2M")
    t = metric_block(inputs["main"], "T2M")
    rows = []
    for spec in row_specs():
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        delta = t_err - v_err
        winner = "V-only" if delta > args.tol else "T-only" if delta < -args.tol else "tie"
        expected = "V-only" if spec["row_id"] in V_DOMAIN_ROWS else "T-only"
        rows.append({
            **spec,
            "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "main_V": round(v_raw, 6), "main_T": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6), "winner": winner,
            "expected_winner": expected, "matches_expected": winner == expected,
            "values": {"Main V-only": v_raw, "Main T-only": t_raw},
        })
    out = args.out
    write_rows(out / "main_branch_table.csv", _csv_rows(rows))
    region_bars(out / "main_branch_panels.png", rows,
                ["Main V-only", "Main T-only"],
                {"Main V-only": V_COLOR, "Main T-only": T_COLOR},
                "C3 main branches by region", layout="columns",
                label_colors=domain_label_colors(rows))
    summary = {"matches_expected": sum(bool(r["matches_expected"]) for r in rows), "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def branch_table(args: argparse.Namespace, inputs: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch == "m1":
        specialist_name, specialist_cfg, main_cfg = "v_specialist", "V2M", "V2M"
        domain_rows = T_DOMAIN_ROWS
        main_label, spec_label = "Main V", "V specialist"
        main_color = V_COLOR
        title = "M1: main V on T-domain rows"
    else:
        specialist_name, specialist_cfg, main_cfg = "t_specialist", "T2M", "T2M"
        domain_rows = V_DOMAIN_ROWS
        main_label, spec_label = "Main T", "T specialist"
        main_color = T_COLOR
        title = "M2: main T on V-domain rows"
    if specialist_name not in inputs:
        raise SystemExit("%s 需要对应专才明细文件" % branch.upper())
    main = metric_block(inputs["main"], main_cfg)
    specialist = metric_block(inputs[specialist_name], specialist_cfg)
    prior = load_grid_prior(args.prior_grid, args.prior_cell)
    rows = []
    for spec in row_specs_in_domain(domain_rows):
        m_key, m_raw, m_err = error_value(main, spec["aliases"], spec["higher"])
        s_key, s_raw, s_err = error_value(specialist, spec["aliases"], spec["higher"])
        p_key, p_raw, p_err = error_value(prior, spec["aliases"], spec["higher"])
        d_spec = m_err - s_err
        d_prior = m_err - p_err
        rows.append({
            **spec,
            "metric": m_key,
            "prior": round(p_raw, 6), "specialist": round(s_raw, 6), "main": round(m_raw, 6),
            "delta_error_main_minus_specialist": round(d_spec, 6),
            "delta_error_main_minus_prior": round(d_prior, 6),
            "beats_specialist": d_spec < -args.tol,
            "beats_prior": d_prior < -args.tol,
            "holds": d_spec < -args.tol and d_prior < -args.tol,
            "values": {main_label: m_raw, spec_label: s_raw, "Prior": p_raw},
        })
    out = args.out
    write_rows(out / "branch_table.csv", _csv_rows(rows))
    series = [main_label, spec_label, "Prior"]
    colors = {main_label: main_color, spec_label: SPEC_COLOR, "Prior": PRIOR_COLOR}
    region_bars(out / "branch_series.png", rows, series, colors, title, layout="grouped")
    region_bars(out / "branch_panels.png", rows, series, colors, title, layout="columns")
    summary = {"holds": [r["row_id"] for r in rows if r["holds"]],
               "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


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
        # --all 一次跑多模式时按模式分目录，避免同名文件互相覆盖
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
