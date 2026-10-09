#!/usr/bin/env python3
"""T 判定后续 · 实验 1（Test3_ResidualTest）：冻结残差三列对比。

三列目标 = V specialist（V3_3B_vonly V2M，冻结基底的既有数字）、main VT
（主线 V3_3B VT2M，C2 的 interference 口径）、residual VT（V3_3C VT2M，
冻结 V + 零初始化 T 残差微调 200ep 的产物）。

行集 = C2 同款 19 行（REGION_GROUPS 上半身/下半身/全身/足-地）+ 2 行新指标
（GlobalMPJPE / GroundPen）。分类口径同 C2：delta 相对最佳者，
synergy / interference。

输出：residual_table.csv / residual_series.png（三列合图）/
residual_panels.png（1×2：V specialist vs residual VT）+ analysis_summary.json
（headline：residual-VT 是否闭合 VT−V 缺口、逐区域判定）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.component_analysis import (
    MAIN_COLOR,
    V_COLOR,
    domain_label_colors,
    error_value,
    load_fseries,
    metric_block,
    render_figure,
)
from utils.missing_rate_analysis import row_specs, write_rows

# 2026-10-09 新指标行（实验 2 的物理口径，追加进「全身」组）
EXTRA_ROWS = (
    ("global_mpjpe", "Global MPJPE (mm)", ("GlobalMPJPE", "global_mpjpe_mm"), False),
    ("ground_pen", "Ground penetration (mm)", ("GroundPen", "ground_pen_mm"), False),
)


def _specs() -> list[dict[str, Any]]:
    specs = row_specs()
    specs.append({
        "row_id": "global_mpjpe", "region": "full",
        "region_label": "Full body (mixed units)",
        "label": "Global MPJPE (mm)", "aliases": ("GlobalMPJPE", "global_mpjpe_mm"),
        "higher": False,
    })
    specs.append({
        "row_id": "ground_pen", "region": "full",
        "region_label": "Full body (mixed units)",
        "label": "Ground penetration (mm)", "aliases": ("GroundPen", "ground_pen_mm"),
        "higher": False,
    })
    return specs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--fs-vspecialist", type=Path, required=True,
                        help="V3_3B_vonly metrics/<split>.json")
    parser.add_argument("--fs-mainvt", type=Path, required=True,
                        help="mainline V3_3B metrics/<split>.json")
    parser.add_argument("--fs-residual", type=Path, required=True,
                        help="V3_3C metrics/<split>.json")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="误差差值分类容差（同 C2）")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test3_ResidualTest" / "V3_3C")
    args = parser.parse_args(argv)

    v = metric_block(load_fseries(args.fs_vspecialist), "V2M")
    main = metric_block(load_fseries(args.fs_mainvt), "VT2M")
    residual = metric_block(load_fseries(args.fs_residual), "VT2M")

    rows = []
    for spec in _specs():
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        main_key, main_raw, main_err = error_value(main, spec["aliases"], spec["higher"])
        r_key, r_raw, r_err = error_value(residual, spec["aliases"], spec["higher"])
        best_err = min(v_err, r_err)
        best_src = "V specialist" if v_err <= r_err else "Residual VT"
        category = ("synergy" if r_err - best_err < -args.tol
                    else "interference" if r_err - best_err > args.tol else "keep or select")
        rows.append({
            **spec,
            "metric": main_key if main_key == r_key else "%s / %s" % (main_key, r_key),
            "V_specialist": round(v_raw, 6), "main_VT": round(main_raw, 6),
            "residual_VT": round(r_raw, 6),
            "delta_mainVT_minus_vspec": round(main_err - v_err, 6),
            "delta_residual_minus_vspec": round(r_err - v_err, 6),
            "delta_residual_minus_mainVT": round(r_err - main_err, 6),
            "category": category,
            "values": {"V specialist": v_raw, "Main VT": main_raw, "Residual VT": r_raw},
        })

    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "residual_table.csv", [
        {k: val for k, val in row.items() if k not in ("values",)}
        for row in rows
    ])
    series = ["V specialist", "Main VT", "Residual VT"]
    render_figure(
        args.out / "residual_series.png", rows, series,
        {"V specialist": V_COLOR, "Main VT": MAIN_COLOR, "Residual VT": "#0f7a52"},
        "Test3 · frozen-V base, main VT and residual VT",
        layout="stacked",
        label_colors=domain_label_colors(rows),
        panel_titles=["Residual VT vs baselines"],
    )
    render_figure(
        args.out / "residual_panels.png", rows, ["V specialist", "Residual VT"],
        {"V specialist": V_COLOR, "Residual VT": "#0f7a52"},
        "Test3 · frozen-V base vs T-residual correction",
        panel_titles=["V specialist", "Residual VT"],
        panel_title_markers=[V_COLOR, "#0f7a52"],
        label_colors=domain_label_colors(rows),
    )
    summary = {
        "split": args.split,
        "synergy": sum(r["category"] == "synergy" for r in rows),
        "selection": sum(r["category"] == "keep or select" for r in rows),
        "interference": sum(r["category"] == "interference" for r in rows),
        "residual_vs_vspec": [
            {"row_id": r["row_id"], "delta": r["delta_residual_minus_vspec"]}
            for r in rows
        ],
        "headline": {
            "overall_delta_residual_minus_vspec": next(
                r["delta_residual_minus_vspec"] for r in rows if r["row_id"] == "overall"),
            "overall_delta_mainVT_minus_vspec": next(
                r["delta_mainVT_minus_vspec"] for r in rows if r["row_id"] == "overall"),
        },
    }
    (args.out / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overall = summary["headline"]
    print("wrote %s (residual-vs-vspec overall %.2fmm; mainVT-vs-vspec %.2fmm)"
          % (args.out, overall["overall_delta_residual_minus_vspec"],
             overall["overall_delta_mainVT_minus_vspec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
