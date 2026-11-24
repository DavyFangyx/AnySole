#!/usr/bin/env python3
"""Q14 纹理遮挡对比导出：tex vs black（2D 各档）与 concrete vs gray（场景）。

主模型 V8_Cv5：cells `<fold>_s1_evalocc{NNN}[_tex_{asset}]` / `_evalscene[clean|tex]`
的 metrics/overlap_s10/test.json（metrics.V2M / metrics.VT2M）。
基线：`results/baselines/<Model>/predictions/e5_occ_lower{NNN}[_tex_{asset}]`
与 `e5_scene[_tex]` 的 eval_motion/metrics/test_comparison.json。

输出到 Supplement/06_实现与可复现性/Q14.../实验结果/tex_occlusion/：
- occ_tex_vs_black.csv      主模型 4 折 mean±std：black vs tex（三资产 × 020/050/090）
- scene_tex_vs_gray.csv     主模型 scene clean/occl 配对：gray vs concrete
- baselines_tex.csv         基线关键档（WHAM/GVHMR/MotionPRO）
- summary.md                结论汇总（tex 与主协议的偏差）
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
BASELINES = REPO / "results" / "baselines"
Q14 = (REPO / "results_display" / "Etest_crossfit_physimu_vgate" / "Supplement"
       / "06_实现与可复现性" / "Q14：视觉退化的实现（三种退化）" / "实验结果"
       / "tex_occlusion")

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
LEVELS = ("010", "020", "030", "040", "050", "060", "070", "080", "090")
ASSETS = ("wood", "fabric", "crate")
CONFIGS = ("V2M", "VT2M")

MAIN_METRICS = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "MPJPE_anklefoot",
                "PA-MPJPE_anklefoot", "traj_ATE"]
BASELINE_METRICS = ["pa_mpjpe_mm", "global_mpjpe_mm", "ankle_foot_mpjpe"]


def main_model_metrics(cond: str, fold: str) -> dict | None:
    """Paper main model V8_IMU_Residual per-condition fold JSON
    (formal eval layout: results/diagnostics/<root>/V8_IMU_Residual/<cond>/<fold>.json)."""
    roots = [
        REPO / "results/diagnostics/20261114_imu_eval_v2_all",     # black reference
        REPO / "results/diagnostics/20261120_q14_tex_eval_v8imu",  # tex variants
    ]
    for root in roots:
        path = root / "V8_IMU_Residual" / cond / ("%s.json" % fold)
        if path.is_file():
            data = json.loads(path.read_text())
            return data.get("metrics", {})
    return None


def mean_std(values: list) -> str:
    if not values:
        return ""
    return "%.1f±%.1f" % (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def baseline_metrics(model: str, variant: str) -> dict | None:
    """Four-fold summary from the Q14 unified eval (same script for tex and
    black, pinned frames + canonical solver)."""
    path = (REPO / "results" / "diagnostics" / "20261120_q14_tex_unified"
            / model / variant / "pinned_fold" / "fourfold_summary.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Q14)
    p.add_argument("--model", default="V8_Cv5")
    p.add_argument("--baselines", default="WHAM,GVHMR,MotionPRO")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 2D strip: black vs tex, per fold ----
    rows = []
    for lvl in LEVELS:
        for asset in ASSETS:
            for metric in MAIN_METRICS:
                black, tex = [], []
                for fold in FOLDS:
                    bm = main_model_metrics("occ%s" % lvl, fold)
                    tm = main_model_metrics("occ%s_tex_%s" % (lvl, asset), fold)
                    if bm and metric in bm["VT2M"]:
                        black.append(bm["VT2M"][metric])
                    if tm and metric in tm["VT2M"]:
                        tex.append(tm["VT2M"][metric])
                if black or tex:
                    rows.append({
                        "level": lvl, "asset": asset, "config": "VT2M",
                        "metric": metric,
                        "black": mean_std(black), "tex": mean_std(tex),
                        "n_folds": "%d/%d" % (len(tex), len(black)),
                    })
    with (args.out / "occ_tex_vs_black.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "asset", "config", "metric",
                                           "black", "tex", "n_folds"])
        w.writeheader(); w.writerows(rows)

    # ---- scene: gray vs concrete (same split, clean pairing) ----
    rows = []
    for metric in MAIN_METRICS:
        for label, cond in (("scene_clean", "sceneocc_clean"),
                            ("gray", "sceneocc"),
                            ("concrete", "scene_tex")):
            vals = []
            for fold in FOLDS:
                om = main_model_metrics(cond, fold)
                if om and metric in om["VT2M"]:
                    vals.append(om["VT2M"][metric])
            rows.append({"label": label, "config": "VT2M", "metric": metric,
                         "value": mean_std(vals), "n_folds": str(len(vals))})
    with (args.out / "scene_tex_vs_gray.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "config", "metric", "value", "n_folds"])
        w.writeheader(); w.writerows(rows)

    # ---- baselines: tex at key levels (fourfold mean±std from unified eval) ----
    rows = []
    for model in [m for m in args.baselines.split(",") if m]:
        variants = ["e5_occ_lower050_tex_%s" % a for a in ASSETS] + \
                   ["e5_occ_lower090_tex_%s" % a for a in ASSETS] + \
                   ["e5_scene_tex"]
        refs = {"e5_occ_lower050_tex_%s" % a: "e5_occ_lower050" for a in ASSETS}
        refs.update({"e5_occ_lower090_tex_%s" % a: "e5_occ_lower090" for a in ASSETS})
        refs["e5_scene_tex"] = "e5_scene"
        for variant in variants:
            tm = baseline_metrics(model, variant)
            bm = baseline_metrics(model, refs[variant])
            for metric in BASELINE_METRICS:
                if not tm and not bm:
                    continue
                def fmt(v):
                    if not v or "mean" not in v:
                        return ""
                    return "%.1f±%.1f" % (v["mean"], v["std"])
                rows.append({
                    "model": model, "variant": variant, "ref": refs[variant],
                    "metric": metric,
                    "ref_value": fmt(bm.get(metric) if bm else None),
                    "tex_value": fmt(tm.get(metric) if tm else None),
                })
    with (args.out / "baselines_tex.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "variant", "ref", "metric",
                                           "ref_value", "tex_value"])
        w.writeheader(); w.writerows(rows)

    print("written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
