#!/usr/bin/env python3
"""V8_Cv5VUG（不确定性专家门）真实评估对比：逐档（clean/020-100）× 逐指标。

参照 = 无门 Cv5（V8_Cv5_<fold>_s1 cells 的 clean + evalocc020-100）。
指标：PA-MPJPE / MPJPE / GlobalMPJPE（GT Anchored）/ FootSliding /
GroundPen / traj_ATE。输出 results/diagnostics/20261107_stage2_vig/vug_compare.json
+ 控制台表（VT2M 逐档 Δ(VUG−Cv5)，四折）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
LEVELS = ("clean", "020", "040", "060", "080", "100")
CONDS = ("VT2M", "V2M", "T2M")
METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE（GT Anchored）",
           "FootSliding（GT Anchored）", "GroundPen（GT Anchored）", "traj_ATE")
OUT = REPO / "results/diagnostics/20261107_stage2_vig"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def metrics(path: Path) -> dict:
    return load(path)["metrics"]


def vug_path(fold, level):
    name = "clean" if level == "clean" else "occ%s" % level
    return OUT / ("eval_vug_%s" % fold) / name / "metrics/test.json"


def cv5_path(fold, level):
    cell = ("V8_Cv5_%s_s1" % fold) if level == "clean" else \
        ("V8_Cv5_%s_s1_evalocc%s" % (fold, level))
    return REPO / "results/cells" / cell / "metrics/test.json"


def main():
    report = {"per_fold": {}}
    for fold in FOLDS:
        rec = {}
        for level in LEVELS:
            vp = vug_path(fold, level)
            if not vp.is_file():
                rec[level] = None
                continue
            vig = metrics(vp)
            c5 = metrics(cv5_path(fold, level))
            row = {}
            for cond in CONDS:
                row[cond] = {m: {"VUG": vig[cond][m], "Cv5": c5[cond][m],
                                 "d": vig[cond][m] - c5[cond][m]}
                             for m in METRICS}
            rec[level] = row
        report["per_fold"][fold] = rec
        print("== %s: VT2M Δ(VUG−Cv5) per level" % fold)
        print("  %-6s %8s %8s %8s %8s %8s %8s" %
              ("level", *[m.split("（")[0][:8] for m in METRICS]))
        for level in LEVELS:
            row = rec[level]
            if row is None:
                print("  %-6s pending" % level)
                continue
            print("  %-6s %s" % (level, " ".join("%+8.2f" % row["VT2M"][m]["d"]
                                                 for m in METRICS)))
    # 四折均值（可用折）
    print("== mean across folds: VT2M Δ(VUG−Cv5)")
    print("  %-6s %8s %8s %8s" % ("level", "PA-MPJPE", "MPJPE", "Global"))
    means = {}
    for level in LEVELS:
        ds = {m: [] for m in METRICS}
        for fold in FOLDS:
            row = report["per_fold"][fold].get(level)
            if row is None:
                continue
            for m in METRICS:
                ds[m].append(row["VT2M"][m]["d"])
        means[level] = {m: float(np.mean(ds[m])) if ds[m] else None for m in METRICS}
        print("  %-6s %8.2f %8.2f %8.2f"
              % (level, means[level]["PA-MPJPE"] or 0.0,
                 means[level]["MPJPE"] or 0.0,
                 means[level]["GlobalMPJPE（GT Anchored）"] or 0.0))
    report["mean"] = {lvl: {m: means[lvl][m] for m in METRICS} for lvl in LEVELS}
    (OUT / "vug_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % (OUT / "vug_compare.json"))


if __name__ == "__main__":
    main()
