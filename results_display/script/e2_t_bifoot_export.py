#!/usr/bin/env python3
"""E2 T block: same-protocol comparison of T+BiFoot against Step2Motion.

One statistical protocol for every row compared in the T block:

  * frame set      - the V8 cell test frame ids of the same fold;
  * aggregation    - frame-weighted (micro) with each read's valid-frame
                     count, matching the cell / baseline summaries;
  * anchor         - GT first-frame root position only (no yaw rotation);
  * trajectory     - the S2M/BiFoot adapter target is the world-frame per-frame
                     root displacement, so both trajectories are reconstructed
                     by direct cumulative sum from the GT first root
                     (``bifoot_clean_e2_row.direct_traj``).  The old
                     ``frame_local`` + ``calibrate_gt_course`` yaw transform is
                     not applied: it rotated an already world-aligned
                     prediction and inflated both arms;
  * metric set     - the official Step2Motion definitions: GlobalMPJPE 3D
                     mean, root ATE 3D mean, W-MPJPE100, GroundPen,
                     FootSliding, Jitter, PA-MPJPE / MPJPE.

Under this protocol the S2M archive reproduces the official published summary
exactly (Global 397.935, root ATE 340.884, W100 444.059, GP 64.300,
FS 7.119, Jitter 368.147), which is the pipeline validation.

Outputs (results_display/Etest_crossfit_v8/E2_MainPerformance/):
  Main_Table1_E2_MainPerformance.csv       same-protocol S2M and T+BiFoot rows
  Main_Table2_E2_T_Official.csv            S2M / T+BiFoot / trajectory ablation / legacy
  detail/Full_Table6_E2_T_BiFoot_PerFold.csv  per-fold detail for every arm

Usage:
  python results_display/script/e2_t_bifoot_export.py
  # s10 口径（2026-11-12）：姿态读 overlap_s10，BiFoot 轨迹列不变
  python results_display/script/e2_t_bifoot_export.py --metrics-subdir overlap_s10 \
      --out-dir results_display/Etest_crossfit_v8_s10/E2_MainPerformance
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "E2_MainPerformance"
DIAG = REPO / "results" / "diagnostics" / "20261111_variant_sweep"
CLEAN = DIAG / "clean_e2_direct"
CLEAN_S2M = DIAG / "clean_e2_s2m"
CELLS = REPO / "results" / "cells"
BASELINE = REPO / "results" / "baselines" / "Step2Motion" / "metrics"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
           "RootOrientationDrift", "GroundPen", "FootSliding")
COLS = METRICS + ("W-MPJPE100", "Jitter_pred")
CLEAN_KEYS = ("GlobalMPJPE", "W-MPJPE100", "traj_ATE", "GroundPen",
              "FootSliding", "Jitter_pred")
CELL_KEYS = {
    "PA-MPJPE": "PA-MPJPE",
    "MPJPE": "MPJPE",
    "RootOrientationDrift": "RootOrientationDrift",
    "GlobalMPJPE": "GlobalMPJPE",
    "traj_ATE": "RootAbs",
    "GroundPen": "GroundPen（GT Anchored）",
    "FootSliding": "FootSliding（GT Anchored）",
}
OFFICIAL_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "traj_ATE": "root_ate_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "GroundPen": "ground_pen_mm",
    "FootSliding": "foot_sliding_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}


def fold_mean_std(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return "%.3f±%.3f" % (a.mean(), a.std()) if len(a) > 1 else "%.3f" % a.mean()


def cell_fold(fold, metrics_subdir=None):
    root = CELLS / ("V8_Cv5_%s_s1" % fold) / "metrics"
    if metrics_subdir:
        root = root / metrics_subdir
    d = json.loads((root / "test.json")
                   .read_text(encoding="utf-8"))["metrics"]["T2M"]
    return {k: float(d[key]) for k, key in CELL_KEYS.items()}


def micro(rows, key):
    return float(np.average([r[key] for r in rows],
                            weights=[r["n_valid"] for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-subdir", default=None,
                    help="metrics 子目录：默认读 metrics/test.json；"
                    "传 overlap_s10 读 metrics/overlap_s10/test.json")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="输出目录，默认 %(default)s")
    args = ap.parse_args()
    out = Path(args.out_dir)
    per_fold = {}
    for fold in FOLDS:
        clean = json.loads((CLEAN / ("clean_e2_%s.json" % fold))
                           .read_text(encoding="utf-8"))
        s2m_full = json.loads((CLEAN_S2M / ("clean_e2_s2m_%s.json" % fold))
                              .read_text(encoding="utf-8"))
        official = json.loads((BASELINE / ("%s_comparison.json" % fold))
                              .read_text(encoding="utf-8"))["summary"]
        per_fold[fold] = {"clean": clean, "s2m_full": s2m_full,
                          "official": official,
                          "cell": cell_fold(fold, args.metrics_subdir)}

    arms = ("S2M-full", "S2M-traj", "BiFoot-traj")
    vals = {a: {k: [] for k in COLS} for a in arms}
    official_vals = {k: [] for k in COLS}

    for fold in FOLDS:
        d = per_fold[fold]
        clean, cell = d["clean"], d["cell"]
        for arm, tag in (("S2M", "S2M-traj"), ("BiFoot", "BiFoot-traj")):
            rows = [r for r in clean if r["arm"] == arm]
            for k in CLEAN_KEYS:
                vals[tag][k].append(micro(rows, k))
            # The two T2M-pose arms share the cell pose read.
            for k in ("PA-MPJPE", "MPJPE", "RootOrientationDrift"):
                vals[tag][k].append(cell[k])
        # S2M full model: own pose + own archive trajectory, position-only.
        rows = d["s2m_full"]
        for k in CLEAN_KEYS:
            vals["S2M-full"][k].append(micro(rows, k))
        for k in ("PA-MPJPE", "MPJPE"):
            vals["S2M-full"][k].append(micro(rows, k))
        vals["S2M-full"]["RootOrientationDrift"].append(
            float(d["official"]["root_orientation_drift_deg"]))
        for k, key in OFFICIAL_KEYS.items():
            official_vals[k].append(float(d["official"][key]))

    same = {k: fold_mean_std(vals["S2M-full"][k]) for k in COLS}
    ours = {k: fold_mean_std(vals["BiFoot-traj"][k]) for k in COLS}
    ablation = {k: fold_mean_std(vals["S2M-traj"][k]) for k in COLS}
    official = {k: fold_mean_std(official_vals[k]) for k in COLS}

    legacy = {k: fold_mean_std([per_fold[f]["cell"][k] for f in FOLDS])
              for k in METRICS}
    legacy["W-MPJPE100"] = "—"
    legacy["Jitter_pred"] = "—"

    # ---- Main_Table1: replace the two T-block system rows -----------------
    main_path = out / "Main_Table1_E2_MainPerformance.csv"
    rows = list(csv.DictReader(main_path.open(encoding="utf-8-sig")))
    header = list(rows[0].keys())
    for r in rows:
        if r["模型"].startswith("Step2Motion"):
            r["模型"] = "Step2Motion（position 单锚同口径）"
            for k in METRICS:
                r[k] = same[k]
        if r["模型"].startswith("V8_Cv5") and r["条件"] == "U@T":
            r["模型"] = "V8_Cv5（T+BiFoot, position 同口径）"
            for k in METRICS:
                r[k] = ours[k]
    with main_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print("[csv] %s" % main_path)

    # ---- Main_Table2: full T block ----------------------------------------
    t_path = out / "Main_Table2_E2_T_Official.csv"
    with t_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "条件", "n"] + list(COLS) + ["口径"])
        w.writerow(["Step2Motion（position 单锚）", "U@T", 4]
                   + [same[k] for k in COLS]
                   + ["GT 首帧位置；世界系直接累加；3D mean；官方汇总复现"])
        w.writerow(["V8_Cv5（T+BiFoot, position 同口径）", "U@T", 4]
                   + [ours[k] for k in COLS]
                   + ["同左；姿态取自 V8 T2M cell"])
        w.writerow(["V8_Cv5（T2M 姿态 + S2M 轨迹，消融）", "U@T", 4]
                   + [ablation[k] for k in COLS]
                   + ["同左；仅换轨迹臂"])
        w.writerow(["V8_Cv5（legacy T2M root head）", "U@T", 4]
                   + [legacy[k] for k in COLS]
                   + ["旧 T2M root 头，cell 自锚"])
    print("[csv] %s" % t_path)

    # ---- per-fold detail ---------------------------------------------------
    d_path = out / "detail" / "Full_Table6_E2_T_BiFoot_PerFold.csv"
    with d_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "arm"] + list(COLS))
        for fold in FOLDS:
            i = FOLDS.index(fold)
            for arm in arms:
                w.writerow([fold, arm] + ["%.3f" % vals[arm][k][i]
                                          for k in COLS])
    print("[csv] %s" % d_path)
    print("\n同口径 S2M vs T+BiFoot (ours−S2M):")
    for k in COLS:
        a = float(same[k].split("±")[0])
        b = float(ours[k].split("±")[0])
        print("  %-20s %10.3f  %10.3f  %+9.2f" % (k, a, b, b - a))
    print("\n轨迹消融 (T2M pose, S2M traj → BiFoot traj):")
    for k in COLS:
        a = float(ablation[k].split("±")[0])
        b = float(ours[k].split("±")[0])
        print("  %-20s %10.3f  %10.3f  %+9.2f" % (k, a, b, b - a))
    print("\n官方汇总复现校验 (同口径 S2M − 官方 summary):")
    for k in COLS:
        a = float(same[k].split("±")[0])
        b = float(official[k].split("±")[0])
        print("  %-20s %+0.6f" % (k, a - b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
