#!/usr/bin/env python3
"""Replace the T rows of E2/E4 with the shared BiFoot trajectory branch.

The main model's T branch has no trajectory head; BiFoot is its dedicated
trajectory branch.  Every V8 model/ablation T row is therefore

    its own T2M pose (PA/MPJPE/MPJRE/ROD unchanged)
  + the shared BiFoot world-frame trajectory (one pass per fold, reused)

Trajectory metrics are the E2 T-block protocol: GT first-frame position anchor,
world-frame per-frame displacement accumulated directly, 3D mean Global/traj;
for the multi-condition tables whose `traj_ATE` column is RootAbs (3D RMS) the
3D-RMS variant is written so the column meaning is preserved.

Inputs: results/diagnostics/20261111_variant_sweep/clean_e2_t_bifoot/
Outputs: E2_MainPerformance and E4_ComponentAblation CSVs, rewritten in place.

Usage:
  python results_display/script/e2_t_bifoot_table_replace.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "results" / "diagnostics" / "20261111_variant_sweep"
E2 = Path(os.environ.get(
    "E2_OUT", REPO / "results_display" / "Etest_crossfit_v8" / "E2_MainPerformance"))
E4 = Path(os.environ.get(
    "E4_OUT", REPO / "results_display" / "Etest_crossfit_v8" / "E4_ComponentAblation"))
METRICS_SUBDIR = os.environ.get("E2_METRICS_SUBDIR", "").strip()
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = ("V8_Cv5", "V8_Cv7", "V8_Cv4", "V8_Cv2", "V8_Cv6")
TRAJ_KEYS = ("GlobalMPJPE", "W-MPJPE100", "traj_ATE", "GroundPen",
             "FootSliding", "Jitter_pred")


def load_model(model):
    per = {}
    for fold in FOLDS:
        path = DIAG / "clean_e2_t_bifoot" / model / ("clean_e2_%s.json" % fold)
        rows = json.loads(path.read_text(encoding="utf-8"))
        w = np.asarray([r["n_valid"] for r in rows], float)
        per[fold] = {k: float(np.average([r[k] for r in rows], weights=w))
                     for k in TRAJ_KEYS + ("traj_ATE_3d_rms",)}
    return per


def ms(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return "%.3f±%.3f" % (a.mean(), a.std())


def mean_of(per, key):
    return float(np.mean([per[f][key] for f in FOLDS]))


def ms_of(per, key):
    return ms([per[f][key] for f in FOLDS])




def cell_pose(model):
    """Per-fold T2M pose/ROD means; returns mean±std strings."""
    vals = {k: [] for k in ("PA-MPJPE", "MPJPE", "RootOrientationDrift")}
    for fold in FOLDS:
        root = (REPO / "results" / "cells" / ("%s_%s_s1" % (model, fold))
                / "metrics")
        if METRICS_SUBDIR:
            root = root / METRICS_SUBDIR
        block = json.loads((root / "test.json")
                           .read_text(encoding="utf-8"))["metrics"]["T2M"]
        for k in vals:
            vals[k].append(float(block[k]))
    return {k: ms(v) for k, v in vals.items()}


def replace_main_table1(data):
    path = E2 / "Main_Table1_E2_MainPerformance.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    cv5 = data["V8_Cv5"]
    for row in body:
        if row[idx["条件"]] != "U@T" or not row[idx["模型"]].startswith("V8_Cv5"):
            continue
        row[idx["GlobalMPJPE"]] = ms_of(cv5, "GlobalMPJPE")
        row[idx["traj_ATE"]] = ms_of(cv5, "traj_ATE")
        row[idx["GroundPen"]] = ms_of(cv5, "GroundPen")
        row[idx["FootSliding"]] = ms_of(cv5, "FootSliding")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e2] %s" % path)


def replace_main_table2(data):
    path = E2 / "Main_Table2_E2_T_Official.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    keep = []
    for row in body:
        label = row[idx["模型"]]
        if (label.startswith("Step2Motion") or "消融" in label
                or "legacy" in label):
            keep.append(row)
    t_rows = []
    for model in MODELS:
        per = data[model]
        pose = cell_pose(model)
        row = [""] * len(header)
        row[idx["模型"]] = model + "（T+BiFoot, position 同口径）"
        row[idx["条件"]] = "U@T"
        row[idx["n"]] = "4"
        row[idx["PA-MPJPE"]] = pose["PA-MPJPE"]
        row[idx["MPJPE"]] = pose["MPJPE"]
        row[idx["GlobalMPJPE"]] = ms_of(per, "GlobalMPJPE")
        row[idx["traj_ATE"]] = ms_of(per, "traj_ATE")
        row[idx["RootOrientationDrift"]] = pose["RootOrientationDrift"]
        row[idx["GroundPen"]] = ms_of(per, "GroundPen")
        row[idx["FootSliding"]] = ms_of(per, "FootSliding")
        row[idx["W-MPJPE100"]] = ms_of(per, "W-MPJPE100")
        row[idx["Jitter_pred"]] = ms_of(per, "Jitter_pred")
        row[idx["口径"]] = ("GT 首帧位置；世界系直接累加；3D mean；"
                            "T+BiFoot（共享轨迹）")
        t_rows.append(row)
    out = [header] + t_rows + keep
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(out)
    print("[e2] %s" % path)


def find_model(label):
    for m in MODELS:
        if m in label:
            return m
    return None


def replace_e2_full_table1(data):
    path = E2 / "detail" / "Full_Table1_E2_Crossfit_Main_test.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        if row[idx["条件"]] != "U@T":
            continue
        model = find_model(row[idx["模型"]])
        if model is None:
            continue
        fold = row[idx["折"]]
        per = data[model]
        if fold in per:
            for k in ("GlobalMPJPE", "W-MPJPE100", "GroundPen",
                      "FootSliding", "Jitter_pred"):
                row[idx[k]] = "%.3f" % per[fold][k]
            row[idx["traj_ATE"]] = "%.3f" % per[fold]["traj_ATE_3d_rms"]
        else:
            for k in ("GlobalMPJPE", "W-MPJPE100", "GroundPen",
                      "FootSliding", "Jitter_pred"):
                row[idx[k]] = ms_of(per, k)
            row[idx["traj_ATE"]] = ms_of(per, "traj_ATE_3d_rms")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e2] %s" % path)


def replace_e2_performance_t(data):
    path = E2 / "detail" / "Full_Table2_E2_Performance_T.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        label = row[idx["模型"]]
        if "legacy" in label:
            continue
        model = find_model(label)
        if model is None:
            continue
        per = data[model]
        row[idx["模型"]] = "U@T（%s + BiFoot）" % model
        row[idx["GlobalMPJPE"]] = ms_of(per, "GlobalMPJPE")
        row[idx["W-MPJPE100"]] = ms_of(per, "W-MPJPE100")
        row[idx["traj_ATE"]] = ms_of(per, "traj_ATE")
        row[idx["GroundPen"]] = ms_of(per, "GroundPen")
        row[idx["FootSliding"]] = ms_of(per, "FootSliding")
        row[idx["Jitter_pred"]] = ms_of(per, "Jitter_pred")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e2] %s" % path)


def replace_e2_summary(data):
    path = E2 / "detail" / "Full_Table4_E2_Performance_Summary.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        label = row[idx["模型"]]
        if "U@T" not in label:
            continue
        model = find_model(label)
        if model is None:
            continue
        per = data[model]
        for k in ("GlobalMPJPE", "W-MPJPE100", "GroundPen",
                  "FootSliding", "Jitter_pred"):
            row[idx[k]] = ms_of(per, k)
        row[idx["traj_ATE"]] = ms_of(per, "traj_ATE_3d_rms")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e2] %s" % path)


def replace_e4(path, data):
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        if not row[idx["条件"]].startswith("U@T"):
            continue
        model = find_model(row[idx["模型"]])
        if model is None:
            continue
        per = data[model]
        for k in ("GlobalMPJPE", "W-MPJPE100", "GroundPen",
                  "FootSliding", "Jitter_pred"):
            row[idx[k]] = ms_of(per, k)
        row[idx["traj_ATE"]] = ms_of(per, "traj_ATE_3d_rms")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e4] %s" % path)


def main() -> int:
    data = {m: load_model(m) for m in MODELS}
    replace_e2_full_table1(data)
    replace_e2_performance_t(data)
    replace_e2_summary(data)
    replace_main_table1(data)
    replace_main_table2(data)
    for name in ("Main_Table1_E4_GateEffect.csv",
                 "Main_Table2_E4_TrainingCondition.csv",
                 "Full_Table1_E4_Ablation_AllMetrics.csv"):
        replace_e4(E4 / "detail" / name, data)
    import subprocess
    subprocess.run([sys.executable, str(REPO / "results_display" / "script" / "e4_ablation_summary_build.py")], check=True)
    print("\nCv5 T+BiFoot (full clean, mean±std):")
    for k in TRAJ_KEYS + ("traj_ATE_3d_rms",):
        print("  %-18s %s" % (k, ms_of(data["V8_Cv5"], k)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
