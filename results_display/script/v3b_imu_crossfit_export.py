#!/usr/bin/env python3
"""Export the V3_Bv1 IMU / PhysIMU four-fold LSO (woS13) main tables.

Protocol matches results_display/Etest_crossfit_v8:
  folds = fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8 (fS6S7 = S13-out)
  only test; each cell = four-fold mean±std (fold-level, not session-weighted)
  metrics: PA / ankle-foot / MPJPE / Global / RootAbs / GroundPen / FootSliding.

Inputs:
  results/cells/<model_id>_<fold>_s1/metrics/test.json

Outputs (new files, existing V8 tables untouched):
  results_display/Etest_crossfit_v8/E2_MainPerformance/Main_Table1_E2_IMU_Fusion.csv
  results_display/Etest_crossfit_v8/E2_MainPerformance/Main_Table2_E2_IMU_Fusion_Delta.csv
  results_display/Etest_crossfit_v8/detail/PerFold_IMU_Fusion.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = REPO / "results_display" / "Etest_crossfit_v8"
MAIN = OUT / "E2_MainPerformance"
DETAIL = OUT / "detail"

FAMS = (
    ("V3_Bv1_IMU", "IMU-S2M（门控）"),
    ("V3_Bv1_PhysIMU", "PhysIMU（门控）"),
    ("V3_Bv1_IMU_NG", "IMU-S2M（无门控）"),
    ("V3_Bv1_PhysIMU_NG", "PhysIMU（无门控）"),
)
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
CONDS = (("VT2M", "U@VT"), ("V2M", "U@V"), ("T2M", "U@T"))
METRICS = (
    ("PA-MPJPE", "PA-MPJPE"),
    ("PA-MPJPE_anklefoot", "PA-MPJPE_anklefoot"),
    ("MPJPE", "MPJPE"),
    ("GlobalMPJPE", "GlobalMPJPE"),
    ("RootAbs", "traj_ATE(RootAbs)"),
    ("GroundPen", "GroundPen"),
    ("FootSliding", "FootSliding"),
    ("Jitter_pred", "Jitter_pred"),
    ("contact_f1", "contact_f1"),
)


def strip_anchor(metrics: dict) -> dict:
    out = {}
    for cond, block in metrics.items():
        if not isinstance(block, dict):
            out[cond] = block
            continue
        default = {str(k): v for k, v in block.items()
                   if not str(k).endswith(("（GT Anchored）", " (anchored)"))}
        anchored = {}
        for k, v in block.items():
            name = str(k)
            for suffix in ("（GT Anchored）", " (anchored)"):
                if name.endswith(suffix):
                    anchored[name[:-len(suffix)]] = v
                    break
        out[cond] = {**anchored, **default}
    return out


def load(fam: str, fold: str) -> dict | None:
    p = CELLS / ("%s_%s_s1" % (fam, fold)) / "metrics" / "test.json"
    if not p.is_file():
        return None
    return strip_anchor(json.loads(p.read_text(encoding="utf-8"))["metrics"])


def val(m: dict | None, cond: str, key: str):
    if not m:
        return None
    v = m.get(cond, {}).get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def mean_std(vals) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return "%.3f" % vals[0]
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> None:
    summary_rows = []
    per_fold_rows = []
    delta_rows = []
    for fam, label in FAMS:
        fold_metrics = {f: load(fam, f) for f in FOLDS}
        # per-fold detail
        for fold in FOLDS:
            for cond, cond_label in CONDS:
                row = [fam, label, fold, cond_label]
                row += [val(fold_metrics[fold], cond, k) for k, _ in METRICS]
                per_fold_rows.append(row)
        # four-fold mean±std
        for cond, cond_label in CONDS:
            row = [fam, label, cond_label, sum(m is not None for m in fold_metrics.values())]
            row += [mean_std([val(fold_metrics[f], cond, k) for f in FOLDS])
                    for k, _ in METRICS]
            summary_rows.append(row)
        # Δ(VT−V) and Δ(VT−T), per-fold then mean±std
        for base in ("V2M", "T2M"):
            row = [fam, label, "ΔVT−%s" % ("V" if base == "V2M" else "T")]
            for key, _ in METRICS:
                diffs = []
                for f in FOLDS:
                    a = val(fold_metrics[f], "VT2M", key)
                    b = val(fold_metrics[f], base, key)
                    if a is not None and b is not None:
                        diffs.append(a - b)
                row.append(mean_std(diffs))
            delta_rows.append(row)

    header = ["模型", "标签", "条件", "n_folds"] + [disp for _, disp in METRICS]
    write_csv(MAIN / "Main_Table1_E2_IMU_Fusion.csv", header, summary_rows)
    delta_header = ["模型", "标签", "对比"] + [disp for _, disp in METRICS]
    write_csv(MAIN / "Main_Table2_E2_IMU_Fusion_Delta.csv", delta_header, delta_rows)
    per_fold_header = ["模型", "标签", "折", "条件"] + [disp for _, disp in METRICS]
    write_csv(DETAIL / "PerFold_IMU_Fusion.csv", per_fold_header, per_fold_rows)


if __name__ == "__main__":
    main()
