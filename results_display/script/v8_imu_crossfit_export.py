#!/usr/bin/env python3
"""V8_Cv5 / V8_Cv7 + IMU fusion four-fold LSO export (woS13).

Protocol = results_display/Etest_crossfit_v8:
  folds fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8 (fS6S7 = S13-out)
  only test; each cell = four-fold mean±std (fold-level)
  metrics: PA / ankle-foot / MPJPE / Global / RootAbs / GroundPen / FootSliding.

Rows include the existing V8_Cv5 / V8_Cv7 bases as references, then the four
IMU variants (fusion 1 = raw108_imu; fusion 2 = raw96_phys_imu), each with the
V8_Cv5 gate and the V8_Cv7 no-gate decoder.

Inputs : results/cells/<model_id>/metrics/test.json
Outputs: results_display/Etest_crossfit_v8/E2_MainPerformance/
           Main_Table1_E2_V8_IMU.csv
           Main_Table2_E2_V8_IMU_Delta.csv
         results_display/Etest_crossfit_v8/detail/PerFold_V8_IMU.csv
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
    ("V8_Cv5", "V8_Cv5（双门，参考）"),
    ("V8_Cv5_IMU", "V8_Cv5 + IMU（融合1）"),
    ("V8_Cv5_PhysIMU", "V8_Cv5 + PhysIMU（融合2）"),
    ("V8_Cv7", "V8_Cv7（无门，参考）"),
    ("V8_Cv7_IMU", "V8_Cv7 + IMU（融合1）"),
    ("V8_Cv7_PhysIMU", "V8_Cv7 + PhysIMU（融合2）"),
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
    summary_rows, per_fold_rows, delta_rows = [], [], []
    for fam, label in FAMS:
        fm = {f: load(fam, f) for f in FOLDS}
        for fold in FOLDS:
            for cond, cond_label in CONDS:
                row = [fam, label, fold, cond_label]
                row += [val(fm[fold], cond, k) for k, _ in METRICS]
                per_fold_rows.append(row)
        for cond, cond_label in CONDS:
            row = [fam, label, cond_label,
                   sum(m is not None for m in fm.values())]
            row += [mean_std([val(fm[f], cond, k) for f in FOLDS])
                    for k, _ in METRICS]
            summary_rows.append(row)
        for base in ("V2M", "T2M"):
            row = [fam, label, "ΔVT−%s" % ("V" if base == "V2M" else "T")]
            for key, _ in METRICS:
                diffs = []
                for f in FOLDS:
                    a = val(fm[f], "VT2M", key)
                    b = val(fm[f], base, key)
                    if a is not None and b is not None:
                        diffs.append(a - b)
                row.append(mean_std(diffs))
            delta_rows.append(row)

    header = ["模型", "标签", "条件", "n_folds"] + [d for _, d in METRICS]
    write_csv(MAIN / "Main_Table1_E2_V8_IMU.csv", header, summary_rows)
    write_csv(MAIN / "Main_Table2_E2_V8_IMU_Delta.csv",
              ["模型", "标签", "对比"] + [d for _, d in METRICS], delta_rows)
    write_csv(DETAIL / "PerFold_V8_IMU.csv",
              ["模型", "标签", "折", "条件"] + [d for _, d in METRICS],
              per_fold_rows)


if __name__ == "__main__":
    main()
