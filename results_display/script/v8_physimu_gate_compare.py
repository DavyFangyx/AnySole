#!/usr/bin/env python3
"""Gate vs no-gate comparison for the two PhysIMU variants (woS13 4-fold test).

Cv5_PhysIMU = V8_Cv5 dual T gate + PhysIMU
Cv7_PhysIMU = V8_Cv7 no gate + PhysIMU

Outputs
  results_display/Etest_crossfit_v8/E2_MainPerformance/PhysIMU_GateVsNoGate.csv
  results_display/Etest_crossfit_v8/detail/PerFold_PhysIMU_GateVsNoGate.csv
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

A = "V8_Cv5_PhysIMU"
B = "V8_Cv7_PhysIMU"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
CONDS = (("VT2M", "U@VT"), ("V2M", "U@V"), ("T2M", "U@T"))

# (key, display, direction) direction: -1 lower better, +1 higher better
METRICS = (
    ("PA-MPJPE", "PA-MPJPE", -1),
    ("PA-MPJPE_anklefoot", "PA-MPJPE_anklefoot", -1),
    ("PA-MPJPE_lower", "PA-MPJPE_lower", -1),
    ("PA-MPJPE_upper", "PA-MPJPE_upper", -1),
    ("MPJPE", "MPJPE", -1),
    ("MPJRE", "MPJRE", -1),
    ("GlobalMPJPE", "GlobalMPJPE", -1),
    ("W-MPJPE100", "W-MPJPE100", -1),
    ("RootAbs", "traj_ATE(RootAbs)", -1),
    ("RootOrientationDrift", "RootOrientationDrift", -1),
    ("GroundPen", "GroundPen", -1),
    ("GroundPenFrac", "GroundPenFrac", -1),
    ("FootSliding", "FootSliding", -1),
    ("Jitter_pred", "Jitter_pred", -1),
    ("contact_f1", "contact_f1", +1),
    ("contact_balanced_acc", "contact_balanced_acc", +1),
    ("air_recall", "air_recall", +1),
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


def val(m, cond, key):
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
    fa = {f: load(A, f) for f in FOLDS}
    fb = {f: load(B, f) for f in FOLDS}
    summary, per_fold = [], []
    for cond, clabel in CONDS:
        for key, disp, direction in METRICS:
            va = [val(fa[f], cond, key) for f in FOLDS]
            vb = [val(fb[f], cond, key) for f in FOLDS]
            diffs = [a - b for a, b in zip(va, vb) if a is not None and b is not None]
            wins = sum(1 for d in diffs if (d < 0 if direction < 0 else d > 0))
            summary.append([
                clabel, disp, mean_std(va), mean_std(vb),
                mean_std(diffs), "%d/%d" % (wins, len(diffs)),
            ])
            for f in FOLDS:
                per_fold.append([
                    clabel, disp, f, val(fa[f], cond, key),
                    val(fb[f], cond, key),
                    (val(fa[f], cond, key) - val(fb[f], cond, key))
                    if val(fa[f], cond, key) is not None and val(fb[f], cond, key) is not None
                    else None,
                ])
    write_csv(MAIN / "PhysIMU_GateVsNoGate.csv",
              ["条件", "指标", "Cv5_PhysIMU(门控)", "Cv7_PhysIMU(无门)",
               "Δ(Cv5−Cv7)", "Cv5胜"], summary)
    write_csv(DETAIL / "PerFold_PhysIMU_GateVsNoGate.csv",
              ["条件", "指标", "折", "Cv5_PhysIMU", "Cv7_PhysIMU", "Δ(Cv5−Cv7)"],
              per_fold)


if __name__ == "__main__":
    main()
