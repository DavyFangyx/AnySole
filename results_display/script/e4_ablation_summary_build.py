#!/usr/bin/env python3
"""Build the single E4 ablation summary table from the two source tables.

The E4 experiment answers two design questions (gate structure and training
condition).  For publication we keep one main summary table:

  E4_ComponentAblation/Main_Table1_E4_Ablation_Summary.csv

and move the two per-question source tables plus the 10-metric full table to
``detail/``.  The summary keeps the six decision-relevant metrics:
PA-MPJPE, MPJPE, GlobalMPJPE, traj_ATE, GroundPen, FootSliding, with the
V / VT / Δ(VT−V) rows for each model.

Usage:
  python results_display/script/e4_ablation_summary_build.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E4 = Path(os.environ.get(
    "E4_OUT", REPO / "results_display" / "Etest_crossfit_v8" / "E4_ComponentAblation"))
SOURCES = (
    ("门控效果", E4 / "detail" / "Main_Table1_E4_GateEffect.csv"),
    ("训练条件", E4 / "detail" / "Main_Table2_E4_TrainingCondition.csv"),
)
COLS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
        "GroundPen", "FootSliding")


def main() -> int:
    rows = []
    for group, path in SOURCES:
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            cond = r["条件"]
            kou = ("T+BiFoot GT-start (RootAbs 3D RMS)" if cond == "U@T"
                   else "VT−V (cell)" if cond.startswith("Δ")
                   else "cell self-anchor (RootAbs 3D RMS)")
            rows.append([group, r["模型"], cond, kou] + [r[c] for c in COLS])
    out = E4 / "Main_Table1_E4_Ablation_Summary.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["实验组", "模型", "条件", "口径"] + list(COLS))
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
