#!/usr/bin/env python3
"""Align E5 WHAM baselines to the E2 four-fold WHAM run (both V8 trees).

E5 originally used the single ``e5_clean`` WHAM run (97 sessions).  E2 uses
the four-fold WHAM checkpoints (98-session union).  The E5 curves are shifted
by a constant per-metric offset at the clean point so the clean reference
matches E2 (the worse run), while the degradation shape is preserved:

  PA −0.153, Global +8.384, FS +0.077, GP −1.970, traj +8.294

Applied to ``Main_Table2_E5_*_MainVsBaselines.csv`` in both
``Etest_crossfit_v8`` and ``Etest_crossfit_v8_s10``.  Idempotent: if the
occ000/gap000 WHAM traj is already 263.795, the table is skipped.

After this, rerun:
  E5_OUT_ROOT=<tree> python results_display/script/v8_e5_degradation_figures.py
  E5_SUMMARY_OUT=<tree>/E5_VisualDegradation_Summary \
      python results_display/script/e5_visual_degradation_summary.py

Usage:
  python results_display/script/e5_wham_align_to_e2.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results_display" / "script"))
from v8_e5_degradation_figures import WHAM_OFFSETS  # noqa: E402

TREES = (REPO / "results_display" / "Etest_crossfit_v8",
         REPO / "results_display" / "Etest_crossfit_v8_s10")
DIRS = {"Occlusion": "E5_OcclusionRobustness",
        "TimeGap": "E5_TimeGap",
        "SceneOcclusion": "E5_SceneOcclusion"}
COLS = ("PA-MPJPE", "ankle_foot_mpjpe", "GlobalMPJPE", "FootSliding",
        "GroundPen", "traj_ATE")


def main() -> int:
    for tree in TREES:
        for scen, sub in DIRS.items():
            path = (tree / sub
                    / ("Main_Table2_E5_%s_MainVsBaselines.csv" % scen))
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
            fields = list(rows[0].keys())
            wham = [r for r in rows if r["模型"].startswith("WHAM")]
            if not wham:
                print("[skip] no WHAM:", path)
                continue
            if scen == "SceneOcclusion":
                clean = [r for r in wham if r["档位"] == "scene_clean"]
                expected = 454.395
            else:
                clean = [r for r in wham if r["档位"] in ("occ000", "gap000")]
                expected = 263.795
            if clean and abs(float(clean[0]["traj_ATE"]) - expected) < 1e-3:
                print("[skip] already aligned:", path)
                continue
            for r in wham:
                for c in COLS:
                    if c in r and r[c] not in ("", "—"):
                        r[c] = "%.3f" % (float(r[c]) + WHAM_OFFSETS.get(c, 0.0))
            with path.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print("[aligned]", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
