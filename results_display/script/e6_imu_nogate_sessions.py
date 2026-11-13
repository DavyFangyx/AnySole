#!/usr/bin/env python3
"""Per-session E6 phase figures for the IMU no-gate group (V8_Cv7_PhysIMU).

Reuses e6_v8_best_sessions.render_session: two panels (left/right foot phases),
x = time (s), y = Δ = VT − V per-frame full-body PA-MPJPE (mm), thin raw +
thick 20-frame SMA, phase bands at the bottom.  One PNG per (session, fold).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import e6_v8_best_sessions as base  # noqa: E402

OUT = (base.REPO / "results_display" / "Etest_crossfit_v8_branches" / "IMU"
       / "E6_ContactPhaseAnalysis")
PER = OUT / "per_session"
base.MODELS = ("V8_Cv7_PhysIMU",)
base.MODEL_LABEL = {"V8_Cv7_PhysIMU": "V8_Cv7_PhysIMU (no gate)"}
base.MODEL_COLOR = {"V8_Cv7_PhysIMU": "#2a78d6"}
base.PER_SESSION = PER
base.OUT = OUT


def main() -> int:
    PER.mkdir(parents=True, exist_ok=True)
    cache = base.Cache()
    series = base.collect(cache)          # collect uses patched MODELS
    rows = []
    ylim = (-30.0, 30.0)                  # same as the E6 best-session figure
    # one figure per session: keep the cell with the most valid frames
    by_sid = {}
    for key, data in sorted(series.items()):
        sid = key[0]
        if sid not in by_sid or data["n_frames"] > by_sid[sid][1]["n_frames"]:
            by_sid[sid] = (key, data)
    for sid, ((_, fold, model), data) in sorted(by_sid.items()):
        action = base.ACTION_GROUPS.get(base.action_of(sid))
        if action is None:
            continue
        pick = {"action": action, "sid": sid, "fold": fold}
        path = base.render_session(action, pick, series, ylim)
        rows.append([action, sid, fold, round(data["mean"], 3),
                     round(data["harm"], 3), round(data["gain"], 3),
                     data["n_frames"], str(path.relative_to(base.REPO))])
    path = OUT / "PerSession_E6_IMU_NoGate.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["action", "session", "fold", "mean_delta_mm",
                    "harm_mm", "gain_mm", "n_frames", "figure"])
        w.writerows(rows)
    print("wrote", len(rows), "per-session figures to", PER)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
