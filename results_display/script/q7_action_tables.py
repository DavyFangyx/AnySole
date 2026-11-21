#!/usr/bin/env python3
"""Q7 supplement: per-action comparison tables (rows = action, cols = arms).

Arms: BiFoot-Entry / BiFoot-Ramp / BiFoot-LERP / BiFoot-Spline / BiFoot-TTA.
Tasks: SceneOcc short (real VT anchors) and HalfPlane long (virtual anchors).

Sources
-------
- Entry/Ramp/TTA per-session 3D rows: the published Full_Table1 CSVs
  (results_display/Etest_crossfit_physimu_vgate/E9_Trajectory/{Short,Long}
  Completion/detail/Full_Table1_E9_Final_PerSession.csv, n=46 per arm).
- LERP/Spline per-session rows:
  results/diagnostics/20261119_q7_interp/q7_interp_per_session.json
  (reproduce with: touch_gait python
   results/diagnostics/20261119_q7_interp/q7_interp_eval.py).
- Entry/Ramp/TTA per-action 2D mean±std: the v8_s10 per-action 2D tables
  (results_display/Etest_crossfit_v8_s10/E9_Trajectory/{Short,Long}
  Completion/detail/Main_Table2_E9_Final_OccAte2D_MeanStd.csv); shared
  columns are bit-identical to the current main-tree tables, which dropped
  the Ramp column.
- LERP/Spline 2D: computed from q7_interp_per_session.json.

Pooled = action-n-weighted mean over the per-action means (same convention
as e9_summary_3d.py).

Outputs (Q7 实验结果/):
  Q7_动作级对照_3D.csv            main: rows=action+pooled, cols=arms, both tasks
  detail/Q7_动作级对照_3D_MeanStd.csv    mean±std (n)
  detail/Q7_动作级对照_2D_MeanStd.csv    mean±std (n)
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUB = REPO / "results_display" / "Etest_crossfit_physimu_vgate" / "E9_Trajectory"
S10 = REPO / "results_display" / "Etest_crossfit_v8_s10" / "E9_Trajectory"
INTERP = REPO / "results" / "diagnostics" / "20261119_q7_interp"
OUT = (REPO / "results_display" / "Etest_crossfit_physimu_vgate" / "Supplement"
       / "03_视觉缺失下的轨迹补全" / "Q7：轨迹适配的插值与端点对齐对照"
       / "实验结果")
ARMS = ["BiFoot-Entry", "BiFoot-Ramp", "BiFoot-LERP", "BiFoot-Spline",
        "BiFoot-TTA"]
ACTIONS = ["03", "04", "05", "09"]
TASKS = (("ShortCompletion", "final", "SceneOcc短时"),
         ("LongCompletion", "halfplane", "HalfPlane长时"))


def per_action_rows():
    """action -> {arm -> (mean, std, n)} for 3D, per task."""
    interp = json.load(open(INTERP / "q7_interp_per_session.json"))
    out = {}
    for sub, proto, _ in TASKS:
        rows = {}
        for r in csv.DictReader(
                (PUB / sub / "detail" / "Full_Table1_E9_Final_PerSession.csv")
                .open(encoding="utf-8-sig")):
            if r["arm"] not in ("BiFoot", "BiFoot-Ramp", "BiFoot-TTA"):
                continue
            arm = {"BiFoot": "BiFoot-Entry"}.get(r["arm"], r["arm"])
            rows.setdefault((r["action"], arm), []).append(
                float(r["occ_ate_3d_rms"]))
        for r in interp:
            if r["proto"] != proto:
                continue
            rows.setdefault((r["action"], r["arm"]), []).append(
                r["occ_ate_3d_rms"])
        out[proto] = {k: (sum(v) / len(v), std(v), len(v))
                      for k, v in rows.items()}
    return out


def std(v):
    import statistics
    return statistics.pstdev(v)


def action_meanstd_2d():
    """action -> {arm -> (mean, std, n)} for 2D, per task, from the v8_s10
    2D per-action tables (Entry/Ramp/TTA) + interp JSON (LERP/Spline)."""
    interp = json.load(open(INTERP / "q7_interp_per_session.json"))
    out = {}
    for sub, proto, _ in TASKS:
        t2 = list(csv.DictReader(
            (S10 / sub / "detail" / "Main_Table2_E9_Final_OccAte2D_MeanStd.csv")
            .open(encoding="utf-8-sig")))
        d = {}
        for r in t2:
            for arm, col in [("BiFoot-Entry", "BiFoot"),
                             ("BiFoot-Ramp", "BiFoot-Ramp"),
                             ("BiFoot-TTA", "BiFoot-TTA")]:
                d[(r["action"], arm)] = (float(r[col + "_mean"]),
                                         float(r[col + "_std"]),
                                         int(r[col + "_n"]))
        for r in interp:
            if r["proto"] != proto:
                continue
            v = [x["occ_ate_2d_rms"] for x in interp
                 if x["proto"] == proto and x["action"] == r["action"]
                 and x["arm"] == r["arm"]]
            d[(r["action"], r["arm"])] = (sum(v) / len(v), std(v), len(v))
        out[proto] = d
    return out


def pooled(vals_ns):
    return sum(v * n for v, _, n in vals_ns) / sum(n for _, _, n in vals_ns)


def main():
    d3 = per_action_rows()
    d2 = action_meanstd_2d()
    os.makedirs(OUT / "detail", exist_ok=True)
    # main table: rows = task × (action | pooled), cols = arms, 3D mean
    with (OUT / "Q7_动作级对照_3D.csv").open("w", newline="",
                                             encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["任务", "动作"] + ARMS)
        for _, proto, name in TASKS:
            w.writerow([name, "", "", "", "", "", ""])
            for act in ACTIONS:
                w.writerow(["", act] + ["%.1f" % d3[proto][(act, a)][0]
                                        if (act, a) in d3[proto] else ""
                                        for a in ARMS])
            w.writerow(["", "pooled(n=46)"] + [
                "%.1f" % pooled([d3[proto][(act, a)] for act in ACTIONS])
                for a in ARMS])
    # detail: mean±std (n) for 3D and 2D
    for name, d in [("Q7_动作级对照_3D_MeanStd.csv", d3),
                    ("Q7_动作级对照_2D_MeanStd.csv", d2)]:
        with (OUT / "detail" / name).open("w", newline="",
                                          encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["任务", "动作"] + ARMS)
            for _, proto, tname in TASKS:
                w.writerow([tname] + [""] * len(ARMS))
                for act in ACTIONS:
                    row = ["", act]
                    for a in ARMS:
                        if (act, a) in d[proto]:
                            m, s, n = d[proto][(act, a)]
                            row.append("%.1f±%.1f (n=%d)" % (m, s, n))
                        else:
                            row.append("")
                    w.writerow(row)
    print("wrote", OUT / "Q7_动作级对照_3D.csv")
    print("wrote", OUT / "detail")


if __name__ == "__main__":
    main()
