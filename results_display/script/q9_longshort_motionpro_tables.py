#!/usr/bin/env python3
"""Q9 supplement: MotionPRO long-vs-short task performance.

No new experiment. Assembles the Q9 artifacts from the existing E9 closeout
tables and the occlusion manifests, and verifies the pooled values against
the published E9 summary CSVs.

Sources
-------
- Per-action occ/full 3D ATE mean±std tables (four-fold LSO woS13, 46
  fold-internal records, actions 03/04/05/09, n = 17/4/6/19):
    results_display/Etest_crossfit_physimu_vgate/E9_Trajectory/
      {Long,Short}Completion/Main_Table1_E9_Final_OccAte3D_MeanStd.csv
      {Long,Short}Completion/Main_Table3_E9_Final_FullAte3D_RMS_MeanStd.csv
- Published pooled-3D summaries (reference for the check):
    results_display/Etest_crossfit_physimu_vgate/E9_Trajectory/
      Main_Table0a_E9_Summary_SceneOcc.csv   (short task)
      Main_Table0b_E9_Summary_HalfPlane.csv  (long task)
- Occlusion interval stats (22 unique sessions shared by both tasks):
    results/diagnostics/20261111_halfplane_occ/manifests/*.json
    results/diagnostics/20261107_e5_sceneocc_full/manifests/*.json

Pooling = action-n-weighted mean of the per-action means (same convention as
e9_summary_3d.py).

Outputs (under results_display/Etest_crossfit_physimu_vgate/Supplement/
03_视觉缺失下的轨迹补全/Q9：MotionPRO 长-短任务表现差异/实验结果)
-------------------------------------------------------------------------------
  Q9_长短任务对照.csv                 pooled short/long/Δ, 5 methods (3D occ ATE)
  Q9_长短任务逐动作.csv               per-action short/long/Δ, occ 3D ATE
  Q9_长短任务逐动作_全程ATE.csv       per-action short/long/Δ, full 3D ATE RMS
  Q9_区间时长统计.csv                 event counts/durations, 22-session pool
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "results_display/Etest_crossfit_physimu_vgate/E9_Trajectory"
OUT = (REPO / "results_display/Etest_crossfit_physimu_vgate/Supplement/"
       "03_视觉缺失下的轨迹补全/Q9：MotionPRO 长-短任务表现差异/实验结果")
HALF_MANI = REPO / "results/diagnostics/20261111_halfplane_occ/manifests"
SCENE_MANI = REPO / "results/diagnostics/20261107_e5_sceneocc_full/manifests"

ARMS = ["MotionPRO", "Step2Motion", "BiFoot", "BiFoot-TTA", "BiFoot-Ramp"]
ACTIONS = ["03", "04", "05", "09"]
OUT.mkdir(parents=True, exist_ok=True)


def read_mean_std(path: str) -> dict:
    """action -> {arm: mean}; also n per action."""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out, ns = {}, {}
    for r in rows:
        a = r["action"]
        ns[a] = int(r["MotionPRO_n"])
        out[a] = {arm: float(r[f"{arm}_mean"]) for arm in ARMS if f"{arm}_mean" in r}
    return out, ns


def pooled(per_action: dict, ns: dict, arm: str) -> float:
    tot, ntot = 0.0, 0
    for a in ACTIONS:
        tot += per_action[a][arm] * ns[a]
        ntot += ns[a]
    return tot / ntot


def read_summary(path: str) -> dict:
    """Read published pooled summary: row1 = title, row2 = 方法,03,04,05,09,pooled 3D."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln for ln in csv.reader(f)]
    header, data = lines[1], lines[2:]
    assert header == ["方法", "03", "04", "05", "09", "pooled 3D"], header
    return {r[0]: float(r[-1]) for r in data}


def main() -> None:
    occ_long, ns = read_mean_std(str(E9 / "LongCompletion/Main_Table1_E9_Final_OccAte3D_MeanStd.csv"))
    occ_short, _ = read_mean_std(str(E9 / "ShortCompletion/Main_Table1_E9_Final_OccAte3D_MeanStd.csv"))
    full_long, _ = read_mean_std(str(E9 / "LongCompletion/Main_Table3_E9_Final_FullAte3D_RMS_MeanStd.csv"))
    full_short, _ = read_mean_std(str(E9 / "ShortCompletion/Main_Table3_E9_Final_FullAte3D_RMS_MeanStd.csv"))

    summ_short = read_summary(str(E9 / "Main_Table0a_E9_Summary_SceneOcc.csv"))
    summ_long = read_summary(str(E9 / "Main_Table0b_E9_Summary_HalfPlane.csv"))

    # ---- verify pooled against published summaries (arms present per-action;
    #      Ramp exists only in the summaries, keep it as-is) ----
    per_arms = [a for a in ARMS if a in occ_long["03"]]
    errs = []
    for arm in per_arms:
        for task, per, summ in (("short", occ_short, summ_short),
                                ("long", occ_long, summ_long)):
            v = pooled(per, ns, arm)
            if abs(v - summ[arm]) > 0.05:
                errs.append(f"{task} {arm}: computed {v:.3f} != summary {summ[arm]:.3f}")
    if errs:
        raise SystemExit("pooled mismatch:\n" + "\n".join(errs))
    print("pooled check OK: 4 per-action arms match Main_Table0a/0b on both tasks")

    # ---- pooled comparison CSV (5 arms) ----
    with open(OUT / "Q9_长短任务对照.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方法", "短时_SceneOcc_pooled_3D_ATE_mm", "长时_HalfPlane_pooled_3D_ATE_mm",
                    "Δ长减短_mm", "Δ相对短时_%"])
        for arm in ARMS:
            s, l = summ_short[arm], summ_long[arm]
            w.writerow([arm, f"{s:.1f}", f"{l:.1f}", f"{l - s:+.1f}", f"{(l - s) / s * 100:+.1f}"])

    # ---- per-action CSVs (occ 3D; full 3D RMS) ----
    for name, long_t, short_t in (("Q9_长短任务逐动作", occ_long, occ_short),
                                  ("Q9_长短任务逐动作_全程ATE", full_long, full_short)):
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            header = ["动作", "n"]
            for arm in per_arms:
                header += [f"{arm}_短时_mm", f"{arm}_长时_mm", f"{arm}_Δ长减短_mm"]
            w.writerow(header)
            for a in ACTIONS:
                row = [a, ns[a]]
                for arm in per_arms:
                    s, l = short_t[a][arm], long_t[a][arm]
                    row += [f"{s:.1f}", f"{l:.1f}", f"{l - s:+.1f}"]
                w.writerow(row)
            row = ["pooled", sum(ns[a] for a in ACTIONS)]
            for arm in per_arms:
                s = pooled(short_t, ns, arm)
                l = pooled(long_t, ns, arm)
                row += [f"{s:.1f}", f"{l:.1f}", f"{l - s:+.1f}"]
            w.writerow(row)

    # ---- interval stats over the 22-session pool ----
    sessions = set()
    for t in ("LongCompletion", "ShortCompletion"):
        p = E9 / t / "detail/Full_Table1_E9_Final_PerSession.csv"
        with open(p, encoding="utf-8-sig") as f:
            sessions |= {r["session"] for r in csv.DictReader(f)}
    with open(OUT / "Q9_区间时长统计.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["任务", "会话数", "事件数", "均值帧", "均值秒", "最小帧", "最大帧",
                    "manifest 来源"])
        for task, mani in (("短时_SceneOcc", SCENE_MANI), ("长时_HalfPlane", HALF_MANI)):
            lens = []
            for s in sorted(sessions):
                j = json.load(open(mani / f"{s}.json"))
                for e in j.get("full_events") or []:
                    lens.append(e["end"] - e["start"])
            w.writerow([task, len(sessions), len(lens),
                        f"{sum(lens) / len(lens):.1f}",
                        f"{sum(lens) / len(lens) / 40:.2f}",
                        min(lens), max(lens), mani.relative_to(REPO)])
    print("written:", OUT)


if __name__ == "__main__":
    main()
