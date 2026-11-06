#!/usr/bin/env python3
"""E6 四折 LSO 门值随遮挡汇总（Etest_crossfit，2026-11-05）。

输入 = results_display/Etest_crossfit/E6_ContactPhaseAnalysis/gate_values/
        <family>/<fold>/gate_values_occlusion_summary.csv
四折按 family × occ × group 取 mean±std；同时给出 g>0.5 占比均值。

产物：
  Full_Table1_E6_Crossfit_GateValues_MeanStd.csv
  Main_Fig2_E6_Crossfit_GateValues.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = (REPO / "results_display" / "Etest_crossfit" /
        "E6_ContactPhaseAnalysis" / "gate_values")
OUT = (REPO / "results_display" / "Etest_crossfit" /
       "E6_ContactPhaseAnalysis" / "gate_values")
FAMS = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")
X = [int(x) for x in LEVELS]
GROUPS = ("vel", "other", "left", "right")
COLORS = {"V7_Cv5": "#2a78d6", "V7_Cv5SD2": "#eb6834",
          "V7_Cv5SD2RT": "#1baf7a", "V7_Cv5_tphys": "#eda100"}
LABELS = {"V7_Cv5": "Cv5", "V7_Cv5SD2": "Cv5SD2",
          "V7_Cv5SD2RT": "Cv5SD2RT", "V7_Cv5_tphys": "Cv5_tphys"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return round(vals[0], 3)
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def collect() -> dict:
    """→ data[(fam, occ)][group] = list of per-fold row dicts."""
    out: dict = {}
    missing = []
    for fam in FAMS:
        for level in LEVELS:
            for fold in FOLDS:
                p = BASE / fam / fold / "gate_values_occlusion_summary.csv"
                if not p.is_file():
                    missing.append((fam, fold))
                    continue
                for r in read(p):
                    if r["occ"] != level:
                        continue
                    out.setdefault((fam, level), {}).setdefault(
                        r["group"], []).append(r)
    if missing:
        print("[missing] %d/%d family-fold runs" % (
            len(missing), len(FAMS) * len(FOLDS)))
    return out


def export_table(data: dict) -> None:
    rows = []
    for fam in FAMS:
        for level in LEVELS:
            for group in GROUPS:
                rs = data.get((fam, level), {}).get(group, [])
                if not rs:
                    continue
                rows.append([
                    fam, "occ" + level, group,
                    mean_std([r["mean"] for r in rs]),
                    mean_std([r["open_frac"] for r in rs]),
                    len(rs),
                ])
    p = OUT / "Full_Table1_E6_Crossfit_GateValues_MeanStd.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "档位", "门组", "gate_mean±std",
                    "g>0.5占比_mean±std", "n_folds"])
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (p, len(rows)))


def figure(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), facecolor=SURFACE)
    for ax, group, title in ((axes[0], "vel", "Root gate"),
                             (axes[1], "other", "Pose gate")):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        for fam in FAMS:
            ys = []
            for level in LEVELS:
                rs = data.get((fam, level), {}).get(group, [])
                ys.append(float(np.mean([float(r["mean"]) for r in rs]))
                          if rs else np.nan)
            ax.plot(X, ys, lw=2.0, marker="o", ms=4.5, color=COLORS[fam],
                    label=LABELS[fam])
        ax.axhline(0.5, color="#c8c8c8", lw=0.9, ls="--", zorder=1)
        ax.set_ylim(0, 1)
        ax.set_title(title + " vs occlusion", fontsize=10.5, color=INK,
                     loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=9.5, color=MUTED)
        ax.set_ylabel("Mean gate g (sigmoid)", fontsize=9.5, color=MUTED)
        ax.set_xticks(X)
    axes[0].legend(frameon=False, fontsize=8.2, loc="best")
    fig.suptitle("E6 crossfit: tactile gate values under occlusion, VT input, "
                 "4-fold mean (dashed = 0.5)", fontsize=11.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "Main_Fig2_E6_Crossfit_GateValues.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % p)


def main() -> int:
    global BASE, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    BASE, OUT = args.base, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    if not data:
        print("[skip] no gate value runs")
        return 0
    export_table(data)
    figure(data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
