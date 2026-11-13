#!/usr/bin/env python3
"""E6 contact-phase export for the IMU no-gate group (V8_Cv7_PhysIMU).

G = VT error − V error (mm; negative = tactile helps), per f6_soft phase:
0 Stance / 1 Swing / 2 Strike / 3 Off.  Four-fold LSO, left/right averaged,
fold mean ± std.  Outputs to results_display/Etest_crossfit_v8_branches/IMU.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results_display/Etest_crossfit_v8/E6_ContactPhaseAnalysis/imu_nogate/pa_mpjpe"
OUT = REPO / "results_display/Etest_crossfit_v8_branches/IMU/E6_ContactPhaseAnalysis"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
ACTIONS = ("walk", "run", "squat", "stand_shift", "single_stand",
           "jump_jack", "lateral", "sit_stand", "stairs")
# actions with all four phases: used for the balanced all-action summary
BALANCED = ("walk", "run", "jump_jack", "lateral", "single_stand", "stairs")
PHASES = (("0", "Stance"), ("1", "Swing"), ("2", "Strike"), ("3", "Off"))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = "#2a78d6"
GRID = "#e6e4df"


def load_fold(fold: str):
    p = SRC / ("fold_" + fold) / "detail" / "e6_table.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    # the table has a single cell-model key
    out = {}
    for action, block in payload.items():
        table = block.get("table") or {}
        if not table:
            continue
        model = next(iter(table))
        feet = table[model]
        for phase_key, phase_name in PHASES:
            vals = []
            for foot in ("left", "right"):
                e = (feet.get(foot) or {}).get(phase_key)
                if e and e.get("G") is not None:
                    vals.append((float(e["G"]), float(e["V"]), float(e["VT"])))
            if vals:
                out[(action, phase_name)] = tuple(
                    np.mean([v[i] for v in vals]) for i in range(3))
    return out


def main():
    per_fold = {}
    for fold in FOLDS:
        data = load_fold(fold)
        if data:
            per_fold[fold] = data
    if not per_fold:
        raise SystemExit("no fold e6_table.json")
    folds = list(per_fold)
    OUT.mkdir(parents=True, exist_ok=True)
    # copy raw fold tables into the branches tree
    for fold in folds:
        dst = OUT / "pa_mpjpe" / ("fold_" + fold) / "detail"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / ("fold_" + fold) / "detail" / "e6_table.json",
                     dst / "e6_table.json")

    # full table (per fold) and main table (mean±std across folds)
    full_rows, main_rows = [], []
    agg = {}
    for action in ACTIONS:
        for phase_key, phase_name in PHASES:
            vals = [per_fold[f][(action, phase_name)] for f in folds
                    if (action, phase_name) in per_fold[f]]
            if not vals:
                continue
            g = np.array([v[0] for v in vals])
            v = np.array([v[1] for v in vals])
            vt = np.array([v[2] for v in vals])
            agg[(action, phase_name)] = (g.mean(), g.std(ddof=0), v.mean(),
                                         vt.mean(), len(vals))
            for f, val in zip(folds, vals):
                full_rows.append([f, action, phase_name,
                                  "%.3f" % val[0], "%.3f" % val[1], "%.3f" % val[2]])
            main_rows.append([action, phase_name,
                              "%.3f±%.3f" % (g.mean(), g.std(ddof=0)),
                              "%.3f" % v.mean(), "%.3f" % vt.mean(), len(vals)])
    with (OUT / "Full_Table1_E6_IMU_NoGate_PhaseGain.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "action", "phase", "G_mm", "V_mm", "VT_mm"])
        w.writerows(full_rows)
    with (OUT / "Main_Table1_E6_IMU_NoGate_PhaseGain.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["action", "phase", "G_mean±std(mm)", "V_mean(mm)",
                    "VT_mean(mm)", "n_folds"])
        w.writerows(main_rows)

    # 3x3 action panels
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 9.9), facecolor=SURFACE)
    for ax, action in zip(axes.flat, ACTIONS):
        xs2, ys2, es2, xs1, ys1 = [], [], [], [], []
        for i, (_, phase_name) in enumerate(PHASES):
            if (action, phase_name) not in agg:
                continue
            mean, std, _, _, n = agg[(action, phase_name)]
            if n >= 2:
                xs2.append(i)
                ys2.append(mean)
                es2.append(std)
            else:
                xs1.append(i)
                ys1.append(mean)
        if xs2:
            ax.errorbar(xs2, ys2, yerr=es2, fmt="o-", color=SERIES, lw=2.0,
                        ms=5, capsize=3, ecolor=MUTED, elinewidth=1.0)
        if xs1:
            ax.plot(xs1, ys1, "o", mfc=SURFACE, mec=SERIES, ms=6, mew=1.5)
        ax.axhline(0.0, color="#9a9890", lw=0.8, ls=":", zorder=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([p[1] for p in PHASES], fontsize=8.5)
        ax.set_title(action, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c6bf")
    for ax in axes[:, 0]:
        ax.set_ylabel("G = VT − V (mm)", fontsize=9, color=INK)
    fig.suptitle("IMU no-gate (V8_Cv7_PhysIMU): tactile gain by contact phase\n"
                 "four-fold LSO; negative = T helps; open marker = one fold only",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "Main_Fig1_E6_IMU_NoGate_PhaseGain.png", dpi=150,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # all-actions phase summary: mean over actions per fold, then fold mean±std
    xs, ys, es = [], [], []
    for i, (_, phase_name) in enumerate(PHASES):
        per_fold_g = []
        for f in folds:
            vals = [per_fold[f][(a, phase_name)][0] for a in BALANCED
                    if (a, phase_name) in per_fold[f]]
            if vals:
                per_fold_g.append(float(np.mean(vals)))
        if not per_fold_g:
            continue
        xs.append(i)
        ys.append(float(np.mean(per_fold_g)))
        es.append(float(np.std(per_fold_g, ddof=0)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    bars = ax.bar(xs, ys, yerr=es, color=SERIES, width=0.55, capsize=4,
                  error_kw={"ecolor": MUTED, "elinewidth": 1.0})
    ax.axhline(0.0, color="#9a9890", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[1] for p in PHASES], fontsize=9)
    ax.set_ylabel("G = VT − V (mm)", fontsize=9, color=INK)
    ax.set_title("IMU no-gate: tactile gain by phase\n"
                 "balanced: walk/run/jump_jack/lateral/single_stand/stairs",
                 fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c6bf")
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2,
                y + (0.6 if y >= 0 else -0.9), "%+.1f" % y,
                ha="center", va="bottom" if y >= 0 else "top",
                fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "Main_Fig2_E6_IMU_NoGate_PhaseGain_AllActions.png",
                dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
