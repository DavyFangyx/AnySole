#!/usr/bin/env python3
"""E6 part-wise absolute error: 4 occlusion blocks x (V,T,VT) x 9 parts.

Unified one-directional color scale (dark = low error, light yellow = high),
all methods and conditions share the same scale.  Uses the existing
E_V/E_T/E_VT table; no new experiment.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E6_ContactPhaseAnalysis"
SRC = OUT / "detail/contribution/Main_Table1_E6_FT_Contribution_byPart.csv"
PARTS = ("root", "torso", "headneck", "l_arm", "r_arm",
         "l_leg", "r_leg", "l_foot", "r_foot")
CONDS = (("clean", "Clean"), ("occ020", "Occ20"),
         ("occ050", "Occ50"), ("occ100", "Occ100"))
MODES = (("E_V", "V"), ("E_T", "T"), ("E_VT", "VT"))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

def val(s):
    try:
        return float(s.split("±")[0])
    except Exception:
        return np.nan

def main():
    rows = list(csv.DictReader(open(SRC)))
    lookup = {(r["part"], r["condition"]): r for r in rows}
    mat = np.full((len(PARTS), len(CONDS) * len(MODES)), np.nan)
    for i, part in enumerate(PARTS):
        for j, (cond, _) in enumerate(CONDS):
            r = lookup.get((part, cond))
            if not r:
                continue
            for k, (key, _) in enumerate(MODES):
                mat[i, j * 3 + k] = val(r[key])
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor=SURFACE)
    im = ax.imshow(mat, cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(PARTS)))
    ax.set_yticklabels(PARTS, fontsize=9, color=MUTED)
    ax.set_xticks([])
    for j, (cond, label) in enumerate(CONDS):
        x0 = j * 3
        ax.text(x0 + 1, -0.85, label, ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
        for k, (_, mlabel) in enumerate(MODES):
            ax.text(x0 + k, -0.25, mlabel, ha="center", va="bottom",
                    fontsize=8.5, color=MUTED)
        if j:
            ax.axvline(x0 - 0.5, color=SURFACE, lw=3.0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=6.8,
                    color=("#0b0b0b" if v < vmin + 0.55 * (vmax - vmin) else "#fcfcfb"))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("PA-MPJPE (mm; dark = low, light = high)", fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("Part-wise absolute error: V / T / VT under four occlusion levels",
                 fontsize=11, color=INK, pad=28)
    fig.tight_layout()
    p = OUT / "Main_Fig_E6_PartAbsoluteError_Blocks.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

if __name__ == "__main__":
    main()
