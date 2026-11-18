#!/usr/bin/env python3
"""E6 categorical part map: ordering of E_V, E_T, E_VT.

Categories:
  VT smallest (fusion best) -> white
  T > VT > V               -> blue
  V > VT > T               -> orange
  VT largest (fusion worst)-> black
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E6_ContactPhaseAnalysis"
SRC = OUT / "detail/contribution/Main_Table1_E6_FT_Contribution_byPart.csv"
PARTS = ("root", "torso", "headneck", "l_arm", "r_arm",
         "l_leg", "r_leg", "l_foot", "r_foot")
CONDS = (("clean", "Clean"), ("occ020", "Occ20"),
         ("occ050", "Occ50"), ("occ100", "Occ100"))
CATS = ("VT smallest", "T > VT > V", "V > VT > T", "VT largest")
COLORS = ("#ffffff", "#2a78d6", "#eb6834", "#0b0b0b")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

def val(s):
    try:
        return float(s.split("±")[0])
    except Exception:
        return np.nan

def classify(ev, et, evt):
    if evt < ev and evt < et:
        return 0
    if evt > ev and evt > et:
        return 3
    if et > evt > ev:
        return 1
    if ev > evt > et:
        return 2
    return None

def main():
    rows = list(csv.DictReader(open(SRC)))
    lookup = {(r["part"], r["condition"]): r for r in rows}
    mat = np.full((len(PARTS), len(CONDS)), np.nan)
    vals = {}
    for i, part in enumerate(PARTS):
        for j, (cond, _) in enumerate(CONDS):
            r = lookup.get((part, cond))
            if not r:
                continue
            ev, et, evt = val(r["E_V"]), val(r["E_T"]), val(r["E_VT"])
            vals[(i, j)] = (ev, et, evt)
            c = classify(ev, et, evt)
            if c is not None:
                mat[i, j] = c
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(figsize=(7.8, 6.6), facecolor=SURFACE)
    ax.set_facecolor("#e8e6e1")
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([l for _, l in CONDS], fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(PARTS)))
    ax.set_yticklabels(PARTS, fontsize=9.5, color=MUTED)
    ax.set_xticks(np.arange(-0.5, len(CONDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PARTS), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for (i, j), (ev, et, evt) in vals.items():
        txt = "%.1f / %.1f / %.1f" % (ev, et, evt)
        c = mat[i, j]
        color = "#fcfcfb" if c == 3 else INK
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=color)
    handles = [Patch(facecolor=COLORS[k], edgecolor="#8a8980" if k == 0 else "none",
                     label=CATS[k]) for k in range(4)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=False, fontsize=8.2)
    ax.set_title("Part-wise error ordering: V / T / VT", fontsize=11, color=INK)
    counts = {CATS[k]: int(np.sum(mat == k)) for k in range(4)}
    fig.text(0.5, 0.015,
             "Observed: %d VT-smallest, %d T>VT>V, %d V>VT>T, %d VT-largest"
             % (counts[CATS[0]], counts[CATS[1]], counts[CATS[2]], counts[CATS[3]]),
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    p = OUT / "Main_Fig_E6_PartOrder_Categories.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    csv_path = OUT / "detail" / "part_order_categories.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "condition", "E_V", "E_T", "E_VT", "category"])
        for (i, j), (ev, et, evt) in vals.items():
            c = int(mat[i, j]) if np.isfinite(mat[i, j]) else -1
            w.writerow([PARTS[i], CONDS[j][0], ev, et, evt,
                        CATS[c] if c >= 0 else "tie/NA"])
    print("wrote", csv_path)
    counts = {CATS[k]: int(np.sum(mat == k)) for k in range(4)}
    print("counts", counts)

if __name__ == "__main__":
    main()
