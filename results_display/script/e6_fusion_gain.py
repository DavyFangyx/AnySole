#!/usr/bin/env python3
"""E6 fusion gain over the best single modality.

G(p,c) = min(E_V(p,c), E_T(p,c)) - E_VT(p,c)   (positive = fusion helps)
Whole-body curves use the formal E5 tables; part map uses the E6 contribution
table.  The signed ΔT (VT-V) is kept in detail; this figure avoids reading
"T harmful" from a red cell.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate"
E5 = OUT / "E5_OcclusionRobustness/Main_Table2_E5_Occlusion_MainVsBaselines.csv"
PART = OUT / "E6_ContactPhaseAnalysis/detail/contribution/Main_Table1_E6_FT_Contribution_byPart.csv"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
LEVELS = (("occ000", "clean", 0), ("occ020", "occ020", 20),
          ("occ050", "occ050", 50), ("occ090", "occ090", 90),
          ("occ100", "occ100", 100))
PARTS = ("root", "torso", "headneck", "l_arm", "r_arm",
         "l_leg", "r_leg", "l_foot", "r_foot")
CONDS = (("clean", "Clean"), ("occ020", "Occ20"), ("occ050", "Occ50"),
         ("occ100", "Occ100"))

def parse(s):
    if not s or s == "—":
        return None
    try:
        return float(s.split("±")[0])
    except ValueError:
        return None

def whole_body():
    rows = list(csv.DictReader(open(E5, encoding="utf-8-sig")))
    out = []
    for level, label, x in LEVELS:
        def val(model):
            r = [q for q in rows if q["档位"] == level and q["模型"] == model]
            return parse(r[0]["PA-MPJPE"]) if r else None
        out.append((x, label, val("V8_IMU_Residual @V"),
                    val("V8_IMU_Residual @T"), val("V8_IMU_Residual @VT")))
    return out

def part_matrix():
    rows = list(csv.DictReader(open(PART)))
    mat = np.full((len(PARTS), len(CONDS)), np.nan)
    for r in rows:
        if r["part"] not in PARTS:
            continue
        cond = r["condition"]
        if cond not in [c for c, _ in CONDS]:
            continue
        i = PARTS.index(r["part"]); j = [c for c, _ in CONDS].index(cond)
        ev, et, evt = parse(r["E_V"]), parse(r["E_T"]), parse(r["E_VT"])
        mat[i, j] = min(ev, et) - evt
    return mat

def main():
    wb = whole_body()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), facecolor=SURFACE)
    ax = axes[0]
    xs = [r[0] for r in wb]
    for idx, (label, color, lw) in enumerate((("V", "#86b6ef", 2.0),
                                               ("T", "#4a3aa7", 2.0),
                                               ("VT", "#2a78d6", 2.8))):
        ys = [r[2 + idx] for r in wb]
        ax.plot(xs, ys, color=color, lw=lw, marker="o", ms=4.5, label=label)
    ax.set_xlabel("Occlusion (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("PA-MPJPE (mm)", fontsize=9, color=MUTED)
    ax.set_title("Whole-body error: V / T / VT", fontsize=10.5, color=INK)
    ax.grid(color="#e6e3de", lw=0.7); ax.set_facecolor(SURFACE)
    ax.legend(frameon=False, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=8)

    ax = axes[1]
    gains = [min(r[2], r[3]) - r[4] for r in wb]
    ax.bar([r[0] for r in wb], gains, width=7.5, color="#2a78d6", alpha=0.9)
    for x, g in zip([r[0] for r in wb], gains):
        ax.text(x, g + 0.25, "%.2f" % g, ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.axhline(0, color="#9a9890", lw=0.8)
    ax.set_xlabel("Occlusion (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("G = min(V,T) − VT (mm)", fontsize=9, color=MUTED)
    ax.set_title("Fusion gain over the best single modality", fontsize=10.5, color=INK)
    ax.grid(axis="y", color="#e6e3de", lw=0.7); ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    p = OUT / "E6_ContactPhaseAnalysis/Main_Fig_E6_FusionGain_Curves.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

    mat = part_matrix()
    cmap = LinearSegmentedColormap.from_list(
        "gain", ["#fcfcfb", "#cfe3f3", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.8, 5.6), facecolor=SURFACE)
    vmax = np.nanmax(mat)
    im = ax.imshow(np.clip(mat, 0, vmax), cmap=cmap, vmin=0, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([l for _, l in CONDS], fontsize=9, color=MUTED)
    ax.set_yticks(range(len(PARTS)))
    ax.set_yticklabels(PARTS, fontsize=9, color=MUTED)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center",
                    fontsize=7.6, color=("#fcfcfb" if v > 0.55 * vmax else INK))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("G = min(E_V,E_T) − E_VT (mm)", fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("Part-wise fusion gain (PA-MPJPE; grey = no gain)",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    p = OUT / "E6_ContactPhaseAnalysis/Main_Fig_E6_FusionGain_Parts.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

if __name__ == "__main__":
    main()
