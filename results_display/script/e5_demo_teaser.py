#!/usr/bin/env python3
"""E5 demo teaser: WHAM (vision-only baseline) vs Ours VT2M, with GT ghost.

2 x 3 grid: top row = WHAM, bottom row = Ours VT2M, columns = time snapshots.
Each panel shows the GT ghost and one prediction on the floor; WHAM failures
are marked in red (body, below-floor ring/bar, GT-mismatch dashed ring).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Baselines"))

from Baselines.utils.gt_loading import protocol_gt, protocol_gt_surface  # noqa: E402
from Baselines.utils.motion_io import load_surface  # noqa: E402

MOTION = REPO / "results/diagnostics/20261116_e5_teaser_motion/V8_IMU_Residual"
BASE = REPO / "results/baselines"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/demo"
SCENE_DIR = {"occ020": "occ020", "occ090": "occ090",
             "gap060": "dropout", "gap080": "dropout"}
SMPL_PATH = str(REPO / "AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl")

GT_GRAY = "#b0b0b0"
WHAM_RED = "#d62728"
VT_BLUE = "#2a78d6"
FLOOR = "#ececea"
GRID = "#c9c9c5"
INK = "#222222"
MUTED = "#555555"

COND = {
    "occ020": "e5_occ_lower020",
    "occ090": "e5_occ_lower090",
    "gap060": "e5_gap_r060",
    "gap080": "e5_gap_r080",
}
TITLE = {
    "occ020": "20% lower-body occlusion",
    "occ090": "90% lower-body occlusion",
    "gap060": "video dropout 60%",
    "gap080": "video dropout 80%",
}
ACTIONS = {
    "01": "stand-shift", "02": "single-stand", "03": "shuttle-walk",
    "04": "shuttle-walk-fast", "05": "run", "06": "squat",
    "07": "jump-jack", "08": "shuttle-run", "09": "lateral-walk",
    "10": "sit-stand", "11": "stairs",
}


def load_manifest(sid: str) -> dict:
    for p in (REPO / "AnysoleWorkspace/protocol/splits/crossfit_woS13").glob("*_manifest.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["session_id"] == sid:
                    return row
    raise KeyError(sid)


def faces_from_smpl():
    import smplx
    sm = smplx.create(SMPL_PATH, model_type="smpl", gender="neutral", batch_size=1)
    return np.asarray(sm.faces, dtype=np.int32)


def load_ours(cond: str, fold: str, sid: str, mode: str):
    p = MOTION / cond / fold / "eval_motion" / ("%s_%s.npz" % (sid, mode))
    z = np.load(p, allow_pickle=True)
    surf = load_surface(p)
    return {
        "v": np.asarray(surf["vertices"], np.float64),
        "j": np.asarray(z["joint_xyz_world"], np.float64),
        "valid": np.asarray(z["valid_mask"], bool),
        "fps": float(np.asarray(z["mocap_frame_rate"])) if "mocap_frame_rate" in z.files else 40.0,
    }


def load_wham(cond: str, sid: str):
    p = BASE / "WHAM/predictions" / COND[cond] / "eval_motion" / ("%s.npz" % sid)
    z = np.load(p, allow_pickle=True)
    return {
        "v": np.asarray(z["vertices_world"], np.float64),
        "j": np.asarray(z["joint_xyz_world"], np.float64),
        "valid": np.asarray(z["valid_mask"], bool),
        "fps": float(np.asarray(z["target_fps"])) if "target_fps" in z.files else 40.0,
    }


def pick_frames(wham_v, wham_j, gt_j, valid, fps, ncol=3, vt_err=None):
    idx = np.flatnonzero(valid)
    # root-anchored pose error (pelvis translation removed); trajectory ignored
    w_rel = wham_j - wham_j[:, 0:1]
    g_rel = gt_j - gt_j[:, 0:1]
    err = np.linalg.norm(w_rel - g_rel, axis=-1).mean(axis=1) * 1000.0
    pen = -wham_v[:, :, 2].min(axis=1)   # wham_v is already anchored at GT pelvis
    ok = valid & (pen < 0.55)
    if vt_err is not None:
        ok &= vt_err < 150.0
    if ok.sum() < max(3, ncol):
        ok = valid
    p = int(np.argmax(np.where(ok, pen, -1.0)))
    e = int(np.argmax(np.where(ok, err, -1.0)))
    min_sep = max(12, int(0.5 * fps))
    picks = []
    for cand in (p, e):
        if all(abs(cand - x) >= min_sep for x in picks):
            picks.append(cand)
    idx_ok = np.flatnonzero(ok)
    grid = idx_ok[np.round(np.linspace(0, len(idx_ok) - 1, 17)).astype(int)]
    while len(picks) < ncol:
        best, best_d = None, -1
        for cand in grid:
            d = min(abs(int(cand) - x) for x in picks)
            if d > best_d:
                best, best_d = int(cand), d
        if best is None or best_d < min_sep:
            break
        picks.append(best)
    return sorted(dict.fromkeys(picks)), pen, err


def gap_picks(sid, cond, n, valid, wv_a, wj, gt_j, fps):
    """Frames: two pre-gap, two in-gap, one post-recovery.

    The gap cache freezes the visual stream for rho% of every cycle; the
    artifact records the exact phase. Choose the gap interval with the largest
    WHAM pose error during the freeze.
    """
    rho = int(str(cond).replace("gap", ""))
    p = (REPO / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet"
         / ("cam3_gap_r%03d" % rho) / ("%s.pt.artifact.json" % sid))
    d = json.loads(p.read_text())
    par = d["parameters"]
    cycle = int(par["cycle"])
    gap_len = int(round(float(par["rho_percent"]) / 100.0 * cycle))
    off = int(par["phase_offset"])
    pos = (np.arange(n) + off) % cycle
    is_gap = pos >= (cycle - gap_len)
    is_gap[0] = False
    runs = []
    i = 0
    while i < n:
        if is_gap[i]:
            j = i
            while j < n and is_gap[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    w_rel = wj - wj[:, 0:1]
    g_rel = gt_j - gt_j[:, 0:1]
    err = np.linalg.norm(w_rel - g_rel, axis=-1).mean(axis=1) * 1000.0
    cand = np.flatnonzero(valid)
    def near(t):
        t = int(np.clip(t, 0, n - 1))
        return int(cand[np.argmin(np.abs(cand - t))])

    def pick_low(lo, hi, k, sep):
        """k valid frames with the lowest pose error in [lo, hi)."""
        xs = [int(i) for i in cand if lo <= i < hi]
        xs.sort(key=lambda i: err[i])
        out = []
        for i in xs:
            if all(abs(i - j) >= sep for j in out):
                out.append(i)
                if len(out) == k:
                    break
        return sorted(out)

    def pick_high(lo, hi, k, sep):
        xs = [int(i) for i in cand if lo <= i < hi]
        xs.sort(key=lambda i: -err[i])
        out = []
        for i in xs:
            if all(abs(i - j) >= sep for j in out):
                out.append(i)
                if len(out) == k:
                    break
        return sorted(out)

    best = None
    for s, e in runs:
        if e - s < 20:
            continue
        pre = pick_low(max(0, s - 40), s, 2, 5)
        dur = pick_high(s + 3, e - 3, 2, 10)
        post = pick_low(e, min(n, e + 30), 1, 1)
        if len(pre) < 2 or len(dur) < 2 or len(post) < 1:
            continue
        pre_e = float(np.mean([err[t] for t in pre]))
        dur_e = float(np.mean([err[t] for t in dur]))
        post_e = float(err[post[0]])
        contrast = dur_e - 0.5 * (pre_e + post_e)
        if best is None or contrast > best[0]:
            best = (contrast, pre + dur + post)
    return [] if best is None else best[1]


def draw_ring(ax, x, y, z, r=0.10, color=WHAM_RED, lw=2.0, ls="-", zorder=30):
    th = np.linspace(0, 2 * np.pi, 72)
    ax.plot(x + r * np.cos(th), y + r * np.sin(th), np.full_like(th, z),
            color=color, lw=lw, ls=ls, zorder=zorder)


def draw_panel(ax, gt_v, mv, faces, body_color, *, pen_ring=True, mismatch_ring=False):
    """One square panel: GT ghost + method body on a local floor."""
    zmax = max(gt_v[:, 2].max(), mv[:, 2].max())
    zmin = min(0.0, gt_v[:, 2].min(), mv[:, 2].min())
    xv = np.concatenate([gt_v[:, 0], mv[:, 0]])
    yv = np.concatenate([gt_v[:, 1], mv[:, 1]])
    x0, x1 = xv.min() - 0.28, xv.max() + 0.28
    y0, y1 = yv.min() - 0.40, yv.max() + 0.40
    z0, z1 = min(-0.03, zmin - 0.04), zmax + 0.10

    ax.set_facecolor("white")
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(
        [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]],
        facecolors=FLOOR, edgecolors="none", alpha=0.65, zorder=0))
    for x in np.linspace(x0, x1, 6):
        ax.plot([x, x], [y0, y1], [0, 0], color=GRID, lw=0.6, zorder=1)
    for y in np.linspace(y0, y1, 6):
        ax.plot([x0, x1], [y, y], [0, 0], color=GRID, lw=0.6, zorder=1)
    ax.plot_trisurf(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], triangles=faces,
                    color=GT_GRAY, shade=True, linewidth=0, alpha=0.40,
                    antialiased=False, zorder=6)
    ax.plot_trisurf(mv[:, 0], mv[:, 1], mv[:, 2], triangles=faces,
                    color=body_color, shade=True, linewidth=0, alpha=0.97,
                    antialiased=False, zorder=12)
    if pen_ring:
        ix = int(np.argmin(mv[:, 2]))
        mz = float(mv[ix, 2])
        if mz < -0.004:
            fx, fy = mv[ix, 0], mv[ix, 1]
            draw_ring(ax, fx, fy, 0.002, r=0.095, color=body_color, lw=2.4)
            ax.plot([fx, fx], [fy, fy], [0.0, mz], color=body_color, lw=2.4, zorder=29)
            ax.scatter([fx], [fy], [mz], s=24, color=body_color,
                       depthshade=False, zorder=30)
        elif mz > 0.05:
            fx, fy = mv[ix, 0], mv[ix, 1]
            draw_ring(ax, fx, fy, 0.002, r=0.095, color=body_color, lw=1.6,
                      ls=(0, (3, 3)))
            ax.plot([fx, fx], [fy, fy], [0.0, mz], color=body_color, lw=1.6,
                    ls=(0, (3, 3)), zorder=29)
            ax.scatter([fx], [fy], [mz], s=20, color=body_color,
                       depthshade=False, zorder=30)
    if mismatch_ring:
        rxy = mv[:, :2].mean(axis=0)
        draw_ring(ax, rxy[0], rxy[1], 0.0018, r=0.52, color=body_color,
                  lw=1.5, ls=(0, (4, 3)), zorder=28)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.35)
    ax.view_init(elev=13.0, azim=-62)
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(COND))
    ap.add_argument("--sid", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--no-text", action="store_true", help="render without any text captions/legend")
    ap.add_argument("--gap-phases", action="store_true", help="dropout group: 2 pre + 2 in-gap + 1 recovery")
    ap.add_argument("--tag", default="", help="optional filename prefix (sliding/pending)")
    ap.add_argument("--world-phys", action="store_true", help="draw world-frame bodies (for true ground penetration)")
    args = ap.parse_args()
    cond, sid, fold = args.cond, args.sid, args.fold
    def put(*a, **k):
        if not args.no_text:
            fig.text(*a, **k)

    row = load_manifest(sid)
    action = ACTIONS.get(str(row["action"]), str(row["action"]))
    trial = row.get("trial", "1")
    gt_s = protocol_gt_surface(row, "smpl24")
    if gt_s is None:
        raise RuntimeError("no GT surface")
    gt_v = np.asarray(gt_s["vertices"], np.float64)
    gt_j = np.asarray(protocol_gt(row, "smpl24")[0], np.float64)
    ours = load_ours(cond, fold, sid, "VT2M")
    wham = load_wham(cond, sid)

    n = min(len(gt_v), len(gt_j), len(ours["v"]), len(ours["j"]),
            len(wham["v"]), len(wham["j"]))
    gt_v, gt_j = gt_v[:n], gt_j[:n]
    ov, oj = ours["v"][:n], ours["j"][:n]
    wv, wj = wham["v"][:n], wham["j"][:n]
    valid = (ours["valid"][:n] & wham["valid"][:n]
             & np.isfinite(wv).all((1, 2)) & np.isfinite(ov).all((1, 2)))
    valid &= np.linalg.norm(gt_j[:, 0], axis=-1) > 0.1
    # root-anchor both predictions on the GT pelvis at every frame
    wv_a = wv - wj[:, 0:1, :] + gt_j[:, 0:1, :]
    ov_a = ov - oj[:, 0:1, :] + gt_j[:, 0:1, :]
    o_rel = oj - oj[:, 0:1]
    g_rel = gt_j - gt_j[:, 0:1]
    err_v = np.linalg.norm(o_rel - g_rel, axis=-1).mean(axis=1) * 1000.0
    if args.gap_phases:
        err_w = np.linalg.norm((wj - wj[:, 0:1]) - g_rel, axis=-1).mean(axis=1) * 1000.0
    if args.world_phys:
        # anchor only horizontal root displacement; keep the prediction's own
        # vertical placement so true floor contact (penetration) remains visible
        wv_xy = wv.copy(); wv_xy[..., :2] += (gt_j[:, 0:1, :2] - wj[:, 0:1, :2])
        ov_xy = ov.copy(); ov_xy[..., :2] += (gt_j[:, 0:1, :2] - oj[:, 0:1, :2])
        draw_wv, draw_ov = wv_xy, ov_xy
    else:
        draw_wv, draw_ov = wv_a, ov_a
    if args.gap_phases:
        picks = gap_picks(sid, cond, n, valid, wv_a, wj, gt_j, ours["fps"])
    else:
        picks, _, err_w = pick_frames(draw_wv, wj, gt_j, valid, ours["fps"], args.ncol, err_v)
        picks = picks[: args.ncol]

    faces = faces_from_smpl()
    fps = ours["fps"]
    fig = plt.figure(figsize=(24.0 if len(picks) >= 5 else 16.8, 9.8))
    fig.patch.set_facecolor("white")

    put(0.5, 0.975, "E5 demo teaser — %s — %s (%s, %s t%s)"
             % (TITLE[cond], sid, row["subject_id"], action, trial),
             ha="center", va="top", fontsize=17, color=INK)
    put(0.5, 0.944,
             "root-anchored at GT pelvis (trajectory ignored)   ·   grey ghost = GT   ·   top: WHAM   ·   bottom: Ours VT2M   ·   "
             "solid red ring/bar = vertex below floor   ·   dashed red ring/bar = feet above floor   ·   dashed ellipse = worst GT mismatch",
             ha="center", va="top", fontsize=10.2, color=MUTED)

    if args.no_text:
        top_y, bot_y, panel_h = 0.515, 0.020, 0.470
    else:
        top_y, bot_y, panel_h = 0.545, 0.065, 0.365
    for k, t in enumerate(picks):
        left = 0.035 + k * (0.93 / len(picks))
        width = 0.93 / len(picks) - 0.012
        # top: WHAM
        ax1 = fig.add_axes([left, top_y, width, panel_h], projection="3d")
        # bottom: VT
        ax2 = fig.add_axes([left, bot_y, width, panel_h], projection="3d")
        # centre both bodies on the GT pelvis of this frame
        c = gt_j[t, 0, :2]
        gv = gt_v[t] - np.array([c[0], c[1], 0.0])
        wh = draw_wv[t] - np.array([c[0], c[1], 0.0])
        vt = draw_ov[t] - np.array([c[0], c[1], 0.0])

        e_w = err_w[t] if np.isfinite(err_w[t]) else float("nan")
        e_v = err_v[t] if np.isfinite(err_v[t]) else float("nan")
        pen_w = -float(wh[:, 2].min())
        worst_w = int(np.argmax([err_w[x] if np.isfinite(err_w[x]) else -1 for x in picks]))

        draw_panel(ax1, gv, wh, faces, WHAM_RED, pen_ring=True,
                   mismatch_ring=(k == worst_w))
        draw_panel(ax2, gv, vt, faces, VT_BLUE, pen_ring=False, mismatch_ring=False)

        put(left + width / 2, 0.925, "t = %.1f s" % (t / fps),
                 ha="center", va="top", fontsize=11, color=MUTED)
        put(left + width / 2, 0.535,
                 "WHAM pose  %.0f mm" % e_w, ha="center", va="top",
                 fontsize=10.0, color=WHAM_RED, fontweight="bold")
        minz_w = float(wh[:, 2].min())
        if pen_w > 0.004:
            foot_txt = "foot %.0f mm below floor" % (pen_w * 1000.0)
        elif minz_w > 0.02:
            foot_txt = "feet +%.0f mm above floor" % (minz_w * 1000.0)
        else:
            foot_txt = "feet on floor"
        put(left + width / 2, 0.505, foot_txt,
                 ha="center", va="top", fontsize=9.0, color=WHAM_RED)
        put(left + width / 2, 0.055,
                 "VT pose  %.0f mm" % e_v, ha="center", va="top",
                 fontsize=10.0, color=VT_BLUE)
        if k == worst_w:
            put(left + width / 2, 0.465, "dashed ring = worst GT mismatch",
                     ha="center", va="top", fontsize=8.0,
                     color=WHAM_RED, style="italic")

    put(0.012, 0.735, "WHAM\nvision-only", rotation=90, va="center",
             ha="center", fontsize=13, color=WHAM_RED, fontweight="bold")
    put(0.012, 0.250, "Ours VT2M\n+ tactile", rotation=90, va="center",
             ha="center", fontsize=13, color=VT_BLUE, fontweight="bold")

    handles = [Patch(facecolor=WHAM_RED, label="WHAM"),
               Patch(facecolor=VT_BLUE, label="Ours VT2M"),
               Patch(facecolor=GT_GRAY, alpha=0.6, label="GT (ghost)"),
               Line2D([0], [0], color=WHAM_RED, lw=2, label="below-floor marker"),
               Line2D([0], [0], color=WHAM_RED, lw=1.5, ls=(0, (3, 3)),
                      label="above-floor marker"),
               Line2D([0], [0], color=WHAM_RED, lw=1.3, ls=(0, (4, 3)),
                      label="GT-mismatch ellipse")]
    if not args.no_text:
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
                   frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.0))

    tag_prefix = ("_" + args.tag + "_") if args.tag else ""
    out = args.out or (OUT / SCENE_DIR[cond] /
                       ("%sTeaser_%s_%s_%s%s.png" % (tag_prefix, cond, sid, action.replace("-", ""),
                                                     "_phases" if args.gap_phases else "")))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print("wrote", out)
    for k, t in enumerate(picks):
        print("  col %d: frame %d t=%.2fs  WHAM-GT %.1f mm  VT-GT %.1f mm  "
              "WHAM min-z %.1f mm"
              % (k, t, t / fps, err_w[t], err_v[t], wv[t, :, 2].min() * 1000.0))


if __name__ == "__main__":
    main()
