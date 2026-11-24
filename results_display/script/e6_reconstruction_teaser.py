#!/usr/bin/env python3
"""E6 reconstruction teaser: V vs VT predictions with GT outline.

Same session/six moments as the existing teaser; four occlusion rows.
Predictions are rendered at the GT root translation so the comparison isolates
pose/foot placement; the GT silhouette edge is overlaid as a dark outline.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for entry in (REPO, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
import e6_ft_contribution as e6
from anysole.phase_common import load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

SURFACE, INK, MUTED = e6.SURFACE, e6.INK, e6.MUTED
CONDS = e6.CONDS
PRED = REPO / "results/diagnostics/20261114_imu_eval_v2_motion/VT_teaser"
COLORS = {"V": "#86b6ef", "VT": "#2a78d6"}
GT_EDGE = "#52514e"

def load_pred(cond, mode, sid):
    p = PRED / cond / ("%s_%s.npz" % (sid, mode))
    d = np.load(p)
    poses = np.asarray(d["poses"], np.float64).reshape(-1, 24, 3)
    trans = np.asarray(d["trans"], np.float64)
    betas = np.asarray(d["betas"], np.float64).reshape(-1)[:10]
    return poses, trans, betas

def silhouette_edge(image):
    bg = np.array([int(SURFACE[1:3], 16), int(SURFACE[3:5], 16),
                   int(SURFACE[5:7], 16)], np.float32)
    mask = np.linalg.norm(image - bg, axis=-1) > 8.0
    er = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~er

def render_one(poses, trans, betas, rest, faces, frame, azim, color):
    verts = e6.heat.smpl_vertices(poses[frame:frame + 1],
                                  trans[frame:frame + 1], betas, rest)[0]
    verts_display = e6.heat.smpl_yup_to_display(verts)
    img = e6.heat.render_mesh_frame(
        verts_display, faces,
        vertex_colors=np.tile(np.array(matplotlib_color(color)), (len(verts_display), 1)),
        ambient=0.85, light_intensity=0.8, bg_color=SURFACE, size=(360, 360),
        elev=8.0, azim=azim)
    return img[70:350, 70:350]

def matplotlib_color(hex_color):
    from matplotlib.colors import to_rgb
    return to_rgb(hex_color)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", default="S11051")
    ap.add_argument("--out-dir", type=Path, default=e6.DEFAULT_OUT.parents[1])
    ap.add_argument("--picks", default="",
                    help="comma-separated frame indices for the six snapshots")
    args = ap.parse_args()
    sid = args.session_id
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % sid))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], np.float64); t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, _ = load_contact(sid)
    motion = e6.load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                          query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]))
    valid_pred = None
    for cond, _ in CONDS:
        for mode in ("V2M", "VT2M"):
            d = np.load(PRED / cond / ("%s_%s.npz" % (sid, mode)))
            v = np.asarray(d["valid_mask"], bool)
            valid_pred = v if valid_pred is None else (valid_pred & v)
    picks = np.flatnonzero(valid[:n] & valid_pred[:n])
    if args.picks:
        cand = [int(x) for x in args.picks.split(",") if x.strip()]
        cand = [x for x in cand if 0 <= x < n and valid[x] and valid_pred[x]]
        if len(cand) == 6:
            picks = np.asarray(sorted(cand))
        else:
            picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    else:
        picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    gt_poses = np.asarray(motion["poses_aa"], np.float64).reshape(-1, 24, 3)
    gt_trans = np.asarray(motion["trans_m"], np.float64)
    gt_betas = np.asarray(motion["betas"], np.float64).reshape(-1)[:10]
    rest = e6.heat._rest_joints({"betas": gt_betas})
    _, faces, weights = e6.heat._canonical_mesh()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(23.0, 12.8), facecolor=SURFACE)
    gs = GridSpec(len(CONDS) + 1, 12, figure=fig,
                  height_ratios=[1.0] * len(CONDS) + [0.32],
                  hspace=0.03, wspace=0.01,
                  left=0.045, right=0.985, top=0.90, bottom=0.06)
    for row, (cond, _) in enumerate(CONDS):
        p_v, tr_v, b_v = load_pred(cond, "V2M", sid)
        p_vt, tr_vt, b_vt = load_pred(cond, "VT2M", sid)
        for col, f in enumerate(picks):
            f = int(f)
            rot = SciRotation.from_rotvec(gt_poses[f, 0]).as_matrix()
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            azim = float(np.degrees(np.arctan2(-fwd[2], fwd[0])) + 20.0)
            # GT is rendered at its own root; predictions use the GT root
            # translation so the panel compares pose, not global drift.
            gt_img = render_one(gt_poses, gt_trans, gt_betas, rest, faces, f, azim, "#cfcdc7")
            edge = ndimage.binary_dilation(silhouette_edge(gt_img), iterations=1)
            for k, (mode, poses, betas) in enumerate(
                    (("V", p_v, b_v), ("VT", p_vt, b_vt))):
                trans = np.repeat(gt_trans[f:f + 1], len(poses), axis=0)
                img = render_one(poses, trans, betas, rest, faces, f, azim,
                                 COLORS[mode])
                img = img.copy()
                img[edge] = np.array([82, 81, 78], np.float32)
                ax = fig.add_subplot(gs[row, 2 * col + k])
                ax.imshow(img.astype(np.uint8))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(mode, fontsize=8.5, color=INK, pad=2)
                if col == 0 and k == 0:
                    ax.set_ylabel(cond, fontsize=10.5, color=INK, rotation=0,
                                  labelpad=16, va="center")
    step = (t[n - 1] - t[0]) / max(len(picks) - 1, 1)
    pad = int(round(contact.shape[0] * (step / 2) / (t[n - 1] - t[0])))
    band = np.pad(contact[:n].T.astype(np.float32), ((0, 0), (pad, pad)),
                  constant_values=0.0)
    axc = fig.add_subplot(gs[-1, :])
    axc.imshow(band, aspect="auto",
               cmap=plt.matplotlib.colors.ListedColormap(["#fcfcfb", "#b9c6cf"]),
               vmin=0.0, vmax=1.0, interpolation="nearest",
               extent=[t[0] - step / 2, t[n - 1] + step / 2, 0, 2])
    axc.set_xlim(t[0] - step / 2, t[n - 1] + step / 2)
    axc.set_yticks([0.5, 1.5]); axc.set_yticklabels(["L", "R"], fontsize=8.0, color=MUTED)
    axc.set_xticks([t[f] for f in picks])
    axc.set_xticklabels(["%.1f" % t[f] for f in picks], fontsize=8.5, color=MUTED)
    for f in picks:
        axc.axvline(t[f], color="#8a8980", lw=0.8, ls=":")
    axc.tick_params(length=2, pad=2)
    for spine in axc.spines.values():
        spine.set_visible(False)
    axc.set_ylabel("contact", fontsize=8.5, color=MUTED, labelpad=6)
    axc.set_xlabel("time (s)", fontsize=9.5, color=MUTED, labelpad=4)
    fig.suptitle("Residual Fusion: V vs VT predictions with GT outline — %s" % sid,
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.975)
    fig.text(0.50, 0.925,
             "Session %s; GT SMPL outline (dark edge); V light blue, VT dark blue; "
             "predictions rendered at the GT root to compare pose/foot placement; "
             "bottom = f6_soft contact." % sid,
             fontsize=8.6, color=MUTED, ha="center")
    p = args.out_dir / ("Main_Fig_E6_Reconstruction_Teaser_%s.png" % sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)

if __name__ == "__main__":
    main()
