#!/usr/bin/env python3
"""E6 part-wise absolute-error teaser, one file per input mode (V/T/VT).

Layout matches the response-ratio teaser: 4 occlusion rows x 6 time snapshots,
GT SMPL carrier, contact band; body colour = per-part PA-MPJPE (mm).
All modes/conditions share one colour scale.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for entry in (REPO, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
import e6_ft_contribution as e6
from anysole.data.dataset import AnySoleDataset  # noqa: E402
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT, PART_JOINTS  # noqa: E402
from anysole.phase_common import load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

SURFACE, INK, MUTED = e6.SURFACE, e6.INK, e6.MUTED
CONDS = e6.CONDS
CIDS = {"V": CONFIG_V, "T": CONFIG_T, "VT": CONFIG_VT}

def collect(model, cfg, split_csv, sid, visual_root, device, batch_size):
    tw = int(cfg["tw"])
    out = {m: {} for m in CIDS}
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=tw, session_ids=[sid],
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        n = len(ds.sessions[0]["V_feat"])
        full = {m: np.full((n, 24), np.nan, np.float32) for m in CIDS}
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            b = batch["V_feat"].shape[0]
            for mode, cidv in CIDS.items():
                cid = torch.full((b,), cidv, dtype=torch.long, device=device)
                with torch.inference_mode():
                    o = model(batch["V_feat"], batch["T_raw"], batch["T_phys"],
                              cid, T_s2m=batch["T_s2m"])
                pj, _ = e6._per_joint_pa(o, batch)
                pj = pj.cpu().numpy()
                for i, left in enumerate(batch["frame_start"].tolist()):
                    full[mode][left:left + tw] = pj[i]
        for mode in CIDS:
            out[mode][cond] = full[mode]
    return out

def part_values(full):
    """(n,24) per-joint error -> (n,24) part-level error repeated on joints."""
    out = full.copy()
    for pi, joints in enumerate(PART_JOINTS):
        joints = list(joints)
        part = np.nanmean(full[:, joints], axis=1)
        out[:, joints] = part[:, None]
    return out

def render(values, mode, sid, fold, out_dir, vmin, vmax, picks_override=None):
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % sid))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64); t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, _ = load_contact(sid)
    motion = e6.load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                          query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]),
            values["clean"].shape[0])
    picks = np.flatnonzero(valid[:n] & np.isfinite(values["clean"][:n, 0]))
    if picks_override:
        cand = [int(x) for x in picks_override
                if 0 <= int(x) < n and valid[int(x)]
                and np.isfinite(values["clean"][int(x), 0])]
        if len(cand) == 6:
            picks = np.asarray(sorted(cand))
        else:
            picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    else:
        picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    poses = np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(motion["trans_m"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)[:10]
    rest = e6.heat._rest_joints({"betas": betas})
    _, faces, weights = e6.heat._canonical_mesh()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    cmap = plt.get_cmap("cividis")
    fig = plt.figure(figsize=(17.2, 11.0), facecolor=SURFACE)
    gs = GridSpec(len(CONDS) + 1, len(picks), figure=fig,
                  height_ratios=[1.0] * len(CONDS) + [0.42],
                  hspace=0.03, wspace=0.015,
                  left=0.045, right=0.925, top=0.895, bottom=0.09)
    for row, (cond, _) in enumerate(CONDS):
        for col, f in enumerate(picks):
            f = int(f)
            per_joint = np.nanmean(values[cond][max(0, f - 4):f + 5], axis=0)
            vertex_values = e6.heat._vertex_values(per_joint, weights)
            verts = e6.heat.smpl_vertices(poses[f:f + 1], trans[f:f + 1],
                                          betas, rest)[0]
            verts_display = e6.heat.smpl_yup_to_display(verts)
            rot = SciRotation.from_rotvec(poses[f, 0]).as_matrix()
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            disp = np.array([fwd[0], -fwd[2], fwd[1]])
            azim = float(np.degrees(np.arctan2(disp[1], disp[0])) + 20.0)
            colors = cmap(Normalize(vmin, vmax, clip=True)(vertex_values))[:, :3]
            image = e6.heat.render_mesh_frame(
                verts_display, faces, vertex_colors=colors, ambient=0.85,
                light_intensity=0.8, bg_color=SURFACE, size=(380, 380),
                elev=8.0, azim=azim)
            image = e6.heat._crop_body(image)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title("t = %.1f s" % t[f], fontsize=9.5, color=INK, pad=3)
            if col == 0:
                ax.set_ylabel(cond, fontsize=10.5, color=INK, rotation=0,
                              labelpad=14, va="center")
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
    cax = fig.add_axes([0.935, 0.27, 0.012, 0.50])
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Part PA-MPJPE (mm; clipped %.0f–%.0f; dark = low, light = high)" % (vmin, vmax),
                 fontsize=9.0, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    label = {"V": "V-only", "T": "T-only", "VT": "VT"}[mode]
    fig.suptitle("%s absolute part error at six moments of one %s trial"
                 % (label, sid), fontsize=12.5, color=INK, x=0.02,
                 ha="left", y=0.975)
    fig.text(0.50, 0.925,
             "Session %s (%s fold); GT SMPL pose; colour = per-part PA-MPJPE (mm), "
             "±100 ms mean; common %.0f–%.0f mm scale across V/T/VT and conditions "
             "(values clipped for visibility)." % (sid, fold, vmin, vmax),
             fontsize=8.4, color=MUTED, ha="center")
    p = out_dir / ("Main_Fig_E6_PartError_Teaser_%s_%s.png" % (mode, sid))
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    detail = out_dir / "detail" / "part_error_teaser"
    detail.mkdir(parents=True, exist_ok=True)
    csv_path = detail / ("E6_PartError_Teaser_%s_%s.csv" % (mode, sid))
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s"] + ["joint_%d" % i for i in range(24)])
        for i in range(n):
            w.writerow([i, float(t[i])] + [float(x) for x in values["clean"][i]])
    print("wrote", csv_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-template", default=str(
        e6.DIAG / "router_ft_only_{fold}_v2rerun.pt"))
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--session-id", default="S11051")
    ap.add_argument("--out-dir", type=Path, default=e6.DEFAULT_OUT.parents[1])
    ap.add_argument("--visual-root", type=Path, default=e6.DEFAULT_VISUAL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--picks", default="",
                    help="comma-separated frame indices for the six snapshots")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split_csv = e6.SPLITS / e6.FOLDS[args.fold]
    model, cfg, _ = e6._load_model(Path(args.checkpoint_template.format(
        fold=args.fold)))
    device = torch.device(args.device)
    model.to(device).eval()
    raw = collect(model, cfg, split_csv, args.session_id, args.visual_root,
                  device, args.batch_size)
    vals = {m: {c: part_values(raw[m][c]) for c in raw[m]} for m in raw}
    # Unified scale taken from the part-mean table (~20-102 mm); clipped for
    # visibility.  T-only high-error parts saturate, which is the intended
    # comparison against V/VT.
    vmin, vmax = 20.0, 110.0
    picks = [int(x) for x in args.picks.split(",") if x.strip()]
    for mode in ("V", "T", "VT"):
        render(vals[mode], mode, args.session_id, args.fold, args.out_dir,
               vmin, vmax, picks)

if __name__ == "__main__":
    main()
