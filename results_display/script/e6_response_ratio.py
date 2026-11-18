#!/usr/bin/env python3
"""E6 part-wise V/T response ratio r_T = ||A_T|| / (||A_V|| + ||A_T||).

Uses the actual cross-attention outputs injected into the decoder residual,
averaged over layers (fixed scheme), then averaged per window/part.
Produces a part x condition heatmap and a dynamic SMPL teaser.
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
import e6_attention_teaser as teaser
from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import CONFIG_VT, PART_JOINTS, PART_NAMES  # noqa: E402
from anysole_tta.phase_common import load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

SURFACE, INK, MUTED = e6.SURFACE, e6.INK, e6.MUTED
CMAP_STOPS = [(0.0, "#0d366b"), (0.25, "#2a78d6"), (0.5, "#dedddb"),
              (0.75, "#e34948"), (1.0, "#7f1d1d")]
CONDS = e6.CONDS

def colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("response_t", CMAP_STOPS)

def _patch(model, cap):
    originals = []
    for layer in model.pose_head.layers:
        for kind, mod in (("v", layer.xattn_v), ("t", layer.xattn_t)):
            orig = mod.forward
            originals.append((mod, orig))
            def make(kind=kind, orig=orig):
                def f(q, k, v, need_weights=False, attn_mask=None, **kw):
                    o, w = orig(q, k, v, need_weights=False,
                                attn_mask=attn_mask, **kw)
                    cap[kind].append(o.detach())
                    return o, w
                return f
            mod.forward = make()
    return originals

def _response(model, batch, device, tw):
    b = batch["V_feat"].shape[0]
    cap = {"v": [], "t": []}
    originals = _patch(model, cap)
    cid = torch.full((b,), CONFIG_VT, dtype=torch.long, device=device)
    with torch.inference_mode():
        model(batch["V_feat"], batch["T_raw"], batch["T_phys"], cid,
              T_s2m=batch["T_s2m"])
    for mod, orig in originals:
        mod.forward = orig
    nq = 10
    per_layer = []
    for v, t in zip(cap["v"], cap["t"]):
        v = v.reshape(b, tw, nq, -1).norm(dim=-1)
        t = t.reshape(b, tw, nq, -1).norm(dim=-1)
        per_layer.append((t / (v + t + 1e-6)).cpu().numpy())
    return np.mean(np.stack(per_layer, 0), 0)  # (b,tw,nq)

def collect(model, cfg, split_csv, ids, visual_root, device, batch_size):
    tw = int(cfg["tw"])
    out = {}
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=tw, session_ids=ids,
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        wins = []
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            r = _response(model, batch, device, tw)  # (b,tw,10)
            for i in range(r.shape[0]):
                wins.append((str(batch["session_id"][i]), int(batch["frame_start"][i]),
                             r[i].mean(axis=0)))
        out[cond] = wins
    return out

def heatmap(wins, out_dir):
    parts = list(PART_NAMES)
    mat = np.zeros((len(parts), len(CONDS)))
    for ci, (cond, _) in enumerate(CONDS):
        arr = np.stack([w[2][1:] for w in wins[cond]], 0)  # slots 1..9
        mat[:, ci] = arr.mean(axis=0)
    p = out_dir / "Main_Fig_E6_ResponseRatio_Parts.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor=SURFACE)
    im = ax.imshow(mat, cmap=colormap(), vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([c for c, _ in CONDS], fontsize=9, color=MUTED)
    ax.set_yticks(range(len(parts)))
    ax.set_yticklabels(parts, fontsize=9, color=MUTED)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%.3f" % mat[i, j], ha="center", va="center",
                    fontsize=7.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("r_T = ||A_T|| / (||A_V|| + ||A_T||)", fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("Part-wise T response share (layer mean)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    csv_path = out_dir / "detail" / "response_ratio" / "E6_ResponseRatio_byPart.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "condition", "r_T", "n_windows"])
        for ci, (cond, _) in enumerate(CONDS):
            n = len(wins[cond])
            for pi, part in enumerate(parts):
                w.writerow([part, cond, float(mat[pi, ci]), n])
    print("wrote", csv_path)

def teaser_values(wins, sid):
    out = {}
    for cond, _ in CONDS:
        rows = [w for w in wins[cond] if w[0] == sid]
        if not rows:
            out[cond] = None
            continue
        n = None
        # reconstruct frame range from window starts; use max start + tw
        tw = 20
        n = max(w[1] for w in rows) + tw
        vals = np.full((n, 24), np.nan, np.float32)
        for _, left, r in rows:
            joint = np.full((tw, 24), np.nan, np.float32)
            for slot in range(1, 10):
                joints = list(PART_JOINTS[slot - 1])
                joint[:, joints] = r[slot]
            vals[left:left + tw] = joint
        out[cond] = vals
    return out

def render_teaser(values, sid, fold, out_dir):
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
            colors = colormap()(Normalize(0.0, 1.0, clip=True)(vertex_values))[:, :3]
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
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=colormap())
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("T response share r_T", fontsize=9.0, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Residual Fusion: T response share at six moments of one %s trial"
                 % sid, fontsize=12.5, color=INK, x=0.02, ha="left", y=0.975)
    fig.text(0.50, 0.925,
             "Session %s (%s fold); GT SMPL pose; colour = r_T = ||A_T||/(||A_V||+||A_T||) "
             "from decoder cross-attention outputs (layer mean), per-window part mean; "
             "blue = V, red = T; common 0–1 scale." % (sid, fold),
             fontsize=8.4, color=MUTED, ha="center")
    p = out_dir / ("Main_Fig_E6_ResponseRatio_Teaser_%s.png" % sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    csv_path = out_dir / "detail" / "response_ratio" / ("E6_ResponseRatio_Teaser_%s.csv" % sid)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split_csv = e6.SPLITS / e6.FOLDS[args.fold]
    ids = load_split_ids(split_csv, "test")
    model, cfg, _ = e6._load_model(Path(args.checkpoint_template.format(
        fold=args.fold)))
    device = torch.device(args.device)
    model.to(device).eval()
    wins = collect(model, cfg, split_csv, ids, args.visual_root, device, args.batch_size)
    heatmap(wins, args.out_dir)
    vals = teaser_values(wins, args.session_id)
    if vals["clean"] is not None:
        render_teaser(vals, args.session_id, args.fold, args.out_dir)

if __name__ == "__main__":
    main()
