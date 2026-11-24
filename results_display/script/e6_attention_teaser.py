#!/usr/bin/env python3
"""E6 attention-share teaser for V8_IMU_Residual (2026-11-15).

Colours the SMPL body by the per-part T attention share
a_T = w_T / (w_V + w_T) from the decoder cross-attention, at six moments of
one trial under four visual conditions.
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

import e6_ft_contribution as e6  # noqa: E402
from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import CONFIG_VT, PART_JOINTS  # noqa: E402
from anysole.phase_common import load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

SURFACE, INK, MUTED = e6.SURFACE, e6.INK, e6.MUTED
CMAP_STOPS = [(0.0, "#0d366b"), (0.25, "#2a78d6"), (0.5, "#dedddb"),
              (0.75, "#e34948"), (1.0, "#7f1d1d")]

def colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("att_t", CMAP_STOPS)

def collect_attention(model, cfg, split_csv, sid, visual_root, device,
                      batch_size):
    """Return {cond: (n_frames, 24) a_T} for one session."""
    out = {}
    for cond, cache in e6.CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=[sid],
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        n_frames = len(ds.sessions[0]["V_feat"])
        full = np.full((n_frames, 24), np.nan, np.float32)
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n_b = batch["V_feat"].shape[0]
            cap = {"v": [], "t": []}
            originals = []
            for layer in model.pose_head.layers:
                for kind, mod in (("v", layer.xattn_v), ("t", layer.xattn_t)):
                    orig = mod.forward
                    originals.append((mod, orig))
                    def make(kind=kind, orig=orig):
                        def f(q, k, v, need_weights=False, attn_mask=None, **kw):
                            o, w = orig(q, k, v, need_weights=True,
                                       attn_mask=attn_mask, **kw)
                            cap[kind].append(w.detach().float().cpu())
                            return o, w
                        return f
                    mod.forward = make()
            cid = torch.full((n_b,), CONFIG_VT, dtype=torch.long, device=device)
            with torch.inference_mode():
                model(batch["V_feat"], batch["T_raw"], batch["T_phys"], cid,
                      T_s2m=batch["T_s2m"])
            for mod, orig in originals:
                mod.forward = orig
            tw, nq = int(cfg["tw"]), 10
            # per-layer share, average over heads (already averaged) and memory
            per_layer = []
            idx = torch.arange(tw)
            for wv, wt in zip(cap["v"], cap["t"]):
                wv = wv.reshape(n_b, tw, nq, tw)
                wt = wt.reshape(n_b, tw, nq, tw)
                bshare = []
                for i in range(n_b):
                    dv = wv[i, idx, :, idx]  # (tw,nq) diagonal
                    dt = wt[i, idx, :, idx]
                    bshare.append((dt / (dv + dt + 1e-6)).numpy())
                per_layer.append(np.stack(bshare, axis=0))  # (B,tw,nq)
            a = np.mean(np.stack(per_layer, axis=0), axis=0)  # (B,tw,nq)
            for i, left in enumerate(batch["frame_start"].tolist()):
                joint = np.full((tw, 24), 0.5, np.float32)
                for slot in range(1, nq):
                    joints = list(PART_JOINTS[slot - 1])
                    joint[:, joints] = a[i, :, slot][:, None]
                full[left:left + tw] = joint
        out[cond] = full
    return out

def render(att, sid, fold, out_dir):
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % sid))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, _ = load_contact(sid)
    motion = e6.load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                          query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]))
    picks = np.flatnonzero(valid[:n] & np.isfinite(att["clean"][:n, 0]))
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
    gs = GridSpec(len(e6.CONDS) + 1, len(picks), figure=fig,
                  height_ratios=[1.0] * len(e6.CONDS) + [0.42],
                  hspace=0.03, wspace=0.015,
                  left=0.045, right=0.925, top=0.895, bottom=0.09)
    for row, (cond, _) in enumerate(e6.CONDS):
        for col, f in enumerate(picks):
            f = int(f)
            per_joint = np.nanmean(att[cond][max(0, f - 4):f + 5], axis=0)
            vertex_values = e6.heat._vertex_values(per_joint, weights)
            verts = e6.heat.smpl_vertices(poses[f:f + 1], trans[f:f + 1],
                                          betas, rest)[0]
            verts_display = e6.heat.smpl_yup_to_display(verts)
            rot = SciRotation.from_rotvec(poses[f, 0]).as_matrix()
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            disp = np.array([fwd[0], -fwd[2], fwd[1]])
            azim = float(np.degrees(np.arctan2(disp[1], disp[0])) + 20.0)
            colors = colormap()(Normalize(0.0, 1.0, clip=True)(
                vertex_values))[:, :3]
            image = e6.heat.render_mesh_frame(
                verts_display, faces, vertex_colors=colors, ambient=0.85,
                light_intensity=0.8, bg_color=SURFACE, size=(380, 380),
                elev=8.0, azim=azim)
            image = e6.heat._crop_body(image)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
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
    axc.set_yticks([0.5, 1.5]); axc.set_yticklabels(["L", "R"], fontsize=8.0,
                                                   color=MUTED)
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
    cb.set_label("T attention share a_T", fontsize=9.0, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Residual Fusion: T attention share at six moments of one %s trial"
                 % sid, fontsize=12.5, color=INK, x=0.02, ha="left", y=0.975)
    fig.text(0.50, 0.925,
             "Session %s (%s fold); GT SMPL pose; colour = a_T = w_T/(w_V+w_T) "
             "from decoder cross-attention, averaged over layers and ±100 ms; "
             "blue = V, red = T. Per-frame part spread is ~0.01: attention is "
             "part-agnostic; see the ΔT teaser for part-wise effects."
             % (sid, fold), fontsize=8.4, color=MUTED,
             ha="center")
    path = out_dir / ("Main_Fig_E6_Attention_Teaser_%s.png" % sid)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)
    # per-frame CSV
    detail = out_dir / "detail" / "attention_teaser"
    detail.mkdir(parents=True, exist_ok=True)
    csv_path = detail / ("E6_AttentionShare_%s.csv" % sid)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s"] + ["joint_%d" % i for i in range(24)])
        for i in range(n):
            w.writerow([i, float(t[i])] + [float(x) for x in att["clean"][i]])
    print("wrote", csv_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-template", default=str(
        e6.DIAG / "router_ft_only_{fold}_v2rerun.pt"))
    ap.add_argument("--session-id", default="S11051")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--out-dir", type=Path, default=e6.DEFAULT_OUT.parents[1])
    ap.add_argument("--visual-root", type=Path, default=e6.DEFAULT_VISUAL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    split_csv = e6.SPLITS / e6.FOLDS[args.fold]
    model, cfg, _ = e6._load_model(Path(args.checkpoint_template.format(
        fold=args.fold)))
    device = torch.device(args.device)
    model.to(device).eval()
    att = collect_attention(model, cfg, split_csv, args.session_id,
                            args.visual_root, device, args.batch_size)
    render(att, args.session_id, args.fold, args.out_dir)

if __name__ == "__main__":
    main()
