#!/usr/bin/env python3
"""E6：FT（enum fine-tune，无 router）的模态贡献图与表（2026-11-14）。

口径
----
- 模型：`V8_Cv7_PhysIMU` enum fine-tune，默认 V2 FT checkpoint（v2rerun）
  `results/diagnostics/20261112_v8_gate_frozen/router_ft_only_<fold>_v2rerun.pt`。
- 四折标准 split；每折 test 上分别跑 V-only / T-only / VT。
- 逐关节 PA-MPJPE（sim(3) 对齐，与 `_pa_frames` 同公式），按 9 个 SMPL
  部位聚合；根速度单独用 `_pa_frames` 的 root 项。
- ΔT = E(VT) − E(V)，负 = T 有帮助；ΔV = E(VT) − E(T)，负 = V 有帮助。
- 产物：
  `Main_Fig_E6_FT_Contribution_SMPL.png`  2 行（ΔT/ΔV）× 4 档 SMPL 图
  `Main_Table1_E6_FT_Contribution_byPart.csv`
  `detail/E6_FT_PerJoint_byCondition.csv`
  `teaser/Main_Fig_E6_FT_Contribution_Teaser_<sid>.png`（--teaser）

只读 checkpoint / 缓存，不训练。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.data.smpl_io import load_smpl  # noqa: E402
from anysole.types import (CONFIG_T, CONFIG_V, CONFIG_VT, JOINT_NAMES,  # noqa: E402
                           PART_JOINTS, PART_NAMES)
from anysole.phase_common import load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

DIAG = REPO / "results" / "diagnostics" / "20261112_v8_gate_frozen"
sys.path.insert(0, str(DIAG))
from train_tgate_win_hb import _fk_pos  # noqa: E402

import e5_v8_smpl_gain_heatmap as heat  # noqa: E402

DEFAULT_OUT = (REPO / "results_display" / "Etest_crossfit_physimu_vgate"
               / "E6_ContactPhaseAnalysis" / "detail" / "contribution")
DEFAULT_VISUAL = (REPO / "AnysoleWorkspace" / "model_inputs" / "AnySole"
                  / "adapter_v1" / "visual" / "hrnet")
SPLITS = REPO / "AnysoleWorkspace" / "protocol" / "splits" / "default"
FOLDS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12.csv",
    "fS5S11S12": "splits_test_S5S11S12_val_S13.csv",
    "fS6S7": "splits_test_S6S7_val_S14.csv",
    "fS5S6S8": "splits_test_S5S6S8_val_S7.csv",
}
CONDS = (
    ("clean", "cam3"),
    ("occ020", "cam3_occ_lower020"),
    ("occ050", "cam3_occ_lower050"),
    ("occ090", "cam3_occ_lower090"),
)
MODES = ("V", "T", "VT")
CIDS = {"V": CONFIG_V, "T": CONFIG_T, "VT": CONFIG_VT}
PARTS = list(PART_NAMES) + ["root_velocity"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def _load_model(ckpt: Path):
    from anysole.models.v8_cv7_physimu import STRUCTURE as BASE_S
    from anysole.models.v8_cv7_physimu import build as base_build
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = base_build({**cfg, **BASE_S})
    model.load_state_dict(ck["model"], strict=True)
    return model, cfg, ck


def _per_joint_pa(out, batch):
    """逐帧逐关节 PA-MPJPE（m）与 root 项；与 `_pa_frames` 同公式。"""
    pred = _fk_pos(out["x0_hat"], batch)
    gt = _fk_pos(batch["pose_gt"].to(pred.dtype), batch)
    b, tw, nj, _ = pred.shape
    p = pred.reshape(-1, nj, 3)
    g = gt.reshape(-1, nj, 3)
    pc = p - p.mean(dim=1, keepdim=True)
    gc = g - g.mean(dim=1, keepdim=True)
    cov = gc.transpose(1, 2) @ pc / float(nj)
    u, s, vh = torch.linalg.svd(cov)
    det = torch.det(u) * torch.det(vh)
    corr = torch.ones_like(s)
    corr[:, -1] = torch.where(det < 0, -torch.ones_like(det),
                              torch.ones_like(det))
    rot = u @ torch.diag_embed(corr) @ vh
    var = pc.square().sum(dim=(1, 2)) / float(nj)
    scale = (s * corr).sum(-1) / var.clamp(min=1e-12)
    aligned = (scale[:, None, None] * (pc @ rot.transpose(1, 2))
               + g.mean(dim=1, keepdim=True))
    per_joint = (aligned - g).norm(dim=-1).view(b, tw, nj) * 1000.0
    # 根速度组：v_hat 对 GT 速度的逐帧误差（不是 cumsum 轨迹误差）。
    root_err = (out["v_hat"] - batch["vel_gt"].to(out["v_hat"].dtype)
                ).norm(dim=-1) * 1000.0
    return per_joint, root_err


def _accumulate(stats, cond, mode, per_joint, root_err, n_frames):
    """stats[part][cond][mode] = [sum_mm, n] (per-joint mean per part)."""
    for part, joints in zip(PART_NAMES, PART_JOINTS):
        vals = per_joint[:, :, list(joints)].mean(dim=-1).reshape(-1)
        a = stats[part][cond][mode]
        a[0] += float(vals.sum())
        a[1] += int(vals.numel())
    a = stats["root_velocity"][cond][mode]
    a[0] += float(root_err.sum())
    a[1] += int(root_err.numel())


def _run_fold(model, cfg, split_csv: Path, ids: list, visual_root: Path,
              device: torch.device, batch_size: int, limit_sessions: int,
              stats, per_joint_stats):
    if limit_sessions:
        ids = ids[:limit_sessions]
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=ids,
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n = batch["V_feat"].shape[0]
            for mode in MODES:
                cid = torch.full((n,), CIDS[mode], dtype=torch.long,
                                 device=device)
                with torch.inference_mode():
                    out = model(batch["V_feat"], batch["T_raw"],
                                batch["T_phys"], cid, T_s2m=batch["T_s2m"])
                pj, root_err = _per_joint_pa(out, batch)
                _accumulate(stats, cond, mode, pj, root_err, n)
                per_joint_stats[cond][mode].append(
                    (pj.reshape(-1, pj.shape[-1]).mean(dim=0).cpu().numpy(),
                     int(pj.shape[0] * pj.shape[1])))
        print("[fold] %-9s %-6s done" % (split_csv.stem, cond), flush=True)


def _means(stats, part, cond, mode):
    s, n = stats[part][cond][mode]
    return s / max(n, 1)


def _ms(x):
    return (float(np.mean(x)), float(np.std(x, ddof=1))
            if len(x) > 1 else 0.0)


def _write_tables(out_dir: Path, fold_rows: dict):
    rows = []
    parts = PARTS
    for part in parts:
        for cond, _ in CONDS:
            vals = {m: [] for m in MODES}
            dts, dvs = [], []
            for fold in fold_rows:
                fr = fold_rows[fold]["parts"][part][cond]
                for m in MODES:
                    vals[m].append(fr[m])
                dts.append(fr["VT"] - fr["V"])
                dvs.append(fr["VT"] - fr["T"])
            ev, es = _ms(vals["V"])
            et, ets = _ms(vals["T"])
            evt, evts = _ms(vals["VT"])
            dt, dts_std = _ms(dts)
            dv, dvs_std = _ms(dvs)
            rows.append({
                "part": part, "condition": cond,
                "E_V": "%.3f±%.3f" % (ev, es),
                "E_T": "%.3f±%.3f" % (et, ets),
                "E_VT": "%.3f±%.3f" % (evt, evts),
                "delta_T": "%.3f±%.3f" % (dt, dts_std),
                "delta_V": "%.3f±%.3f" % (dv, dvs_std),
            })
    path = out_dir / "Main_Table1_E6_FT_Contribution_byPart.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[csv] %s" % path)
    md = out_dir / "Main_Table1_E6_FT_Contribution_byPart.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    nf = len(fold_rows)
    lines = ["# E6 FT 模态贡献（ΔT = VT−V，ΔV = VT−T，mm）", "",
             "负 = 该模态有帮助；%d 折 mean±std；差值按同折配对后计算。" % nf, "",
             "| 部位 | 条件 | E_V | E_T | E_VT | ΔT | ΔV |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % (r["part"], r["condition"], r["E_V"], r["E_T"],
                        r["E_VT"], r["delta_T"], r["delta_V"]))
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[md] %s" % md)

    # 逐关节（24）四折表，供复查热力图
    jpath = out_dir / "E6_FT_PerJoint_byCondition.csv"
    with jpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["joint", "part", "condition",
                         "delta_T_mean", "delta_T_std",
                         "delta_V_mean", "delta_V_std"])
        for cond, _ in CONDS:
            for j, name in enumerate(JOINT_NAMES):
                part = next(p for p, js in zip(PART_NAMES, PART_JOINTS)
                            if j in js)
                dt = [fold_rows[f]["per_joint"][cond]["T"][j]
                      for f in fold_rows]
                dv = [fold_rows[f]["per_joint"][cond]["V"][j]
                      for f in fold_rows]
                dtm, dts = _ms(dt)
                dvm, dvs = _ms(dv)
                writer.writerow([j, part, cond,
                                 "%.3f" % dtm, "%.3f" % dts,
                                 "%.3f" % dvm, "%.3f" % dvs])
    print("[csv] %s" % jpath)
    return rows


def _joint_vector(fold_rows: dict, cond: str, metric: str):
    """四折 mean per-joint Δ (metric = 'T' or 'V')."""
    vals = []
    for fold in fold_rows:
        vals.append(fold_rows[fold]["per_joint"][cond][metric])
    return np.mean(np.stack(vals, axis=0), axis=0)


def _compose(images, row_labels, col_labels, vmax, title, path, footnote):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_rows, n_cols = images.shape
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.3 * n_cols + 1.7, 2.6 * n_rows + 1.2),
                             facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(images[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10.5, color=INK, pad=4)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=10.5, color=INK,
                              labelpad=8)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.5, 0.015, footnote, fontsize=8.4, color=MUTED, ha="center")
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=heat._colormap())
    cax = fig.add_axes([0.91, 0.20, 0.012, 0.64])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("PA-MPJPE contribution (mm)", fontsize=9.0, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.055, right=0.885, top=0.90, bottom=0.12,
                        wspace=0.03, hspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def _render_main(out_dir: Path, fold_rows: dict):
    verts, faces, weights = heat._canonical_mesh()
    verts_display = heat.smpl_yup_to_display(verts)
    dT = {cond: heat._vertex_values(_joint_vector(fold_rows, cond, "T"), weights)
          for cond, _ in CONDS}
    dV = {cond: heat._vertex_values(_joint_vector(fold_rows, cond, "V"), weights)
          for cond, _ in CONDS}
    peak = max(float(np.abs(x).max()) for x in
               list(dT.values()) + list(dV.values()))
    vmax = min(40.0, max(5.0, float(np.ceil(peak / 5.0) * 5.0)))
    images = np.empty((2, len(CONDS)), dtype=object)
    for r, (values, label) in enumerate(((dT, "ΔT"), (dV, "ΔV"))):
        for c, (cond, _) in enumerate(CONDS):
            images[r, c] = heat._render_one(verts_display, faces,
                                            values[cond], vmax,
                                            azim=-64.0, elev=8.0, size=480)
    _compose(images,
             ["ΔT = VT − V", "ΔV = VT − T"],
             [c for c, _ in CONDS],
             vmax,
             "FT (enum fine-tune): per-part modality contribution under 2D occlusion",
             out_dir / "Main_Fig_E6_FT_Contribution_SMPL.png",
             "ΔT = PA-MPJPE(VT) − PA-MPJPE(V); ΔV = PA-MPJPE(VT) − PA-MPJPE(T), mm; "
             "blue = modality helps, red = modality harms; joint errors smoothed by "
             "SMPL skinning weights; %d-fold LSO mean. Root velocity is in the table."
             % len(fold_rows))


def _teaser_session(fold: str, sid: str, model, cfg, split_csv: Path,
                    visual_root: Path, device: torch.device,
                    batch_size: int, out_dir: Path, picks_override=None):
    """单会话 6 快照的 ΔT 贡献 teaser（GT SMPL 姿态）。"""
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % sid))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, _ = load_contact(sid)
    motion = load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                       query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]))
    delta = {}
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=[sid],
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        full_v = np.full((len(ds.sessions[0]["V_feat"]), 24), np.nan, np.float32)
        full_t = np.full_like(full_v, np.nan)
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n_b = batch["V_feat"].shape[0]
            errs = {}
            for mode in ("V", "VT"):
                cid = torch.full((n_b,), CIDS[mode], dtype=torch.long,
                                 device=device)
                with torch.inference_mode():
                    out = model(batch["V_feat"], batch["T_raw"],
                                batch["T_phys"], cid, T_s2m=batch["T_s2m"])
                pj, _ = _per_joint_pa(out, batch)
                # (b, tw, 24): keep each window's own per-frame errors.
                errs[mode] = pj.cpu().numpy()
            tw_cur = int(cfg["tw"])
            for i, left in enumerate(batch["frame_start"].tolist()):
                full_v[left:left + tw_cur] = errs["V"][i]
                full_t[left:left + tw_cur] = errs["VT"][i]
        delta[cond] = (full_t - full_v)[:n]
    picks = np.flatnonzero(valid[:n] & np.isfinite(delta["clean"][:, 0]))
    if picks_override:
        cand = [int(x) for x in picks_override
                if 0 <= int(x) < n and valid[int(x)]
                and np.isfinite(delta["clean"][int(x), 0])]
        if len(cand) == 6:
            picks = np.asarray(sorted(cand))
        else:
            picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    else:
        picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]

    poses = np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(motion["trans_m"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)[:10]
    rest = heat._rest_joints({"betas": betas})
    _, faces, weights = heat._canonical_mesh()
    vmax = 0.0
    for cond, _ in CONDS:
        for f in picks:
            v = np.nanmean(delta[cond][max(0, f - 4):f + 5], axis=0)
            vmax = max(vmax, float(np.nanmax(np.abs(v))))
    vmax = min(40.0, max(5.0, float(np.ceil(vmax / 5.0) * 5.0)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Rectangle

    COND_LABEL = {"clean": "clean", "occ020": "+20% occ", "occ050": "+50% occ",
                  "occ090": "+90% occ"}
    # one variation axis only (tone, not weight): a plain grayscale ramp
    # reads as ordinal severity without the jumpy look of mixing bold+dark.
    COND_TONE = {"clean": "#8a8980", "occ020": "#5f5e57", "occ050": "#363530",
                 "occ090": "#0b0b0b"}
    RULE = "#aaa69b"
    CARD = "#f2f1ee"
    FOOT_COLOR = {"L": "#3f6b73", "R": "#b9893c"}
    SWING_FILL = "#e9e7e2"
    GUIDE = "#8a8980"

    fig = plt.figure(figsize=(16.8, 11.0), facecolor=SURFACE)
    gs = GridSpec(len(CONDS) + 1, len(picks), figure=fig,
                  height_ratios=[1.0] * len(CONDS) + [0.62],
                  hspace=0.08, wspace=0.03,
                  left=0.075, right=0.915, top=0.93, bottom=0.10)
    body_axes = np.empty((len(CONDS), len(picks)), dtype=object)
    for row, (cond, _) in enumerate(CONDS):
        for col, f in enumerate(picks):
            f = int(f)
            per_joint = np.nanmean(delta[cond][max(0, f - 4):f + 5], axis=0)
            vertex_values = heat._vertex_values(per_joint, weights)
            verts = heat.smpl_vertices(poses[f:f + 1], trans[f:f + 1],
                                       betas, rest)[0]
            verts_display = heat.smpl_yup_to_display(verts)
            rot = SciRotation.from_rotvec(poses[f, 0]).as_matrix()
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            disp = np.array([fwd[0], -fwd[2], fwd[1]])
            azim = float(np.degrees(np.arctan2(disp[1], disp[0])) + 20.0)
            colors = heat._colormap()(Normalize(-vmax, vmax, clip=True)(
                vertex_values))[:, :3]
            image = heat.render_mesh_frame(
                verts_display, faces, vertex_colors=colors, ambient=0.85,
                light_intensity=0.8, bg_color=SURFACE, size=(380, 380),
                elev=8.0, azim=azim)
            image = heat._crop_body(image)
            ax = fig.add_subplot(gs[row, col])
            body_axes[row, col] = ax
            # light rounded "card" behind the (narrow, portrait) body crop:
            # turns the unavoidable side margin into deliberate framing
            # instead of dead space, and keeps the render undistorted.
            ax.add_patch(FancyBboxPatch(
                (0.03, 0.03), 0.94, 0.94, transform=ax.transAxes,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=CARD, edgecolor="none", zorder=0))
            ax.imshow(image, zorder=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.text(0.5, 1.07, "t = %.1f s" % t[f], transform=ax.transAxes,
                        fontsize=9.8, color=INK, ha="center", va="bottom",
                        fontweight="medium")
            if col == 0:
                ax.text(-0.14, 0.5, COND_LABEL[cond], transform=ax.transAxes,
                        fontsize=12.5, color=COND_TONE[cond], rotation=90,
                        ha="center", va="center", fontweight="medium")

    # booktabs-style rules framing the condition-row table: a heavier rule
    # top/bottom, plain hairlines between rows — standard, not decorative.
    left_edge = body_axes[0, 0].get_position().x0 - 0.04
    right_edge = body_axes[0, -1].get_position().x1
    top_edge = body_axes[0, 0].get_position().y1 + 0.012
    bottom_edge = body_axes[-1, 0].get_position().y0 - 0.012
    for y, lw in ((top_edge, 1.3), (bottom_edge, 1.3)):
        fig.add_artist(Line2D([left_edge, right_edge], [y, y],
                              transform=fig.transFigure, color=RULE, lw=lw,
                              zorder=0))
    for row in range(1, len(CONDS)):
        y = (body_axes[row - 1, 0].get_position().y0
             + body_axes[row, 0].get_position().y1) / 2.0
        fig.add_artist(Line2D([left_edge, right_edge], [y, y],
                              transform=fig.transFigure, color=RULE, lw=0.8,
                              zorder=0))

    step = (t[n - 1] - t[0]) / max(len(picks) - 1, 1)

    axc = fig.add_subplot(gs[-1, :])
    track_y = {"L": 1.5, "R": 0.5}
    bar_half = 0.34
    for col_name, foot in (("L", 0), ("R", 1)):
        y = track_y[col_name]
        mask = contact[:n, foot] > 0.5
        # one continuous two-tone strip for the whole recording (stance vs
        # swing fill, same height throughout) instead of blocks floating on
        # a thin baseline — no separate "skewer" line underneath.
        bounds = np.flatnonzero(np.diff(mask.astype(np.int8)))
        bounds = np.concatenate(([0], bounds + 1, [n]))
        xranges = [(t[s], (t[min(e, n - 1)] - t[s]) or step * 0.5)
                   for s, e in zip(bounds[:-1], bounds[1:])]
        colors = [FOOT_COLOR[col_name] if mask[s] else SWING_FILL
                 for s, e in zip(bounds[:-1], bounds[1:])]
        axc.broken_barh(xranges, (y - bar_half, 2 * bar_half),
                        facecolors=colors, edgecolor="none", zorder=3)
    axc.set_xlim(t[0] - step / 2, t[n - 1] + step / 2)
    axc.set_ylim(0, 2)
    axc.set_yticks([track_y["R"], track_y["L"]])
    axc.set_yticklabels(["R", "L"], fontsize=9.0, fontweight="semibold")
    for tick, foot in zip(axc.get_yticklabels(), ["R", "L"]):
        tick.set_color(FOOT_COLOR[foot])
    axc.set_xticks([t[f] for f in picks])
    axc.set_xticklabels(["%.1f" % t[f] for f in picks], fontsize=8.6,
                        color=MUTED)
    for f in picks:
        axc.axvline(t[f], color=GUIDE, lw=0.8, ls=(0, (1, 1.6)), zorder=0)
    axc.tick_params(length=2, pad=3)
    for spine in axc.spines.values():
        spine.set_visible(False)
    axc.set_title("ground contact", loc="left", fontsize=9.2, color=MUTED,
                 pad=4)
    axc.set_xlabel("time (s)", fontsize=9.5, color=MUTED, labelpad=4)

    # ΔT trend overlaid on the contact bars: whole-body mean ΔT per
    # condition, on its own (twin) scale so it reads independently of the
    # L/R bar lanes beneath it.
    CURVE_COLOR = {"clean": "#c3cacc", "occ020": "#8ba0a8", "occ050": "#4c6b74",
                   "occ090": "#17262a"}
    ax2 = axc.twinx()
    curves, curve_vmax = {}, 0.0
    for cond, _ in CONDS:
        c = np.nanmean(delta[cond], axis=1)
        smooth = np.convolve(np.nan_to_num(c), np.ones(9) / 9.0, mode="same")
        curves[cond] = smooth
        finite = smooth[np.isfinite(c)]
        if len(finite):
            curve_vmax = max(curve_vmax, float(np.nanmax(np.abs(finite))))
    curve_vmax = min(60.0, max(10.0, float(np.ceil(curve_vmax / 5.0) * 5.0)))
    for cond, _ in CONDS:
        ax2.plot(t, curves[cond], color=CURVE_COLOR[cond], lw=1.5, alpha=0.95,
                 zorder=4, label=COND_LABEL[cond])
    ax2.axhline(0.0, color=MUTED, lw=0.6, ls="--", zorder=2)
    ax2.set_xlim(axc.get_xlim())
    ax2.set_ylim(-curve_vmax, curve_vmax)
    ax2.set_ylabel("ΔT (mm)", fontsize=8.6, color=MUTED, labelpad=6)
    ax2.tick_params(labelsize=8.0, colors=MUTED, length=2)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=4,
              fontsize=7.6, frameon=False, handlelength=1.3,
              columnspacing=0.9, borderaxespad=0.0)

    # dotted connectors tying each snapshot column to its instant on the
    # ground-contact timeline, spanning the gap below the body grid.
    axc_pos = axc.get_position()
    for col in range(len(picks)):
        pos = body_axes[-1, col].get_position()
        xc = (pos.x0 + pos.x1) / 2.0
        fig.add_artist(Line2D([xc, xc], [pos.y0, axc_pos.y1],
                              transform=fig.transFigure, color=GUIDE,
                              lw=0.8, ls=(0, (1, 1.6)), zorder=0))

    cax = fig.add_axes([0.93, 0.27, 0.013, 0.50])
    sm = ScalarMappable(norm=Normalize(-vmax, vmax), cmap=heat._colormap())
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    cb.ax.text(0.5, 1.03, "T harms", transform=cb.ax.transAxes, fontsize=8.0,
              color=MUTED, ha="center", va="bottom")
    cb.ax.text(0.5, -0.03, "T helps", transform=cb.ax.transAxes, fontsize=8.0,
              color=MUTED, ha="center", va="top")
    cb.set_label("ΔT = PA-MPJPE(VT) − PA-MPJPE(V)  (mm)", fontsize=8.8,
                color=MUTED, labelpad=10)

    # inline swatch legend for the ground-contact panel.
    legend_y = 0.978
    fig.add_artist(Rectangle((0.595, legend_y - 0.007), 0.016, 0.014,
                             transform=fig.transFigure, facecolor="#777d80",
                             edgecolor="none"))
    fig.text(0.615, legend_y, "stance (foot down)", transform=fig.transFigure,
             fontsize=8.6, color=MUTED, ha="left", va="center")
    fig.add_artist(Rectangle((0.760, legend_y - 0.007), 0.016, 0.014,
                             transform=fig.transFigure, facecolor=SWING_FILL,
                             edgecolor="none"))
    fig.text(0.782, legend_y, "swing (foot lifted)", transform=fig.transFigure,
             fontsize=8.6, color=MUTED, ha="left", va="center")
    path = out_dir.parents[1] / ("Main_Fig_E6_FT_Contribution_Teaser_%s.png"
                             % sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


# clean/occ020/occ050/occ090 (not occ100): the four raw-element rows used
# for manual figure compositing, per 2026-11-16 ruling.
RAW_CONDS = (
    ("clean", "cam3"),
    ("occ020", "cam3_occ_lower020"),
    ("occ050", "cam3_occ_lower050"),
    ("occ090", "cam3_occ_lower090"),
)


def _mainfig_raw(sid: str, model, cfg, split_csv: Path, visual_root: Path,
                 device: torch.device, batch_size: int, out_dir: Path,
                 picks_override=None):
    """Bare, unstyled raw elements for manual compositing: no cards, rules,
    axes, legends or captions — just the SMPL renders and the contact bars.
    """
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % sid))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, _ = load_contact(sid)
    motion = load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                       query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]))
    delta = {}
    for cond, cache in RAW_CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=[sid],
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        full_v = np.full((len(ds.sessions[0]["V_feat"]), 24), np.nan, np.float32)
        full_t = np.full_like(full_v, np.nan)
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            n_b = batch["V_feat"].shape[0]
            tw_cur = int(cfg["tw"])
            for mode, dest in (("V", full_v), ("VT", full_t)):
                cid = torch.full((n_b,), CIDS[mode], dtype=torch.long,
                                 device=device)
                with torch.inference_mode():
                    out = model(batch["V_feat"], batch["T_raw"],
                                batch["T_phys"], cid, T_s2m=batch["T_s2m"])
                pj, _ = _per_joint_pa(out, batch)
                pj = pj.cpu().numpy()
                for i, left in enumerate(batch["frame_start"].tolist()):
                    dest[left:left + tw_cur] = pj[i]
        delta[cond] = (full_t - full_v)[:n]

    picks = np.flatnonzero(valid[:n] & np.isfinite(delta["clean"][:, 0]))
    if picks_override:
        cand = [int(x) for x in picks_override
                if 0 <= int(x) < n and valid[int(x)]
                and np.isfinite(delta["clean"][int(x), 0])]
        if len(cand) == 6:
            picks = np.asarray(sorted(cand))
        else:
            picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]
    else:
        picks = picks[np.round(np.linspace(0, len(picks) - 1, 6)).astype(int)]

    poses = np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(motion["trans_m"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)[:10]
    rest = heat._rest_joints({"betas": betas})
    _, faces, weights = heat._canonical_mesh()
    vmax = 0.0
    for cond, _ in RAW_CONDS:
        for f in picks:
            v = np.nanmean(delta[cond][max(0, int(f) - 4):int(f) + 5], axis=0)
            vmax = max(vmax, float(np.nanmax(np.abs(v))))
    vmax = min(40.0, max(5.0, float(np.ceil(vmax / 5.0) * 5.0)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    out_dir.mkdir(parents=True, exist_ok=True)

    # element 1: bare 4x6 SMPL grid — bodies only, no cards/rules/labels.
    fig = plt.figure(figsize=(2.0 * len(picks), 2.6 * len(RAW_CONDS)),
                     facecolor=SURFACE)
    gs = GridSpec(len(RAW_CONDS), len(picks), figure=fig,
                  hspace=0.0, wspace=0.0, left=0.0, right=1.0, top=1.0,
                  bottom=0.0)
    for row, (cond, _) in enumerate(RAW_CONDS):
        for col, f in enumerate(picks):
            f = int(f)
            per_joint = np.nanmean(delta[cond][max(0, f - 4):f + 5], axis=0)
            vertex_values = heat._vertex_values(per_joint, weights)
            verts = heat.smpl_vertices(poses[f:f + 1], trans[f:f + 1],
                                       betas, rest)[0]
            verts_display = heat.smpl_yup_to_display(verts)
            rot = SciRotation.from_rotvec(poses[f, 0]).as_matrix()
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            disp = np.array([fwd[0], -fwd[2], fwd[1]])
            azim = float(np.degrees(np.arctan2(disp[1], disp[0])) + 20.0)
            colors = heat._colormap()(Normalize(-vmax, vmax, clip=True)(
                vertex_values))[:, :3]
            image = heat.render_mesh_frame(
                verts_display, faces, vertex_colors=colors, ambient=0.85,
                light_intensity=0.8, bg_color=SURFACE, size=(380, 380),
                elev=8.0, azim=azim)
            image = heat._crop_body(image)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image)
            ax.axis("off")
    body_path = out_dir / ("Fig_SMPL_Grid_%s.png" % sid)
    fig.savefig(body_path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % body_path)

    # element 2: bare L/R ground-contact strips — no ticks/labels/legend.
    FOOT_COLOR = {"L": "#3f6b73", "R": "#b9893c"}
    SWING_FILL = "#e9e7e2"
    fig2, ax2 = plt.subplots(figsize=(12.0, 1.6), facecolor=SURFACE)
    track_y = {"L": 1.5, "R": 0.5}
    bar_half = 0.42
    for col_name, foot in (("L", 0), ("R", 1)):
        y = track_y[col_name]
        mask = contact[:n, foot] > 0.5
        bounds = np.flatnonzero(np.diff(mask.astype(np.int8)))
        bounds = np.concatenate(([0], bounds + 1, [n]))
        xranges = [(t[s], (t[min(e, n - 1)] - t[s]) or 0.01)
                  for s, e in zip(bounds[:-1], bounds[1:])]
        colors = [FOOT_COLOR[col_name] if mask[s] else SWING_FILL
                 for s, e in zip(bounds[:-1], bounds[1:])]
        ax2.broken_barh(xranges, (y - bar_half, 2 * bar_half),
                        facecolors=colors, edgecolor="none")
    ax2.set_xlim(t[0], t[n - 1])
    ax2.set_ylim(0, 2)
    ax2.axis("off")
    contact_path = out_dir / ("Fig_Contact_%s.png" % sid)
    fig2.savefig(contact_path, dpi=220, facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig2)
    print("[png] %s" % contact_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-template", default=str(
        DIAG / "router_ft_only_{fold}_v2rerun.pt"))
    ap.add_argument("--tag", default="FT_v2")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--teaser", action="store_true")
    ap.add_argument("--teaser-session", default="S11051")
    ap.add_argument("--teaser-fold", default="fS5S6S11")
    ap.add_argument("--teaser-picks", default="37,96,138,213,263,316",
                    help="comma-separated frame indices for the six snapshots; "
                         "default = the highest-score frames recorded in "
                         "E6_TeaserSelection_S11051.json for this session")
    ap.add_argument("--teaser-only", action="store_true",
                    help="skip the (expensive) full-fold table/heatmap pass; "
                         "only (re)render --teaser, for fast style iteration")
    ap.add_argument("--mainfig", action="store_true",
                    help="emit bare, unstyled raw elements (SMPL grid + "
                         "contact strips, no cards/rules/labels/captions) "
                         "for manual compositing, skipping the full-fold "
                         "pass; writes into --mainfig-out-dir")
    ap.add_argument("--mainfig-out-dir", type=Path, default=(
        REPO / "results_display" / "Etest_crossfit_physimu_vgate"
        / "E6_ContactPhaseAnalysis" / "MainFig"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    if args.mainfig:
        fold = args.teaser_fold
        ckpt = Path(args.checkpoint_template.format(fold=fold))
        model, cfg, _ = _load_model(ckpt)
        model.to(device).eval()
        picks = [int(x) for x in args.teaser_picks.split(",") if x.strip()]
        _mainfig_raw(args.teaser_session, model, cfg, SPLITS / FOLDS[fold],
                    args.visual_root, device, args.batch_size,
                    args.mainfig_out_dir, picks)
        print("done", args.tag)
        return 0
    if args.teaser_only:
        if not args.teaser:
            raise SystemExit("--teaser-only requires --teaser")
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fold = args.teaser_fold
        ckpt = Path(args.checkpoint_template.format(fold=fold))
        model, cfg, _ = _load_model(ckpt)
        model.to(device).eval()
        picks = [int(x) for x in args.teaser_picks.split(",") if x.strip()]
        _teaser_session(fold, args.teaser_session, model, cfg,
                        SPLITS / FOLDS[fold], args.visual_root, device,
                        args.batch_size, out_dir, picks)
        print("done", args.tag)
        return 0
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    fold_rows = {}
    for fold in folds:
        ckpt = Path(args.checkpoint_template.format(fold=fold))
        model, cfg, _ = _load_model(ckpt)
        model.to(device).eval()
        split_csv = SPLITS / FOLDS[fold]
        ids = load_split_ids(split_csv, "test")
        stats = {p: {c: {m: [0.0, 0] for m in MODES} for c, _ in CONDS}
                 for p in PARTS}
        per_joint_stats = {c: {m: [] for m in MODES} for c, _ in CONDS}
        _run_fold(model, cfg, split_csv, ids, args.visual_root, device,
                  args.batch_size, args.limit_sessions, stats, per_joint_stats)
        parts = {p: {c: {m: _means(stats, p, c, m) for m in MODES}
                     for c, _ in CONDS} for p in PARTS}
        # 逐关节四折前的每折均值：先按模式聚合，再在折内做差值。
        per_joint = {c: {"T": None, "V": None} for c, _ in CONDS}
        for cond, _ in CONDS:
            mode_mean = {}
            for mode in MODES:
                sums = np.zeros(24)
                count = 0
                for arr, n in per_joint_stats[cond][mode]:
                    sums += arr * n
                    count += n
                mode_mean[mode] = sums / max(count, 1)
            per_joint[cond]["T"] = mode_mean["VT"] - mode_mean["V"]
            per_joint[cond]["V"] = mode_mean["VT"] - mode_mean["T"]
        fold_rows[fold] = {"parts": parts, "per_joint": per_joint}
        print("[fold] %s done" % fold, flush=True)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(out_dir, fold_rows)
    if not args.skip_render:
        _render_main(out_dir, fold_rows)
    if args.teaser:
        fold = args.teaser_fold
        ckpt = Path(args.checkpoint_template.format(fold=fold))
        model, cfg, _ = _load_model(ckpt)
        model.to(device).eval()
        picks = [int(x) for x in args.teaser_picks.split(",") if x.strip()]
        _teaser_session(fold, args.teaser_session, model, cfg,
                        SPLITS / FOLDS[fold], args.visual_root, device,
                        args.batch_size, out_dir, picks)
    print("done", args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
