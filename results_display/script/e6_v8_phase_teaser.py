#!/usr/bin/env python3
"""E6 V8 四折 LSO：单动作「相位 teaser + 曲线」图（2026-11-09）。

上面是四帧 SMPL 身体增益 teaser，两行 = V8_Cv5（双门）/ V8_Cv7（无门）；
每具身体使用**该模型自己 VT 推理出来的 SMPL pose**，颜色 = 该帧短窗内的
逐关节 PA-MPJPE Δ = VT − V（蓝 = 触觉增益，红 = 伤害），经 skinning
weights 铺到该 pose 的 mesh。四相位齐全时每相位取门控对比最大的帧；
短动作/单相位动作用等距帧补齐到 4 列，保证所有动作的 teaser 图同规格。
下面是同一会话的逐帧 Δ 曲线，竖虚线标出四个 teaser 时刻；曲线上下两
panel 分别是左/右脚相位带。

teaser 帧的选择：左脚相位状态机里，每个相位取「Cv7 Δ − Cv5 Δ」最大的帧
（距相位边界 ≥3 帧），即门控对比最清楚的时刻；这是 teaser，不是无偏
相位平均。该时刻的 heatmap 值 = 同相位 ±5 帧（40 Hz，±0.125 s）的
逐关节 Δ 平均，避免单帧噪声。

用法（touch_gait env，仓库根）：
  python results_display/script/e6_v8_phase_teaser.py --action run
  python results_display/script/e6_v8_phase_teaser.py --action all
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import e5_v8_smpl_gain_heatmap as heat  # noqa: E402
from anysole.utils.metrics import similarity_align  # noqa: E402
from anysole_tta.phase_common import (CELL_ROOT, load_contact,  # noqa: E402
                                      load_gt_joints, load_pred_joints,
                                      metric_error, phase_labels, PHASE_NAMES)

BASE = REPO / "results_display" / "Etest_crossfit_v8" / "E6_ContactPhaseAnalysis" / "pa_mpjpe"
PICKS = BASE / "detail" / "E6_BestSessions_V8.csv"
OUT = BASE / "teaser"
MODELS = ("V8_Cv5", "V8_Cv7")
MODEL_LABEL = {"V8_Cv5": "V8_Cv5 (double gate)", "V8_Cv7": "V8_Cv7 (no gate)"}
MODEL_COLOR = {"V8_Cv5": "#2a78d6", "V8_Cv7": "#52514e"}
ACTION_TITLE = {"walk": "Walk", "run": "Run", "squat": "Squat",
                "stand_shift": "Stand-shift", "single_stand": "Single-stand",
                "jump_jack": "Jump-jack", "lateral": "Lateral",
                "sit_stand": "Sit-stand", "stairs": "Stairs"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
FPS = 40.0
HALF_WINDOW = 5


def _read_picks() -> dict:
    if not PICKS.is_file():
        raise SystemExit("missing %s; run e6_v8_best_sessions.py first" % PICKS)
    with PICKS.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {row["动作"]: {"sid": row["会话"], "fold": row["折"]}
            for row in rows}


def _load(sid: str, fold: str) -> dict:
    gt = load_gt_joints(sid)
    contact, _, contact_valid = load_contact(sid)
    data = {"gt": gt, "contact": contact,
            "valid": np.asarray(contact_valid).astype(bool),
            "pred": {}, "err": {}, "delta": None, "pose": {}}
    n = min(len(gt), len(contact), len(data["valid"]))
    for model in MODELS:
        cell = "%s_%s_s1" % (model, fold)
        pv, _, vv = load_pred_joints(cell, sid, "V2M")
        pt, _, vt = load_pred_joints(cell, sid, "VT2M")
        npz = np.load(CELL_ROOT / cell / "predictions/eval_motion"
                      / ("%s_VT2M.npz" % sid), allow_pickle=True)
        poses = np.asarray(npz["poses"], dtype=np.float64).reshape(-1, 24, 3)
        trans = np.asarray(npz["pred_pelvis_trans"], dtype=np.float64)
        betas = np.asarray(npz["betas"], dtype=np.float64).reshape(-1)[:10]
        n = min(n, len(pv), len(pt), len(poses), len(trans))
        data["pred"][model] = (pv, pt, np.asarray(vv).astype(bool)
                               & np.asarray(vt).astype(bool))
        data["pose"][model] = (poses, trans, betas)
    data["gt"] = gt[:n]
    data["contact"] = contact[:n]
    data["valid"] = data["valid"][:n]
    for model in data["pred"]:
        poses, trans, betas = data["pose"][model]
        data["pose"][model] = (poses[:n], trans[:n], betas)
    for model, (pv, pt, mask) in data["pred"].items():
        pv, pt = pv[:n], pt[:n]
        mask = mask[:n] & data["valid"]
        err_v = metric_error(pv, data["gt"], "pa_mpjpe", "full")
        err_t = metric_error(pt, data["gt"], "pa_mpjpe", "full")
        data["pred"][model] = (pv, pt, mask)
        data["err"][model] = (err_v, err_t, mask & np.isfinite(err_v) & np.isfinite(err_t))
    data["delta"] = {model: data["err"][model][1] - data["err"][model][0]
                     for model in MODELS}
    data["phase"] = phase_labels(data["contact"])
    return data


def _phase_frames(data: dict) -> dict:
    """Per left-foot phase: the frame with the largest gate contrast.

    Contrast = Δ_VT−V(Cv7) − Δ_VT−V(Cv5), i.e. the moment where the no-gate
    model is most clearly worse than the double-gate model.  Frames within
    3 samples of a phase boundary are excluded so the annotated phase is
    unambiguous; this is a teaser, not an unbiased phase average.
    """
    phase = data["phase"][:, 0]
    ok = np.ones(len(phase), dtype=bool)
    for model in MODELS:
        ok &= data["err"][model][2]
    contrast = data["delta"]["V8_Cv7"] - data["delta"]["V8_Cv5"]
    out = {}
    for k in range(4):
        sel = (phase == k) & ok & np.isfinite(contrast)
        if not sel.any():
            continue
        boundaries = np.flatnonzero(np.diff(np.r_[False, sel, False].astype(np.int8)))
        runs = list(zip(boundaries[::2], boundaries[1::2]))
        best: Optional[tuple] = None
        for start, stop in runs:
            lo = start + 3 if stop - start > 6 else start
            hi = stop - 3 if stop - start > 6 else stop
            for frame in range(lo, hi):
                score = float(contrast[frame])
                if best is None or score > best[0]:
                    best = (score, frame)
        if best is not None:
            out[k] = int(best[1])
    return out


def _frame_delta(data: dict, model: str, center: int, phase_value: int) -> np.ndarray:
    """Average per-joint Δ over ±HALF_WINDOW frames inside the same phase."""
    lo = max(center - HALF_WINDOW, 0)
    hi = min(center + HALF_WINDOW + 1, len(data["gt"]))
    pv, pt, mask = data["pred"][model]
    phase = data["phase"][:, 0]
    total = np.zeros(24, dtype=np.float64)
    count = 0
    for t in range(lo, hi):
        if not mask[t] or phase[t] != phase_value:
            continue
        gt = data["gt"][t][None]
        ev = np.linalg.norm(similarity_align(pv[t][None], gt)[0] - gt[0], axis=-1) * 1000.0
        et = np.linalg.norm(similarity_align(pt[t][None], gt)[0] - gt[0], axis=-1) * 1000.0
        total += et - ev
        count += 1
    if count == 0:
        return np.zeros(24, dtype=np.float64)
    return total / count


def _pose_vertices(data: dict, model: str, frame: int) -> np.ndarray:
    """VT-predicted SMPL mesh of this model at ``frame`` (display z-up)."""
    poses, trans, betas = data["pose"][model]
    rest = heat._rest_joints({"betas": betas})
    verts = heat.smpl_vertices(poses[frame:frame + 1], trans[frame:frame + 1],
                               betas, rest)[0]
    return heat.smpl_yup_to_display(verts)


def _column_azim(data: dict, frame: int, offset: float = 20.0) -> float:
    """Camera azimuth in front of the action, shared by both model rows.

    The body's native forward axis is +Z; the root rotation at the frame maps
    it to the display frame.  Placing the camera along that forward vector
    (plus ``offset`` for a 3/4 view) keeps the VT-inferred action facing the
    viewer instead of rendering a random back.
    """
    azimuths = []
    for model in MODELS:
        poses = data["pose"][model][0]
        rot = SciRotation.from_rotvec(poses[frame, 0]).as_matrix()
        forward = rot @ np.array([0.0, 0.0, 1.0])
        display = np.array([forward[0], -forward[2], forward[1]])
        azimuths.append(np.degrees(np.arctan2(display[1], display[0])))
    # Circular mean: the two models should look roughly the same way.
    mean = np.angle(np.mean(np.exp(1j * np.deg2rad(azimuths))))
    return float(np.rad2deg(mean) + offset)


def _curve(data: dict) -> tuple:
    t = np.arange(len(data["gt"])) / FPS
    curves = {}
    for model in MODELS:
        delta = data["delta"][model].copy()
        delta[~data["err"][model][2]] = np.nan
        curves[model] = delta
    return t, curves


def _snapshots(data: dict, frames: dict, count: int = 4) -> list:
    """Exactly ``count`` (phase, frame) columns, same spec for every action.

    Actions with four contact phases use one max-contrast frame each.  Short /
    single-phase actions are filled with evenly spaced valid frames so they
    still get a four-column teaser instead of a single lonely body.
    """
    entries = [(k, frames[k]) for k in range(4) if k in frames]
    if len(entries) >= count:
        return entries[:count]
    ok = np.ones(len(data["gt"]), dtype=bool)
    for model in MODELS:
        ok &= data["err"][model][2]
    candidates = np.flatnonzero(ok)
    used = {frame for _, frame in entries}
    if len(candidates):
        for frac in np.linspace(0.0, 1.0, count + 2)[1:-1]:
            frame = int(candidates[int(round(frac * (len(candidates) - 1)))])
            if frame in used:
                continue
            entries.append((int(data["phase"][frame, 0]), frame))
            used.add(frame)
            if len(entries) >= count:
                break
    if len(entries) < count:
        for frame in candidates:
            frame = int(frame)
            if frame in used:
                continue
            entries.append((int(data["phase"][frame, 0]), frame))
            used.add(frame)
            if len(entries) >= count:
                break
    return entries[:count]


def _render_teaser(action: str, sid: str, fold: str, data: dict,
                   frames: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _, faces, weights = heat._canonical_mesh()
    entries = _snapshots(data, frames, count=4)
    if not entries:
        raise SystemExit("%s %s: no complete phase segment" % (action, sid))

    t, curves = _curve(data)
    values = {(model, col): heat._vertex_values(
        _frame_delta(data, model, center, phase_id), weights)
        for model in MODELS for col, (phase_id, center) in enumerate(entries)}
    # One symmetric scale per column: the two models in a column stay
    # directly comparable, while a violent phase cannot wash out the rest.
    vmax = {}
    for col, (phase_id, center) in enumerate(entries):
        # 75th percentile, capped: the hand/foot vertices can carry a single
        # 100+ mm outlier at the contrast frame, which must not wash out the
        # torso/leg structure.  Values beyond the column scale are clipped.
        peak = max(float(np.percentile(np.abs(values[(model, col)]), 75.0))
                   for model in MODELS)
        vmax[col] = min(40.0, max(15.0, float(np.ceil(peak / 5.0) * 5.0)))
    fig = plt.figure(figsize=(18.0, 10.2), facecolor=SURFACE)
    gs = GridSpec(4, len(entries), figure=fig,
                  height_ratios=[1.0, 1.0, 1.25, 1.25],
                  hspace=0.18, wspace=0.04,
                  left=0.06, right=0.98, top=0.90, bottom=0.08)

    for row, model in enumerate(MODELS):
        for col, (phase_id, center) in enumerate(entries):
            ax = fig.add_subplot(gs[row, col])
            verts_display = _pose_vertices(data, model, center)
            image = heat._render_one(verts_display, faces,
                                     values[(model, col)],
                                     vmax[col],
                                     azim=_column_azim(data, center),
                                     elev=8.0, size=360)
            ax.imshow(image)
            ax.axis("off")
            if row == 0:
                ax.set_title("%s  (t=%.2fs, ±%.0f mm)"
                             % (PHASE_NAMES[phase_id], t[center], vmax[col]),
                             fontsize=10.0, color=INK, pad=5)
            if col == 0:
                ax.text(-0.04, 0.5, MODEL_LABEL[model], transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10.5,
                        color=INK)

    phase_colors = {0: "#d9d9d9", 1: "#ffffff", 2: "#f4a582", 3: "#92c5de"}
    axes = []
    for panel, foot_i in ((2, 0), (3, 1)):
        ax = fig.add_subplot(gs[panel, :])
        axes.append(ax)
        ph = data["phase"][:, foot_i]
        ax.set_xlim(0, t[-1])
        for k in range(4):
            sel = ph == k
            if sel.any():
                bounds = np.flatnonzero(np.diff(np.r_[False, sel, False].astype(np.int8)))
                for number, (start, stop) in enumerate(zip(bounds[::2], bounds[1::2])):
                    ax.axvspan(t[start], t[stop - 1] + 1.0 / FPS,
                               ymin=0.0, ymax=0.08, color=phase_colors[k],
                               alpha=0.6, lw=0,
                               label=PHASE_NAMES[k] if number == 0 else None)
        for model in MODELS:
            delta = curves[model]
            smooth = np.convolve(np.nan_to_num(delta), np.ones(20) / 20.0, mode="same")
            ax.plot(t, delta, color=MODEL_COLOR[model], alpha=0.22, lw=0.8)
            ax.plot(t, smooth, color=MODEL_COLOR[model], lw=1.8,
                    label=MODEL_LABEL[model] if foot_i == 0 else None)
        for phase_id, center in entries:
            ax.axvline(t[center], color="#8a8980", lw=0.9, ls=":")
        ax.axhline(0, color=INK, lw=0.7, ls="--")
        ax.set_ylim(-30, 30)
        ax.set_ylabel("Δ = VT − V (mm)", fontsize=9.5, color=MUTED)
        ax.set_title("full body — %s-foot phases" % ("left" if foot_i == 0 else "right"),
                     fontsize=10, color=INK, loc="left")
        ax.tick_params(labelsize=8.5, colors=MUTED)
        ax.grid(alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("time (s)", fontsize=9.5, color=MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("%s — %s — %s: tactile body-part effect at four snapshots"
                 % (ACTION_TITLE.get(action, action), sid, fold),
                 fontsize=12.0, color=INK, y=0.985)
    fig.text(0.5, 0.945,
             "Bodies = each model's own VT-predicted SMPL pose at the frame; "
             "colour = per-joint PA-MPJPE Δ averaged over ±0.125 s. Blue = tactile "
             "benefit, red = tactile harm. Phase frames use the largest Cv7−Cv5 "
             "contrast; short actions are filled with evenly spaced frames. "
             "Each column has its own 75th-percentile scale capped at ±40 mm "
             "(shown in the column title).",
             fontsize=8.5, color=MUTED, ha="center")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("Full_Fig_E6_PhaseTeaser_%s.png" % action)
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % path)
    return path


def _render_single_teaser(action: str, sid: str, fold: str, data: dict,
                          frames: dict) -> Path:
    """One striking moment: V5 vs V7 bodies above, the motion curve below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    if not frames:
        raise SystemExit("%s %s: no complete phase segment" % (action, sid))
    from anysole.types import JOINT_HIERARCHY
    upper = list(JOINT_HIERARCHY["upper"])
    d7 = data["delta"]["V8_Cv7"]
    d5 = data["delta"]["V8_Cv5"]

    def score(phase_id: int) -> tuple:
        d7_joint = _frame_delta(data, "V8_Cv7", frames[phase_id], phase_id)
        d5_joint = _frame_delta(data, "V8_Cv5", frames[phase_id], phase_id)
        # Widespread upper-body gate benefit, with each joint capped at 10 mm:
        # a single wrist outlier must not outvote several flipped joints.
        upper_gain = float(np.clip(d7_joint[upper] - d5_joint[upper], 0, 10).sum())
        return upper_gain, float(d7[frames[phase_id]] - d5[frames[phase_id]])

    # Prefer the phase with the most widespread upper-body gate flip
    # (the "T hurts V's upper body" story), then the largest mean contrast.
    phase_id = max(frames, key=score)
    center = frames[phase_id]

    _, faces, weights = heat._canonical_mesh()
    values = {model: heat._vertex_values(
        _frame_delta(data, model, center, phase_id), weights)
        for model in MODELS}
    peak = max(float(np.percentile(np.abs(values[model]), 75.0))
               for model in MODELS)
    vmax = min(40.0, max(15.0, float(np.ceil(peak / 5.0) * 5.0)))

    t, curves = _curve(data)
    fig = plt.figure(figsize=(12.5, 9.2), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0],
                  hspace=0.16, wspace=0.04,
                  left=0.05, right=0.88, top=0.88, bottom=0.10)
    azim = _column_azim(data, center)
    for col, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, col])
        verts_display = _pose_vertices(data, model, center)
        ax.imshow(heat._render_one(verts_display, faces, values[model],
                                   vmax, azim=azim, elev=8.0, size=440))
        ax.axis("off")
        ax.set_title(MODEL_LABEL[model], fontsize=11, color=INK, pad=4)
    fig.suptitle("%s — %s — %s   |   %s (t=%.2f s)"
                 % (ACTION_TITLE.get(action, action), sid, fold,
                    PHASE_NAMES[phase_id], t[center]),
                 fontsize=12.5, color=INK, y=0.975)
    fig.text(0.5, 0.925,
             "Body = per-joint PA-MPJPE Δ = VT − V averaged over ±0.125 s; "
             "blue = tactile benefit, red = tactile harm; scale ±%.0f mm."
             % vmax, fontsize=8.8, color=MUTED, ha="center")
    cax = fig.add_axes([0.90, 0.52, 0.014, 0.32])
    sm = ScalarMappable(norm=Normalize(-vmax, vmax), cmap=heat._colormap())
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("ΔPA-MPJPE (mm)", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    phase_colors = {0: "#d9d9d9", 1: "#ffffff", 2: "#f4a582", 3: "#92c5de"}
    ph = data["phase"][:, 0]
    ax.set_xlim(0, t[-1])
    for k in range(4):
        sel = ph == k
        if sel.any():
            bounds = np.flatnonzero(np.diff(np.r_[False, sel, False].astype(np.int8)))
            for number, (start, stop) in enumerate(zip(bounds[::2], bounds[1::2])):
                ax.axvspan(t[start], t[stop - 1] + 1.0 / FPS,
                           ymin=0.0, ymax=0.08, color=phase_colors[k],
                           alpha=0.6, lw=0,
                           label=PHASE_NAMES[k] if number == 0 else None)
    for model in MODELS:
        delta = curves[model]
        smooth = np.convolve(np.nan_to_num(delta), np.ones(20) / 20.0, mode="same")
        ax.plot(t, delta, color=MODEL_COLOR[model], alpha=0.22, lw=0.8)
        ax.plot(t, smooth, color=MODEL_COLOR[model], lw=1.8,
                label=MODEL_LABEL[model])
    ax.axvline(t[center], color="#8a8980", lw=1.0, ls=":")
    ax.axhline(0, color=INK, lw=0.7, ls="--")
    ax.set_ylim(-30, 30)
    ax.set_xlabel("time (s)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Δ = VT − V (mm)", fontsize=9.5, color=MUTED)
    ax.set_title("full body — left-foot phases", fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = OUT / ("Main_Fig_E6_Teaser_%s.png" % action)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="run",
                        help="action name or 'all'")
    parser.add_argument("--single", action="store_true",
                        help="also render the one-moment V5-vs-V7 teaser")
    args = parser.parse_args()
    picks = _read_picks()
    actions = (list(picks) if args.action == "all" else [args.action])
    for action in actions:
        if action not in picks:
            raise SystemExit("no selected session for action %r" % action)
        sid, fold = picks[action]["sid"], picks[action]["fold"]
        data = _load(sid, fold)
        frames = _phase_frames(data)
        print("[teaser] %s %s %s phases=%s" % (action, sid, fold,
                                               sorted(frames)))
        _render_teaser(action, sid, fold, data, frames)
        if args.single:
            _render_single_teaser(action, sid, fold, data, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
