#!/usr/bin/env python3
"""E5 crossfit：V 门多指标 Absolute + Relative（V8 主基座口径）。

指标（用户 2026-11-08 选定，4 主 + 2 补充）：
  PA-MPJPE（全身体）、ankle_foot_mpjpe（骨盆对齐踝足 MPJPE）、
  GlobalMPJPE（GT Anchored）、FootSliding（GT Anchored）、
  GroundPen（GT Anchored）、traj_ATE。

数据源：
- 无 V 门 = V8_Cv5（=Cv5SD2；T 门开、V 门关）；
- 无 T 门 = V8_Cv7；消融 = V8_Cv4/Cv2/Cv6；
- V 门 = results/diagnostics/20261108_gate_level/report_<fold>.json 的
  (tau,width) 作用在 rows_<fold>_test.npz 的 s_ps 上，输出
  y=(1-g)*T2M+g*VT2M（用 V8_Cv5 的 eval_motion 两条 motion 融合）。
- V only = V8_Cv5 的 V2M；T only = V8_Cv5 的 T2M。
- eval_motion 的 joint_xyz_world 是 world_z_up；GT kp_gt 是 Y-up，
  统一转成 (x,z,-y)。

产物（results_display/Etest_crossfit/E5_OcclusionRobustness/）：
  Main_Fig6_Gate_Metrics_Absolute.png   2×3 六指标（V gate / V8 / V8_Cv7 + V/T refs）
  Main_Fig7_Gate_Metrics_Relative.png   2×3 六指标（Gate−V8 / Gate−Cv7 / V8−Cv7 + 消融）
  Main_Table6_Gate_Metrics_Absolute.csv
  Main_Table7_Gate_Metrics_Relative.csv
  Full_Table4_Gate_Metrics_PerFold.csv

用法：python results_display/script/gate_metric_crossfit_export.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display/script"))

from anysole.data.smpl_io import estimate_floor_y  # noqa: E402
from anysole.types import ANKLE_FOOT_JOINTS  # noqa: E402
from anysole.utils.metrics import (  # noqa: E402
    foot_sliding_joints, ground_penetration, mean_point_error, pa_mpjpe,
    pelvis_align, root_trajectory_metrics)

GATE_OUT = REPO / "results/diagnostics/20261108_gate_level"
CELLS = REPO / "results/cells"
OUT = REPO / "results_display/Etest_crossfit/E5_OcclusionRobustness"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")
METRICS = ("PA-MPJPE", "ankle_foot_mpjpe", "GlobalMPJPE",
           "FootSliding", "GroundPen", "traj_ATE")
METRIC_LABELS = {
    "PA-MPJPE": "PA-MPJPE (mm)",
    "ankle_foot_mpjpe": "ankle-foot MPJPE, pelvis-aligned (mm)",
    "GlobalMPJPE": "GlobalMPJPE, GT anchored (mm)",
    "FootSliding": "FootSliding, GT anchored (mm)",
    "GroundPen": "GroundPen, GT anchored (mm)",
    "traj_ATE": "traj_ATE (mm)",
}
GROUPS = ("Gate", "V8_Cv5", "V8_Cv7")
ABLATIONS = ("V8_Cv4", "V8_Cv2", "V8_Cv6")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
COLORS = {"Gate": "#2a78d6", "V8_Cv5": "#eb6834", "V8_Cv7": "#52514e",
          "V8_Cv4": "#1baf7a", "V8_Cv2": "#eda100", "V8_Cv6": "#7a5af8"}
LABELS = {"Gate": "V gate (T gate + V gate)",
          "V8_Cv5": "V8 / Cv5SD2 (T gate, no V gate)",
          "V8_Cv7": "V8 ablation: no gate",
          "V8_Cv4": "V8 ablation: single gate",
          "V8_Cv2": "V8 ablation: dropout",
          "V8_Cv6": "V8 ablation: always VT"}


def cell_dir(fam: str, fold: str, level: str) -> Path:
    base = "%s_%s_s1" % (fam, fold)
    if level != "000":
        base += "_evalocc%s" % level
    return CELLS / base


def load_motion(fam: str, fold: str, level: str, session: str, cond: str) -> dict | None:
    p = cell_dir(fam, fold, level) / "predictions" / "eval_motion" / ("%s_%s.npz" % (session, cond))
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=True)
    j = np.asarray(d["joint_xyz_world"], dtype=np.float64)
    # world_z_up -> Y-up (GT kp_gt 口径)
    j = np.stack([j[..., 0], j[..., 2], -j[..., 1]], axis=-1)
    return {"joint": j, "frame": np.asarray(d["frame_indices"]),
            "time": np.asarray(d["source_frame_times_s"], dtype=np.float64)}


def gate_from_s(s, tau, width):
    if width <= 1e-9:
        return (s >= tau).astype(np.float64)
    lo = tau - 0.5 * width
    hi = tau + 0.5 * width
    return np.clip((s - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def session_metrics(pred, gt, times, floor_y):
    pa = float(pa_mpjpe(pred, gt).mean())
    pa_p, _ = pelvis_align(pred)
    gt_p, _ = pelvis_align(gt)
    ankle = float(mean_point_error(pa_p[:, ANKLE_FOOT_JOINTS],
                                   gt_p[:, ANKLE_FOOT_JOINTS], scale=1000.0).mean())
    glob = float(mean_point_error(pred, gt, scale=1000.0).mean())
    fs, _ = foot_sliding_joints(pred[:, ANKLE_FOOT_JOINTS],
                                gt[:, ANKLE_FOOT_JOINTS], times)
    gp = float(ground_penetration(pred, floor_y)["ground_pen_mean_mm"])
    tr = float(root_trajectory_metrics(pred[:, 0, :], gt[:, 0, :])["root_ate_mm"])
    return {"PA-MPJPE": pa, "ankle_foot_mpjpe": ankle, "GlobalMPJPE": glob,
            "FootSliding": float(fs), "GroundPen": gp, "traj_ATE": tr}


CELL_KEYS = {
    "PA-MPJPE": "PA-MPJPE",
    "ankle_foot_mpjpe": "MPJPE_anklefoot",
    "GlobalMPJPE": "GlobalMPJPE（GT Anchored）",
    "FootSliding": "FootSliding（GT Anchored）",
    "GroundPen": "GroundPen（GT Anchored）",
    "traj_ATE": "traj_ATE",
}


def cell_values(fam: str, fold: str, level: str, cond: str) -> dict:
    p = cell_dir(fam, fold, level) / "metrics" / "test.json"
    d = json.loads(p.read_text(encoding="utf-8"))["metrics"][cond]
    return {m: float(d[k]) for m, k in CELL_KEYS.items()}


def collect_fold_level(fold: str, level: str) -> dict:
    """Gate = eval_motion 融合后的 per-session 六指标；V8/消融/T/V 直接读 cells。"""
    rep = json.loads((GATE_OUT / ("report_%s.json" % fold)).read_text(encoding="utf-8"))
    tau, width = float(rep["tau"]), float(rep["width"])
    rows = np.load(GATE_OUT / ("rows_%s_test.npz" % fold), allow_pickle=True)
    lev = rows["level"]
    ses = rows["session"]
    st = rows["start"]
    s_ps = rows["s_ps"]
    gt_all = rows["gt"]
    out = {g: {m: [] for m in METRICS}
           for g in GROUPS + ABLATIONS + ("V_only", "T_only")}

    # ---- V8 / 消融 / T/V：标准 eval cells（fold/level 聚合） -------------
    for group in GROUPS + ABLATIONS + ("V_only", "T_only"):
        if group == "Gate":
            continue
        if group == "T_only":
            fam, cond = "V8_Cv5", "T2M"
        elif group == "V_only":
            fam, cond = "V8_Cv5", "V2M"
        else:
            fam, cond = group, "VT2M"
        vals = cell_values(fam, fold, level, cond)
        for metric in METRICS:
            out[group][metric].append(vals[metric])

    # ---- Gate：eval_motion T2M/VT2M 按 session 融合 ----------------------
    sessions = [s for s in np.unique(ses[lev == int(level)])]
    for session in sessions:
        m = (ses == session) & (lev == int(level))
        if not m.any():
            continue
        starts = st[m].astype(int)
        gt_win = gt_all[m]
        g_win = gate_from_s(s_ps[m], tau, width)
        base = load_motion("V8_Cv5", fold, level, str(session), "VT2M")
        if base is None:
            continue
        frames = base["frame"]
        P_vt = base["joint"]
        T_m = load_motion("V8_Cv5", fold, level, str(session), "T2M")
        if T_m is None:
            continue
        indices, gt, times, gs = [], [], [], []
        for wi, start in enumerate(starts):
            for j in range(gt_win.shape[1]):
                idx = np.where(frames == (start + j))[0]
                if len(idx) == 0:
                    continue
                indices.append(int(idx[0]))
                gt.append(gt_win[wi, j])
                times.append(T_m["time"][idx[0]])
                gs.append(g_win[wi])
        if not indices:
            continue
        indices = np.asarray(indices)
        gt = np.asarray(gt); times = np.asarray(times); gs = np.asarray(gs)
        floor_y = float(estimate_floor_y(gt))
        pred_t = T_m["joint"][indices]
        pred_vt = P_vt[indices]
        gate_pred = (1.0 - gs)[:, None, None] * pred_t + gs[:, None, None] * pred_vt
        for metric, v in session_metrics(gate_pred, gt, times, floor_y).items():
            out["Gate"][metric].append(v)
    return out


def collect() -> dict:
    """data[fold][level][group][metric] = list of session values."""
    data = {}
    for fold in FOLDS:
        data[fold] = {}
        for level in LEVELS:
            data[fold][level] = collect_fold_level(fold, level)
            print("[%s/%s] sessions=%d" % (fold, level,
                  len(data[fold][level]["Gate"]["PA-MPJPE"])))
    return data


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def four_fold(data, group, metric, level):
    return [v for fold in FOLDS for v in data[fold][level].get(group, {}).get(metric, [])]


def four_fold_mean(data, group, metric, level):
    vals = four_fold(data, group, metric, level)
    return float(np.mean(vals)) if vals else np.nan


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _fig_absolute(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), facecolor=SURFACE)
    x = np.asarray([int(l) for l in LEVELS])
    for ax, metric in zip(axes.flat, METRICS):
        _style(ax)
        for group in ("V8_Cv5", "V8_Cv7"):
            y = [four_fold_mean(data, group, metric, l) for l in LEVELS]
            ax.plot(x, y, lw=1.8, marker="o", ms=4.0, color=COLORS[group],
                    ls="--" if group == "V8_Cv7" else "-", label=LABELS[group])
        y = [four_fold_mean(data, "Gate", metric, l) for l in LEVELS]
        ax.plot(x, y, lw=2.1, marker="o", ms=4.5, color=COLORS["Gate"],
                label=LABELS["Gate"])
        # V only / T only references
        yv = [four_fold_mean(data, "V_only", metric, l) for l in LEVELS]
        yt = [four_fold_mean(data, "T_only", metric, l) for l in LEVELS]
        ax.plot(x, yv, lw=1.3, ls="-.", color="#8c6d31",
                label="V only (visual, no tactile)")
        ax.plot(x, yt, lw=1.3, ls=(0, (2, 2)), color="#6b6b6b",
                label="T only (tactile, no V)")
        ax.set_title(metric, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=8.5, color=MUTED)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8.0, color=MUTED)
        ax.set_xticks(x)
    axes.flat[0].legend(frameon=False, fontsize=7.2, loc="best")
    fig.suptitle("E5 crossfit: multi-metric absolute (4-fold mean; "
                 "V8 main line; fS6S7 = woS13)", fontsize=11.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig6_Gate_Metrics_Absolute.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def _fig_relative(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), facecolor=SURFACE)
    x = np.asarray([int(l) for l in LEVELS])
    for ax, metric in zip(axes.flat, METRICS):
        _style(ax)
        ax.axhline(0.0, color="#9a9890", lw=0.9, ls=":", zorder=1)
        gate = [four_fold_mean(data, "Gate", metric, l) for l in LEVELS]
        v8 = [four_fold_mean(data, "V8_Cv5", metric, l) for l in LEVELS]
        c7 = [four_fold_mean(data, "V8_Cv7", metric, l) for l in LEVELS]
        ax.plot(x, np.asarray(gate) - np.asarray(v8), lw=2.0, marker="o",
                ms=4.2, color=COLORS["Gate"], label="Gate − V8 (V gate)")
        ax.plot(x, np.asarray(gate) - np.asarray(c7), lw=1.7, marker="s",
                ms=3.8, color="#b04a8a", label="Gate − Cv7 (full stack)")
        ax.plot(x, np.asarray(v8) - np.asarray(c7), lw=1.7, marker="^",
                ms=3.8, color=COLORS["V8_Cv5"], label="V8 − Cv7 (T gate)")
        for a in ABLATIONS:
            ya = [four_fold_mean(data, a, metric, l) for l in LEVELS]
            ax.plot(x, np.asarray(ya) - np.asarray(c7), lw=1.1, ls=":",
                    color=COLORS[a], label="%s − Cv7" % a)
        ax.set_title(metric, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Occlusion (%)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("Δ %s (mm; negative = better)" % metric, fontsize=7.8,
                      color=MUTED)
        ax.set_xticks(x)
    axes.flat[0].legend(frameon=False, fontsize=6.8, loc="best")
    fig.suptitle("E5 crossfit: multi-metric relative (4-fold mean; "
                 "V8 main line; negative = improvement)", fontsize=11.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig7_Gate_Metrics_Relative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % out)


def export_tables(data: dict) -> None:
    # absolute: fold-level mean±std
    rows = []
    for group in GROUPS + ABLATIONS + ("V_only", "T_only"):
        for metric in METRICS:
            vals = [four_fold_mean(data, group, metric, l) for l in LEVELS]
            rows.append([group, metric] + [("%.3f" % v) if np.isfinite(v) else ""
                                           for v in vals])
    write_csv(OUT / "Main_Table6_Gate_Metrics_Absolute.csv",
              ["group", "metric"] + ["occ" + l for l in LEVELS], rows)

    # relative: gate-v8, gate-cv7, v8-cv7, ablations
    rows = []
    for metric in METRICS:
        for label, fn in (
                ("Gate−V8", lambda l: four_fold_mean(data, "Gate", metric, l)
                 - four_fold_mean(data, "V8_Cv5", metric, l)),
                ("Gate−Cv7", lambda l: four_fold_mean(data, "Gate", metric, l)
                 - four_fold_mean(data, "V8_Cv7", metric, l)),
                ("V8−Cv7", lambda l: four_fold_mean(data, "V8_Cv5", metric, l)
                 - four_fold_mean(data, "V8_Cv7", metric, l))):
            rows.append([metric, label] + ["%+.3f" % fn(l) for l in LEVELS])
        for a in ABLATIONS:
            rows.append([metric, "%s−Cv7" % a]
                        + ["%+.3f" % (four_fold_mean(data, a, metric, l)
                                     - four_fold_mean(data, "V8_Cv7", metric, l))
                           for l in LEVELS])
    write_csv(OUT / "Main_Table7_Gate_Metrics_Relative.csv",
              ["metric", "quantity"] + ["occ" + l for l in LEVELS], rows)

    # per-fold long: fold, level, metric, group, mean
    rows = []
    for fold in FOLDS:
        for level in LEVELS:
            for group in GROUPS + ABLATIONS + ("V_only", "T_only"):
                for metric in METRICS:
                    vals = data[fold][level].get(group, {}).get(metric, [])
                    if vals:
                        rows.append([fold, "occ" + level, metric, group,
                                     "%.3f" % float(np.mean(vals)),
                                     "%.3f" % float(np.std(vals)), len(vals)])
    write_csv(OUT / "Full_Table4_Gate_Metrics_PerFold.csv",
              ["fold", "level", "metric", "group", "mean", "std", "n"], rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    export_tables(data)
    _fig_absolute(data)
    _fig_relative(data)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
