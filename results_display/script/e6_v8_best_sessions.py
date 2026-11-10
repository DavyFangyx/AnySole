#!/usr/bin/env python3
"""E6 V8 四折 LSO：逐动作「无门退化 → 双门翻转」最佳完整会话图（2026-11-09）。

对应旧 E6 的 ``Full_Fig_E6_BestSessions.png``，但数据源改为 V8 四折 LSO
（fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8，只评各折 test）。每个候选 =
一个 (session, fold) cell；同一会话在不同折里是不同训练集的预测。

选图准则（用户 2026-11-09 口径：只留好效果）
- Cv7（无门）平均 Δ ≥ 0.5 mm：无门确实退化；
- Cv5（双门）平均 Δ ≤ 0：门控不退化/翻转成增益；
- 在满足者中取 ``flip = mean(Δ7) − mean(Δ5)`` 最大。
若某动作无满足者，退化为 flip 最大且 Δ7 > 0；再无则 Δ7 最大，并红框标注。

曲线口径 = 项目 pa_mpjpe：全 24 关节逐帧相似变换对齐 MPJPE（mm），
Δ = VT − V（负 = 触觉有帮助）；上下两面板分别是左/右脚相位带，
全身曲线在两面板重复（与旧 E6 pa_mpjpe 图一致）。

产物
----
``results_display/Etest_crossfit_v8/E6_ContactPhaseAnalysis/pa_mpjpe/``
  Full_Fig_E6_BestSessions_V8.png
  detail/E6_BestSessions_V8.csv
  detail/E6_BestSessions_V8.md
  detail/per_session/Full_Fig_{action}_{sid}_{fold}.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anysole_tta.phase_common import (action_of, load_contact, load_gt_joints,  # noqa: E402
                                      load_pred_joints, metric_error, phase_labels,
                                      PHASE_NAMES)

FOLDS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12.csv",
    "fS5S11S12": "splits_test_S5S11S12_val_S13.csv",
    "fS6S7": "splits_test_S6S7_val_S14.csv",
    "fS5S6S8": "splits_test_S5S6S8_val_S7.csv",
}
SPLITS = REPO / "AnysoleWorkspace" / "protocol" / "splits" / "default"
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "E6_ContactPhaseAnalysis" / "pa_mpjpe"
DETAIL = OUT / "detail"
PER_SESSION = DETAIL / "per_session"

MODELS = ("V8_Cv5", "V8_Cv7")
MODEL_LABEL = {"V8_Cv5": "V8_Cv5 (double gate)",
               "V8_Cv7": "V8_Cv7 (no gate)"}
MODEL_COLOR = {"V8_Cv5": "#2a78d6", "V8_Cv7": "#52514e"}
ACTION_ORDER = ["walk", "run", "squat", "stand_shift", "single_stand",
                "jump_jack", "lateral", "sit_stand", "stairs"]
ACTION_TITLE = {"walk": "Walk", "run": "Run", "squat": "Squat",
                "stand_shift": "Stand-shift", "single_stand": "Single-stand",
                "jump_jack": "Jump-jack", "lateral": "Lateral",
                "sit_stand": "Sit-stand", "stairs": "Stairs"}
ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}
HARM_MIN = 0.5
GATE_MAX = 0.0
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def _cells() -> list:
    out = []
    for fold, filename in FOLDS.items():
        with (SPLITS / filename).open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for sid in sorted({str(row["test"]).strip() for row in rows
                           if str(row.get("test", "")).strip()}):
            out.append((sid, fold))
    return out


class Cache:
    def __init__(self) -> None:
        self.gt: dict = {}
        self.contact: dict = {}

    def joints(self, sid: str) -> np.ndarray:
        if sid not in self.gt:
            self.gt[sid] = load_gt_joints(sid)
        return self.gt[sid]

    def labels(self, sid: str) -> tuple:
        if sid not in self.contact:
            self.contact[sid] = load_contact(sid)
        return self.contact[sid]


def _series(sid: str, fold: str, model: str, cache: Cache) -> Optional[dict]:
    cell = "%s_%s_s1" % (model, fold)
    try:
        pred_v, _, valid_v = load_pred_joints(cell, sid, "V2M")
        pred_t, _, valid_t = load_pred_joints(cell, sid, "VT2M")
    except FileNotFoundError:
        return None
    gt = cache.joints(sid)
    contact, _, contact_valid = cache.labels(sid)
    n = min(len(gt), len(contact), len(pred_v), len(pred_t))
    gt = gt[:n]
    contact = contact[:n]
    pred_v, pred_t = pred_v[:n], pred_t[:n]
    mask = (np.asarray(valid_v[:n]).astype(bool)
            & np.asarray(valid_t[:n]).astype(bool)
            & np.asarray(contact_valid[:n]).astype(bool))
    err_v = metric_error(pred_v, gt, "pa_mpjpe", "full")
    err_t = metric_error(pred_t, gt, "pa_mpjpe", "full")
    ok = mask & np.isfinite(err_v) & np.isfinite(err_t)
    if int(ok.sum()) < 20:
        return None
    delta = err_t - err_v
    return {
        "sid": sid, "fold": fold, "model": model,
        "delta": delta, "ok": ok, "phase": phase_labels(contact),
        "mean": float(delta[ok].mean()),
        "harm": float(np.maximum(delta[ok], 0.0).mean()),
        "gain": float(np.maximum(-delta[ok], 0.0).mean()),
        "n_frames": int(ok.sum()),
    }


def collect(cache: Cache) -> tuple:
    series: dict = {}
    for sid, fold in _cells():
        try:
            cache.joints(sid)
            cache.labels(sid)
        except Exception:
            continue
        for model in MODELS:
            result = _series(sid, fold, model, cache)
            if result is not None:
                series[(sid, fold, model)] = result
    return series


def select(series: dict) -> dict:
    by_action: dict = {}
    for (sid, fold, model), data in series.items():
        if model != "V8_Cv7":
            continue
        gate = series.get((sid, fold, "V8_Cv5"))
        if gate is None:
            continue
        action = ACTION_GROUPS.get(action_of(sid), None)
        if action is None:
            continue
        item = {
            "action": action, "sid": sid, "fold": fold,
            "d7": data["mean"], "d5": gate["mean"],
            "flip": data["mean"] - gate["mean"],
            "removed": data["harm"] - gate["harm"],
            "gain_drop": data["gain"] - gate["gain"],
            "h7": data["harm"], "h5": gate["harm"],
            "a7": data["gain"], "a5": gate["gain"],
            "n_frames": data["n_frames"],
        }
        by_action.setdefault(action, []).append(item)
    picks = {}
    for action in ACTION_ORDER:
        rows = by_action.get(action, [])
        if not rows:
            continue
        qualified = [row for row in rows
                     if row["d7"] >= HARM_MIN and row["d5"] <= GATE_MAX]
        pool = qualified
        qualify = True
        if not pool:
            pool = [row for row in rows if row["d7"] > 0.0]
            qualify = False
        if not pool:
            pool = rows
            qualify = False
        pick = max(pool, key=lambda row: (row["flip"], row["removed"]))
        pick = dict(pick)
        pick["qualify"] = qualify
        picks[action] = pick
    return picks


def render_session(action: str, pick: dict, series: dict, ylim: tuple) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sid, fold = pick["sid"], pick["fold"]
    data = {model: series[(sid, fold, model)] for model in MODELS
            if (sid, fold, model) in series}
    any_data = next(iter(data.values()))
    t = np.arange(len(any_data["delta"])) / 40.0
    phase_colors = {0: "#d9d9d9", 1: "#ffffff", 2: "#f4a582", 3: "#92c5de"}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, squeeze=True)
    for foot_i, (ax, foot) in enumerate(zip(axes, ("left", "right"))):
        ax.set_xlim(0, t[-1])
        ph = any_data["phase"][:, foot_i]
        for k in range(4):
            sel = ph == k
            if sel.any():
                boundaries = np.flatnonzero(np.diff(np.r_[False, sel, False].astype(np.int8)))
                for number, (start, stop) in enumerate(zip(boundaries[::2], boundaries[1::2])):
                    ax.axvspan(t[start], t[stop - 1] + 1.0 / 40.0,
                               ymin=0.0, ymax=0.08, color=phase_colors[k],
                               alpha=0.6, lw=0,
                               label=PHASE_NAMES[k] if number == 0 else None)
        for model, mdata in data.items():
            delta = mdata["delta"].copy()
            delta[~mdata["ok"]] = np.nan
            smooth = np.convolve(np.nan_to_num(delta), np.ones(20) / 20.0, mode="same")
            ax.plot(t, delta, color=MODEL_COLOR[model], alpha=0.22, lw=0.8)
            ax.plot(t, smooth, color=MODEL_COLOR[model], lw=1.8,
                    label=MODEL_LABEL[model] if foot_i == 0 else None)
        ax.axhline(0, color=INK, lw=0.7, ls="--")
        ax.set_ylim(*ylim)
        ax.set_ylabel("Δ = VT − V (mm)", fontsize=9.5, color=MUTED)
        ax.set_title("full body — %s-foot phases" % foot, fontsize=10, color=INK,
                     loc="left")
        ax.tick_params(labelsize=8.5, colors=MUTED)
        ax.grid(alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("time (s)", fontsize=9.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 7), fontsize=8.2, frameon=False)
    fig.suptitle("%s — %s — %s" % (ACTION_TITLE[action], sid, fold),
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    PER_SESSION.mkdir(parents=True, exist_ok=True)
    path = PER_SESSION / ("Full_Fig_%s_%s_%s.png" % (action, sid, fold))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compose(picks: dict, paths: dict, ylim: tuple) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 15.0), facecolor="#fcfcfb")
    for ax, action in zip(axes.ravel(), ACTION_ORDER):
        pick = picks.get(action)
        if pick is None:
            ax.axis("off")
            continue
        ax.imshow(imread(str(paths[action])))
        ax.set_axis_off()
        title = ("%s — %s (%s)\n"
                 "mean Δ: Cv7 %+.1f → Cv5 %+.1f mm (flip %+.1f)  |  "
                 "harm %+.1f → %+.1f mm"
                 % (ACTION_TITLE[action], pick["sid"], pick["fold"],
                    pick["d7"], pick["d5"], pick["flip"],
                    pick["h7"], pick["h5"]))
        ax.set_title(title, fontsize=10.2, color=INK, pad=4)
        if not pick["qualify"]:
            ax.add_patch(Rectangle((-0.004, -0.008), 1.008, 1.016,
                                   transform=ax.transAxes, fill=False,
                                   edgecolor="#d03b3b", lw=4, zorder=10))
    fig.suptitle(
        "E6 (V8 four-fold LSO) — per action, the session with the largest "
        "no-gate→gate flip: Cv7 mean Δ ≥ %.1f mm, Cv5 mean Δ ≤ %.1f mm; "
        "V8_Cv5 = double gate, V8_Cv7 = no gate. "
        "Curves = per-frame full-body PA-MPJPE Δ (thin = raw, thick = 20-frame SMA)."
        % (HARM_MIN, GATE_MAX),
        y=0.995, fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Full_Fig_E6_BestSessions_V8.png"
    fig.savefig(path, dpi=140, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % path)


def write_all_cells(series: dict) -> dict:
    """Every (session, fold) Cv7/Cv5 pair, plus prevalence of the flip."""
    rows = []
    for (sid, fold, model), data in series.items():
        if model != "V8_Cv7":
            continue
        gate = series.get((sid, fold, "V8_Cv5"))
        if gate is None:
            continue
        action = ACTION_GROUPS.get(action_of(sid), "")
        rows.append([action, sid, fold,
                     round(data["mean"], 3), round(gate["mean"], 3),
                     round(data["mean"] - gate["mean"], 3),
                     round(data["harm"] - gate["harm"], 3),
                     data["n_frames"]])
    path = DETAIL / "E6_AllCells_V8.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["动作", "会话", "折", "Cv7_meanΔ", "Cv5_meanΔ",
                         "flip", "removed", "帧数"])
        writer.writerows(rows)
    n = len(rows)
    degrade7 = sum(1 for row in rows if row[3] > 0.0)
    safe5 = sum(1 for row in rows if row[4] <= 0.0)
    flip = sum(1 for row in rows if row[3] > 0.0 and row[4] <= 0.0)
    stats = {"n": n, "degrade7": degrade7, "safe5": safe5, "flip": flip}
    print("[csv] %s (%d cells; Cv7>0: %d, Cv5<=0: %d, flip: %d)"
          % (path, n, degrade7, safe5, flip))
    return stats


def write_tables(picks: dict, stats: dict) -> None:
    header = ["动作", "会话", "折", "Cv7_meanΔ", "Cv5_meanΔ",
              "flip", "removed", "gain_drop", "H7", "H5", "A7", "A5",
              "帧数", "达标"]
    rows = []
    for action in ACTION_ORDER:
        pick = picks.get(action)
        if pick is None:
            continue
        rows.append([action, pick["sid"], pick["fold"],
                     round(pick["d7"], 3), round(pick["d5"], 3),
                     round(pick["flip"], 3), round(pick["removed"], 3),
                     round(pick["gain_drop"], 3), round(pick["h7"], 3),
                     round(pick["h5"], 3), round(pick["a7"], 3),
                     round(pick["a5"], 3), pick["n_frames"],
                     "Y" if pick["qualify"] else "N"])
    path = DETAIL / "E6_BestSessions_V8.csv"
    DETAIL.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("[csv] %s" % path)

    lines = [
        "# E6 V8 四折 LSO：逐动作最佳会话（无门退化 → 双门翻转）",
        "",
        "选图准则：Cv7（无门）平均 Δ ≥ %.1f mm，Cv5（双门）平均 Δ ≤ %.1f mm；"
        "在达标者中取 flip = mean(Δ7) − mean(Δ5) 最大。" % (HARM_MIN, GATE_MAX),
        "Δ = VT − V（pa_mpjpe，mm；负 = 触觉有帮助）。",
        "",
        "全量候选（%d 个 session×fold cell）：Cv7 平均 Δ > 0 的 %d/%d（%.0f%%），"
        "Cv5 平均 Δ ≤ 0 的 %d/%d（%.0f%%），两者同时成立的 flip cell %d/%d（%.0f%%）。"
        % (stats["n"], stats["degrade7"], stats["n"],
           100.0 * stats["degrade7"] / max(stats["n"], 1),
           stats["safe5"], stats["n"],
           100.0 * stats["safe5"] / max(stats["n"], 1),
           stats["flip"], stats["n"],
           100.0 * stats["flip"] / max(stats["n"], 1)),
        "",
        "遮挡档位的身体部位结构见 "
        "`E5_OcclusionRobustness/SMPLHeatmap/`（SMPL heatmap；occ040 对账表）。",
        "单动作的 VT 姿态 teaser 见 `pa_mpjpe/teaser/`（V5/V7 + 曲线）。",
        "",
        "| 动作 | 会话 | 折 | Cv7 Δ | Cv5 Δ | flip | 伤害 H7→H5 | 达标 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for action in ACTION_ORDER:
        pick = picks.get(action)
        if pick is None:
            continue
        lines.append("| %s | %s | %s | %+.2f | %+.2f | %+.2f | %.2f → %.2f | %s |"
                     % (ACTION_TITLE[action], pick["sid"], pick["fold"],
                        pick["d7"], pick["d5"], pick["flip"],
                        pick["h7"], pick["h5"],
                        "Y" if pick["qualify"] else "N"))
    (DETAIL / "E6_BestSessions_V8.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="pa_mpjpe", choices=("pa_mpjpe",))
    parser.add_argument("--ylim", type=float, default=30.0,
                        help="symmetric y limit in mm (default: 30, as old E6)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cache = Cache()
    series = collect(cache)
    picks = select(series)
    for action in ACTION_ORDER:
        pick = picks.get(action)
        if pick is None:
            print("%-13s no candidate" % action)
            continue
        print("%-13s %s %-9s d7=%+6.2f d5=%+6.2f flip=%+6.2f "
              "removed=%+5.2f %s"
              % (action, pick["sid"], pick["fold"], pick["d7"], pick["d5"],
                 pick["flip"], pick["removed"],
                 "QUALIFY" if pick["qualify"] else "RELAXED"))
    stats = write_all_cells(series)
    write_tables(picks, stats)
    if args.dry_run:
        return 0
    ylim = (-args.ylim, args.ylim)
    print("[ylim] ±%.1f mm" % args.ylim)
    paths = {action: render_session(action, pick, series, ylim)
             for action, pick in picks.items()}
    compose(picks, paths, ylim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
