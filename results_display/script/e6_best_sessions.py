#!/usr/bin/env python3
"""E6 各运动「门控滤除无门伤害」最佳完整会话合并图（2026-11-01 用户裁定）。

每运动选表现最好的会话 = 该会话全帧（双脚合并）上无门 Cv7 的伤害量
H = mean(max(Δ,0)) 大、双门 B 的 H 明显更小（滤除量 = H7 − H5 最大），
且双门的增益 A = mean(max(−Δ,0)) 基本持平无门（A7 − A5 ≤ GAIN_TOL）。
子图 = 该会话已有的 Full_Fig_{action}_{sid}.png 原图（e6_phase_analysis
导出，2 面板分脚 Δ 曲线 + 相位底色），imshow 原样嵌入，不做任何截断或
处理。若某动作没有任何会话满足条件（H7 ≥ HARM_MIN、滤除 ≥ REMOVE_MIN、
增益损失 ≤ GAIN_TOL），该动作子图加红框。

用法（touch_gait env，仓库根）：
  python results_display/script/e6_best_sessions.py [--dry-run] [--metric pa_mpjpe]
  或经统一导出管线：python results_display/script/e_results_export.py
  --experiment E6 --best-sessions
产物（2026-11-02 起按指标子文件夹）：
  results_display/Etest/E6_ContactPhaseAnalysis/{ankle_foot_mpjpe,pa_mpjpe}/Full_Fig_E6_BestSessions.png
  …/{metric}/detail/e6_best_sessions.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_note/HMR2.0/hmr2_gait"))
from anysole_tta.e6_phase_analysis import build_session_sets, load_session  # noqa: E402
from anysole_tta.phase_common import METRICS  # noqa: E402

E6_BASE = REPO / "results_display/Etest/E6_ContactPhaseAnalysis"
# 位置误差指标文件夹（2026-11-02 裁定：逐指标同模式生成）
GATE, NOGATE = "V7_Cv5_s1", "V7_Cv7_s1"
# 选图阈值随指标缩放（pa_mpjpe 全身误差量级约为踝足的 1/4）
THRESHOLDS = {"ankle_foot_mpjpe": (5.0, 2.0, 4.0),
              "pa_mpjpe": (1.25, 0.5, 1.0)}
ACTION_ORDER = ["walk", "run", "squat", "stand_shift", "single_stand",
                "jump_jack", "lateral", "sit_stand", "stairs"]
ACTION_TITLE = {"walk": "Walk", "run": "Run", "squat": "Squat",
                "stand_shift": "Stand-shift", "single_stand": "Single-stand",
                "jump_jack": "Jump-jack", "lateral": "Lateral",
                "sit_stand": "Sit-stand", "stairs": "Stairs"}


def harm(data):
    """双脚合并 H（伤害量）/ A（收益量），mm。"""
    valid = np.asarray(data["valid"])
    hs = []
    for foot in ("left", "right"):
        g = np.asarray(data[f"VT2M_{foot}"]) - np.asarray(data[f"V2M_{foot}"])
        m = valid & np.isfinite(g)
        g = g[m]
        hs.append((float(np.mean(np.maximum(g, 0))), float(np.mean(np.maximum(-g, 0)))))
    return (float(np.mean([h[0] for h in hs])), float(np.mean([h[1] for h in hs])))


def main() -> int:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.patches import Rectangle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--metric", choices=METRICS, default="ankle_foot_mpjpe",
                    help="位置误差指标（缺省 ankle_foot_mpjpe）")
    args = ap.parse_args()
    OUT = E6_BASE / args.metric
    DETAIL = OUT / "detail"
    HARM_MIN, REMOVE_MIN, GAIN_TOL = THRESHOLDS[args.metric]

    session_sets = build_session_sets()
    picks = {}
    for act in ACTION_ORDER:
        best, best_q = None, None
        for sid in session_sets[act]:
            h7, a7 = harm(load_session(NOGATE, sid, args.metric))
            h5, a5 = harm(load_session(GATE, sid, args.metric))
            removed, gain_drop = h7 - h5, a7 - a5
            qualify = h7 >= HARM_MIN and removed >= REMOVE_MIN and gain_drop <= GAIN_TOL
            key = (removed, h7)
            if best is None or key > best["key"]:
                best = {"sid": sid, "h7": h7, "h5": h5, "a7": a7, "a5": a5,
                        "removed": removed, "gain_drop": gain_drop,
                        "qualify": qualify, "key": key}
            if qualify and (best_q is None or key > best_q["key"]):
                best_q = dict(best)
        picks[act] = best_q if best_q is not None else best
        b = picks[act]
        print(f"{act:13s} {b['sid']}  H7={b['h7']:+5.1f} H5={b['h5']:+5.1f} removed={b['removed']:+5.1f}  "
              f"A7={b['a7']:+4.1f} A5={b['a5']:+4.1f} gainDrop={b['gain_drop']:+4.1f}  "
              f"{'QUALIFY' if b['qualify'] else 'RED-FRAME'}", flush=True)
    (DETAIL / "e6_best_sessions.json").write_text(
        json.dumps(picks, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        return 0

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 15.0))
    for ax, act in zip(axes.ravel(), ACTION_ORDER):
        info = picks[act]
        png = DETAIL / "per_session" / f"Full_Fig_{act}_{info['sid']}.png"
        assert png.exists(), str(png)
        img = imread(str(png))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            f"{ACTION_TITLE[act]} — {info['sid']}\n"
            f"harm {info['h7']:+.1f} $\\to$ {info['h5']:+.1f} mm "
            f"(removed {info['removed']:+.1f})  |  "
            f"gain {info['a7']:+.1f} $\\to$ {info['a5']:+.1f} mm",
            fontsize=10.5, pad=4)
        if not info["qualify"]:
            ax.add_patch(Rectangle((-0.004, -0.008), 1.008, 1.016,
                                   transform=ax.transAxes, fill=False,
                                   edgecolor="#d62728", lw=4, zorder=10))
            ax.set_title(ax.get_title() + "\nno qualifying session",
                         fontsize=10.5, pad=4, color="#d62728")
    fig.suptitle("E6 — best full session per motion (unmodified per-session "
                 "figures); dual gate (B) vs no gate (Cv7); qualifier = "
                 "harm removed $\\geq$%.1f mm with gain drop $\\leq$%.1f mm; "
                 "red frame = no qualifying session" % (REMOVE_MIN, GAIN_TOL),
                 y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / "Full_Fig_E6_BestSessions.png", dpi=130)
    print("saved", OUT / "Full_Fig_E6_BestSessions.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
