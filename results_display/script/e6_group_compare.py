#!/usr/bin/env python3
"""E6 分组对比曲线（B/T/LR/TLR/Q × 左右脚，2026-10-31）。

口径同 E6：逐帧未平滑 Δ = VT − V（mm，骨盆对齐踝足 MPJPE），有效帧分脚；
ckpt_last；B = V7_Cv5（双门基线）/ T / LR / TLR / Q + 对照 Cv4（单门）/
Cv7（无门）。展示用途（§9：总图平滑只用于展示，验收用分脚未平滑数据）。

图布局（2026-10-31 用户裁定：1×2 上下分布，上=左脚、下=右脚，左右脚
分开不合并不平均）：
  - curves/ 每动作 × 每会话一张图：上下两面板，各面板 = 各模型 Δ
    曲线（原始 α 淡 + 20 帧滑动均值主线）+ 零线。
  - scatter/ 每动作 × 每会话一张图（与 curves 同一件事的散点表述，
    同 1×2 上下分布）：各面板 = 各模型 × 该会话全部有效帧 Δ 散点
    （jitter，构图阈值 ±30：>30 判定为 +30、<−30 判定为 −30）+ 中位数。
  - 汇总表：分脚分模型 mean Δ / H / A / E（会话内均值再跨会话均值）。

用法（touch_gait env，仓库根）：
  python results_display/script/e6_group_compare.py
产物：results_display/Etest/E6_ContactPhaseAnalysis/group_compare/{curves,scatter}/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_note/HMR2.0/hmr2_gait"))
from anysole_tta.e6_phase_analysis import FPS, build_session_sets, load_session  # noqa: E402

OUT = REPO / "results_display/Etest/E6_ContactPhaseAnalysis" / "group_compare"
# B + 候选五组 + 对照两组（Cv4 单门 / Cv7 无门）
MODELS = ("V7_Cv5_s1", "V7_Cv4_s1", "V7_Cv7_s1", "V7_Cv5T_s1", "V7_Cv5LR_s1",
          "V7_Cv5TLR_s1", "V7_Cv5Q_s1")
MODEL_COLORS = {"V7_Cv5_s1": "#111111", "V7_Cv4_s1": "#8c564b",
                "V7_Cv7_s1": "#9467bd", "V7_Cv5T_s1": "#2a78d6",
                "V7_Cv5LR_s1": "#f4a582", "V7_Cv5TLR_s1": "#2ca25f",
                "V7_Cv5Q_s1": "#d62728"}
MODEL_LABELS = {"V7_Cv5_s1": "B (Cv5)", "V7_Cv4_s1": "Cv4 (1-gate)",
                "V7_Cv7_s1": "Cv7 (no gate)", "V7_Cv5T_s1": "T",
                "V7_Cv5LR_s1": "LR", "V7_Cv5TLR_s1": "TLR",
                "V7_Cv5Q_s1": "Q"}
ACTION_NUM = {"walk": 1, "run": 3, "squat": 5, "stand_shift": 6,
              "single_stand": 7, "jump_jack": 8, "lateral": 9,
              "sit_stand": 10, "stairs": 11}


def main() -> int:
    global MODELS
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS),
                        help="逗号分隔模型（缺省 = B/Cv4/Cv7/T/LR/TLR/Q 七模型）")
    parser.add_argument("--actions", default=None,
                        help="动作组逗号分隔（缺省 = canonical test 全部）")
    args = parser.parse_args()
    MODELS = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if args.actions:
        keep = {a.strip() for a in args.actions.split(",") if a.strip()}
    session_sets = build_session_sets()
    if args.actions:
        session_sets = {a: v for a, v in session_sets.items() if a in keep}
    sids = sorted({sid for v in session_sets.values() for sid in v})
    cache = {}

    def data_of(model, sid):
        key = (model, sid)
        if key not in cache:
            cache[key] = load_session(model, sid)
        return cache[key]

    per_session = {m: {f: {} for f in ("left", "right")} for m in MODELS}
    for m in MODELS:
        for sid in sids:
            d = data_of(m, sid)
            valid = np.asarray(d["valid"])
            for foot in ("left", "right"):
                v = np.asarray(d["V2M_" + foot], dtype=np.float64)
                vt = np.asarray(d["VT2M_" + foot], dtype=np.float64)
                g = vt - v
                mask = valid & np.isfinite(g)
                g = np.where(mask, g, np.nan)
                per_session[m][foot][sid] = g
                per_session[m][foot]["valid_" + sid] = mask

    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    (OUT / "scatter").mkdir(parents=True, exist_ok=True)

    # ---- curves：每动作 × 每会话一张，1×2 上下（上左/下右）----
    for action, act_sids in session_sets.items():
        for sid in act_sids:
            fig, axes = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True)
            for foot_i, (ax, foot) in enumerate(zip(axes, ("left", "right"))):
                for m in MODELS:
                    g = per_session[m][foot][sid]
                    t = np.arange(len(g)) / FPS
                    sma = np.convolve(np.nan_to_num(g), np.ones(20) / 20,
                                      mode="same")
                    sma = np.where(np.isnan(g), np.nan, sma)
                    ax.plot(t, g, color=MODEL_COLORS[m], alpha=0.12, lw=0.7)
                    ax.plot(t, sma, color=MODEL_COLORS[m], lw=1.4,
                            label=MODEL_LABELS[m] if foot_i == 0 else None)
                ax.axhline(0, color="k", lw=0.6, ls="--")
                ax.set_ylim(-45, 45)
                ax.set_yticks([-40, -20, 0, 20, 40])
                ax.set_ylabel("%s foot\nΔ = VT - V (mm)" % foot)
            axes[-1].set_xlabel("time (s)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8,
                       frameon=False)
            fig.suptitle("E6 group compare — %s — %s (thin = raw, "
                         "thick = 20-frame SMA)" % (action, sid), y=0.99)
            fig.tight_layout(rect=(0, 0.05, 1, 0.96))
            fig.savefig(OUT / "curves" / ("Full_GFig%d_E6_Group_%s_%s.png"
                                          % (ACTION_NUM[action], action, sid)),
                        dpi=130)
            plt.close(fig)
        print("curves %s (%d sessions)" % (action, len(act_sids)), flush=True)

    # ---- scatter：每动作 × 每会话一张，1×2 上下，帧级散点（±30）----
    rng = np.random.default_rng(0)
    for action, act_sids in session_sets.items():
        for sid in act_sids:
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.6))
            for foot_i, (ax, foot) in enumerate(zip(axes, ("left", "right"))):
                for mi, m in enumerate(MODELS):
                    g = per_session[m][foot][sid]
                    vals = g[per_session[m][foot]["valid_" + sid]]
                    vals = np.clip(vals, -30, 30)   # 构图阈值：±30 判定
                    x = np.full_like(vals, mi, dtype=float) + rng.uniform(
                        -0.2, 0.2, size=len(vals))
                    ax.scatter(x, vals, s=2, color=MODEL_COLORS[m], alpha=0.22,
                               lw=0)
                    ax.hlines(np.median(vals), mi - 0.32, mi + 0.32,
                              color=MODEL_COLORS[m], lw=2.4)
                ax.axhline(0, color="k", lw=0.7, ls="--")
                ax.set_xticks(range(len(MODELS)))
                ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                                   rotation=40, ha="right", fontsize=8)
                ax.set_ylabel("%s foot\nframe Δ (mm, clipped at ±30)" % foot)
                ax.set_ylim(-30, 30)
            fig.suptitle("E6 group compare — %s — %s (dot = 1 valid frame, "
                         "bar = median)" % (action, sid))
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(OUT / "scatter" / ("Full_GFig%d_E6_Group_%s_%s_scatter.png"
                                           % (ACTION_NUM[action], action, sid)),
                        dpi=130)
            plt.close(fig)
        print("scatter %s (%d sessions)" % (action, len(act_sids)),
              flush=True)

    # ---- 汇总表（分脚分模型，会话内均值再跨会话均值；展示口径）----
    lines = ["# E6 分组对比汇总（分脚，会话内均值 → 跨会话均值；展示口径）",
             "", "| 模型 | 脚 | mean Δ | H | A | E(VT) |", "|---|---|---|---|---|---|"]
    stats = {}
    for m in MODELS:
        for foot in ("left", "right"):
            rows = []
            for sid in sids:
                g = per_session[m][foot][sid]
                mask = per_session[m][foot]["valid_" + sid]
                gv = g[mask]
                if len(gv) == 0:
                    continue
                rows.append((float(np.nanmean(gv)),
                             float(np.maximum(gv, 0).mean()),
                             float(np.maximum(-gv, 0).mean()),
                             float(np.nanmean(np.asarray(
                                 data_of(m, sid)["VT2M_" + foot],
                                 dtype=np.float64)[mask]))))
            arr = np.array(rows)
            stats[(m, foot)] = arr.mean(axis=0)
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.2f |"
                         % (MODEL_LABELS[m], "左" if foot == "left" else "右",
                            *stats[(m, foot)]))
    lines.append("")
    lines.append("注：展示口径（§9 展示项）；正式判定 = 逐会话配对统计"
                 "（E6_配对统计*.md）。")
    (OUT / "E6_分组对比汇总.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print("wrote %s" % OUT)
    for m in MODELS:
        print("  %-14s 左 Δ%+.2f H%.2f A%.2f | 右 Δ%+.2f H%.2f A%.2f"
              % (MODEL_LABELS[m], *(stats[(m, "left")][:3]),
                 *(stats[(m, "right")][:3])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
