"""把 E9 四折 crossfit 的 combined 产物拆成 V7/V8 两个根（2026-11-09）。

输入：Etest_crossfit/E9_TrajectorySceneOcc/crossfit/ 的 combined 表。
输出：
  Etest_crossfit/E9_TrajectorySceneOcc/crossfit/      V7_Cv5 only
  Etest_crossfit_v8/E9_TrajectorySceneOcc/crossfit/   V8_Cv5 only
  Etest_crossfit/E9_TrajectorySceneOcc/crossbase/     combined（保留）
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/data/fangyuxuan/projects/gait"
V7 = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc"
V8 = ROOT + "/results_display/Etest_crossfit_v8/E9_TrajectorySceneOcc"
_HAS_CB = os.path.isfile(V7 + "/crossbase/Main_Table1_E9_Crossfit_OccAte_FoldAction.csv")
SRC = V7 + "/crossbase" if _HAS_CB else V7 + "/crossfit"
BASES = {"V7_Cv5": V7, "V8_Cv5": V8}
ARMS = ["Step2Motion", "Anysolo-TI", "Anysolo-I", "VT", "Hybrid"]
COLORS = {"Step2Motion": "#2a78d6", "Anysolo-TI": "#1baf7a",
          "Anysolo-I": "#eda100", "VT": "#8a6d3b", "Hybrid": "#7b52ab"}
DROP = ("Hybrid+VT0.50", "Hybrid+VT0.75")


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [k for k in rows[0].keys() if not any(d in k for d in DROP)]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])


def fig_for(mean_rows, base, path):
    acts = sorted({r["action"] for r in mean_rows})
    x = np.arange(len(acts)); width = 0.15
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, arm in enumerate(ARMS):
        vals = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_mean"])
                for a in acts]
        errs = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_std"])
                for a in acts]
        ax.bar(x + (k - 2.0) * width, vals, width, color=COLORS[arm], label=arm,
               yerr=errs, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(acts)
    ax.set_ylabel("occlusion ATE (mm)")
    ax.set_title("%s — four-fold fold mean±std" % base)
    ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def report_for(mean_rows, base, path):
    lines = ["# E9 四折 crossfit 场景遮挡轨迹（%s，2026-11-09）" % base, "",
             "## 口径", "",
             "- 四折 LSO woS13；每折 test 的 scene-occ manifest 会话（03/04/05/08/09）。",
             "- 本文件只含 **%s**；V7/V8 另一基座见对应根目录。" % base,
             "- S2M：逐折 ckpt 的 per-fold 预测（`gait_model_smpl24_<fold>`），不是零样本。",
             "- Hybrid：逐折 `_last` 模型；四折 test 受试者留出。",
             "- 指标：遮挡内 ATE（full_event 并集）；四折报 fold-level mean±std，不池化。",
             "- VT 是 occluded-V prior，不是视觉证据。",
             "- 空间遮挡：阴影楔形 = cam3 通过柱子的地面阴影（整片调暗）；柱脚深灰；不是 GT 轨迹带。", "",
             "## 四折 fold-level mean±std（mm）", "",
             "| 动作 | " + " | ".join(ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for r in mean_rows:
        lines.append("| %s | %s |" % (r["action"], " | ".join(
            "%.1f±%.1f" % (float(r[a + "_mean"]), float(r[a + "_std"]))
            for a in ARMS)))
    lines += ["", "## 读数", "",
              "- 纯 Hybrid 与 VT 的逐动作读数见上表；本表只作 fold-level 汇总。",
              "- 固定 α 的 H+VT0.50/0.75 已按 2026-11-09 裁定移除：两个近似独立轨迹逐帧加权会产生 Z 字抖动，不作为方法臂。",
              "- S2M 是逐折模型；与 7 会话 S13/S14 eval-only 表不是同一池，禁止跨表比较。",
              "- 四折 std 很大（尤其 05/09），不做显著性主张。", "",
              "## 文件", "",
              "- `Main_Table1_E9_Crossfit_OccAte_FoldAction.csv`：逐折×动作",
              "- `Main_Table2_E9_Crossfit_OccAte_MeanStd.csv`：四折 mean±std",
              "- `Full_Table1_E9_Crossfit_PerSession.csv`：逐会话全臂",
              "- `Main_Fig1_E9_Crossfit_OccAte.png`：本基座柱图",
              "- `Main_Fig2-6_E9_Crossfit_Teaser_action{03,04,05,08,09}.png`：轨迹 teaser",
              "- `figures/`：全量逐会话 PNG",
              "- `Full_Fig2_E9_Crossfit_PerSessionATE.png`：逐会话 ATE"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def fig_combined(mean_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, base in zip(axes, ["V7_Cv5", "V8_Cv5"]):
        rr = {r["action"]: r for r in mean_rows if r["base"] == base}
        acts = sorted(rr)
        x = np.arange(len(acts)); width = 0.15
        for k, arm in enumerate(ARMS):
            vals = [float(rr[a][arm + "_mean"]) for a in acts]
            errs = [float(rr[a][arm + "_std"]) for a in acts]
            ax.bar(x + (k - 2.0) * width, vals, width, color=COLORS[arm],
                   label=arm if base == "V7_Cv5" else None, yerr=errs, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(acts)
        ax.set_title("%s — four-fold mean±std" % base)
        ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    axes[0].set_ylabel("occlusion ATE (mm)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def report_combined(mean_rows, path):
    lines = ["# E9 四折 crossfit 场景遮挡轨迹（V7_Cv5 / V8_Cv5，2026-11-09）", "",
             "- 四折 LSO woS13；V7/V8 各自根目录；本文件为双基座合并参考。",
             "- H+VT0.50/0.75 已按 2026-11-09 裁定移除（Z 字抖动）。", "",
             "| 基座 | 动作 | " + " | ".join(ARMS) + " |",
             "|---|---|" + "---|" * len(ARMS)]
    for r in mean_rows:
        lines.append("| %s | %s | %s |" % (r["base"], r["action"], " | ".join(
            "%.1f±%.1f" % (float(r[a + "_mean"]), float(r[a + "_std"]))
            for a in ARMS)))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    t1 = read(os.path.join(SRC, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"))
    t2 = read(os.path.join(SRC, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"))
    t3 = read(os.path.join(SRC, "Full_Table1_E9_Crossfit_PerSession.csv"))
    # t1 是 fold/base/action 宽表（臂是列），只在 write 里删列；t3 是长表，按 arm 删行
    t3 = [r for r in t3 if r["arm"] not in DROP]
    # combined -> crossbase（始终写过滤后的表/图）
    cb = os.path.join(V7, "crossbase")
    os.makedirs(cb, exist_ok=True)
    write(os.path.join(cb, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"), t1)
    write(os.path.join(cb, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"), t2)
    write(os.path.join(cb, "Full_Table1_E9_Crossfit_PerSession.csv"), t3)
    fig_combined(t2, os.path.join(cb, "Main_Fig1_E9_Crossfit_OccAte.png"))
    report_combined(t2, os.path.join(cb, "E9_四折Crossfit说明.md"))
    for base, root in BASES.items():
        out = os.path.join(root, "V7_Cv5" if base == "V7_Cv5" else "V8_Cv5")
        os.makedirs(os.path.join(out, "detail"), exist_ok=True)
        r1 = [r for r in t1 if r["base"] == base]
        r2 = [r for r in t2 if r["base"] == base]
        r3 = [r for r in t3 if r["base"] == base]
        write(os.path.join(out, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"), r1)
        write(os.path.join(out, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"), r2)
        write(os.path.join(out, "Full_Table1_E9_Crossfit_PerSession.csv"), r3)
        fig_for(r2, base, os.path.join(out, "Main_Fig1_E9_Crossfit_OccAte.png"))
        report_for(r2, base, os.path.join(out, "detail", "E9_四折Crossfit说明.md"))
        print("wrote", out, len(r1), len(r2), len(r3))


if __name__ == "__main__":
    main()
