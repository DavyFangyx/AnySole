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
ARMS = ["Step2Motion", "Anysolo-TI", "VT", "Hybrid",
        "Hybrid+VT0.50", "Hybrid+VT0.75"]
COLORS = {"Step2Motion": "#2a78d6", "Anysolo-TI": "#1baf7a",
          "VT": "#8a6d3b", "Hybrid": "#7b52ab",
          "Hybrid+VT0.50": "#c58b2a", "Hybrid+VT0.75": "#d64545"}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fig_for(mean_rows, base, path):
    acts = sorted({r["action"] for r in mean_rows})
    x = np.arange(len(acts)); width = 0.13
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, arm in enumerate(ARMS):
        vals = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_mean"])
                for a in acts]
        errs = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_std"])
                for a in acts]
        ax.bar(x + (k - 2.5) * width, vals, width, color=COLORS[arm], label=arm,
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
             "- VT 是 occluded-V prior，不是视觉证据。", "",
             "## 四折 fold-level mean±std（mm）", "",
             "| 动作 | " + " | ".join(ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for r in mean_rows:
        lines.append("| %s | %s |" % (r["action"], " | ".join(
            "%.1f±%.1f" % (float(r[a + "_mean"]), float(r[a + "_std"]))
            for a in ARMS)))
    lines += ["", "## 读数", "",
              "- H+VT0.50 与纯 Hybrid 的逐动作读数见上表；本表只作 fold-level 汇总。",
              "- S2M 是逐折模型；与 7 会话 S13/S14 eval-only 表不是同一池，禁止跨表比较。",
              "- 四折 std 很大（尤其 05/09），不做显著性主张。", "",
              "## 文件", "",
              "- `Main_Table1_E9_Crossfit_OccAte_FoldAction.csv`：逐折×动作",
              "- `Main_Table2_E9_Crossfit_OccAte_MeanStd.csv`：四折 mean±std",
              "- `Full_Table1_E9_Crossfit_PerSession.csv`：逐会话全臂",
              "- `Main_Fig1_E9_Crossfit_OccAte.png`：本基座柱图"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    t1 = read(os.path.join(SRC, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"))
    t2 = read(os.path.join(SRC, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"))
    t3 = read(os.path.join(SRC, "Full_Table1_E9_Crossfit_PerSession.csv"))
    # combined -> crossbase（只在首次；之后 SRC 就是 crossbase）
    cb = os.path.join(V7, "crossbase")
    os.makedirs(cb, exist_ok=True)
    if not _HAS_CB:
        for fn in ("Main_Table1_E9_Crossfit_OccAte_FoldAction.csv",
                   "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv",
                   "Full_Table1_E9_Crossfit_PerSession.csv",
                   "Main_Fig1_E9_Crossfit_OccAte.png"):
            shutil.copy2(os.path.join(SRC, fn), os.path.join(cb, fn))
        shutil.copy2(os.path.join(SRC, "detail", "E9_四折Crossfit说明.md"),
                     os.path.join(cb, "E9_四折Crossfit说明.md"))
    for base, root in BASES.items():
        out = os.path.join(root, "crossfit")
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
