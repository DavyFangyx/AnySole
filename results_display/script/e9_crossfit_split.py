"""E9 主评估三臂拆分（2026-11-10）。

主表：Step2Motion / VT / Hybrid（VT 是主模型遮挡前向，Hybrid 是纯 IMU 学习臂）。
Anysolo-TI/I 保留为 supplementary 经典基线；H+VT0.50/0.75 已按 11-09 裁定移除。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/data/fangyuxuan/projects/gait"
V7 = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc"
V8 = ROOT + "/results_display/Etest_crossfit_v8/E9_TrajectorySceneOcc"
_HAS_CB = os.path.isfile(V7 + "/crossbase/Main_Table1_E9_Crossfit_OccAte_FoldAction.csv")
_HAS_FRESH = os.path.isfile(V7 + "/crossfit/Main_Table1_E9_Crossfit_OccAte_FoldAction.csv")
SRC = (V7 + "/crossfit") if _HAS_FRESH else (V7 + "/crossbase" if _HAS_CB else V7 + "/crossfit")
BASES = {"V7_Cv5": V7, "V8_Cv5": V8}
ARMS = ["Step2Motion-VT", "VT", "Hybrid"]
SUPP = ["Step2Motion", "Anysolo-TI", "Anysolo-I"]
COLORS = {"Step2Motion-VT": "#5b7ac2", "VT": "#8a6d3b", "Hybrid": "#7b52ab"}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fields_for(rows, arms):
    keys = list(rows[0].keys())
    out = []
    for k in keys:
        if k in ("fold", "base", "action", "n", "n_folds", "session", "arm",
                 "full_ate_mm", "occ_ate_mm"):
            out.append(k)
        elif any(k.startswith(a) for a in arms):
            out.append(k)
    return out


def write(path, rows, arms):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = fields_for(rows, arms)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])


def fig_for(mean_rows, base, path):
    acts = sorted({r["action"] for r in mean_rows})
    x = np.arange(len(acts)); width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, arm in enumerate(ARMS):
        vals = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_mean"])
                for a in acts]
        errs = [float(next(r for r in mean_rows if r["action"] == a)[arm + "_std"])
                for a in acts]
        ax.bar(x + (k - 1.0) * width, vals, width, color=COLORS[arm], label=arm,
               yerr=errs, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(acts)
    ax.set_ylabel("occlusion ATE (mm)")
    ax.set_title("%s — four-fold fold mean±std" % base)
    ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_combined(mean_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, base in zip(axes, ["V7_Cv5", "V8_Cv5"]):
        rr = {r["action"]: r for r in mean_rows if r["base"] == base}
        acts = sorted(rr)
        x = np.arange(len(acts)); width = 0.22
        for k, arm in enumerate(ARMS):
            vals = [float(rr[a][arm + "_mean"]) for a in acts]
            errs = [float(rr[a][arm + "_std"]) for a in acts]
            ax.bar(x + (k - 1.0) * width, vals, width, color=COLORS[arm],
                   label=arm if base == "V7_Cv5" else None, yerr=errs, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(acts)
        ax.set_title("%s — four-fold mean±std" % base)
        ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    axes[0].set_ylabel("occlusion ATE (mm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def report_for(mean_rows, base, path):
    lines = ["# E9 四折 crossfit 场景遮挡轨迹（%s，2026-11-10）" % base, "",
             "## 口径", "",
             "- 四折 LSO woS13；每折 test 的 scene-occ manifest 会话（03/04/05/08/09）。",
             "- 主表三臂：Step2Motion-VT（逐折 ckpt + 同 VT 前后锚，公平基线）、VT（主模型遮挡前向）、Hybrid（纯 IMU 学习臂）。",
             "- Step2Motion GT-start、Anysolo-TI/I 放 supplementary。",
             "- H+VT0.50/0.75 已按 2026-11-09 裁定移除（Z 字抖动）。",
             "- 指标：遮挡内 ATE（full_event 并集）；四折报 fold-level mean±std，不池化。",
             "- VT 是 occluded-V prior，不是视觉证据；空间遮挡阴影是 cam3 通过柱子的地面楔形。", "",
             "## 四折 fold-level mean±std（mm）", "",
             "| 动作 | " + " | ".join(ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for r in mean_rows:
        lines.append("| %s | %s |" % (r["action"], " | ".join(
            "%.1f±%.1f" % (float(r[a + "_mean"]), float(r[a + "_std"]))
            for a in ARMS)))
    lines += ["", "## 读数", "",
              "- 纯 Hybrid vs S2M-VT / VT 的逐动作读数见上表；S2M-VT 是公平基线（同一 VT 前后锚）。",
              "- 经典 TI / GP-SSM / gate / token 线不再作为方法臂；Anysolo-TI/I 仅在 supplementary。",
              "- 四折 std 很大（尤其 05/09），不做显著性主张。", "",
              "## 文件", "",
              "- `Main_Table1_E9_Crossfit_OccAte_FoldAction.csv`：逐折×动作",
              "- `Main_Table2_E9_Crossfit_OccAte_MeanStd.csv`：四折 mean±std",
              "- `Main_Fig1_E9_Crossfit_OccAte.png`：三臂柱图",
              "- `Main_Fig2-6_E9_Crossfit_Teaser_action{03,04,05,08,09}.png`：轨迹 teaser（空间阴影楔形）",
              "- `Full_Table1_E9_Crossfit_PerSession.csv`：逐会话三臂",
              "- `Full_Fig2_E9_Crossfit_PerSessionATE.png`：逐会话 ATE",
              "- `figures/`：全量逐会话 PNG",
              "- `supplementary/`：Anysolo 经典基线"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def report_combined(mean_rows, path):
    lines = ["# E9 四折 crossfit（V7_Cv5 / V8_Cv5，2026-11-10）", "",
             "- 主表三臂：S2M-VT / VT / Hybrid；S2M GT-start、Anysolo 在 supplementary。",
             "- H+VT0.50/0.75 已移除；S2M-VT 与 Hybrid 共用 VT 前后锚；VT 是主模型遮挡前向。", "",
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
    cb = os.path.join(V7, "crossbase")
    os.makedirs(cb, exist_ok=True)
    write(os.path.join(cb, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"), t1, ARMS)
    write(os.path.join(cb, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"), t2, ARMS)
    write(os.path.join(cb, "Full_Table1_E9_Crossfit_PerSession.csv"),
          [r for r in t3 if r["arm"] in ARMS], ARMS)
    write(os.path.join(cb, "Supplement_Table1_E9_Crossfit_Baselines.csv"), t1, SUPP)
    write(os.path.join(cb, "Supplement_Table2_E9_Crossfit_Baselines_MeanStd.csv"), t2, SUPP)
    write(os.path.join(cb, "Supplement_Table3_E9_Crossfit_Baselines_PerSession.csv"),
          [r for r in t3 if r["arm"] in SUPP], SUPP)
    fig_combined(t2, os.path.join(cb, "Main_Fig1_E9_Crossfit_OccAte.png"))
    report_combined(t2, os.path.join(cb, "E9_四折Crossfit说明.md"))
    for base, root in BASES.items():
        out = os.path.join(root, "V7_Cv5" if base == "V7_Cv5" else "V8_Cv5")
        os.makedirs(os.path.join(out, "detail"), exist_ok=True)
        os.makedirs(os.path.join(out, "supplementary"), exist_ok=True)
        r1 = [r for r in t1 if r["base"] == base]
        r2 = [r for r in t2 if r["base"] == base]
        r3 = [r for r in t3 if r["base"] == base]
        write(os.path.join(out, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"), r1, ARMS)
        write(os.path.join(out, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"), r2, ARMS)
        write(os.path.join(out, "Full_Table1_E9_Crossfit_PerSession.csv"),
              [r for r in r3 if r["arm"] in ARMS], ARMS)
        write(os.path.join(out, "supplementary", "Supplement_Table1_E9_Crossfit_Baselines.csv"),
              r1, SUPP)
        write(os.path.join(out, "supplementary", "Supplement_Table2_E9_Crossfit_Baselines_MeanStd.csv"),
              r2, SUPP)
        write(os.path.join(out, "supplementary", "Supplement_Table3_E9_Crossfit_Baselines_PerSession.csv"),
              [r for r in r3 if r["arm"] in SUPP], SUPP)
        fig_for(r2, base, os.path.join(out, "Main_Fig1_E9_Crossfit_OccAte.png"))
        report_for(r2, base, os.path.join(out, "detail", "E9_四折Crossfit说明.md"))
        print("wrote", out, len(r1), len(r2), len(r3))


if __name__ == "__main__":
    main()
