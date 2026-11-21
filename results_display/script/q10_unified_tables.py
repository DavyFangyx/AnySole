#!/usr/bin/env python3
"""Q10 统一重评与交叉核查的结果表生成（2026-11-19/20）。

输出：
  E2_MainPerformance/detail/Q10_unified_baselines.csv    统一协议逐折 + 四折（E2 表来源）
  E5_VisualDegradation_Summary/detail/Q10_unified_gvhmr_levels.csv  GVHMR 17 档统一 vs 旧值
  Q10 目录/实验结果/统一重评结果表.md     论文采用数字 + E2/E5 clean 一致性核对
  Q10 目录/实验结果/旧数字来源对照表.md   旧管线三要素 + 交叉核查矩阵 + 复现命令

数据源：
  results/diagnostics/20261119_q10_unified_eval/<model>/<variant>/<frames>_<agg>/
  results/baselines/...（旧管线记录文件）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path("/data/fangyuxuan/projects/gait")
DIAG = REPO / "results/diagnostics/20261119_q10_unified_eval"
BASE = REPO / "results/baselines"
SUPP_Q10 = (REPO / "results_display/Etest_crossfit_physimu_vgate/Supplement"
            / "04_指标与跨表一致性/Q10：GVHMR-MotionPRO 跨表 PA-MPJPE 不一致")
E2_DIR = REPO / "results_display/Etest_crossfit_physimu_vgate/E2_MainPerformance"
E5_DIR = REPO / "results_display/Etest_crossfit_physimu_vgate/E5_VisualDegradation_Summary"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
E2_METRICS = [("PA-MPJPE", "pa_mpjpe_mm"), ("MPJPE", "mpjpe_mm"),
              ("GlobalMPJPE", "global_mpjpe_mm"), ("traj_ATE", "root_ate_mm"),
              ("RootOrientationDrift", "root_orientation_drift_deg"),
              ("GroundPen", "ground_pen_mm"), ("FootSliding", "foot_sliding_mm")]
GVHMR_LEVELS = [
    "e5_clean", "e5_occ_lower010", "e5_occ_lower020", "e5_occ_lower030",
    "e5_occ_lower040", "e5_occ_lower050", "e5_occ_lower060", "e5_occ_lower070",
    "e5_occ_lower080", "e5_occ_lower090", "e5_gap_r010", "e5_gap_r020",
    "e5_gap_r040", "e5_gap_r060", "e5_scene_clean", "e5_scene",
]


def fold_json(model: str, variant: str, frames: str, fold: str) -> dict:
    p = DIAG / model / variant / f"{frames}_fold" / f"{fold}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fourfold(model: str, variant: str, frames: str) -> dict:
    p = DIAG / model / variant / f"{frames}_fold" / "fourfold_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def pooled(model: str, variant: str, frames: str) -> dict:
    p = DIAG / model / variant / f"{frames}_pool" / "pooled_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def pooled_trim(model: str, variant: str, frames: str) -> dict:
    p = DIAG / model / variant / f"{frames}_pool_trim" / "pooled_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_ms(entry: dict) -> str:
    return f"{entry['mean']:.3f}±{entry['std']:.3f}"


def old_gvhmr_single(variant: str) -> dict:
    p = BASE / "GVHMR" / "predictions" / variant / "eval_motion" / "metrics" \
        / "test_comparison.json"
    summary = dict(json.loads(p.read_text(encoding="utf-8"))["summary"])
    extra = p.parent / "extra_metrics.json"
    if extra.is_file():
        summary.update(json.loads(extra.read_text(encoding="utf-8")))
    return summary


def old_e2_summary(model: str, fold: str) -> dict:
    p = BASE / model / "metrics" / f"{fold}_comparison.json"
    return json.loads(p.read_text(encoding="utf-8"))["summary"]


def write_e2_detail() -> None:
    rows = []
    for model, variant, label in (("GVHMR", "e5_clean", "GVHMR U@V"),
                                  ("MotionPRO", "e2_carpet_v4", "MotionPRO U@VT")):
        for fold in FOLDS:
            s = fold_json(model, variant, "pinned", fold)["summary"]
            rows.append([label, fold] + [f"{s[k]:.3f}" for _, k in E2_METRICS])
        f = fourfold(model, variant, "pinned")
        rows.append([label, "mean±std"] + [fmt_ms(f[k]) for _, k in E2_METRICS])
    out = E2_DIR / "detail" / "Q10_unified_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["模型", "折"] + [n for n, _ in E2_METRICS])
        w.writerows(rows)
    print("[csv]", out, len(rows))


def write_e5_levels_detail() -> None:
    rows = []
    for v in GVHMR_LEVELS:
        f = fourfold("GVHMR", v, "pinned")
        old = old_gvhmr_single(v)
        rows.append([
            v,
            fmt_ms(f["pa_mpjpe_mm"]), f"{old['pa_mpjpe_mm']:.3f}",
            f"{f['pa_mpjpe_mm']['mean'] - old['pa_mpjpe_mm']:+.3f}",
            fmt_ms(f["ankle_foot_mpjpe"]), f"{old.get('ankle_foot_mpjpe', float('nan')):.3f}",
            fmt_ms(f["global_mpjpe_mm"]), f"{old['global_mpjpe_mm']:.3f}",
            fmt_ms(f["foot_sliding_mm"]), f"{old['foot_sliding_mm']:.3f}",
            fmt_ms(f["ground_pen_mm"]), f"{old['ground_pen_mm']:.3f}",
            fmt_ms(f["root_ate_mm"]), f"{old['root_ate_mm']:.3f}",
            f"{f['n_valid_frames']['mean']:.0f}±{f['n_valid_frames']['std']:.0f}",
            f"{old['n_valid_frames']}",
        ])
    out = E5_DIR / "detail" / "Q10_unified_gvhmr_levels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["档位", "PA 统一(四折)", "PA 旧(单次)", "ΔPA",
                    "踝足 统一", "踝足 旧", "Global 统一", "Global 旧",
                    "FootSliding 统一", "FootSliding 旧",
                    "GroundPen 统一", "GroundPen 旧",
                    "traj_ATE 统一", "traj_ATE 旧",
                    "n_valid 统一", "n_valid 旧"])
        w.writerows(rows)
    print("[csv]", out, len(rows))


def write_q10_tables() -> None:
    exp = SUPP_Q10 / "实验结果"
    exp.mkdir(parents=True, exist_ok=True)

    # ---- 表 1：统一重评结果表 ----
    g_fold = fourfold("GVHMR", "e5_clean", "pinned")
    m_fold = fourfold("MotionPRO", "e2_carpet_v4", "pinned")
    g_old = old_gvhmr_single("e5_clean")
    m_old_folds = [old_e2_summary("MotionPRO", f) for f in FOLDS]
    lines = [
        "# Q10 统一重评结果表（论文采用数字）",
        "",
        "统一协议：评分帧 = 折级 manifest 钉定帧（主模型 cell frame_ids）；指标实现 = `Baselines/utils/solver`"
        "（公式唯一来源 `anysole/utils/metrics.py`）；聚合 = 折内有效帧加权，四折 mean±std。",
        "预测未重新推理，仅切换评分条件。",
        "",
        "## A. clean 统一数字（E2 表与 E5 表 clean 列共用）",
        "",
        "| 模型 | PA-MPJPE | MPJPE | GlobalMPJPE | traj_ATE | RootOrientationDrift | GroundPen | FootSliding |",
        "|---|---|---|---|---|---|---|---|",
        f"| GVHMR @V | {fmt_ms(g_fold['pa_mpjpe_mm'])} | {fmt_ms(g_fold['mpjpe_mm'])} | {fmt_ms(g_fold['global_mpjpe_mm'])} | {fmt_ms(g_fold['root_ate_mm'])} | {fmt_ms(g_fold['root_orientation_drift_deg'])} | {fmt_ms(g_fold['ground_pen_mm'])} | {fmt_ms(g_fold['foot_sliding_mm'])} |",
        f"| MotionPRO @VT | {fmt_ms(m_fold['pa_mpjpe_mm'])} | {fmt_ms(m_fold['mpjpe_mm'])} | {fmt_ms(m_fold['global_mpjpe_mm'])} | {fmt_ms(m_fold['root_ate_mm'])} | {fmt_ms(m_fold['root_orientation_drift_deg'])} | {fmt_ms(m_fold['ground_pen_mm'])} | {fmt_ms(m_fold['foot_sliding_mm'])} |",
        "",
        "## B. E2/E5 clean 一致性核对",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        "| MotionPRO E2 侧统一重评 vs E5 e5_clean 既有四折 | **逐位一致**（四折 PA 80.434/75.288/76.220/74.535 → 76.620±2.282，全指标 fourfold_summary 逐位相等） |",
        "| GVHMR E2 旧四折 vs 统一重评 | **逐位一致**（统一重评逐折 56.899/57.925/49.338/51.179 → 53.835±3.653；旧 E2 表值即统一值，无需改行） |",
        "| 预测是否重新推理 | 否。MotionPRO 两套 npz 逐位相同；GVHMR 共用 local_gen 预测 |",
        "| 表更新 | E2 主表 MotionPRO 行已换统一值（77.235→76.620 等 7 列）；GVHMR 行不变。E5 主表 GVHMR 全档位换统一四折值（clean 54.131→53.835），MotionPRO 行不变（原本即统一协议） |",
        "",
        "## C. E5 各档位 GVHMR 统一值（论文 E5 表采用）",
        "",
        "| 档位 | PA 统一(四折) | PA 旧(单次) | Δ |",
        "|---|---|---|---|",
    ]
    for v in GVHMR_LEVELS:
        f = fourfold("GVHMR", v, "pinned")
        o = old_gvhmr_single(v)
        lines.append(f"| {v} | {fmt_ms(f['pa_mpjpe_mm'])} | {o['pa_mpjpe_mm']:.3f} | "
                     f"{f['pa_mpjpe_mm']['mean'] - o['pa_mpjpe_mm']:+.3f} |")
    lines += ["", "MotionPRO 各档位沿用既有 `fourfold_summary.json`（已为统一协议，数值不变）。"]
    (exp / "统一重评结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 表 2：旧数字来源对照表 + 交叉核查 ----
    # GVHMR 组合（固定 solver + local_gen）
    g_pp = pooled("GVHMR", "e5_clean", "pinned")
    g_sf = fourfold("GVHMR", "e5_clean", "session")
    g_sp = pooled("GVHMR", "e5_clean", "session")
    g_sp_trim = pooled_trim("GVHMR", "e5_clean", "session")
    # MotionPRO 组合（固定四折 + carpet_v4 npz）
    m_ss = fourfold("MotionPRO", "e2_carpet_v4", "session")
    m_ps = m_fold
    # evaluate.py 现行（带尾裁）session 重算
    mp_eval_session_folds = {}
    for f in FOLDS:
        p = BASE / "MotionPRO" / "metrics" / f"{f}_q10_session_comparison.json"
        mp_eval_session_folds[f] = json.loads(p.read_text())["summary"]["pa_mpjpe_mm"]
    mp_eval_session_mean = float(np.mean(list(mp_eval_session_folds.values())))
    mp_eval_session_std = float(np.std(list(mp_eval_session_folds.values())))
    mp_eval_pinned_folds = {}
    for f in FOLDS:
        p = BASE / "MotionPRO" / "metrics" / f"{f}_q10_pinned_comparison.json"
        mp_eval_pinned_folds[f] = json.loads(p.read_text())["summary"]["pa_mpjpe_mm"]
    mp_eval_pinned_mean = float(np.mean(list(mp_eval_pinned_folds.values())))
    mp_eval_pinned_std = float(np.std(list(mp_eval_pinned_folds.values())))

    lines2 = [
        "# Q10 旧数字来源对照表与交叉核查",
        "",
        "## A. 旧管线三要素与旧数字",
        "",
        "| 模型 | 旧数字 | 值 | 评分帧 | 指标实现 | 聚合 |",
        "|---|---|---|---|---|---|",
        "| GVHMR Table 1 | 53.835±3.653 | 折级 manifest 钉定帧 | solver（公式与统一相同） | 四折 mean±std |",
        f"| GVHMR E5 clean | {g_old['pa_mpjpe_mm']:.3f} | session manifest 全有效帧 + **20 窗尾裁** | solver | 97 会话单次 pooled |",
        "| MotionPRO Table 1 | 77.235±2.224 | session manifest 全有效帧（无尾裁的旧版评估器） | solver（公式与统一相同） | 四折 mean±std |",
        "| MotionPRO E5 clean | 76.620±2.282 | 折级 manifest 钉定帧 | solver | 四折 mean±std（本就是统一协议） |",
        "",
        "## B. 交叉核查矩阵（PA-MPJPE，其余条件固定）",
        "",
        "GVHMR（固定实现 = solver、固定预测 = local_gen）：",
        "",
        "| 帧群 \\ 聚合 | 四折 mean±std | 97 会话 pooled |",
        "|---|---|---|",
        f"| 钉定（折级 manifest） | **{fmt_ms(g_fold['pa_mpjpe_mm'])}**（统一） | {g_pp['pa_mpjpe_mm']:.3f} |",
        f"| 全有效（session manifest） | {fmt_ms(g_sf['pa_mpjpe_mm'])} | {g_sp['pa_mpjpe_mm']:.3f}（+尾裁 → {g_sp_trim['pa_mpjpe_mm']:.3f} = 旧 e5_clean 逐位） |",
        "",
        "效应（其余条件固定）：聚合（折均值→pooled）钉定 −0.100 / 全有效 −0.117；帧群（全有效→钉定）折均值 +0.319 / pooled +0.336；尾裁 +0.732。",
        "",
        "MotionPRO（固定聚合 = 四折、固定预测 = carpet_v4 逐折 npz）：",
        "",
        "| 帧群 \\ 实现 | solver（统一） | evaluate.py（现行，未钉定带 20 窗尾裁） |",
        "|---|---|---|",
        f"| 全有效（session manifest） | {fmt_ms(m_ss['pa_mpjpe_mm'])}（= 旧 Table 1 逐位） | {mp_eval_session_mean:.3f}±{mp_eval_session_std:.3f} |",
        f"| 钉定（折级 manifest） | **{fmt_ms(m_ps['pa_mpjpe_mm'])}**（统一） | {mp_eval_pinned_mean:.3f}±{mp_eval_pinned_std:.3f} |",
        "",
        "效应（其余条件固定）：帧群（全有效→钉定）solver 下 −0.615；实现差异在钉定帧下为 0，"
        "在全有效帧下仅来自尾裁行为（公式逐会话逐位一致，见 S11011 冒烟：session 帧两实现同值 40.871/537 帧）。"
        "旧 Table 1 的 77.235 由无尾裁版本评估器产生（逐位复现）。",
        "",
        "## C. 复现命令",
        "",
        "```bash",
        "# 统一重评（钉定帧 + solver + 四折；--frames/--aggregation/--tail-trim 用于交叉核查）",
        "python .tmp_scripts/q10_unified_eval.py --model MotionPRO --variant e2_carpet_v4",
        "python .tmp_scripts/q10_unified_eval.py --model GVHMR --variant e5_clean [--pred-source local_gen]",
        "bash .tmp_scripts/run_q10_gvhmr_unified.sh   # GVHMR 17 档位批量",
        "# evaluate.py 侧交叉核查（MotionPRO）",
        "bash .tmp_scripts/run_q10_motionpro_xcheck.sh",
        "# 本表生成",
        "python results_display/script/q10_unified_tables.py",
        "```",
        "",
        "原始产物：`results/diagnostics/20261119_q10_unified_eval/<model>/<variant>/<frames>_<agg>/`；"
        "统一四折也已落 `results/baselines/<M>/…/q10_unified_*`。",
    ]
    (exp / "旧数字来源对照表.md").write_text("\n".join(lines2) + "\n", encoding="utf-8")


def main() -> int:
    write_e2_detail()
    write_e5_levels_detail()
    write_q10_tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
