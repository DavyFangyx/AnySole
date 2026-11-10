#!/usr/bin/env python3
"""V8 系列纯导出（2026-11-08）：E2 / E4 / E5 多指标 / E6-B。

口径：四折 LSO woS13（fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8），只评 test；
每格 = 四折 mean±std。导出族 = 已就绪的 8 族（失败族 T1/VG/ST/VGST/VIG/VUG 排除）。

数据源：results/cells/<fam>_<fold>_s1/metrics/{test.json,session_detail_test.json}
        + <...>_evalocc<level>/metrics/test.json
产物：results_display/Etest_crossfit_v8/
  E2_MainPerformance/Full_Table1–4_E2_Performance_*.csv
  E4_ComponentAblation/Full_Table1_E4_Ablation_AllMetrics.csv
  E5_OcclusionRobustness/Full_Table1_E5_Occlusion.csv
                         Full_Table2_E5_TactileGain.csv
                         Full_Table4_E5_Degradation.csv
  E6_ContactPhaseAnalysis/gate_tables/Full_Table1/2_*.csv
  detail/PerFold_*.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = REPO / "results_display" / "Etest_crossfit_v8"
# 2026-11-08 用户裁定：RM/PO 已判失败，从主表移除；Cv8 是候选路由，不进
# 五模型消融。正文消融集 = Cv5/Cv7/Cv4/Cv2/Cv6。
FAMS = ("V8_Cv5", "V8_Cv4", "V8_Cv2", "V8_Cv6", "V8_Cv7")
# E6 只做 Cv5（门控）vs Cv7（无门控）；单门/dropout/恒 VT 留给 E4 消融。
E6_FAMS = ("V8_Cv5", "V8_Cv7")
REFERENCE = "V8_Cv7"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
LEVELS = ("000", "020", "040", "060", "080", "100")

# 指标列序（2026-11-08 用户裁定）：姿态 → 轨迹/全局 → 物理 → 时序。
METRICS10 = ["PA-MPJPE", "MPJPE", "MPJRE",
             "GlobalMPJPE", "W-MPJPE100", "traj_ATE", "RootOrientationDrift",
             "GroundPen", "FootSliding", "Jitter_pred"]
E2_METRICS10 = METRICS10
E5_TACTILE = [("PA-MPJPE_anklefoot", "↓"), ("PA-MPJPE_l_foot", "↓"),
              ("PA-MPJPE_r_foot", "↓"), ("MPJPE_anklefoot", "↓"),
              ("PA-MPJPE_upper", "↓"),
              ("GroundPenFrac", "↓"), ("ground_pen_foot_mm", "↓"),
              ("traj_ATE", "↓")]
E5_MAIN_METRICS = ["PA-MPJPE", "PA-MPJPE_anklefoot",
                   "GlobalMPJPE", "traj_ATE", "RootOrientationDrift",
                   "GroundPen", "FootSliding"]
COND_ZH = {"V2M": "U@V", "T2M": "U@T", "VT2M": "U@VT"}
METRICS_E6 = {
    "PA-MPJPE": "pa_mpjpe_mm", "MPJPE": "mpjpe_mm", "MPJRE": "mpjae_deg",
    "GlobalMPJPE": "global_mpjpe_mm", "W-MPJPE100": "w_mpjpe100_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "GroundPen": "ground_pen_mm", "FootSliding": "foot_sliding_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
# E6 主表只保留三层各一个代表指标（2026-11-09 用户裁定）。
E6_MAIN_METRICS = ("PA-MPJPE", "GlobalMPJPE", "GroundPen")

# E4 消融表的模型标签（用户裁定：模型名后标注对应消融项）
FAM_LABEL = {
    "V8_Cv5": "V8_Cv5（完整：双门 + 三条件枚举）",
    "V8_Cv7": "V8_Cv7（无门控）",
    "V8_Cv4": "V8_Cv4（单门：姿态/根共享）",
    "V8_Cv2": "V8_Cv2（双门 + 随机模态 dropout）",
    "V8_Cv6": "V8_Cv6（双门 + 始终 VT 训练）",
}


def strip_anchor(metrics: dict) -> dict:
    out = {}
    for cond, block in metrics.items():
        if not isinstance(block, dict):
            out[cond] = block
            continue
        default = {str(k): v for k, v in block.items()
                   if not str(k).endswith(("（GT Anchored）", " (anchored)"))}
        anchored = {}
        for k, v in block.items():
            name = str(k)
            for suffix in ("（GT Anchored）", " (anchored)"):
                if name.endswith(suffix):
                    anchored[name[:-len(suffix)]] = v
                    break
        out[cond] = {**anchored, **default}
    return out


def cell_id(fam: str, fold: str, level: str) -> str:
    base = "%s_%s_s1" % (fam, fold)
    return base if level == "000" else "%s_evalocc%s" % (base, level)


def load_test(fam: str, fold: str, level: str) -> dict | None:
    p = CELLS / cell_id(fam, fold, level) / "metrics" / "test.json"
    if not p.is_file():
        return None
    return strip_anchor(json.loads(p.read_text(encoding="utf-8"))["metrics"])


def mean_std(vals) -> str:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return round(vals[0], 3)
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def fold_metric(fam: str, fold: str, cond: str, key: str, level="000"):
    m = load_test(fam, fold, level)
    if not m:
        return None
    # 2026-11-08 用户裁定：论文的 traj_ATE = RootAbs（自锚逐帧绝对根），
    # 不是 cell 里的链式无锚 traj_ATE（root_ate_unanchored_mm）。
    read_key = "RootAbs" if key == "traj_ATE" else key
    return m.get(cond, {}).get(read_key)


def ms_across(fam: str, cond: str, key: str, level="000") -> str:
    return mean_std([fold_metric(fam, f, cond, key, level) for f in FOLDS])


# ---------------------------------------------------------------- E2
def export_e2() -> None:
    out = OUT / "E2_MainPerformance"
    detail = OUT / "detail"
    header = ["类型", "模型", "口径"] + E2_METRICS10
    by_block = {"V": [], "T": [], "VT": []}
    per_fold = []
    for fam in FAMS:
        for cond, block in (("V2M", "V"), ("T2M", "T"), ("VT2M", "VT")):
            row = ["U", "%s（%s）" % (COND_ZH[cond], fam), "last"]
            row += [ms_across(fam, cond, k) for k in E2_METRICS10]
            by_block[block].append(row)
        for fold in FOLDS:
            for cond, block in (("V2M", "V"), ("T2M", "T"), ("VT2M", "VT")):
                vals = [fold_metric(fam, fold, cond, k) for k in E2_METRICS10]
                per_fold.append([fam, fold, block, "last"]
                                + ["" if v is None else round(float(v), 3)
                                   for v in vals])
    for n, block in enumerate(("V", "T", "VT"), 1):
        write_csv(out / ("Full_Table%d_E2_Performance_%s.csv" % (n, block)),
                  header, by_block[block])
    summary = [["块"] + r for block in ("V", "T", "VT")
               for r in by_block[block]]
    write_csv(out / "Full_Table4_E2_Performance_Summary.csv",
              ["块"] + header, summary)
    write_csv(detail / "PerFold_E2_Performance_V8.csv",
              ["模型", "折", "块", "口径"] + E2_METRICS10, per_fold)


# ---------------------------------------------------------------- E4
def _e4_blocks(fams) -> list[list]:
    """按条件分块（U@V / U@T / U@VT / Δ），块内列给定模型。"""
    header = ["条件", "模型"] + METRICS10
    blocks = {c: [] for c in ("U@V", "U@T", "U@VT", "Δ(VT−V)")}
    for fam in fams:
        label = FAM_LABEL.get(fam, fam)
        for cond in ("V2M", "T2M", "VT2M"):
            blocks[COND_ZH[cond]].append(
                [COND_ZH[cond], label] + [ms_across(fam, cond, k) for k in METRICS10])
        delta = []
        for k in METRICS10:
            vals = []
            for fold in FOLDS:
                vt = fold_metric(fam, fold, "VT2M", k)
                v = fold_metric(fam, fold, "V2M", k)
                if vt is not None and v is not None:
                    vals.append(float(vt) - float(v))
            delta.append(mean_std(vals))
        blocks["Δ(VT−V)"].append(["Δ(VT−V)", label] + delta)
    return [r for cond in ("U@V", "U@T", "U@VT", "Δ(VT−V)") for r in blocks[cond]]


def export_e4() -> None:
    out = OUT / "E4_ComponentAblation"
    detail = OUT / "detail"
    header = ["条件", "模型"] + METRICS10
    # 全模型（补充材料）
    write_csv(out / "Full_Table1_E4_Ablation_AllMetrics.csv", header, _e4_blocks(FAMS))
    # 表 1：门控效果（固定三条件枚举；无门 / 单门 / 双门）
    write_csv(out / "Main_Table1_E4_GateEffect.csv", header,
              _e4_blocks(("V8_Cv7", "V8_Cv4", "V8_Cv5")))
    # 表 2：训练条件 / 数据 dropout（固定双门；枚举 / 随机 dropout / 恒 VT）
    write_csv(out / "Main_Table2_E4_TrainingCondition.csv", header,
              _e4_blocks(("V8_Cv5", "V8_Cv2", "V8_Cv6")))
    per_fold = []
    for fam in FAMS:
        for fold in FOLDS:
            for cond in ("V2M", "T2M", "VT2M"):
                per_fold.append([fam, fold, COND_ZH[cond]]
                                + [round(float(v), 3) if (v := fold_metric(fam, fold, cond, k)) is not None else ""
                                   for k in METRICS10])
    write_csv(detail / "PerFold_E4_Ablation_V8.csv",
              ["模型", "折", "条件"] + METRICS10, per_fold)


# ---------------------------------------------------------------- E5
def export_e5() -> None:
    out = OUT / "E5_OcclusionRobustness"
    detail = OUT / "detail"
    header = ["模型", "口径", "档位", "行"] + METRICS10
    header2 = ["模型", "口径", "档位", "行"] + [k + d for k, d in E5_TACTILE]
    rows, rows2 = [], []
    per_fold = []
    for fam in FAMS:
        for level in LEVELS:
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                rows.append([fam, "last", "occ" + level, row_key]
                            + [ms_across(fam, cond, k, level) for k in METRICS10])
                rows2.append([fam, "last", "occ" + level, row_key]
                             + [ms_across(fam, cond, k, level) for k, _ in E5_TACTILE])
            delta = []
            for k in METRICS10:
                vals = []
                for fold in FOLDS:
                    vt = fold_metric(fam, fold, "VT2M", k, level)
                    v = fold_metric(fam, fold, "V2M", k, level)
                    if vt is not None and v is not None:
                        vals.append(float(vt) - float(v))
                delta.append(mean_std(vals))
            rows.append([fam, "last", "occ" + level, "Δ(VT−V)"] + delta)
            delta2 = []
            for k, _ in E5_TACTILE:
                vals = []
                for fold in FOLDS:
                    vt = fold_metric(fam, fold, "VT2M", k, level)
                    v = fold_metric(fam, fold, "V2M", k, level)
                    if vt is not None and v is not None:
                        vals.append(float(vt) - float(v))
                delta2.append(mean_std(vals))
            rows2.append([fam, "last", "occ" + level, "Δ(VT−V)"] + delta2)
            if level != "000":
                for fold in FOLDS:
                    for cond in ("V2M", "VT2M"):
                        per_fold.append([fam, fold, "occ" + level, cond]
                                        + [round(float(v), 3) if (v := fold_metric(fam, fold, cond, k, level)) is not None else ""
                                           for k in METRICS10])
    write_csv(out / "Full_Table1_E5_Occlusion.csv", header, rows)
    write_csv(out / "Full_Table2_E5_TactileGain.csv", header2, rows2)
    # 退化表：值 − occ000
    h4 = ["模型", "口径", "档位", "行"] + E5_MAIN_METRICS
    rows4 = []
    for fam in FAMS:
        for level in LEVELS:
            if level == "000":
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                cells = []
                for k in E5_MAIN_METRICS:
                    cur = fold_metric(fam, FOLDS[0], cond, k, level)  # values via ms not needed
                    vals = []
                    for fold in FOLDS:
                        a = fold_metric(fam, fold, cond, k, level)
                        b = fold_metric(fam, fold, cond, k, "000")
                        if a is not None and b is not None:
                            vals.append(float(a) - float(b))
                    cells.append(mean_std(vals))
                rows4.append([fam, "last", "occ" + level, row_key] + cells)
            cells = []
            for k in E5_MAIN_METRICS:
                vals = []
                for fold in FOLDS:
                    vt = fold_metric(fam, fold, "VT2M", k, level)
                    v = fold_metric(fam, fold, "V2M", k, level)
                    bvt = fold_metric(fam, fold, "VT2M", k, "000")
                    bv = fold_metric(fam, fold, "V2M", k, "000")
                    if None not in (vt, v, bvt, bv):
                        vals.append((float(vt) - float(bvt)) - (float(v) - float(bv)))
                cells.append(mean_std(vals))
            rows4.append([fam, "last", "occ" + level, "Δ退化(VT−V)"] + cells)
    write_csv(out / "Full_Table4_E5_Degradation.csv", h4, rows4)
    write_csv(detail / "PerFold_E5_Occlusion_V8.csv",
              ["模型", "折", "档位", "行"] + METRICS10, per_fold)


# ---------------------------------------------------------------- E6-B
def decompose(fam: str, fold: str) -> dict | None:
    p = CELLS / cell_id(fam, fold, "000") / "metrics" / "session_detail_test.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["conditions"]
    vt = {e["session_id"]: e["metrics"] for e in d["VT2M"]}
    v = {e["session_id"]: e["metrics"] for e in d["V2M"]}
    n = {e["session_id"]: e["n_frames"] for e in d["V2M"]}
    sids = sorted(set(v) & set(vt))
    out = {}
    for name, key in METRICS_E6.items():
        pairs = [(vt[s].get(key), v[s].get(key)) for s in sids]
        if any(a is None or b is None for a, b in pairs):
            continue
        delta = np.array([a - b for a, b in pairs], dtype=float)
        w = np.array([float(n[s]) for s in sids])
        total = w.sum()
        out[name] = {
            "net": float((w * delta).sum() / total),
            "plus": float((w * np.maximum(delta, 0.0)).sum() / total),
            "minus": float((w * np.minimum(delta, 0.0)).sum() / total),
        }
    return out


def export_e6b() -> None:
    out = OUT / "E6_ContactPhaseAnalysis" / "gate_tables"
    detail = OUT / "detail"
    data = {}
    for fam in E6_FAMS:
        for fold in FOLDS:
            d = decompose(fam, fold)
            if d is not None:
                data[(fam, fold)] = d
    per_fold = []
    for (fam, fold), d in data.items():
        for metric in E6_MAIN_METRICS:
            if metric not in d:
                continue
            x = d[metric]
            per_fold.append([fam, fold, metric, round(x["net"], 3),
                             round(x["plus"], 3), round(x["minus"], 3)])
    write_csv(detail / "PerFold_E6_Crossfit_GateDecomposition_V8.csv",
              ["模型", "折", "指标", "ΔE_net", "ΔE_plus", "ΔE_minus"], per_fold)

    h1 = ["模型", "指标", "ΔE⁺_mean±std(mm)", "抑制率_mean±std(%)", "n_folds"]
    h2 = ["模型", "指标", "ΔE⁻_mean±std(mm)", "保留率_mean±std(%)", "n_folds"]
    rows1, rows2 = [], []
    for fam in E6_FAMS:
        for metric in E6_MAIN_METRICS:
            plus, minus, supp, pres = [], [], [], []
            for fold in FOLDS:
                ref = data.get((REFERENCE, fold), {}).get(metric)
                cand = data.get((fam, fold), {}).get(metric)
                if not ref or not cand:
                    continue
                plus.append(cand["plus"])
                minus.append(cand["minus"])
                if fam != REFERENCE:
                    if abs(ref["plus"]) >= 0.05:
                        supp.append(100.0 * (1.0 - cand["plus"] / ref["plus"]))
                    if abs(ref["minus"]) >= 0.05:
                        pres.append(100.0 * cand["minus"] / ref["minus"])
            rows1.append([fam, metric, mean_std(plus),
                          "" if fam == REFERENCE else mean_std(supp),
                          len(plus)])
            rows2.append([fam, metric, mean_std(minus),
                          "" if fam == REFERENCE else mean_std(pres),
                          len(minus)])
    write_csv(out / "Full_Table1_E6_Crossfit_GateHarmSuppression.csv", h1, rows1)
    write_csv(out / "Full_Table2_E6_Crossfit_GateGainPreservation.csv", h2, rows2)

    # 全指标合并表（补充材料）；主表只保留三层三个代表指标。
    h3 = ["模型", "指标", "ΔE⁺有害(mm)", "抑制率(%)", "ΔE⁻有益(mm)", "保留率(%)", "净Δ(VT−V)"]

    def combined(metrics):
        rows = []
        for fam in E6_FAMS:
            for metric in metrics:
                plus, minus, supp, pres, net = [], [], [], [], []
                for fold in FOLDS:
                    ref = data.get((REFERENCE, fold), {}).get(metric)
                    cand = data.get((fam, fold), {}).get(metric)
                    if not cand:
                        continue
                    plus.append(cand["plus"])
                    minus.append(cand["minus"])
                    net.append(cand["net"])
                    if fam != REFERENCE and ref:
                        if abs(ref["plus"]) >= 0.05:
                            supp.append(100.0 * (1.0 - cand["plus"] / ref["plus"]))
                        if abs(ref["minus"]) >= 0.05:
                            pres.append(100.0 * cand["minus"] / ref["minus"])
                rows.append([FAM_LABEL.get(fam, fam), metric,
                             mean_std(plus),
                             "" if fam == REFERENCE else mean_std(supp),
                             mean_std(minus),
                             "" if fam == REFERENCE else mean_std(pres),
                             mean_std(net)])
        return rows

    write_csv(out / "Main_Table1_E6_Gate_ThreeLayers.csv", h3, combined(E6_MAIN_METRICS))
    # 旧 9 指标 Main 表作废（避免主表口径混乱）
    for stale_name in ("Main_Table2_E6_Gate_AllMetrics.csv",
                       "Full_Table3_E6_Gate_AllMetrics.csv"):
        stale = out / stale_name
        if stale.exists():
            stale.unlink()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    export_e2()
    export_e4()
    export_e5()
    export_e6b()
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
