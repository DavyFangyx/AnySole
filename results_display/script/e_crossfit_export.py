#!/usr/bin/env python3
"""协议 A 4 折主表导出（2026-11-05）。

数据源：results/cells/<model>_s1/metrics/{test,val}.json + session_detail_*.json
产物：results_display/Etest_crossfit/
  Full_Table1_Crossfit_Main_test.csv    V/T/VT × 每折 + mean±std
  Full_Table2_Crossfit_Delta_test.csv   Δ(VT−V) × 每折 + mean±std
  Full_Table3_Crossfit_Main_val.csv     val 版主表（单受试者，辅助）
  Full_Table4_Crossfit_SubjectDelta_test.csv  按受试者聚合的 Δ（n=7）

指标列序 = E2_METRICS10（2026-11-05 用户裁定：GroundPen/FootSliding/
RootOrientationDrift 领头，PA 随后；自锚 GlobalMPJPE 优先）。

留出池口径（2026-11-05 用户裁定）：S13 只作 train/val 受试者，不作留出 test
受试者 —— 见 `FOLD_TEST_CELL` 常量上方的理由与代价。test 留出池因此是
S5/S6/S7/S8/S11/S12 六人（S10、S14 本来也从未入 test）。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results_display" / "Etest_crossfit"
CELLS = REPO / "results" / "cells"

# 与 results_display/script/e_results_export.py 保持一致（避免第二套口径）
METRICS = ["GroundPen", "FootSliding", "RootOrientationDrift",
           "PA-MPJPE", "MPJPE", "MPJRE", "W-MPJPE100",
           "GlobalMPJPE", "traj_ATE", "Jitter_pred"]
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
FAMILIES = ("V7_Cv5", "V7_Cv5_tphys", "V7_Cv5_tphysRC", "V7_Cv5SD2",
            "V7_Cv5SD2RT", "V7_Cv4", "V7_Cv2", "V7_Cv6", "V7_Cv7")
CONDS = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))
PUB_SUBJECTS = (5, 6, 7, 8, 10, 11, 12, 13, 14)

# S13 退出留出池（2026-11-05 用户裁定）：S13 采集时鞋垫硬件异常——压力单格峰值
# 915 到不了 clip@1023、只用掉 71% 值域、饱和率严格 0.000%、活跃格占比 z(train)
# = −3.01，底层不可弥合，故 S13 只作 train/val 受试者、不作留出 test 受试者。
# 四折里唯一把 S13 放进 test 的是 `fS6S7S13` → 该折 test 口径改读 `_s13out`
# eval-only 重评格（test 池 S6/S7，49→31 会话；train/val 列与原折逐位一致、
# 复用同一 ckpt_last、零重训）。折标签统一显示为留出 test 受试者 `fS6S7`。
# val 口径仍读原折（`_s13out` 是 eval-only、只评 test，没有 val.json；
# 该折 val = S14，两份 split 的 val 列逐位相同）。
# 口径代价：该折 test 动作覆盖 11/11 → 10/11（动作 08 全库仅 3 条 = S5×2 + S13，
# 本折 S5 在 train）；动作 08 的留出读数由 S5 在 test 的另三折提供。
FOLD_TEST_CELL = {"fS6S7S13": "fS6S7S13_s1_s13out"}
FOLD_LABEL = {"fS6S7S13": "fS6S7"}


def cell_id(fam: str, fold: str, split: str) -> str:
    if split == "test" and fold in FOLD_TEST_CELL:
        return "%s_%s" % (fam, FOLD_TEST_CELL[fold])
    return "%s_%s_s1" % (fam, fold)


def fold_label(fold: str) -> str:
    return FOLD_LABEL.get(fold, fold)


def _strip_anchor(metrics: dict) -> dict:
    """默认（自锚）列优先；吃锚列去后缀后只在缺省时补位。"""
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


def load_metrics(model_id: str, split: str) -> dict | None:
    p = CELLS / model_id / "metrics" / ("%s.json" % split)
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return _strip_anchor(payload.get("metrics") or {})


def fmt(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(f) else round(f, 3)


def mean_std(vals: list[float]):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return round(vals[0], 3)
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def metric_vals(m: dict, cond: str) -> dict[str, float | None]:
    return {k: m.get(cond, {}).get(k) for k in METRICS}


def subject_of(session_id: str) -> int:
    for su in PUB_SUBJECTS:
        prefix = "S%d" % su
        if session_id.startswith(prefix) and len(session_id) == len(prefix) + 3:
            return su
    m = re.match(r"S(\d+)", session_id)
    return int(m.group(1)) if m else -1


def subject_delta(model_id: str, metric_key: str) -> dict[int, float]:
    """按受试者聚合 Δ(VT−V)：session_detail_test.json 的 n_frames 加权均值，
    再对同一受试者出现的多个 fold 取平均。"""
    p = CELLS / model_id / "metrics" / "session_detail_test.json"
    if not p.is_file():
        return {}
    detail = json.loads(p.read_text(encoding="utf-8"))
    conds = detail.get("conditions") or {}
    if "V2M" not in conds or "VT2M" not in conds:
        return {}
    per = {}
    for cond in ("V2M", "VT2M"):
        agg = {}
        for s in conds[cond]:
            su = subject_of(str(s.get("session_id", "")))
            w = int(s.get("n_frames") or 0)
            v = (s.get("metrics") or {}).get(metric_key)
            if su < 0 or w <= 0 or v is None:
                continue
            cur = agg.setdefault(su, [0.0, 0])
            cur[0] += float(v) * w
            cur[1] += w
        per[cond] = {su: a / w for su, (a, w) in agg.items() if w}
    out = {}
    for su in per["V2M"]:
        if su in per["VT2M"]:
            out[su] = per["VT2M"][su] - per["V2M"][su]
    return out


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def export_main(split: str, out_name: str) -> None:
    header = ["模型", "条件", "折", "n"] + METRICS
    rows = []
    for cond, label in CONDS:
        for fam in FAMILIES:
            fold_m = {}
            for fold in FOLDS:
                m = load_metrics(cell_id(fam, fold, split), split)
                if m:
                    fold_m[fold] = m
            for fold in FOLDS:
                m = fold_m.get(fold)
                if not m:
                    rows.append([fam, label, fold_label(fold), 0]
                                + [""] * len(METRICS))
                    continue
                vals = metric_vals(m, cond)
                rows.append([fam, label, fold_label(fold), 1]
                            + [fmt(vals[k]) for k in METRICS])
            if fold_m:
                vals_by_k = {k: [metric_vals(m, cond)[k] for m in fold_m.values()]
                             for k in METRICS}
                rows.append([fam, label, "mean±std", len(fold_m)]
                            + [mean_std(vals_by_k[k]) for k in METRICS])
    write_csv(OUT / out_name, header, rows)


def export_delta(out_name: str) -> None:
    header = ["模型", "折", "n"] + METRICS
    rows = []
    for fam in FAMILIES:
        fold_delta = {}
        for fold in FOLDS:
            m = load_metrics(cell_id(fam, fold, "test"), "test")
            if not m or "V2M" not in m or "VT2M" not in m:
                continue
            fold_delta[fold] = {k: (m["VT2M"].get(k) - m["V2M"].get(k))
                                if m["VT2M"].get(k) is not None and m["V2M"].get(k) is not None
                                else None for k in METRICS}
        for fold in FOLDS:
            d = fold_delta.get(fold)
            if d is None:
                rows.append([fam, fold_label(fold), 0] + [""] * len(METRICS))
            else:
                rows.append([fam, fold_label(fold), 1]
                            + [fmt(d[k]) for k in METRICS])
        if fold_delta:
            rows.append([fam, "mean±std", len(fold_delta)]
                        + [mean_std([fold_delta[f][k] for f in fold_delta])
                           for k in METRICS])
    write_csv(OUT / out_name, header, rows)


def export_subject_delta(out_name: str) -> None:
    header = ["模型", "受试者", "折数", "ΔPA-MPJPE", "ΔGroundPen"]
    rows = []
    for fam in FAMILIES:
        # 每个受试者在不同 fold 的 Δ 先按 fold 平均；session_detail 已是单 fold。
        per_subj = {}
        for fold in FOLDS:
            model_id = cell_id(fam, fold, "test")
            if not (CELLS / model_id / "metrics" / "session_detail_test.json").is_file():
                continue
            dpa = subject_delta(model_id, "pa_mpjpe_mm")
            dgp = subject_delta(model_id, "ground_pen_mm")
            for su, v in dpa.items():
                per_subj.setdefault(su, {"pa": [], "gp": []})["pa"].append(v)
            for su, v in dgp.items():
                per_subj.setdefault(su, {"pa": [], "gp": []})["gp"].append(v)
        for su in sorted(per_subj):
            pa = per_subj[su]["pa"]
            gp = per_subj[su]["gp"]
            n = max(len(pa), len(gp))
            rows.append([fam, "S%d" % su, n,
                         round(float(np.mean(pa)), 3) if pa else "",
                         round(float(np.mean(gp)), 3) if gp else ""])
        if per_subj:
            rows.append([fam, "受试者 mean±std", len(per_subj),
                         mean_std([np.mean(v["pa"]) for v in per_subj.values() if v["pa"]]),
                         mean_std([np.mean(v["gp"]) for v in per_subj.values() if v["gp"]])])
    write_csv(OUT / out_name, header, rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    export_main("test", "Full_Table1_Crossfit_Main_test.csv")
    export_delta("Full_Table2_Crossfit_Delta_test.csv")
    export_main("val", "Full_Table3_Crossfit_Main_val.csv")
    export_subject_delta("Full_Table4_Crossfit_SubjectDelta_test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
