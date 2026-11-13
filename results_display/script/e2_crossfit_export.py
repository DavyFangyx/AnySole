#!/usr/bin/env python3
"""E2 主性能：四折 LSO woS13（协议 A）+ 外部基线。

T 行后处理（2026-11-13）：所有 V8 模型/消融的 U@T 行 = 各模型 T2M 姿态 +
共享 BiFoot 轨迹（无 TTA/Ramp）。本脚本生成 cell 版 T 行后，必须再跑
``results_display/script/e2_t_bifoot_table_replace.py`` 覆盖。

数据源
------
- 主模型 = ``results/cells/<model>_<fold>_s1/metrics/test.json``。
- 外部基线 = ``results/baselines/<MODEL>/metrics/<fold>_comparison.json``
  （由 ``AnysoleWorkspace/tool/adapters/visual_baselines/run_crossfit.py --phase eval``
  产出；未落盘的折跳过，表内标 —）。

口径（2026-11-08 用户裁定）
--------------------------
主口径 = 四折 LSO woS13，只评 test：

    fS5S6S11  test S5/S6/S11   fS5S11S12  test S5/S11/S12
    fS6S7     test S6/S7       fS5S6S8    test S5/S6/S8

**GlobalMPJPE / W-MPJPE100 用无锚列**（cells 的默认 ``GlobalMPJPE`` /
``W-MPJPE100``），因为外部基线的 ``global_mpjpe_mm`` / ``w_mpjpe100_mm``
是无锚世界系误差；canonical 旧表在 cells 缺默认列时回落到 GT Anchored，
与基线不可比。FootSliding / GroundPen 用吃锚列（与基线 `foot_sliding_mm` /
`ground_pen_mm` 同源）。

产物（``results_display/Etest_crossfit_v8/E2_MainPerformance/``）
--------------------------------------------------------------
  detail/Full_Table1_E2_Crossfit_Main_test.csv   模型×条件×折 + mean±std（十指标）
  Full_Table2_E2_Crossfit_Delta_test.csv  Δ(VT−V) × 折 + mean±std
  Main_Table1_E2_MainPerformance.csv      V/T/VT 三块紧凑表（主模型 + 基线）

用法：
  python results_display/script/e2_crossfit_export.py
  python results_display/script/e2_crossfit_export.py --models V8_Cv5,V8_Cv7
  python results_display/script/e2_crossfit_export.py --with-baselines
  # 重叠 s10 口径（2026-11-12）：读 metrics/overlap_s10/test.json，输出到新根
  python results_display/script/e2_crossfit_export.py --models V8_Cv5 \
      --metrics-subdir overlap_s10 --with-baselines \
      --out-dir results_display/Etest_crossfit_v8_s10/E2_MainPerformance
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
BASE_METRICS = REPO / "results" / "baselines"
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "E2_MainPerformance"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
FOLD_LABEL = {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
              "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"}
CONDS = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))

# 论文主表 = E4 消融五模型。已判失败的实验组（RM/PO/VG/T1）与候选路由
# （Cv8）不进主表；需要时用 --models 显式指定。
DEFAULT_MODELS = ("V8_Cv5", "V8_Cv7", "V8_Cv4", "V8_Cv2", "V8_Cv6")

# 展示指标（十指标，顺序按用户 2026-11-08 裁定：姿态 → 轨迹/全局 → 物理 → 时序）
METRICS = [
    ("PA-MPJPE", "PA-MPJPE"),
    ("MPJPE", "MPJPE"),
    ("MPJRE", "MPJRE"),
    ("GlobalMPJPE", "GlobalMPJPE"),        # 无锚，与基线同源
    ("W-MPJPE100", "W-MPJPE100"),          # 无锚，与基线同源
    # 论文口径：traj_ATE = RootAbs（自锚逐帧绝对根，RMS），不是 cell 的
    # 链式无锚 traj_ATE（root_ate_unanchored_mm）。基线 root_ate_mm 同义。
    ("traj_ATE", "RootAbs"),
    ("RootOrientationDrift", "RootOrientationDrift"),
    ("GroundPen", "GroundPen（GT Anchored）"),
    ("FootSliding", "FootSliding（GT Anchored）"),
    ("Jitter_pred", "Jitter_pred"),
]
MAIN_METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
                "RootOrientationDrift", "GroundPen", "FootSliding")

# 基线 summary key → 本表指标
BASELINE_KEYS = {
    "GroundPen": "ground_pen_mm",
    "FootSliding": "foot_sliding_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "MPJRE": "mpjae_deg",
    "W-MPJPE100": "w_mpjpe100_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "traj_ATE": "root_ate_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
BASELINE_FILES = {
    "GVHMR": "GVHMR",
    "SAM3DB": "SAM3DB",
    "WHAM": "WHAM",
    "Step2Motion": "Step2Motion",
    "MotionPRO": "MotionPRO",
    "MMVP (FPP-Net + PoseTransOpt)": "VP-MoCap",
    "MMVP pressure_toolkit": "pressure_toolkit",
}
BASELINE_BLOCK = {"V2M": "V", "T2M": "T", "VT2M": "VT"}
# 只有 canonical 单折（default splits，test=S13+S14=36）的基线：不能进四折
# mean±std，但 2026-11-08 用户要求暂时以「仅参照」行放进主表并显式标注。
CANONICAL_SINGLE_FOLD = {"MotionPRO": "MotionPRO"}


def cell_path(model: str, fold: str, metrics_subdir: str | None = None) -> Path:
    root = CELLS / ("%s_%s_s1" % (model, fold)) / "metrics"
    if metrics_subdir:
        root = root / metrics_subdir
    return root / "test.json"


def read_cell(model: str, fold: str, metrics_subdir: str | None = None) -> dict | None:
    path = cell_path(model, fold, metrics_subdir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test":
        raise ValueError("split mismatch: %s" % path)
    return payload.get("metrics")


def metric_value(block: dict, key: str) -> float | None:
    v = block.get(key)
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, len(vals)


def fmt(ms: tuple[float | None, float | None, int]) -> str:
    mean, std, n = ms
    if mean is None:
        return "—"
    return "%.3f±%.3f" % (mean, std) if n > 1 else "%.3f" % mean


def build_rows(models: list[str], metrics_subdir: str | None = None) -> tuple[list[list], list[list], dict]:
    """返回 (主表行, Δ表行, 缺失清单)。"""
    main_rows: list[list] = []
    delta_rows: list[list] = []
    missing: dict[str, list[str]] = {}
    for model in models:
        per_fold_metrics: dict[str, dict[str, list[float | None]]] = {
            cond: {name: [] for name, _ in METRICS} for cond, _ in CONDS
        }
        fold_blocks: dict[str, dict] = {}
        for fold in FOLDS:
            payload = read_cell(model, fold, metrics_subdir)
            if payload is None:
                missing.setdefault(model, []).append(fold)
                continue
            fold_blocks[fold] = payload
            for cond, cond_label in CONDS:
                block = payload.get(cond) or {}
                for name, key in METRICS:
                    per_fold_metrics[cond][name].append(metric_value(block, key))
                main_rows.append(["U", model, cond_label, FOLD_LABEL[fold], 1]
                                 + [metric_value(block, key) for _, key in METRICS])
        for cond, cond_label in CONDS:
            row = ["U", model, cond_label, "mean±std", len(fold_blocks)]
            for name, _ in METRICS:
                row.append(fmt(mean_std(per_fold_metrics[cond][name])))
            main_rows.append(row)
        # Δ(VT−V)，逐折 + mean±std
        for fold in FOLDS:
            payload = fold_blocks.get(fold)
            if payload is None:
                continue
            v = payload.get("V2M") or {}
            vt = payload.get("VT2M") or {}
            row = [model, FOLD_LABEL[fold], 1]
            for name, key in METRICS:
                a, b = metric_value(vt, key), metric_value(v, key)
                row.append(round(a - b, 3) if a is not None and b is not None else None)
            delta_rows.append(row)
        row = [model, "mean±std", len(fold_blocks)]
        for name, key in METRICS:
            diffs = []
            for fold in FOLDS:
                payload = fold_blocks.get(fold)
                if payload is None:
                    continue
                a = metric_value(payload.get("VT2M") or {}, key)
                b = metric_value(payload.get("V2M") or {}, key)
                if a is not None and b is not None:
                    diffs.append(a - b)
            row.append(fmt(mean_std(diffs)))
        delta_rows.append(row)
    return main_rows, delta_rows, missing


def baseline_rows(min_folds: int = 4, min_sessions: int = 20) -> list[list]:
    """外部基线行：逐折 comparison.json → mean±std；缺折/部分评估跳过。"""
    rows = []
    for display, folder in BASELINE_FILES.items():
        if display in CANONICAL_SINGLE_FOLD:
            # 2026-11-09: MotionPRO now has a four-fold run; keep the canonical
            # single-fold row only as a fallback until all folds are present.
            available = sum(
                (BASE_METRICS / folder / "metrics" / ("%s_comparison.json" % fold)).is_file()
                for fold in FOLDS)
            if available < min_folds:
                continue
        per_fold = {name: [] for name, _ in METRICS}
        n_folds = 0
        last_mode = ""
        for fold in FOLDS:
            path = BASE_METRICS / folder / "metrics" / ("%s_comparison.json" % fold)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("split") != fold:
                continue
            if int(payload.get("n_sessions") or 0) < min_sessions:
                print("[skip] baseline %s %s: n_sessions=%s < %d (partial eval)"
                      % (display, fold, payload.get("n_sessions"), min_sessions))
                continue
            summary = payload.get("summary") or {}
            n_folds += 1
            last_mode = str(payload.get("mode") or "")
            for name, _ in METRICS:
                v = summary.get(BASELINE_KEYS[name])
                per_fold[name].append(float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None)
        if n_folds < min_folds:
            print("[skip] baseline %s: %d/%d folds (need %d)"
                  % (display, n_folds, len(FOLDS), min_folds))
            continue
        block = BASELINE_BLOCK.get(last_mode)
        cond = {"V": "U@V", "T": "U@T", "VT": "U@VT"}.get(block, "外部基线")
        # 2026-11-10: Step2Motion has no world anchor; GlobalMPJPE / traj_ATE /
        # GroundPen are filled from its own inferred trajectory (the world-frame
        # translation head), rigidly aligned once at the first valid frame to the
        # GT root (position + hip-line yaw) — the same start alignment as the E9
        # Step2Motion arm.  Anchor-invariant metrics are unchanged; the row label
        # makes the trajectory口径 explicit.
        display_out = "Step2Motion（轨迹 GT-start Anchored）" if display == "Step2Motion" else display
        row = ["外部基线", display_out, cond, "mean±std", n_folds]
        row += [fmt(mean_std(per_fold[name])) for name, _ in METRICS]
        rows.append(row)
    return rows


def canonical_baseline_rows() -> list[list]:
    """Canonical 单折基线行（仅参照，显式标注，不参与四折 mean±std）。"""
    rows = []
    for display, folder in CANONICAL_SINGLE_FOLD.items():
        if all((BASE_METRICS / folder / "metrics" / ("%s_comparison.json" % fold)).is_file()
               for fold in FOLDS):
            continue
        path = BASE_METRICS / folder / "metrics" / "test_comparison.json"
        if not path.is_file():
            print("[skip] canonical baseline %s: 无 %s" % (display, path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        row = ["外部基线",
               "%s（canonical 单折 test36，非四折；仅参照）" % display,
               "U@VT", "canonical test36", 1]
        row += [fmt(mean_std([summary.get(BASELINE_KEYS[name])])) for name, _ in METRICS]
        rows.append(row)
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="逗号分隔 cell 家族名")
    ap.add_argument("--with-baselines", action="store_true",
                    help="追加外部基线行（需要 results/baselines/<M>/metrics/<fold>_comparison.json）")
    ap.add_argument("--metrics-subdir", default=None,
                    help="metrics 子目录：默认读 metrics/test.json；"
                    "传 overlap_s10 读 metrics/overlap_s10/test.json")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="输出目录，默认 %(default)s")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)

    main_rows, delta_rows, missing = build_rows(models, args.metrics_subdir)
    header = ["模型类型", "模型", "条件", "折", "n"] + [name for name, _ in METRICS]
    if args.with_baselines:
        for row in baseline_rows() + canonical_baseline_rows():
            main_rows.append(row)
    cond_rank = {"U@V": 0, "U@T": 1, "U@VT": 2}
    fold_rank = {fold: i for i, fold in enumerate(FOLDS)}
    fold_rank["mean±std"] = 9
    main_rows.sort(key=lambda row: (cond_rank.get(row[2], 9),
                                    0 if row[0] == "外部基线" else 1,
                                    row[1], fold_rank.get(row[3], 9)))
    write_csv(out_dir / "detail" / "Full_Table1_E2_Crossfit_Main_test.csv", header, main_rows)
    write_csv(out_dir / "detail" / "Full_Table2_E2_Crossfit_Delta_test.csv",
              ["模型", "折", "n"] + [name for name, _ in METRICS], delta_rows)

    # 紧凑主表（论文 Table 1）：每块 = 外部基线 + 主模型 U，不是全部消融五模型。
    # 五模型完整行留在 Full_Table1，供 E4/Table 2 使用。
    MAIN_MODEL = "V8_Cv5"
    compact = []
    for row in main_rows:
        if row[3] not in ("mean±std", "canonical test36"):
            continue
        if row[1] != MAIN_MODEL and row[0] != "外部基线":
            continue
        compact.append([row[1], row[2], row[4]] +
                       [row[5 + i] for i, (name, _) in enumerate(METRICS)
                        if name in MAIN_METRICS])
    write_csv(out_dir / "Main_Table1_E2_MainPerformance.csv",
              ["模型", "条件", "n"] + [name for name, _ in METRICS if name in MAIN_METRICS],
              compact)
    if missing:
        print("[missing folds]")
        for model, folds in sorted(missing.items()):
            print("  %s: %s" % (model, ",".join(folds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
