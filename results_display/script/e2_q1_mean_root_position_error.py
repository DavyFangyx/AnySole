#!/usr/bin/env python3
"""Q1 supplement: E2 main performance with the unified root-trajectory metric.

2026-11-19 user ruling: every row's root-trajectory column is the same
statistic — Mean Root Position Error = (1/T) sum_t ||p_hat_t - p_t||_2 (mm),
zero GT anchor (main model = per-window self anchor; baselines = raw absolute
world; T+BiFoot = shared BiFoot trajectory from the GT first frame).  The old
table mixed the main model's RootAbs (RMSE) with the baselines' root_ate_mm
(mean); this script rebuilds the table with the mean-only column.

Sources
-------
- Main model V/VT: ``results/diagnostics/20261119_e2_meanroot/V8_IMU_Residual/clean``
  (four-fold LSO; ``MeanRootPositionError`` = root_mae_selfanchor_mm).
- Main model T: ``V8_IMU_Residual_TBiFoot_yaw_all.json`` clean
  (``traj_ATE`` in that file is already the mean root error).
- Baselines: ``results_display/Etest_crossfit_v8/E2_MainPerformance`` four-fold
  summary (``root_ate_mm`` = mean absolute root error, smpl24 absolute).

Outputs (under ``results_display/Etest_crossfit_physimu_vgate/Supplement/04_指标与跨表一致性/Q1：轨迹指标不一致/E2``)
------------------------------------------------------------------------------------------------
  Main_Table1_E2_MeanRootPositionError.csv   主表（四折 mean±std）
  detail/E2_PerFold_RootMetrics.csv          主模型逐折 RMS vs mean
  detail/RootMetric_BeforeAfter.csv          旧 RootAbs(RMS) 与新 mean 对照
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/diagnostics/20261119_e2_meanroot/V8_IMU_Residual/clean"
TB = (REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
      / "V8_IMU_Residual_TBiFoot_yaw_all.json")
OLD = REPO / "results_display/backup/Etest_crossfit_v8/E2_MainPerformance/Main_Table1_E2_MainPerformance.csv"
OUT = (REPO / "results_display/Etest_crossfit_physimu_vgate"
       / "Supplement" / "04_指标与跨表一致性"
       / "Q1：轨迹指标不一致" / "E2")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODES = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))

# 输出列 = (JSON key / T 行 key, 表头)
MAIN = (
    ("PA-MPJPE", "PA-MPJPE"),
    ("MPJPE", "MPJPE"),
    ("GlobalMPJPE", "GlobalMPJPE"),
    ("MeanRootPositionError", "Mean Root Position Error (mm)"),
    ("RootOrientationDrift", "RootOrientationDrift"),
    ("GroundPen（GT Anchored）", "GroundPen"),
    ("FootSliding（GT Anchored）", "FootSliding"),
)
T_KEYS = {
    "PA-MPJPE": "PA-MPJPE",
    "MPJPE": "MPJPE",
    "GlobalMPJPE": "GlobalMPJPE",
    "MeanRootPositionError": "traj_ATE",   # BiFoot 的 traj_ATE 已是 mean
    "RootOrientationDrift": "RootOrientationDrift",
    "GroundPen（GT Anchored）": "GroundPen",
    "FootSliding（GT Anchored）": "FootSliding",
}

# 定位信息分组：自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
LOC_INFO = {
    "GVHMR": "测试集校准（世界系注册拟合平移）",
    "MMVP (FPP-Net + PoseTransOpt)": "测试集校准（世界系注册拟合平移）",
    "WHAM": "测试集校准（世界系注册拟合平移）",
    "Step2Motion": "GT 初始化（首帧位置单锚）",
    "MotionPRO": "动捕定位辅助（输入含真实定位）",
    "V8_IMU_Residual": "",
}
V8_LOC = {"U@V": "自主定位", "U@T": "GT 初始化（首帧位置+yaw 单锚）", "U@VT": "自主定位"}
GROUP_ORDER = {"自主定位": 0, "GT 初始化": 1, "测试集校准": 2, "动捕定位辅助": 3}


def ms(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def fmt(values):
    mean, std = ms(values)
    return "-" if mean is None else "%.3f±%.3f" % (mean, std)


def read_cell(fold: str) -> dict:
    # 2026-11-25：物理列与 E2 主表同源——20261124 重评树（自锚版）优先，
    # MeanRootPositionError 与其余列仍从 20261119 meanroot 树读。
    m = json.loads((SRC / ("%s.json" % fold)).read_text(encoding="utf-8"))["metrics"]
    rev = REPO / "results/diagnostics/20261124_e5_meanroot_selfphys/V8_IMU_Residual/clean"
    p = rev / ("%s.json" % fold)
    if p.is_file():
        r = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        for mode in ("V2M", "T2M", "VT2M"):
            for col, key in (("GroundPen（GT Anchored）", "GroundPen"),
                             ("FootSliding（GT Anchored）", "FootSliding")):
                if key in r.get(mode, {}):
                    m.setdefault(mode, {})[col] = r[mode][key]
    return m


def load_tb() -> dict | None:
    if not TB.is_file():
        return None
    return json.loads(TB.read_text(encoding="utf-8")).get("clean")


def load_baselines() -> list[dict]:
    rows = []
    if not OLD.is_file():
        return rows
    for row in csv.DictReader(OLD.open(encoding="utf-8-sig")):
        if row["模型"].startswith("V8_") or row["模型"].startswith("SAM3DB"):
            continue
        if row.get("条件") not in ("U@V", "U@T", "U@VT"):
            continue
        if row["模型"] == "Step2Motion（position 单锚）":
            row["模型"] = "Step2Motion"
        # 基线 root_ate_mm 已是 Mean Root Position Error（逐帧平均），
        # 旧表列名 traj_ATE；改写到统一表头。
        row["Mean Root Position Error (mm)"] = row.get("traj_ATE", "")
        row["定位信息"] = LOC_INFO.get(row["模型"], "")
        rows.append(row)
    return rows


def load_q10_motionpro() -> dict | None:
    """MotionPRO U@VT 行改用 Q10 统一四折值（钉定帧 + solver），不再沿用
    旧树 77.235（session 全有效帧、无尾裁旧评估器）。"""
    p = (REPO / "results_display/Etest_crossfit_physimu_vgate"
         / "E2_MainPerformance" / "detail" / "Q10_unified_baselines.csv")
    if not p.is_file():
        return None
    for row in csv.DictReader(p.open(encoding="utf-8-sig")):
        if row.get("模型") == "MotionPRO U@VT" and row.get("折") == "mean±std":
            return row
    return None


def apply_q10_motionpro(rows: list[dict]) -> list[dict]:
    q10 = load_q10_motionpro()
    if q10 is None:
        return rows
    for row in rows:
        if row["模型"] != "MotionPRO":
            continue
        row["PA-MPJPE"] = q10["PA-MPJPE"]
        row["MPJPE"] = q10["MPJPE"]
        row["GlobalMPJPE"] = q10["GlobalMPJPE"]
        row["Mean Root Position Error (mm)"] = q10["traj_ATE"]
        row["RootOrientationDrift"] = q10["RootOrientationDrift"]
        row["GroundPen"] = q10["GroundPen"]
        row["FootSliding"] = q10["FootSliding"]
        row["定位信息"] = LOC_INFO["MotionPRO"]
    return rows


def build_main_rows(tb: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for mode, label in MODES:
        rec = {"模型": "V8_IMU_Residual", "条件": label, "n": 4,
               "定位信息": V8_LOC[label]}
        for key, header in MAIN:
            if mode == "T2M" and tb is not None:
                tkey = T_KEYS.get(key, key)
                if tkey in tb["mean"]:
                    rec[header] = "%.3f±%.3f" % (tb["mean"][tkey], tb["std"][tkey])
                    continue
            values = [read_cell(f).get(mode, {}).get(key) for f in FOLDS]
            rec[header] = fmt(values)
        out[label] = rec
    return out


def write_main_table(main_rows: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["模型", "条件", "n", "定位信息"] + [header for _, header in MAIN]
    rows = []
    baseline_rows = apply_q10_motionpro(load_baselines())
    # 2026-11-25：MMVP/Step2Motion/WHAM 行与正式 E2 主表同源
    # （钉定帧重评全列 + Q10 统一）。
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("e2_v2_table", str(REPO / "results_display/script/e2_v2_table.py"))
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    baseline_rows = _mod.apply_rerun(baseline_rows)
    for label in ("U@V", "U@T", "U@VT"):
        block = [r for r in baseline_rows if r.get("条件") == label]
        block.append(main_rows[label])
        # 自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
        block.sort(key=lambda r: GROUP_ORDER.get(
            (r.get("定位信息") or "").split("（")[0], 9))
        rows.extend(block)
    path = OUT / "Main_Table1_E2_MeanRootPositionError.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print("wrote", path)


def write_detail() -> None:
    detail = OUT / "detail"
    detail.mkdir(parents=True, exist_ok=True)
    tb = load_tb()
    # 1) per-fold main model: RMS vs mean vs Global
    rows = []
    for mode, label in MODES:
        for fold in FOLDS:
            if mode == "T2M":
                # T 行 = T2M pose + 共享 BiFoot 轨迹；该文件只有 mean 根误差。
                pf = (tb or {}).get("per_fold", {}).get(fold, {})
                block = {
                    "RootAbs": None,
                    "MeanRootPositionError": pf.get("traj_ATE"),
                    "GlobalMPJPE": pf.get("GlobalMPJPE"),
                    "AnchorErr": None,
                }
            else:
                block = read_cell(fold).get(mode, {})
            rows.append({
                "模型": "V8_IMU_Residual",
                "条件": label,
                "fold": fold,
                "RootAbs (RMS)": block.get("RootAbs"),
                "MeanRootPositionError": block.get("MeanRootPositionError"),
                "GlobalMPJPE": block.get("GlobalMPJPE"),
                "AnchorErr": block.get("AnchorErr"),
            })
    path = detail / "E2_PerFold_RootMetrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)

    # 2) before/after four-fold comparison for the two V rows
    rows = []
    for mode, label in (("V2M", "U@V"), ("VT2M", "U@VT")):
        rms = [read_cell(f)[mode].get("RootAbs") for f in FOLDS]
        mean = [read_cell(f)[mode].get("MeanRootPositionError") for f in FOLDS]
        glob = [read_cell(f)[mode].get("GlobalMPJPE") for f in FOLDS]
        rows.append({
            "条件": label,
            "RootAbs (RMS, old)": fmt(rms),
            "Mean Root Position Error (mm, new)": fmt(mean),
            "GlobalMPJPE": fmt(glob),
        })
    path = detail / "RootMetric_BeforeAfter.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


def main() -> None:
    tb = load_tb()
    main_rows = build_main_rows(tb)
    write_main_table(main_rows)
    write_detail()


if __name__ == "__main__":
    main()
