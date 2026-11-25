#!/usr/bin/env python3
"""E2 main-performance table for V8_IMU_Residual + baselines (V2 formal eval)."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual/clean"
MEANROOT = REPO / "results/diagnostics/20261119_e2_meanroot/V8_IMU_Residual/clean"
REV_20261124 = REPO / "results/diagnostics/20261124_e5_meanroot_selfphys/V8_IMU_Residual/clean"
OLD = REPO / "results_display/backup/Etest_crossfit_v8/E2_MainPerformance/Main_Table1_E2_MainPerformance.csv"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate/E2_MainPerformance"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODES = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))
MAIN = (("PA-MPJPE", "PA-MPJPE"), ("MPJPE", "MPJPE"),
        ("GlobalMPJPE", "GlobalMPJPE"), ("RootAbs", "平均根位置误差 (mm)"),
        ("RootOrientationDrift", "RootOrientationDrift"),
        ("GroundPen（GT Anchored）", "GroundPen"),
        ("FootSliding（GT Anchored）", "FootSliding"))
DETAIL = MAIN + (("MPJPE_anklefoot", "MPJPE_anklefoot"), ("MPJRE", "MPJRE"),
                 ("W-MPJPE100", "W-MPJPE100"), ("RootAbs", "RootAbs"),
                 ("Jitter_pred", "Jitter_pred"))

# 定位信息分组：自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
LOC_INFO = {
    "GVHMR": "测试集校准（世界系注册拟合平移）",
    "MMVP (FPP-Net + PoseTransOpt)": "测试集校准（世界系注册拟合平移）",
    "WHAM": "测试集校准（世界系注册拟合平移）",
    "Step2Motion": "GT 初始化（首帧位置单锚）",
    "MotionPRO": "动捕定位辅助（输入含真实定位）",
    "V8_IMU_Residual": "",  # 按条件填写
}
V8_LOC = {"U@V": "自主定位", "U@T": "GT 初始化（首帧位置+yaw 单锚）", "U@VT": "自主定位"}
GROUP_ORDER = {"自主定位": 0, "GT 初始化": 1, "测试集校准": 2, "动捕定位辅助": 3}

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def fmt(vals):
    m, s = ms(vals)
    return "-" if m is None else "%.3f±%.3f" % (m, s)

def load_baselines():
    rows = []
    if not OLD.is_file():
        return rows
    for r in csv.DictReader(open(OLD, encoding="utf-8-sig")):
        if r["模型"].startswith("V8_") or r["模型"].startswith("SAM3DB"):
            continue
        if r.get("条件") not in ("U@V", "U@T", "U@VT"):
            continue
        if r["模型"] == "Step2Motion（position 单锚）":
            r["模型"] = "Step2Motion"
        r["定位信息"] = LOC_INFO.get(r["模型"], "")
        # 旧树列名 traj_ATE 改写到统一表头；基线 root_ate_mm 已是 mean 口径。
        r["平均根位置误差 (mm)"] = r.get("traj_ATE", "")
        rows.append(r)
    return rows


def load_q10_motionpro():
    """MotionPRO U@VT 行改用 Q10 统一四折值（钉定帧 + solver）；旧树行的
    77.235 是 session 全有效帧、无尾裁旧评估器口径，不再沿用。"""
    p = (REPO / "results_display/Etest_crossfit_physimu_vgate"
         / "E2_MainPerformance" / "detail" / "Q10_unified_baselines.csv")
    if not p.is_file():
        return None
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if r.get("模型") == "MotionPRO U@VT" and r.get("折") == "mean±std":
            return r
    return None


def apply_q10_motionpro(rows):
    q10 = load_q10_motionpro()
    if q10 is None:
        return rows
    for r in rows:
        if r["模型"] != "MotionPRO":
            continue
        r["PA-MPJPE"] = q10["PA-MPJPE"]
        r["MPJPE"] = q10["MPJPE"]
        r["GlobalMPJPE"] = q10["GlobalMPJPE"]
        r["平均根位置误差 (mm)"] = q10["traj_ATE"]
        r["RootOrientationDrift"] = q10["RootOrientationDrift"]
        r["GroundPen"] = q10["GroundPen"]
        r["FootSliding"] = q10["FootSliding"]
    return rows


def apply_rerun(rows):
    """MMVP/Step2Motion/WHAM 行改用 2026-11-25 钉定帧重评产物（全部列）：
    折级钉定帧 manifest + canonical solver（solver 已修正物理口径），
    与 GVHMR/MotionPRO 的 Q10 统一协议同款。Step2Motion GroundPen 在评估
    入口为 nan（solver 直调有值，入口差异待查）→ 该列回填历史值并标注。
    """
    import math

    def fourfold(model, key):
        vals = []
        for f in FOLDS:
            p = (REPO / "results/baselines" / model / "metrics"
                 / f"{f}_20261125_phys_rerun_comparison.json")
            if not p.is_file():
                return None
            s = json.loads(p.read_text())["summary"]
            v = s.get(key)
            if v is None or not (isinstance(v, (int, float))):
                return None
            if not math.isfinite(float(v)):
                return None
            vals.append(float(v))
        if len(vals) < 4:
            return None
        return "%.3f±%.3f" % (statistics.mean(vals), statistics.pstdev(vals))

    COL_MAP = (("PA-MPJPE", "pa_mpjpe_mm"), ("MPJPE", "mpjpe_mm"),
               ("GlobalMPJPE", "global_mpjpe_mm"),
               ("平均根位置误差 (mm)", "root_ate_mm"),
               ("Mean Root Position Error (mm)", "root_ate_mm"),
               ("RootOrientationDrift", "root_orientation_drift_deg"),
               ("GroundPen", "ground_pen_mm"), ("FootSliding", "foot_sliding_mm"))

    for r in rows:
        if r["模型"] == "MMVP (FPP-Net + PoseTransOpt)":
            src = lambda key: fourfold("VP-MoCap", key)
        elif r["模型"] == "Step2Motion":
            src = lambda key: fourfold("Step2Motion", key)
        elif r["模型"] == "WHAM":
            p = (REPO / "results/baselines/WHAM/predictions/e5_clean"
                 / "metrics" / "fourfold_summary.json")
            if not p.is_file():
                continue
            d = json.loads(p.read_text())

            def src(key):
                e = d.get(key)
                return ("%.3f±%.3f" % (e["mean"], e["std"])
                        if e and e.get("n_folds", 0) >= 4 else None)
        else:
            continue
        for col, key in COL_MAP:
            v = src(key)
            if v:
                r[col] = v
    return rows


def meanroot_value(mode, fold, key):
    p = MEANROOT / ("%s.json" % fold)
    if not p.is_file():
        return None
    return json.loads(p.read_text())["metrics"].get(mode, {}).get(key)


def eval_value(fold, mode, key):
    """逐折值：20261124 重评树优先（含自锚物理列），EV_ALL 回落。"""
    p = REV_20261124 / ("%s.json" % fold)
    if p.is_file():
        m = json.loads(p.read_text())["metrics"][mode]
    else:
        m = json.loads((SRC / ("%s.json" % fold)).read_text())["metrics"][mode]
    if key == "GroundPen（GT Anchored）" and "GroundPen" in m:
        return m.get("GroundPen")
    if key == "FootSliding（GT Anchored）" and "FootSliding" in m:
        return m.get("FootSliding")
    return m.get(key)


# MAIN/DETAIL 的 JSON key → TBiFoot JSON key（T 通道 = T2M pose + BiFoot 轨迹）
TB_KEY = {"RootAbs": "traj_ATE",
          "GroundPen（GT Anchored）": "GroundPen",
          "FootSliding（GT Anchored）": "FootSliding"}


def tb_key(key):
    return TB_KEY.get(key, key)


def main():
    tb_path = (REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
               / "V8_IMU_Residual_TBiFoot_yaw_all.json")
    tb_all = json.loads(tb_path.read_text()) if tb_path.is_file() else None
    tb = tb_all.get("clean") if tb_all else None
    res_rows = {}
    for mode, cond_label in MODES:
        rec = {"模型": "V8_IMU_Residual", "条件": cond_label, "n": 4,
               "定位信息": V8_LOC[cond_label]}
        for key, label in MAIN:
            tbkey = tb_key(key)
            if mode == "T2M" and tb is not None and tbkey in tb["mean"]:
                # T channel is one whole: T2M pose + E9 BiFoot trajectory.
                rec[label] = "%.3f±%.3f" % (tb["mean"][tbkey], tb["std"][tbkey])
                continue
            if label == "平均根位置误差 (mm)" and mode != "T2M":
                # 2026-11-23 口径规范：轨迹列统一为平均根位置误差（mean），
                # 主模型 V/VT 用 20261119 meanroot 重评（自锚 mean）。
                rec[label] = fmt([meanroot_value(mode, f, "MeanRootPositionError")
                                  for f in FOLDS])
                continue
            rec[label] = fmt([eval_value(f, mode, key) for f in FOLDS])
        res_rows[cond_label] = rec
    base_rows = apply_rerun(apply_q10_motionpro(load_baselines()))
    rows = []
    for cond_label in ("U@V", "U@T", "U@VT"):
        block = [r for r in base_rows if r.get("条件") == cond_label]
        block.append(res_rows[cond_label])
        # 自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
        block.sort(key=lambda r: GROUP_ORDER.get(
            (r.get("定位信息") or "").split("（")[0], 9))
        rows.extend(block)
    fields = ["模型", "条件", "n", "定位信息"] + [x for _, x in MAIN]
    p = OUT / "Main_Table1_E2_MainPerformance.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", p)

    # detail: per-mode full metrics + per-fold
    detail = []
    for mode, cond_label in MODES:
        rec = {"模型": "V8_IMU_Residual", "条件": cond_label,
               "定位信息": V8_LOC[cond_label]}
        for key, label in DETAIL:
            tbkey = tb_key(key)
            if mode == "T2M" and tb is not None and tbkey in tb["mean"]:
                rec[label] = "%.3f±%.3f" % (tb["mean"][tbkey], tb["std"][tbkey])
                continue
            if label == "平均根位置误差 (mm)" and mode != "T2M":
                rec[label] = fmt([meanroot_value(mode, f, "MeanRootPositionError")
                                  for f in FOLDS])
                continue
            rec[label] = fmt([eval_value(f, mode, key) for f in FOLDS])
        detail.append(rec)
    df = ["模型", "条件", "定位信息"] + [x for _, x in DETAIL]
    dp = OUT / "detail" / "Main_Table1_E2_MainPerformance.md"
    lines = ["# E2 原生通道诊断明细（V8_IMU_Residual，Clean，四折 mean±std）", "",
             "| " + " | ".join(df) + " |", "|" + "---|" * len(df)]
    for r in detail:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in df) + " |")
    dp.write_text("\n".join(lines) + "\n")
    print("wrote", dp)

    per = []
    for mode, cond_label in MODES:
        for f in FOLDS:
            d = json.loads((SRC / ("%s.json" % f)).read_text())["metrics"][mode]
            rec = {"模型": "V8_IMU_Residual", "条件": cond_label, "fold": f,
                   "定位信息": V8_LOC[cond_label]}
            pf = None
            if mode == "T2M" and tb is not None:
                # T 通道 = T2M pose + BiFoot 轨迹；逐折明细与主表同装配，
                # 不混用原生 T2M 根头。
                pf = (tb or {}).get("per_fold", {}).get(f, {})
            for key, label in DETAIL:
                tbkey = tb_key(key)
                if pf is not None and tbkey in pf:
                    rec[label] = pf.get(tbkey)
                    continue
                if label == "平均根位置误差 (mm)" and mode != "T2M":
                    rec[label] = meanroot_value(mode, f, "MeanRootPositionError")
                    continue
                # 与主表同源：20261124 重评树优先（物理列取自锚版）。
                rec[label] = eval_value(f, mode, key)
            per.append(rec)
    pf = ["模型", "条件", "fold", "定位信息"] + [x for _, x in DETAIL]
    pp = OUT / "detail" / "E2_MainPerformance_perFold.csv"
    with pp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pf)
        w.writeheader()
        for r in per:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in pf})
    print("wrote", pp)

if __name__ == "__main__":
    main()
