#!/usr/bin/env python3
"""E5 tables for V8_IMU_Residual from the formal V2 eval (2026-11-15)."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual"
OLD = REPO / "results_display/Etest_crossfit_v8"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
TBI_PATH = (REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
            / "V8_IMU_Residual_TBiFoot_yaw_all.json")
TBI_ALL = json.loads(TBI_PATH.read_text()) if TBI_PATH.is_file() else {}
TBI = TBI_ALL.get("clean")
TBI_SCEN = TBI_ALL
MODE_KEY = {"V2M": "V2M", "T2M": "T2M", "VT2M": "VT2M"}
METRICS = ("PA-MPJPE", "PA-MPJPE_anklefoot", "MPJPE", "MPJPE_anklefoot", "MPJRE",
           "GlobalMPJPE", "W-MPJPE100", "RootOrientationDrift", "RootAbs",
           "FootSliding（GT Anchored）", "GroundPen（GT Anchored）",
           "traj_ATE", "Jitter_pred")

BASE_ROOT = REPO / "results/baselines"
# 档位 -> each baseline model's predictions subdir name
LEVEL_TO_BASELINE_DIR = {
    **{("occ%03d" % i): ("e5_clean" if i == 0 else "e5_occ_lower%03d" % i)
       for i in range(0, 91, 10)},
    **{("gap%03d" % i): ("e5_clean" if i == 0 else "e5_gap_r%03d" % i)
       for i in (0, 10, 20, 40, 60)},
    "scene_clean": "e5_scene_clean", "scene": "e5_scene",
}

# 定位信息分组：自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
LOC_INFO = {
    "GVHMR @V": "测试集校准（世界系注册拟合平移）",
    "WHAM @V": "测试集校准（世界系注册拟合平移）",
    "MotionPRO @VT": "动捕定位辅助（输入含真实定位）",
    "V8_IMU_Residual @V": "自主定位",
    "V8_IMU_Residual @VT": "自主定位",
    "V8_IMU_Residual @T": "GT 初始化（首帧位置锚）",
    "V8_IMU_Residual": "",  # Main_Table1 的 V2M/T2M/VT2M 行
}
V8_LOC = {"V2M": "自主定位", "T2M": "GT 初始化（首帧位置锚）", "VT2M": "自主定位"}
GROUP_ORDER = {"自主定位": 0, "GT 初始化": 1, "测试集校准": 2, "动捕定位辅助": 3}

# Q10 统一四折（2026-11-19）：GVHMR 基线行改用钉定帧 + canonical solver 的
# 四折 mean±std，不再沿用旧 pooled（session 全有效帧 + 20 窗尾裁）。
GVHMR_Q10_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "ankle_foot_mpjpe": "ankle_foot_mpjpe",
    "MPJPE": "mpjpe_mm",
    "MPJRE": "mpjae_deg",
    "GlobalMPJPE": "global_mpjpe_mm",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "根轨迹误差 (mm)": "root_ate_mm",
}


def wham_raw_row(level):
    """WHAM 行的 E5 预测值。

    2026-11-24 口径规范：正式曲线/表不使用 WHAM_OFFSETS 分数常数平移，
    且统计方式与其它基线统一为四折（每折分别统计再汇总四折）。
    优先读四折版（`e5_ft_<variant>/metrics/fourfold_summary.json`，钉定帧
    + solver + 逐折 b 校准，run_wham_e5_fourfold.py 生成）；四折版未生成
    前回落单次 run（97 会话 pooled）并保留待办。
    """
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return None
    # 零样本四折版（2026-11-24，eval_zs/fourfold_zs：现役单次预测按四折
    # split 分组逐折评估，钉定帧 + solver，与 GVHMR Q10 同协议；clean 与
    # scene_clean 的逐会话 npz 未保存，回落单次 pooled 并保留待办）。
    ff = (BASE_ROOT / "WHAM" / "predictions" / level_dir
          / "metrics" / "fourfold_summary.json")
    row = {"模型": "WHAM @V", "档位": level, "行": "@V",
           "定位信息": LOC_INFO["WHAM @V"]}
    if ff.is_file():
        d = json.loads(ff.read_text(encoding="utf-8"))
        for col, key in (("PA-MPJPE", "pa_mpjpe_mm"), ("ankle_foot_mpjpe", "ankle_foot_mpjpe"),
                         ("MPJPE", "mpjpe_mm"), ("MPJRE", "mpjae_deg"),
                         ("GlobalMPJPE", "global_mpjpe_mm"), ("FootSliding", "foot_sliding_mm"),
                         ("GroundPen", "ground_pen_mm"), ("根轨迹误差 (mm)", "root_ate_mm")):
            entry = d.get(key)
            row[col] = ("%.3f±%.3f" % (entry["mean"], entry["std"])
                        if entry and entry.get("n_folds", 0) >= 4 else "—")
        row["PA-MPJPE_anklefoot"] = baseline_anklefoot_pa("WHAM", level)
        return row
    p = (BASE_ROOT / "WHAM" / "predictions" / level_dir
         / "eval_motion" / "metrics" / "test_comparison.json")
    if not p.is_file():
        return None
    s = dict(json.loads(p.read_text(encoding="utf-8"))["summary"])
    extra = p.parent / "extra_metrics.json"
    if extra.is_file():
        s.update(json.loads(extra.read_text(encoding="utf-8")))
    for col, key in (("PA-MPJPE", "pa_mpjpe_mm"), ("ankle_foot_mpjpe", "ankle_foot_mpjpe"),
                     ("GlobalMPJPE", "global_mpjpe_mm"), ("FootSliding", "foot_sliding_mm"),
                     ("GroundPen", "ground_pen_mm"), ("根轨迹误差 (mm)", "root_ate_mm")):
        row[col] = "%.3f" % s[key] if s.get(key) is not None else "—"
    row["MPJPE"], row["MPJRE"] = baseline_mpjpe_mpjre("WHAM", level)
    row["PA-MPJPE_anklefoot"] = baseline_anklefoot_pa("WHAM", level)
    return row


def gvhmr_q10_row(level):
    """GVHMR 行的 Q10 统一四折值（q10_unified_fourfold_summary.json）。"""
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return None
    p = (BASE_ROOT / "GVHMR" / "predictions" / level_dir
         / "eval_motion" / "metrics" / "q10_unified_fourfold_summary.json")
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    row = {"模型": "GVHMR @V", "档位": level, "行": "@V",
           "定位信息": LOC_INFO["GVHMR @V"]}
    for col, key in GVHMR_Q10_KEYS.items():
        entry = d.get(key)
        if entry and entry.get("n_folds", 0) >= 4:
            row[col] = "%.3f±%.3f" % (entry["mean"], entry["std"])
        else:
            row[col] = "—"
    side = BASE_ROOT / "GVHMR" / "_e5_anklefoot_mpjre.json"
    row["PA-MPJPE_anklefoot"] = "—"
    if side.is_file():
        rec = json.loads(side.read_text(encoding="utf-8")).get(level_dir, {})
        v = rec.get("pa_mpjpe_anklefoot_mm")
        row["PA-MPJPE_anklefoot"] = "%.3f" % v if v is not None else "—"
        # q10 四折 JSON 仅 clean 档含 mpjae_deg；其余档位 fallback 侧文件旧值。
        if row["MPJRE"] == "—":
            v = rec.get("mpjae_deg")
            row["MPJRE"] = "%.3f" % v if v is not None else "—"
    return row


def gvhmr_ft_row(level):
    """GVHMR (FT) 行的 Q10 统一四折值（q10 统一评估对四折微调 ckpt 预测，
    run_q10_gvhmr_ft_unified.sh 生成，2026-11-24）。"""
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return None
    ft_tag = level_dir[3:] if level_dir.startswith("e5_") else level_dir
    p = (REPO / "results" / "diagnostics" / "20261119_q10_unified_eval"
         / "GVHMR" / f"ft_{ft_tag}" / "pinned_fold" / "fourfold_summary.json")
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    row = {"模型": "GVHMR @V (FT)", "档位": level, "行": "@V",
           "定位信息": "四折微调 + 测试集校准"}
    for col, key in GVHMR_Q10_KEYS.items():
        entry = d.get(key)
        if entry and entry.get("n_folds", 0) >= 4:
            row[col] = "%.3f±%.3f" % (entry["mean"], entry["std"])
        else:
            row[col] = "—"
    entry = d.get("ankle_foot_mpjpe")
    row["PA-MPJPE_anklefoot"] = ("%.3f±%.3f" % (entry["mean"], entry["std"])
                                 if entry and entry.get("n_folds", 0) >= 4 else "—")
    return row


def wham_ft_row(level):
    """WHAM (FT) 行的四折汇总（run_wham_e5_fourfold.py --stage fourfold，
    2026-11-24；occ060-080 三档逐折评估未补齐时返回 None；clean 档暂无
    四折汇总）。"""
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return None
    ft_tag = level_dir[3:] if level_dir.startswith("e5_") else level_dir
    ff = (BASE_ROOT / "WHAM" / "predictions" / f"e5_ft_{ft_tag}"
          / "metrics" / "fourfold_summary.json")
    if not ff.is_file():
        return None
    d = json.loads(ff.read_text(encoding="utf-8"))
    row = {"模型": "WHAM @V (FT)", "档位": level, "行": "@V",
           "定位信息": "四折微调 + 测试集校准"}
    for col, key in (("PA-MPJPE", "pa_mpjpe_mm"), ("MPJPE", "mpjpe_mm"),
                     ("MPJRE", "mpjae_deg"), ("GlobalMPJPE", "global_mpjpe_mm"),
                     ("FootSliding", "foot_sliding_mm"), ("GroundPen", "ground_pen_mm"),
                     ("根轨迹误差 (mm)", "root_ate_mm")):
        entry = d.get(key)
        row[col] = ("%.3f±%.3f" % (entry["mean"], entry["std"])
                    if entry and entry.get("n_folds", 0) >= 4 else "—")
    entry = d.get("ankle_foot_mpjpe")
    row["PA-MPJPE_anklefoot"] = ("%.3f±%.3f" % (entry["mean"], entry["std"])
                                 if entry and entry.get("n_folds", 0) >= 4 else "—")
    return row


def baseline_mpjpe_mpjre(model, level):
    """MPJPE/MPJRE read directly from each baseline's raw eval json.

    Not present in the legacy Etest_crossfit_v8 MainVsBaselines CSV, so this
    pulls straight from predictions/ instead of going through baseline_rows().
    MotionPRO's fourfold_summary has no MPJRE; use the side file instead
    (2026-11-17，results/baselines/<model>/_e5_anklefoot_mpjre.json — the
    fourfold eval script never passed rotations into the canonical solver).
    """
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return "—", "—"
    if model == "MotionPRO":
        p = BASE_ROOT / model / "predictions" / level_dir / "metrics/fourfold_summary.json"
        if not p.is_file():
            return "—", "—"
        entry = json.loads(p.read_text(encoding="utf-8")).get("mpjpe_mm")
        mpjpe = "%.3f±%.3f" % (entry["mean"], entry["std"]) if entry and entry.get("n_folds", 0) >= 4 else "—"
        side = BASE_ROOT / model / "_e5_anklefoot_mpjre.json"
        mpjre = "—"
        if side.is_file():
            rec = json.loads(side.read_text(encoding="utf-8")).get(level_dir, {}).get("mpjae_deg")
            if rec and rec.get("n_folds", 0) >= 4:
                mpjre = "%.3f±%.3f" % (rec["mean"], rec["std"])
        return mpjpe, mpjre
    p = (BASE_ROOT / model / "predictions" / level_dir
         / "eval_motion/metrics/test_comparison.json")
    if not p.is_file():
        return "—", "—"
    summary = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
    mpjpe = summary.get("mpjpe_mm")
    mpjre = summary.get("mpjae_deg")
    return ("%.3f" % mpjpe if mpjpe is not None else "—",
            "%.3f" % mpjre if mpjre is not None else "—")


def baseline_anklefoot_pa(model, level):
    """PA-MPJPE_anklefoot from the 2026-11-17 side-computation (none of the
    baselines' canonical solvers output a per-part PA-MPJPE)."""
    level_dir = LEVEL_TO_BASELINE_DIR.get(level)
    if level_dir is None:
        return "—"
    side = BASE_ROOT / model / "_e5_anklefoot_mpjre.json"
    if not side.is_file():
        return "—"
    rec = json.loads(side.read_text(encoding="utf-8")).get(level_dir, {})
    if model == "MotionPRO":
        v = rec.get("pa_mpjpe_anklefoot_mm")
        return "%.3f±%.3f" % (v["mean"], v["std"]) if v and v.get("n_folds", 0) >= 4 else "—"
    v = rec.get("pa_mpjpe_anklefoot_mm")
    return "%.3f" % v if v is not None else "—"

OCC = [("occ000", "clean", 0), ("occ010", "occ010", 10),
       ("occ020", "occ020", 20), ("occ030", "occ030", 30),
       ("occ040", "occ040", 40), ("occ050", "occ050", 50),
       ("occ060", "occ060", 60), ("occ070", "occ070", 70),
       ("occ080", "occ080", 80), ("occ090", "occ090", 90)]
SCENE = [("scene_clean", "sceneocc_clean", 0), ("scene", "sceneocc", 1)]
GAP = [("gap000", "clean", 0), ("gap010", "gap010", 10),
       ("gap020", "gap020", 20), ("gap040", "gap040", 40),
       ("gap060", "gap060", 60)]

EVAL_20261124 = REPO / "results/diagnostics/20261124_e5_meanroot_selfphys/V8_IMU_Residual"


def metric(cond, fold, mode, key):
    # 2026-11-24 重评树优先（含 MeanRootPositionError 与自锚物理列），
    # 旧 EV_ALL 仅作回落（条件缺失时）。
    p = EVAL_20261124 / cond / ("%s.json" % fold)
    if not p.is_file():
        p = EVAL / cond / ("%s.json" % fold)
        if not p.is_file():
            return None
    d = json.loads(p.read_text())
    m = d.get("metrics", {}).get(mode, {})
    if key == "RootAbs" and "MeanRootPositionError" in m:
        # 2026-11-23 口径规范：轨迹统一为平均根位置误差（mean）。
        # 当前 JSON 无该字段时回落 RootAbs（RMS），重评后自动切换。
        return m.get("MeanRootPositionError")
    if key == "GroundPen（GT Anchored）" and "GroundPen" in m:
        # 2026-11-24：物理列优先自锚版（与自锚 Global 同一世界轨迹，08 §3.3）；
        # 重评前 JSON 无该字段，回落每窗 GT 锚诊断。
        return m.get("GroundPen")
    if key == "FootSliding（GT Anchored）" and "FootSliding" in m:
        return m.get("FootSliding")
    return m.get(key)

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def fmt(vals):
    m, s = ms(vals)
    return "-" if m is None else "%.3f±%.3f" % (m, s)

def residual_row(cond, mode):
    return {k: fmt([metric(cond, f, mode, k) for f in FOLDS]) for k in METRICS}

def tb_row(cond, tbdata):
    """T channel as one whole: T2M pose + E9 BiFoot trajectory.

    Pose-only metrics (ankle MPJPE, MPJRE, RootOrientationDrift) come from the
    T2M row; trajectory/Global/GroundPen/FootSliding/W/Jitter come from the
    T+BiFoot combination. RootAbs is not defined for this mixed-anchor row.
    """
    out = residual_row(cond, "T2M")
    if tbdata is None:
        out["traj_ATE"] = "—"
        return out
    mapping = {
        "PA-MPJPE": "PA-MPJPE", "MPJPE": "MPJPE",
        "GlobalMPJPE": "GlobalMPJPE", "W-MPJPE100": "W-MPJPE100",
        "traj_ATE": "traj_ATE",
        "FootSliding（GT Anchored）": "FootSliding",
        "GroundPen（GT Anchored）": "GroundPen",
        "Jitter_pred": "Jitter_pred",
        "RootOrientationDrift": "RootOrientationDrift",
        "MPJPE_anklefoot": "MPJPE_anklefoot",
    }
    for k, src in mapping.items():
        out[k] = ("%.3f±%.3f" % (tbdata["mean"][src], tbdata["std"][src])
                  if src in tbdata["mean"] else "—")
    out["RootAbs"] = "—"
    return out

def residual_delta(cond):
    return {k: fmt([(metric(cond, f, "VT2M", k) or 0) - (metric(cond, f, "V2M", k) or 0)
                    for f in FOLDS]) for k in METRICS}

def baseline_rows(csv_path, level):
    out = []
    # GVHMR：Q10 统一四折（不读旧树 CSV）
    g_row = gvhmr_q10_row(level)
    if g_row is not None:
        out.append(g_row)
    # GVHMR (FT)：四折微调 ckpt 预测的 Q10 统一四折
    gft_row = gvhmr_ft_row(level)
    if gft_row is not None:
        out.append(gft_row)
    # WHAM：原始 E5 预测（无分数常数平移）
    w_row = wham_raw_row(level)
    if w_row is not None:
        out.append(w_row)
    # WHAM (FT)：四折微调 ckpt 预测的四折汇总（run_wham_e5_fourfold.py）
    wft_row = wham_ft_row(level)
    if wft_row is not None:
        out.append(wft_row)
    if not csv_path.is_file():
        return out
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if r["档位"] != level:
            continue
        if r["模型"].startswith("V8_") or r["模型"].startswith("GVHMR") \
                or r["模型"].startswith("WHAM"):
            continue
        model = r["模型"].split(" ")[0]
        mpjpe, mpjre = baseline_mpjpe_mpjre(model, level)
        row = {k: r.get(k, "") for k in
               ("模型", "档位", "行", "PA-MPJPE", "ankle_foot_mpjpe",
                "GlobalMPJPE", "FootSliding", "GroundPen", "traj_ATE")}
        row["根轨迹误差 (mm)"] = row.pop("traj_ATE", "")
        row["MPJPE"] = mpjpe
        row["MPJRE"] = mpjre
        row["PA-MPJPE_anklefoot"] = baseline_anklefoot_pa(model, level)
        row["定位信息"] = LOC_INFO.get(row["模型"], "")
        out.append(row)
    return out

def write_table1(path, levels, tbdata=None):
    fields = ["模型", "档位", "行", "定位信息", "PA-MPJPE", "MPJPE", "MPJRE",
              "GlobalMPJPE", "W-MPJPE100", "根轨迹误差 (mm)", "RootOrientationDrift", "RootAbs",
              "GroundPen", "FootSliding", "Jitter_pred"]
    rows = []
    for level, cond, _ in levels:
        block = []
        for mode in ("V2M", "T2M", "VT2M"):
            rec = tb_row(cond, tbdata) if (tbdata is not None and mode == "T2M") else residual_row(cond, mode)
            block.append({"模型": "V8_IMU_Residual", "档位": level, "行": mode,
                          "定位信息": V8_LOC[mode],
                          "PA-MPJPE": rec["PA-MPJPE"], "MPJPE": rec["MPJPE"],
                          "MPJRE": rec["MPJRE"],
                          "GlobalMPJPE": rec["GlobalMPJPE"],
                          "W-MPJPE100": rec["W-MPJPE100"],
                          "根轨迹误差 (mm)": (rec["traj_ATE"] if (tbdata is not None and mode == "T2M")
                                              else "—" if mode == "T2M" else rec["RootAbs"]),
                          "RootOrientationDrift": rec["RootOrientationDrift"],
                          "RootAbs": ("—" if mode == "T2M" else rec["RootAbs"]),
                          "GroundPen": rec["GroundPen（GT Anchored）"],
                          "FootSliding": rec["FootSliding（GT Anchored）"],
                          "Jitter_pred": rec["Jitter_pred"]})
        d = residual_delta(cond)
        block.append({"模型": "V8_IMU_Residual", "档位": level, "行": "Δ(VT−V)",
                      "定位信息": "—",
                      "PA-MPJPE": d["PA-MPJPE"], "MPJPE": d["MPJPE"],
                      "MPJRE": d["MPJRE"], "GlobalMPJPE": d["GlobalMPJPE"],
                      "W-MPJPE100": d["W-MPJPE100"],
                      "根轨迹误差 (mm)": d["RootAbs"],
                      "RootOrientationDrift": d["RootOrientationDrift"],
                      "RootAbs": d["RootAbs"],
                      "GroundPen": d["GroundPen（GT Anchored）"],
                      "FootSliding": d["FootSliding（GT Anchored）"],
                      "Jitter_pred": d["Jitter_pred"]})
        # 自主定位 → GT 初始化（Δ 行排末尾）
        block.sort(key=lambda r: (r["行"] == "Δ(VT−V)",
                                  GROUP_ORDER.get(r["定位信息"].split("（")[0], 9)))
        rows.extend(block)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", path)

def write_table2(path, levels, old_csv, tbdata=None):
    fields = ["模型", "档位", "行", "定位信息", "PA-MPJPE", "PA-MPJPE_anklefoot", "MPJPE",
              "MPJRE", "ankle_foot_mpjpe", "GlobalMPJPE", "FootSliding",
              "GroundPen", "根轨迹误差 (mm)"]
    rows = []
    for level, cond, _ in levels:
        block = []
        for label, mode in (("@VT", "VT2M"), ("@V", "V2M"), ("@T", "T2M")):
            rec = tb_row(cond, tbdata) if (tbdata is not None and label == "@T") else residual_row(cond, mode)
            block.append({"模型": "V8_IMU_Residual " + label, "档位": level,
                          "行": label,
                          "定位信息": LOC_INFO["V8_IMU_Residual " + label],
                          "PA-MPJPE": rec["PA-MPJPE"],
                          "PA-MPJPE_anklefoot": rec["PA-MPJPE_anklefoot"],
                          "MPJPE": rec["MPJPE"],
                          "MPJRE": rec["MPJRE"],
                          "ankle_foot_mpjpe": rec["MPJPE_anklefoot"],
                          "GlobalMPJPE": rec["GlobalMPJPE"],
                          "FootSliding": rec["FootSliding（GT Anchored）"],
                          "GroundPen": rec["GroundPen（GT Anchored）"],
                          "根轨迹误差 (mm)": (rec["traj_ATE"] if (tbdata is not None and label == "@T")
                                       else "—" if label == "@T" else rec["RootAbs"])})
        block.extend(baseline_rows(old_csv, level))
        # 自主定位 → GT 初始化 → 测试集校准 → 动捕定位辅助
        block.sort(key=lambda r: GROUP_ORDER.get(
            (r.get("定位信息") or "").split("（")[0], 9))
        rows.extend(block)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", path)


def write_per_fold(scenario_dir, levels, tag):
    out = OUT / scenario_dir / "detail"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "condition", "fold"] + list(METRICS)
    rows = []
    for _, cond, _ in levels:
        for mode in ("V2M", "T2M", "VT2M"):
            for fold in FOLDS:
                rec = {"mode": mode, "condition": cond, "fold": fold}
                for key in METRICS:
                    rec[key] = metric(cond, fold, mode, key)
                rows.append(rec)
    p = out / ("E5_%s_perFold.csv" % tag)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("wrote", p)

def main():
    specs = [
        ("E5_OcclusionRobustness", OCC, OLD / "E5_OcclusionRobustness/Main_Table2_E5_Occlusion_MainVsBaselines.csv", "Occlusion", TBI),
        ("E5_SceneOcclusion", SCENE, OLD / "E5_SceneOcclusion/Main_Table2_E5_SceneOcclusion_MainVsBaselines.csv", "SceneOcclusion", TBI_SCEN.get("sceneocc")),
        ("E5_TimeGap", GAP, OLD / "E5_TimeGap/Main_Table2_E5_TimeGap_MainVsBaselines.csv", "TimeGap", TBI_SCEN.get("gap010")),
    ]
    for d, levels, old_csv, tag, tbdata in specs:
        out = OUT / d
        out.mkdir(parents=True, exist_ok=True)
        write_table1(out / ("Main_Table1_E5_%s_Models.csv" % tag), levels, tbdata)
        write_table2(out / ("Main_Table2_E5_%s_MainVsBaselines.csv" % tag), levels, old_csv, tbdata)
        write_per_fold(d, levels, tag)

if __name__ == "__main__":
    main()
