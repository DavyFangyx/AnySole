#!/usr/bin/env python3
"""E 系列实验结果导出器（2026-10-27 用户裁定：results_display 存展示产物）。

数据源与去向：
- E1：z_note/E_Test/C版本对照/E1_同架构专家对比.md 表格 → E1_.../Full_Table1_E1_Experts.csv
- E2：cells（Cv1–7 last / V7_Cv1_evalbest）+ 基线公共评估产物（results/baselines/*/metrics/test_comparison.json）→ Full_Table1_E2_Performance.csv
- E3：Etest/10_21/e3_multiview.json → Full_Table1_E3_MultiView.csv
- E4：cells（Cv1–7 stat，三条件十指标 + Δ）→ Full_Table1_E4_Ablation_AllMetrics.csv
- E5：主模型 V7_Cv5（last/best × occ000-100 十一档）→ Full_Table1_E5_Occlusion.csv
- E6：E6_接触相位/e6_table.json + PNG → Full_Table1_E6_PhaseGain.csv
  （+ 相位简表 Full_Table1_E6_PhaseGain_Simple.csv，四相位 × 双门/无门）
  + Full_FigN_E6_Phase_{g}.png；--best-sessions → 门控最佳完整会话合并图
  Full_Fig_E6_BestSessions.png + e6_best_sessions.json（滤伤害+保增益双条件）；
  2026-11-03 门控主表 = 根层两张（Full_Table1_E6_GateHarmSuppression.csv
  退化抑制 / Full_Table2_E6_GateGainPreservation.csv 增长保留，三组门控）
- E7：E7_动态输入/e7_table.json + PNG → Full_Table1_E7_DynamicStages.csv + Full_FigN_E7_Stages_{a}.png
- 2026-11-02：E6/E7 产物按位置误差指标子文件夹（ankle_foot_mpjpe/pa_mpjpe）同模式生成

表 = CSV（utf-8-sig）；十指标固定列序与实验计划一致。幂等可重复执行。
用法（touch_gait env 非必需）：python results_display/script/e_results_export.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ETEST = REPO / "results_display" / "Etest"
# 2026-11-02 用户裁定终版：根层 = 论文级主产物（主图 + 主干表 + 各指标表）；
# supplementary/ = 逐运动图与完整表；detail/ = json 源/缓存/报告 md。
E5_DIR = ETEST / "E5_OcclusionRobustness"
E5_SUPP = E5_DIR / "supplementary"
E5_DETAIL = E5_DIR / "detail"
# 2026-11-02 用户裁定：E6/E7 按位置误差指标建子文件夹，同一套表/图模式
# 逐指标生成（ankle_foot_mpjpe = 现行踝足口径；pa_mpjpe = 全身相似对齐）。
E6_BASE = ETEST / "E6_ContactPhaseAnalysis"
E7_BASE = ETEST / "E7_DynamicModalityInputs"
POSITION_METRICS = ("ankle_foot_mpjpe", "pa_mpjpe")
RECEIPTS = ETEST / "receipts"
# z_note/E_Test 已并入 E 目录（2026-10-27）；下方源路径全部指向 E 目录
CELLS = REPO / "results" / "cells"

METRICS10 = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
             "traj_ATE", "RootOrientationDrift", "FootSliding", "GroundPen",
             "Jitter_pred"]
# 表 1（E2 主性能表）指标顺序（2026-11-05 用户裁定）：领头物理/接触量与
# PA，长程 Global 如实报但**不领头**；短程轨迹列 traj_ATE@0.5s 已取消
# （同日裁定：不进入评估指标），轨迹只保留无锚 traj_ATE。
E2_METRICS10 = ["GroundPen", "FootSliding", "RootOrientationDrift",
                "PA-MPJPE", "MPJPE", "MPJRE", "W-MPJPE100",
                "GlobalMPJPE", "traj_ATE", "Jitter_pred"]
# E5 表 2 = 触觉有利指标表（2026-11-02 用户裁定：A 层 cells 已落盘指标，
# 零重算；箭头 = 更优方向，Δ 列同向为正即触觉有帮助）。接触类指标已移出
# （cell 级混两套机制），统一入表 3（Full_Table3_E5_ContactFair.csv）。
E5_TACTILE_METRICS = [("traj_ATE", "↓"), ("GroundPenFrac", "↓"),
                      ("ground_pen_foot_mm", "↓"),
                      ("PA-MPJPE_anklefoot", "↓"), ("PA-MPJPE_l_foot", "↓"),
                      ("PA-MPJPE_r_foot", "↓"), ("MPJPE_anklefoot", "↓"),
                      ("PA-MPJPE_upper", "↓")]
E5_TACTILE_KEYS = [k for k, _ in E5_TACTILE_METRICS]
# E5 遮挡档位（2026-11-02 用户裁定：occ010–100 步长 10，occ100 = 视觉全无；
# occ000 = 无遮挡基线）
E5_OCC_LEVELS = ("010", "020", "030", "040", "050", "060", "070", "080",
                 "090", "100")
E5_OCCS = ("occ000",) + tuple("occ" + o for o in E5_OCC_LEVELS)
# Main_Fig2 / Main_Table2 = 论文主图/主表（2026-11-02 用户裁定：occ 20–90
# 八点 × 7 精选指标；必选 PA-MPJPE/traj_ATE@0.5s/GroundPen/FootSliding，
# 可选 GlobalMPJPE/RootOrientationDrift，另加 PA-MPJPE_anklefoot）
E5_MAIN_LEVELS = ("020", "030", "040", "050", "060", "070", "080", "090")
E5_MAIN_METRICS = ["PA-MPJPE", "traj_ATE", "GroundPen", "FootSliding",
                   "GlobalMPJPE", "RootOrientationDrift", "PA-MPJPE_anklefoot"]
CONTACT_NAMESPACE = "state_smpl24"  # 当前默认线（f6_soft_time_accel_state_smpl24_v1）
EVAL_SPLIT = "test"
CELL_MAP = {}
STRICT_EXPORT = False
SOURCE_RECEIPTS = []


def _namespace_cell(model_id):
    if (CONTACT_NAMESPACE in ("legacy", "state_smpl24", "tafix_smpl24")
            or not model_id.startswith("V7_Cv")
            or "_statefix" in model_id or "_tafix" in model_id):
        # 短名即当前默认线（后缀清理 2026-10-29）；smpl24 命名空间直通
        return model_id
    suffix = {"state_v2": "_statefix", "time_accel_v1": "_tafix"}[CONTACT_NAMESPACE]
    match = re.match(r"^(V7_Cv\d+(?:_SV1|_ST1)?)(.*)$", model_id)
    if not match:
        return model_id
    base, tail = match.groups()
    if not tail:
        tail = "_statistics"
    if tail.startswith("_statistics_evalocc"):
        tail = tail.replace("_statistics_evalocc", "_evalocc", 1)
    return base + suffix + tail


CONDS = ("T2M", "V2M", "VT2M")
COND_ZH = {"T2M": "U@T", "V2M": "U@V", "VT2M": "U@VT"}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def display_name(cell_id: str) -> str:
    """cell id → 展示模型名：V7_Cv5_s1_evalocc010 → V7_Cv5。"""
    for suffix in ("_s1", "_s2", "_s3", "_statistics_evalocc", "_evalbest_occ",
                   "_evalocc", "_evalgap", "_evalbest", "_statistics",
                   "_sel_evalocc", "_sel"):
        if suffix in cell_id:
            return cell_id.split(suffix)[0]
    return cell_id


def dedupe(rows: list[list]) -> list[list]:
    """按整行去重（同 ckpt 不同 cell id 的历史遗留双 id → 一行）。"""
    seen, out = set(), []
    for r in rows:
        key = tuple(str(x) for x in r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _strip_anchor_suffix(metrics):
    """修改 10：吃锚列展示名带（GT Anchored）后缀；导出层按原名读取。

    默认列（无后缀）= 自锚读数；吃锚列去掉后缀后与默认同名时**不覆盖**
    默认值（默认优先）——老文件（无自锚列）则原样保留吃锚值。
    """
    if not isinstance(metrics, dict):
        return metrics
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
        out[cond] = {**anchored, **default}   # 默认（自锚）优先
    return out


def cell_metrics(model_id: str, split=None):
    split = split or EVAL_SPLIT
    key = _namespace_cell(model_id)
    mapped = CELL_MAP.get(key)
    p = Path(mapped) if mapped else CELLS / key / "metrics" / ("%s.json" % split)
    if not p.is_file():
        if STRICT_EXPORT:
            raise FileNotFoundError("Missing required result: " + str(p))
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("split") != split:
        raise ValueError("Result split mismatch: " + str(p))
    expected = {"time_accel_v1": "f6_soft_time_accel_v1",
                "state_v2": "f6_soft_time_accel_state_v2",
                "state_smpl24": "f6_soft_time_accel_state_smpl24_v1",
                "tafix_smpl24": "f6_soft_time_accel_smpl24_v1"}.get(CONTACT_NAMESPACE)
    if expected and model_id.startswith("V7_Cv"):
        if payload.get("contact_method") != expected or not payload.get("contact_label_package_sha256") or not payload.get("checkpoint_sha256"):
            raise ValueError("Result contact/weight provenance mismatch: " + str(p))
    import hashlib
    SOURCE_RECEIPTS.append({"cell": key, "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "split": split, "checkpoint_sha256": payload.get("checkpoint_sha256"),
                            "contact_label_package_sha256": payload.get("contact_label_package_sha256")})
    return _strip_anchor_suffix(payload.get("metrics"))


def metric_row(m: dict, cond: str) -> list:
    return [round(m[cond].get(k, float("nan")), 3) if m[cond].get(k) is not None
            else "" for k in METRICS10]


def seeded_metrics(model_id: str, split=None) -> dict | None:
    """自动扫描 {base} 与 {base}_sN 种子变体 cell，聚合为 mean / mean±std
    （2026-10-29 用户裁定：方案 A——多 seed 存 mean±std、单 seed 存单值；
    seed 1 显式 _s1；_statistics 后缀作废，训练 cell 即评估 cell）。

    扫描在 namespace 映射后的目录名上进行（变体 = mapped_key + _sN，
    另兼容无 _sN 的历史 cell）；聚合只在同一条件内做；值为标量 = 单 seed，
    值为 (mean, std) = 多 seed。
    """
    key = _namespace_cell(model_id)
    # _sN 插在子后缀前（与 conf 层 seed_models 一致）：
    # V7_Cv5_evalocc010 → V7_Cv5_s2_evalocc010
    sub_pos = None
    for sub in ("_evalocc", "_evalgap", "_evalbest", "_precheck", "_sel",
                "_statistics"):
        pos = key.rfind(sub)
        if pos != -1 and pos > 0:
            sub_pos = pos
            break

    def _variant(n):
        if sub_pos is not None:
            return key[:sub_pos] + "_s%d" % n + key[sub_pos:]
        return key + "_s%d" % n

    cands = sorted(
        p.name for p in (CELLS).iterdir()
        if p.name in {key, _variant(1)})
    ms = [cell_metrics(c, split) for c in cands]
    ms = [m for m in ms if m]
    if not ms:
        return None
    if len(ms) == 1:
        return ms[0]
    agg = {}
    for cond in CONDS:
        agg[cond] = {}
        for k in ms[0][cond]:
            vals = [m[cond][k] for m in ms if m[cond].get(k) is not None]
            if not vals:
                agg[cond][k] = None
            elif len(vals) == 1:
                agg[cond][k] = vals[0]
            else:
                agg[cond][k] = (float(np.mean(vals)), float(np.std(vals)))
    return agg


def metric_row(m: dict, cond: str, keys: list[str] = METRICS10,
               key_map: dict[str, str] | None = None) -> list:
    key_map = key_map or {}
    out = []
    for k in keys:
        v = m[cond].get(key_map.get(k, k))
        if v is None:
            out.append("")
        elif isinstance(v, tuple):
            out.append("%.3f±%.3f" % v)
        else:
            out.append(round(float(v), 3))
    return out


def delta_row(m: dict, keys: list[str] = METRICS10) -> list:
    """Δ(VT−V) 行：多 seed 按 seed 对先算 Δ 再聚合（误差传播）。"""
    out = []
    for k in keys:
        vv = m["VT2M"].get(k)
        v = m["V2M"].get(k)
        if vv is None or v is None:
            out.append("")
            continue
        if isinstance(vv, tuple) and isinstance(v, tuple):
            out.append("%.3f±%.3f" % (vv[0] - v[0], (vv[1] ** 2 + v[1] ** 2) ** 0.5))
        elif isinstance(vv, tuple):
            out.append("%.3f±%.3f" % (vv[0] - float(v), vv[1]))
        elif isinstance(v, tuple):
            out.append("%.3f±%.3f" % (float(vv) - v[0], v[1]))
        else:
            out.append(round(float(vv) - float(v), 3))
    return out


# ---------------- E1：专家对比（cells 驱动） ----------------
# 专家条目已随历史清理删除（2026-10-29）；当前 E1 面板 = Cv1–7 统一模型
# 的 U@V / U@T / U@VT（同一训练 cell，_statistics 后缀作废）。
def export_e1(models: list[str]) -> None:
    header = ["家族", "测试设定", "模型"] + METRICS10
    rows = []
    families = models or ["V7_Cv%d" % n for n in range(1, 8)]
    for fam in families:
        u = seeded_metrics(fam + ("_f6_statistics" if CONTACT_NAMESPACE == "legacy" else ""))
        if not u:
            print("[skip] E1 %s: 无 cell 结果" % fam)
            continue
        for cond, setting in (("T2M", "T"), ("V2M", "V"), ("VT2M", "VT")):
            rows.append([fam, setting, "U@" + setting] + metric_row(u, cond))
    write_csv(ETEST / "E1_CleanPerformance" / "Full_Table1_E1_Experts.csv", header, rows)


# ---------------- E2：主性能（cells 驱动 + 基线公共评估产物） ----------------
def _selected(cell_ids: list[str], models: list[str]) -> list[str]:
    return [c for c in cell_ids if not models or any(m in c for m in models)]


BASELINE_METRIC_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "MPJRE": "mpjae_deg",
    "traj_ATE": "root_ate_mm",
    "traj_ATE@0.5s": "root_ate_seg20_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}


def _baseline_rows(keys: list[str] = None) -> dict[str, list[list]]:
    """基线行 = 各基线公共评估产物 test_comparison.json（2026-10-31 裁定：
    数据驱动，不读 md）。mode 决定块；V2T 与无运动指标者不产出行。"""
    keys = keys or E2_METRICS10
    rows_by_block: dict[str, list[list]] = {"V": [], "T": [], "VT": []}
    block_of = {"V2M": "V", "T2M": "T", "VT2M": "VT"}
    for path in sorted((REPO / "results" / "baselines").glob("*/metrics/test_comparison.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = block_of.get(str(payload.get("mode") or ""))
        if not block:
            continue
        s = payload.get("summary") or {}
        name = payload.get("display_name") or path.parents[2].name
        row = ["外部基线", name, "—"]
        for key in (BASELINE_METRIC_KEYS[k] for k in keys):
            v = s.get(key)
            row.append(round(v, 3) if isinstance(v, (int, float)) and np.isfinite(v) else "—")
        rows_by_block[block].append(row)
    return rows_by_block


# E0 参照地板（2026-11-03 修改 01）：读 e0_reference_floors.py 的产物 json，
# 与 _baseline_rows 同一数据驱动路径（读 json、不读 md）。
E0_REFERENCE_JSON = ETEST / "E0_ReferenceFloors" / "detail" / "e0_reference_floors.json"


def _reference_row(keys: list[str]) -> list:
    """参照行 = E0 参照地板「常数均值姿态（train 拟合）」；json 缺失时返回
    空行（不插入）。可定义五枚指标（PA-MPJPE/MPJPE/FootSliding/GroundPen/
    Jitter_pred）外留 —（metric_reasons=not_applicable）。"""
    path = E0_REFERENCE_JSON
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    s = (payload.get("references") or {}).get(
        "constant_pose_train_fit", {}).get("summary") or {}
    if not s:
        return []
    import hashlib
    SOURCE_RECEIPTS.append({"cell": "E0_ReferenceFloors", "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "split": EVAL_SPLIT, "checkpoint_sha256": None,
                            "contact_label_package_sha256": None})
    row = ["参照", "Reference / 常数均值姿态", "—"]
    for key in (BASELINE_METRIC_KEYS[k] for k in keys):
        v = s.get(key)
        row.append(round(v, 3) if isinstance(v, (int, float)) and np.isfinite(v) else "—")
    return row


def export_e2(models: list[str], ckpt: str) -> None:
    """四张表（2026-10-27 用户裁定）：V 块 / T 块 / VT 块 / 汇总。

    模型类型列：外部基线 = 各基线公共评估产物（mode 决定块）；U = 统一模型
    （Cv1–7，Cv6 为 S_VT 专家角色）。旧候选 A/B 不入表（2026-10-30 用户裁定）。
    """
    header = ["类型", "模型", "口径"] + E2_METRICS10
    # 当前线：训练 cell 即评估 cell（V7_CvN → 自动扫 _s1/_s2/_s3）；
    # legacy 线保持 _f6_statistics 旧命名（display 层只读专门测试 cell）
    last_cells = _selected(["V7_Cv%d%s%s" % (n,
                                           "_f6" if CONTACT_NAMESPACE == "legacy" else "",
                                           "_statistics" if CONTACT_NAMESPACE == "legacy" else "")
                              for n in range(1, 8)], models)
    best_cells = _selected(["V7_Cv1_evalbest"], models)
    rows_by_block = {"V": [], "T": [], "VT": []}
    # 表结构（实验计划 E2）：每块先列外部基线，最后一行 U。
    for block, rows in _baseline_rows().items():
        rows_by_block[block].extend(rows)
    # 2026-11-03 修改 01：各条件块首行插入参照行（E0 参照地板，数据驱动）。
    reference = _reference_row(E2_METRICS10)
    if reference:
        for block in rows_by_block:
            rows_by_block[block].insert(0, list(reference))
    for cond, block in (("V2M", "V"), ("T2M", "T"), ("VT2M", "VT")):
        if ckpt != "best":
            for model in last_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["S_VT" if model == "V7_Cv6" else "U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "last"]
                        + metric_row(m, cond, keys=E2_METRICS10))
        if ckpt != "last":
            for model in best_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "best"]
                        + metric_row(m, cond, keys=E2_METRICS10))
    out = ETEST / "E2_MainPerformance"
    for table_no, block in enumerate(("V", "T", "VT"), 1):
        write_csv(out / ("Full_Table%d_E2_Performance_%s.csv" % (table_no, block)),
                  header, dedupe(rows_by_block.get(block, [])))
    summary = []
    for block in ("V", "T", "VT"):
        for r in rows_by_block.get(block, []):
            summary.append([block] + r)
    write_csv(out / "Full_Table4_E2_Performance_Summary.csv",
              ["块"] + header, dedupe(summary))
    # 旧的单表文件已拆分为四张
    legacy = out / "Full_Table1_E2_Performance.csv"
    if legacy.is_file():
        legacy.unlink()
        print("[rm] %s" % legacy)


# ---------------- E4：消融组（cells 驱动，三条件 + Δ） ----------------
def export_e4(models: list[str], ckpt: str) -> None:
    if ckpt != "last":
        raise SystemExit("E4 消融组目前只有 last 口径（Cv 家族 best 口径 cells 未注册）")
    header = ["模型", "条件"] + METRICS10
    rows = []
    # 2026-11-03 修改 01：表首插入参照行（E0 参照地板，数据驱动；参照对
    # 三条件同一值，单行即可判读）。
    reference = _reference_row(METRICS10)
    if reference:
        rows.append([reference[1], "—"] + reference[3:])
    cells = _selected(["V7_Cv%d%s%s" % (n,
                                        "_f6" if CONTACT_NAMESPACE == "legacy" else "",
                                        "_statistics" if CONTACT_NAMESPACE == "legacy" else "")
                       for n in range(1, 8)], models)
    for cell_id in cells:
        m = seeded_metrics(cell_id)
        if not m:
            continue
        for cond in (("VT2M",) if cell_id == "V7_Cv6" else CONDS):
            rows.append([display_name(cell_id), "S_VT@VT" if cell_id == "V7_Cv6" else COND_ZH[cond]] + metric_row(m, cond))
        if cell_id == "V7_Cv6":
            continue
        rows.append([display_name(cell_id), "Δ(VT−V)"] + delta_row(m))
    write_csv(ETEST / "E4_ComponentAblation" / "Full_Table1_E4_Ablation_AllMetrics.csv",
              header, rows)


# ---------------- E5：遮挡（cells 驱动，档×行） ----------------
def export_e5(models: list[str], ckpt: str) -> None:
    header = ["模型", "口径", "档位", "行"] + METRICS10
    rows = []
    # 表 2 行同表 1 结构，列为触觉有利指标（↑=越高越好 / ↓=越低越好；
    # Δ(VT−V) 同向为正 = 触觉有帮助，2026-11-02 用户裁定）
    header2 = ["模型", "口径", "档位", "行"] + \
        [k + d for k, d in E5_TACTILE_METRICS]
    rows2 = []
    # E5 = 单一主模型（2026-10-30 用户裁定）：U = V7_Cv5 一份 ckpt × occ 五档
    model = models[0] if models else "V7_Cv5"
    last_cells = {"occ000": [model]}
    best_cells = {"occ000": ["%s_evalbest" % model]}
    for occ in E5_OCC_LEVELS:
        last_cells["occ%s" % occ] = ["%s_evalocc%s" % (model, occ)]
        best_cells["occ%s" % occ] = ["%s_evalbest_occ%s" % (model, occ)]
    for occ in E5_OCCS:
        last_ids = _selected(last_cells[occ], models)
        best_ids = _selected(best_cells.get(occ, []), models)
        for model in last_ids if ckpt != "best" else []:
            m = seeded_metrics(model)
            if not m:
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                rows.append([display_name(model), "last", occ, row_key]
                            + metric_row(m, cond))
                rows2.append([display_name(model), "last", occ, row_key]
                             + metric_row(m, cond, keys=E5_TACTILE_KEYS))
            rows.append([display_name(model), "last", occ, "Δ(VT−V)"]
                        + delta_row(m))
            rows2.append([display_name(model), "last", occ, "Δ(VT−V)"]
                         + delta_row(m, keys=E5_TACTILE_KEYS))
        for model in best_ids if ckpt != "last" else []:
            m = seeded_metrics(model)
            if not m:
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                rows.append([display_name(model), "best", occ, row_key]
                            + metric_row(m, cond))
                rows2.append([display_name(model), "best", occ, row_key]
                             + metric_row(m, cond, keys=E5_TACTILE_KEYS))
            rows.append([display_name(model), "best", occ, "Δ(VT−V)"]
                        + delta_row(m))
            rows2.append([display_name(model), "best", occ, "Δ(VT−V)"]
                         + delta_row(m, keys=E5_TACTILE_KEYS))
    write_csv(E5_SUPP / "Full_Table1_E5_Occlusion.csv", header, dedupe(rows))
    write_csv(E5_DIR / "Full_Table2_E5_TactileGain.csv", header2, dedupe(rows2))
    # 表 3 = 接触口径统一表（2026-11-02 用户裁定「接触指标口径统一且公平」）：
    # 足高判据行 = V/VT 同机制（e5_contact_kin.json 离线池化，两脚两条件）；
    # 接触头行 = 仅 VT 推理有触觉 token 驱动，V-only 无触觉通道故无接触头。
    # 「V=足高判据 vs VT=接触头」即各条件最强接触感知方式的对比，
    # Δ(头−V足高判据) = 触觉通道本身的贡献；足高判据 Δ(VT−V) ≈ 0 排除
    # 「骨骼修正副产物」质疑。
    # balanced（类别平衡口径）：V 的 acc/f1 在 occ030 有回升假象（脚部全遮
    # → 预测塌缩到地面先验，recall↑/air_recall↓），balanced 单调退化——
    # 趋势判读用 balanced 列。
    kin_path = E5_DETAIL / "e5_contact_kin.json"
    if kin_path.is_file():
        kin = json.loads(kin_path.read_text(encoding="utf-8"))
        header3 = ["档位", "足高判据acc(V)", "足高判据acc(VT)", "Δacc(VT−V)",
                   "足高判据f1(V)", "足高判据f1(VT)", "Δf1(VT−V)",
                   "足高判据balanced(V)", "足高判据balanced(VT)",
                   "Δbalanced(VT−V)", "接触头acc(VT)", "接触头f1(VT)",
                   "接触头balanced(VT)", "Δacc(头−V足高判据)",
                   "Δbalanced(头−V足高判据)"]
        rows3 = []

        def _bal(cell: dict) -> float:
            return 0.5 * (cell["recall"] + cell["air_recall"])

        for occ, ids in last_cells.items():
            m = seeded_metrics(ids[0])
            if not m or occ not in kin:
                continue
            k = kin[occ]
            kv, kvt = k["V2M"], k["VT2M"]
            head_acc = m["VT2M"].get("contact_acc", float("nan"))
            head_f1 = m["VT2M"].get("contact_f1", float("nan"))
            head_bal = m["VT2M"].get("contact_balanced_acc", float("nan"))
            rows3.append([occ,
                          round(kv["acc"], 3), round(kvt["acc"], 3),
                          round(kvt["acc"] - kv["acc"], 3),
                          round(kv["f1"], 3), round(kvt["f1"], 3),
                          round(kvt["f1"] - kv["f1"], 3),
                          round(_bal(kv), 3), round(_bal(kvt), 3),
                          round(_bal(kvt) - _bal(kv), 3),
                          round(head_acc, 3), round(head_f1, 3),
                          round(head_bal, 3),
                          round(head_acc - kv["acc"], 3),
                          round(head_bal - _bal(kv), 3)])
        write_csv(E5_SUPP / "Full_Table3_E5_ContactFair.csv", header3, rows3)
    # 表 4 = 图 3 的对应数值表（2026-11-02 用户裁定：主干表，根层）：
    # 各指标随 occ 的退化 = 值 − occ000 基线；行 V2M / VT2M /
    # Δ退化(VT−V)。Δ退化 为负 = VT 退化更轻 = 触觉有帮助；traj 列 =
    # traj_ATE@0.5s（与图 3 子图口径一致）。
    E5_FIG_METRICS = list(METRICS10)

    def _scalar(x):
        return x[0] if isinstance(x, tuple) else x

    def _degradation_table(metrics: list[str], levels: tuple[str, ...]
                           ) -> tuple[list[str], list[list]]:
        header = ["模型", "口径", "档位", "行"] + list(metrics)
        rows = []
        for occ in ("occ" + o for o in levels):
            m = seeded_metrics(last_cells[occ][0])
            base = seeded_metrics(last_cells["occ000"][0])
            if not m or not base:
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                cells = []
                for k in metrics:
                    v = m[cond].get(k)
                    b = base[cond].get(k)
                    cells.append("" if v is None or b is None
                                 else round(float(_scalar(v)) - float(_scalar(b)), 3))
                rows.append([display_name(last_cells[occ][0]), "last", occ, row_key]
                            + cells)
            deg = []
            for k in metrics:
                vv = m["VT2M"].get(k)
                v = m["V2M"].get(k)
                bv = base["VT2M"].get(k)
                b = base["V2M"].get(k)
                if None in (vv, v, bv, b):
                    deg.append("")
                else:
                    deg.append(round((float(_scalar(vv)) - float(_scalar(bv)))
                                     - (float(_scalar(v)) - float(_scalar(b))), 3))
            rows.append([display_name(last_cells[occ][0]), "last", occ,
                         "Δ退化(VT−V)"] + deg)
        return header, rows

    header4, rows4 = _degradation_table(E5_FIG_METRICS, E5_OCC_LEVELS)
    write_csv(E5_DIR / "Full_Table4_E5_Degradation.csv", header4, rows4)
    main_header, main_rows = _degradation_table(E5_MAIN_METRICS, E5_MAIN_LEVELS)
    write_csv(E5_DIR / "Main_Table2_E5_Degradation.csv", main_header, main_rows)
    export_e5_figure(models)
    export_e5_visual_failure(models)


# E5 视觉退化扩展（2026-11-04，z_note/E5_视觉退化扩展说明.md §3）：
# 同一 ckpt、固定 VT 输入，只替换视觉特征缓存；单元格 = Δ(VT−V)，负值 =
# 触觉有帮助（与 E6 的 ΔE 同号）。空间轴 = 现成 occ 缓存；时间轴 =
# 连续断流（冻结）。光度/曝光轴 2026-11-08 用户裁定整组删除。
E5_GAP_RHOS = ("010", "020", "040", "060", "080")
E5_VISUAL_MAIN_METRICS = ["PA-MPJPE", "MPJPE", "GlobalMPJPE",
                          "FootSliding", "GroundPen"]


def _e5_visual_conditions(model: str) -> list[tuple[str, str, str, str]]:
    """(轴, 档位, 条件说明, cell id) — 总表行序 = 设计文档表序。"""
    conditions = [("空间", "occ000", "干净", model)]
    for occ in ("010", "020", "040", "060", "080"):
        conditions.append(("空间", "occ" + occ, "下肢截断 %d%%" % int(occ),
                           "%s_evalocc%s" % (model, occ)))
    for rho in E5_GAP_RHOS:
        conditions.append(("时间", "gap_r" + rho, "连续断流 ρ=%d%%" % int(rho),
                           "%s_evalgap%s" % (model, rho)))
    return conditions


def export_e5_visual_failure(models: list[str]) -> None:
    """E5 扩展总表（Main_Table3）+ Full 表（十指标 × V / VT / Δ）。

    Full 表编号顺延为 Table5：Table4 已被 Main/Full_Table4_E5_Degradation
    占用（设计文档 §3 注明文件名可改）。
    """
    model = models[0] if models else "V7_Cv5"
    conditions = _e5_visual_conditions(model)
    header = ["模型", "口径", "轴", "档位", "条件"] + E5_VISUAL_MAIN_METRICS
    header_full = ["模型", "口径", "轴", "档位", "条件", "行"] + METRICS10
    rows, rows_full = [], []
    for axis, tag, label, cell_id in conditions:
        m = seeded_metrics(cell_id)
        if not m:
            print("[skip] E5 visual %s: 无 cell 结果" % cell_id)
            continue
        rows.append([display_name(cell_id), "last", axis, tag, label]
                    + delta_row(m, keys=E5_VISUAL_MAIN_METRICS))
        for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
            rows_full.append([display_name(cell_id), "last", axis, tag, label, row_key]
                             + metric_row(m, cond))
        rows_full.append([display_name(cell_id), "last", axis, tag, label, "Δ(VT−V)"]
                         + delta_row(m))
    write_csv(E5_DIR / "Main_Table3_E5_VisualFailure.csv", header, rows)
    write_csv(E5_SUPP / "Full_Table5_E5_VisualFailure.csv", header_full, rows_full)
    export_e5_pure_visual(model)


# E5 纯视觉基线对照（2026-11-04 口径更正版，z_note/E5_纯视觉基线扩展说明.md §3.3）：
# 同一退化条件下 V7_Cv5 的 VT / V 与外部纯视觉基线的差，Δ<0 = VT（或 V）误差更低；
# 当前只有 WHAM（WHAM first, other baselines pending）。表号 6 为顺延占位。
E5_PURE_VISUAL_BASELINES = ("WHAM",)


def _e5_baseline_pred_dir(tag: str) -> str:
    """条件档位 → 基线 predictions 子目录（occ000 = 干净档 local_gen）。"""
    if tag == "occ000":
        return "local_gen"
    if tag.startswith("occ"):
        return "e5_occ_lower" + tag[3:]
    return "e5_" + tag


def _e5_baseline_payload(baseline: str, tag: str) -> dict | None:
    path = (REPO / "results" / "baselines" / baseline / "predictions"
            / _e5_baseline_pred_dir(tag) / "eval_motion" / "metrics"
            / ("%s_comparison.json" % EVAL_SPLIT))
    if not path.is_file():
        print("[skip] E5 pure-visual %s %s: 无 %s" % (baseline, tag, path))
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != EVAL_SPLIT:
        raise ValueError("Result split mismatch: " + str(path))
    import hashlib
    SOURCE_RECEIPTS.append({"cell": "%s/%s" % (baseline, _e5_baseline_pred_dir(tag)),
                            "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "split": EVAL_SPLIT, "checkpoint_sha256": None,
                            "contact_label_package_sha256": None})
    return payload


def _e5_num(v):
    if isinstance(v, tuple):
        v = v[0]
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _e5_diff(a, b):
    return round(a - b, 3) if a is not None and b is not None else ""


def export_e5_pure_visual(model: str) -> None:
    """Main_Table6：Δ(VT−基线) / Δ(V−基线)；Full_Table6：V / VT / 基线绝对值
    与相对干净档的退化量（Δ干净），用于比较各方法在遮挡下的退化幅度。"""
    conditions = _e5_visual_conditions(model)
    metrics = E5_VISUAL_MAIN_METRICS
    header = ["轴", "档位", "条件"]
    for b in E5_PURE_VISUAL_BASELINES:
        header += ["%s_Δ(VT−%s) %s" % (b, b, k) for k in metrics]
        header += ["%s_Δ(V−%s) %s" % (b, b, k) for k in metrics]
    header_full = ["轴", "档位", "条件", "行"] + metrics + ["会话数"]
    rows, rows_full, clean, series = [], [], {}, {}
    for axis, tag, label, cell_id in conditions:
        m = seeded_metrics(cell_id)
        if not m:
            print("[skip] E5 pure-visual %s: 无 cell 结果" % cell_id)
            continue
        vals = {"V": [_e5_num(m["V2M"].get(k)) for k in metrics],
                "VT": [_e5_num(m["VT2M"].get(k)) for k in metrics]}
        counts = {"V": "", "VT": ""}
        row = [axis, tag, label]
        for b in E5_PURE_VISUAL_BASELINES:
            payload = _e5_baseline_payload(b, tag)
            summary = (payload or {}).get("summary") or {}
            vals[b] = [_e5_num(summary.get(BASELINE_METRIC_KEYS[k])) for k in metrics]
            counts[b] = (payload or {}).get("n_sessions", "")
            row += [_e5_diff(a, w) for a, w in zip(vals["VT"], vals[b])]
            row += [_e5_diff(a, w) for a, w in zip(vals["V"], vals[b])]
        rows.append(row)
        series[tag] = {n: list(v) for n, v in vals.items()}
        if tag == "occ000":
            clean = {n: list(v) for n, v in vals.items()}
        for n, v in vals.items():
            rows_full.append([axis, tag, label, {"V": "V2M", "VT": "VT2M"}.get(n, n)]
                             + ["" if x is None else round(x, 3) for x in v] + [counts[n]])
        for n, v in vals.items():
            if n in clean:
                rows_full.append([axis, tag, label, "Δ干净(%s)" % {"V": "V2M", "VT": "VT2M"}.get(n, n)]
                                 + [_e5_diff(x, c) for x, c in zip(v, clean[n])] + [""])
    write_csv(E5_DIR / "Main_Table6_E5_PureVisual.csv", header, rows)
    write_csv(E5_SUPP / "Full_Table6_E5_PureVisual.csv", header_full, rows_full)
    _e5_pure_visual_figure(model, metrics, series)


def _e5_pure_visual_figure(model: str, metrics: list[str],
                           series: dict[str, dict[str, list]]) -> None:
    """Main_Fig5 = 三条线的退化曲线：主模型 V / 主模型 VT / 外部纯视觉基线
    （2026-11-05 用户要求）。y = 指标绝对值，三轴各一张子图行。
    图号 5 为顺延占位（Main_Fig2/3/4 已占用），与 Table6 同一对照口径。

    数字直接复用 Main/Full_Table6 的同一批取数（series），表图同源——
    不另读基线 json，避免表与图各算一次而对不上。
    基线只有五主指标与 occ 10/20/40/60/80，故曲线按各轴**实际落地的档位**
    画，缺的点留空（不插值）。判读只读符号与排序。

    只画**有数据的轴**：act8020 口径只有遮挡轴的 eval 格，断流轴整行无点，
    留着就是空白子图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = E5_PURE_VISUAL_BASELINES[0]
    # 两轴 ×（档位, x 值, x 轴标题）：x 用物理量。
    axes_spec = [
        ("Occlusion (%)", [("occ%s" % o, float(int(o))) for o in ("010", "020", "040", "060", "080")], False),
        ("Dropout rho (%)", [("gap_r%s" % r, float(int(r))) for r in E5_GAP_RHOS], False),
    ]
    names = ("V", "VT", baseline)

    def xy(points: list, clean_x: float, name: str, col: int) -> tuple[list, list]:
        # 干净档在每轴都作为参照：遮挡/断流的 0 点。
        xs, ys = [], []
        for tag, xv in [("occ000", clean_x)] + points:
            value = (series.get(tag) or {}).get(name)
            if value is None or value[col] is None:
                continue
            xs.append(xv)
            ys.append(value[col])
        return xs, ys

    live = []
    for xlabel, points, logx in axes_spec:
        clean_x = 1.0 if logx else 0.0
        if any(len(xy(points, clean_x, name, c)[1]) >= 2
               for name in names for c in range(len(metrics))):
            live.append((xlabel, points, logx, clean_x))
    if not live:
        print("[skip] E5 pure-visual figure: 无足够落地档位")
        return
    color_v, color_vt, color_b = "#2a78d6", "#eb6834", "#4f4f4f"
    nrow, ncol = len(live), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow))
    axes = np.atleast_2d(axes)
    drawn = False
    for r, (xlabel, points, logx, clean_x) in enumerate(live):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for name, color, style, marker in (
                    ("V", color_v, "-", "o"),
                    ("VT", color_vt, "--", "s"),
                    (baseline, color_b, ":", "^")):
                xs, ys = xy(points, clean_x, name, c)
                if len(ys) < 2:
                    continue
                order = np.argsort(xs)
                ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], color=color,
                        ls=style, lw=1.9, marker=marker, ms=4.5,
                        label=name if name != baseline else "%s (pure visual)" % baseline)
                drawn = True
            if logx:
                ax.set_xscale("log", base=2)
            ax.axvline(clean_x, color="#c8c8c8", ls=":", lw=0.9, zorder=0)
            ax.set_title(metric if r == 0 else "", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8.5)
            if c == 0:
                ax.set_ylabel("mm", fontsize=8.5)
            ax.grid(alpha=0.25, lw=0.6)
            ax.margins(y=0.14)
    if not drawn:
        plt.close(fig)
        print("[skip] E5 pure-visual figure: 无足够落地档位")
        return
    # 图例跨所有子图去重收集：某个子图少画一条线时图例不能跟着少一条。
    handles, labels = [], []
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("E5 pure-visual comparison: %s V / VT vs %s (ckpt_last)"
                 % (model, baseline), fontsize=12)
    out = E5_DIR / "Main_Fig5_E5_PureVisual.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % out)


def _curve_crossings(xs, a, b) -> list[tuple[float, float]]:
    """两条折线的交点：d = a − b 变号处按线性插值求 x 与 y（跳过缺失点）。
    起点重合（两条曲线同起点，d[0] == 0）不算交点。"""
    x = np.asarray(xs, dtype=float)
    ya = np.asarray(a, dtype=float)
    d = np.asarray(b, dtype=float) - ya
    out = []
    for i in range(len(x) - 1):
        if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
            continue
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            out.append((x[i] + t * (x[i + 1] - x[i]),
                        ya[i] + t * (ya[i + 1] - ya[i])))
        elif d[i] == 0 and i > 0 and d[i - 1] * d[i + 1] < 0:
            out.append((x[i], ya[i]))
    return out


def _mark_crossings(ax, xs, v, vt) -> None:
    """在 VT 与 V 的交点画竖虚线 + × 标记 + 交点档位文字（靠顶时文字放下方）。"""
    y0, y1 = ax.get_ylim()
    for xi, yi in _curve_crossings(xs, v, vt):
        if xi < 1.0:  # 起点附近的伪交点（两线相差 <1% 档位）不标
            continue
        ax.axvline(xi, color="#8a8a8a", ls=":", lw=0.9, zorder=1)
        ax.plot([xi], [yi], marker="x", color="#333333", ms=7, mew=1.5,
                zorder=5, label="_nolegend_")
        above = (yi - y0) < 0.75 * (y1 - y0)
        ax.annotate("cross %.0f%%" % xi, xy=(xi, yi),
                    xytext=(0, 9 if above else -14),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color="#4a4a4a")


def _e5_degradation_figure(models: list[str], metrics: list[str],
                           units: dict[str, str], scales: dict[str, float],
                           suptitle: str, fname: str,
                           titles: dict[str, str] | None = None,
                           levels: tuple[str, ...] | None = None,
                           absolute: bool = False,
                           crossover: bool = False,
                           t_reference: bool = False) -> None:
    """E5 遮挡曲线通用绘图：x = 遮挡档，V 蓝实线 / VT 橙虚线，子图 = metrics。
    absolute=False：y = 相对 occ000 的 Δ（×scale），判读 VT 退化应轻于 V；
    absolute=True：y = 指标绝对值（×scale），直接看 V / VT 两条曲线的高低
    与间距（多 seed 时画 mean ± std 带）。
    crossover=True：标注 VT / V 曲线交点（绝对误差图的判据，实验计划 §4.3）；
    t_reference=True：叠加 U@T（纯触觉）水平参考线，判断视觉退化到哪一档时
    触觉开始主导。
    titles = 子图标题覆盖（接触类标注两套机制口径）；
    levels = 档位子集（缺省 = E5_OCC_LEVELS 全部十档）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = tuple(levels) if levels is not None else E5_OCC_LEVELS
    model = models[0] if models else "V7_Cv5"
    cells = {"occ000": model}
    for occ in levels:
        cells["occ%s" % occ] = "%s_evalocc%s" % (model, occ)
    xs = [0] + [int(o) for o in levels]
    color_v, color_vt = "#2a78d6", "#eb6834"  # 校验过的双色（CVD 分离 24.7）
    ncol = 4 if len(metrics) >= 6 else len(metrics)
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.1 * nrow))
    base = seeded_metrics(cells["occ000"])
    for ax, metric in zip(np.asarray(axes).flat, metrics):
        scale = scales.get(metric, 1.0)
        if not base:
            continue

        def _mean(v):
            return v[0] if isinstance(v, tuple) else float(v)

        def _std(v):
            return v[1] if isinstance(v, tuple) else 0.0

        def series(cond):
            vals, errs = [], []
            for occ, cell in cells.items():
                m = seeded_metrics(cell)
                raw = m[cond].get(metric) if m else None
                if raw is None:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
                    continue
                if absolute:
                    vals.append(_mean(raw) * scale)
                    errs.append(_std(raw) * scale)
                else:
                    b = base[cond].get(metric)
                    if b is None:
                        vals.append(float("nan"))
                        errs.append(float("nan"))
                        continue
                    vals.append((_mean(raw) - _mean(b)) * scale)
                    errs.append(((_std(raw) ** 2 + _std(b) ** 2) ** 0.5) * scale)
            return vals, errs

        v, v_err = series("V2M")
        vt, vt_err = series("VT2M")
        ax.plot(xs, v, color=color_v, lw=2, marker="o", ms=5, label="V")
        ax.plot(xs, vt, color=color_vt, lw=2, ls="--", marker="s", ms=5, label="VT")
        for vals, errs, color in ((v, v_err, color_v), (vt, vt_err, color_vt)):
            arr_v = np.asarray(vals, dtype=float)
            arr_e = np.asarray(errs, dtype=float)
            ok = np.isfinite(arr_v) & np.isfinite(arr_e) & (arr_e > 0)
            if ok.any():
                ax.fill_between(np.asarray(xs, dtype=float)[ok],
                                (arr_v - arr_e)[ok], (arr_v + arr_e)[ok],
                                color=color, alpha=0.15, lw=0)
        if t_reference:
            tv = base["T2M"].get(metric)
            if tv is not None:
                ax.axhline(_mean(tv) * scale, color="#7a7a7a", ls="-.",
                           lw=1.1, zorder=0, label="T-only")
        if crossover:
            _mark_crossings(ax, xs, v, vt)
        if all(np.isfinite(v)) and all(np.isfinite(vt)):
            ax.annotate("V", (xs[-1], v[-1]), color=color_v, fontweight="bold",
                        xytext=(4, 0), textcoords="offset points", va="center")
            ax.annotate("VT", (xs[-1], vt[-1]), color=color_vt, fontweight="bold",
                        xytext=(4, 0), textcoords="offset points", va="center")
        ax.set_title((titles or {}).get(metric, metric), fontsize=10)
        ax.set_xlabel("Occlusion (%)", fontsize=9)
        ax.set_ylabel((units.get(metric, "mm") if absolute
                       else "Δ vs occ000 (%s)" % units.get(metric, "mm")),
                      fontsize=9)
        if absolute:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.6)
        ax.margins(y=0.15)
    for ax in np.asarray(axes).flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(suptitle % model, fontsize=12)
    handles = [h for h in np.asarray(axes).flat[0].lines
               if h.get_label() and not str(h.get_label()).startswith("_")]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
    out = (E5_DIR if "/" not in fname else ETEST / "E5_OcclusionRobustness") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % out)


def _e5_global_figure(models: list[str], metrics: list[str], fname: str,
                      levels: tuple[str, ...] | None = None,
                      suptitle: str = "E5 global degradation under occlusion - %s (ckpt_last)"
                      ) -> None:
    """全局退化量图：每个指标先除以 **V 的 occ000 值**（干净 V = 100%），
    再对指标等权平均，得到单条 V / VT 全局曲线；标注两者交点与 T-only
    水平线。共用同一基准，干净档 VT 相对 V 的代价因此可见。
    y 越低越好；>100% = 比干净 V 退化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = tuple(levels) if levels is not None else E5_OCC_LEVELS
    model = models[0] if models else "V7_Cv5"
    cells = {"occ000": model}
    for occ in levels:
        cells["occ%s" % occ] = "%s_evalocc%s" % (model, occ)
    xs = [0] + [int(o) for o in levels]
    base = seeded_metrics(cells["occ000"])

    def _mean(v):
        return v[0] if isinstance(v, tuple) else float(v)

    def global_series(cond):
        vals = []
        for _, cell in cells.items():
            m = seeded_metrics(cell)
            rel = []
            for k in metrics:
                raw = m[cond].get(k) if m else None
                b = base["V2M"].get(k) if base else None
                if raw is None or b is None or float(_mean(b)) == 0:
                    continue
                rel.append(float(_mean(raw)) / float(_mean(b)) * 100.0)
            vals.append(float(np.mean(rel)) if rel else float("nan"))
        return vals

    v, vt = global_series("V2M"), global_series("VT2M")
    t_ref = []
    for _ in cells:
        rel = []
        for k in metrics:
            raw = base["T2M"].get(k) if base else None
            b = base["V2M"].get(k) if base else None
            if raw is None or b is None or float(_mean(b)) == 0:
                continue
            rel.append(float(_mean(raw)) / float(_mean(b)) * 100.0)
        t_ref.append(float(np.mean(rel)) if rel else float("nan"))

    color_v, color_vt = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, v, color=color_v, lw=2.2, marker="o", ms=5, label="V")
    ax.plot(xs, vt, color=color_vt, lw=2.2, ls="--", marker="s", ms=5, label="VT")
    if np.isfinite(t_ref[0]):
        ax.axhline(t_ref[0], color="#7a7a7a", ls="-.", lw=1.2,
                   label="T-only", zorder=0)
    ax.axhline(100.0, color="#c8c8c8", ls=":", lw=0.9, zorder=0,
               label="clean (100%)")
    crosses = _curve_crossings(xs, v, vt)
    _mark_crossings(ax, xs, v, vt)
    if not crosses:
        ax.text(0.98, 0.03, "VT ≤ V at all occlusion levels (no crossing)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#4a4a4a")
    ax.set_xlabel("Occlusion (%)", fontsize=10)
    ax.set_ylabel("Mean error relative to clean V (%)", fontsize=10)
    ax.set_title(suptitle % model, fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(y=0.12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    out = E5_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % out)


def export_e5_figure(models: list[str]) -> None:
    """E5 图 1 + 图 2 + Main_Fig2 + Main_Fig3（2026-11-02 用户裁定）。

    x = 遮挡档，y = 相对 occ000 的退化（Δ），V（蓝实线）与 VT（橙虚线）
    两条曲线——判读：VT 退化应轻于 V（曲线更低/更平）。
    - 图 1 = 十指标完整面板（occ000–100 十一档，工作视图）
    - 图 2 = contact / 地面 / 轨迹 / 踝足 vs 上肢对照（十一档）
    - Main_Fig2 = 论文主图（2026-11-02 用户裁定：occ 20–90 八点，
      7 精选指标 = PA-MPJPE / traj_ATE@0.5s / GroundPen / FootSliding /
      GlobalMPJPE / RootOrientationDrift / PA-MPJPE_anklefoot）。
    - Main_Fig3 = 绝对误差遮挡曲线（2026-11-04 用户要求）：同 Main_Fig2 的
      指标与档位，y 改为指标绝对值，标注 V / VT 交点与 U@T 参考线。
    - Main_Fig4 = 全局退化量图（2026-11-04 用户要求）：七指标各自除以 V 的
      occ000 值后等权平均为单条全局曲线（干净 V = 100%），标出 V / VT 交点。
    主模型缺省 V7_Cv5（last 口径）。图内英文。
    """
    units = {"MPJRE": "deg", "RootOrientationDrift": "deg", "Jitter_pred": "m/s³",
             "contact_acc": "%", "contact_f1": "%", "contact_balanced_acc": "%",
             "contact_recall": "%", "air_recall": "%", "GroundPenFrac": "frac"}
    scales = {k: 100.0 for k in ("contact_acc", "contact_f1",
                                 "contact_balanced_acc", "contact_recall",
                                 "air_recall")}
    # traj 子图 = traj_ATE@0.5s（20 帧段首对齐全方法同一处理，11-01 裁定；
    # occ 四档 11-02 重评后五 cell 齐列）；表 1 保留未对齐 traj_ATE 单列。
    _e5_degradation_figure(
        models, list(METRICS10), units, scales,
        "E5 degradation under occlusion - %s (ckpt_last)",
        "Full_Fig1_E5_Degradation.png")
    _e5_degradation_figure(
        models, ["contact_acc", "contact_f1", "contact_balanced_acc",
                 "traj_ATE", "GroundPenFrac", "ground_pen_foot_mm",
                 "RootOrientationDrift", "PA-MPJPE_anklefoot",
                 "PA-MPJPE_upper"],
        units, scales,
        "E5 tactile-gain metrics under occlusion - %s (ckpt_last)",
        "Full_Fig2_E5_TactileGain.png",
        titles={"contact_acc": "contact_acc (V: kinematic / VT: head)",
                "contact_f1": "contact_f1 (V: kinematic / VT: head)",
                "contact_balanced_acc": "contact_balanced_acc (V: kinematic / VT: head)"})
    # Main_Fig2 = 论文主图（2026-11-02 用户裁定：occ 20–90 八点 × 7 精选指标，
    # 与 Main_Table2_E5_Degradation.csv 同指标同切点，常量见模块头）。
    _e5_degradation_figure(
        models, list(E5_MAIN_METRICS),
        units, scales,
        "E5 degradation under occlusion - %s (ckpt_last)",
        "Main_Fig2_E5_Degradation.png",
        levels=E5_MAIN_LEVELS,
        titles={"PA-MPJPE_anklefoot": "PA-MPJPE (ankle-foot)"})
    # Main_Fig3 = 绝对误差遮挡曲线（2026-11-04 用户要求；实验计划 §4.3 正文图）：
    # 与 Main_Fig2 同指标同档位，y 改为指标绝对值——直接对照 V / VT 曲线高低，
    # 标注两者交点，并叠加 U@T 水平参考线判断视觉退化到哪一档触觉开始主导。
    _e5_degradation_figure(
        models, list(E5_MAIN_METRICS),
        units, scales,
        "E5 absolute error under occlusion - %s (ckpt_last)",
        "Main_Fig3_E5_Absolute.png",
        levels=E5_MAIN_LEVELS,
        titles={"PA-MPJPE_anklefoot": "PA-MPJPE (ankle-foot)"},
        absolute=True, crossover=True, t_reference=True)
    # Main_Fig4 = 全局退化量图（2026-11-04 用户要求）：七指标各自除以 V 的
    # occ000 值后等权平均为单条全局曲线（干净 V = 100%，两条件共用基准），
    # 标出 V / VT 交点。
    _e5_global_figure(
        models, list(E5_MAIN_METRICS), "Main_Fig4_E5_Global.png",
        levels=E5_MAIN_LEVELS)


# ---------------- E3：多视角（表 4 骨架：落地的填数、未落地的填 —） ----------------
# 行 = 实验计划表 4 四条对比线 + TTA 原方法（设置五线）；列 = n=1..4，
# 指标 = MPJPE / GlobalMPJPE 均值±标准差（组合间，n=4 单一组合无标准差）。
# 朴素无标定晚融合 = pilot 的 plain_fusion（eval_multiview.py 定义：
# V3_Bv1 原模型 per-view V2M 世界系平均）——明确对应，已落地填数。
# 其余线来源待用户确认后填入；未落地填 "—"。
E3_LINES = ["标定多视角（参考上限）", "TTA 原方法",
            "AnySole TTA", "AnySole TTA wLcon"]


def _stage1_agg(json_path: Path, key: str) -> tuple[str, str]:
    """run_tta stage1 产物逐会话聚合（PA-MPJPE，nan 剔除）。"""
    if not json_path.is_file():
        return "—", "—"
    import numpy as np
    sessions = json.loads(json_path.read_text(encoding="utf-8"))["sessions"]
    vals = np.array([v[key] for v in sessions.values()], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return "—", "—"
    return "%.3f" % vals.mean(), "%.3f" % vals.std()


def _combo_stats(base: Path, model: str, n: int, key: str) -> tuple[str, str]:
    """组合间口径（2026-10-30 用户裁定）：std 列人口 = 相机组合间。

    来源 = run_tta --cams 模式逐组合产物（{model}_test_i{n}_cams*.json）；
    每个文件一个组合的会话均值 → 跨组合 mean±std。
    n=4 单一组合：std = —（表注：无组合间标准差）。
    """
    import numpy as np
    files = sorted(base.glob(f"anysole_tta_stage1_{model}_test_i{n}_cams*.json"))
    if not files:
        # 兜底：--i 单进程产物（内部含 combos+overall）
        merged = base / f"anysole_tta_stage1_{model}_test_i{n}_lr0.01.json"
        if merged.is_file():
            d = json.loads(merged.read_text(encoding="utf-8"))
            cell = d.get("overall", {}).get(key)
            if cell and cell.get("mean") is not None:
                std = cell.get("std")
                return "%.3f" % cell["mean"], ("%.3f" % std if std is not None else "—")
        if n == 4:
            # i=4 = 固定四相机组合 = 单组合模式产物（sessions 平铺，无组合间 std）。
            # 口径守卫：只接受无标定产物（camera_calibration == "none"；
            # 标定时代的存量 json 不入表）
            plain = base / f"anysole_tta_stage1_{model}_test_lr0.01.json"
            if plain.is_file():
                d = json.loads(plain.read_text(encoding="utf-8"))
                if d.get("camera_calibration") == "none":
                    vals = [v[key] for v in d["sessions"].values()
                            if v.get(key) is not None and np.isfinite(v[key])]
                    if vals:
                        return "%.3f" % np.mean(vals), "—"
        return "—", "—"
    means = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for cm in d.get("combos", {}).values():
            if cm.get(key) is not None:
                means.append(cm[key])
    means = [m for m in means if np.isfinite(m)]
    if not means:
        return "—", "—"
    if len(means) == 1:
        return "%.3f" % means[0], "—"
    return "%.3f" % np.mean(means), "%.3f" % np.std(means)


def _line1_stats(base: Path, n: int) -> tuple[str, str, str, str, str, str]:
    """标定多视角线（EasyMocap，2026-10-30 集成）：组合间口径（同 _combo_stats）。

    首选读聚合缓存 metrics/e3_multiview.json 的 easymocap 键（由
    anysole_tta/easymocap_baseline.py --aggregate 生成，纯 json 读取、
    本导出器无 touch_gait 依赖也可跑）；键缺失时兜底从
    predictions/easymocap/{sid}_n{n}_{camstr}.npz 重算（每组合取会话均值、
    再跨组合 mean±std；n=4 单一组合 std = —，重算需 touch_gait 依赖）。
    返回 (PA_mean, PA_std, MPJPE_mean, MPJPE_std, GMPJPE_mean, GMPJPE_std)。
    """
    cache = base / "e3_multiview.json"
    if cache.is_file():
        try:
            cell = json.loads(cache.read_text(encoding="utf-8")) \
                .get("easymocap", {}).get(str(n))
            if cell and cell.get("mpjpe_mean") is not None:
                def _fmt(v):
                    return "%.3f" % v if v is not None else "—"
                return (_fmt(cell.get("pa_mpjpe_mean")),
                        _fmt(cell.get("pa_mpjpe_std")),
                        _fmt(cell["mpjpe_mean"]), _fmt(cell["mpjpe_std"]),
                        _fmt(cell["gmpjpe_mean"]), _fmt(cell["gmpjpe_std"]))
        except Exception:
            pass
    import numpy as np
    pred_dir = base.parent / "predictions/easymocap"
    if not pred_dir.is_dir():
        return "—", "—", "—", "—", "—", "—"
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from anysole_tta.calib_baseline import load_gt_joints
        from anysole_tta.easymocap_baseline import _eval3
    except Exception:
        return "—", "—", "—", "—", "—", "—"
    combos = [c for c in ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4),
                          (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)) if len(c) == n]
    p_means, m_means, g_means = [], [], []
    for combo in combos:
        camstr = "".join(str(c) for c in combo)
        gts, p_vals, m_vals, g_vals = {}, [], [], []
        for f in sorted(pred_dir.glob(f"*_n{n}_{camstr}.npz")):
            sid = f.name.split("_")[0]
            if sid not in gts:
                gts[sid] = load_gt_joints(sid)[0]
            d = np.load(f)
            pa, m, g = _eval3(d["joints"], gts[sid])
            p_vals.append(pa)
            m_vals.append(m)
            g_vals.append(g)
        if m_vals:
            p_means.append(np.mean(p_vals))
            m_means.append(np.mean(m_vals))
            g_means.append(np.mean(g_vals))
    if not m_means:
        return "—", "—", "—", "—", "—", "—"
    if len(m_means) == 1:
        return ("%.3f" % p_means[0], "—", "%.3f" % m_means[0], "—",
                "%.3f" % g_means[0], "—")
    return ("%.3f" % np.mean(p_means), "%.3f" % np.std(p_means),
            "%.3f" % np.mean(m_means), "%.3f" % np.std(m_means),
            "%.3f" % np.mean(g_means), "%.3f" % np.std(g_means))


def _hmr_stats(base: Path, n: int, arm: str, metric: str) -> tuple[str, str]:
    """HMR2.0 TTA 阶段 2 产物（tta_theta_windows_i{N}.json）的 overall 组合间
    mean±std；i=4 单一组合 std=None → —。"""
    import glob
    import numpy as np
    files = sorted(glob.glob(str(base / "zero" / f"tta_theta_windows_i{n}*.json")))
    if not files:
        return "—", "—"
    means = []
    for f in files:
        d = json.loads(open(f, encoding="utf-8").read())
        if "overall" in d:      # --i 单进程产物：直接取 overall
            cell = d["overall"].get(arm, {}).get(metric, {})
            if cell and cell.get("mean") is not None:
                means.append(cell["mean"])
        else:                   # --cams 逐组合产物：取该组合会话均值
            for cm in d.get("combos", {}).values():
                v = cm.get(arm, {}).get(metric)
                if v is not None:
                    means.append(v)
    if not means:
        return "—", "—"
    if len(means) == 1:
        return "%.3f" % means[0], "—"
    return "%.3f" % np.mean(means), "%.3f" % np.std(means)


def export_e3() -> None:
    """四条对照线 × 视角数 1-4（2026-10-31 用户裁定：只此四线，
    朴素无标定晚融合不作数）。

    - 标定多视角（参考上限）= EasyMocap（RTMPose 检测 + 真值外参三角化 +
      原生 SMPL 拟合；_line1_stats 读 predictions/easymocap 缓存）
    - TTA 原方法 = HMR 2.0（无标定 TTA；_hmr_stats 读
      metrics/zero/tta_theta_windows_i{N}.json：i=1 单视角 pa_init、
      i>=2 virtual 臂；三指标 PA/MPJPE/GlobalMPJPE）
    - AnySole TTA = V7_Cv5 无标定纯视觉 TTA（_combo_stats 读
      anysole_tta_stage1_V7_Cv5_test_i{N}_cams*.json：i=1 pa_init、
      i>=2 pa_virt_refined/mpjpe_virt_refined/gmpjpe_virt_refined）
    - AnySole TTA wLcon = 待跑（接触约束第二版）→ 填 —
    std 列人口 = 组合间；i=4 单一组合无组合间标准差 → —（表注）。
    """
    stage1_dir = REPO / "results/baselines/HMR2.0/metrics"
    header = ["对比线", "模型", "视角数", "PA-MPJPE_mean", "PA-MPJPE_std",
              "MPJPE_mean", "MPJPE_std", "GlobalMPJPE_mean", "GlobalMPJPE_std"]
    rows = []
    for line in E3_LINES:
        for n in ("1", "2", "3", "4"):
            ni = int(n)
            if line == "标定多视角（参考上限）":
                pa_mean, pa_std, m_mean, m_std, g_mean, g_std = \
                    _line1_stats(stage1_dir, ni)
                rows.append([line, "EasyMocap", n, pa_mean, pa_std,
                             m_mean, m_std, g_mean, g_std])
            elif line == "TTA 原方法":
                arm = "pa_init" if n == "1" else "pa_virtual"
                pa_mean, pa_std = _hmr_stats(stage1_dir, ni, arm, "pa")
                m_mean, m_std = _hmr_stats(stage1_dir, ni, arm, "mpjpe")
                g_mean, g_std = _hmr_stats(stage1_dir, ni, arm, "gmpjpe")
                rows.append([line, "HMR 2.0", n, pa_mean, pa_std,
                             m_mean, m_std, g_mean, g_std])
            elif line == "AnySole TTA":
                if n == "1":
                    pa_mean, pa_std = _combo_stats(stage1_dir, "V7_Cv5", ni,
                                                   "pa_init")
                    rows.append([line, "V7_Cv5", n, pa_mean, pa_std,
                                 "—", "—", "—", "—"])
                else:
                    pa_mean, pa_std = _combo_stats(stage1_dir, "V7_Cv5", ni,
                                                   "pa_virt_refined")
                    m_mean, m_std = _combo_stats(stage1_dir, "V7_Cv5", ni,
                                                 "mpjpe_virt_refined")
                    g_mean, g_std = _combo_stats(stage1_dir, "V7_Cv5", ni,
                                                 "gmpjpe_virt_refined")
                    rows.append([line, "V7_Cv5", n, pa_mean, pa_std,
                                 m_mean, m_std, g_mean, g_std])
            else:
                rows.append([line, "V7_Cv5", n, "—", "—", "—", "—", "—", "—"])
        write_csv(ETEST / "E3_MultiViewConsistency" / "Full_Table1_E3_MultiView.csv",
              header, rows)


# ---------------- E6：接触相位（json + PNG） ----------------
E6_GROUP_ORDER = ["walk", "run", "squat", "stand_shift", "single_stand",
                  "jump_jack", "lateral", "sit_stand", "stairs"]
# 编号与驱动一致（ACTION_GROUPS.values().index 首现：walk=1、run=3、squat=5…）
E6_ACTION_VALUES = ["walk", "walk", "run", "run", "squat", "stand_shift",
                    "single_stand", "jump_jack", "lateral", "sit_stand", "stairs"]
E6_PHASE_ZH = {"0": "稳定支撑", "1": "摆动", "2": "落脚附近", "3": "离地附近"}
E6_METRIC_ZH = {"vert": "足垂直误差(mm)", "pen": "地面穿透(mm)",
                "contact": "足高判据接触准确率(%)"}

# ---------------- E6 门控主表（2026-11-03 用户裁定：三组门控两张主表） ----------------
# 三组 = 无门 Cv7 / 双门 Cv5 / 单门 Cv4（同配方同枚举训练，唯一差异是门）。
# 门控实验本质 = VT 相较于 V 的情况 → 数值全部是 Δ(VT−V)（2026-11-03 用户
# 裁定：去掉 T2M 参照列，全 10 指标；traj 列 = traj_ATE@0.5s 口径）。
E6_GATE_ROWS = [("V7_Cv7_s1", "无门"), ("V7_Cv5_s1", "双门"),
                ("V7_Cv4_s1", "单门")]
# 动作分层 80/20 新 split（2026-11-04）只有 Cv5/Cv7 两格：环境变量 E6_GATE_ROWS 覆盖，
# 格式 cell:标签,cell:标签（首项 = 无门参照）；不设即旧行为逐位不变。
import os as _os
if _os.environ.get("E6_GATE_ROWS"):
    E6_GATE_ROWS = [tuple(item.split(":", 1)) for item in _os.environ["E6_GATE_ROWS"].split(",") if item]
E6_GATE_METRICS = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
                   "traj_ATE", "RootOrientationDrift", "FootSliding",
                   "GroundPen", "Jitter_pred"]
E6_SESSION_KEYS = {"PA-MPJPE": "pa_mpjpe_mm", "MPJPE": "mpjpe_mm",
                   "GlobalMPJPE": "global_mpjpe_mm",
                   "W-MPJPE100": "w_mpjpe100_mm", "MPJRE": "mpjae_deg",
                   "traj_ATE@0.5s": "root_ate_seg20_mm",
                   "RootOrientationDrift": "root_orientation_drift_deg",
                   "FootSliding": "foot_sliding_mm",
                   "GroundPen": "ground_pen_mm",
                   "Jitter_pred": "jitter_pred_m_s3"}


E6_MIN_EFFECT = 0.05          # 比例分母下限：低于此值比例失真，只报绝对量


def _e6_gate_decomposition() -> dict:
    """三组门控的逐会话分解 ΔE = ΔE⁺ + ΔE⁻（2026-11-03 用户裁定）。

    旧表缺陷：表 1 用池化净值（cell test.json，VT−V），表 2 用会话级
    mean(min(Δ,0))，两表混用「净值」与「分量」两种口径。新口径两表同源：

        Δ_s = VT_s − V_s（正 = 触觉引入退化）
        w_s = 该指标的聚合权重（帧数；FootSliding = GT 接触帧数）
        ΔE   = Σ w_s Δ_s / Σ w_s      （= test.json 池化净值，逐位复核）
        ΔE⁺ = Σ w_s max(Δ_s, 0) / Σ w_s   （有害分量，表 1）
        ΔE⁻ = Σ w_s min(Δ_s, 0) / Σ w_s   （有益分量，表 2，负数）

    故 ΔE⁺ + ΔE⁻ = ΔE 精确成立，且 ΔE 与 E2 表 1 的 VT−V 逐位吻合
    （FootSliding 因接触帧权重 ≤0.003 差）。权重取指标自身的聚合口径，
    不是额外统计假设：帧数权重即池化含义，FootSliding 的接触帧数来自
    detail/session_foot_contact_counts.json（GT 足速 <0.3 m/s）。
    """
    weight_path = E6_BASE / "detail" / "session_foot_contact_counts.json"
    contact_counts = {}
    if weight_path.is_file():
        contact_counts = json.loads(
            weight_path.read_text(encoding="utf-8"))["counts"]
    else:
        print("[warn] %s 缺失；FootSliding 退回按帧数加权" % weight_path)
    comp: dict = {}
    for cell, _ in E6_GATE_ROWS:
        d = json.loads((CELLS / cell / "metrics" / ("session_detail_%s.json" % EVAL_SPLIT))
                       .read_text(encoding="utf-8"))["conditions"]
        vt = {e["session_id"]: e["metrics"] for e in d["VT2M"]}
        v = {e["session_id"]: e["metrics"] for e in d["V2M"]}
        n_frames = {e["session_id"]: e["n_frames"] for e in d["V2M"]}
        sids = sorted(set(v) & set(vt))
        comp[cell] = {}
        for m in E6_GATE_METRICS:
            key = E6_SESSION_KEYS[m]
            delta = np.array([vt[s][key] - v[s][key] for s in sids], dtype=float)
            w = np.array([
                float(contact_counts.get(s, n_frames[s]))
                if m == "FootSliding" else float(n_frames[s])
                for s in sids])
            total = w.sum()
            comp[cell][m] = {
                "net": float((w * delta).sum() / total),
                "plus": float((w * np.maximum(delta, 0.0)).sum() / total),
                "minus": float((w * np.minimum(delta, 0.0)).sum() / total),
            }
    return comp


def export_e6_gate_tables() -> None:
    """E6 门控两张主表（2026-11-03 用户裁定）：三组门控 × 全 10 指标。

    表 1 = 有害分量 + 抑制率（5 行）：
        3 行模型 = ΔE⁺；2 行抑制率 = 1 − ΔE⁺_门控/ΔE⁺_无门。
    表 2 = 有益分量 + 保留率（5 行）：
        3 行模型 = ΔE⁻（负 = 触觉有帮助）；2 行保留率 =
        ΔE⁻_门控/ΔE⁻_无门（两者同为负，比值 = 保留比例）。
    两表同一总体、同一权重，故 ΔE⁺ + ΔE⁻ = ΔE（cell test.json 池化净值）；
    净值不重复入表（见 E2 表 1），保持两表各自只讲一件事。
    统计口径见 _e6_gate_decomposition。
    """
    comp = _e6_gate_decomposition()
    nogate = E6_GATE_ROWS[0][0]
    h1 = ["模型"] + E6_GATE_METRICS
    rows1 = []
    for cell, zh in E6_GATE_ROWS:
        rows1.append(["%s（%s）" % (cell, zh)]
                     + ["%.3f" % comp[cell][k]["plus"]
                        for k in E6_GATE_METRICS])
    for cell, zh in E6_GATE_ROWS[1:]:
        rows1.append(["抑制率（%s vs 无门）" % zh]
                     + ["%.1f%%" % (100.0 * (comp[nogate][k]["plus"]
                                             - comp[cell][k]["plus"])
                                    / comp[nogate][k]["plus"])
                        if comp[nogate][k]["plus"] >= E6_MIN_EFFECT
                        and comp[cell][k]["plus"] < comp[nogate][k]["plus"]
                        else "—"
                        for k in E6_GATE_METRICS])
    write_csv(E6_BASE / "Full_Table1_E6_GateHarmSuppression.csv", h1, rows1)
    h2 = ["模型"] + E6_GATE_METRICS
    rows2 = []
    for cell, zh in E6_GATE_ROWS:
        rows2.append(["%s（%s）" % (cell, zh)]
                     + ["%+.3f" % comp[cell][k]["minus"]
                        for k in E6_GATE_METRICS])
    for cell, zh in E6_GATE_ROWS[1:]:
        rows2.append(["保留率（%s vs 无门）" % zh]
                     + ["%.1f%%" % (100.0 * comp[cell][k]["minus"]
                                    / comp[nogate][k]["minus"])
                        if abs(comp[nogate][k]["minus"]) >= E6_MIN_EFFECT
                        and abs(comp[cell][k]["minus"]) >= E6_MIN_EFFECT
                        else "—"
                        for k in E6_GATE_METRICS])
    write_csv(E6_BASE / "Full_Table2_E6_GateGainPreservation.csv", h2, rows2)


def export_e6() -> None:
    """E6 各位置误差指标文件夹同模式导出（2026-11-02 用户裁定）。
    门控主表（抑制/保留）在 E6 根层，由 export_e6_gate_tables 导出。"""
    for metric in POSITION_METRICS:
        _export_e6_metric(metric)
    export_e6_gate_tables()


def _export_e6_metric(metric: str) -> None:
    e6_dir = E6_BASE / metric
    supp = e6_dir / "supplementary"
    detail = e6_dir / "detail"
    data = json.loads((detail / "e6_table.json").read_text(encoding="utf-8"))
    header = ["动作", "模型", "脚", "相位"] + ["V误差", "VT误差", "G(V−VT)"]
    rows = []
    for group in E6_GROUP_ORDER:
        if group not in data:
            continue
        table = data[group]["table"]
        for model in sorted(table):
            for foot in ("left", "right"):
                for phase in ("0", "1", "2", "3"):
                    v = table[model][foot].get(phase)
                    if v is None:
                        continue
                    rows.append([group, model, foot, E6_PHASE_ZH[phase],
                                 round(v["V"], 3), round(v["VT"], 3), round(v["G"], 3)])
    write_csv(supp / "Full_Table1_E6_PhaseGain.csv", header, rows)
    # 相位简表（每动作一行，四相位 × 双门/无门，G=V−VT 论文口径）：
    # 2026-11-03 起降级进 supplementary（E6 主表 = 根层门控两表）
    write_csv(supp / "Full_Table1_E6_PhaseGain_Simple.csv",
              *e6_simple_csv(data))
    # 表 2 = 多指标相位表（2026-11-02 用户裁定）：足垂直误差 / 地面穿透 /
    # 接触准确率，逐相位 V / VT / Δ(VT−V)。误差与穿透 Δ 为负 = 触觉有帮助；
    # 接触准确率 Δ 为正 = 触觉有帮助（方向随指标，见指标名）。
    # 足类指标与位置误差指标无关，各指标文件夹内容相同（同模式复制）。
    metrics_src = detail / "e6_table_metrics.json"
    if metrics_src.is_file():
        mdata = json.loads(metrics_src.read_text(encoding="utf-8"))
        header2 = ["动作", "模型", "脚", "相位", "指标", "V", "VT", "Δ(VT−V)"]
        rows2 = []
        for group in E6_GROUP_ORDER:
            if group not in mdata:
                continue
            mtable = mdata[group]["table"]
            for model in sorted(mtable):
                for foot in ("left", "right"):
                    for phase in ("0", "1", "2", "3"):
                        row = mtable[model][foot].get(phase)
                        if not row:
                            continue
                        for mk, zh in E6_METRIC_ZH.items():
                            d = row.get(mk)
                            if not d:
                                continue
                            rows2.append([group, model, foot, E6_PHASE_ZH[phase],
                                          zh, round(d["V"], 3),
                                          round(d["VT"], 3), round(d["G"], 3)])
        write_csv(supp / "Full_Table2_E6_PhaseMetrics.csv", header2, rows2)
    # 新版口径（2026-10-31 用户裁定）：驱动直写逐会话 1×2 上下分布图
    # Full_Fig_{action}_{sid}.png（上左脚/下右脚，分脚不合并不平均）；
    # group_compare/{curves,scatter}/ 由 results_display/script/
    # e6_group_compare.py 生成。旧合并双足命名
    # （Full_FigN_E6_Phase_{group}.png / e6_phase_*.png）已废止，不再复制。
    n_figs = len(list((detail / "per_session").glob("Full_Fig_*.png")))
    print("[png] E6[%s] per-session figures: %d (Full_Fig_{action}_{sid}.png)"
          % (metric, n_figs))


def e6_simple_csv(data: dict) -> tuple[list[str], list[list[str]]]:
    """E6 简表（2026-11-02 用户裁定）：每动作一行，四相位 × 双门/无门。
    口径 = 论文 G = V − VT（正=触觉有帮助），左右脚均值（缺相位留空）；
    注意完整表 G 列为 Δ=VT−V（正=伤害），两者互为反号。"""
    simple_models = ("V7_Cv5_s1", "V7_Cv7_s1")
    header = ["动作", "会话数"] + [
        f"{zh}G_{m}" for zh in E6_PHASE_ZH.values() for m in simple_models]
    rows = []
    for group in E6_GROUP_ORDER:
        if group not in data:
            continue
        table = data[group]["table"]
        line = [group, str(len(data[group]["sessions"]))]
        for phase in ("0", "1", "2", "3"):
            for m in simple_models:
                vals = [table[m][foot][phase]["V"] - table[m][foot][phase]["VT"]
                        for foot in ("left", "right")
                        if table[m][foot].get(phase) is not None]
                line.append(f"{sum(vals) / len(vals):+.1f}" if vals else "")
        rows.append(line)
    return header, rows


def _touch_gait_python() -> str:
    """touch_gait 解释器（E6 明细图/teaser 子进程需要 cv2/torch；调用方可能
    在 base 环境跑本导出脚本，子进程必须显式用 touch_gait python）。"""
    import os
    cand = Path.home() / "miniconda3/envs/touch_gait/bin/python"
    if cand.is_file():
        return str(cand)
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).name == "touch_gait":
        return str(Path(prefix) / "bin/python")
    return sys.executable


def export_e6_group_compare(models: list[str]) -> None:
    """--detail（2026-10-31 用户裁定）：E6 明细对比图——group_compare/
    {curves,scatter}/，逐会话 1×2 上下分布（上左脚/下右脚）、左右脚分开
    不平滑。生成器 = results_display/script/e6_group_compare.py
    （--models 缺省七模型；--actions 过滤动作；--metric 逐指标文件夹）。"""
    for metric in POSITION_METRICS:
        cmd = [_touch_gait_python(),
               str(REPO / "results_display/script/e6_group_compare.py"),
               "--metric", metric]
        if models:
            cmd += ["--models", ",".join(models)]
        subprocess.run(cmd, cwd=str(REPO), check=True)


def export_e6_best_sessions() -> None:
    """--best-sessions（2026-11-02 用户裁定）：E6 门控最佳完整会话合并图
    Full_Fig_E6_BestSessions.png + e6_best_sessions.json——逐会话「滤伤害
    +保增益」双条件选会话（阈值随指标缩放），子图为该会话
    Full_Fig 原图原样嵌入，无合格会话的动作红框。生成器 =
    results_display/script/e6_best_sessions.py（--metric 逐指标文件夹）。"""
    for metric in POSITION_METRICS:
        cmd = [_touch_gait_python(),
               str(REPO / "results_display/script/e6_best_sessions.py"),
               "--metric", metric]
        subprocess.run(cmd, cwd=str(REPO), check=True)


# ---------------- E7：动态输入（json + PNG） ----------------
def export_e7() -> None:
    """统一入口（2026-10-31 用户裁定）：先跑 anysole_tta/e7_driver.py
    生成图 + e7_table.json，再导出 CSV（2026-11-02 起逐位置误差指标
    子文件夹各跑一遍）。TTA dump 已有时驱动自动跳过
    （仅装配+绘图，分钟级）；缺 dump 的动作会触发 TTA 子进程。
    驱动需 touch_gait（cv2），用 _touch_gait_python 显式指定
    （2026-11-02 修复：原 sys.executable 在 base 环境缺 cv2 失败）。"""
    import subprocess
    driver = REPO / "anysole_tta" / "e7_driver.py"
    for metric in POSITION_METRICS:
        dst_dir = E7_BASE / metric
        print("[e7] run driver: %s --metric %s" % (driver, metric))
        subprocess.run([_touch_gait_python(), str(driver), "--metric", metric],
                       check=True, cwd=str(REPO))
        data = json.loads((dst_dir / "detail" / "e7_table.json").read_text(encoding="utf-8"))
        stages = ["V", "T", "VT", "Vi", "ViT"]
        systems = sorted({s for a in data.values() for st in a["stages"].values() for s in st})
        header = ["动作", "系统"] + stages
        rows = []
        for action, entry in sorted(data.items()):
            for sys_name in systems:
                rows.append([action, sys_name]
                            + [round(entry["stages"][st].get(sys_name, float("nan")), 3)
                               if entry["stages"][st].get(sys_name) is not None else ""
                               for st in stages])
        write_csv(dst_dir / "Full_Table1_E7_DynamicStages.csv", header, rows)


# ---------------- teaser（E6/E7 可见人体 SMPL） ----------------
def export_teasers(exp: str) -> None:
    """--teaser：E6/E7 teaser 图（anysole_tta/e_teaser.py，--count 按总时长
    1/N 间距抽取，偏航置零侧视）。需 touch_gait 环境。默认关闭。"""
    import subprocess
    # HMR2.0 线已从 z_note/HMR2.0/hmr2_gait 迁到仓库根 anysole_tta/
    # （2026-10-31 用户重组；e_teaser.py 现位于 REPO/anysole_tta/）。
    script = REPO / "anysole_tta/e_teaser.py"
    cmds = []
    if exp in ("all", "e6"):
        cmds.append([_touch_gait_python(), str(script), "--mode", "e6"])
    if exp in ("all", "e7"):
        cmds.append([_touch_gait_python(), str(script), "--mode", "e7"])
    for cmd in cmds:
        print("[teaser]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(REPO))
    print("[teaser] done")


# ---------------- 报告 md 迁移 ----------------
def migrate_reports() -> None:
    """报告 md 已随 2026-10-27 迁移落位 E 目录；保留空实现占位。"""


def main() -> int:
    global CONTACT_NAMESPACE, EVAL_SPLIT, ETEST, CELL_MAP, STRICT_EXPORT, E5_DIR, E5_SUPP, E5_DETAIL, E6_BASE, E7_BASE, RECEIPTS
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="all",
                    help="只导出指定实验（E1..E7；缺省 = 全部）")
    ap.add_argument("--models", default="",
                    help="逗号分隔模型过滤（E1 = 家族如 V7_Cv5；E2/E4/E5 = cell 名子串）")
    ap.add_argument("--ckpt", choices=("last", "best"), default="last",
                    help="ckpt 口径（缺省 last；best 需对应 cells 已注册）")
    ap.add_argument("--contact-version", choices=("legacy", "time_accel_v1", "state_v2", "state_smpl24", "tafix_smpl24"), default="state_smpl24")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cell-map", type=Path, help="Explicit cell ID to native result JSON mapping")
    ap.add_argument("--strict", action="store_true", help="Fail rather than silently omit required results")
    ap.add_argument("--detail", action="store_true",
                    help="E6 生成明细对比图（group_compare/{curves,scatter}/："
                         "逐会话 1×2 上下分布、左右脚分开不平滑）")
    ap.add_argument("--teaser", action="store_true",
                    help="E6/E7 生成可见人体 SMPL teaser 图（默认关闭；需 touch_gait）")
    ap.add_argument("--best-sessions", action="store_true",
                    help="E6 生成门控最佳完整会话合并图（Full_Fig_E6_BestSessions.png；"
                         "滤伤害+保增益双条件选会话，无合格会话红框）")
    args = ap.parse_args()
    CONTACT_NAMESPACE, EVAL_SPLIT, STRICT_EXPORT = args.contact_version, args.split, args.strict
    if args.cell_map:
        CELL_MAP = json.loads(args.cell_map.read_text())
    if args.out_dir:
        ETEST = args.out_dir
        # E5/E6/E7/receipts 目录在 import 时按默认 ETEST 拼成常量；--out-dir 必须重算，
        # 否则这些实验仍写回 Etest 旧目录，覆盖按 test 口径生成的论文表图（2026-11-04 act8020 val 导出前发现）。
        E5_DIR = ETEST / "E5_OcclusionRobustness"
        E5_SUPP = E5_DIR / "supplementary"
        E5_DETAIL = E5_DIR / "detail"
        E6_BASE = ETEST / "E6_ContactPhaseAnalysis"
        E7_BASE = ETEST / "E7_DynamicModalityInputs"
        RECEIPTS = ETEST / "receipts"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    exp = args.experiment.lower()
    if exp in ("all", "e1"):
        export_e1(models)
    if exp in ("all", "e2"):
        export_e2(models, args.ckpt)
    if exp in ("all", "e3"):
        export_e3()
    if exp in ("all", "e4"):
        export_e4(models, args.ckpt)
    if exp in ("all", "e5"):
        export_e5(models, args.ckpt)
    if exp in ("all", "e6"):
        export_e6()
        if args.detail:
            export_e6_group_compare(models)
    if exp in ("all", "e7"):
        export_e7()
    if args.teaser:
        if exp in ("all", "e6", "e7"):
            export_teasers(exp)
        else:
            print("[skip] --teaser 只适用于 E6/E7")
    if args.best_sessions:
        if exp in ("all", "e6"):
            export_e6_best_sessions()
        else:
            print("[skip] --best-sessions 只适用于 E6")
    ETEST.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / ("export_receipt_" + exp + ".json")).write_text(json.dumps({
        "split": EVAL_SPLIT, "contact_version": CONTACT_NAMESPACE,
        "sources": SOURCE_RECEIPTS}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
