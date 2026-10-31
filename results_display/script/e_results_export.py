#!/usr/bin/env python3
"""E 系列实验结果导出器（2026-10-27 用户裁定：results_display 存展示产物）。

数据源与去向：
- E1：z_note/E_Test/C版本对照/E1_同架构专家对比.md 表格 → E1_.../Full_Table1_E1_Experts.csv
- E2：cells（Cv1–7 last / V7_Cv1_evalbest）+ 基线公共评估产物（results/baselines/*/metrics/test_comparison.json）→ Full_Table1_E2_Performance.csv
- E3：Etest/10_21/e3_multiview.json → Full_Table1_E3_MultiView.csv
- E4：cells（Cv1–7 stat，三条件十指标 + Δ）→ Full_Table1_E4_Ablation_AllMetrics.csv
- E5：主模型 V7_Cv5（last/best × occ000-050 五档）→ Full_Table1_E5_Occlusion.csv
- E6：E6_接触相位/e6_table.json + PNG → Full_Table1_E6_PhaseGain.csv + Full_FigN_E6_Phase_{g}.png
- E7：E7_动态输入/e7_table.json + PNG → Full_Table1_E7_DynamicStages.csv + Full_FigN_E7_Stages_{a}.png

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
# z_note/E_Test 已并入 E 目录（2026-10-27）；下方源路径全部指向 E 目录
CELLS = REPO / "results" / "cells"

METRICS10 = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
             "traj_ATE", "RootOrientationDrift", "FootSliding", "GroundPen",
             "Jitter_pred"]
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
                   "_evalocc", "_evalbest", "_statistics", "_sel_evalocc", "_sel"):
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
    return payload.get("metrics")


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
    for sub in ("_evalocc", "_evalbest", "_precheck", "_sel", "_statistics"):
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
        if p.name in {key, _variant(1), _variant(2), _variant(3)})
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


def metric_row(m: dict, cond: str) -> list:
    out = []
    for k in METRICS10:
        v = m[cond].get(k)
        if v is None:
            out.append("")
        elif isinstance(v, tuple):
            out.append("%.3f±%.3f" % v)
        else:
            out.append(round(float(v), 3))
    return out


def delta_row(m: dict) -> list:
    """Δ(VT−V) 行：多 seed 按 seed 对先算 Δ 再聚合（误差传播）。"""
    out = []
    for k in METRICS10:
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
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}


def _baseline_rows() -> dict[str, list[list]]:
    """基线行 = 各基线公共评估产物 test_comparison.json（2026-10-31 裁定：
    数据驱动，不读 md）。mode 决定块；V2T 与无运动指标者不产出行。"""
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
        for key in BASELINE_METRIC_KEYS.values():
            v = s.get(key)
            row.append(round(v, 3) if isinstance(v, (int, float)) and np.isfinite(v) else "—")
        rows_by_block[block].append(row)
    return rows_by_block


def export_e2(models: list[str], ckpt: str) -> None:
    """四张表（2026-10-27 用户裁定）：V 块 / T 块 / VT 块 / 汇总。

    模型类型列：外部基线 = 各基线公共评估产物（mode 决定块）；U = 统一模型
    （Cv1–7，Cv6 为 S_VT 专家角色）。旧候选 A/B 不入表（2026-10-30 用户裁定）。
    """
    header = ["类型", "模型", "口径"] + METRICS10
    # 当前线：训练 cell 即评估 cell（V7_CvN → 自动扫 _s1/_s2/_s3）；
    # legacy 线保持 _f6_statistics 旧命名（display 层只读专门测试 cell）
    last_cells = _selected(["V7_Cv%d%s%s" % (n,
                                           "_f6" if CONTACT_NAMESPACE == "legacy" else "",
                                           "_statistics" if CONTACT_NAMESPACE == "legacy" else "")
                              for n in range(1, 8)], models)
    best_cells = _selected(["V7_Cv1_evalbest"], models)
    rows_by_block = {"V": [], "T": [], "VT": []}
    for cond, block in (("V2M", "V"), ("T2M", "T"), ("VT2M", "VT")):
        if ckpt != "best":
            for model in last_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["S_VT" if model == "V7_Cv6" else "U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "last"]
                        + metric_row(m, cond))
        if ckpt != "last":
            for model in best_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "best"]
                        + metric_row(m, cond))
    # 基线行：公共评估产物驱动（Ours 自身模型不在此目录，天然不入表）
    for block, rows in _baseline_rows().items():
        rows_by_block[block].extend(rows)
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
    # E5 = 单一主模型（2026-10-30 用户裁定）：U = V7_Cv5 一份 ckpt × occ 五档
    model = models[0] if models else "V7_Cv5"
    last_cells = {"occ000": [model]}
    best_cells = {"occ000": ["%s_evalbest" % model]}
    for occ in ("010", "020", "030", "050"):
        last_cells["occ%s" % occ] = ["%s_evalocc%s" % (model, occ)]
        best_cells["occ%s" % occ] = ["%s_evalbest_occ%s" % (model, occ)]
    for occ in ("occ000", "occ010", "occ020", "occ030", "occ050"):
        last_ids = _selected(last_cells[occ], models)
        best_ids = _selected(best_cells.get(occ, []), models)
        for model in last_ids if ckpt != "best" else []:
            m = seeded_metrics(model)
            if not m:
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                rows.append([display_name(model), "last", occ, row_key]
                            + metric_row(m, cond))
            rows.append([display_name(model), "last", occ, "Δ(VT−V)"]
                        + delta_row(m))
        for model in best_ids if ckpt != "last" else []:
            m = seeded_metrics(model)
            if not m:
                continue
            for row_key, cond in (("V2M", "V2M"), ("VT2M", "VT2M")):
                rows.append([display_name(model), "best", occ, row_key]
                            + metric_row(m, cond))
            rows.append([display_name(model), "best", occ, "Δ(VT−V)"]
                        + delta_row(m))
    write_csv(ETEST / "E5_OcclusionRobustness" / "Full_Table1_E5_Occlusion.csv",
              header, dedupe(rows))
    export_e5_figure(models)


def export_e5_figure(models: list[str]) -> None:
    """E5 图 1：退化曲线（2026-10-27 用户要求）。

    x = 遮挡档（0/10/20/30/50%），y = 相对 occ000 的退化（Δ），
    V（蓝实线）与 VT（橙虚线）两条曲线，子图 = 六个关键指标——
    判读：VT 退化应轻于 V（曲线更低/更平）。主模型缺省 V7_Cv5（last 口径）。
    图内英文（项目口径）。产物：E5_OcclusionRobustness/Full_Fig1_E5_Degradation.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = models[0] if models else "V7_Cv5"
    cells = {"occ000": model}
    for occ in ("010", "020", "030", "050"):
        cells["occ%s" % occ] = "%s_evalocc%s" % (model, occ)
    metrics = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE",
               "GroundPen", "FootSliding"]
    xs = [0, 10, 20, 30, 50]
    color_v, color_vt = "#2a78d6", "#eb6834"  # 校验过的双色（CVD 分离 24.7）
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.8))
    for ax, metric in zip(axes.flat, metrics):
        base = seeded_metrics(cells["occ000"])
        if not base:
            continue
        def _val(v):
            return v[0] if isinstance(v, tuple) else float(v)

        def series(cond):
            vals = []
            for occ, cell in cells.items():
                m = seeded_metrics(cell)
                vals.append(_val(m[cond][metric]) - _val(base[cond][metric])
                            if m else float("nan"))
            return vals
        v, vt = series("V2M"), series("VT2M")
        ax.plot(xs, v, color=color_v, lw=2, marker="o", ms=5, label="V")
        ax.plot(xs, vt, color=color_vt, lw=2, ls="--", marker="s", ms=5, label="VT")
        if all(np.isfinite(v)) and all(np.isfinite(vt)):
            ax.annotate("V", (xs[-1], v[-1]), color=color_v, fontweight="bold",
                        xytext=(4, 0), textcoords="offset points", va="center")
            ax.annotate("VT", (xs[-1], vt[-1]), color=color_vt, fontweight="bold",
                        xytext=(4, 0), textcoords="offset points", va="center")
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Occlusion (%)", fontsize=9)
        ax.set_ylabel("Δ vs occ000 (mm)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.margins(y=0.15)
    fig.suptitle("E5 degradation under occlusion - %s (ckpt_last)" % model,
                 fontsize=12)
    fig.legend(handles=axes.flat[0].lines, labels=["V", "VT"],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    out = ETEST / "E5_OcclusionRobustness" / "Full_Fig1_E5_Degradation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % out)


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


def export_e6() -> None:
    src_dir = ETEST / "E6_ContactPhaseAnalysis"
    dst_dir = ETEST / "E6_ContactPhaseAnalysis"
    data = json.loads((src_dir / "e6_table.json").read_text(encoding="utf-8"))
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
    write_csv(dst_dir / "Full_Table1_E6_PhaseGain.csv", header, rows)
    # 新版口径（2026-10-31 用户裁定）：驱动直写逐会话 1×2 上下分布图
    # Full_Fig_{action}_{sid}.png（上左脚/下右脚，分脚不合并不平均）；
    # group_compare/{curves,scatter}/ 由 results_display/script/
    # e6_group_compare.py 生成。旧合并双足命名
    # （Full_FigN_E6_Phase_{group}.png / e6_phase_*.png）已废止，不再复制。
    n_figs = len(list(dst_dir.glob("Full_Fig_*.png")))
    print("[png] E6 per-session figures: %d (Full_Fig_{action}_{sid}.png)"
          % n_figs)


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
    （--models 缺省七模型；--actions 过滤动作）。"""
    cmd = [_touch_gait_python(),
           str(REPO / "results_display/script/e6_group_compare.py")]
    if models:
        cmd += ["--models", ",".join(models)]
    subprocess.run(cmd, cwd=str(REPO), check=True)


# ---------------- E7：动态输入（json + PNG） ----------------
def export_e7() -> None:
    src_dir = ETEST / "E7_DynamicModalityInputs"
    dst_dir = ETEST / "E7_DynamicModalityInputs"
    data = json.loads((src_dir / "e7_table.json").read_text(encoding="utf-8"))
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
    for i, action in enumerate(("walk", "run", "squat"), 1):
        # 新驱动直写 Full_ 命名 PNG；旧命名（e7_phase_*.png）仅作兜底
        src_png = dst_dir / ("Full_Fig%d_E7_Stages_%s.png" % (i, action))
        if not src_png.is_file():
            src_png = src_dir / ("e7_phase_%s.png" % action)
        if src_png.is_file():
            dst_png = dst_dir / ("Full_Fig%d_E7_Stages_%s.png" % (i, action))
            if src_png != dst_png:
                shutil.copyfile(src_png, dst_png)
                print("[png] %s" % dst_png)


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
    global CONTACT_NAMESPACE, EVAL_SPLIT, ETEST, CELL_MAP, STRICT_EXPORT
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
    args = ap.parse_args()
    CONTACT_NAMESPACE, EVAL_SPLIT, STRICT_EXPORT = args.contact_version, args.split, args.strict
    if args.cell_map:
        CELL_MAP = json.loads(args.cell_map.read_text())
    if args.out_dir:
        ETEST = args.out_dir
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
    ETEST.mkdir(parents=True, exist_ok=True)
    (ETEST / ("export_receipt_" + exp + ".json")).write_text(json.dumps({
        "split": EVAL_SPLIT, "contact_version": CONTACT_NAMESPACE,
        "sources": SOURCE_RECEIPTS}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
