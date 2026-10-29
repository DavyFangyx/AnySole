#!/usr/bin/env python3
"""E 系列实验结果导出器（2026-10-27 用户裁定：results_display 存展示产物）。

数据源与去向：
- E1：z_note/E_Test/C版本对照/E1_同架构专家对比.md 表格 → E1_.../Full_Table1_E1_Experts.csv
- E2：cells（A/B/C last/best + Cv1–7 last）+ 外部基线 md → Full_Table1_E2_Performance.csv
- E3：Etest/10_21/e3_multiview.json → Full_Table1_E3_MultiView.csv
- E4：cells（Cv1–7 stat，三条件十指标 + Δ）→ Full_Table1_E4_Ablation_AllMetrics.csv
- E5：cells（A/B/C last/best × occ 四档 + Cv1–7 last occ）→ Full_Table1_E5_Occlusion.csv
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


def md_tables(path: Path) -> list[dict]:
    """解析 md 中的 pipe 表：返回 [{section, header, rows}]。"""
    tables2, cur2, section2 = [], None, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("##") or line.startswith("###"):
            section2 = line.lstrip("#").strip()
        if not line.strip().startswith("|"):
            if cur2 is not None:
                tables2.append(cur2)
                cur2 = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cur2 is None:
            cur2 = {"section": section2, "header": cells, "rows": []}
        elif all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
            continue
        else:
            cur2["rows"].append(cells)
    if cur2 is not None:
        tables2.append(cur2)
    return tables2


def display_name(cell_id: str) -> str:
    """cell id → 展示模型名：V7_Cv5_statistics_evalocc010 → V7_Cv5。"""
    for suffix in ("_statistics_evalocc", "_evalbest_occ", "_evalocc", "_evalbest",
                   "_statistics", "_sel_evalocc", "_sel"):
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
    """自动扫描 base 与 *_sN 种子变体 cell，聚合为 mean / mean±std
    （2026-10-29 用户裁定：方案 A——多 seed 存 mean±std、单 seed 存单值）。

    扫描在 namespace 映射后的目录名上进行（变体 = mapped_key + _sN）；
    聚合只在同一条件内做；值为标量 = 单 seed，值为 (mean, std) = 多 seed。
    """
    key = _namespace_cell(model_id)
    # _sN 插在子后缀前（与 conf 层 seed_models 一致）：
    # V7_Cv5_statefix_smpl24_statistics → V7_Cv5_statefix_smpl24_s2_statistics
    sub_pos = None
    for sub in ("_statistics", "_evalocc", "_evalbest", "_precheck", "_sel"):
        pos = key.rfind(sub)
        if pos != -1 and pos > 0:
            sub_pos = pos
            break
    def _variant(n):
        if n == 1:
            return key
        if sub_pos is not None:
            return key[:sub_pos] + "_s%d" % n + key[sub_pos:]
        return key + "_s%d" % n

    cands = [key] + sorted(
        p.name for p in (CELLS).iterdir()
        if p.name in {_variant(2), _variant(3)})
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
# 专家口径（2026-10-26 裁定）：Cv1–4 家族共享 A 骨架专家 V7_SV1/V7_ST1；
# Cv5 家族 = 同架构专家 V7_Cv5_SV1_stat/V7_Cv5_ST1_stat（v7_c2 双门）。
EXPERT_CELLS = {"V7_SV1": "S_V@V", "V7_ST1": "S_T@T",
                "V7_Cv5_SV1_statistics": "S_V@V", "V7_Cv5_ST1_statistics": "S_T@T"}


def export_e1(models: list[str]) -> None:
    header = ["家族", "测试设定", "模型"] + METRICS10
    rows = []
    families = models or ["V7_Cv%d" % n for n in range(1, 8)]
    for fam in families:
        u = seeded_metrics(fam + ("_f6_statistics" if CONTACT_NAMESPACE == "legacy" else "_statistics"))
        if not u:
            print("[skip] E1 %s: %s_stat 无 cell 结果" % (fam, fam))
            continue
        for cond, setting in (("T2M", "T"), ("V2M", "V"), ("VT2M", "VT")):
            rows.append([fam, setting, "U@" + {"T": "T", "V": "V", "VT": "VT"}[setting]]
                        + metric_row(u, cond))
        if CONTACT_NAMESPACE != "legacy" and fam not in ("V7_Cv2", "V7_Cv5"):
            raise ValueError("E1 needs explicitly matched experts for family: " + fam)
        experts = (["V7_Cv5_SV1_statistics", "V7_Cv5_ST1_statistics"] if fam in ("V7_Cv2", "V7_Cv5")
                   else ["V7_SV1", "V7_ST1"])
        for cell, setting in zip(experts, ("S_V@V", "S_T@T")):
            e = seeded_metrics(cell)
            if not e:
                print("[skip] E1 %s: 专家 cell %s 无结果" % (fam, cell))
                continue
            cond = "V2M" if setting == "S_V@V" else "T2M"
            rows.append([fam, setting[2], setting] + metric_row(e, cond))
    write_csv(ETEST / "E1_CleanPerformance" / "Full_Table1_E1_Experts.csv", header, rows)


# ---------------- E2：主性能（cells 驱动 + 基线 md） ----------------
def _selected(cell_ids: list[str], models: list[str]) -> list[str]:
    return [c for c in cell_ids if not models or any(m in c for m in models)]


def export_e2(models: list[str], ckpt: str) -> None:
    """四张表（2026-10-27 用户裁定）：V 块 / T 块 / VT 块 / 汇总。

    模型类型列：外部基线 = 四外部方法；Ours = AnySole 自家对照模型
    （V4_Av1_wo05_drop / V5_Av4_wo05_drop，wo05 时代）；U = 统一模型。
    """
    header = ["类型", "模型", "口径"] + METRICS10
    # Cv1 的 first_round cell 已删（双 id 遗留）；Cv1 行由 V7_Cv1_stat 提供
    # （display 层只读专门统计测试 cell，2026-10-27 用户裁定）
    last_cells = _selected(["V7_Av1", "V7_Bv1"]
                           + ["V7_Cv%d%s_statistics" % (n, "_f6" if CONTACT_NAMESPACE == "legacy" else "") for n in range(1, 8)], models)
    best_cells = _selected(["V7_Av1_evalbest", "V7_Bv1_evalbest", "V7_Cv1_evalbest"], models)
    rows_by_block = {"V": [], "T": [], "VT": []}
    for cond, block in (("V2M", "V"), ("T2M", "T"), ("VT2M", "VT")):
        if ckpt != "best":
            for model in last_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["S_VT" if model == "V7_Cv6_statistics" else "U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "last"]
                        + metric_row(m, cond))
        if ckpt != "last":
            for model in best_cells:
                m = seeded_metrics(model)
                if m and cond in m:
                    rows_by_block[block].append(
                        ["U", "%s（%s）" % (COND_ZH[cond], display_name(model)), "best"]
                        + metric_row(m, cond))
    # 基线表（md 驱动；首列即块）。Ours（V4_Av1_wo05_drop 等）是 AnySole
    # 自家模型但不属于本实验默认生成名单 → 不入表（2026-10-27 用户裁定）。
    base_src = (REPO / "results_display/Etest/E2_MainPerformance/E2_外部基线主结果.md") if EVAL_SPLIT == "test" else Path("/nonexistent/validation_external_baselines")
    if base_src.is_file():
        for t in md_tables(base_src):
            for r in t["rows"]:
                if r[1].startswith("Ours"):
                    continue
                rows_by_block.setdefault(r[0], []).append(
                    ["外部基线", r[1], "—"] + r[2:])
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
    cells = _selected(["V7_Cv%d%s_statistics" % (n, "_f6" if CONTACT_NAMESPACE == "legacy" else "") for n in range(1, 8)], models)
    for cell_id in cells:
        m = seeded_metrics(cell_id)
        if not m:
            continue
        for cond in (("VT2M",) if cell_id == "V7_Cv6_statistics" else CONDS):
            rows.append([display_name(cell_id), "S_VT@VT" if cell_id == "V7_Cv6_statistics" else COND_ZH[cond]] + metric_row(m, cond))
        if cell_id == "V7_Cv6_statistics":
            continue
        rows.append([display_name(cell_id), "Δ(VT−V)"] + delta_row(m))
    write_csv(ETEST / "E4_ComponentAblation" / "Full_Table1_E4_Ablation_AllMetrics.csv",
              header, rows)


# ---------------- E5：遮挡（cells 驱动，档×行） ----------------
def export_e5(models: list[str], ckpt: str) -> None:
    header = ["模型", "口径", "档位", "行"] + METRICS10
    rows = []
    last_cells = {"occ000": ["V7_Av1", "V7_Bv1", "V7_Cv1"]}
    best_cells = {"occ000": ["V7_Av1_evalbest", "V7_Bv1_evalbest", "V7_Cv1_evalbest"]}
    for occ in ("010", "020", "030", "050"):
        last_cells["occ%s" % occ] = ["V7_Av1_evalocc%s" % occ,
                                     "V7_Bv1_evalocc%s" % occ,
                                     "V7_Cv1_evalocc%s" % occ]
        best_cells["occ%s" % occ] = ["V7_Av1_evalbest_occ%s" % occ,
                                     "V7_Bv1_evalbest_occ%s" % occ,
                                     "V7_Cv1_evalbest_occ%s" % occ]
    for occ, ids in last_cells.items():
        for n in range(1, 8):
            if occ == "occ000":
                ids.append("V7_Cv%d%s_statistics" % (n, "_f6" if CONTACT_NAMESPACE == "legacy" else ""))
            else:
                ids.append("V7_Cv%d%s_statistics_evalocc%s" % (n, "_f6" if CONTACT_NAMESPACE == "legacy" else "", occ[3:]))
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
    cells = {"occ000": model + "_statistics"}
    for occ in ("010", "020", "030", "050"):
        cells["occ%s" % occ] = "%s_statistics_evalocc%s" % (model, occ)
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
E3_LINES = ["标定多视角（参考上限）", "TTA 原方法", "朴素无标定晚融合",
            "U@V{i}（TTA 框架）", "U@V{i} + 接触约束（TTA 框架）"]


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


def export_e3() -> None:
    src = ETEST / "E3_MultiViewConsistency" / "e3_multiview.json"
    plain = {}
    if src.is_file():
        data = json.loads(src.read_text(encoding="utf-8"))
        plain = {n: data["plain_fusion"][n] for n in data.get("plain_fusion", {})}
    # U@V{i}（TTA 框架）已落地产物（E3 口径说明 2026-10-27）：
    # V3_Bv1 i=1 = cam3_init（val 11 会话）；i=4 = stage1 lr0.06 定案
    # pa_virt_refined（精修后 virtual 聚合，30.7）。i=2/3 无子集支持；
    # V7C Cv1 运行中、Cv2–5 未跑——均填 —。
    tta_metrics = REPO / "results/baselines/HMR2.0/metrics"
    header = ["对比线", "模型", "视角数", "PA-MPJPE_mean", "PA-MPJPE_std",
              "MPJPE_mean", "MPJPE_std", "GlobalMPJPE_mean", "GlobalMPJPE_std"]
    rows = []
    for line in E3_LINES:
        for n in ("1", "2", "3", "4"):
            if line == "朴素无标定晚融合" and n in plain:
                v = plain[n]
                rows.append([line, "V3_Bv1", n, "—", "—",
                             round(v.get("mpjpe_mean", 0), 3),
                             round(v.get("mpjpe_std", 0), 3),
                             round(v.get("gmpjpe_mean", 0), 3),
                             round(v.get("gmpjpe_std", 0), 3)])
            elif line == "U@V{i}（TTA 框架）" and n in ("1", "4"):
                if n == "1":
                    mean, std = _stage1_agg(
                        tta_metrics / "anysole_tta_stage1_val_cam3_init.json",
                        "pa_init_cam3")
                else:
                    mean, std = _stage1_agg(
                        tta_metrics / "anysole_tta_stage1_val_lr0.06.json",
                        "pa_virt_refined")
                rows.append([line, "V3_Bv1", n, mean, std, "—", "—", "—", "—"])
            else:
                rows.append([line, "—", n, "—", "—", "—", "—", "—", "—"])
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
    for group in E6_GROUP_ORDER:
        # 新驱动直写 Full_ 命名 PNG；旧命名（e6_phase_*.png）仅作兜底
        src_png = dst_dir / ("Full_Fig%d_E6_Phase_%s.png"
                             % (E6_ACTION_VALUES.index(group) + 1, group))
        if not src_png.is_file():
            src_png = src_dir / ("e6_phase_%s.png" % group)
        if src_png.is_file():
            dst_png = dst_dir / ("Full_Fig%d_E6_Phase_%s.png"
                                 % (E6_ACTION_VALUES.index(group) + 1, group))
            if src_png != dst_png:
                shutil.copyfile(src_png, dst_png)
                print("[png] %s" % dst_png)


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
        src_png = src_dir / ("e7_phase_%s.png" % action)
        if src_png.is_file():
            dst_png = dst_dir / ("Full_Fig%d_E7_Stages_%s.png" % (i, action))
            shutil.copyfile(src_png, dst_png)
            print("[png] %s" % dst_png)


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
    args = ap.parse_args()
    CONTACT_NAMESPACE, EVAL_SPLIT, STRICT_EXPORT = args.contact_version, args.split, args.strict
    if args.cell_map:
        CELL_MAP = json.loads(args.cell_map.read_text())
    if args.out_dir:
        ETEST = args.out_dir
    elif CONTACT_NAMESPACE != "legacy" or EVAL_SPLIT != "test":
        ETEST = REPO / "results_display" / ("Etest_" + CONTACT_NAMESPACE + "_" + EVAL_SPLIT)
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
    if exp in ("all", "e7"):
        export_e7()
    ETEST.mkdir(parents=True, exist_ok=True)
    (ETEST / ("export_receipt_" + exp + ".json")).write_text(json.dumps({
        "split": EVAL_SPLIT, "contact_version": CONTACT_NAMESPACE,
        "sources": SOURCE_RECEIPTS}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
