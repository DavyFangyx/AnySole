#!/usr/bin/env python3
"""Consume versioned RGB-SMPL tables; publish paired audits and Chinese report."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from AnysoleWorkspace.tool.eval_rgb_smpl import (  # noqa: E402
    BASELINES, REFERENCE, VERSIONS, digest, read_csv, write_csv, write_json,
)
from anysole.utils.vision_smpl_metrics import REQUESTED_METRICS  # noqa: E402
from anysole.utils.metrics import paired_cluster_bootstrap  # noqa: E402

LABELS = {
    "mpjpe_mm": "MPJPE (mm)", "pa_mpjpe_mm": "PA-MPJPE (mm)",
    "mpjae_deg": "MPJAE (°)", "pve_mm": "PVE (mm)", "pve_t_mm": "PVE-T (mm)",
    "accel_error_m_s2": "ACC-ERR (m/s²)", "jitter_pred_m_s3": "Prediction Jerk (m/s³)",
    "jitter_gt_m_s3": "GT Jerk (m/s³)", "w_mpjpe100_mm": "W-MPJPE100 (mm)",
    "root_ate_mm": "Root ATE (mm)", "foot_sliding_mm": "Foot sliding (mm/contact transition)",
}
# Set after the bounded pilot, before the full run. Absolute tolerances;
# not bitwise identity. Third derivatives amplify float32 SMPL roundoff.
def tolerance(metric):
    return 0.005 if metric.startswith("jitter_") else 1e-6 if metric == "mpjae_deg" else 1e-4


def number(value):
    return None if value in (None, "", "None") else float(value)


def fmt(value):
    return "N/A" if value is None else f"{value:.6f}"


def compare(root: Path):
    comparisons, paired, audit, headline = [], [], {}, {}
    for baseline in BASELINES:
        reference = root / baseline / REFERENCE
        old_units = {row["unit_id"]: row for row in read_csv(reference / "unit_metrics.csv")}
        old_sessions = {(row["date"], row["session"]): row for row in read_csv(reference / "session_metrics.csv")}
        old_overall = next(row for row in read_csv(reference / "aggregate_metrics.csv") if row["stratum"] == "overall")
        for version in VERSIONS:
            destination = root / baseline / version
            units = read_csv(destination / "unit_metrics.csv")
            sessions = {(row["date"], row["session"]): row for row in read_csv(destination / "session_metrics.csv")}
            if {row["unit_id"] for row in units} != set(old_units) or len(units) != 2056 or len(sessions) != 257:
                raise ValueError(f"roster mismatch: {baseline}/{version}")
            if set(sessions) != set(old_sessions) or sum(int(row["frame_count"]) for row in units) != 785088:
                raise ValueError("session/frame coverage mismatch")
            overall = next(row for row in read_csv(destination / "aggregate_metrics.csv") if row["stratum"] == "overall")
            headline[(baseline, version)] = overall
            diffs = [{"unit_id": row["unit_id"], "baseline": baseline,
                      **{metric: None if metric == "pve_t_mm" else
                         float(row[metric]) - float(old_units[row["unit_id"]][metric])
                         for metric in REQUESTED_METRICS}} for row in units]
            write_csv(destination / "unit_differences_to_2_code.csv", diffs)
            details = [f"# {baseline} / {version}\n",
                       "覆盖257 sessions / 2056 units / 785088 frames；物理session宏平均。\n",
                       "| 指标 | 2_Code历史 | 当前版本 | 差值(当前−历史) | unit最大绝对差 |",
                       "|---|---:|---:|---:|---:|"]
            for metric in REQUESTED_METRICS:
                old, new = number(old_overall[metric]), number(overall[metric])
                delta = None if new is None else new - old
                max_abs = None if new is None else max(abs(row[metric]) for row in diffs)
                if new is None:
                    failures, low, high = None, None, None
                else:
                    keys = sorted(sessions)
                    values = np.array([float(sessions[key][metric]) for key in keys])
                    refs = np.array([float(old_sessions[key][metric]) for key in keys])
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite session {metric}")
                    recomputed = values.mean()
                    if abs(recomputed - new) > 1e-8:
                        raise ValueError("macro aggregation mismatch")
                    _, low, high = paired_cluster_bootstrap(values, refs)
                    failures = sum(abs(row[metric]) > tolerance(metric) for row in diffs)
                comparisons.append({"baseline": baseline, "metric_version": version, "metric": metric,
                                    "reference_value": old, "new_value": new, "delta": delta,
                                    "delta_ci95_low": low, "delta_ci95_high": high,
                                    "max_abs_unit_delta": max_abs,
                                    "abs_tolerance": tolerance(metric), "units_outside_tolerance": failures,
                                    "status": "not_applicable_removed" if new is None else
                                    "within_numeric_tolerance" if not failures else "different",
                                    "paired_sessions": 257})
                if version == VERSIONS[1] and new is not None:
                    audit[f"{baseline}/{metric}"] = {"max_abs_unit_delta": max_abs,
                        "abs_tolerance": tolerance(metric), "units_outside_tolerance": failures}
                details.append(f"| {LABELS[metric]} | {fmt(old)} | {fmt(new)} | {fmt(delta)} | {fmt(max_abs)} |")
            details.extend(["", "PVE-T 已被e9accee0废除，留空并标为N/A。",
                            "" if version == VERSIONS[0] else
                            "本组是相同输入的公式控制诊断；body21、去骨盆时序、顶点足滑，不作为AnySole原生公共指标。",
                            "来源、采样、聚合、SHA-256见run.json；unit_differences_to_2_code.csv保存逐unit差值。"])
            (destination / "report.md").write_text("\n".join(details) + "\n", encoding="utf-8")
        headline[(baseline, REFERENCE)] = old_overall
    for first, second in combinations(BASELINES, 2):
        a = {(row["date"], row["session"]): row for row in read_csv(root / first / VERSIONS[0] / "session_metrics.csv")}
        b = {(row["date"], row["session"]): row for row in read_csv(root / second / VERSIONS[0] / "session_metrics.csv")}
        keys = sorted(a)
        for metric in REQUESTED_METRICS:
            if metric in ("pve_t_mm", "jitter_gt_m_s3"):
                continue
            difference, low, high = paired_cluster_bootstrap(
                np.array([float(a[key][metric]) for key in keys]),
                np.array([float(b[key][metric]) for key in keys]))
            paired.append({"first": first, "second": second, "metric": metric,
                           "difference_first_minus_second": difference, "ci95_low": low,
                           "ci95_high": high, "paired_sessions": 257, "bootstrap_samples": 10000})
    write_csv(root / "metric_version_comparison.csv", comparisons)
    write_csv(root / "paired_baselines_anysole_e9accee0.csv", paired)
    passed = all(value["units_outside_tolerance"] == 0 for value in audit.values())
    write_json(root / "formula_control_audit.json", {"passed": passed, "checks": audit,
        "comparison": "recomputed AnySole formula-control vs historical published unit values; absolute tolerance, not bit identity",
        "coverage_per_baseline": {"sessions": 257, "units": 2056, "frames": 785088},
        "pve_t": "not_applicable_removed_in_AnySole"})
    inputs = []
    for baseline in BASELINES:
        for row in read_csv(root / baseline / VERSIONS[0] / "unit_metrics.csv"):
            directory = REPO.parent / "3_Result" / f"{baseline}_baseline" / row["short_id"]
            state = json.loads((directory / "state.json").read_text())
            filename = "result.npz" if baseline == "GVHMR" else "result.pkl"
            recorded = (state["artifacts"][filename]["sha256"] if baseline == "GVHMR" else
                        state["inference"]["sha256"] if baseline == "WHAM" else state["result_sha256"])
            stat = (directory / filename).stat()
            inputs.append({"baseline": baseline, "unit_id": row["unit_id"],
                           "prediction_path": str(directory / filename),
                           "sha256_recorded_by_baseline_state": recorded,
                           "size_bytes": stat.st_size, "mtime_ns_at_comparison": stat.st_mtime_ns})
    write_csv(root / "prediction_input_inventory.csv", inputs)
    write_json(root / "comparison_provenance.json", {
        "comparison_script_sha256": digest(Path(__file__)), "command": sys.argv,
        "prediction_inventory_sha256": digest(root / "prediction_input_inventory.csv"),
        "prediction_hash_status": "baseline-state-recorded hashes; not independently rehashed in this comparison",
        "formula_control_audit_sha256": digest(root / "formula_control_audit.json"),
        "legacy_copies_sha256": {f"{b}/{name}": digest(root / b / REFERENCE / name)
                                 for b in BASELINES for name in
                                 ("unit_metrics.csv", "session_metrics.csv", "aggregate_metrics.csv", "run.json")}})
    report = ["# AnySole e9accee0 与 2_Code 三基线指标比较\n",
              "**完整评估协议的结果不同。** 相同输入的公式控制组" +
              ("全部通过数值容差检查" if passed else "存在超出容差项，详见审计JSON") +
              "；原生协议改变了关节构造、旋转集合、时序坐标/修复帧规则和足部取点，因此不能把两套正式数值视为相同。PVE-T在AnySole该提交中已经废除。\n",
              "分支为`metrics`，基于`e9accee0`，公共公式`anysole/utils/metrics.py`未修改。三baseline的既有预测不重跑推理。每个baseline使用相同257个物理session、2056个method×camera单元和785088帧。\n",
              "## 指标版本与文件\n",
              "每个baseline目录保留原报告，并新增三个版本目录：\n",
              "- `2_code_reference_20261006/`：已发表历史结果的字节相同副本，**不是本次重跑2_Code**；reference_provenance.json记录来源/hash。",
              "- `anysole_e9accee0/`：AnySole原生SMPL-24语义，指标全部调用该提交的公共公式。",
              "- `anysole_e9accee0_formula_control/`：保留旧输入/子集/掩码，改用AnySole公共公式；顶点足滑是诊断控制，不能与原生公共关节足滑混名引用。\n",
              "每个新版本有unit/session/aggregate_metrics.csv、run.json、report.md与unit_differences_to_2_code.csv。根metric_version_comparison.csv列出宏平均差值、配对session 95% CI、最大unit差及容差外unit数量。\n",
              "## 正式口径的汇总数值\n",
              "以下格式为`2_Code历史 → AnySole原生`，均按物理session宏平均；差值CSV保留完整精度。\n",
              "| 指标 | WHAM | SAM3DB | GVHMR |", "|---|---:|---:|---:|"]
    for metric in REQUESTED_METRICS:
        cells = [f"{fmt(number(headline[(b, REFERENCE)][metric]))} → {fmt(number(headline[(b, VERSIONS[0])][metric]))}" for b in BASELINES]
        report.append(f"| {LABELS[metric]} | " + " | ".join(cells) + " |")
    report += ["", "## 口径差异和解释\n",
               "| 指标 | AnySole本次原生语义 | 与2_Code的差异 |", "|---|---|---|",
               "| MPJPE / PA-MPJPE | 原生SMPL24，hip(1,2)平均作骨盆；PA为每帧Sim(3) | WHAM旧版从posed mesh重新回归24关节；本次用SMPL FK关节中心，与AnySole的FK等价，测试误差<1µm |",
               "| MPJAE | 全24局部旋转SO(3)角误差均值；root先转共同world | 旧版body21排除root与两末端hand，分母及集合均不同；不能简单乘21/24，因为新增关节误差非零 |",
               "| PVE | 各baseline自己的预测beta/网格与GT网格，hip平均去平移 | 三baseline确实预测形状，因此适用；不采用主模型诊断里的GT-beta补造表面。WHAM统一重建网格会有约0.1mm replay差异 |",
               "| PVE-T | N/A | AnySole metrics.py及指标改动说明明确取消；不复制2_Code T-pose指标公式补造数值 |",
               "| ACC-ERR / Jitter | 世界关节，全部配对帧，实际不规则时间；canonical duplicate/gap处理 | 旧版骨盆局部、剔除修复参考帧；平移运动与参考修复都影响结果。Jerk是三阶导范数，非pred−GT误差；GT jerk同时提供背景 |",
               "| W-MPJPE100 | 每100个输入帧，前2帧关节点拟合Sim(3)，评估窗口含末尾≥2帧短窗 | 公式与旧版一致；WHAM native/FK输入不同。100是帧数，不是100秒或固定时间宽度 |",
               "| Root ATE | native pelvis joint0在标定world的绝对距离均值，无ATE轨迹对齐 | 公式与旧版一致；WHAM root不再由posed网格重新回归。GVHMR只使用incam，不混入gravity/global gauge |",
               "| Foot sliding | ankle/foot joints 7,8,10,11；GT速度<0.3m/s的相邻transition上预测位移均值 | 旧版使用vertices3216,3387,6617,6787；两类contact数和位移都可能不同；该项不是速度，也不是预测接触分类 |\n",
               "### 采样与聚合边界\n",
               "本次为同一RGB预测样本的指标复核：CSV两锚点把各相机JPEG物理时间映射到mocap，GT旋转SO(3)插值、translation线性插值。保留重复时间给canonical时序函数合并，长间隔切段；W-MPJPE100沿输入帧窗口。",
               "AnySole主模型eval_protocol.py在自己的40Hz数据网格上用arange(T)/40。本次没有把RGB不规则样本虚构为40Hz，也没有改成主模型train/val/test split或只cam3；因此这是**同一257-session corpus上AnySole公式及原生关节语义的迁移评估**，不是原主模型40Hz全流程的逐字复刻。",
               "每个unit先算序列均值；每个(date,session)等权平均两方法×四相机的8个unit；最后对257个session等权。没有按frame_count加权。CI按物理session配对bootstrap10000次，非独立帧bootstrap。\n",
               "## 基线排名变化\n",
               "| 指标 | 历史排序（从小到大） | AnySole原生排序（从小到大） |", "|---|---|---|"]
    for metric in REQUESTED_METRICS:
        if metric in ("pve_t_mm", "jitter_gt_m_s3"):
            continue
        order = lambda version: " < ".join(sorted(BASELINES, key=lambda b: float(headline[(b, version)][metric])))
        report.append(f"| {LABELS[metric]} | {order(REFERENCE)} | {order(VERSIONS[0])} |")
    changed_ranks = [LABELS[metric] for metric in REQUESTED_METRICS
                     if metric not in ("pve_t_mm", "jitter_gt_m_s3") and
                     sorted(BASELINES, key=lambda b: float(headline[(b, REFERENCE)][metric])) !=
                     sorted(BASELINES, key=lambda b: float(headline[(b, VERSIONS[0])][metric]))]
    report += ["", "排序变化项：" + ("、".join(changed_ranks) if changed_ranks else "没有") + "。",
               "下面的时序倍率描述两种协议数值差异，不能解读为模型在同一指标下性能退化："]
    for baseline in BASELINES:
        old = headline[(baseline, REFERENCE)]
        new = headline[(baseline, VERSIONS[0])]
        report.append(f"- {baseline}：ACC-ERR变为旧值的{float(new['accel_error_m_s2']) / float(old['accel_error_m_s2']):.3f}倍，"
                      f"Prediction Jerk变为{float(new['jitter_pred_m_s3']) / float(old['jitter_pred_m_s3']):.3f}倍；"
                      f"关节足滑相对旧顶点足滑差{float(new['foot_sliding_mm']) - float(old['foot_sliding_mm']):+.6f}mm/transition。")
    report += ["", "Jerk与足滑是平滑性/运动品质统计，低值本身不保证姿态更准确；应连同GT jerk和MPJPE解读。基线之间的配对差值与95% CI见paired_baselines_anysole_e9accee0.csv。\n",
               "## 相同输入公式控制的数值证据\n",
               "先于全量运行，以8个跨日期/方法/相机unit的pilot设置绝对容差：几何/足滑0.0001mm，MPJAE0.000001°，ACC0.0001m/s²，Jerk0.005m/s³。它们用于检查公式一致性，不是允许的模型质量误差。SMPL GPU float32批量重建的微小差异会被三阶导放大；不要求字节相同。\n",
               "| 指标 | WHAM最大unit差 | SAM3DB最大unit差 | GVHMR最大unit差 |", "|---|---:|---:|---:|"]
    for metric in REQUESTED_METRICS:
        if metric == "pve_t_mm":
            continue
        report.append(f"| {LABELS[metric]} | " + " | ".join(f"{audit[f'{b}/{metric}']['max_abs_unit_delta']:.9g}" for b in BASELINES) + " |")
    failures = {key: value for key, value in audit.items() if value["units_outside_tolerance"]}
    report += ["", "公式控制是否全通过：**" + str(passed) + "**。完整逐项容差/超限计数见formula_control_audit.json。",
               f"30项严格控制检查中，{len(audit) - len(failures)}项通过，{len(failures)}项超限。没有放宽预设容差，超限项如实保留："]
    for name, value in failures.items():
        report.append(f"- {name}：{value['units_outside_tolerance']}/2056个unit超限；最大差{value['max_abs_unit_delta']:.12f}，"
                      f"原绝对容差{value['abs_tolerance']}。")
    source_audit = root / "shared_formula_source_audit.json"
    if source_audit.exists():
        report += ["", "补充只读源码AST审计显示，15个共享求解函数的计算实现完全相同（忽略docstring，"
                   "旧foot_sliding对应AnySole的foot_sliding_vertices）；未导入或执行2_Code指标。证据见shared_formula_source_audit.json。"]
    probe = root / "w_mpjpe_precision_probe.json"
    if probe.exists():
        records = json.loads(probe.read_text())["records"]
        report += ["", "最大差出现在同一个316帧unit：20260808/method_1_offset/S10081/camera_2。"
                   "使用AnySole求解器，对每个窗口首两帧GT施加一个float32 ULP扰动，关节点最大扰动"
                   f"{max(row['max_point_perturbation_mm'] for row in records):.12f}mm，"
                   f"可使WHAM的W值变化{max(abs(row['w_change_mm']) for row in records if row['baseline'] == 'WHAM'):.12f}mm。"
                   "这复现了同量级的浮点敏感性；历史运行未保存中间GT张量，因而只能推断微差与浮点输入/重建一致，"
                   "不能逐位定位历史误差来源，也不能宣称严格相等。完整探针见w_mpjpe_precision_probe.json。"]
    report += ["", "比较入口在严格控制失败时返回exit1，但仍完整输出差值和报告。"
               "这表示上述严格数值相等检查未全通过，不表示baseline重评缺失或把失败记录清零。\n",
               "## 代码来源、验证与限制\n",
               "公式唯一来源：AnySole/anysole/utils/metrics.py；native装配：anysole/utils/vision_smpl_metrics.py；入口：AnysoleWorkspace/tool/eval_rgb_smpl.py；消费端：results_display/script/compare_rgb_smpl_metrics.py。没有从2_Code导入具体指标、runner或GVHMR evaluation。",
               "仅复用2_Code/smpl_evaluation/{io,alignment,coordinates,model}.py。隔离namespace绕过原package __init__的metrics副作用导入；worker结束检查禁止模块未加载。run.json保存源码/GT/标定/manifest/模型hash及命令。",
               "prediction_input_inventory.csv列出6168个预测输入及baseline state原先记录的SHA-256、当前大小/mtime；此处未重新hash全部预测，不将state记录冒充本次独立hash验证。comparison_provenance.json记录比较端、参考表与审计表hash。",
               "7项针对测试涵盖identity、root参与MPJAE、世界平移加速度、公共关节足滑、session宏平均、禁止导入和SMPL前向与AnySole FK等价。真实8-unit×三baseline×两版本试跑48 rows零失败。",
               "已有public metric消费者测试依赖未随该提交提供的Baselines包，不能直接运行；本次直接验证已有canonical公式及新增装配，不伪造该套测试通过。",
               "GT′是marker拟合的SMPL参考，含已记录修复帧；全帧原生时序结果对修复有敏感性。SAM3DB含MHR→SMPL拟合与similarity比例，GVHMR含SMPL-X→SMPL转换约束，比较对象为保存的SMPL预测。",
               "本次未修改2_Code、原SMPL_Motion引用或历史发布文件；没有commit/push。PVE-T无法按e9accee0给出合法数值，这是该版本定义边界。\n"]
    (root / "比较分析_AnySole_e9accee0.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("formula-control comparison exceeds declared numerical tolerance; review audit")
    return {"formula_control_passed": passed, "checks": len(audit), "comparisons": len(comparisons)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO.parent / "3_Result/SMPL_evaluation")
    args = parser.parse_args()
    print(compare(args.root.resolve()))
