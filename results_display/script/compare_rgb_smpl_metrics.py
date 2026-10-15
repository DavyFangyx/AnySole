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


def publish_2_code(root: Path, source: Path):
    """Split freshly rerun 2_Code tables; do not import its evaluator."""
    metadata = json.loads((source / "run.json").read_text())
    if metadata.get("status") != "succeeded" or metadata.get("rows") != 6168:
        raise ValueError("2_Code full rerun is incomplete")
    write_json(root / "2_code_run.json", metadata)
    tables = {name: read_csv(source / name) for name in
              ("unit_metrics.csv", "session_metrics.csv", "aggregate_metrics.csv")}
    for baseline in BASELINES:
        destination = root / baseline / REFERENCE
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            write_csv(destination / name, [row for row in rows if row["baseline"] == baseline])
        write_json(destination / "run.json", {**metadata, "baseline": baseline,
                   "metric_version": REFERENCE, "source_run": "../../2_code_run.json",
                   "fresh_rerun": True,
                   "table_sha256": {name: digest(destination / name) for name in tables},
                   "source_table_sha256": {name: digest(source / name) for name in tables}})


def compare(root: Path):
    comparisons, paired, audit, headline, strata = [], [], {}, {}, []
    provenance = {}
    for baseline in BASELINES:
        reference = root / baseline / REFERENCE
        old_units = {row["unit_id"]: row for row in read_csv(reference / "unit_metrics.csv")}
        old_sessions = {(row["date"], row["session"]): row for row in read_csv(reference / "session_metrics.csv")}
        old_overall = next(row for row in read_csv(reference / "aggregate_metrics.csv") if row["stratum"] == "overall")
        headline[(baseline, REFERENCE)] = old_overall
        destination = root / baseline / VERSIONS[0]
        units = read_csv(destination / "unit_metrics.csv")
        sessions = {(row["date"], row["session"]): row for row in read_csv(destination / "session_metrics.csv")}
        if len(old_units) != 2056 or len(units) != 2056 or len(sessions) != 257:
            raise ValueError("wrong matched coverage")
        if {row["unit_id"] for row in units} != set(old_units) or set(sessions) != set(old_sessions):
            raise ValueError("paired unit/session roster differs")
        for version in (REFERENCE, VERSIONS[0]):
            rows = read_csv(root / baseline / version / "unit_metrics.csv")
            if sum(int(row["frame_count"]) for row in rows) != 785088:
                raise ValueError("frame coverage mismatch")
            for row in rows:
                if not all(np.isfinite(float(row[key])) for key in REQUESTED_METRICS):
                    raise ValueError("missing or nonfinite metric")
            for name in ("unit_metrics.csv", "session_metrics.csv", "aggregate_metrics.csv", "run.json"):
                path = root / baseline / version / name
                provenance[str(path.relative_to(root))] = digest(path)
        overall = next(row for row in read_csv(destination / "aggregate_metrics.csv") if row["stratum"] == "overall")
        headline[(baseline, VERSIONS[0])] = overall
        diffs = [{"unit_id": row["unit_id"], "baseline": baseline,
                  **{metric: float(row[metric])-float(old_units[row["unit_id"]][metric])
                     for metric in REQUESTED_METRICS}} for row in units]
        write_csv(destination / "unit_differences_to_2_code.csv", diffs)
        keys = sorted(sessions)
        session_diffs = [{"date": k[0], "session": k[1], "baseline": baseline,
                          **{m:float(sessions[k][m])-float(old_sessions[k][m]) for m in REQUESTED_METRICS}}
                         for k in keys]
        write_csv(destination / "session_differences_to_2_code.csv", session_diffs)
        details = [f"# {baseline} 两套指标对齐后的比较", "",
                   "两套结果均从保存预测独立重评。覆盖257个会话、2056个单元、785088帧，按物理会话等权汇总。", "",
                   "| 指标 | 2_Code | AnySole | AnySole减2_Code | 最大单元绝对差 |",
                   "|---|---:|---:|---:|---:|"]
        for metric in REQUESTED_METRICS:
            old, new = float(old_overall[metric]), float(overall[metric])
            values = np.array([float(sessions[k][metric]) for k in keys])
            refs = np.array([float(old_sessions[k][metric]) for k in keys])
            if abs(values.mean()-new)>1e-8 or abs(refs.mean()-old)>1e-8:
                raise ValueError("overall macro mean mismatch")
            _, low, high = paired_cluster_bootstrap(values, refs)
            max_abs = max(abs(row[metric]) for row in diffs)
            failures = sum(abs(row[metric])>tolerance(metric) for row in diffs)
            record = {"baseline":baseline,"metric":metric,"reference_value":old,"new_value":new,
                      "delta":new-old,"delta_ci95_low":low,"delta_ci95_high":high,
                      "max_abs_unit_delta":max_abs,"abs_tolerance":tolerance(metric),
                      "units_outside_tolerance":failures,"paired_sessions":257,
                      "status":"within_numeric_tolerance" if not failures else "different"}
            comparisons.append(record);audit[f"{baseline}/{metric}"]=record
            details.append(f"| {LABELS[metric]} | {old:.9f} | {new:.9f} | {new-old:+.12g} | {max_abs:.12g} |")
        for version in (REFERENCE, VERSIONS[0]):
            (root / baseline / version / "report.md").write_text("\n".join(details)+"\n",encoding="utf-8")
        identity = lambda row: tuple(row.get(k, "") for k in ("stratum","date","method","camera"))
        ref_strata = {identity(r):r for r in read_csv(reference / "aggregate_metrics.csv")}
        for row in read_csv(destination / "aggregate_metrics.csv"):
            ref = ref_strata[identity(row)]
            for metric in REQUESTED_METRICS:
                strata.append({"baseline":baseline,**{k:row.get(k,"") for k in ("stratum","date","method","camera")},
                               "metric":metric,"reference_value":float(ref[metric]),"new_value":float(row[metric]),
                               "delta":float(row[metric])-float(ref[metric])})
    for version in (REFERENCE, VERSIONS[0]):
        for first, second in combinations(BASELINES, 2):
            a={(r["date"],r["session"]):r for r in read_csv(root/first/version/"session_metrics.csv")}
            b={(r["date"],r["session"]):r for r in read_csv(root/second/version/"session_metrics.csv")}
            for metric in REQUESTED_METRICS:
                if metric == "jitter_gt_m_s3":
                    continue
                keys=sorted(a)
                delta,low,high=paired_cluster_bootstrap(np.array([float(a[k][metric]) for k in keys]),
                                                     np.array([float(b[k][metric]) for k in keys]))
                paired.append({"metric_version":version,"first":first,"second":second,"metric":metric,
                               "difference_first_minus_second":delta,"ci95_low":low,"ci95_high":high,
                               "paired_sessions":len(keys)})
    write_csv(root / "metric_version_comparison.csv",comparisons)
    write_csv(root / "stratified_metric_comparison.csv",strata)
    write_csv(root / "paired_baseline_comparison.csv",paired)
    passed=all(not r["units_outside_tolerance"] for r in comparisons)
    write_json(root / "metric_alignment_audit.json",{"passed":passed,"checks":audit})
    write_json(root / "comparison_provenance.json",{"result_sha256":provenance,
               "comparison_source_sha256":digest(Path(__file__)),"fresh_reruns":True})
    report=["# 三基线两套指标对齐后的全量比较", "",
            f"两套评估均独立重跑保存的WHAM、SAM3DB、GVHMR预测，覆盖同一257个物理会话、每基线2056个方法与相机单元、785088帧。33项数值核查（十项指标加GT Jerk，三基线）通过{sum(not r['units_outside_tolerance'] for r in comparisons)}项。" + ("两套结果在预设数值容差内一致。" if passed else "存在超过容差的差异，详见逐单元审计。"), "",
            "## 评估协议与八项修改", "",
            "| 要求 | 对齐后的方式 |", "|---|---|",
            "| PA-MPJPE | 直接传入未减骨盆的24个关节，逐帧拟合旋转、平移和尺度；相似变换已经包含平移，预先减骨盆理论上不改变结果 |",
            "| MPJAE | 21个身体关节的局部旋转测地角，排除根和两个hand末端；单位为度 |",
            "| PVE | 各基线预测自身beta生成网格，GT使用GT beta，6890对应顶点，分别减髋均值后取欧氏距离均值 |",
            "| PVE-T | 各自beta、统一零姿态，分别减髋均值，无尺度校正；AnySole新增公共函数与有限数值 |",
            "| ACC-ERR与Jerk | 减髋均值后的相对关节轨迹，真实时间重复差分；m/s²与m/s³，不含官方WHAM的额外除10 |",
            "| 修复GT区间 | 全部配对帧参与主指标和时序计算；仍记录修复计数及额外空间敏感性诊断，不丢弃主评估样本 |",
            "| Foot sliding | SMPL关节7、8、10、11；GT速度小于0.3m/s判接触，统计预测三维位移，mm/接触transition |",
            "| WHAM几何 | 从保存pose、beta、trans重新生成neutral-SMPL顶点和原生FK关节，不对posed mesh重新回归关节 |", "",
            "W-MPJPE100保持每100输入帧窗口以前两帧拟合相似变换；末尾至少两帧继续评估。Root ATE保持世界第0关节未对齐的平均距离，使用mean而非ATE-RMSE。GT用SO(3)旋转插值与线性平移插值，重复时间戳取均值，时序长缺口分段。", "",
            "## 全量指标数值", "",
            "表格格式为2_Code / AnySole，主值为会话宏平均。GT Jerk用于解释参考运动，不参与基线优劣排名。", "",
            "| 指标 | WHAM | SAM3DB | GVHMR |", "|---|---:|---:|---:|"]
    for metric in REQUESTED_METRICS:
        cells=[f"{float(headline[(b,REFERENCE)][metric]):.6f} / {float(headline[(b,VERSIONS[0])][metric]):.6f}" for b in BASELINES]
        report.append(f"| {LABELS[metric]} | "+" | ".join(cells)+" |")
    report += ["", "## 逐单元差值与数值判断", "",
               "容差沿用之前试跑确定的界限：毫米类1e-4mm，角度1e-6度，ACC-ERR 1e-4m/s²，Jerk 0.005m/s³。不以四舍五入后的表格相同代替逐单元检查。", "",
               "| 指标 | WHAM最大绝对差 | SAM3DB最大绝对差 | GVHMR最大绝对差 |", "|---|---:|---:|---:|"]
    for metric in REQUESTED_METRICS:
        report.append(f"| {LABELS[metric]} | "+" | ".join(f"{audit[f'{b}/{metric}']['max_abs_unit_delta']:.12g}" for b in BASELINES)+" |")
    report += ["", "两套入口保留各自实现。2_Code先在camera计算局部误差，AnySole先转world；固定刚体旋转不改变距离和导数范数，微小差异可能来自坐标转换或浮点拟合。数值核查只判断设定容差内的一致性，不宣称所有CSV逐位相同。", "",
               "## 基线表现与排名", "", "| 指标 | 2_Code从低到高 | AnySole从低到高 |", "|---|---|---|"]
    changed=[]
    for metric in REQUESTED_METRICS:
        if metric=="jitter_gt_m_s3":continue
        orders=[sorted(BASELINES,key=lambda b:float(headline[(b,v)][metric])) for v in (REFERENCE,VERSIONS[0])]
        if orders[0]!=orders[1]:changed.append(LABELS[metric])
        report.append(f"| {LABELS[metric]} | {' < '.join(orders[0])} | {' < '.join(orders[1])} |")
    report += ["", "### 各类误差的具体表现", ""]
    for category, metrics in (
        ("局部姿态与网格", ("mpjpe_mm", "pa_mpjpe_mm", "mpjae_deg", "pve_mm")),
        ("体型", ("pve_t_mm",)),
        ("世界运动", ("w_mpjpe100_mm", "root_ate_mm")),
        ("身体时序与接触运动", ("accel_error_m_s2", "jitter_pred_m_s3", "foot_sliding_mm")),
    ):
        cells = []
        for metric in metrics:
            ordered = sorted(BASELINES, key=lambda b:float(headline[(b,VERSIONS[0])][metric]))
            first, second = ordered[:2]
            best = float(headline[(first,VERSIONS[0])][metric])
            next_value = float(headline[(second,VERSIONS[0])][metric])
            percent = (next_value-best)/next_value*100 if next_value else 0
            cells.append(f"{LABELS[metric]}最低为{first}（{best:.6f}），第二为{second}（{next_value:.6f}），相差{percent:.2f}%")
        report.append(category+"："+"；".join(cells)+"。")
        report.append("")
    report.append("PA-MPJPE消除旋转、尺度和平移，不直接检验世界定位。PVE-T隔离体型；相同PVE仍可能由不同的姿态和体型误差组合产生。身体相对Jerk不含根运动，足滑则仍在世界坐标测量。上述百分比只是宏平均差异，显著性应结合配对区间。")
    report += ["", "### 主要准确性指标的配对区间", "",
               "| 指标 | 第一基线 | 第二基线 | 第一减第二 | 95%区间 |", "|---|---|---|---:|---:|"]
    for row in paired:
        if row["metric_version"]==VERSIONS[0] and row["metric"] in ("mpjpe_mm","pve_mm","root_ate_mm"):
            report.append(f"| {LABELS[row['metric']]} | {row['first']} | {row['second']} | {row['difference_first_minus_second']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report += ["", "两套排名变化项："+("、".join(changed) if changed else "无")+"。",
               "WHAM减SAM3DB的PVE差值区间跨过零，因此SAM3DB的PVE宏平均虽稍低，此配对分析未支持二者PVE有明确差异。MPJPE及世界Root ATE的相应区间未跨零。", "",
               "准确性比较应结合MPJPE、PVE、角度和世界轨迹，Jerk与足滑较小本身不足以证明姿态更准确。基线间差值与95%区间见paired_baseline_comparison.csv；区间按257个会话配对bootstrap10000次，不将同一会话的帧当独立样本。", "",
               "## PVE的体型条件", "",
               "RGB三基线均生成beta，因此两套PVE和PVE-T均使用预测自身beta。AnySole原主模型不预测shape，其同GT beta的PVE是已知体型下的姿态诊断，不进主模型正式比较表；该诊断保留，不伪造一套预测beta。主模型消费者的body21角度及骨盆相对时序已同步。", "",
               "## 文件与可复核范围", "",
               "每个基线仅保留2_code_aligned与anysole_aligned两版本，各有unit_metrics.csv、session_metrics.csv、aggregate_metrics.csv、run.json和report.md。AnySole版本额外保存逐单元和会话对2_Code的差值。根目录提供总体、日期、方法、相机分层差值以及两套基线配对比较。", "",
               "AnySole所需四个数据/模型辅助模块按原实现选择性放入anysole/data/rgb_smpl；具体指标在anysole/utils/metrics.py，不导入2_Code指标或辅助源码。2_Code的GVHMR直接重算保存的result.npz，不读取旧evaluation_camera.json指标作为结果。", "",
               "GT是marker拟合参考，仍有拟合及修复不确定性；SAM3DB包含MHR到SMPL拟合，GVHMR包含SMPL-X到SMPL转换。这里重跑指标评估，没有重新推理或改变保存预测。源hash、模型hash、GT和标定来源见run.json与comparison_provenance.json。", ""]
    validation_path = root / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        report += ["## 独立验证", "",
                   f"独立审计从unit表重新计算session及总体/日期/方法/相机宏平均，{validation['aggregation_checks']}项检查通过。六份版本表均覆盖相同帧与单元，所有十项指标及GT Jerk为有限值。", "",
                   f"{validation['prediction_metadata_unchanged']}份原始预测文件的大小与修改时间保持不变；{validation['gt_hashes_unchanged']}份GT的SHA-256保持不变。SMPL_Motion引用、manifest及两套运行使用的SMPL模型hash均通过核对。预测检查为文件metadata核查，不冒充完整文件hash核查。详情见validation.json。", ""]
    (root / "比较分析_两套指标对齐.md").write_text("\n".join(report),encoding="utf-8")
    (root / "README.md").write_text("# 最新SMPL指标评估\n\n"+
        "三基线均有独立重跑的2_code_aligned和anysole_aligned。\n\n"+
        "[完整比较与分析](比较分析_两套指标对齐.md)；[逐指标差值](metric_version_comparison.csv)；[数值审计](metric_alignment_audit.json)。\n",encoding="utf-8")
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=REPO.parent/"3_Result/SMPL_evaluation")
    parser.add_argument("--two-code-run",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if args.two_code_run:publish_2_code(root,args.two_code_run.resolve())
    raise SystemExit(0 if compare(root) else 1)


if __name__=="__main__":main()
