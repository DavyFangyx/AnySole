#!/usr/bin/env python3
"""E0 重叠拼接推理对照表（修改 02「接缝归因与重叠拼接推理」，2026-11-03）：

把 --overlap-stride 扫描（stride ∈ {20,10,5,2} × 模型 × 条件）的 brief 指标
落成对照表 + 接缝分解（jerk 冻结口径 frame%20，同 e0_window_profile）。

- 指标来源：results/cells/<cell>/metrics/overlap_s<N>/test_brief.json
  （十指标口径 = 现役 brief：PA-MPJPE 等 11 列，同 E2 主表头）；
  n_gt_anchors_used 读同目录 test.json 顶层字段（每条件）。
- 接缝分解：npz joint_xyz_world + valid_mask + frame_indices，逐帧
  j[t] = FPS³·(J[t] − 2·J[t−1] + J[t−2])（冻结实测口径）按窗内帧号
  %20 分桶；接缝帧 = {18,19,0,1}；接缝占总量 = 4·接缝均值/(20·全帧
  均值)（与 e0_window_profile.py 同函数）。
- 基线行：现有 cell metrics/test_brief.json（stride=20 = 现协议）；
  stride=20 扫描行与基线逐位一致的等价性自检在扫描阶段完成（硬门），
  本表不重复校验。

产物（幂等）：
- results_display/Etest/E0_ReferenceFloors/Full_Table3_E0_OverlapStride.csv
- results_display/Etest/E0_ReferenceFloors/detail/e0_overlap_stride.json
- results_display/Etest/E0_ReferenceFloors/detail/E0_重叠拼接推理说明.md
- results_display/Etest/receipts/e0_overlap_stride_receipt.json

用法（touch_gait env）：
  python results_display/script/e0_overlap_stride.py \
      --cells V7_Cv5_s1,V7_Cv5AR2M_s1 --strides 20,10,5,2 --modes T2M,V2M,VT2M
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display" / "script"))

import e0_window_profile as e0  # noqa: E402

ETEST = REPO / "results_display" / "Etest"
OUT_DIR = ETEST / "E0_ReferenceFloors"
DETAIL = OUT_DIR / "detail"
RECEIPTS = ETEST / "receipts"
CELLS_ROOT = REPO / "results" / "cells"

METRIC_COLS = [
    "PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
    "traj_ATE@0.5s", "traj_ATE", "RootOrientationDrift", "FootSliding",
    "GroundPen", "Jitter_pred",
]
EXTRA_COLS = ["n_gt_anchors_used", "jerk_全帧(冻结)", "接缝占总量%"]
DEFAULT_CELLS = ("V7_Cv5_s1", "V7_Cv5AR2M_s1")
DEFAULT_STRIDES = (20, 10, 5, 2)
DEFAULT_MODES = ("T2M", "V2M", "VT2M")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_brief(cell: str, stride) -> dict:
    if stride == "baseline":
        p = CELLS_ROOT / cell / "metrics" / "test_brief.json"
    else:
        p = CELLS_ROOT / cell / "metrics" / ("overlap_s%d" % stride) / \
            "test_brief.json"
    return json.loads(p.read_text())


def _read_anchors(cell: str, stride) -> dict:
    p = CELLS_ROOT / cell / "metrics" / ("overlap_s%d" % stride) / "test.json"
    payload = json.loads(p.read_text())
    return payload.get("n_gt_anchors_used", {})


def _seam_row(pred_dir: Path, mode: str, gt: dict) -> dict:
    """jerk 冻结口径剖面 + 接缝分解（同 e0_window_profile 函数）。"""
    sources = []
    for sid, g in gt.items():
        f = pred_dir / ("%s_%s.npz" % (sid, mode))
        if not f.is_file():
            continue
        d = np.load(f)
        J = np.asarray(d["joint_xyz_world"], np.float64)
        valid = (np.asarray(d["valid_mask"]).astype(bool)
                 if "valid_mask" in d.files else None)
        fi = np.asarray(d["frame_indices"], np.int64)
        sources.append((J, valid, fi, g))
    if not sources:
        return {"all": float("nan"), "share": float("nan"), "files": []}
    row = e0.profile_row("seam", sources)
    return {"all": row["jerk"]["all"], "share": row["jerk"]["seam_share_pct"],
            "files": [str(s) for s in sources]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    ap.add_argument("--strides", default=",".join(map(str, DEFAULT_STRIDES)))
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--include-baseline", action="store_true",
                    help="额外输出基线行（cell 现有 test_brief，stride=20 现协议）")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    strides = [int(s) for s in args.strides.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    gt = e0.build_gt()
    rows = []           # 长格式：{模型,模式,stride,<指标>...}
    sources = []        # receipt 源登记
    for cell in cells:
        baseline = _read_brief(cell, "baseline")["metrics"]
        baseline_anchor = None
        for stride in strides:
            brief_path = (CELLS_ROOT / cell / "metrics"
                          / ("overlap_s%d" % stride) / "test_brief.json")
            if not brief_path.is_file():
                print("[skip] %s stride=%d: 无 brief json" % (cell, stride))
                continue
            brief = _read_brief(cell, stride)["metrics"]
            anchors = _read_anchors(cell, stride)
            sources.append({"kind": "brief", "cell": cell, "stride": stride,
                            "path": str(brief_path),
                            "sha256": _sha256(brief_path)})
            detail_path = (CELLS_ROOT / cell / "metrics"
                           / ("overlap_s%d" % stride) / "test.json")
            if detail_path.is_file():
                sources.append({"kind": "detail", "cell": cell,
                                "stride": stride, "path": str(detail_path),
                                "sha256": _sha256(detail_path)})
            pred_dir = (CELLS_ROOT / cell / "predictions"
                        / ("overlap_s%d" % stride) / "eval_motion")
            for mode in modes:
                row = {"模型": cell, "模式": mode, "stride": stride}
                block = brief.get(mode, {})
                for m in METRIC_COLS:
                    row[m] = block.get(m)
                row["n_gt_anchors_used"] = anchors.get(mode)
                seam = _seam_row(pred_dir, mode, gt) if pred_dir.is_dir() \
                    else {"all": None, "share": None}
                row["jerk_全帧(冻结)"] = seam["all"]
                row["接缝占总量%"] = seam["share"]
                rows.append(row)
        if args.include_baseline:
            for mode in modes:
                row = {"模型": cell, "模式": mode, "stride": "baseline"}
                block = baseline.get(mode, {})
                for m in METRIC_COLS:
                    row[m] = block.get(m)
                row["n_gt_anchors_used"] = baseline_anchor
                rows.append(row)

    # 宽表：行 = 模型×模式，列 = stride × (指标 + 附加列)
    header = ["模型", "模式"]
    for stride in (["baseline"] if args.include_baseline else []) + strides:
        for m in METRIC_COLS + EXTRA_COLS:
            header.append("%s_s%s" % (m, stride))
    out_rows = []
    for cell in cells:
        for mode in modes:
            cell_rows = [r for r in rows if r["模型"] == cell
                         and r["模式"] == mode]
            if not cell_rows:
                continue
            line = [cell, mode]
            for r in cell_rows:
                for m in METRIC_COLS + EXTRA_COLS:
                    v = r.get(m)
                    line.append("" if v is None else (
                        "%.4f" % v if isinstance(v, (int, float)) else str(v)))
            out_rows.append(line)
    csv_path = OUT_DIR / "Full_Table3_E0_OverlapStride.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print("[csv] %s (%d rows)" % (csv_path, len(out_rows)))

    # 长格式 detail json（含全部数值）
    json_path = DETAIL / "e0_overlap_stride.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "e0_overlap_stride_v1",
        "split": "test",
        "cells": cells,
        "strides": strides,
        "modes": modes,
        "metric_columns": METRIC_COLS,
        "extra_columns": EXTRA_COLS,
        "seam_definition": (
            "jerk 冻结口径（同 e0_window_profile）：j[t] = FPS³·(J[t] − "
            "2·J[t−1] + J[t−2])，按窗内帧号 frame%20 分桶；接缝帧 = "
            "{18,19,0,1}；接缝占总量 = 4·接缝均值/(20·全帧均值)。"),
        "n_gt_anchors_used_definition": (
            "预测侧 GT 平移锚点消费数（每条件）：标准路径 = 覆盖 tile 数"
            "（重叠 stride 下锚点网格保持非重叠网格 → 与现协议每窗一锚"
            "逐位相同）；AR2 自由积分 = 0。"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)
                         + "\n")
    print("[json] %s" % json_path)

    # 说明 md（detail/ 下沉；新文档）
    md_lines = ["# E0 重叠拼接推理说明（修改 02）", ""]
    md_lines.append("声明口径：--overlap-stride N（N ∈ [1, tw]，N=tw 为等价性"
                    "对照）；协议与导出改走三角权重混合组装（w(k) = 1 − "
                    "|k − (tw−1)/2| / (tw/2)，窗中心 0.95、窗缘 1/tw > 0，每帧"
                    "按覆盖窗权重归一化）；pred_pose 6D 通道加权平均后经 "
                    "6D→rotmat→6D 投影回 SO(3)，平移/速度通道线性加权；"
                    "GT/contact 键取首覆盖窗。")
    md_lines.append("")
    md_lines.append("GT 泄漏硬约束：锚点网格保持原非重叠网格（仅 tile*tw 处取"
                    "锚），重叠窗继承所属 tile 的锚；n_gt_anchors_used 与现"
                    "协议逐位相同。AR2 自由积分轨迹无每窗 GT 锚（登记 0）。")
    md_lines.append("")
    md_lines.append("等价性硬门：stride=20 走新代码路径，十指标与现协议逐位"
                    "一致（扫描阶段已验；不一致先修代码再出数）。")
    md_lines.append("")
    md_lines.append("| 模型 | 模式 | stride | PA-MPJPE | Jitter_pred | 接缝占总量% | n_gt_anchors |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md_lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["模型"], r["模式"], r["stride"],
            ("%.3f" % r["PA-MPJPE"]) if r.get("PA-MPJPE") is not None else "—",
            ("%.2f" % r["Jitter_pred"]) if r.get("Jitter_pred") is not None else "—",
            ("%.1f" % r["接缝占总量%"]) if r.get("接缝占总量%") is not None
            and np.isfinite(r["接缝占总量%"]) else "—",
            r.get("n_gt_anchors_used", "—")))
    md_lines.append("")
    md_lines.append("判读：stride<20 列是否把 Jitter 与接缝份额压下去、PA 是否"
                    "保持（±1.0mm 内）见回报与验收表；本表只作对照，不直接"
                    "写入 E2 主表。")
    md_path = DETAIL / "E0_重叠拼接推理说明.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("[md] %s" % md_path)

    # receipt（沿用现有 schema：sources + artifacts）
    for cell in cells:
        for stride in strides:
            pred_dir = (CELLS_ROOT / cell / "predictions"
                        / ("overlap_s%d" % stride) / "eval_motion")
            if not pred_dir.is_dir():
                continue
            for f in sorted(pred_dir.glob("*.npz")):
                sources.append({"kind": "npz", "cell": cell, "stride": stride,
                                "path": str(f), "sha256": _sha256(f)})
    receipt_path = RECEIPTS / "e0_overlap_stride_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [csv_path, json_path, md_path]
    receipt_path.write_text(json.dumps(
        {"sources": sources,
         "artifacts": [{"path": str(p), "sha256": _sha256(p)}
                       for p in artifacts]},
        ensure_ascii=False, indent=2) + "\n")
    print("[receipt] %s" % receipt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
