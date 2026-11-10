#!/usr/bin/env python3
"""V8 四折重叠窗评估对照（2026-11-09，接缝归因 02 的 V8 落点）。

目的：把 `anysole.eval --overlap-stride N` 在 V8 四折 LSO 上的结果与现协议
（stride=tw 硬接）并排，验证「相邻窗接缝去抖动、姿态零税」是否在 V8 重演。
不替换现协议、不改 E2 主表口径；本表只作声明口径对照。

数据源（逐折 cell）：
- 基线：results/cells/<model>_<fold>_s1/metrics/test.json
- 重叠：results/cells/<model>_<fold>_s1/metrics/overlap_s<N>/test.json
- 接缝剖面：两侧 predictions/{eval_motion,overlap_s<N>/eval_motion} 的
  npz joint_xyz_world + valid_mask + frame_indices，jerk 冻结口径同
  e0_window_profile（接缝帧 = 窗内帧号 {18,19,0,1}，接缝份额 = 4·接缝均值/
  (20·全帧均值)）。

产物（幂等）：
- results_display/Etest_crossfit_v8/E2_MainPerformance/
    Full_Table5_E2_OverlapStride_<model>.csv    逐折 base/overlap/Δ 长表
    Main_Table2_E2_OverlapZeroTax_<model>.csv   条件级 mean±std 紧凑表
- results_display/Etest_crossfit_v8/detail/e2_overlap_stride_<model>.json

用法（touch_gait env，仓库根）：
  python results_display/script/e2_overlap_v8_export.py --models V8_Cv5 --stride 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display" / "script"))

import e0_window_profile as e0  # noqa: E402

CELLS = REPO / "results" / "cells"
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "E2_MainPerformance"
DETAIL = REPO / "results_display" / "Etest_crossfit_v8" / "detail"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
CONDS = (("V2M", "U@V"), ("T2M", "U@T"), ("VT2M", "U@VT"))
# (展示名, cell 键)。traj_ATE 与 RootAbs 都出：前者是 cell 链式口径（非主表），
# 后者是 E2 主表/外部基线同源的逐帧绝对根口径。
# 顺序对齐用户 2026-11-08 裁定的论文表序（姿态 → 轨迹/全局 → 物理 → 时序）。
METRICS = (
    ("PA-MPJPE", "PA-MPJPE"),
    ("MPJPE", "MPJPE"),
    ("MPJRE", "MPJRE"),
    ("GlobalMPJPE", "GlobalMPJPE"),
    ("W-MPJPE100", "W-MPJPE100"),
    ("RootAbs", "RootAbs"),
    ("traj_ATE", "traj_ATE"),
    ("RootOrientationDrift", "RootOrientationDrift"),
    ("GroundPen", "GroundPen（GT Anchored）"),
    ("FootSliding", "FootSliding（GT Anchored）"),
    ("Jitter_pred", "Jitter_pred"),
    ("Jitter_gt", "Jitter_gt"),
)


def _read(model: str, fold: str, stride) -> dict | None:
    if stride is None:
        p = CELLS / ("%s_%s_s1" % (model, fold)) / "metrics" / "test.json"
    else:
        p = (CELLS / ("%s_%s_s1" % (model, fold)) / "metrics"
             / ("overlap_s%d" % stride) / "test.json")
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _seam(model: str, fold: str, cond: str, stride) -> dict:
    d = CELLS / ("%s_%s_s1" % (model, fold)) / "predictions"
    pred = d / "eval_motion" if stride is None else d / ("overlap_s%d" % stride) / "eval_motion"
    if not pred.is_dir():
        return {"jerk": None, "seam_pct": None}
    sources = []
    for f in sorted(pred.glob("*_%s.npz" % cond)):
        z = np.load(f)
        J = np.asarray(z["joint_xyz_world"], np.float64)
        valid = (np.asarray(z["valid_mask"]).astype(bool)
                 if "valid_mask" in z.files else None)
        fi = np.asarray(z["frame_indices"], np.int64)
        sources.append((J, valid, fi, None))
    if not sources:
        return {"jerk": None, "seam_pct": None}
    row = e0.profile_row(cond, sources)
    return {"jerk": row["jerk"]["all"],
            "seam_pct": row["jerk"]["seam_share_pct"]}


def _val(block: dict, key: str):
    v = block.get(key)
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, len(vals)


def _fmt(ms):
    m, s, n = ms
    if m is None:
        return "—"
    return "%.3f±%.3f" % (m, s) if n > 1 else "%.3f" % m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="V8_Cv5")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in models:
        long_rows: list[list] = []
        agg: dict[str, dict[str, dict[str, list]]] = {}
        for cond, label in CONDS:
            agg[label] = {side: {name: [] for name, _ in METRICS}
                          for side in ("base", "ov")}
            agg[label]["base"]["jerk"] = []
            agg[label]["base"]["seam_pct"] = []
            agg[label]["ov"]["jerk"] = []
            agg[label]["ov"]["seam_pct"] = []
            agg[label]["base"]["n_gt"] = []
            agg[label]["ov"]["n_gt"] = []

        for fold in FOLDS:
            base = _read(model, fold, None)
            ov = _read(model, fold, args.stride)
            if base is None or ov is None:
                print("[skip] %s %s: base=%s overlap=%s"
                      % (model, fold, base is not None, ov is not None))
                continue
            base_m, ov_m = base.get("metrics", {}), ov.get("metrics", {})
            base_ngt = base.get("n_gt_anchors_used") or {}
            ov_ngt = ov.get("n_gt_anchors_used") or {}
            for cond, label in CONDS:
                b = base_m.get(cond) or {}
                o = ov_m.get(cond) or {}
                bj = _seam(model, fold, cond, None)
                oj = _seam(model, fold, cond, args.stride)
                row = [model, label, fold]
                for name, key in METRICS:
                    vb, vo = _val(b, key), _val(o, key)
                    row += [vb, vo, (None if vb is None or vo is None else vo - vb)]
                    agg[label]["base"][name].append(vb)
                    agg[label]["ov"][name].append(vo)
                row += [bj["jerk"], oj["jerk"], bj["seam_pct"], oj["seam_pct"],
                        base_ngt.get(cond), ov_ngt.get(cond)]
                long_rows.append(row)
                agg[label]["base"]["jerk"].append(bj["jerk"])
                agg[label]["ov"]["jerk"].append(oj["jerk"])
                agg[label]["base"]["seam_pct"].append(bj["seam_pct"])
                agg[label]["ov"]["seam_pct"].append(oj["seam_pct"])
                agg[label]["base"]["n_gt"].append(base_ngt.get(cond))
                agg[label]["ov"]["n_gt"].append(ov_ngt.get(cond))

        # 长表
        header = ["模型", "条件", "折"]
        for name, _ in METRICS:
            header += ["%s_base" % name, "%s_ov" % name, "%s_Δ" % name]
        header += ["jerk_base", "jerk_ov", "seam%_base", "seam%_ov",
                   "n_gt_base", "n_gt_ov"]
        path = OUT / ("Full_Table5_E2_OverlapStride_%s.csv" % model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(long_rows)
        print("[csv] %s (%d rows)" % (path, len(long_rows)))

        # 紧凑表：条件 × mean±std
        compact = []
        for cond, label in CONDS:
            row = [model, label]
            for name, _ in METRICS:
                b = _mean_std(agg[label]["base"][name])
                o = _mean_std(agg[label]["ov"][name])
                d = [ob - bb for bb, ob in zip(agg[label]["base"][name],
                                               agg[label]["ov"][name])
                     if bb is not None and ob is not None]
                row += [_fmt(b), _fmt(o), _fmt(_mean_std(d))]
            row += [_fmt(_mean_std(agg[label]["base"]["jerk"])),
                    _fmt(_mean_std(agg[label]["ov"]["jerk"])),
                    _fmt(_mean_std(agg[label]["base"]["seam_pct"])),
                    _fmt(_mean_std(agg[label]["ov"]["seam_pct"]))]
            compact.append(row)
        cheader = ["模型", "条件"]
        for name, _ in METRICS:
            cheader += ["%s_base" % name, "%s_ov" % name, "%s_Δ" % name]
        cheader += ["jerk_base", "jerk_ov", "seam%_base", "seam%_ov"]
        cpath = OUT / ("Main_Table2_E2_OverlapZeroTax_%s.csv" % model)
        with cpath.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cheader)
            w.writerows(compact)
        print("[csv] %s (%d rows)" % (cpath, len(compact)))

        # detail json：长表同源，便于复核
        jpath = DETAIL / ("e2_overlap_stride_%s.json" % model)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "e2_overlap_stride_v8_v1",
            "model": model,
            "stride": args.stride,
            "folds": list(FOLDS),
            "metrics": [{"name": n, "key": k} for n, k in METRICS],
            "long_rows": long_rows,
        }
        jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
        print("[json] %s" % jpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
