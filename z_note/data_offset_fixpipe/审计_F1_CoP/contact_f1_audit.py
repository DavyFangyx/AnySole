"""接触 F1 审计：GT 修正轴（2026-10-20，offset 修正表）。

复制自 audit_20261020_v3/contact_f1/contact_f1_audit.py，两处改动：
  1. GT 加载改用 anysole 私有标签 load_contact_label(shared, gt_method)
     （--gt-method，缺省 f6_soft 保持既有 GT 口径；f6_soft_offfix =
      修正轴标签，由 contact_adapter --corrected-offset 生成）；
  2. tactile 只审 raw108（压力派生四方法，10-19 同口径，val 36 会话）。

tactile 预测在视频轴（生产 raw108，不修正）；GT 在修正轴（f6_soft_offfix）
或原生轴（f6_soft）——F1 = 触觉派生接触 vs 运动学 GT 的轴差异敏感度。
GT 排除规则：某脚 GT 全 0（无接触帧）或全 1（无离地帧）→ 该侧 F1 记 excl。
输出 pooled precision/recall/F1 + per-session mean F1 (±std)。

只读：不写任何共享数据/模型产物；输出 csv+json 到 --out-dir。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "AnysoleWorkspace" / "tool"))

from contact_labels import METHODS, load_aligned, seq_dirs  # noqa: E402
from anysole.types import SEQ_ROOT, SPLIT_CSV  # noqa: E402
from anysole.data.workspace_adapter import (  # noqa: E402
    load_contact_label,
    load_shared_session,
)
from anysole.data.dataset import load_split_ids  # noqa: E402

METHODS_AUDIT = ("tactile_abs", "tactile_rel", "tactile_gmm", "pressure_f6")
FEET = ("left", "right")


def build_ctx(seq_dir: Path) -> dict:
    """raw108 ctx（视频轴，不修正）——10-19 同口径 load_aligned。"""
    return load_aligned(seq_dir)


def f1_components(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    """TP/FP/FN，contact=1 为正类。"""
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    tp = int(np.count_nonzero(pred_b & gt_b))
    fp = int(np.count_nonzero(pred_b & ~gt_b))
    fn = int(np.count_nonzero(~pred_b & gt_b))
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "z_note" / "data_offset_fixpipe"
                    / "审计_F1_CoP")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: only first N sessions")
    ap.add_argument("--split-col", default="val")
    ap.add_argument("--gt-method", default="f6_soft",
                    help="GT 标签 method（anysole 私有标签）；"
                         "f6_soft = 原生轴，f6_soft_offfix = 修正轴")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = load_split_ids(SPLIT_CSV, args.split_col)
    dirs = [d for d in seq_dirs(SEQ_ROOT) if d.name in set(split_ids)]
    if args.limit:
        dirs = dirs[: args.limit]
    print("split col=%s: %d ids; session dirs=%d%s"
          % (args.split_col, len(split_ids), len(dirs),
             " (SMOKE)" if args.limit else ""))

    # 汇总结构：results[method][foot] = {tp,fp,fn,f1_list,excl,missing}
    results: dict = {m: {f: {"tp": 0, "fp": 0, "fn": 0,
                             "f1s": [], "excl": 0, "n_incl": 0}
                         for f in FEET}
                     for m in METHODS_AUDIT}
    load_fail: list[str] = []
    gt_missing: list[str] = []
    gt_len_mismatch: list[str] = []
    per_session: dict = {}

    for seq_dir in dirs:
        sid = seq_dir.name
        try:
            ctx = build_ctx(seq_dir)
        except (FileNotFoundError, ValueError) as exc:
            load_fail.append("%s: %s" % (sid, exc))
            continue
        try:
            shared = load_shared_session(sid)
            gt = load_contact_label(shared, args.gt_method)
        except (FileNotFoundError, ValueError) as exc:
            gt_missing.append("%s: %s" % (sid, exc))
            continue
        n = ctx["n"]
        if gt.shape[0] != n:
            gt_len_mismatch.append("%s: gt=%d ctx=%d" % (sid, gt.shape[0], n))
            continue
        per_session[sid] = {}
        for foot_i, foot in enumerate(FEET):
            gt_f = gt[:, foot_i]
            degener = bool(gt_f.all()) or not bool(gt_f.any())
            per_session[sid][foot] = {"gt_degenerate": degener, "f1": {}}
            if degener:
                for m in METHODS_AUDIT:
                    results[m][foot]["excl"] += 1
                continue
            for m in METHODS_AUDIT:
                pred = METHODS[m]["fn"](ctx, None)[:, foot_i]
                tp, fp, fn = f1_components(pred, gt_f)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
                cell = results[m][foot]
                cell["tp"] += tp
                cell["fp"] += fp
                cell["fn"] += fn
                cell["f1s"].append(f1)
                cell["n_incl"] += 1
                per_session[sid][foot]["f1"][m] = f1
        print("done %s" % sid, flush=True)

    # ---- 汇总输出 ----
    rows = []
    summary = {}
    for m in METHODS_AUDIT:
        summary[m] = {}
        for foot in FEET:
            cell = results[m][foot]
            tp, fp, fn = cell["tp"], cell["fp"], cell["fn"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1p = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
            f1s = np.asarray(cell["f1s"], dtype=np.float64)
            f1_mean = float(f1s.mean()) if f1s.size else float("nan")
            f1_std = float(f1s.std()) if f1s.size else float("nan")
            summary[m][foot] = {
                "n_incl": cell["n_incl"], "n_excl": cell["excl"],
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec,
                "f1_pooled": f1p, "f1_mean": f1_mean, "f1_std": f1_std,
            }
            rows.append({
                "method": m, "foot": foot,
                "n_incl": cell["n_incl"], "n_excl": cell["excl"],
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1_pooled": round(f1p, 4),
                "f1_mean": round(f1_mean, 4), "f1_std": round(f1_std, 4),
            })

    csv_path = out_dir / ("contact_f1_results_%s.csv" % args.gt_method)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    excl_by_foot = {foot: results[METHODS_AUDIT[0]][foot]["excl"]
                    for foot in FEET}
    payload = {
        "split_col": args.split_col,
        "n_split_ids": len(split_ids),
        "n_sessions": len(dirs),
        "n_loaded": len(per_session),
        "load_fail": load_fail,
        "gt_missing": gt_missing,
        "gt_len_mismatch": gt_len_mismatch,
        "gt_degenerate_excl_by_foot": excl_by_foot,
        "methods": list(METHODS_AUDIT),
        "gt_method": args.gt_method,
        "tactile": "raw108 (视频轴, 不修正)",
        "gt": "load_contact_label binarized >0.5（%s 轴）" % (
            "修正" if args.gt_method == "f6_soft_offfix" else "原生"),
        "exclusion_rule": "GT foot all-0 or all-1 -> excl",
        "f1_definition": "binary F1, contact=positive; pooled + per-session mean",
        "summary": summary,
        "per_session": per_session,
    }
    json_path = out_dir / ("contact_f1_results_%s.json" % args.gt_method)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("load_fail=%d gt_missing=%d gt_len_mismatch=%d"
          % (len(load_fail), len(gt_missing), len(gt_len_mismatch)))
    print("excl by foot:", excl_by_foot)
    print("wrote", csv_path)
    print("wrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
