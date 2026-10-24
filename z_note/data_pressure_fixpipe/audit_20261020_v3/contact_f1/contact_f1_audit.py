"""接触 F1 三版审计（2026-10-20，v3 收口）。

复用 2026-10-19 audit_20261019/contact_f1/contact_f1_audit.py 口径，唯一
改动 = 版本列表与包映射（v1 已于 10-19 证与 v2 无差别，不再纳入）。
对 TEST split（splits.csv val 列，36 会话），用四种压力派生接触方法
（tactile_abs / tactile_rel / tactile_gmm / pressure_f6，生产阈值/公式原样，
contact_labels.METHODS 直接复用）在三版压力数据上出标签，与固定 GT
（f6_soft 二值化，load_labels 同口径）逐帧比对：
  raw108         = shared facts pressure_48.npz 的 left48/right48（原值，帧对齐）
  denoised108_v2 = 交接包 v2（无平滑）processed48 经 resample_handoff 映射
  denoised108_v3 = 交接包 v3（v2 前三项 + 故障格突发清洗）processed48 同映射
v2/v3 版 ctx 只替换压力数组（left48/right48/sums_l/sums_r），BVH 关节、
floor、air 信号等与 raw108 相同。

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

from contact_labels import METHODS, load_aligned, load_labels, seq_dirs  # noqa: E402
from anysole.types import SEQ_ROOT, SPLIT_CSV  # noqa: E402
from anysole.data.workspace_adapter import load_shared_session  # noqa: E402
from anysole.data.dataset import load_split_ids  # noqa: E402
from anysole.data.handoff_pressure import (  # noqa: E402
    handoff_root,
    resample_handoff,
    video_start_for,
)

METHODS_AUDIT = ("tactile_abs", "tactile_rel", "tactile_gmm", "pressure_f6")
VERSIONS = ("raw108", "denoised108_v2", "denoised108_v3")
PKG_BY_VERSION = {"denoised108_v2": "v2", "denoised108_v3": "v3"}
FEET = ("left", "right")


def build_ctxs(seq_dir: Path) -> dict[str, dict]:
    """三版 ctx：raw108 = load_aligned 原样；v1/v2 = 只换压力数组与 sums。"""
    ctx_raw = load_aligned(seq_dir)
    session_id = seq_dir.name
    shared = load_shared_session(session_id)
    meta = shared["meta"]
    n = ctx_raw["n"]
    date = str(meta["date"])
    subject = str(meta["subject"])
    video_start = video_start_for(shared["dir"])
    ctxs = {"raw108": ctx_raw}
    for version in VERSIONS[1:]:
        package = PKG_BY_VERSION[version]
        left48, right48 = resample_handoff(
            session_id, date, subject, video_start, n,
            package=package, root=handoff_root(package))
        ctx = dict(ctx_raw)
        ctx["left48"] = left48
        ctx["right48"] = right48
        ctx["sums_l"] = left48.sum(axis=1).astype(np.float64)
        ctx["sums_r"] = right48.sum(axis=1).astype(np.float64)
        ctxs[version] = ctx
    return ctxs


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
                    default=REPO_ROOT / "z_note" / "data_pressure_fixpipe"
                    / "audit_20261020_v3" / "contact_f1")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: only first N sessions")
    ap.add_argument("--split-col", default="val")
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

    # 汇总结构：results[method][version][foot] = {tp,fp,fn,f1_list,excl,missing}
    results: dict = {m: {v: {f: {"tp": 0, "fp": 0, "fn": 0,
                                 "f1s": [], "excl": 0, "n_incl": 0}
                             for f in FEET}
                         for v in VERSIONS}
                     for m in METHODS_AUDIT}
    load_fail: list[str] = []
    gt_missing: list[str] = []
    gt_len_mismatch: list[str] = []
    per_session: dict = {}

    for seq_dir in dirs:
        sid = seq_dir.name
        try:
            ctxs = build_ctxs(seq_dir)
        except (FileNotFoundError, ValueError) as exc:
            load_fail.append("%s: %s" % (sid, exc))
            continue
        gt = load_labels(seq_dir, "f6_soft")
        if gt is None:
            gt_missing.append(sid)
            continue
        n = ctxs["raw108"]["n"]
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
                    for v in VERSIONS:
                        results[m][v][foot]["excl"] += 1
                continue
            for m in METHODS_AUDIT:
                for v in VERSIONS:
                    pred = METHODS[m]["fn"](ctxs[v], None)[:, foot_i]
                    tp, fp, fn = f1_components(pred, gt_f)
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec = tp / (tp + fn) if tp + fn else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
                    cell = results[m][v][foot]
                    cell["tp"] += tp
                    cell["fp"] += fp
                    cell["fn"] += fn
                    cell["f1s"].append(f1)
                    cell["n_incl"] += 1
                    per_session[sid][foot]["f1"]["%s/%s" % (m, v)] = f1
        print("done %s" % sid, flush=True)

    # ---- 汇总输出 ----
    rows = []
    summary = {}
    for m in METHODS_AUDIT:
        summary[m] = {}
        for v in VERSIONS:
            summary[m][v] = {}
            for foot in FEET:
                cell = results[m][v][foot]
                tp, fp, fn = cell["tp"], cell["fp"], cell["fn"]
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1p = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
                f1s = np.asarray(cell["f1s"], dtype=np.float64)
                f1_mean = float(f1s.mean()) if f1s.size else float("nan")
                f1_std = float(f1s.std()) if f1s.size else float("nan")
                summary[m][v][foot] = {
                    "n_incl": cell["n_incl"], "n_excl": cell["excl"],
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": prec, "recall": rec,
                    "f1_pooled": f1p, "f1_mean": f1_mean, "f1_std": f1_std,
                }
                rows.append({
                    "method": m, "version": v, "foot": foot,
                    "n_incl": cell["n_incl"], "n_excl": cell["excl"],
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1_pooled": round(f1p, 4),
                    "f1_mean": round(f1_mean, 4), "f1_std": round(f1_std, 4),
                })

    csv_path = out_dir / "contact_f1_results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    excl_by_foot = {foot: results[METHODS_AUDIT[0]]["raw108"][foot]["excl"]
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
        "versions": list(VERSIONS),
        "gt": "f6_soft binarized >0.5 (load_labels 同口径)",
        "exclusion_rule": "GT foot all-0 or all-1 -> excl",
        "f1_definition": "binary F1, contact=positive; pooled + per-session mean",
        "summary": summary,
        "per_session": per_session,
    }
    json_path = out_dir / "contact_f1_results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("load_fail=%d gt_missing=%d gt_len_mismatch=%d"
          % (len(load_fail), len(gt_missing), len(gt_len_mismatch)))
    print("excl by foot:", excl_by_foot)
    print("wrote", csv_path)
    print("wrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
