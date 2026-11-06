#!/usr/bin/env python3
"""E6 门控 × 遮挡（四折 LSO，Etest_crossfit 口径，2026-11-05）。

与 ``e6_gate_occlusion.py`` 同一分解口径，只把数据源从 act8020 val 换成
四折 LSO test：

    Δ_s = VT_s − V_s（逐会话，n_frames 加权）
    ΔE⁺ = Σ w_s max(Δ_s,0)/Σw_s      （有害分量）
    ΔE⁻ = Σ w_s min(Δ_s,0)/Σw_s      （有益分量）
    抑制率 = 1 − ΔE⁺_cand / ΔE⁺_ref   （越大越好）
    保留率 = ΔE⁻_cand / ΔE⁻_ref       （越大越好）
    净值差 = ΔE_cand − ΔE_ref

参照 = Cv7（无门，g≡1）；候选 = Cv5 / Cv5SD2 / Cv5SD2RT / Cv5_tphys。
数据源 = ``results/cells/<fam>_<fold>_s1_evalocc<level>/metrics/
session_detail_test.json``（E5 四折遮挡交叉格的 eval 产物）。

用法（base env 即可；E5 格跑完后执行）：
  python results_display/script/e6_fold_gate_occlusion.py
  python results_display/script/e6_fold_gate_occlusion.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = REPO / "results_display" / "Etest_crossfit" / "E6_gate_occlusion"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")
REFERENCE = "V7_Cv7"
CANDIDATES = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT", "V7_Cv5_tphys")
LEVELS = ("020", "040", "060", "080", "100")
METRICS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "MPJRE": "mpjae_deg",
    "traj_ATE@0.5s": "root_ate_seg20_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
KEY_METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE@0.5s",
               "FootSliding", "GroundPen", "Jitter_pred")
MIN_EFFECT = 0.05
_SKIPPED_METRICS: set[str] = set()
S13OUT = False
S13OUT_FOLD = "fS6S7S13"


def cell_id(fam: str, fold: str, level: str) -> str:
    base = "%s_%s_s1" % (fam, fold)
    if S13OUT and fold == S13OUT_FOLD:
        base += "_s13out"
    return "%s_evalocc%s" % (base, level)


def decompose(cell: str, split: str) -> dict | None:
    path = CELLS / cell / "metrics" / ("session_detail_%s.json" % split)
    if not path.is_file():
        return None
    detail = json.loads(path.read_text(encoding="utf-8"))["conditions"]
    vt = {e["session_id"]: e["metrics"] for e in detail["VT2M"]}
    v = {e["session_id"]: e["metrics"] for e in detail["V2M"]}
    n_frames = {e["session_id"]: e["n_frames"] for e in detail["V2M"]}
    sids = sorted(set(v) & set(vt))
    if not sids:
        return None
    out = {}
    for name, key in METRICS.items():
        pairs = [(vt[s].get(key), v[s].get(key)) for s in sids]
        if any(a is None or b is None for a, b in pairs):
            # Fold E5 eval 的 session_detail 不含 act8020 的短窗
            # root_ate_seg20_mm；跳过该列而不是让整张表 KeyError。
            _SKIPPED_METRICS.add(name)
            continue
        delta = np.array([a - b for a, b in pairs], dtype=float)
        w = np.array([float(n_frames[s]) for s in sids])
        total = w.sum()
        out[name] = {
            "net": float((w * delta).sum() / total),
            "plus": float((w * np.maximum(delta, 0.0)).sum() / total),
            "minus": float((w * np.minimum(delta, 0.0)).sum() / total),
        }
    return out


def fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return round(float(value), 3)


def pct(num: float | None, den: float, signed: bool = False) -> str:
    if num is None or abs(den) < MIN_EFFECT:
        return ""
    if signed:
        return "%+.1f%%" % (100.0 * num / den)
    return "%.1f%%" % (100.0 * num / den)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def mean_std(values: list[float]) -> str:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return ""
    if len(values) == 1:
        return "%.3f" % values[0]
    return "%.3f±%.3f" % (float(np.mean(values)), float(np.std(values)))


def main() -> int:
    global S13OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--include-s13", action="store_true",
                    help="改用含 S13 的旧口径（写到 detail/include_s13/）；"
                         "缺省 = woS13 正式口径，写根层")
    args = ap.parse_args()
    S13OUT = not args.include_s13
    out_dir: Path = args.out_dir or (OUT / "detail" / "include_s13"
                                     if args.include_s13 else OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    summary_rows = []
    per_family_metric: dict[tuple[str, str], dict[str, list[float]]] = {}
    missing = []

    for fold in FOLDS:
        for level in LEVELS:
            ref = decompose(cell_id(REFERENCE, fold, level), args.split)
            if ref is None:
                missing.append("%s/%s" % (fold, level))
                continue
            for fam in CANDIDATES:
                cand = decompose(cell_id(fam, fold, level), args.split)
                if cand is None:
                    missing.append("%s/%s/%s" % (fold, level, fam))
                    continue
                for metric in KEY_METRICS:
                    if metric not in ref or metric not in cand:
                        _SKIPPED_METRICS.add(metric)
                        continue
                    r, c = ref[metric], cand[metric]
                    suppression = (1.0 - c["plus"] / r["plus"]) if abs(r["plus"]) >= MIN_EFFECT else None
                    preservation = (c["minus"] / r["minus"]) if abs(r["minus"]) >= MIN_EFFECT else None
                    net_delta = c["net"] - r["net"]
                    long_rows.append([
                        fold, level, fam, metric,
                        fmt(r["net"]), fmt(r["plus"]), fmt(r["minus"]),
                        fmt(c["net"]), fmt(c["plus"]), fmt(c["minus"]),
                        "" if suppression is None else round(100.0 * suppression, 1),
                        "" if preservation is None else round(100.0 * preservation, 1),
                        fmt(net_delta),
                    ])
                    summary_rows.append([
                        fold, level, fam, metric,
                        fmt(r["net"]), fmt(c["net"]), fmt(net_delta),
                        pct(suppression, 1.0), pct(preservation, 1.0),
                    ])
                    bucket = per_family_metric.setdefault((fam, metric), {
                        "suppression": [], "preservation": [], "net_delta": [],
                        "d_plus": [], "d_minus": []})
                    if suppression is not None:
                        bucket["suppression"].append(suppression)
                    if preservation is not None:
                        bucket["preservation"].append(preservation)
                    bucket["net_delta"].append(net_delta)
                    bucket["d_plus"].append(c["plus"] - r["plus"])
                    bucket["d_minus"].append(c["minus"] - r["minus"])

    write_csv(out_dir / "fold_gate_occlusion_long.csv",
              ["fold", "level", "family", "metric",
               "ref_net", "ref_plus", "ref_minus",
               "cand_net", "cand_plus", "cand_minus",
               "suppression_pct", "preservation_pct", "net_delta"],
              long_rows)
    write_csv(out_dir / "fold_gate_occlusion_summary.csv",
              ["fold", "level", "family", "metric",
               "ref_net", "cand_net", "net_delta",
               "suppression", "preservation"],
              summary_rows)

    mean_rows = []
    for fam in CANDIDATES:
        for metric in KEY_METRICS:
            bucket = per_family_metric.get((fam, metric))
            if not bucket:
                continue
            mean_rows.append([
                fam, metric,
                mean_std(bucket["d_plus"]),
                mean_std(bucket["d_minus"]),
                mean_std(bucket["net_delta"]),
                mean_std([100.0 * x for x in bucket["suppression"]]),
                mean_std([100.0 * x for x in bucket["preservation"]]),
                len(bucket["net_delta"]),
            ])
    # 主列 = 绝对值差（d_plus/d_minus/net_delta，mm），与 E6 两张主表
    # 「保留%因小分母失真改报绝对值」的既有口径一致；百分比仅作辅助列。
    write_csv(out_dir / "fold_gate_occlusion_mean.csv",
              ["family", "metric", "d_plus_mean_std", "d_minus_mean_std",
               "net_delta_mean_std", "suppression_pct_mean_std",
               "preservation_pct_mean_std", "n_fold_level"],
              mean_rows)

    if missing:
        print("[missing] %d fold/level/family cells" % len(missing))
        for item in missing[:12]:
            print("  -", item)
    if _SKIPPED_METRICS:
        print("[skip metrics] %s" % ", ".join(sorted(_SKIPPED_METRICS)))
    print("[done] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
