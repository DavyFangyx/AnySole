#!/usr/bin/env python3
"""Cross-check exported visual baseline metrics against the package reference.

The two calibres are NOT numerically comparable (decision archive 11): the
package's anysole_aligned uses native RGB sampling / 257 sessions / session
macro means / smplx J_regressor joints, while the project evaluation uses the
canonical 40 Hz grid / test 36 sessions / frame weighting / protocol FK
joints.  This script therefore checks what survives the calibre change:
per-session rank structure and root-trajectory magnitude.

Usage (touch_gait):
  python AnysoleWorkspace/tool/adapters/visual_baselines/crosscheck_package.py \
      --package-root Vision_Baseline/visual_baselines [--split test]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"
BASELINES = ("WHAM", "SAM3DB", "GVHMR")
RANK_METRICS = ("mpjpe_mm", "pa_mpjpe_mm", "mpjae_deg", "pve_mm", "pve_t_mm",
                "accel_error_m_s2", "jitter_pred_m_s3", "w_mpjpe100_mm",
                "root_ate_mm", "foot_sliding_mm")
RHO_LIMIT = 0.9


def project_rows(model: str, split: str) -> dict[str, dict]:
    path = RESULTS / "baselines" / model / "metrics" / f"{split}_comparison.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in value["sessions"]:
        if row.get("status") != "ok":
            continue
        rows[row["session_id"]] = row
    return rows


def package_rows(package_root: str, baseline: str) -> dict[tuple[str, str], dict]:
    path = (Path(package_root) / "reference_results" / "3_Result" / "SMPL_evaluation"
            / baseline / "anysole_aligned" / "session_metrics.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["session"]: row for row in csv.DictReader(handle)}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",
                        default=str(ROOT / "Vision_Baseline" / "visual_baselines"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    summary = {}
    for model in BASELINES:
        mine = project_rows(model, args.split)
        reference = package_rows(args.package_root, model)
        shared = sorted(set(mine) & set(reference))
        summary[model] = {"n_sessions": len(mine), "shared_sessions": len(shared)}
        if len(shared) < 10:
            print(f"{model}: only {len(shared)} shared sessions; rank check skipped")
            continue
        checks = {}
        for metric in RANK_METRICS:
            a = np.asarray([mine[s].get(metric, np.nan) for s in shared], dtype=float)
            b = np.asarray([float(reference[s][metric]) for s in shared], dtype=float)
            keep = np.isfinite(a) & np.isfinite(b)
            if keep.sum() < 10:
                checks[metric] = {"rho": None, "ok": None}
                continue
            rho = float(spearmanr(a[keep], b[keep]).statistic)
            checks[metric] = {"rho": round(rho, 4), "ok": bool(rho >= RHO_LIMIT)}
        # Root trajectory magnitudes must stay in the same order of magnitude.
        ate_mine = np.median([mine[s]["root_ate_mm"] for s in shared])
        ate_ref = np.median([float(reference[s]["root_ate_mm"]) for s in shared])
        ratio = max(ate_mine, ate_ref) / (min(ate_mine, ate_ref) + 1e-9)
        checks["root_ate_median"] = {"mine": round(float(ate_mine), 1),
                                     "package": round(float(ate_ref), 1),
                                     "ratio": round(float(ratio), 2),
                                     "ok": bool(ratio < 2.0)}
        summary[model]["checks"] = checks
        bad = {k: v for k, v in checks.items() if v.get("ok") is False}
        print(f"{model}: shared={len(shared)}  "
              + " ".join(f"{k}={v.get('rho', v.get('ratio', ''))}" for k, v in checks.items()
                         if k in ("mpjpe_mm", "pa_mpjpe_mm", "root_ate_mm", "foot_sliding_mm"))
              + ("  FAIL: " + str(bad) if bad else "  OK"))
    output = ROOT / "AnysoleWorkspace" / "work" / "visual_baselines" / "crosscheck_project_vs_package.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
