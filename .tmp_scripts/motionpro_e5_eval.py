#!/usr/bin/env python3
"""Evaluate MotionPRO E5 variants per fold (V8 four-fold口径).

Per fold: evaluate the fold's split test sessions against the fold manifest
(its test-session frame ids are identical to the V8 cells), frame-weighted
aggregation via the canonical solver.  Writes

  results/baselines/MotionPRO/predictions/<variant>/<fold>/eval_motion/metrics/
      test_comparison.json   per-fold summary (canonical solver keys)
      extra_metrics.json     frame-weighted ankle_foot_mpjpe / traj_ATE

and one four-fold summary per variant:

  results/baselines/MotionPRO/predictions/<variant>/metrics/fourfold_summary.json
      metric -> {"mean": ..., "std": ...} over the four folds
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Baselines"))

from Baselines.utils.gt_loading import protocol_gt, valid_mask  # noqa: E402
from Baselines.utils.solver import METRICS, aggregate, metrics, pelvis_align  # noqa: E402
from Baselines.utils.protocols import PROTOCOL_PELVIS  # noqa: E402

RESULTS = ROOT / "results/baselines/MotionPRO"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
FOLD_SPLITS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12",
    "fS5S11S12": "splits_test_S5S11S12_val_S13",
    "fS6S7": "splits_test_S6S7_val_S14",
    "fS5S6S8": "splits_test_S5S6S8_val_S7",
}
VARIANTS = [
    "e5_clean",
    "e5_occ_lower010", "e5_occ_lower020", "e5_occ_lower030", "e5_occ_lower040",
    "e5_occ_lower050", "e5_occ_lower060", "e5_occ_lower070", "e5_occ_lower080",
    "e5_occ_lower090",
    "e5_gap_r010", "e5_gap_r020", "e5_gap_r040", "e5_gap_r060",
    "e5_scene_clean", "e5_scene",
]

_mans: dict[str, dict] = {}


def fold_manifest(fold: str) -> dict:
    if fold not in _mans:
        rows = {}
        path = ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13" / f"{fold}_manifest.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
        _mans[fold] = rows
    return _mans[fold]


def fold_test_sessions(fold: str, kind: str) -> list[str]:
    csv_name = f"splits_sceneocc_{fold}" if kind == "scene" else FOLD_SPLITS[fold]
    path = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{csv_name}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    sessions = [(r.get("test") or "").strip() for r in rows if (r.get("test") or "").strip()]
    return [s for s in sessions if s != "S12102"]


def ankle_foot_mpjpe_frames(pred: np.ndarray, gt: np.ndarray, names,
                            keep: np.ndarray) -> tuple[float, int]:
    """Frame-weighted ankle/foot MPJPE: per-frame mean over the 4 joints."""
    pelvis = PROTOCOL_PELVIS["smpl24"]
    p_aligned, _ = pelvis_align(pred, pelvis_indices=pelvis)
    g_aligned, _ = pelvis_align(gt, pelvis_indices=pelvis)
    idx = [i for i, n in enumerate(names)
           if n in ("left_ankle", "right_ankle", "left_foot", "right_foot")]
    d = np.linalg.norm(p_aligned[keep][:, idx] - g_aligned[keep][:, idx], axis=-1)
    # sum of per-frame 4-joint means; denominator = frame count
    return float(d.mean(axis=1).sum() * 1000.0), int(len(d))


def evaluate_variant(variant: str, kind: str) -> dict[str, dict]:
    fold_vals: dict[str, dict] = {}
    for fold in FOLDS:
        root = RESULTS / "predictions" / variant / fold / "eval_motion"
        expected = fold_test_sessions(fold, kind)
        present = {p.stem for p in root.glob("*.npz")} if root.is_dir() else set()
        missing = [s for s in expected if s not in present]
        extra = [s for s in present if s not in expected]
        if missing or extra:
            raise RuntimeError(
                f"{variant} {fold}: prediction set mismatch "
                f"missing={missing[:5]} extra={extra[:5]}")
        rows = []
        ankle_sum = 0.0
        ankle_n = 0
        rms_root: list[float] = []
        for sid in expected:
            npz = root / f"{sid}.npz"
            meta = fold_manifest(fold)[sid]
            z = np.load(npz, allow_pickle=True)
            pred = np.asarray(z["joint_xyz_world"], dtype=np.float64)
            pred_mask = np.asarray(z["valid_mask"], dtype=bool)
            names = tuple(str(x) for x in z["joint_names"])
            gt, _ = protocol_gt(meta, "smpl24")
            n_eval = min(len(pred), len(gt), int(meta["n_frames"]))
            keep = valid_mask(meta, n_eval) & pred_mask[:n_eval]
            if not keep.any():
                continue
            values = metrics(pred[:n_eval], gt[:n_eval], keep, 40.0,
                             protocol="smpl24", names=names)
            values["n_valid_frames"] = int(keep.sum())
            values["session_id"] = sid
            values["status"] = "ok"
            values["protocol"] = "smpl24-native"
            rows.append(values)
            s, n = ankle_foot_mpjpe_frames(pred[:n_eval], gt[:n_eval], names, keep)
            ankle_sum += s
            ankle_n += n
            d = np.linalg.norm(pred[keep, 0] - gt[keep, 0], axis=-1)
            rms_root.append(float(np.sqrt((d ** 2).mean()) * 1000.0))
        if len(rows) != len(expected):
            raise RuntimeError(
                f"{variant} {fold}: evaluated {len(rows)}/{len(expected)} sessions")
        summary = aggregate(rows)
        out_metrics = root / "metrics"
        out_metrics.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "mmvp_native_metrics_v2",
            "model": "MotionPRO",
            "display_name": "MotionPRO",
            "family": "baseline",
            "mode": "VT2M",
            "split": fold,
            "evaluation_protocol": "smpl24-native",
            "summary": {key: float(summary[key]) for key in ("n_valid_frames", *METRICS)},
            "n_sessions": len(rows),
        }
        (out_metrics / "test_comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        extra = {
            "ankle_foot_mpjpe": float(ankle_sum / ankle_n) if ankle_n else float("nan"),
            "traj_ATE": float(np.mean(rms_root)) if rms_root else float("nan"),
            "n_sessions": len(rows),
            "n_frames": int(summary["n_valid_frames"]),
        }
        (out_metrics / "extra_metrics.json").write_text(
            json.dumps(extra, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fold_vals[fold] = {key: float(summary[key]) for key in ("n_valid_frames", *METRICS)}
        fold_vals[fold]["ankle_foot_mpjpe"] = extra["ankle_foot_mpjpe"]
        print(f"[eval] {variant} {fold}: n={len(rows)} "
              f"pa={summary['pa_mpjpe_mm']:.2f} global={summary['global_mpjpe_mm']:.1f} "
              f"ate={summary['root_ate_mm']:.1f} pen={summary['ground_pen_mm']:.1f} "
              f"ankle={extra['ankle_foot_mpjpe']:.1f}", flush=True)
    if len(fold_vals) != 4:
        raise RuntimeError(f"{variant}: only {len(fold_vals)}/4 folds evaluated")
    summary_out = RESULTS / "predictions" / variant / "metrics"
    summary_out.mkdir(parents=True, exist_ok=True)
    fourfold = {}
    keys = ("n_valid_frames", *METRICS, "ankle_foot_mpjpe")
    for key in keys:
        vals = [fold_vals[f].get(key) for f in FOLDS if f in fold_vals]
        vals = [v for v in vals if v is not None and np.isfinite(float(v))]
        if vals:
            fourfold[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                            "n_folds": len(vals)}
    (summary_out / "fourfold_summary.json").write_text(
        json.dumps(fourfold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return fourfold


def main() -> int:
    failed = []
    for variant in VARIANTS:
        kind = "scene" if variant.startswith("e5_scene") else "occ"
        try:
            evaluate_variant(variant, kind)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {variant}: {exc}", flush=True)
            failed.append(variant)
    if failed:
        print(f"eval incomplete: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
