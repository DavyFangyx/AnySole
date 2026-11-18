#!/usr/bin/env python3
"""Compute PA-MPJPE_anklefoot for E5 baselines, and MotionPRO's missing MPJRE.

PA-MPJPE_anklefoot is never written by Baselines/utils/solver.py::metrics()
(it only returns whole-body pa_mpjpe_mm); anysole/utils/eval_protocol.py's
version does one full-24-joint per-frame Procrustes alignment, then reads
the error on the ankle/foot joint subset from the aligned points. This
reruns exactly that (similarity_align + subset readout) against the same
predictions/GT the existing baseline test_comparison.json already evaluated,
without touching the canonical Baselines schema.

MotionPRO's mpjae_deg is NaN everywhere because .tmp_scripts/motionpro_e5_eval.py
never passed pred_rotations/gt_rotations into metrics(); the archive's `poses`
field (axis-angle) has always been there, so this loads it and recomputes.

Output: one JSON per baseline next to its existing predictions:
  results/baselines/<Model>/predictions/_e5_anklefoot_mpjre.json
  {condition: {"pa_mpjpe_anklefoot_mm": float, "mpjae_deg": float or null}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Baselines"))

from Baselines.utils.gt_loading import (  # noqa: E402
    protocol_gt, protocol_gt_rotations, read_manifest, valid_mask,
)
from anysole.types import ANKLE_FOOT_JOINTS  # noqa: E402
from anysole.utils.metrics import (  # noqa: E402
    matrix_rotation_error_degrees, mean_point_error, similarity_align,
)

PRED_ROOT = ROOT / "results/baselines"

WHAM_GVHMR_CONDITIONS = (
    "e5_clean",
    *("e5_occ_lower%03d" % i for i in range(10, 91, 10)),
    *("e5_gap_r%03d" % i for i in (10, 20, 40, 60)),
    "e5_scene_clean", "e5_scene",
)
MOTIONPRO_FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MOTIONPRO_VARIANTS = (
    "e5_clean",
    *("e5_occ_lower%03d" % i for i in range(10, 91, 10)),
    *("e5_gap_r%03d" % i for i in (10, 20, 40, 60)),
    "e5_scene_clean", "e5_scene",
)

_manifest_cache: dict[tuple[str, str], dict[str, dict]] = {}


def manifest_rows(manifest_path: str, split: str, split_csv: str) -> dict[str, dict]:
    key = (manifest_path, split_csv)
    if key not in _manifest_cache:
        rows = read_manifest(Path(manifest_path), split, split_csv=Path(split_csv))
        _manifest_cache[key] = {r["session_id"]: r for r in rows}
    return _manifest_cache[key]


def keep_mask(row: dict, n_eval: int, pred_mask: np.ndarray) -> np.ndarray:
    keep = valid_mask(row, n_eval) & pred_mask[:n_eval]
    if not str(row.get("frame_ids_pinned") or "").lower() in {"1", "true", "yes"}:
        pos = np.flatnonzero(keep)
        if len(pos) >= 20:
            keep[pos[20 * (len(pos) // 20):]] = False
    return keep


def pa_mpjpe_anklefoot(pred: np.ndarray, gt: np.ndarray) -> float:
    aligned = similarity_align(pred, gt)
    return float(mean_point_error(
        aligned[:, ANKLE_FOOT_JOINTS], gt[:, ANKLE_FOOT_JOINTS], scale=1000.0).mean())


_gt_cache: dict[str, np.ndarray] = {}


def cached_gt(row: dict) -> np.ndarray:
    sid = row["session_id"]
    if sid not in _gt_cache:
        gt, _ = protocol_gt(row, "smpl24")
        _gt_cache[sid] = gt
    return _gt_cache[sid]


def wham_gvhmr(model: str) -> dict[str, dict]:
    out = {}
    for cond in WHAM_GVHMR_CONDITIONS:
        js_path = PRED_ROOT / model / "predictions" / cond / "eval_motion/metrics/test_comparison.json"
        if not js_path.is_file():
            continue
        doc = json.loads(js_path.read_text(encoding="utf-8"))
        rows_meta = manifest_rows(doc["manifest"], doc["split"], doc["split_csv"])
        npz_dir = PRED_ROOT / model / "predictions" / cond / "eval_motion"
        fallback_dir = PRED_ROOT / model / "predictions/local_gen/eval_motion"
        vals, weights = [], []
        mismatches = []
        for sess in doc["sessions"]:
            if sess.get("status") != "ok":
                continue
            sid = sess["session_id"]
            p = npz_dir / f"{sid}.npz"
            if not p.is_file():
                p = fallback_dir / f"{sid}.npz"
            if not p.is_file():
                continue
            row = rows_meta[sid]
            z = np.load(p, allow_pickle=True)
            pred = np.asarray(z["joint_xyz_world"], dtype=np.float64)
            pred_mask = np.asarray(z["valid_mask"], dtype=bool)
            gt = cached_gt(row)
            n_eval = min(len(pred), len(gt), int(float(row["n_frames"])))
            keep = keep_mask(row, n_eval, pred_mask)
            if not keep.any():
                continue
            if keep.sum() != int(sess["n_valid_frames"]):
                mismatches.append(sid)
            v = pa_mpjpe_anklefoot(pred[:n_eval][keep], gt[:n_eval][keep])
            vals.append(v)
            weights.append(int(keep.sum()))
        agg = float(np.average(vals, weights=weights)) if vals else float("nan")
        mpjre_vals = [s["mpjae_deg"] for s in doc["sessions"]
                      if s.get("status") == "ok" and np.isfinite(s.get("mpjae_deg", float("nan")))]
        mpjre_w = [s["n_valid_frames"] for s in doc["sessions"]
                   if s.get("status") == "ok" and np.isfinite(s.get("mpjae_deg", float("nan")))]
        mpjre = float(np.average(mpjre_vals, weights=mpjre_w)) if mpjre_vals else None
        out[cond] = {"pa_mpjpe_anklefoot_mm": agg, "mpjae_deg": mpjre,
                     "n_sessions": len(vals), "mismatches": mismatches}
        print(f"[{model}] {cond}: pa_mpjpe_anklefoot={agg:.3f} mpjae_deg={mpjre} "
              f"n={len(vals)} mismatches={len(mismatches)}")
    return out


def fold_test_sessions(fold: str, kind: str) -> list[str]:
    import csv
    csv_name = f"splits_sceneocc_{fold}" if kind == "scene" else {
        "fS5S6S11": "splits_test_S5S6S11_val_S12",
        "fS5S11S12": "splits_test_S5S11S12_val_S13",
        "fS6S7": "splits_test_S6S7_val_S14",
        "fS5S6S8": "splits_test_S5S6S8_val_S7",
    }[fold]
    path = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{csv_name}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    sessions = [(r.get("test") or "").strip() for r in rows if (r.get("test") or "").strip()]
    return [s for s in sessions if s != "S12102"]


_fold_manifest_cache: dict[str, dict] = {}


def fold_manifest(fold: str) -> dict:
    if fold not in _fold_manifest_cache:
        rows = {}
        path = ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13" / f"{fold}_manifest.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
        _fold_manifest_cache[fold] = rows
    return _fold_manifest_cache[fold]


_gt_rot_cache: dict[str, np.ndarray] = {}


def cached_gt_rot(row: dict) -> np.ndarray:
    sid = row["session_id"]
    if sid not in _gt_rot_cache:
        data = protocol_gt_rotations(row, "smpl24")
        _gt_rot_cache[sid] = np.asarray(data["rotations"], dtype=np.float64)
    return _gt_rot_cache[sid]


def motionpro() -> dict[str, dict]:
    out = {}
    for variant in MOTIONPRO_VARIANTS:
        kind = "scene" if variant.startswith("e5_scene") else "other"
        fold_pa, fold_mpjre = [], []
        for fold in MOTIONPRO_FOLDS:
            root = PRED_ROOT / "MotionPRO/predictions" / variant / fold / "eval_motion"
            sessions = fold_test_sessions(fold, kind)
            pa_vals, pa_w, mpjre_vals, mpjre_w = [], [], [], []
            for sid in sessions:
                npz = root / f"{sid}.npz"
                if not npz.is_file():
                    continue
                row = fold_manifest(fold)[sid]
                z = np.load(npz, allow_pickle=True)
                pred = np.asarray(z["joint_xyz_world"], dtype=np.float64)
                pred_mask = np.asarray(z["valid_mask"], dtype=bool)
                poses = np.asarray(z["poses"], dtype=np.float64).reshape(-1, 24, 3)
                gt = cached_gt(row)
                n_eval = min(len(pred), len(gt), int(float(row["n_frames"])))
                keep = valid_mask(row, n_eval) & pred_mask[:n_eval]
                if not keep.any():
                    continue
                pa_vals.append(pa_mpjpe_anklefoot(pred[:n_eval][keep], gt[:n_eval][keep]))
                pa_w.append(int(keep.sum()))
                pred_rot = Rotation.from_rotvec(
                    poses[:n_eval][keep].reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
                gt_rot = cached_gt_rot(row)[:n_eval][keep]
                mpjre_vals.append(float(matrix_rotation_error_degrees(pred_rot, gt_rot).mean()))
                mpjre_w.append(int(keep.sum()))
            if pa_vals:
                fold_pa.append(float(np.average(pa_vals, weights=pa_w)))
            if mpjre_vals:
                fold_mpjre.append(float(np.average(mpjre_vals, weights=mpjre_w)))
        pa_mean = float(np.mean(fold_pa)) if len(fold_pa) == 4 else float("nan")
        pa_std = float(np.std(fold_pa)) if len(fold_pa) == 4 else float("nan")
        mpjre_mean = float(np.mean(fold_mpjre)) if len(fold_mpjre) == 4 else float("nan")
        mpjre_std = float(np.std(fold_mpjre)) if len(fold_mpjre) == 4 else float("nan")
        out[variant] = {
            "pa_mpjpe_anklefoot_mm": {"mean": pa_mean, "std": pa_std, "n_folds": len(fold_pa)},
            "mpjae_deg": {"mean": mpjre_mean, "std": mpjre_std, "n_folds": len(fold_mpjre)},
        }
        print(f"[MotionPRO] {variant}: pa_anklefoot={pa_mean:.3f}±{pa_std:.3f} "
              f"mpjre={mpjre_mean:.3f}±{mpjre_std:.3f} (folds={len(fold_pa)}/{len(fold_mpjre)})")
    return out


def main():
    for model, fn in (("WHAM", wham_gvhmr), ("GVHMR", wham_gvhmr)):
        result = fn(model)
        p = PRED_ROOT / model / "_e5_anklefoot_mpjre.json"
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("wrote", p)
    result = motionpro()
    p = PRED_ROOT / "MotionPRO/_e5_anklefoot_mpjre.json"
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
