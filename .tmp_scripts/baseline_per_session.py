#!/usr/bin/env python3
"""Per-session visual-baseline metrics for E5 occlusion/gap screening.

WHAM/GVHMR: results/baselines/<model>/predictions/<variant>/eval_motion/*.npz
MotionPRO:  results/baselines/MotionPRO/predictions/<variant>/<fold>/eval_motion/*.npz

Uses the same canonical solver as .tmp_scripts/motionpro_e5_eval.py.
Writes results/diagnostics/20261116_e5_session_detail/screening/baseline_<variant>.csv
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
from Baselines.utils.solver import metrics, pelvis_align  # noqa: E402
from Baselines.utils.protocols import PROTOCOL_PELVIS  # noqa: E402

OUT = ROOT / "results/diagnostics/20261116_e5_session_detail/screening"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")


def load_manifests() -> dict[str, dict]:
    rows = {}
    for fold in FOLDS:
        p = ROOT / "AnysoleWorkspace/protocol/splits/crossfit_woS13" / ("%s_manifest.jsonl" % fold)
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def ankle_foot(pred, gt, names, keep) -> float:
    pelvis = PROTOCOL_PELVIS["smpl24"]
    p_al, _ = pelvis_align(pred, pelvis_indices=pelvis)
    g_al, _ = pelvis_align(gt, pelvis_indices=pelvis)
    idx = [i for i, n in enumerate(names)
           if n in ("left_ankle", "right_ankle", "left_foot", "right_foot")]
    d = np.linalg.norm(p_al[keep][:, idx] - g_al[keep][:, idx], axis=-1)
    return float(d.mean(axis=1).mean() * 1000.0)


def eval_session(npz: Path, meta: dict) -> dict | None:
    z = np.load(npz, allow_pickle=True)
    pred = np.asarray(z["joint_xyz_world"], dtype=np.float64)
    pred_mask = np.asarray(z["valid_mask"], dtype=bool)
    names = tuple(str(x) for x in z["joint_names"])
    gt, _ = protocol_gt(meta, "smpl24")
    n = min(len(pred), len(gt), int(meta["n_frames"]))
    keep = valid_mask(meta, n) & pred_mask[:n]
    if not keep.any():
        return None
    m = metrics(pred[:n], gt[:n], keep, 40.0, protocol="smpl24", names=names)
    return {
        "mpjpe_mm": float(m["mpjpe_mm"]),
        "pa_mpjpe_mm": float(m["pa_mpjpe_mm"]),
        "mpjpe_anklefoot_mm": ankle_foot(pred[:n], gt[:n], names, keep),
        "ground_pen_mm": float(m["ground_pen_mm"]),
        "foot_sliding_mm": float(m["foot_sliding_mm"]),
        "global_mpjpe_mm": float(m["global_mpjpe_mm"]),
        "n_frames": int(keep.sum()),
    }


def main() -> None:
    variants = sys.argv[1:] or ["e5_occ_lower020", "e5_occ_lower090",
                                "e5_gap_r060", "e5_gap_r080"]
    mans = load_manifests()
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        rows = []
        for model in ("WHAM", "GVHMR", "MotionPRO"):
            root = ROOT / "results/baselines" / model / "predictions" / variant
            if not root.is_dir():
                print("missing", model, variant)
                continue
            if model == "MotionPRO":
                npz_dirs = [(f.name, f / "eval_motion") for f in sorted(root.glob("f*"))
                            if (f / "eval_motion").is_dir()]
            else:
                npz_dirs = [("", root / "eval_motion")]
            for fold, d in npz_dirs:
                for npz in sorted(d.glob("*.npz")):
                    sid = npz.stem
                    meta = mans.get(sid)
                    if meta is None:
                        continue
                    try:
                        vals = eval_session(npz, meta)
                    except Exception as exc:  # noqa: BLE001
                        print("ERR", model, variant, sid, exc)
                        continue
                    if vals is None:
                        continue
                    rows.append({"model": model, "fold": fold, "session_id": sid, **vals})
        path = OUT / ("baseline_%s.csv" % variant)
        if rows:
            with path.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print("[csv]", path, len(rows))


if __name__ == "__main__":
    main()
