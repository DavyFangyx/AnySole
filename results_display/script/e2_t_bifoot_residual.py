#!/usr/bin/env python3
"""E2 T+BiFoot row for V8_IMU_Residual: own T2M pose + shared BiFoot trajectory."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "results/diagnostics/20261111_variant_sweep"
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(REPO))
from bifoot_clean_e2_row import metrics_for_session  # noqa: E402
import anysole.phase_common as PC  # noqa: E402

MOTION = REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
CACHE = DIAG / "clean_e2_t_bifoot/_cache"
PROTO = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual/clean"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
KEYS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "traj_ATE",
        "GroundPen", "FootSliding", "Jitter_pred")

def main():
    per_fold = {}
    for fold in FOLDS:
        z = np.load(CACHE / ("tr_%s.npz" % fold))
        proto = json.loads((PROTO / ("%s.json" % fold)).read_text())
        sessions = {str(e["session_id"]): e
                    for e in proto["evaluation_contract"]["sessions"]}
        rows = []
        for sid, entry in sessions.items():
            npz = MOTION / fold / ("%s_T2M.npz" % sid)
            if sid not in z.files or not npz.is_file():
                continue
            d = np.load(npz)
            disp = d["joint_xyz_world"]
            native = np.stack((disp[..., 0], disp[..., 2], -disp[..., 1]), -1)
            gt = PC.load_gt_joints(sid)
            tr = np.asarray(z[sid], np.float64)
            m = metrics_for_session(sid, fold, tr, gt, native,
                                    d["frame_indices"], d["valid_mask"],
                                    entry["frame_ids"])
            rows.append(m)
        w = np.asarray([r["n_valid"] for r in rows], float)
        micro = {k: float(np.average([r[k] for r in rows], weights=w)) for k in KEYS}
        micro["n_valid"] = int(w.sum())
        per_fold[fold] = micro
        print(fold, {k: round(micro[k], 3) for k in KEYS}, "n", int(w.sum()))
    mean = {}
    std = {}
    for k in KEYS:
        vals = [per_fold[f][k] for f in FOLDS]
        mean[k] = float(np.mean(vals))
        std[k] = float(np.std(vals))
    out = MOTION / "V8_IMU_Residual_TBiFoot.json"
    out.write_text(json.dumps({"per_fold": per_fold, "mean": mean,
                               "std": std}, indent=1))
    print("wrote", out)
    print("mean", {k: round(mean[k], 3) for k in KEYS})

if __name__ == "__main__":
    main()
