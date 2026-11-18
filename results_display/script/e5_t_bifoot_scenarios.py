#!/usr/bin/env python3
"""T+BiFoot reference for E5 scene/gap subsets (same whole T channel)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "results/diagnostics/20261111_variant_sweep"
sys.path.insert(0, str(DIAG)); sys.path.insert(0, str(REPO))
from bifoot_clean_e2_row import metrics_for_session  # noqa: E402
import anysole_tta.phase_common as PC  # noqa: E402

MOTION = REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
CACHE = DIAG / "clean_e2_t_bifoot/_cache"
PROTO = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
KEYS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "traj_ATE",
        "GroundPen", "FootSliding", "Jitter_pred")
SCEN = ("sceneocc", "gap010", "gap020", "gap040", "gap060")

def one(fold, cond):
    z = np.load(CACHE / ("tr_%s.npz" % fold))
    proto = json.loads((PROTO / cond / ("%s.json" % fold)).read_text())
    rows = []
    for e in proto["evaluation_contract"]["sessions"]:
        sid = str(e["session_id"])
        npz = MOTION / fold / ("%s_T2M.npz" % sid)
        if sid not in z.files or not npz.is_file():
            continue
        d = np.load(npz)
        disp = d["joint_xyz_world"]
        native = np.stack((disp[..., 0], disp[..., 2], -disp[..., 1]), -1)
        gt = PC.load_gt_joints(sid)
        m = metrics_for_session(sid, fold, np.asarray(z[sid], np.float64), gt,
                                native, d["frame_indices"], d["valid_mask"],
                                e["frame_ids"])
        rows.append(m)
    if not rows:
        return None
    w = np.asarray([r["n_valid"] for r in rows], float)
    return {k: float(np.average([r[k] for r in rows], weights=w)) for k in KEYS}

def main():
    out = {}
    for cond in SCEN:
        per = {}
        for fold in FOLDS:
            m = one(fold, cond)
            if m:
                per[fold] = m
        if not per:
            continue
        mean = {k: float(np.mean([per[f][k] for f in per])) for k in KEYS}
        std = {k: float(np.std([per[f][k] for f in per])) for k in KEYS}
        out[cond] = {"per_fold": per, "mean": mean, "std": std}
        print(cond, {k: round(mean[k], 3) for k in KEYS})
    p = MOTION / "V8_IMU_Residual_TBiFoot_scenarios.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)

if __name__ == "__main__":
    main()
