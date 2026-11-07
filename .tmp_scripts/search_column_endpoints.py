"""Cheap geometric search: fixed column, endpoints clear, middle occluded.

Body proxy = convex hull of projected SMPL-24 joints; final candidates are
re-checked with the exact SAM-mask manifest builder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
from anysole.data.workspace_adapter import load_shared_session, shared_smpl_path  # noqa: E402
from anysole.data.smpl_io import load_smpl  # noqa: E402
from anysole.data.rgb_smpl.coordinates import (  # noqa: E402
    gt_smpl_to_checkerboard, checkerboard_to_camera)
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402

SESSIONS = ["S5033", "S11051", "S5083", "S11091"]
CAL_ROOT = ROOT / "AnysoleWorkspace/hf-raw/config/calibration"


def project(pts, cam, K, D):
    cp = checkerboard_to_camera(pts, cam)
    p, _ = cv2.projectPoints(cp.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return p.reshape(pts.shape[:-1] + (2,))


def poly_for(x0, y0, w, cam, K, D):
    pts = np.array([[x, y, z] for x in (x0 - w / 2, x0 + w / 2)
                    for y in (y0 - w / 2, y0 + w / 2) for z in (0, -2.2)])
    return cv2.convexHull(project(pts, cam, K, D).astype(np.float32))


def ratio(hull, poly):
    inter, _ = cv2.intersectConvexConvex(hull, poly)
    return float(inter) / max(cv2.contourArea(hull), 1e-6)


def main():
    geom = {}
    for sid in SESSIONS:
        sess = load_shared_session(sid)
        date = sess["meta"]["date"]
        cal = json.load(open(CAL_ROOT / f"{date}.json"))
        cam = cal["cameras"]["cam3"]
        K = np.asarray(cam["K"], float); D = np.asarray(cam["D"], float)
        q = sess["frames"]["mocap_time_s"]
        d = load_smpl(shared_smpl_path(sess), query_t=q)
        J = fk_pose6d_np(d["pose_6d"], d["trans_m"], d["offsets_m"], d["parents"])
        W = gt_smpl_to_checkerboard(J, cal)
        C = checkerboard_to_camera(W, cam)
        proj = project(W, cam, K, D)
        hulls = [cv2.convexHull(proj[t].astype(np.float32)) for t in range(len(proj))]
        depth = W[:, 0, [0, 1]]
        geom[sid] = dict(cam=cam, K=K, D=D, hulls=hulls, xy=depth,
                         fps=float(sess["meta"]["fps"]), n=len(proj))
    results = []
    for x0 in np.arange(1.0, 2.25, 0.05):
        for y0 in np.arange(-2.6, 0.21, 0.1):
            for w in (0.3, 0.4, 0.5, 0.6, 0.8):
                end_max, mids, clears, fulls = 0.0, [], [], []
                for sid, g in geom.items():
                    poly = poly_for(x0, y0, w, g["cam"], g["K"], g["D"])
                    dist = np.linalg.norm(g["xy"] - np.array([x0, y0]), axis=1).min()
                    clears.append(dist - w / 2)
                    ends = [ratio(g["hulls"][t], poly)
                            for t in list(range(10)) + list(range(g["n"] - 10, g["n"]))]
                    end_max = max(end_max, max(ends) if ends else 0.0)
                    mid = [ratio(g["hulls"][t], poly) for t in range(10, g["n"] - 10)]
                    mid = np.asarray(mid)
                    full = mid > 0.75
                    runs, run = [], 0
                    for v in full:
                        if v:
                            run += 1
                        elif run:
                            runs.append(run); run = 0
                    if run:
                        runs.append(run)
                    fulls.append(max(runs) / g["fps"] if runs else 0.0)
                    mids.append(float(full.mean()))
                if end_max > 0.1 or min(clears) < 0.5:
                    continue
                results.append(dict(x0=round(float(x0), 2), y0=round(float(y0), 2), w=w,
                                    clear_min=round(float(min(clears)), 2),
                                    end_max=round(end_max, 2),
                                    cover=int(np.sum(np.asarray(fulls) >= 0.3)),
                                    maxfull=round(float(np.mean(fulls)), 2),
                                    mid_full=round(float(np.mean(mids)), 2),
                                    min_full=round(float(np.min(fulls)), 2)))
    ok = [r for r in results if r["cover"] == 4]
    ok.sort(key=lambda r: (abs(r["maxfull"] - 1.5), -r["mid_full"]))
    print("top candidates (endpoints clear, all covered, clearance>=0.5):")
    for r in ok[:40]:
        print(r)
    print("total", len(results), "ok", len(ok))


if __name__ == "__main__":
    main()
