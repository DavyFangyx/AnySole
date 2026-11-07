"""Grid search a room-fixed column with path clearance + hull occlusion."""
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

SESSION_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
CAL_ROOT = ROOT / "AnysoleWorkspace/hf-raw/config/calibration"
ACTIONS = (3, 4, 5, 8, 9)


def parse_action(sid, subject):
    return int(sid[len(subject):len(subject) + 2])


def project(pts, cal, cam, K, D):
    cp = checkerboard_to_camera(pts, cam)
    p, _ = cv2.projectPoints(cp.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return p.reshape(pts.shape[:-1] + (2,))


def column_poly(x0, y0, w, top, cal, cam, K, D):
    pts = np.array([[x, y, z] for x in (x0 - w / 2, x0 + w / 2)
                    for y in (y0 - w / 2, y0 + w / 2) for z in (0.0, -top)])
    return cv2.convexHull(project(pts, cal, cam, K, D).astype(np.float32))


def main() -> int:
    cal_cache, geom = {}, {}
    for sd in sorted(SESSION_ROOT.glob("*/*/*")):
        if not sd.is_dir():
            continue
        sid, subject = sd.name, sd.parent.name
        action = parse_action(sid, subject)
        if action not in ACTIONS:
            continue
        sess = load_shared_session(sid)
        date = sess["meta"]["date"]
        if date not in cal_cache:
            cal_cache[date] = json.loads((CAL_ROOT / f"{date}.json").read_text())
        cal = cal_cache[date]; cam = cal["cameras"]["cam3"]
        K = np.asarray(cam["K"], float); D = np.asarray(cam["D"], float)
        q = sess["frames"]["mocap_time_s"]
        try:
            d = load_smpl(shared_smpl_path(sess), query_t=q)
        except Exception:  # noqa: BLE001
            continue
        if not np.asarray(d["query_valid"]).all():
            continue
        joints = fk_pose6d_np(d["pose_6d"], d["trans_m"], d["offsets_m"], d["parents"])
        world = gt_smpl_to_checkerboard(joints, cal)
        proj = project(world, cal, cam, K, D)
        hulls = [cv2.convexHull(proj[t].astype(np.float32)) for t in range(len(proj))]
        xy = world[:, 0, [0, 1]]
        geom[sid] = dict(action=action, hulls=hulls, xy=xy, cal=cal, cam=cam,
                         K=K, D=D, fps=float(sess["meta"]["fps"]))
    print("usable sessions", len(geom))
    results = []
    for x0 in np.arange(1.5, 2.45, 0.1):
        for y0 in np.arange(-1.8, 0.61, 0.2):
            for w in (0.6, 0.8, 1.0):
                clears, covers, maxfull, anyfrac = [], [], [], []
                for sid, g in geom.items():
                    dist = np.linalg.norm(g["xy"] - np.array([x0, y0]), axis=1).min()
                    clears.append(dist - w / 2)
                    poly = column_poly(x0, y0, w, 2.2, g["cal"], g["cam"], g["K"], g["D"])
                    ratios = []
                    for hull in g["hulls"]:
                        inter, _ = cv2.intersectConvexConvex(hull, poly)
                        ratios.append(float(inter) / max(cv2.contourArea(hull), 1e-6))
                    ratios = np.asarray(ratios)
                    full = ratios > 0.75
                    runs, run = [], 0
                    for v in full:
                        if v:
                            run += 1
                        elif run:
                            runs.append(run); run = 0
                    if run:
                        runs.append(run)
                    covers.append(1 if (runs and max(runs) / g["fps"] >= 0.3) else 0)
                    maxfull.append(max(runs) / g["fps"] if runs else 0.0)
                    anyfrac.append(float((ratios > 0.1).mean()))
                clear_min = float(np.min(clears))
                cover = int(np.sum(covers))
                long = int(np.sum(np.asarray(maxfull) > 2.0))
                results.append(dict(x0=round(float(x0), 2), y0=round(float(y0), 2), w=w,
                                    clear_min=round(clear_min, 2), cover=cover, long=long,
                                    maxfull=round(float(np.mean(maxfull)), 2),
                                    anyfrac=round(float(np.mean(anyfrac)), 2)))
    # require clearance >= 0.5 m; prefer cover, low long, moderate anyfrac
    ok = [r for r in results if r["clear_min"] >= 0.5]
    ok.sort(key=lambda r: (r["cover"], -r["long"], -abs(r["anyfrac"] - 0.3)), reverse=True)
    print("top with clearance>=0.5:")
    for r in ok[:30]:
        print(r)
    out = ROOT / "results/diagnostics/20261107_e5_static_occ_audit/column_clearance.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
