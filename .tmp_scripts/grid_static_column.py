"""Simulate a static vertical occluder against GT body silhouettes (cam3).

Cheap geometric proxy for choosing one room-fixed column:
  body silhouette  = convex hull of projected SMPL-24 joints
  column silhouette = convex hull of projected 8 cuboid corners
  overlap ratio    = hull intersection area / body hull area
Outputs per candidate column: per-session event counts/durations.
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

SESSION_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
CAL_ROOT = ROOT / "AnysoleWorkspace/hf-raw/config/calibration"
ACTIONS = (3, 4, 5, 8, 9)


def parse_action(sid: str, subject: str) -> int:
    return int(sid[len(subject):len(subject) + 2])


def session_frames():
    out = []
    for sd in sorted(SESSION_ROOT.glob("*/*/*")):
        if not sd.is_dir():
            continue
        sid, subject = sd.name, sd.parent.name
        action = parse_action(sid, subject)
        if action not in ACTIONS:
            continue
        sess = load_shared_session(sid)
        out.append((sid, subject, action, sess))
    return out


def project(pts_world, cal, cam, K, D):
    cam_pts = checkerboard_to_camera(pts_world, cam)
    proj, _ = cv2.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return proj.reshape(pts_world.shape[:-1] + (2,))


def body_hull(proj):
    hull = cv2.convexHull(proj.astype(np.float32))
    return hull


def column_poly(x0, y0, w, d, top, cal, cam, K, D):
    xs = [x0 - w / 2, x0 + w / 2]
    ys = [y0 - d / 2, y0 + d / 2]
    pts = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, -top)],
                   dtype=np.float64)
    p = project(pts, cal, cam, K, D)
    return cv2.convexHull(p.astype(np.float32))


def main() -> int:
    srows = session_frames()
    cache = {}
    geom = {}
    print("sessions", len(srows))
    for sid, subject, action, sess in srows:
        date = sess["meta"]["date"]
        if date not in cache:
            cache[date] = json.loads((CAL_ROOT / f"{date}.json").read_text())
        cal = cache[date]
        cam = cal["cameras"]["cam3"]
        K = np.asarray(cam["K"], float); D = np.asarray(cam["D"], float)
        q = sess["frames"]["mocap_time_s"]
        try:
            d = load_smpl(shared_smpl_path(sess), query_t=q)
        except Exception as exc:  # noqa: BLE001
            print("[skip]", sid, exc)
            continue
        if not np.asarray(d["query_valid"]).all():
            print("[skip]", sid, "frames outside mocap span")
            continue
        joints = fk_pose6d_np(d["pose_6d"], d["trans_m"], d["offsets_m"], d["parents"])
        world = gt_smpl_to_checkerboard(joints, cal)
        proj = project(world, cal, cam, K, D)
        hulls = [body_hull(proj[t]) for t in range(len(proj))]
        geom[sid] = dict(hulls=hulls, cal=cal, cam=cam, K=K, D=D,
                         fps=float(sess["meta"]["fps"]), action=action)

    # candidate grid: one fixed column in the room
    best = []
    for x0 in (1.0, 1.2, 1.4, 1.6):
        for y0 in (-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4):
            for w in (0.6, 0.8, 1.0, 1.2):
                stats = []
                for sid, subject, action, sess in srows:
                    if sid not in geom:
                        continue
                    g = geom[sid]
                    poly = column_poly(x0, y0, w, w, 2.2, g["cal"], g["cam"],
                                       g["K"], g["D"])
                    ratios = []
                    for hull in g["hulls"]:
                        inter, _ = cv2.intersectConvexConvex(hull, poly)
                        a = cv2.contourArea(hull)
                        ratios.append(float(inter) / max(a, 1e-6))
                    ratios = np.asarray(ratios)
                    full = ratios > 0.75
                    # runs of full
                    runs, run = [], 0
                    for v in full:
                        if v:
                            run += 1
                        elif run:
                            runs.append(run); run = 0
                    if run:
                        runs.append(run)
                    stats.append(dict(sid=sid, action=action,
                                      frac=float((ratios > 0.1).mean()),
                                      full_frac=float(full.mean()),
                                      n_events=len([r for r in runs if r >= 5]),
                                      max_full_s=max(runs) / g["fps"] if runs else 0.0,
                                      runs=[r / g["fps"] for r in runs]))
                if not stats:
                    continue
                # coverage: sessions with at least one full event of >=0.3 s
                cover = sum(1 for s in stats if s["max_full_s"] >= 0.3)
                long = [s for s in stats if s["max_full_s"] > 2.0]
                best.append((cover, -len(long), sum(s["frac"] for s in stats), x0, y0, w, stats))
    best.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
    print("\nTop candidates (cover, -long, frac_sum, x0,y0,w):")
    for cover, nlong, frac, x0, y0, w, stats in best[:12]:
        acts = {}
        for s in stats:
            acts.setdefault(s["action"], []).append(s)
        print("x0=%.1f y0=%.1f w=%.1f cover=%d long=%d frac_sum=%.2f" % (
            x0, y0, w, cover, -nlong, frac))
        for a in sorted(acts):
            ss = acts[a]
            print("   act%02d n=%2d full_frac=%.2f n_events=%.1f max_full_s=%.2f frac=%.2f" % (
                a, len(ss), np.mean([s["full_frac"] for s in ss]),
                np.mean([s["n_events"] for s in ss]),
                np.mean([s["max_full_s"] for s in ss]),
                np.mean([s["frac"] for s in ss])))
    Path(ROOT / "results/diagnostics/20261107_e5_static_occ_audit/column_grid.json").write_text(
        json.dumps([dict(cover=c, long=(-nl), frac_sum=f, x0=x0, y0=y0, w=w,
                         stats=st)
                    for c, nl, f, x0, y0, w, st in best],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
