"""Exact SAM-mask overlap for candidate static columns (cam3)."""
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
MASK_INDEX = ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"
ACTIONS = (3, 4, 5, 8, 9)
CANDS = [(1.6, 0.4, 1.2), (1.4, 0.4, 1.0), (1.2, 0.4, 0.8), (1.0, -1.6, 0.6),
         (1.4, 0.0, 0.6), (1.6, -0.4, 0.4)]


def parse_action(sid: str, subject: str) -> int:
    return int(sid[len(subject):len(subject) + 2])


def project(pts, cal, cam, K, D):
    cp = checkerboard_to_camera(pts, cam)
    p, _ = cv2.projectPoints(cp.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return p.reshape(pts.shape[:-1] + (2,))


def column_poly(x0, y0, w, d, top, cal, cam, K, D):
    pts = np.array([[x, y, z] for x in (x0 - w / 2, x0 + w / 2)
                    for y in (y0 - d / 2, y0 + d / 2) for z in (0.0, -top)])
    return cv2.convexHull(project(pts, cal, cam, K, D).astype(np.float32))


def main() -> int:
    index = {}
    for line in MASK_INDEX.read_text().splitlines():
        row = json.loads(line)
        if row["camera"] != "cam3":
            continue
        index.setdefault(row["session"], {})[int(row["frame_id"])] = row
    cal_cache = {}
    sessions = []
    for sd in sorted(SESSION_ROOT.glob("*/*/*")):
        if not sd.is_dir():
            continue
        sid, subject = sd.name, sd.parent.name
        action = parse_action(sid, subject)
        if action not in ACTIONS:
            continue
        sessions.append((sid, action, load_shared_session(sid)))
    for x0, y0, w in CANDS:
        out = []
        for sid, action, sess in sessions:
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
            cam_pts = checkerboard_to_camera(world, cam)
            depth = cam_pts[:, :, 2].mean(1)
            poly = column_poly(x0, y0, w, w, 2.2, cal, cam, K, D)
            col_depth = checkerboard_to_camera(
                np.array([[x0, y0, -1.1]]), cam)[0, 2]
            masks = index.get(sid, {})
            ratios = []
            for t in range(len(joints)):
                row = masks.get(t)
                if row is None:
                    ratios.append(0.0); continue
                mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    ratios.append(0.0); continue
                body = mask > 0
                occ = np.zeros_like(mask, np.uint8)
                cv2.fillConvexPoly(occ, poly.astype(np.int32), 255)
                occ = occ > 0
                area = body.sum()
                if area == 0:
                    ratios.append(0.0); continue
                overlap = (body & occ).sum()
                behind = depth[t] > col_depth + 0.05
                ratios.append(float(overlap) / float(area) if behind else 0.0)
            ratios = np.asarray(ratios)
            full = ratios > 0.8
            runs, run = [], 0
            for v in full:
                if v:
                    run += 1
                elif run:
                    runs.append(run); run = 0
            if run:
                runs.append(run)
            fps = float(sess["meta"]["fps"])
            out.append(dict(sid=sid, action=action,
                            frac=float((ratios > 0.1).mean()),
                            full_frac=float(full.mean()),
                            max_full_s=max(runs) / fps if runs else 0.0,
                            n_events=len([r for r in runs if r >= 4]),
                            runs=[round(r / fps, 2) for r in runs]))
        print("\n=== column x0=%.1f y0=%.1f w=%.1f ===" % (x0, y0, w))
        for a in sorted(set(s["action"] for s in out)):
            ss = [s for s in out if s["action"] == a]
            print(" act%02d n=%2d full=%.2f maxfull=%.2f events=%.1f frac=%.2f" % (
                a, len(ss), np.mean([s["full_frac"] for s in ss]),
                np.mean([s["max_full_s"] for s in ss]),
                np.mean([s["n_events"] for s in ss]),
                np.mean([s["frac"] for s in ss])))
        print("  per-session maxfull:", [(s["sid"], round(s["max_full_s"], 2),
                                          s["n_events"]) for s in out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
