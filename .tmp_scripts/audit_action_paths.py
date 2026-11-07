"""Audit session root trajectories per action: suitability for a static occluder.

For every shared cam3 session, load the SMPL GT at the video frame grid,
FK the 24 joints, project them into cam3, and summarise:
  - world path length / net displacement / straightness (net/path)
  - reversal count along the principal horizontal axis (shuttle motion)
  - angular coverage around the path centroid (loop motion)
  - projected pelvis image range and body-bbox size

Output: results/diagnostics/20261107_e5_static_occ_audit/{summary.csv,per_session.json}
"""
from __future__ import annotations

import csv
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

OUT = ROOT / "results/diagnostics/20261107_e5_static_occ_audit"
SESSION_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
CAL_ROOT = ROOT / "AnysoleWorkspace/hf-raw/config/calibration"

ACTION_NAMES = {
    1: "站立重心左右移", 2: "单脚站立落地", 3: "折返走路慢", 4: "折返走路快",
    5: "绕圈跑", 6: "深蹲", 7: "开合跳", 8: "折返跑", 9: "横移",
    10: "坐下-站起", 11: "上楼梯",
}


def parse_action(sid: str, subject: str) -> tuple[int, int]:
    tail = sid[len(subject):]
    return int(tail[:2]), int(tail[2:])


def smooth(x: np.ndarray, k: int = 9) -> np.ndarray:
    if len(x) < k:
        return x
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(x, ker, mode="same")


def reversals_along_axis(xy: np.ndarray, frac: float = 0.25) -> int:
    """Count direction reversals of the dominant horizontal axis (shuttle)."""
    if len(xy) < 10:
        return 0
    d = np.diff(xy, axis=0)
    # principal axis
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    v = smooth(d @ axis, 9)
    # ignore velocity near the median speed
    thr = frac * np.median(np.abs(v)) if np.median(np.abs(v)) > 0 else 0.0
    v = np.where(np.abs(v) > thr, np.sign(v), 0.0)
    v = v[v != 0]
    if len(v) < 2:
        return 0
    return int(np.sum(v[1:] != v[:-1]))


def angular_coverage(xy: np.ndarray) -> float:
    """Fraction of the circle covered by the path around its centroid."""
    c = xy.mean(0)
    d = xy - c
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)
    ang = np.sort(ang)
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return float(1.0 - gaps.max() / (2 * np.pi))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cal_cache = {}
    rows, detail = [], {}
    for sd in sorted(SESSION_ROOT.glob("*/*/*")):
        if not sd.is_dir():
            continue
        sid = sd.name
        subject = sd.parent.name
        sess = load_shared_session(sid)
        meta = sess["meta"]
        date = meta["date"]
        action, trial = parse_action(sid, subject)
        if date not in cal_cache:
            cal_cache[date] = json.loads((CAL_ROOT / f"{date}.json").read_text())
        cal = cal_cache[date]
        cam = cal["cameras"]["cam3"]
        K = np.asarray(cam["K"], dtype=np.float64)
        D = np.asarray(cam["D"], dtype=np.float64)
        q = sess["frames"]["mocap_time_s"]
        try:
            smpl = load_smpl(shared_smpl_path(sess), query_t=q)
        except Exception as exc:  # noqa: BLE001
            print("[skip] %s: %s" % (sid, exc))
            continue
        if not np.asarray(smpl["query_valid"]).all():
            print("[skip] %s: %d/%d frames outside mocap span" % (
                sid, int((~np.asarray(smpl["query_valid"])).sum()), len(q)))
            continue
        joints = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"],
                              smpl["offsets_m"], smpl["parents"])
        world = gt_smpl_to_checkerboard(joints, cal)
        cam_pts = checkerboard_to_camera(world, cam)
        proj, _ = cv2.projectPoints(
            cam_pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        proj = proj.reshape(len(joints), -1, 2)
        pelvis_w = world[:, 0, :]
        # Checkerboard world: z = 0 is the floor plane (prepare_mmvp_observations),
        # so the horizontal ground plane is x-y and z is height.
        xy = pelvis_w[:, [0, 1]]
        dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        path_len = float(dxy.sum())
        net = float(np.linalg.norm(xy[-1] - xy[0]))
        speed = dxy * float(meta["fps"])
        px = proj[:, :, 0]
        py = proj[:, :, 1]
        body_w = px.max(1) - px.min(1)
        body_h = py.max(1) - py.min(1)
        cx = px.mean(1)
        cy = py.mean(1)
        rec = {
            "sid": sid, "subject": subject, "date": date, "action": action,
            "action_name": ACTION_NAMES.get(action, "?"), "trial": trial,
            "n_frames": int(len(q)),
            "world_x_range": float(np.ptp(xy[:, 0])),
            "world_z_range": float(np.ptp(xy[:, 1])),
            "path_len_m": path_len,
            "net_disp_m": net,
            "straightness": net / path_len if path_len > 0 else 0.0,
            "reversals": reversals_along_axis(xy),
            "angular_coverage": angular_coverage(xy),
            "speed_mean_mps": float(speed.mean()),
            "speed_p95_mps": float(np.percentile(speed, 95)),
            "img_cx_range": float(np.ptp(cx)),
            "img_cy_range": float(np.ptp(cy)),
            "img_bbox_x_min": float(px.min()),
            "img_bbox_x_max": float(px.max()),
            "img_bbox_y_min": float(py.min()),
            "img_bbox_y_max": float(py.max()),
            "img_body_w_med": float(np.median(body_w)),
            "img_body_h_med": float(np.median(body_h)),
            "img_cx_min": float(cx.min()), "img_cx_max": float(cx.max()),
        }
        rows.append(rec)
        detail[sid] = rec
        print("%s %-2d %-8s f=%4d path=%5.1fm net=%5.1fm str=%.2f rev=%2d ang=%.2f "
              "imgx=[%4.0f,%4.0f] imgy=[%4.0f,%4.0f]" % (
                  sid, action, ACTION_NAMES.get(action, "?"), rec["n_frames"],
                  path_len, net, rec["straightness"], rec["reversals"],
                  rec["angular_coverage"], cx.min(), cx.max(), cy.min(), cy.max()))
    keys = list(rows[0].keys()) if rows else []
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "per_session.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    # action aggregates
    print("\n=== action aggregates ===")
    for a in sorted(ACTION_NAMES):
        rs = [r for r in rows if r["action"] == a]
        if not rs:
            continue
        def ms(key):
            v = [r[key] for r in rs]
            return "%s=%.2f±%.2f" % (key, np.mean(v), np.std(v))
        print("%2d %-8s n=%2d frames=%4d %s %s %s %s" % (
            a, ACTION_NAMES[a], len(rs), sum(r["n_frames"] for r in rs),
            ms("path_len_m"), ms("net_disp_m"), ms("reversals"),
            ms("angular_coverage")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
