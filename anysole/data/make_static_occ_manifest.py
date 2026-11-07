"""Static scene-occluder manifest for the E5 scene-occlusion pilot.

A calibrated vertical column is placed in the room.  For every session frame
we compute:
  - active: the column polygon intersects the person crop (HRNet must rerun)
  - behind: the person is farther from cam3 than the column at the overlap
  - vis:    visible fraction of the SAM3.1 person mask (1.0 = fully visible)
and aggregate the same non-overlapping 20-frame windows used by eval.

The column geometry is fixed in checkerboard world coordinates; each recording
date uses its own camera calibration.  Output is a per-session JSON manifest
consumed by ``make_scene_occ_cache.py`` (and later by the E5 exporters).

Pilot scope: actions 03/05/08/09, four representative sessions.
"""
from __future__ import annotations

import argparse
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

CAL_ROOT = ROOT / "AnysoleWorkspace/hf-raw/config/calibration"
MASK_INDEX = ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"
BBOX_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox_cam3"
SESSION_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"

# Fixed room column (checkerboard world metres; z=0 floor, z negative = up).
COLUMN = {
    "type": "cuboid",
    "center_xy": [1.45, -2.3],
    "size_xy": [0.4, 0.4],
    "height_m": 2.2,
    "fill": "gray_gradient",
}

TW = 20
FPS_DEFAULT = 40.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", default="",
                   help="Comma-separated session ids; empty = enumerate --actions.")
    p.add_argument("--actions", default="3,4,5,8,9",
                   help="Action numbers enumerated when --sessions is empty.")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results/diagnostics/20261107_e5_sceneocc_full/manifests")
    return p.parse_args(argv)


def load_mask_index() -> dict:
    index = {}
    with MASK_INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("camera") != "cam3":
                continue
            index.setdefault(row["session"], {})[int(row["frame_id"])] = row
    return index


def project(pts: np.ndarray, cam: dict, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    cp = checkerboard_to_camera(pts, cam)
    proj, _ = cv2.projectPoints(cp.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return proj.reshape(pts.shape[:-1] + (2,))


def column_corners() -> np.ndarray:
    x0, y0 = COLUMN["center_xy"]
    w, d = COLUMN["size_xy"]
    top = COLUMN["height_m"]
    return np.array([[x, y, z] for x in (x0 - w / 2, x0 + w / 2)
                     for y in (y0 - d / 2, y0 + d / 2) for z in (0.0, -top)],
                    dtype=np.float64)


def column_polygon(cam: dict, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    return cv2.convexHull(project(column_corners(), cam, K, D).astype(np.float32))


def expanded_bbox(box: np.ndarray, img_w: int, img_h: int, factor: float = 1.5) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(float(img_w - 1), cx + w / 2), min(float(img_h - 1), cy + h / 2)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def mask_overlap(person: np.ndarray, poly: np.ndarray):
    occ = np.zeros(person.shape, dtype=np.uint8)
    cv2.fillConvexPoly(occ, poly.astype(np.int32), 255)
    occ = occ > 0
    area = int(person.sum())
    inter = int((person & occ).sum())
    return area, inter, occ


def aggregate_windows(vis: np.ndarray, tw: int = TW) -> list:
    rows = []
    for start in range(0, len(vis), tw):
        v = vis[start:start + tw]
        if len(v) == 0:
            continue
        full_frac = float((v <= 0.05).mean())
        mean = float(v.mean())
        if full_frac >= 0.5:
            state = "full"
        elif mean >= 0.95:
            state = "clean"
        else:
            state = "partial"
        rows.append({
            "start": int(start),
            "n_frames": int(len(v)),
            "vis_mean": round(mean, 4),
            "vis_min": round(float(v.min()), 4),
            "full_frac": round(full_frac, 4),
            "state": state,
        })
    return rows


def build_session(sid: str, mask_index: dict, out_dir: Path) -> dict:
    sess = load_shared_session(sid)
    meta = sess["meta"]
    date = meta["date"]
    n_frames = int(meta["frame_count"])
    fps = float(meta.get("fps", FPS_DEFAULT))
    cal = json.loads((CAL_ROOT / f"{date}.json").read_text())
    cam = cal["cameras"]["cam3"]
    K = np.asarray(cam["K"], dtype=np.float64)
    D = np.asarray(cam["D"], dtype=np.float64)

    # SMPL at the video frame grid (same query as the dataset).  Frames outside
    # the mocap span are clipped by load_smpl; they are flagged, not fatal -- the
    # eval support policy drops their windows.  Geometry still needs a depth
    # estimate, so use the clipped values for those few frames.
    q = sess["frames"]["mocap_time_s"]
    smpl = load_smpl(shared_smpl_path(sess), query_t=q)
    query_valid = np.asarray(smpl["query_valid"], dtype=bool)
    joints = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"],
                          smpl["offsets_m"], smpl["parents"])
    world = gt_smpl_to_checkerboard(joints, cal)
    cam_pts = checkerboard_to_camera(world, cam)
    depth = cam_pts[:, :, 2]  # (T, 24)

    poly = column_polygon(cam, K, D)
    x0, y0 = COLUMN["center_xy"]
    col_depth = float(checkerboard_to_camera(
        np.array([[x0, y0, -COLUMN["height_m"] / 2.0]]), cam)[0, 2])

    bbox = np.load(BBOX_ROOT / ("%s.npy" % sid))
    if bbox.shape[0] != n_frames:
        raise ValueError("%s: bbox length %d != %d" % (sid, bbox.shape[0], n_frames))

    rows = mask_index.get(sid, {})
    if not rows:
        raise ValueError("%s: no cam3 SAM3.1 mask rows" % sid)
    first = rows[min(rows)]
    img_h, img_w = cv2.imread(first["image_path"]).shape[:2]
    active = np.zeros(n_frames, dtype=np.uint8)
    behind = np.zeros(n_frames, dtype=np.uint8)
    vis = np.ones(n_frames, dtype=np.float64)
    mask_area = np.zeros(n_frames, dtype=np.int64)

    for t in range(n_frames):
        row = rows.get(t)
        if row is None:
            continue
        person = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        mask_area[t] = int(person.sum())
        area, inter, occ = mask_overlap(person, poly)
        crop = expanded_bbox(np.asarray(bbox[t][1:5], dtype=np.float64), img_w, img_h)
        inter_crop, _ = cv2.intersectConvexConvex(poly, crop)
        active[t] = 1 if inter_crop > 0 else 0
        # Depth order at the overlap: median camera depth of joints inside the
        # column silhouette; fall back to the root when no joint is inside.
        inside = np.array([_point_in_poly(cam_pts[t, j, :2], poly)
                           for j in range(cam_pts.shape[1])], dtype=bool)
        d = float(np.median(depth[t][inside])) if inside.any() else float(depth[t, 0])
        behind[t] = 1 if d > col_depth + 0.05 else 0
        if area > 0 and behind[t]:
            vis[t] = 1.0 - inter / float(area)
        else:
            vis[t] = 1.0

    windows = aggregate_windows(vis)
    events = []
    start = None
    for t, v in enumerate(vis):
        if v <= 0.05 and start is None:
            start = t
        elif v > 0.05 and start is not None:
            events.append({"start": int(start), "end": int(t - 1),
                           "duration_s": round((t - start) / fps, 3)})
            start = None
    if start is not None:
        events.append({"start": int(start), "end": int(n_frames - 1),
                       "duration_s": round((n_frames - start) / fps, 3)})

    subject = sess["dir"].parent.name
    endpoint = np.concatenate([vis[:10], vis[-10:]])
    manifest = {
        "session_id": sid,
        "action": sid[len(subject):len(subject) + 2],
        "date": date,
        "fps": fps,
        "n_frames": n_frames,
        "calibration": str(CAL_ROOT / f"{date}.json"),
        "column": COLUMN,
        "column_polygon_xy": poly.tolist(),
        "column_depth_m": round(col_depth, 4),
        "frames": {
            "active": active.tolist(),
            "behind": behind.tolist(),
            "vis": [round(float(v), 4) for v in vis],
            "mask_area": mask_area.tolist(),
        },
        "windows": windows,
        "full_events": events,
        "n_full_frames": int((vis <= 0.05).sum()),
        "full_frac": round(float((vis <= 0.05).mean()), 4),
        "active_frac": round(float(active.mean()), 4),
        "n_full_events": int(len(events)),
        "endpoint_vis_min": round(float(endpoint.min()), 4),
        "invalid_mocap_frames": np.where(~query_valid)[0].tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s.json" % sid)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("%s: frames=%d active=%.2f full=%.2f events=%s" % (
        sid, n_frames, manifest["active_frac"], manifest["full_frac"],
        [(e["start"], e["end"], e["duration_s"]) for e in events]))
    return manifest


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def enumerate_sessions(actions: set) -> list:
    out = []
    for sd in sorted(SESSION_ROOT.glob("*/*/*")):
        if not sd.is_dir():
            continue
        sid, subject = sd.name, sd.parent.name
        try:
            action = int(sid[len(subject):len(subject) + 2])
        except (ValueError, IndexError):
            continue
        if action in actions:
            out.append((sid, action))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    mask_index = load_mask_index()
    if args.sessions.strip():
        sessions = [(x.strip(), None) for x in args.sessions.split(",") if x.strip()]
    else:
        actions = {int(x) for x in args.actions.split(",") if x.strip()}
        sessions = enumerate_sessions(actions)
    summary = []
    for sid, action in sessions:
        if sid not in mask_index:
            raise KeyError("no cam3 mask index rows for %s" % sid)
        man = build_session(sid, mask_index, args.out)
        if action is None:
            action = int(man["action"])
        durs = [e["duration_s"] for e in man["full_events"]]
        summary.append({
            "session_id": sid,
            "action": action,
            "n_frames": man["n_frames"],
            "active_frac": man["active_frac"],
            "full_frac": man["full_frac"],
            "n_full_events": man["n_full_events"],
            "max_full_s": max(durs) if durs else 0.0,
            "endpoint_vis_min": man["endpoint_vis_min"],
            "invalid_mocap_frames": len(man["invalid_mocap_frames"]),
        })
    import csv
    (args.out / "summary.csv").unlink(missing_ok=True)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()) if summary else [])
        w.writeheader(); w.writerows(summary)
    print("manifests written to %s (%d sessions)" % (args.out, len(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
