#!/usr/bin/env python3
"""Per-unit SAM3DB inference worker (runs inside the vb_tactile environment).

Mirrors the source worker's run_unit assembly: per-frame Engine.infer with
the workspace-frontend bbox, then pickles the io.load_sam result contract.
Invoked by generate.py via subprocess with the vb_tactile python.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AnysoleWorkspace.tool.adapters.visual_baselines import common  # noqa: E402

FRONTEND_BBOX = ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1"


def _write_gap_result(work_dir: Path, row: dict, cam: int, unit: dict,
                      level: float) -> int:
    """Gap axis is free for the per-frame SAM3DB: reorder clean outputs by anchor."""
    import pickle
    from anysole.data.make_gap_cache import (CYCLE_DEFAULT, SEED_DEFAULT,
                                             build_gap_array)
    from AnysoleWorkspace.tool.adapters.visual_baselines import generate
    session = row["session_id"]
    date, _ = generate.session_parts(row)
    short = "%s_m1_%s_c%d" % (date, session, cam)
    clean_path = (generate.LOCAL_ROOT / "reference_results" / "3_Result" /
                  "SAM3DB_baseline" / short / "result.pkl")
    if not clean_path.is_file():
        raise FileNotFoundError("SAM3DB clean result missing: %s" % clean_path)
    with clean_path.open("rb") as fh:
        payload = pickle.load(fh)
    unit_paths = generate.unit_frame_paths(row, cam, unit)
    grid = generate.cache_grid(session, cam)
    rows = generate.unit_cache_rows(unit_paths, grid)
    anchors = build_gap_array(
        np.arange(len(grid), dtype=np.int64)[:, None], level,
        CYCLE_DEFAULT, SEED_DEFAULT, session)[:, 0]
    row_to_unit = {}
    for index, cache_row in enumerate(rows):
        row_to_unit.setdefault(int(cache_row), index)
    order = []
    for cache_row in rows:
        target = int(anchors[int(cache_row)])
        if target in row_to_unit:
            order.append(row_to_unit[target])
        else:
            order.append(min(range(len(rows)),
                             key=lambda j: abs(int(rows[j]) - target)))
    arrays = payload.get("arrays") or {}
    n = len(unit_paths)
    reordered = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if key == "frame" or array.ndim == 0 or array.shape[0] != n:
            reordered[key] = array
        else:
            reordered[key] = array[order]
    payload["arrays"] = reordered
    destination = Path(work_dir) / "result.pkl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fh:
        pickle.dump(payload, fh)
    print("wrote %s (gap reorder, %d frames)" % (destination, n))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--degrade", default="none",
                        help="none | occ:<pct> | gap:<pct> | scene")
    args = parser.parse_args()

    import json as _json
    rows = {}
    for line in (ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl").open(encoding="utf-8"):
        row = _json.loads(line)
        rows[row["session_id"]] = row
    row = rows[args.session]
    from AnysoleWorkspace.tool.workspace import resolve_uri
    recording = resolve_uri(row["video_path"], must_exist=True)
    date = recording.parts[-3]
    unit_manifest = ROOT / "AnysoleWorkspace/work/visual_baselines/reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl"
    unit_id = f"{date}/method_1_offset/{args.session}/camera_{args.cam}"
    unit = None
    for line in unit_manifest.open(encoding="utf-8"):
        value = _json.loads(line)
        if value.get("unit_id") == unit_id:
            unit = value
            break
    if unit is None:
        raise FileNotFoundError(unit_id)

    from AnysoleWorkspace.tool.adapters.visual_baselines import generate
    kind, level = generate.parse_degrade(args.degrade)
    if kind == "gap":
        return _write_gap_result(args.work_dir, row, args.cam, unit, level)
    unit_paths = generate.unit_frame_paths(row, args.cam, unit)
    stage_dir = None
    if kind in ("occ", "scene"):
        stage_dir = Path(args.work_dir) / "frames"
        frames = generate.stage_frames(row, args.cam, unit_paths, stage_dir,
                                       kind, level)
    else:
        frames = unit_paths
    calibration, _, _ = common.load_calibration(date)
    width, height = [int(v) for v in calibration["image_size_px"]]
    subject = recording.parts[-2]
    side = np.load(FRONTEND_BBOX / date / subject / args.session / "bbox.npz",
                   allow_pickle=False)
    visual = np.asarray(side["visual_time_s"], dtype=np.float64)
    boxes_norm = np.asarray(side["bbox_xyxy_norm"], dtype=np.float64)
    start = common.recording_start_day_s(row)
    boxes_px = []
    for record in unit["frames"]:
        canonical = int(np.argmin(np.abs(visual - (float(record["timestamp_seconds"]) - start))))
        norm = boxes_norm[canonical]
        boxes_px.append(np.asarray([norm[0] * width, norm[1] * height,
                                    norm[2] * width, norm[3] * height], dtype=np.float32))

    sys.path.insert(0, str(ROOT / "AnysoleWorkspace/tool/adapters"))
    sys.path.insert(0, str(ROOT / "Baselines/SAM3DB/fast-sam-3d-body"))
    from SAM3DB.config import runtime_environment
    import os as _os
    _os.environ.update(runtime_environment(args.gpu))
    from SAM3DB.runtime import Engine
    # runtime_environment 已把 CUDA_VISIBLE_DEVICES 隔离为单卡，设备号恒为 0。
    engine = Engine(device="cuda:0")
    try:
        array_lists = {"frame": [], "bbox_xyxy": [], "global_orient": [], "body_pose": [],
                       "betas": [], "camera_translation": [], "focal_length": [],
                       "similarity_scale": [], "similarity_rotation": [],
                       "similarity_translation": [], "smpl_joints_2d": [],
                       "alignment_rmse": []}
        for frame_path, bbox, record in zip(frames, boxes_px, unit["frames"]):
            output = engine.infer(frame_path, bbox, (width, height))
            array_lists["frame"].append(int(record["frame"]))
            for name in array_lists:
                if name != "frame":
                    array_lists[name].append(output[name])
    finally:
        engine.close()
        if stage_dir is not None:
            import shutil
            shutil.rmtree(stage_dir, ignore_errors=True)
    arrays = {name: np.asarray(values) for name, values in array_lists.items()}
    arrays["frame"] = arrays["frame"].astype(np.int32)
    for name in arrays:
        if name != "frame":
            arrays[name] = arrays[name].astype(np.float32)
    result = {"schema_id": "sam3db_result_v1", "schema_version": 1,
              "unit_id": unit["unit_id"], "config_hash": "project-layout",
              "frame_count": len(frames), "image_size": [width, height],
              "fps": 30.0, "arrays": arrays}
    destination = Path(args.work_dir) / "result.pkl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(result, handle)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
