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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work-dir", required=True)
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
    frames = generate.unit_frame_paths(row, args.cam, unit)
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
    engine = Engine(device=f"cuda:{args.gpu}")
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
