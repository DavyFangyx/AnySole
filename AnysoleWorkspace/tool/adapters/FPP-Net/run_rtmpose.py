#!/usr/bin/env python3
"""Produce RTMPose HALPE-26 keypoint sidecars with shared frame ids.

One sidecar per canonical frame, written to the FPP-Net (or PoseTransOpt)
model_inputs tree:

    <output-root>/<date>/<subject>/<session>/keypoints/<frame_id:06d>.npy

Each sidecar stores ``keypoints`` (26,2), ``keypoint_scores`` (26,),
``frame_id`` (the canonical shared frame id), the source image name and the
frontend model.  The stored frame id lets every downstream consumer verify
the join instead of trusting file order.

MMPose is imported lazily: this module can be syntax-checked anywhere and
only executed in an environment that provides ``mmpose`` (with the mmpose
tree at ``AnysoleWorkspace/assets/third_party/mmpose`` on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

MMPOSE_ROOT = WORKSPACE / "assets/third_party/mmpose"
MODEL_DEFAULT = (
    WORKSPACE / "assets/third_party/rtmpose"
    / "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
)
CONFIG_DEFAULT = (
    MMPOSE_ROOT / "configs/body_2d_keypoint/rtmpose/body8"
    / "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
)
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
DEFAULT_OUTPUT = WORKSPACE / "model_inputs/FPP-Net/adapter_v1"


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def first_person(prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    people = prediction.get("predictions", [[]])
    people = people[0] if people else []
    if not people:
        raise RuntimeError("RTMPose returned no person")
    person = max(people, key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))))
    points = np.asarray(person["keypoints"], dtype=np.float32)
    scores = np.asarray(person["keypoint_scores"], dtype=np.float32).reshape(-1)
    if points.shape != (26, 2) or scores.shape != (26,):
        raise ValueError(f"expected HALPE-26, got keypoints={points.shape}, scores={scores.shape}")
    return points, scores


def run_session(session: str, row: dict, inferencer, output_root: Path,
                force: bool, shard: int = 0, shards: int = 1,
                also_root: Path | None = None, limit: int = 0) -> dict:
    date, subject, sid = session_parts(row)
    color_root = FACTS_ROOT / date / subject / sid / "rgb"
    images = sorted(p for p in color_root.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(images) != int(row["n_frames"]):
        raise ValueError(f"{sid}: RGB count {len(images)} != manifest {row['n_frames']}")
    if limit > 0:
        images = images[:limit]
    out_dirs = [output_root / date / subject / sid / "keypoints"]
    if also_root is not None:
        out_dirs.append(also_root / date / subject / sid / "keypoints")
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    pending = [(index, image) for index, image in enumerate(images)
               if index % shards == shard
               and (any(not (out_dir / f"{index:06d}.npy").is_file()
                        for out_dir in out_dirs) or force)]
    for index, image in pending:
        prediction = next(inferencer(str(image)))
        points, scores = first_person(prediction)
        payload = {"keypoints": points, "keypoint_scores": scores,
                   "frame_id": index,
                   "source_image": image.name,
                   "model": "RTMPose HALPE-26"}
        for out_dir in out_dirs:
            np.save(out_dir / f"{index:06d}.npy", payload)
    return {"session": sid, "frames": len(images), "written": len(pending),
            "outputs": [str(d) for d in out_dirs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--also-output-root", default="",
                        help="secondary tree (e.g. PoseTransOpt adapter root) "
                             "written in the same inference pass")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="smoke limit: only the first N frames per session")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if str(MMPOSE_ROOT) not in sys.path:
        sys.path.insert(0, str(MMPOSE_ROOT))
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(
        pose2d=args.config, pose2d_weights=args.model,
        device=args.device, show_progress=False)
    rows = read_manifest()
    sessions = [args.session] if args.session else split_sessions(args.split)
    output_root = Path(args.output_root)
    also_root = Path(args.also_output_root) if args.also_output_root else None
    result = [run_session(session, rows[session], inferencer, output_root,
                          args.force, args.shard, args.shards, also_root,
                          args.limit_frames)
              for session in sessions]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
