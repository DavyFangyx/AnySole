#!/usr/bin/env python3
"""Run real RTMPose HALPE-26 observations into both MMVP input trees.

The script intentionally imports MMPose lazily.  It can therefore be syntax
checked in the main training environment while execution remains explicit in
the dedicated pose environment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
MODEL_DEFAULT = WORKSPACE / "dependencies/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_paths(row: dict) -> tuple[str, str, Path]:
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    date, subject, session = parts[cam + 1 : cam + 4]
    color = WORKSPACE / "derived/pressure_tookit/images" / date / subject / session / "color"
    return date, subject, color


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


def run_session(session: str, row: dict, inferencer, force: bool) -> dict:
    date, subject, color_root = session_paths(row)
    images = sorted(p for p in color_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(images) != int(row["n_frames"]):
        raise ValueError(f"{session}: RGB count {len(images)} != manifest {row['n_frames']}")
    roots = [
        WORKSPACE / "derived/pressure_tookit/input" / subject / session / "keypoints",
        WORKSPACE / "derived/VP-MoCap" / date / subject / session / "keypoints",
    ]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        outputs = [root / f"{index:03d}.npy" for root in roots]
        if all(path.is_file() for path in outputs) and not force:
            continue
        prediction = next(inferencer(str(image)))
        points, scores = first_person(prediction)
        payload = {"keypoints": points, "keypoint_scores": scores,
                   "source_image": image.name, "model": "RTMPose HALPE-26"}
        for output in outputs:
            np.save(output, payload)
    return {"session": session, "frames": len(images), "outputs": [str(root) for root in roots]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--config", required=True, help="MMPose HALPE-26 config")
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(
        pose2d=args.config, pose2d_weights=args.model,
        device=args.device, show_progress=False)
    rows = read_manifest()
    sessions = [args.session] if args.session else split_sessions(args.split)
    result = [run_session(session, rows[session], inferencer, args.force) for session in sessions]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
