"""Convert one raw BVH to a visual HIF-style skeleton sequence.

The BVH conversion is intentionally imported from the original
MocapVideoAligner_0811 project.  This script does not touch either existing
MotionPRO or Step2Motion renderer.

Example:
python resultsdisplay/script/BVH2HIF.py \
    --bvh /data/lizhe/projects/Tactile/MocapVideoAligner_0811/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from bvh_aligner_pose import parse_bvh_aligner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a raw BVH with MocapVideoAligner_0811 conversion.")
    parser.add_argument("--bvh", default="", help="Optional single BVH path. Default processes all test samples.")
    parser.add_argument("--split-csv", default="AnysoleWorkspace/splits/default/splits.csv")
    parser.add_argument("--seq-root", default="AnysoleWorkspace/derived/MotionPRO/sequences/cam3")
    parser.add_argument(
        "--out-dir",
        default="resultsdisplay/test",
        help="Output directory; default: resultsdisplay/test",
    )
    parser.add_argument("--stride", type=int, default=4, help="Render every Nth BVH frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all sampled frames.")
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS; 0 uses BVH FPS / stride.")
    return parser.parse_args()


def discover_test_bvhs(split_csv: Path, seq_root: Path) -> list[tuple[str, Path]]:
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        test_ids = [
            row["test"].strip()
            for row in csv.DictReader(handle)
            if row.get("test", "").strip()
        ]
    found = []
    for session_id in dict.fromkeys(test_ids):
        matches = sorted(path for path in seq_root.glob(f"*/*/{session_id}") if path.is_dir())
        if not matches:
            print(f"[skip] no sequence directory for {session_id}")
            continue
        meta_path = matches[0] / "align_meta.json"
        if not meta_path.is_file():
            print(f"[skip] missing align_meta.json for {session_id}")
            continue
        bvh_path = Path(json.loads(meta_path.read_text())["bvh_path"]).expanduser()
        if not bvh_path.is_file():
            print(f"[skip] missing BVH for {session_id}: {bvh_path}")
            continue
        found.append((session_id, bvh_path))
    return found


def render_frame(positions: np.ndarray, edges: list[tuple[int, int]], frame_index: int, total: int) -> np.ndarray:
    fig = plt.figure(figsize=(7, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F3E6D7")
    fig.patch.set_facecolor("#F3E6D7")
    ax.view_init(elev=18, azim=-62)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    for parent, child in edges:
        points = positions[[parent, child]]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#AF5A3B", linewidth=2.4)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="#355166", s=24)

    center = positions.mean(axis=0)
    span = max(float(np.max(np.ptp(positions, axis=0))), 1.0)
    radius = span * 0.65
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(f"BVH2HIF  frame {frame_index}/{total - 1}")

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def convert_one(bvh_path: Path, output_root: Path, sample_id: str, args: argparse.Namespace) -> None:
    stride = max(1, int(args.stride))

    parsed = parse_bvh_aligner(bvh_path)
    frame_ids = list(range(0, len(parsed["joints"]), stride))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError("BVH contains no motion frames after trimming")

    positions = []
    for frame_index in frame_ids:
        positions.append(parsed["joints"][frame_index])
    positions = np.stack(positions, axis=0)
    edges = [
        (int(parent), int(child))
        for child, parent in enumerate(parsed["parents"])
        if parent >= 0
    ]

    output_dir = output_root / sample_id / bvh_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "skeleton_zup.npz",
        positions=positions,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        parents=np.asarray([parent for parent, _ in edges], dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        raw_fps=np.asarray([parsed["fps"]], dtype=np.float64),
        trimmed_frame_count=np.asarray([len(parsed["joints"])], dtype=np.int64),
    )

    images = [render_frame(frame, edges, frame_id, len(parsed["joints"])) for frame, frame_id in zip(positions, frame_ids)]
    gif_path = output_dir / "skeleton_zup.gif"
    Image.fromarray(images[0]).save(
        gif_path,
        save_all=True,
        append_images=[Image.fromarray(image) for image in images[1:]],
        duration=max(20, int(round(1000.0 / (args.fps or parsed["fps"] / stride)))),
        loop=0,
    )

    mp4_path = output_dir / "skeleton_zup.mp4"
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps or parsed["fps"] / stride, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {mp4_path}")
    for image in images:
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"BVH: {bvh_path}")
    print(f"raw_fps={parsed['fps']:.6f} trimmed_frames={len(parsed['joints'])} rendered={len(images)}")
    print(f"output: {output_dir}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.out_dir).expanduser()
    if args.bvh:
        bvh_path = Path(args.bvh).expanduser().resolve()
        if not bvh_path.is_file():
            raise FileNotFoundError(f"BVH not found: {bvh_path}")
        convert_one(bvh_path, output_root, bvh_path.stem, args)
        return

    split_csv = Path(args.split_csv).expanduser().resolve()
    seq_root = Path(args.seq_root).expanduser().resolve()
    samples = discover_test_bvhs(split_csv, seq_root)
    print(f"Found {len(samples)} test BVH files from {split_csv}")
    for sample_id, bvh_path in samples:
        convert_one(bvh_path, output_root, sample_id, args)


if __name__ == "__main__":
    main()
