"""Render raw GT mocap BVHs as HIF-style skeleton animations (Test1).

The BVH conversion is intentionally imported from the original
MocapVideoAligner_0811 project.  This script does not touch either existing
MotionPRO or Step2Motion renderer.

Outputs are written below ``results_display/Test1_visualization/gt``:
    <session>/<bvh_stem>/skeleton_zup.npz         data file (always written)
    <session>/<bvh_stem>/{gif,mp4}/skeleton_zup.* animation (selected by --gen)

``ANYSOLE_RESULTSDISPLAY`` overrides the display root.

Example:
    python results_display/script/d_test1_data_viz.py
    python results_display/script/d_test1_data_viz.py --session S7013 --gen gif
    python results_display/script/d_test1_data_viz.py --bvh /path/to/single.bvh --gen mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import cli_common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.bvh_aligner_pose import parse_bvh_aligner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a raw BVH with MocapVideoAligner_0811 conversion.")
    cli_common.add_common_args(
        parser,
        seq_root=True,
        fps_default=0.0,
        stride_default=4,
        out_dir_default=cli_common.DISPLAY_ROOT / "Test1_visualization" / "gt",
    )
    parser.add_argument("--bvh", default="", help="Optional single BVH path. Default processes all test samples.")
    return parser.parse_args()


def discover_test_bvhs(session_ids: list, seq_root: Path) -> list[tuple[str, Path]]:
    found = []
    for session_id in session_ids:
        matches = sorted(path for path in seq_root.glob(f"*/*/{session_id}") if path.is_dir())
        if not matches:
            print(f"[skip] no sequence directory for {session_id}")
            continue
        meta_path = matches[0] / "align_meta.json"
        if not meta_path.is_file():
            print(f"[skip] missing align_meta.json for {session_id}")
            continue
        bvh_path = Path(cli_common.resolve_path(json.loads(meta_path.read_text())["bvh_path"])).expanduser()
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
    ax.set_title(f"GT BVH  frame {frame_index}/{total - 1}")

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def convert_one(bvh_path: Path, output_root: Path, sample_id: str, args: argparse.Namespace) -> None:
    stride = max(1, int(args.stride))
    output_dir = output_root / sample_id / bvh_path.stem
    npz_path = output_dir / "skeleton_zup.npz"
    gen_path = cli_common.media_path(output_dir, "skeleton_zup", args.gen)
    if cli_common.outputs_ready([npz_path, gen_path]) and not args.force:
        print(f"[skip] {sample_id}: outputs already exist under {output_dir}")
        return

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

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        positions=positions,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        parents=np.asarray([parent for parent, _ in edges], dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        raw_fps=np.asarray([parsed["fps"]], dtype=np.float64),
        trimmed_frame_count=np.asarray([len(parsed["joints"])], dtype=np.int64),
    )

    images = [render_frame(frame, edges, frame_id, len(parsed["joints"])) for frame, frame_id in zip(positions, frame_ids)]
    frame_fps = cli_common.viz_fps(args.fps or parsed["fps"], stride)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gen == "gif":
        cli_common.write_gif(images, gen_path, frame_fps)
    else:
        cli_common.write_mp4(images, gen_path, frame_fps)
    print(f"BVH: {bvh_path}")
    print(f"raw_fps={parsed['fps']:.6f} trimmed_frames={len(parsed['joints'])} rendered={len(images)}")
    print(f"output: {output_dir}")


def main() -> None:
    args = parse_args()
    output_root = Path(cli_common.resolve_path(args.out_dir))
    if args.bvh:
        bvh_path = Path(cli_common.resolve_path(args.bvh)).expanduser()
        if not bvh_path.is_file():
            raise FileNotFoundError(f"BVH not found: {bvh_path}")
        convert_one(bvh_path, output_root, bvh_path.stem, args)
        return

    split_csv = Path(cli_common.resolve_path(args.split_csv))
    seq_root = Path(cli_common.resolve_path(args.seq_root))
    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    samples = discover_test_bvhs(session_ids, seq_root)
    print(f"Found {len(samples)} test BVH files from {split_csv}")
    for sample_id, bvh_path in samples:
        convert_one(bvh_path, output_root, sample_id, args)


if __name__ == "__main__":
    main()
