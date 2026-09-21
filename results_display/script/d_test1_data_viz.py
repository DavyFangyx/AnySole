"""Render raw GT mocap as skeleton animations (D_Test1), SMPL-24 preferred.

Per session the GT source is auto-detected: native SMPL-24 NPZ when the
session has one (the current protocol), falling back to the legacy BVH-23
path (``--protocol bvh23`` forces it).  SMPL data is read through
``utils.motion_io.load_session_gt`` (display frame, z-up meters); the BVH
path uses the original MocapVideoAligner_0811 conversion.  This script does
not touch either existing MotionPRO or Step2Motion renderer.

Outputs are written below ``results_display/data/d_test1_data_viz/``:
    <session-or-stem>/<smp24|bvh23>/skeleton_zup.npz   data file (always written)
    <session-or-stem>/<smp24|bvh23>/{gif,mp4}/skeleton_zup.* animation (--gen)
    <session-or-stem>/smp24/mesh_zup.npz + {gif,mp4}/mesh_zup.* (--mesh)

``ANYSOLE_RESULTSDISPLAY`` overrides the display root.

``--mesh`` renders the full SMPL surface (6890 vertices, LBS from
``utils.smpl_mesh``) with the skeleton overlaid, replacing the skeleton-only
outputs for SMPL-24 sessions; BVH-23 sessions fall back to skeleton-only with
a warning.  The default (and ``--only-bone``) keeps the skeleton-only output.

Example:
    python results_display/script/d_test1_data_viz.py
    python results_display/script/d_test1_data_viz.py --session S7013 --gen gif
    python results_display/script/d_test1_data_viz.py --session S7013 --mesh --gen mp4
    python results_display/script/d_test1_data_viz.py --protocol bvh23
    python results_display/script/d_test1_data_viz.py --bvh /path/to/single.bvh --gen mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # script/ -> results_display/ -> gait repo root (parents[0] = results_display/)
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from utils import cli_common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anysole.data.smpl_io import resolve_smpl_path
from anysole.types import SMPL_ROOTS
from utils.bvh_aligner_pose import parse_bvh_aligner
from utils.motion_io import load_session_gt, smpl_yup_to_display
from utils.smpl_mesh import mesh_from_archive, render_mesh_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render raw GT (SMPL-24 preferred, BVH-23 fallback).")
    cli_common.add_common_args(
        parser,
        seq_root=True,
        fps_default=0.0,
        stride_default=4,
        out_dir_default=cli_common.DISPLAY_ROOT / "data/d_test1_data_viz",
    )
    parser.add_argument("--bvh", default="", help="Optional single BVH path. Default processes all test samples.")
    parser.add_argument(
        "--protocol",
        choices=("auto", "smp24", "bvh23"),
        default="auto",
        help="GT source per session: auto = SMPL-24 when available else BVH-23 (default); "
        "smp24 / bvh23 force one protocol and skip sessions that lack it.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render the full SMPL surface (mesh + skeleton overlay) instead of skeleton-only. "
        "Needs SMPL-24 sessions; BVH-23 sessions fall back to skeleton-only.",
    )
    parser.add_argument(
        "--only-bone",
        action="store_true",
        help="Explicit skeleton-only rendering (the default); conflicts with --mesh.",
    )
    args = parser.parse_args()
    if args.mesh and args.only_bone:
        parser.error("--mesh and --only-bone are mutually exclusive")
    return args


def discover_test_sessions(session_ids: list, seq_root: Path) -> list[tuple[str, Path]]:
    found = []
    for session_id in session_ids:
        matches = sorted(path for path in seq_root.glob(f"*/*/{session_id}") if path.is_dir())
        if not matches:
            print(f"[skip] no sequence directory for {session_id}")
            continue
        seq_dir = matches[0]
        meta_path = seq_dir / "align_meta.json"
        if not meta_path.is_file():
            print(f"[skip] missing align_meta.json for {session_id}")
            continue
        found.append((session_id, seq_dir))
    return found


def session_source(seq_dir: Path, protocol: str) -> str:
    """'smp24' / 'bvh23' per the requested protocol and what the session has."""
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    has_smpl = False
    try:
        resolve_smpl_path(meta, tuple(SMPL_ROOTS))
        has_smpl = True
    except FileNotFoundError:
        pass
    if protocol == "smp24":
        return "smp24" if has_smpl else ""
    if protocol == "bvh23":
        bvh = Path(str(meta.get("bvh_path", ""))).expanduser()
        return "bvh23" if bvh.is_file() else ""
    return "smp24" if has_smpl else "bvh23"


def render_frame(positions: np.ndarray, edges: list[tuple[int, int]], frame_index: int, total: int, title: str) -> np.ndarray:
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
    ax.set_title(f"{title}  frame {frame_index}/{total - 1}")

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def _write_outputs(output_dir: Path, positions: np.ndarray, edges: list, frame_ids: np.ndarray,
                   args: argparse.Namespace, raw_fps: float, total_frames: int, title: str) -> None:
    stride = max(1, int(args.stride))
    npz_path = output_dir / "skeleton_zup.npz"
    gen_path = cli_common.media_path(output_dir, "skeleton_zup", args.gen)
    if cli_common.outputs_ready([npz_path, gen_path]) and not args.force:
        print(f"[skip] outputs already exist under {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        positions=positions,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        parents=np.asarray([parent for parent, _ in edges], dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        raw_fps=np.asarray([raw_fps], dtype=np.float64),
        trimmed_frame_count=np.asarray([total_frames], dtype=np.int64),
    )
    images = [render_frame(frame, edges, frame_id, total_frames, title) for frame, frame_id in zip(positions, frame_ids)]
    frame_fps = cli_common.viz_fps(args.fps or raw_fps, stride)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gen == "gif":
        cli_common.write_gif(images, gen_path, frame_fps)
    else:
        cli_common.write_mp4(images, gen_path, frame_fps)
    print(f"rendered={len(images)} raw_fps={raw_fps:.6f} trimmed_frames={total_frames}")
    print(f"output: {output_dir}")


def _write_mesh_outputs(output_dir: Path, verts: np.ndarray, faces: np.ndarray, positions: np.ndarray,
                        edges: list, frame_ids: np.ndarray, args: argparse.Namespace,
                        raw_fps: float, total_frames: int, title: str) -> None:
    stride = max(1, int(args.stride))
    npz_path = output_dir / "mesh_zup.npz"
    gen_path = cli_common.media_path(output_dir, "mesh_zup", args.gen)
    if cli_common.outputs_ready([npz_path, gen_path]) and not args.force:
        print(f"[skip] outputs already exist under {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int64),
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        raw_fps=np.asarray([raw_fps], dtype=np.float64),
        trimmed_frame_count=np.asarray([total_frames], dtype=np.int64),
    )
    images = [
        render_mesh_frame(v, faces, joints=j, edges=edges, title=title, frame_index=i, total=total_frames)
        for v, j, i in zip(verts, positions, frame_ids)
    ]
    frame_fps = cli_common.viz_fps(args.fps or raw_fps, stride)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gen == "gif":
        cli_common.write_gif(images, gen_path, frame_fps)
    else:
        cli_common.write_mp4(images, gen_path, frame_fps)
    print(f"rendered={len(images)} raw_fps={raw_fps:.6f} trimmed_frames={total_frames}")
    print(f"output: {output_dir}")


def convert_session(seq_dir: Path, output_root: Path, sample_id: str, source: str, args: argparse.Namespace) -> None:
    """SMPL-24 (via load_session_gt) or BVH-23 (via aligner) session render."""
    stride = max(1, int(args.stride))
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    if source == "smp24":
        fps = 40.0
        loaded = load_session_gt(seq_dir, n_frames, fps)
        positions_all, parents = loaded["joints"], loaded["parents"]
        title = "GT SMPL-24"
    else:
        bvh_path = Path(cli_common.resolve_path(meta["bvh_path"])).expanduser()
        parsed = parse_bvh_aligner(bvh_path)
        fps = parsed["fps"]
        positions_all, parents = parsed["joints"], parsed["parents"]
        title = "GT BVH-23"
    frame_ids = list(range(0, len(positions_all), stride))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError("motion contains no frames after trimming")
    positions = np.stack([positions_all[i] for i in frame_ids], axis=0)
    edges = [(int(parent), int(child)) for child, parent in enumerate(parents) if parent >= 0]
    if source == "smp24" and args.mesh:
        mesh = mesh_from_archive(Path(loaded["source_path"]), query_t=loaded["t_mocap"])
        verts_display = smpl_yup_to_display(mesh["verts"])
        _write_mesh_outputs(
            output_root / sample_id / source, verts_display[np.asarray(frame_ids)], mesh["faces"],
            positions, edges, np.asarray(frame_ids), args, fps, len(positions_all), f"{title} mesh",
        )
        return
    if args.mesh:
        print(f"[warn] {sample_id}: {source} has no SMPL params; --mesh falls back to skeleton-only")
    _write_outputs(output_root / sample_id / source, positions, edges,
                   np.asarray(frame_ids), args, fps, len(positions_all), title)


def convert_one(bvh_path: Path, output_root: Path, sample_id: str, args: argparse.Namespace) -> None:
    """Single-file BVH render (--bvh mode), protocol forced to bvh23."""
    if args.mesh:
        print("[warn] single-BVH mode has no SMPL params; --mesh falls back to skeleton-only")
    stride = max(1, int(args.stride))
    parsed = parse_bvh_aligner(bvh_path)
    frame_ids = list(range(0, len(parsed["joints"]), stride))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError("BVH contains no motion frames after trimming")
    positions = np.stack([parsed["joints"][i] for i in frame_ids], axis=0)
    edges = [
        (int(parent), int(child))
        for child, parent in enumerate(parsed["parents"])
        if parent >= 0
    ]
    _write_outputs(output_root / sample_id / "bvh23", positions, edges,
                   np.asarray(frame_ids), args, parsed["fps"], len(parsed["joints"]), "GT BVH-23")
    print(f"BVH: {bvh_path}")


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
    samples = discover_test_sessions(session_ids, seq_root)
    print(f"Found {len(samples)} test sessions from {split_csv} (protocol={args.protocol})")
    for sample_id, seq_dir in samples:
        source = session_source(seq_dir, args.protocol)
        if not source:
            print(f"[skip] {sample_id}: no {args.protocol} GT source")
            continue
        convert_session(seq_dir, output_root, sample_id, source, args)


if __name__ == "__main__":
    main()
