"""Render AnySole tactile input, predicted BVH, and GT as one animation (Test1).

Predictions are read from the eval BVH files already written by
``anysole.eval`` (``results/AnySole/<modal>/predictions/eval_bvh/<session>_<config>.bvh``);
no model inference happens here.  Panel rendering reuses the MotionPRO
renderer helpers so every Test1 output shares the same visual style.

Outputs are written below ``results_display/Test1_visualization/AnySole/<modal>/<config>``
(or the ``ANYSOLE_RESULTSDISPLAY`` override).

Usage (run from the repository root):
    python results_display/script/visualize_anysole.py
    python results_display/script/visualize_anysole.py --modal anysolev1 --config-id VT2M
    python results_display/script/visualize_anysole.py --session S7013
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402
from bvh_aligner_pose import parse_bvh_aligner  # noqa: E402
from visualize_motionpro import (  # noqa: E402
    LEFT_FOOT_BOX,
    RIGHT_FOOT_BOX,
    compose_frame,
    crop_foot,
    interp_joints,
    joints_to_meters,
    parents_to_edges,
    session_dir,
)

PRED_ROOT = cli_common.RESULTS_ROOT / "AnySole"


def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake_path = seq_dir / "fake_mask.npy"
    if fake_path.is_file():
        fake = np.load(fake_path).astype(np.uint8).reshape(-1)
    else:
        fake = np.zeros((pressure.shape[0],), dtype=np.uint8)
    return pressure, fake


def load_gt(seq_dir: Path, n_frames: int, fps: float = 40.0):
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    bvh_path = Path(cli_common.resolve_path(meta["bvh_path"]))
    # AnySole BVHs are already aligned to visual_start_s; the default 0.4s
    # trim would shift the skeleton panel past the pressure timeline.
    parsed = parse_bvh_aligner(bvh_path, trim_leading_seconds=0.0)
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    joints = interp_joints(parsed["joints"], parsed["frame_time"], t_mocap)
    return joints_to_meters(joints), parsed["parents"]


def load_pred(pred_bvh: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    parsed = parse_bvh_aligner(pred_bvh, trim_leading_seconds=0.0)
    return joints_to_meters(parsed["joints"]), parents_to_edges(parsed["parents"])


def render_session(seq_dir: Path, pred_bvh: Path, session_id: str, config_id: str, args: argparse.Namespace, session_out: Path):
    gen_path = cli_common.media_path(session_out, f"{session_id}_{config_id}_compare", args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}_{config_id}: already exists under {session_out}")
        return "skip"

    pressure, fake = load_pressure(seq_dir)
    pred, pred_edges = load_pred(pred_bvh)
    n = min(pressure.shape[0], pred.shape[0], fake.shape[0])
    if n < pred.shape[0]:
        log.warning(f"{session_id}_{config_id}: pred has {pred.shape[0]} frames, rendering first {n}")
    gt, gt_parents = load_gt(seq_dir, n, args.fps)
    gt_edges = parents_to_edges(gt_parents)

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    frames = []
    for t in frame_ids:
        valid = not bool(fake[t])
        frames.append(
            compose_frame(
                crop_foot(pressure[t], LEFT_FOOT_BOX),
                crop_foot(pressure[t], RIGHT_FOOT_BOX),
                pred[t],
                gt[t],
                gt_edges,
                session_id,
                t,
                n,
                args.fps,
                valid,
                pred_edges=pred_edges,
            )
        )
    if not frames:
        raise RuntimeError(f"No frames rendered for {session_id}_{config_id}")

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")
    log.info(f"{session_id}_{config_id}: rendered={len(frames)} frames gt_joints={gt.shape[1]}")
    return "write"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AnySole eval BVH vs tactile input and GT BVH.")
    cli_common.add_common_args(parser, seq_root=True, modal=True, config_id=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test1_visualization" / "AnySole")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orig_cwd = os.getcwd()
    seq_root = Path(cli_common.resolve_path(args.seq_root, orig_cwd))
    split_csv = Path(cli_common.resolve_path(args.split_csv, orig_cwd))
    out_dir = Path(cli_common.resolve_path(args.out_dir, orig_cwd))
    out_dir.mkdir(parents=True, exist_ok=True)

    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    if args.session:
        log.info(f"Sessions from --session: {session_ids}")
    else:
        log.info(f"Sessions from {args.split} split ({len(session_ids)}): {session_ids}")

    for modal in cli_common.split_csv_arg(args.modal):
        for config_id in cli_common.split_csv_arg(args.config_id):
            pred_root = PRED_ROOT / modal / "predictions" / "eval_bvh"
            if not pred_root.is_dir():
                log.warning(f"No prediction directory for {modal}: {pred_root}")
                continue
            session_out = out_dir / modal / config_id
            for session_id in session_ids:
                pred_bvh = pred_root / f"{session_id}_{config_id}.bvh"
                if not pred_bvh.is_file():
                    log.warning(f"Skip {modal}/{config_id}/{session_id}: no {pred_bvh.name}")
                    continue
                try:
                    seq_dir = session_dir(seq_root, session_id)
                except FileNotFoundError as exc:
                    log.warning(str(exc))
                    continue
                render_session(seq_dir, pred_bvh, session_id, config_id, args, session_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
