"""Render AnySole tactile input, predicted motion, and GT as one animation.

Predictions are read from standard SMPL NPZ files written by ``anysole.eval``;
legacy BVH files are detected automatically for Step2Motion results. Panel rendering uses the shared
``render_common`` helpers so every Test1 output keeps the same visual style.

Outputs are written below ``results_display/Test1_visualization/AnySole/<modal>/<config>``
(or the ``ANYSOLE_RESULTSDISPLAY`` override).

Usage (run from the repository root):
    python results_display/script/r_test1_visualize_anysole.py
    python results_display/script/r_test1_visualize_anysole.py --modal anysolev1 --config-id VT2M
    python results_display/script/r_test1_visualize_anysole.py --session S7013
    python results_display/script/r_test1_visualize_anysole.py --modal anysolev1 --contact-method bvh_soft,joint_and
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

from utils import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402
from utils.motion_io import load_motion, load_session_gt  # noqa: E402
from utils.render_common import (  # noqa: E402
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
    loaded = load_session_gt(seq_dir, n_frames, fps)
    return loaded["joints"], loaded["parents"]


def load_pred(pred_path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    loaded = load_motion(pred_path)
    return loaded["joints"], parents_to_edges(loaded["parents"])


def render_session(seq_dir: Path, pred_path: Path, session_id: str, config_id: str, args: argparse.Namespace, session_out: Path):
    gen_path = cli_common.media_path(session_out, f"{session_id}_{config_id}_compare", args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}_{config_id}: already exists under {session_out}")
        return "skip"

    pressure, fake = load_pressure(seq_dir)
    pred, pred_edges = load_pred(pred_path)
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
    parser = argparse.ArgumentParser(description="Visualize AnySole motion vs tactile input and SMPL GT.")
    cli_common.add_common_args(parser, seq_root=True, config_id=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test1_visualization" / "AnySole")
    parser.add_argument(
        "--modal",
        type=str,
        default="auto",
        help="Modal names, comma-separated; 'auto' (default) scans every dir under results/AnySole/. "
        "Model dir is <modal>_<contact-method> (e.g. anysolev1_bvh_soft).",
    )
    parser.add_argument("--contact-method", type=str, default="tactile_abs", help="Contact-label scheme(s), comma-separated; ignored when --modal is 'auto'.")
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

    if args.modal == "auto":
        # Archived BVH-era runs live under *_backup dirs; keep them out of
        # the automatic sweep.
        model_dirs = sorted(
            p.name for p in PRED_ROOT.iterdir()
            if p.is_dir() and "_backup" not in p.name
        ) if PRED_ROOT.is_dir() else []
        if not model_dirs:
            raise SystemExit("No model dirs under %s" % PRED_ROOT)
        log.info(f"Models from {PRED_ROOT} ({len(model_dirs)}): {model_dirs}")
    else:
        model_dirs = [
            cli_common.anysole_model_dir(modal, contact_method)
            for modal in cli_common.split_csv_arg(args.modal)
            for contact_method in cli_common.split_csv_arg(args.contact_method)
        ]

    for model_dir in model_dirs:
        for config_id in cli_common.split_csv_arg(args.config_id):
            pred_root = PRED_ROOT / model_dir / "predictions"
            if not pred_root.is_dir():
                log.warning(f"No prediction directory for {model_dir}: {pred_root}")
                continue
            session_out = out_dir / model_dir / config_id
            for session_id in session_ids:
                candidates = [
                    pred_root / "eval_motion" / f"{session_id}_{config_id}.npz",
                    # Legacy BVH output remains readable for old experiments;
                    # new AnySole eval writes standard SMPL NPZ instead.
                    pred_root / "eval_bvh" / f"{session_id}_{config_id}.bvh",
                ]
                pred_path = next((path for path in candidates if path.is_file()), None)
                if pred_path is None:
                    log.warning(f"Skip {model_dir}/{config_id}/{session_id}: no SMPL/BVH motion file")
                    continue
                try:
                    seq_dir = session_dir(seq_root, session_id)
                except FileNotFoundError as exc:
                    log.warning(str(exc))
                    continue
                render_session(seq_dir, pred_path, session_id, config_id, args, session_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
