"""Render AnySole tactile input, predicted motion, and GT as one animation.

Predictions are read from standard SMPL NPZ files written by ``anysole.eval``;
legacy BVH files are detected automatically for Step2Motion results. Panel rendering uses the shared
``render_common`` helpers so every Test1 output keeps the same visual style.

Outputs are written below ``results_display/r_test1_visualize/AnySole/<modal>/<config>``
(or the ``ANYSOLE_RESULTSDISPLAY`` override).

``--mesh`` replaces the Predicted/GT skeleton panels with full SMPL surface
renders (mesh + skeleton overlay via ``utils.smpl_mesh``); panels whose
motion file has no SMPL params (legacy BVH) keep the classic skeleton panel.
The default (and ``--only-bone``) renders skeleton panels only.

Usage (run from the repository root):
    python results_display/script/r_test1_visualize_anysole.py
    python results_display/script/r_test1_visualize_anysole.py --modal anysolev1 --config-id VT2M
    python results_display/script/r_test1_visualize_anysole.py --session S7013
    python results_display/script/r_test1_visualize_anysole.py --modal anysolev1 --contact-method bvh_soft,joint_and
    python results_display/script/r_test1_visualize_anysole.py --session S7013 --mesh --max-frames 40
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
from utils.motion_io import load_motion, load_session_gt, smpl_yup_to_display  # noqa: E402
from utils.render_common import (  # noqa: E402
    CANVAS_H,
    LEFT_FOOT_BOX,
    PANEL_W,
    RIGHT_FOOT_BOX,
    crop_foot,
    parents_to_edges,
    render_foot_panel,
    render_skeleton_panel,
    session_dir,
)
from utils.smpl_mesh import mesh_from_archive, render_mesh_frame, smpl_faces  # noqa: E402

PRED_ROOT = cli_common.RESULTS_ROOT / "AnySole"


def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake_path = seq_dir / "fake_mask.npy"
    if fake_path.is_file():
        fake = np.load(fake_path).astype(np.uint8).reshape(-1)
    else:
        fake = np.zeros((pressure.shape[0],), dtype=np.uint8)
    return pressure, fake


def load_gt(seq_dir: Path, n_frames: int, fps: float = 40.0) -> dict:
    return load_session_gt(seq_dir, n_frames, fps)


def load_pred(pred_path: Path) -> dict:
    return load_motion(pred_path)


def render_mesh_panel(verts: np.ndarray, faces: np.ndarray, joints: np.ndarray,
                      edges: list, title: str, color: tuple, frame_idx: int,
                      n_frames: int) -> np.ndarray:
    """Full-SMPL panel at the classic PANEL_W x CANVAS_H size (mesh + skeleton)."""
    return render_mesh_frame(
        verts, faces, joints=joints, edges=edges, title=title,
        frame_index=frame_idx, total=n_frames,
        mesh_color=tuple(c / 255.0 for c in color),
        bone_color="#C46A4A", joint_color="#{:02x}{:02x}{:02x}".format(*color),
        title_color="#{:02x}{:02x}{:02x}".format(*color),
        bg_color="#0A0C10",
        size=(PANEL_W, CANVAS_H),
    )


def render_session(seq_dir: Path, pred_path: Path, session_id: str, config_id: str, args: argparse.Namespace, session_out: Path):
    stem = f"{session_id}_{config_id}_compare" + ("_mesh" if args.mesh else "")
    gen_path = cli_common.media_path(session_out, stem, args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}_{config_id}: already exists under {session_out}")
        return "skip"

    pressure, fake = load_pressure(seq_dir)
    pred_loaded = load_pred(pred_path)
    pred, pred_edges = pred_loaded["joints"], parents_to_edges(pred_loaded["parents"])
    n = min(pressure.shape[0], pred.shape[0], fake.shape[0])
    if n < pred.shape[0]:
        log.warning(f"{session_id}_{config_id}: pred has {pred.shape[0]} frames, rendering first {n}")
    gt_loaded = load_gt(seq_dir, n, args.fps)
    gt, gt_edges = gt_loaded["joints"], parents_to_edges(gt_loaded["parents"])

    # Full-surface panels where SMPL params exist; legacy BVH keeps skeletons.
    pred_verts = gt_verts = None
    faces = smpl_faces() if args.mesh else None
    if args.mesh:
        if pred_loaded["format"] == "smpl":
            pred_verts = smpl_yup_to_display(mesh_from_archive(pred_path)["verts"])
        else:
            log.warning(f"{session_id}_{config_id}: pred is {pred_loaded['format']}; mesh unavailable, keeping skeleton panel")
        if gt_loaded["format"] == "smpl":
            gt_verts = smpl_yup_to_display(
                mesh_from_archive(Path(gt_loaded["source_path"]), query_t=gt_loaded["t_mocap"])["verts"]
            )
        else:
            log.warning(f"{session_id}_{config_id}: GT is {gt_loaded['format']}; mesh unavailable, keeping skeleton panel")
        log.info(f"mesh mode: up to 2 surface panels per frame; use --stride/--max-frames to trim render time")

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    frames = []
    for t in frame_ids:
        valid = not bool(fake[t])
        left = render_foot_panel(
            crop_foot(pressure[t], LEFT_FOOT_BOX),
            crop_foot(pressure[t], RIGHT_FOOT_BOX),
            session_id, t, n, args.fps, valid,
        )
        if pred_verts is not None:
            pred_panel = render_mesh_panel(pred_verts[t], faces, pred[t], pred_edges, "Predicted", (80, 200, 255), t, n)
        else:
            pred_panel = render_skeleton_panel(pred[t], "Predicted", (80, 200, 255), t, n, edges=pred_edges)
        if gt_verts is not None:
            gt_panel = render_mesh_panel(gt_verts[t], faces, gt[t], gt_edges, "GT Motion", (255, 170, 80), t, n)
        else:
            gt_panel = render_skeleton_panel(gt[t], "GT Motion", (255, 170, 80), t, n, edges=gt_edges)
        frames.append(np.concatenate([left, pred_panel, gt_panel], axis=1))
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
    cli_common.add_common_args(parser, seq_root=True, config_id=True, out_dir_default=cli_common.DISPLAY_ROOT / "result/r_test1_visualize" / "AnySole")
    parser.add_argument(
        "--modal",
        type=str,
        default="auto",
        help="Modal names, comma-separated; 'auto' (default) scans every dir under results/AnySole/. "
        "Model dir is <modal>_<contact-method> (e.g. anysolev1_bvh_soft).",
    )
    parser.add_argument("--contact-method", type=str, default="tactile_abs", help="Contact-label scheme(s), comma-separated; ignored when --modal is 'auto'.")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Stacked hyperparameter subdir under the model dir (e.g. tw40 / tw40_st20); "
        "model+contact+variant == the ckpt address (2026-09-22 stacked naming). "
        "Ignored when --modal is 'auto'.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render full SMPL surface panels (mesh + skeleton overlay) for Predicted/GT instead of skeleton-only. "
        "Motion files without SMPL params (legacy BVH) keep the classic skeleton panel.",
    )
    parser.add_argument(
        "--only-bone",
        action="store_true",
        help="Explicit skeleton-only panels (the default); conflicts with --mesh.",
    )
    args = parser.parse_args()
    if args.mesh and args.only_bone:
        parser.error("--mesh and --only-bone are mutually exclusive")
    return args


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
            cli_common.anysole_model_dir(modal, contact_method, args.variant)
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
