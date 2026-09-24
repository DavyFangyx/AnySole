"""R_Test1 comparison view: one row of panels per generation mode.

Row layout is mode-locked by ``models_modes.yaml`` (rows of different modes
never share one animation):

    VT2M: [Tactile] [AnySole VT2M] [MotionPRO] [MMVP_pressure_toolkit] [GT]
    V2M:  [Video]   [AnySole V2M]  [MMVP_FPP-Net]                    [GT]
    T2M:  [Tactile] [AnySole T2M]  [Step2Motion]                     [GT]

The input panel follows the mode's real input: tactile heatmap for
VT2M/T2M, the actual RGB camera frame for V2M.  Every skeleton panel draws
the same common19 semantic joint set with the shared ``render_common``
renderer, so poses are directly comparable at a glance; each model panel's
footer shows its own per-frame MPJPE.  A model without an exported
prediction gets a grey placeholder panel naming the missing path (never
silently skipped).

Outputs land in ``results_display/result/r_test1_visualize/compare/<mode>/``
and do not touch the per-model visualization directories.

Usage (run from the repository root):
    python results_display/script/r_test1_compare.py --mode VT2M --session S13013 --gen gif
    python results_display/script/r_test1_compare.py --mode V2M --split test
    python results_display/script/r_test1_compare.py --split test   # all modes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402
from utils.compare_core import (  # noqa: E402
    COMMON_JOINTS,
    GT_COLOR,
    MAIN_COLOR,
    MISSING_COLOR,
    MODES,
    array_from_file,
    common_edges_for,
    frame_mpjpe_mm,
    load_mode_registry,
    select_common_joints,
)
from utils.motion_io import load_motion, load_session_gt  # noqa: E402
from utils.render_common import (  # noqa: E402
    CANVAS_H,
    FOOT_W,
    INFO_FONT,
    LEFT_FOOT_BOX,
    RIGHT_FOOT_BOX,
    TITLE_FONT,
    crop_foot,
    draw_text,
    render_foot_panel,
    render_skeleton_panel,
    session_dir,
)

PRED_ROOT = cli_common.RESULTS_ROOT / "AnySole"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = str(value or "").strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"bad color: {value!r}")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def protocol_of(fmt: str) -> str:
    return {"smpl": "smpl24", "bvh": "bvh23"}.get(fmt, fmt)


def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake_path = seq_dir / "fake_mask.npy"
    fake = np.load(fake_path).astype(np.uint8).reshape(-1) if fake_path.is_file() else np.zeros((pressure.shape[0],), dtype=np.uint8)
    return pressure, fake


def video_frames(session_id: str) -> list[Path]:
    """RGB camera frames for the V2M input panel (0..N-1 aligned to the grid)."""
    images = cli_common.WORKSPACE_ROOT / "derived/pressure_tookit/images"
    for d in sorted(images.glob(f"*/*/{session_id}")):
        color = d / "color"
        if color.is_dir():
            frames = sorted(p for p in color.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if frames:
                return frames
    return []


def render_video_panel(frame_path: Path, session_id: str, frame_idx: int, n_frames: int, fps: float) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (FOOT_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, FOOT_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (FOOT_W // 2, 12), "Video", TITLE_FONT, fill=(180, 210, 255), anchor="mt")
    img = Image.open(frame_path).convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)  # raw capture is dark; display-only boost
    img.thumbnail((FOOT_W - 24, CANVAS_H - 110))
    canvas.paste(img, ((FOOT_W - img.width) // 2, 84))
    t_s = frame_idx / fps if fps else 0.0
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 18), f"{session_id}  t={t_s:.2f}s  {frame_idx}/{n_frames - 1}", INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_missing_panel(label: str, reason: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (460, CANVAS_H), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 459, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (230, 12), label, TITLE_FONT, fill=MISSING_COLOR, anchor="mt")
    draw_text(draw, (230, CANVAS_H // 2 - 30), "missing", TITLE_FONT, fill=MISSING_COLOR, anchor="mm")
    tail = "/".join(str(reason).split("/")[-2:])
    draw_text(draw, (230, CANVAS_H // 2 + 6), tail, INFO_FONT, fill=MISSING_COLOR, anchor="mm")
    return np.asarray(canvas)


def load_anysole_pred(model_dir: str, session_id: str, config_id: str):
    """Load the main model's common19 joints for one config; None when missing."""
    pred_root = PRED_ROOT / model_dir / "predictions"
    for cand in (
        pred_root / "eval_motion" / f"{session_id}_{config_id}.npz",
        pred_root / "eval_bvh" / f"{session_id}_{config_id}.bvh",
    ):
        if cand.is_file():
            loaded = load_motion(cand)
            protocol = protocol_of(loaded["format"])
            joints = select_common_joints(loaded["joints"], tuple(loaded["names"]), protocol)
            return joints, str(cand)
    return None, str(pred_root / "eval_motion" / f"{session_id}_{config_id}.npz")


def load_baseline_pred(entry: dict, session_id: str):
    """Load one registry baseline's common19 joints; None when missing."""
    root = cli_common.resolve_path(entry.get("prediction_root", ""))
    from utils.compare_core import find_prediction
    path = find_prediction(root, session_id, entry.get("pattern") or None)
    if path is None:
        return None, f"{root}/{session_id}"
    joints, _, names, protocol = array_from_file(path)
    return select_common_joints(joints, names, protocol), str(path)


def render_session(mode: str, session_id: str, model_dir: str, baselines: list[dict], args: argparse.Namespace, out_mode: Path, seq_root: Path):
    stem = f"{session_id}_{mode}_compare"
    gen_path = cli_common.media_path(out_mode, stem, args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}_{mode}: already exists")
        return "skip"

    seq_dir = session_dir(seq_root, session_id)
    pressure, fake = load_pressure(seq_dir)
    n_ref = pressure.shape[0]
    gt_loaded = load_session_gt(seq_dir, n_ref, args.fps)
    gt = select_common_joints(gt_loaded["joints"], tuple(gt_loaded["names"]), protocol_of(gt_loaded["format"]))
    gt_edges = common_edges_for(tuple(COMMON_JOINTS))

    main_joints, main_path = load_anysole_pred(model_dir, session_id, mode)
    models: list[dict] = [{
        "label": f"AnySole {mode}", "color": MAIN_COLOR, "joints": main_joints, "reason": main_path,
    }]
    for entry in baselines:
        joints, path = load_baseline_pred(entry, session_id)
        color = hex_to_rgb(entry.get("color", "#787880"))
        models.append({"label": str(entry.get("name", "")), "color": color, "joints": joints, "reason": path})

    n = min([n_ref, gt.shape[0], fake.shape[0]] + [m["joints"].shape[0] for m in models if m["joints"] is not None])
    frames_path = video_frames(session_id)

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    out_frames = []
    for t in frame_ids:
        if mode == "V2M":
            input_panel = (render_video_panel(frames_path[min(t, len(frames_path) - 1)], session_id, t, n, args.fps)
                           if frames_path else render_missing_panel("Video", "no RGB frames"))
        else:
            input_panel = render_foot_panel(
                crop_foot(pressure[t], LEFT_FOOT_BOX), crop_foot(pressure[t], RIGHT_FOOT_BOX),
                session_id, t, n, args.fps, not bool(fake[t]),
            )
        panels = [input_panel]
        for m in models:
            if m["joints"] is None:
                panels.append(render_missing_panel(m["label"], m["reason"]))
                continue
            mpjpe = frame_mpjpe_mm(m["joints"][:n], gt[:n])
            panels.append(render_skeleton_panel(m["joints"][t], m["label"], m["color"], t, n, edges=gt_edges, mpjpe_mm=float(mpjpe[t])))
        panels.append(render_skeleton_panel(gt[t], "GT Motion", GT_COLOR, t, n, edges=gt_edges))
        out_frames.append(np.concatenate(panels, axis=1))
    if not out_frames:
        raise RuntimeError(f"No frames rendered for {session_id}_{mode}")

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(out_frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(out_frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")
    log.info(f"{session_id}_{mode}: row_width={out_frames[0].shape[1]}px frames={len(out_frames)}")
    return "write"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Side-by-side main-vs-baseline comparison animation, one row per mode.")
    cli_common.add_common_args(parser, seq_root=True, gen=True, fps=True, stride=True, max_frames=True, force=True,
                               out_dir_default=cli_common.DISPLAY_ROOT / "result/r_test1_visualize" / "compare")
    parser.add_argument("--mode", default="all", help="Generation mode(s): VT2M,V2M,T2M or 'all'.")
    parser.add_argument("--model-name", "--modal", dest="modal", metavar="MODEL_NAME", default="V4B", help="AnySole model name(s), comma-separated; legacy alias: --modal.")
    parser.add_argument("--contact-method", default="joint_and", help="Contact-label scheme(s), comma-separated.")
    parser.add_argument("--modes-config", type=Path, default=SCRIPT_DIR / "models_modes.yaml", help="Mode registry.")
    parser.set_defaults(split="val")  # iteration split by default; pass --split test for the formal 36-session set
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    orig_cwd = os.getcwd()
    seq_root = Path(cli_common.resolve_path(args.seq_root, orig_cwd))
    split_csv = Path(cli_common.resolve_path(args.split_csv, orig_cwd))
    out_dir = Path(cli_common.resolve_path(args.out_dir, orig_cwd))
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_mode_registry(args.modes_config)
    modes = list(MODES) if args.mode == "all" else cli_common.split_csv_arg(args.mode)
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise SystemExit(f"Unknown --mode: {unknown}; choices={list(MODES)} or 'all'")
    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    log.info(f"Sessions ({len(session_ids)}): {session_ids}")
    log.info(f"Modes: {modes}")

    model_dirs = [
        cli_common.anysole_model_dir(modal, contact_method)
        for modal in cli_common.split_csv_arg(args.modal)
        for contact_method in cli_common.split_csv_arg(args.contact_method)
    ]
    for model_dir in model_dirs:
        if not (PRED_ROOT / model_dir / "predictions").is_dir():
            log.warning(f"No prediction directory for {model_dir}: {PRED_ROOT / model_dir / 'predictions'}")
    for mode in modes:
        baselines = list(registry.get(mode, []))
        log.info(f"mode {mode}: main=AnySole {mode}, baselines={[b.get('name') for b in baselines]}")
        out_mode = out_dir / mode
        for session_id in session_ids:
            for model_dir in model_dirs:
                try:
                    render_session(mode, session_id, model_dir, baselines, args, out_mode, seq_root)
                except FileNotFoundError as exc:
                    log.warning(f"{session_id}_{mode}: {exc}")
                    continue
                except Exception as exc:  # one bad session must not stop the sweep
                    log.error(f"{session_id}_{mode}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
