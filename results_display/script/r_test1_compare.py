"""R_Test1 comparison view: one row of panels per generation mode.

评估整改任务 03：删除 common19 裁剪与公共骨架边。每个模型画自己的原生
骨架（SMPL-24 用 SMPL 树、BVH-23 用解析出的 Skeleton3 层级），逐帧 MPJPE
只在模型与其**配对同协议 GT** 之间计算。混合协议的一行不再用一个
``GT Motion`` 面板代表所有模型：每个出现过的协议各给一栏 GT
（``GT Motion — SMPL-24 native`` / ``GT Motion — BVH-23 native``）。

Row layout is mode-locked by ``models_modes.yaml`` (rows of different modes
never share one animation):

    VT2M: [Tactile] [AnySole VT2M] [MotionPRO] [MMVP pressure_toolkit] [GT SMPL-24]
    V2M:  [Video]   [AnySole V2M]  [MMVP_VP-MoCap]                   [GT SMPL-24]
    T2M:  [Tactile] [AnySole T2M]  [Step2Motion]         [GT SMPL-24] [GT BVH-23]

The input panel follows the mode's real input: tactile heatmap for
VT2M/T2M, the actual RGB camera frame for V2M.  Every model panel title
carries its ``SMPL-24 native`` / ``BVH-23 native`` protocol marker.  A model
without an exported prediction gets a grey placeholder panel naming the
missing path (never silently skipped, never substituted by another model's
GT).

Outputs land in ``results_display/ResultTest/R1Test_visualize/compare/<mode>/``
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
    GT_COLOR,
    MAIN_COLOR,
    MISSING_COLOR,
    MOTION_MODES as MODES,
    PROTOCOL_LABELS,
    array_from_file,
    bvh_joints,
    find_prediction,
    load_mode_registry,
    native_edges,
    native_frame_mpjpe_mm,
    normalize_protocol,
    protocol_gt,
    read_manifest,
)
from utils.motion_io import load_session_gt  # noqa: E402
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


def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(seq_dir / "pressure_48.npz", allow_pickle=True) as data:
        pressure = np.stack((data["left48"], data["right48"]), axis=1).reshape(-1, 2, 4, 12).astype(np.float32)
        fake = np.asarray(data["fake"], dtype=np.uint8).reshape(-1)
    return pressure, fake


def video_frames(session_id: str) -> list[Path]:
    """RGB camera frames for the V2M input panel (0..N-1 aligned to the grid)."""
    images = cli_common.WORKSPACE_ROOT / "shared/facts/sessions/cam3"
    for d in sorted(images.glob(f"*/*/{session_id}")):
        color = d / "rgb"
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


def anysole_prediction_roots(model_dir: Path, variant: str | None) -> list[tuple[str, Path]]:
    """AnySole run subdirs (stacked-hyperparameter naming) or the flat layout."""
    if variant:
        return [(variant, model_dir / variant)]
    if not model_dir.is_dir():
        return []
    runs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and ((p / "checkpoints").is_dir() or (p / "predictions").is_dir())
    )
    return [(p.name, p) for p in runs] or [("", model_dir)]


def load_anysole_pred(model_dir: Path, run_name: str, session_id: str, config_id: str, row: dict):
    """Load the main model's native SMPL-24 joints (+ names + protocol); None when missing."""
    pred_root = (model_dir / run_name if run_name else model_dir) / "predictions"
    for cand in (
        pred_root / "eval_motion" / f"{session_id}_{config_id}.npz",
        pred_root / "eval_bvh" / f"{session_id}_{config_id}.bvh",
    ):
        if cand.is_file():
            try:
                joints, mask, names, protocol = array_from_file(cand)
            except Exception as exc:
                return None, f"{cand}: {exc}"
            if mask is None:
                return None, f"legacy motion archive without valid_mask: {cand}"
            return {"joints": joints, "names": names, "protocol": protocol}, str(cand)
    return None, str(pred_root / "eval_motion" / f"{session_id}_{config_id}.npz")


def load_baseline_pred(entry: dict, session_id: str, row: dict):
    """Load one registry baseline's native joints; None when missing."""
    root = cli_common.resolve_path(entry.get("prediction_root", ""))
    path = find_prediction(root, session_id, entry.get("pattern") or None)
    if path is None:
        return None, f"{root}/{session_id}"
    try:
        protocol = normalize_protocol(str(entry.get("protocol") or ""))
        if path.suffix.lower() == ".bvh":
            # A prediction BVH owns its own frame axis.  Do not query it with
            # the GT manifest's n_frames/offset; pairing is handled by the
            # explicit prediction length and frame axis downstream.
            joints, names, _parents = bvh_joints(path)
            return {"joints": joints, "names": names,
                    "protocol": PROTOCOL_LABELS[protocol]}, str(path)
        joints, mask, names, protocol_label = array_from_file(path)
        return {"joints": joints, "names": names, "protocol": protocol_label}, str(path)
    except Exception as exc:
        return None, f"{path}: {exc}"


def render_session(mode: str, session_id: str, model_dir: Path, run_name: str,
                   baselines: list[dict], args: argparse.Namespace, out_mode: Path,
                   seq_root: Path, manifest_rows: dict[str, dict]):
    stem = f"{session_id}_{mode}_{run_name}_compare" if run_name else f"{session_id}_{mode}_compare"
    gen_path = cli_common.media_path(out_mode, stem, args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}_{mode}: already exists")
        return "skip"

    row = manifest_rows[session_id]
    seq_dir = session_dir(seq_root, session_id)
    pressure, fake = load_pressure(seq_dir)
    n_ref = pressure.shape[0]

    # Main model: native SMPL-24 with its paired SMPL GT.
    main, main_path = load_anysole_pred(model_dir, run_name, session_id, mode, row)
    models: list[dict] = []
    if main is not None:
        main_protocol = normalize_protocol(main["protocol"])
        gt_joints, gt_names = protocol_gt(row, main_protocol)
        models.append({
            "label": f"AnySole {mode}", "color": MAIN_COLOR,
            "joints": main["joints"], "names": main["names"], "protocol": main_protocol,
            "edges": native_edges(main["names"]),
            "gt": gt_joints, "gt_names": gt_names,
            "gt_edges": native_edges(gt_names),
            "reason": main_path,
        })
    else:
        models.append({
            "label": f"AnySole {mode}", "color": MAIN_COLOR,
            "joints": None, "names": (), "protocol": "smpl24",
            "edges": [], "gt": None, "gt_names": (), "gt_edges": [],
            "reason": main_path,
        })
    for entry in baselines:
        color = hex_to_rgb(entry.get("color", "#787880"))
        label = str(entry.get("display_name") or entry.get("name") or "")
        pred, path = load_baseline_pred(entry, session_id, row)
        if pred is None:
            models.append({
                "label": label, "color": color,
                "joints": None, "names": (), "protocol": "",
                "edges": [], "gt": None, "gt_names": (), "gt_edges": [],
                "reason": path,
            })
            continue
        protocol = normalize_protocol(pred["protocol"])
        gt_joints, gt_names = protocol_gt(row, protocol)
        models.append({
            "label": label, "color": color,
            "joints": pred["joints"], "names": pred["names"], "protocol": protocol,
            "edges": native_edges(pred["names"]),
            "gt": gt_joints, "gt_names": gt_names,
            "gt_edges": native_edges(gt_names),
            "reason": path,
        })

    lengths = [n_ref, fake.shape[0]]
    for m in models:
        if m["joints"] is not None:
            lengths.append(m["joints"].shape[0])
        if m["gt"] is not None:
            lengths.append(m["gt"].shape[0])
    n = min(lengths)
    frames_path = video_frames(session_id)

    # One GT column per protocol present in this row (never a single shared
    # GT panel standing for mixed SMPL/BVH predictions).
    gt_columns: list[dict] = []
    seen = set()
    for m in models:
        if m["gt"] is not None and m["protocol"] and m["protocol"] not in seen:
            seen.add(m["protocol"])
            gt_columns.append({
                "label": f"GT Motion — {PROTOCOL_LABELS[m['protocol']]}",
                "joints": m["gt"][:n], "edges": m["gt_edges"],
            })

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
                np.rot90(pressure[t, 0], k=1), np.rot90(pressure[t, 1], k=1),
                session_id, t, n, args.fps, not bool(fake[t]),
            )
        panels = [input_panel]
        for m in models:
            if m["joints"] is None:
                panels.append(render_missing_panel(m["label"], m["reason"]))
                continue
            marker = PROTOCOL_LABELS.get(m["protocol"], m["protocol"])
            mpjpe = native_frame_mpjpe_mm(m["joints"][:n], m["gt"][:n], protocol=m["protocol"])
            panels.append(render_skeleton_panel(
                m["joints"][t], f"{m['label']} — {marker}", m["color"], t, n,
                edges=m["edges"], mpjpe_mm=float(mpjpe[t]),
            ))
        for gt_col in gt_columns:
            panels.append(render_skeleton_panel(
                gt_col["joints"][t], gt_col["label"], GT_COLOR, t, n,
                edges=gt_col["edges"],
            ))
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
    protocols = ", ".join(m["protocol"] for m in models if m["joints"] is not None)
    log.info(f"{session_id}_{mode}: row_width={out_frames[0].shape[1]}px frames={len(out_frames)} protocols=[{protocols}]")
    return "write"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Side-by-side main-vs-baseline comparison animation, one row per mode.")
    cli_common.add_common_args(parser, seq_root=True, gen=True, fps=True, stride=True, max_frames=True, force=True,
                               out_dir_default=cli_common.DISPLAY_ROOT / "ResultTest/R1Test_visualize" / "compare")
    parser.add_argument("--mode", default="all", help="Generation mode(s): VT2M,V2M,T2M or 'all'.")
    parser.add_argument("--model-name", "--modal", dest="modal", metavar="MODEL_NAME", default="V4B", help="AnySole model name(s), comma-separated; legacy alias: --modal.")
    parser.add_argument("--contact-method", default="joint_and", help="Contact-label scheme(s), comma-separated.")
    parser.add_argument("--variant", default=None, help="AnySole stacked-hyperparameter run subdir (omit to scan all runs).")
    parser.add_argument("--modes-config", type=Path, default=SCRIPT_DIR.parent / "models_modes.yaml", help="Mode registry.")
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
    session_ids = cli_common.load_evaluable_sessions(args.session, split_csv, args.split)
    log.info(f"Sessions ({len(session_ids)}): {session_ids}")
    log.info(f"Modes: {modes}")
    # Paired native GT per session comes from the canonical manifest row.
    manifest_rows = {
        row["session_id"]: row
        for row in read_manifest(cli_common.DEFAULT_MANIFEST, args.split,
                                 split_csv=split_csv, evaluable_only=True)
    }
    missing_manifest = [sid for sid in session_ids if sid not in manifest_rows]
    if missing_manifest:
        log.warning(f"Sessions without manifest row (skipped): {missing_manifest}")

    model_dirs = [
        cli_common.anysole_model_dir(modal, contact_method)
        for modal in cli_common.split_csv_arg(args.modal)
        for contact_method in cli_common.split_csv_arg(args.contact_method)
    ]
    for model_dir in model_dirs:
        if not (PRED_ROOT / model_dir).is_dir():
            log.warning(f"No model directory for {model_dir}: {PRED_ROOT / model_dir}")
    for mode in modes:
        baselines = list(registry.get(mode, []))
        log.info(f"mode {mode}: main=AnySole {mode}, baselines={[b.get('name') for b in baselines]}")
        out_mode = out_dir / mode
        for session_id in session_ids:
            if session_id not in manifest_rows:
                continue
            for model_dir in model_dirs:
                for run_name, run_dir in anysole_prediction_roots(PRED_ROOT / model_dir, args.variant):
                    try:
                        render_session(mode, session_id, PRED_ROOT / model_dir, run_name,
                                       baselines, args, out_mode, seq_root, manifest_rows)
                    except FileNotFoundError as exc:
                        log.warning(f"{session_id}_{mode}: {exc}")
                        continue
                    except Exception as exc:  # one bad session must not stop the sweep
                        log.error(f"{session_id}_{mode}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
