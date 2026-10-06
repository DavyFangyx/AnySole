"""R_Test1 姿态可视化统一入口（2026-10-06 用户裁定：选中 + 生成什么）。

单独生成（默认）：``--models`` 选中谁出谁；AnySole 一次出四个系列 mode
（VT2M/V2M/T2M/V2T），基线各出自己注册的那个 mode。mode 不是参数。
每模型输出 = 输入 | 预测 | GT 三栏动画（仅 gif/mp4，无中间帧产物）。

    python results_display/script/r_test1_visualize.py --models anysole,motionpro --session S13011
    python results_display/script/r_test1_visualize.py          # 全部模型

对比（--compare）：同 mode 模型 1×N 横排一行——**输入只放一次**（随 mode：
VT2M/T2M 触觉热力图，V2M/V2T 视频帧），中间是各模型的预测骨架（各自原生
协议 + 逐帧 MPJPE），右侧每个出现过的协议一栏 GT，不重复放输入与 GT。

    python results_display/script/r_test1_visualize.py --compare

产物：

    ResultTest/R1Test_visualize/<model>/<mode>/{gif,mp4}/<session>.{ext}          # 单独生成
    ResultTest/R1Test_visualize/compare/<mode>/{gif,mp4}/<session>_compare.{ext}  # 1×N 横排
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
    MOTION_MODES,
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
from utils.render_common import (  # noqa: E402
    CANVAS_H,
    FOOT_W,
    INFO_FONT,
    TITLE_FONT,
    draw_text,
    render_foot_panel,
    render_skeleton_panel,
    session_dir,
)

PRED_ROOT = cli_common.RESULTS_ROOT / "AnySole"
ANYSOLE_MODES = tuple(MOTION_MODES) + ("V2T",)  # AnySole 四个系列
DEFAULT_MODES_CONFIG = SCRIPT_DIR.parent / "models_modes.yaml"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = str(value or "").strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"bad color: {value!r}")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


# ---- 输入/预测加载（原 r_test1_compare 的实现，唯一实现点） ----

def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(seq_dir / "pressure_48.npz", allow_pickle=True) as data:
        pressure = np.stack((data["left48"], data["right48"]), axis=1).reshape(-1, 2, 4, 12).astype(np.float32)
        fake = np.asarray(data["fake"], dtype=np.uint8).reshape(-1)
    return pressure, fake


def video_frames(session_id: str) -> list[Path]:
    """RGB camera frames for the V2M/V2T input panel (0..N-1 aligned to the grid)."""
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
    """AnySole run subdirs (stacked-hyperparameter naming) or the flat layout.

    Omitted --variant: exactly one run dir -> use it; several -> newest by
    mtime (logged); ``all`` -> every run dir (historical sweep).
    """
    if variant == "all":
        variant = None
    if variant:
        return [(variant, model_dir / variant)]
    if not model_dir.is_dir():
        return []
    runs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and ((p / "checkpoints").is_dir() or (p / "predictions").is_dir())
    )
    if len(runs) > 1:
        newest = max(runs, key=lambda p: p.stat().st_mtime)
        log.warning(f"{model_dir.name}: {len(runs)} run dirs; picking newest: {newest.name} (--variant to choose, 'all' for every run)")
        return [(newest.name, newest)]
    if runs:
        return [(runs[0].name, runs[0])]
    return [("", model_dir)]


def load_anysole_pred(model_dir: Path, run_name: str, session_id: str, mode: str, row: dict):
    """Load the main model's native SMPL-24 joints (+ names + protocol); None when missing."""
    pred_root = (model_dir / run_name if run_name else model_dir) / "predictions"
    for cand in (
        pred_root / "eval_motion" / f"{session_id}_{mode}.npz",
        pred_root / "eval_bvh" / f"{session_id}_{mode}.bvh",
    ):
        if cand.is_file():
            try:
                joints, mask, names, protocol = array_from_file(cand)
            except Exception as exc:
                return None, f"{cand}: {exc}"
            if mask is None:
                return None, f"legacy motion archive without valid_mask: {cand}"
            return {"joints": joints, "names": names, "protocol": protocol}, str(cand)
    return None, str(pred_root / "eval_motion" / f"{session_id}_{mode}.npz")


def load_baseline_pred(entry: dict, session_id: str, row: dict):
    """Load one registry baseline's native joints; None when missing."""
    root = cli_common.resolve_path(entry.get("prediction_root", ""))
    path = find_prediction(root, session_id, entry.get("pattern") or None)
    if path is None:
        return None, f"{root}/{session_id}"
    try:
        protocol = normalize_protocol(str(entry.get("protocol") or ""))
        if path.suffix.lower() == ".bvh":
            joints, names, _parents = bvh_joints(path)
            return {"joints": joints, "names": names,
                    "protocol": PROTOCOL_LABELS[protocol]}, str(path)
        joints, mask, names, protocol_label = array_from_file(path)
        return {"joints": joints, "names": names, "protocol": protocol_label}, str(path)
    except Exception as exc:
        return None, f"{path}: {exc}"


# ---- 模型解析（选中 = 参数） ----

def resolve_models(models_arg: str, modal_csv: str, contact_csv: str,
                   variant: str | None, registry: dict) -> list[dict]:
    """--models csv -> 统一模型条目（anysole = 四 mode，基线 = 注册 mode）。"""
    names = cli_common.split_csv_arg(models_arg)
    want_all = not names or names == ["all"]
    selected = set(n.lower() for n in names if n.lower() != "all")
    models: list[dict] = []
    registry_names = {}
    for mode, entries in registry.items():
        for entry in entries:
            registry_names[str(entry.get("name") or "").lower()] = (mode, entry)

    if want_all or "anysole" in selected:
        for modal in cli_common.split_csv_arg(modal_csv):
            for contact in cli_common.split_csv_arg(contact_csv):
                model_dir = PRED_ROOT / cli_common.anysole_model_dir(modal, contact)
                if not model_dir.is_dir():
                    log.warning(f"No AnySole model dir: {model_dir}")
                    continue
                for run_name, run_dir in anysole_prediction_roots(model_dir, variant):
                    label = model_dir.name if not run_name else f"{model_dir.name}/{run_name}"
                    models.append({
                        "label": label, "kind": "anysole",
                        "model_dir": model_dir, "run_name": run_name,
                        "modes": list(ANYSOLE_MODES), "color": MAIN_COLOR,
                    })
    for name in selected if selected else registry_names:
        if name == "anysole":
            continue
        if name not in registry_names:
            log.warning(f"Unknown --models entry (skipped): {name}")
            continue
        mode, entry = registry_names[name]
        if not dict(entry.get("capabilities") or {}).get("joint_positions"):
            log.warning(f"{entry.get('name')} has no pose output; skipped in R_Test1")
            continue
        models.append({
            "label": str(entry.get("name") or name), "kind": "baseline",
            "entry": entry, "modes": [mode],
            "color": hex_to_rgb(entry.get("color", "#787880")),
        })
    return models


def load_pred(model: dict, mode: str, session_id: str, row: dict):
    if model["kind"] == "anysole":
        return load_anysole_pred(model["model_dir"], model["run_name"], session_id, mode, row)
    return load_baseline_pred(model["entry"], session_id, row)


def models_of_mode(models: list[dict], mode: str) -> list[dict]:
    return [m for m in models if mode in m["modes"]]


# ---- 单独生成：输入 | 预测 | GT 三面板动画（仅 gif/mp4） ----

def render_single(model: dict, mode: str, session_id: str, row: dict,
                  seq_dir: Path, args: argparse.Namespace, out_base: Path) -> str:
    anim_path = cli_common.media_path(out_base / model["label"] / mode, session_id, args.gen)
    if cli_common.outputs_ready([anim_path]) and not args.force:
        return "skip"

    pred, reason = load_pred(model, mode, session_id, row)
    pressure, fake = load_pressure(seq_dir)
    n_ref = pressure.shape[0]

    lengths = [n_ref, fake.shape[0]]
    gt_joints = gt_names = None
    protocol = ""
    if pred is not None:
        protocol = normalize_protocol(pred["protocol"])
        gt_joints, gt_names = protocol_gt(row, protocol)
        lengths.append(pred["joints"].shape[0])
        lengths.append(gt_joints.shape[0])
    n = min(lengths)
    frames_path = video_frames(session_id) if mode in ("V2M", "V2T") else []

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    out_frames = []
    for t in frame_ids:
        if mode in ("V2M", "V2T"):
            input_panel = (render_video_panel(frames_path[min(t, len(frames_path) - 1)], session_id, t, n, args.fps)
                           if frames_path else render_missing_panel("Video", "no RGB frames"))
        else:
            input_panel = render_foot_panel(
                np.rot90(pressure[t, 0], k=1), np.rot90(pressure[t, 1], k=1),
                session_id, t, n, args.fps, not bool(fake[t]),
            )
        panels = [input_panel]
        if pred is None:
            panels.append(render_missing_panel(model["label"], reason))
        else:
            marker = PROTOCOL_LABELS.get(protocol, protocol)
            mpjpe = native_frame_mpjpe_mm(pred["joints"][:n], gt_joints[:n], protocol=protocol)
            panels.append(render_skeleton_panel(
                pred["joints"][t], f"{model['label']} — {marker}", model["color"], t, n,
                edges=native_edges(pred["names"]), mpjpe_mm=float(mpjpe[t]),
            ))
            panels.append(render_skeleton_panel(
                gt_joints[t], f"GT — {marker}", GT_COLOR, t, n,
                edges=native_edges(gt_names),
            ))
        out_frames.append(np.concatenate(panels, axis=1))
    if not out_frames:
        raise RuntimeError(f"No frames rendered for {model['label']}_{mode}_{session_id}")

    anim_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(out_frames, anim_path, frame_fps)
    else:
        cli_common.write_mp4(out_frames, anim_path, frame_fps)
    log.info(f"Wrote {anim_path} ({len(out_frames)} frames)")
    return "write"


# ---- 对比：同 mode 1×N 横排（输入一次 + 各模型预测 + 每协议一栏 GT） ----

def render_compare(mode: str, models: list[dict], session_id: str, row: dict,
                   seq_dir: Path, args: argparse.Namespace, out_base: Path) -> str:
    anim_path = cli_common.media_path(out_base / "compare" / mode, f"{session_id}_{mode}_compare", args.gen)
    if cli_common.outputs_ready([anim_path]) and not args.force:
        return "skip"

    pressure, fake = load_pressure(seq_dir)
    n_ref = pressure.shape[0]
    loaded: list[dict] = []
    for model in models:
        pred, path = load_pred(model, mode, session_id, row)
        if pred is None:
            loaded.append({
                "label": model["label"], "color": model["color"],
                "joints": None, "names": (), "protocol": "",
                "edges": [], "gt": None, "gt_names": (), "gt_edges": [],
                "reason": path,
            })
            continue
        protocol = normalize_protocol(pred["protocol"])
        gt_joints, gt_names = protocol_gt(row, protocol)
        loaded.append({
            "label": model["label"], "color": model["color"],
            "joints": pred["joints"], "names": pred["names"], "protocol": protocol,
            "edges": native_edges(pred["names"]),
            "gt": gt_joints, "gt_names": gt_names,
            "gt_edges": native_edges(gt_names),
            "reason": path,
        })

    lengths = [n_ref, fake.shape[0]]
    for m in loaded:
        if m["joints"] is not None:
            lengths.append(m["joints"].shape[0])
        if m["gt"] is not None:
            lengths.append(m["gt"].shape[0])
    n = min(lengths)
    frames_path = video_frames(session_id) if mode in ("V2M", "V2T") else []

    # One GT column per protocol present in this row (never a single shared
    # GT panel standing for mixed SMPL/BVH predictions).
    gt_columns: list[dict] = []
    seen = set()
    for m in loaded:
        if m["gt"] is not None and m["protocol"] and m["protocol"] not in seen:
            seen.add(m["protocol"])
            gt_columns.append({
                "label": f"GT — {PROTOCOL_LABELS[m['protocol']]}",
                "joints": m["gt"][:n], "edges": m["gt_edges"],
            })

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    out_frames = []
    for t in frame_ids:
        if mode in ("V2M", "V2T"):
            input_panel = (render_video_panel(frames_path[min(t, len(frames_path) - 1)], session_id, t, n, args.fps)
                           if frames_path else render_missing_panel("Video", "no RGB frames"))
        else:
            input_panel = render_foot_panel(
                np.rot90(pressure[t, 0], k=1), np.rot90(pressure[t, 1], k=1),
                session_id, t, n, args.fps, not bool(fake[t]),
            )
        panels = [input_panel]
        for m in loaded:
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
        raise RuntimeError(f"No frames rendered for compare {mode}_{session_id}")

    anim_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(out_frames, anim_path, frame_fps)
    else:
        cli_common.write_mp4(out_frames, anim_path, frame_fps)
    log.info(f"Wrote {anim_path} ({len(out_frames)} frames, {len(loaded)} models)")
    return "write"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test1 pose visualization: --models selection, single or --compare.")
    cli_common.add_common_args(
        parser, seq_root=True, gen=True, fps=True, stride=True, max_frames=True, force=True,
        modal=True, variant=True, contact_method=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "ResultTest/R1Test_visualize",
    )
    parser.add_argument("--models", type=str, default="all",
                        help="Select models: anysole and/or registry baseline names, comma-separated; default all.")
    parser.add_argument("--compare", action="store_true",
                        help="One shared input column + one column per same-mode model + one GT column per protocol.")
    parser.add_argument("--modes-config", type=Path, default=DEFAULT_MODES_CONFIG, help="Mode registry.")
    parser.set_defaults(split="val")  # iteration split by default; pass --split test for the formal 36-session set
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orig_cwd = os.getcwd()
    seq_root = Path(cli_common.resolve_path(args.seq_root, orig_cwd))
    split_csv = Path(cli_common.resolve_path(args.split_csv, orig_cwd))
    out_base = Path(cli_common.resolve_path(args.out_dir, orig_cwd))
    out_base.mkdir(parents=True, exist_ok=True)

    registry = load_mode_registry(args.modes_config)
    models = resolve_models(args.models, args.modal, args.contact_method, args.variant, registry)
    if not models:
        raise SystemExit("No models resolved (check --models/--model-name/--contact-method)")
    for model in models:
        log.info(f"selected: {model['label']} modes={model['modes']}")

    session_ids = cli_common.load_evaluable_sessions(args.session, split_csv, args.split)
    log.info(f"Sessions ({len(session_ids)}): {session_ids}")
    manifest_rows = {
        row["session_id"]: row
        for row in read_manifest(cli_common.DEFAULT_MANIFEST, args.split,
                                 split_csv=split_csv, evaluable_only=True)
    }
    missing_manifest = [sid for sid in session_ids if sid not in manifest_rows]
    if missing_manifest:
        log.warning(f"Sessions without manifest row (skipped): {missing_manifest}")

    for session_id in session_ids:
        if session_id not in manifest_rows:
            continue
        row = manifest_rows[session_id]
        try:
            seq_dir = session_dir(seq_root, session_id)
        except FileNotFoundError as exc:
            log.warning(str(exc))
            continue
        if not args.compare:
            for model in models:
                for mode in model["modes"]:
                    try:
                        render_single(model, mode, session_id, row, seq_dir, args, out_base)
                    except Exception as exc:
                        log.error(f"{model['label']}_{mode}_{session_id}: {exc}")
        else:
            modes = sorted({mode for model in models for mode in model["modes"]})
            for mode in modes:
                in_mode = models_of_mode(models, mode)
                if len(in_mode) < 1:
                    continue
                try:
                    render_compare(mode, in_mode, session_id, row, seq_dir, args, out_base)
                except Exception as exc:
                    log.error(f"compare {mode}/{session_id}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
