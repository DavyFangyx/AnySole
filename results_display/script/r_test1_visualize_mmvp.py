#!/usr/bin/env python3
"""Render a lightweight R_Test1 animation for the two MMVP method lines.

The script consumes the unified ``eval_motion/<session>.npz`` contract.  It
does not run either optimizer and therefore remains usable as soon as a
baseline exporter writes its result, while the missing prediction is reported
explicitly instead of silently falling back to GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.motion_io import load_session_gt  # noqa: E402
from r_test2_compare import array_from_file  # noqa: E402

MANIFEST = ROOT / "AnysoleWorkspace/manifests/session_manifest.jsonl"
DISPLAY = ROOT / "results_display/result/r_test1_visualize"
INFERNO = np.asarray([
    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
    [186, 54, 85], [227, 89, 51], [249, 140, 10], [252, 194, 39],
    [252, 255, 164],
], dtype=np.float32)

EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
    ("pelvis", "neck"), ("neck", "head"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]


def manifest_row(session: str) -> dict:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["session_id"] == session:
                return row
    raise KeyError(session)


def seq_dir(row: dict) -> Path:
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    return ROOT / Path(*parts[:cam + 4])


def prediction_path(model: str, session: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if model == "MMVP_pressure_toolkit":
        root = ROOT / "results/pressure_tookit/predictions/eval_motion"
    elif model in {"MMVP_FPP-Net", "MMVP_VP-MoCap"}:
        root = ROOT / "results/VP-MoCap/predictions/eval_motion"
    else:
        raise ValueError(f"unsupported MMVP model label: {model}")
    return root / f"{session}.npz"


def normalize_xy(points: np.ndarray, size=(560, 400)) -> np.ndarray:
    xy = np.asarray(points)[..., (0, 2)].astype(np.float64)
    lo, hi = np.nanpercentile(xy.reshape(-1, 2), [2, 98], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    xy = (xy - lo) / span
    xy[..., 0] = 40 + xy[..., 0] * (size[0] - 80)
    xy[..., 1] = size[1] - 30 - xy[..., 1] * (size[1] - 60)
    return xy


def heatmap(frame: np.ndarray, size=(280, 340)) -> Image.Image:
    grid = np.asarray(frame, dtype=np.float32)
    vmax = max(float(grid.max()), 1.0)
    scaled = np.clip(grid / vmax, 0, 1) * (len(INFERNO) - 1)
    left = np.floor(scaled).astype(int)
    right = np.minimum(left + 1, len(INFERNO) - 1)
    frac = scaled - left
    rgb = (INFERNO[left] * (1 - frac[..., None]) + INFERNO[right] * frac[..., None]).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def render(args: argparse.Namespace) -> Path:
    row = manifest_row(args.session)
    pred_path = prediction_path(args.model, args.session, args.prediction)
    if not pred_path.is_file():
        raise FileNotFoundError(f"unified prediction missing: {pred_path}")
    pred, _, pred_names, _ = array_from_file(pred_path)
    gt = load_session_gt(seq_dir(row), int(row["n_frames"]), float(row.get("target_fps") or 40.0))
    n = min(len(pred), len(gt["joints"]))
    pred, gt_points = np.asarray(pred[:n]), np.asarray(gt["joints"][:n])
    pred_xy, gt_xy = normalize_xy(np.concatenate([pred, gt_points], axis=0))[:n], normalize_xy(np.concatenate([pred, gt_points], axis=0))[n:]
    names = {name: i for i, name in enumerate(pred_names)}
    edges = [(names[a], names[b]) for a, b in EDGES if a in names and b in names]
    row_parts = Path(row["pressure_path"]).parts
    cam = row_parts.index("cam3")
    date, subject = row_parts[cam + 1], row_parts[cam + 2]
    insole_root = (ROOT / "AnysoleWorkspace/derived/pressure_tookit/images" / date / subject / args.session / "insole"
                   if args.model == "MMVP_pressure_toolkit" else
                   ROOT / "AnysoleWorkspace/derived/VP-MoCap" / date / subject / args.session / "insole")
    output = DISPLAY / args.model / f"{args.session}.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in range(min(n, args.max_frames) if args.max_frames > 0 else n):
        canvas = Image.new("RGB", (860, 440), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), f"{args.model}  {args.session}  frame={frame}", fill="black")
        for ia, ib in edges:
            draw.line([tuple(pred_xy[frame, ia]), tuple(pred_xy[frame, ib])], fill=(210, 40, 40), width=3)
            draw.line([tuple(gt_xy[frame, ia]), tuple(gt_xy[frame, ib])], fill=(40, 90, 210), width=2)
        for point in pred_xy[frame]:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(210, 40, 40))
        draw.text((40, 408), "red=prediction  blue=GT", fill="black")
        insole_path = insole_root / f"{frame:03d}.npy"
        if insole_path.is_file():
            payload = np.load(insole_path, allow_pickle=True).item()
            grid = np.concatenate(payload["insole"], axis=1)
            canvas.paste(heatmap(grid), (570, 65))
        frames.append(canvas)
    if not frames:
        raise ValueError("no frames to render")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=25, loop=0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("MMVP_pressure_toolkit", "MMVP_FPP-Net", "MMVP_VP-MoCap"),
        required=True,
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--prediction", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    print(render(args))


if __name__ == "__main__":
    main()
