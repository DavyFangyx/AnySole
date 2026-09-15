"""Standalone test of the insole-drift spatial compensator (Test4).

The ablation component ``anysole.ablations.insole_drift.InsoleDriftCompensator``
is a tactile-in / tactile-out module: both its input and output are
``(B, TW, 96)`` normalized pressure windows (left 48 cells + right 48 cells).
This script loads the *trained* compensator weights from the
``anysolev1_insole_drift`` checkpoint and runs the component alone -- no
diffusion, no encoders -- over real sessions.

Data path mirrors ``anysole.data.pressure.load_session_pressure`` exactly:
per-session ``align_meta.json`` -> fake_marked ``pressure_left/right.csv`` ->
48 cells per foot interpolated onto the 40 Hz video grid -> concatenated 96
raw values -> clip to [0, 1023] / 1023 -> non-overlapping 20-frame windows
(same windowing as the eval dataset).  The compensated output is stitched
back into a full-session sequence.

Outputs land under ``results_display/Test4_insole_drift/``:
    gif/ or mp4/ directory                left panel = original tactile,
      <session>_drift_compare.gif / .mp4  right panel = compensated tactile
    theta_summary.csv                    per-session drift statistics

Usage (run from the repository root):
    python results_display/script/test4_insole_drift.py
    python results_display/script/test4_insole_drift.py --session S7013
    python results_display/script/test4_insole_drift.py --session S7013 --stride 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cli_common  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from loguru import logger as log  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from anysole.ablations.insole_drift import InsoleDriftCompensator  # noqa: E402
from anysole.ablations.insole_drift.templates import load_template_bank  # noqa: E402
from anysole.types import FAKE_MARKED_ROOT, PRESSURE_CLIP, TW  # noqa: E402

# --- Rendering constants, same visual style as Test1's tactile panels. ---
FOOT_W = 360
INSOLE_W = 84
INSOLE_H = 252
CANVAS_H = 760
SENSOR_ROWS = 4
SENSOR_COLS = 12


def load_font(size):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except Exception:
                continue
    return ImageFont.load_default()


TITLE_FONT = load_font(30)
LABEL_FONT = load_font(22)
INFO_FONT = load_font(18)


def cells_to_heatmap(cells48, vmax=1.0):
    """(48,) or (4, 12) pressure cells -> insole heatmap, Test1 style.

    Stored as 4 medial-lateral rows x 12 heel-to-toe columns; rotate to a
    vertical insole like ``crop_foot`` does for the raster input.
    """
    cells = np.asarray(cells48, dtype=np.float32).reshape(SENSOR_ROWS, SENSOR_COLS)
    cells = np.rot90(cells, k=1)
    cells = np.clip(cells, 0.0, vmax) / max(vmax, 1e-6)
    rgb = (cm.inferno(cells)[..., :3] * 255.0).astype(np.uint8)
    rgb = cv2.resize(rgb, (INSOLE_W, INSOLE_H), interpolation=cv2.INTER_NEAREST)
    rows, cols = cells.shape[:2]
    cell_w = INSOLE_W / cols
    cell_h = INSOLE_H / rows
    for r in range(rows + 1):
        y = min(int(round(r * cell_h)), INSOLE_H - 1)
        rgb[y, :] = np.minimum(rgb[y, :] + 28, 255)
    for c in range(cols + 1):
        x = min(int(round(c * cell_w)), INSOLE_W - 1)
        rgb[:, x] = np.minimum(rgb[:, x] + 28, 255)
    return rgb


def draw_text(draw, xy, text, font, fill=(235, 235, 235), anchor="lt"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def render_pressure_panel(left48, right48, title, title_color, theta_line, session_id, frame_idx, n_frames, fps, valid):
    """One 360x760 panel with left/right insole heatmaps and a status footer."""
    canvas = Image.new("RGB", (FOOT_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, FOOT_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (FOOT_W // 2, 12), title, TITLE_FONT, fill=title_color, anchor="mt")

    left_img = Image.fromarray(cells_to_heatmap(left48))
    right_img = Image.fromarray(cells_to_heatmap(right48))
    gap = 24
    pair_w = left_img.width + gap + right_img.width
    x0 = (FOOT_W - pair_w) // 2
    y0 = 92
    canvas.paste(left_img, (x0, y0))
    canvas.paste(right_img, (x0 + left_img.width + gap, y0))
    draw_text(draw, (x0 + left_img.width // 2, 58), "Left", LABEL_FONT, fill=(180, 210, 255), anchor="mt")
    draw_text(draw, (x0 + left_img.width + gap + right_img.width // 2, 58), "Right", LABEL_FONT, fill=(255, 190, 160), anchor="mt")

    if theta_line:
        draw_text(draw, (FOOT_W // 2, CANVAS_H - 52), theta_line, INFO_FONT, fill=(160, 230, 170), anchor="mb")
    t_s = frame_idx / fps if fps else 0.0
    status = "valid" if valid else "invalid"
    info = f"{session_id}  t={t_s:.2f}s  {frame_idx}/{n_frames - 1}  {status}"
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 18), info, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


# --- Data loading: mirror anysole.data.pressure.load_session_pressure. ---


def session_dir(seq_root, session_id):
    matches = sorted(Path(seq_root).glob(f"*/*/{session_id}"))
    dirs = [path for path in matches if path.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No sequence dir for {session_id} under {seq_root}")
    return dirs[0]


def load_session_raw(seq_dir: Path) -> tuple[np.ndarray, str, int]:
    """Load (n, 96) normalized raw pressure for a session, plus subject id."""
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    rec_dir = FAKE_MARKED_ROOT / meta["date"] / meta["subject"] / meta["rec_name"]
    left_path = rec_dir / "pressure_left.csv"
    right_path = rec_dir / "pressure_right.csv"
    if not (left_path.is_file() and right_path.is_file()):
        raise FileNotFoundError(f"No fake_marked pressure CSV under {rec_dir}")

    def read_48(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty pressure CSV: {csv_path}")
        value_cols = [name for name in rows[0] if name.isdigit()]
        t_us = np.asarray([float(row["t_us"]) for row in rows], dtype=np.float64)
        values = np.asarray([[float(row[col]) for col in value_cols] for row in rows], dtype=np.float32)
        return t_us, values

    t_l, left = read_48(left_path)
    t_r, right = read_48(right_path)
    n = int(meta["n_frames"])
    fps = float(meta.get("target_fps", 40.0))
    t_grid = (float(meta["visual_start_s"]) + np.arange(n, dtype=np.float64) / fps) * 1e6
    interp = lambda t, v: np.stack([np.interp(t_grid, t, v[:, c]) for c in range(v.shape[1])], axis=1)
    t_raw = np.concatenate([interp(t_l, left), interp(t_r, right)], axis=1)
    t_raw = np.clip(t_raw, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
    return t_raw.astype(np.float32), str(meta["subject"]), n


def run_compensator(comp: InsoleDriftCompensator, t_raw: np.ndarray, subject_id: str):
    """Non-overlapping TW windows through the compensator, stitched back.

    Returns (n, 96) compensated pressure and (n_windows, 3) per-window thetas
    for each foot.
    """
    n = t_raw.shape[0]
    n_windows = n // TW
    if n_windows == 0:
        raise ValueError(f"Session has {n} frames < window length {TW}")
    used = n_windows * TW
    windows = t_raw[:used].reshape(n_windows, TW, 96)
    device = next(comp.parameters()).device
    x = torch.from_numpy(windows).to(device)
    with torch.inference_mode():
        compensated, theta_l, theta_r = comp(x, [subject_id] * n_windows)
    return (
        compensated.cpu().numpy().reshape(-1, 96),
        theta_l.cpu().numpy(),
        theta_r.cpu().numpy(),
    )


# --- Main flow ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test4: standalone insole-drift compensator (tactile in -> tactile out).")
    cli_common.add_common_args(parser, seq_root=True, contact_method=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test4_insole_drift")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained anysolev1_insole_drift checkpoint; only the drift_compensator weights are used. "
        "Default: results/AnySole/anysolev1_insole_drift_<contact-method>/checkpoints/ckpt_last.pt.",
    )
    parser.add_argument("--templates", type=str, default=str(cli_common.WORKSPACE_ROOT / "calibration/insole_templates.json"))
    parser.add_argument(
        "--no-subject-cond",
        action="store_true",
        help="Pass session ids instead of subject ids (replicates the current train/eval call, "
        "which misses the template lookup and always falls back to template index 0).",
    )
    return parser.parse_args()


def load_compensator(args: argparse.Namespace, device: torch.device) -> tuple[InsoleDriftCompensator, dict, int]:
    templates, subject_map = load_template_bank(args.templates)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "drift_compensator."
    sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub_state:
        raise ValueError(f"No drift_compensator weights in {args.checkpoint}; is it an anysolev1_insole_drift checkpoint?")
    comp = InsoleDriftCompensator(templates, subject_map)
    comp.load_state_dict(sub_state)
    comp.eval()
    comp.to(device)
    return comp, subject_map, checkpoint.get("epoch", -1)


def render_session(args: argparse.Namespace, comp, session_id: str, out_dir: Path) -> str:
    gen_path = cli_common.media_path(out_dir, f"{session_id}_drift_compare", args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}: outputs already exist under {out_dir}")
        return "skip"

    seq_dir = session_dir(args.seq_root, session_id)
    t_raw, subject_id, n_frames = load_session_raw(seq_dir)
    fake_path = Path(seq_dir) / "fake_mask.npy"
    fake = np.load(fake_path).astype(np.uint8).reshape(-1) if fake_path.is_file() else np.zeros(n_frames, dtype=np.uint8)
    subject_arg = session_id if args.no_subject_cond else subject_id
    compensated, theta_l, theta_r = run_compensator(comp, t_raw, subject_arg)

    n_used = compensated.shape[0]
    if n_used < n_frames:
        log.warning(f"{session_id}: rendering first {n_used}/{n_frames} frames (multiple of TW={TW})")
    theta_l_deg = np.degrees(theta_l)
    theta_r_deg = np.degrees(theta_r)

    frame_ids = list(range(0, n_used, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    frame_fps = cli_common.viz_fps(args.fps, args.stride)

    frames = []
    for t in frame_ids:
        w = t // TW
        tl = f"L(tx,ty,rot) {theta_l[w, 0]:+.1f}mm {theta_l[w, 1]:+.1f}mm {theta_l_deg[w, 2]:+.1f}deg   " \
             f"R(tx,ty,rot) {theta_r[w, 0]:+.1f}mm {theta_r[w, 1]:+.1f}mm {theta_r_deg[w, 2]:+.1f}deg"
        valid = not bool(fake[t])
        left_panel = render_pressure_panel(
            t_raw[t, :48], t_raw[t, 48:], "Original", (200, 210, 230), None,
            session_id, t, n_used, args.fps, valid,
        )
        right_panel = render_pressure_panel(
            compensated[t, :48], compensated[t, 48:], "Compensated", (150, 230, 170), tl,
            session_id, t, n_used, args.fps, valid,
        )
        frames.append(np.concatenate([left_panel, right_panel], axis=1))
    if not frames:
        raise RuntimeError(f"No frames rendered for {session_id}")

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")

    mae = float(np.abs(t_raw[:n_used] - compensated).mean())
    row = {
        "session_id": session_id,
        "subject_id": subject_id,
        "subject_arg": subject_arg,
        "n_frames": n_used,
        "n_windows": compensated.shape[0] // TW,
        "mae": round(mae, 5),
        "theta_l_tx_mm_mean": round(float(theta_l[:, 0].mean()), 2),
        "theta_l_ty_mm_mean": round(float(theta_l[:, 1].mean()), 2),
        "theta_l_rot_deg_mean": round(float(theta_l_deg[:, 2].mean()), 2),
        "theta_l_rot_deg_std": round(float(theta_l_deg[:, 2].std()), 2),
        "theta_r_tx_mm_mean": round(float(theta_r[:, 0].mean()), 2),
        "theta_r_ty_mm_mean": round(float(theta_r[:, 1].mean()), 2),
        "theta_r_rot_deg_mean": round(float(theta_r_deg[:, 2].mean()), 2),
        "theta_r_rot_deg_std": round(float(theta_r_deg[:, 2].std()), 2),
    }
    log.info(f"{session_id}: rendered={len(frames)} frames, mae={mae:.4f}, "
             f"theta_l={row['theta_l_tx_mm_mean']},{row['theta_l_ty_mm_mean']},{row['theta_l_rot_deg_mean']} "
             f"theta_r={row['theta_r_tx_mm_mean']},{row['theta_r_ty_mm_mean']},{row['theta_r_rot_deg_mean']}")
    return row


def main() -> int:
    args = parse_args()
    orig_cwd = os.getcwd()
    out_dir = Path(cli_common.resolve_path(args.out_dir, orig_cwd))
    out_dir.mkdir(parents=True, exist_ok=True)

    split_csv = Path(cli_common.resolve_path(args.split_csv, orig_cwd))
    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    if args.session:
        log.info(f"Sessions from --session: {session_ids}")
    else:
        log.info(f"Sessions from {args.split} split ({len(session_ids)}): {session_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is None:
        args.checkpoint = str(cli_common.RESULTS_ROOT / "AnySole" / cli_common.anysole_model_dir("anysolev1_insole_drift", args.contact_method) / "checkpoints" / "ckpt_last.pt")
    comp, subject_map, epoch = load_compensator(args, device)
    log.info(f"Compensator loaded from {args.checkpoint} (epoch {epoch}, {len(subject_map)} subjects), device={device}")

    summary_path = out_dir / "theta_summary.csv"
    rows = []
    for session_id in session_ids:
        try:
            result = render_session(args, comp, session_id, out_dir)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"Skip {session_id}: {exc}")
            continue
        if isinstance(result, dict):
            rows.append(result)
    if rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
