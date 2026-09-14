"""Test5: GT BVH + tactile insoles + contact indicator animation (contact label check).

Renders, per session, the ground-truth mocap BVH skeleton aligned to the
40 Hz session grid next to the tactile insole heatmaps, with a per-foot
contact indicator driven by ``contact.npy`` (red = contact, green = no
contact), plus a full-session contact timeline strip.  A per-frame text
shows the 48-cell CSV pressure sum vs ``CONTACT_SUM_THRESH`` -- the exact
quantity ``prepare_sequences.py:contact_from_insoles`` thresholds -- so the
"labels are 97% always 1" issue is visible at a glance.

Threshold analysis outputs (candidate thresholds are applied to the 48-cell
CSV sums, NOT to the pressure.npz raster):

    gif/ or mp4/ directory          animation (tactile + contact badges | GT
      <session>_contact.gif / .mp4  skeleton with recolored feet | timeline)
    contact_summary.csv            per-session contact rates, sum stats,
                                   contact rate under candidate thresholds
    threshold_analysis.png         sum histograms + rate-vs-threshold curves

Outputs land under ``results_display/Test5_contact/`` (or the
``ANYSOLE_RESULTSDISPLAY`` override).

Usage (run from the repository root):
    python results_display/script/test5_contact.py
    python results_display/script/test5_contact.py --session S10103 --max-frames 100
    python results_display/script/test5_contact.py --force
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cli_common  # noqa: E402
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger as log  # noqa: E402
from matplotlib import cm, font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from anysole.data.pressure import load_session_pressure  # noqa: E402
from anysole.types import CONTACT_SUM_THRESH  # noqa: E402
from bvh_aligner_pose import parse_bvh_aligner  # noqa: E402

# --- Rendering constants, same visual style as Test1's panels. ---
LEFT_FOOT_BOX = (slice(40, 120), slice(6, 54))
RIGHT_FOOT_BOX = (slice(40, 120), slice(66, 114))
PANEL_W = 460
FOOT_W = 360
INSOLE_W = 84
INSOLE_H = 252
CANVAS_H = 760
STRIP_H = 120
TOTAL_W = FOOT_W + PANEL_W
TOTAL_H = CANVAS_H + STRIP_H
SENSOR_ROWS = 4
SENSOR_COLS = 12

LEFT_CONTACT_COL = 6
RIGHT_CONTACT_COL = 7
C_RED = (200, 50, 50)
C_GREEN = (60, 180, 90)
C_YELLOW = (200, 180, 50)
C_GRAY = (140, 140, 150)
C_BONE = (255, 170, 80)
CANDIDATE_THRESHOLDS = (100.0, 1000.0, 3000.0, 5000.0, 7000.0, 9000.0, 11000.0)


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
LEGEND_FONT = load_font(14)


def session_dir(seq_root, session_id):
    matches = sorted(Path(seq_root).glob(f"*/*/{session_id}"))
    dirs = [p for p in matches if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No sequence dir for {session_id} under {seq_root}")
    return dirs[0]


def joints_to_meters(joints):
    pts = np.asarray(joints, dtype=np.float32)
    if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0:
        return pts * 0.01
    return pts


def interp_joints(joints, frame_time, query_t):
    src_t = np.arange(joints.shape[0], dtype=np.float64) * float(frame_time)
    out = np.empty((query_t.size,) + joints.shape[1:], dtype=np.float64)
    clipped = np.clip(query_t, src_t[0], src_t[-1]) if src_t.size else query_t
    for joint_i in range(joints.shape[1]):
        for axis_i in range(3):
            out[:, joint_i, axis_i] = np.interp(clipped, src_t, joints[:, joint_i, axis_i])
    return out.astype(np.float32)


def parents_to_edges(parents):
    edges = []
    for child, parent in enumerate(np.asarray(parents).tolist()):
        if parent >= 0:
            edges.append((parent, child))
    return edges


def crop_foot(pressure_t, box):
    y_slice, x_slice = box
    block = np.asarray(pressure_t[y_slice, x_slice], dtype=np.float32)
    if block.shape != (SENSOR_ROWS * 20, SENSOR_COLS * 4):
        raise ValueError(f"unexpected insole crop {block.shape}")
    cells = block.reshape(SENSOR_ROWS, 20, SENSOR_COLS, 4).mean(axis=(1, 3))
    # stored as 4 medial-lateral rows x 12 heel-to-toe columns; rotate to a vertical insole
    return np.rot90(cells, k=1)


def pressure_to_heatmap(cells, vmax=255.0):
    cells = np.clip(np.asarray(cells, dtype=np.float32), 0.0, vmax) / max(vmax, 1e-6)
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


def project_joints(joints, width, height, margin=70):
    pts = np.asarray(joints, dtype=np.float64)
    # keep the figure body-sized: center on the hip so walking translation
    # does not shrink the skeleton.
    pts = pts - pts[0]
    # MocapVideoAligner uses z-up: z is vertical, x/y form the ground plane.
    horizontal = pts[:, 0] * 0.94 + pts[:, 1] * 0.34
    depth = -pts[:, 0] * 0.34 + pts[:, 1] * 0.94
    vertical = pts[:, 2] - np.min(pts[:, 2])
    ground = np.stack([horizontal, depth], axis=1)
    body_h = max(float(np.max(vertical)), 0.8)
    span = max(np.max(np.abs(ground)), body_h * 0.55, 0.7)
    span = min(float(span), 1.15)
    scale = (min(width, height) - 2 * margin) / (2.0 * span)
    u = width * 0.50 + ground[:, 0] * scale
    v = height * 0.84 - vertical * scale * 1.18 - ground[:, 1] * scale * 0.05
    return np.stack([u, v], axis=1)


def load_pressure(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake_path = seq_dir / "fake_mask.npy"
    if fake_path.is_file():
        fake = np.load(fake_path).astype(np.uint8).reshape(-1)
    else:
        fake = np.zeros((pressure.shape[0],), dtype=np.uint8)
    return pressure, fake


def load_gt(seq_dir: Path, n_frames: int, fps: float = 40.0):
    """GT BVH aligned to the 40 Hz session grid, dataset-consistent.

    ``trim_leading_seconds=0.0`` is required: the parser's 0.40 default
    shifts the returned joints by 0.4 s (z_note/Agent_05_AnySole_缺陷修复任务书.md Task 1).
    """
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    bvh_path = Path(cli_common.resolve_path(meta["bvh_path"]))
    parsed = parse_bvh_aligner(bvh_path, trim_leading_seconds=0.0)
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    joints = interp_joints(parsed["joints"], parsed["frame_time"], t_mocap)
    return joints_to_meters(joints), parsed["parents"], parsed["names"]


def foot_joint_indices(names: list[str]) -> dict[str, list[int]]:
    """Map foot joints by name (skeletons are Skeleton0/1/5 with different
    joint counts, so indices must not be hardcoded).  Captures ``LeftFoot``,
    ``LeftToeBase`` and the ``..._EndSite`` variants automatically."""
    out: dict[str, list[int]] = {"left": [], "right": []}
    for idx, name in enumerate(names):
        lower = name.lower()
        for side in ("left", "right"):
            if lower.startswith(f"{side}foot") or lower.startswith(f"{side}toe"):
                out[side].append(idx)
    for side in ("left", "right"):
        if not out[side]:
            log.warning(f"No {side} foot joints matched by name; names={names}")
    return out


def contact_from_sums(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    """Replicate prepare_sequences.py:contact_from_insoles for the foot columns."""
    contact = np.zeros((left48.shape[0], 2), dtype=np.float32)
    contact[left48.sum(axis=1) > CONTACT_SUM_THRESH, 0] = 1.0
    contact[right48.sum(axis=1) > CONTACT_SUM_THRESH, 1] = 1.0
    return contact


def load_contact(seq_dir: Path, n: int, left48, right48) -> tuple[np.ndarray, bool]:
    """Return (n, 2) bool contact [left, right] and an availability flag."""
    path = Path(seq_dir) / "contact.npy"
    if path.is_file():
        raw = np.load(path).astype(np.float32)
        if raw.ndim == 2 and raw.shape[1] >= 8:
            return raw[:n, [LEFT_CONTACT_COL, RIGHT_CONTACT_COL]] > 0.5, True
        log.warning(f"{path}: unexpected shape {raw.shape}, deriving contact from CSV sums")
    if left48 is not None and right48 is not None:
        log.warning(f"No usable contact.npy under {seq_dir}; deriving contact from 48-cell CSV sums")
        return contact_from_sums(left48[:n], right48[:n]) > 0.5, True
    log.warning(f"No contact data under {seq_dir}")
    return np.zeros((n, 2), dtype=bool), False


def side_color(on: bool, valid: bool) -> tuple[int, int, int]:
    if not valid:
        return C_GRAY
    return C_RED if on else C_GREEN


def render_tactile_panel(
    left_block, right_block, left_on, right_on, left_sum, right_sum,
    session_id, frame_idx, n_frames, fps, valid, contact_available, csv_available, rate_l, rate_r,
):
    canvas = Image.new("RGB", (FOOT_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, FOOT_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (FOOT_W // 2, 12), "Tactile + Contact", TITLE_FONT, fill=(230, 230, 235), anchor="mt")

    left_img = Image.fromarray(pressure_to_heatmap(left_block))
    right_img = Image.fromarray(pressure_to_heatmap(right_block))
    gap = 24
    pair_w = left_img.width + gap + right_img.width
    x0 = (FOOT_W - pair_w) // 2
    y0 = 92
    canvas.paste(left_img, (x0, y0))
    canvas.paste(right_img, (x0 + left_img.width + gap, y0))
    draw_text(draw, (x0 + left_img.width // 2, 58), "Left", LABEL_FONT, fill=(180, 210, 255), anchor="mt")
    draw_text(draw, (x0 + left_img.width + gap + right_img.width // 2, 58), "Right", LABEL_FONT, fill=(255, 190, 160), anchor="mt")

    # Contact indicator: colored outline + badge under each insole.
    sides = (
        (x0, left_img.width, left_on, left_sum),
        (x0 + left_img.width + gap, right_img.width, right_on, right_sum),
    )
    for insole_x, insole_w, on, s in sides:
        color = side_color(bool(on), valid)
        draw.rectangle(
            [insole_x - 3, y0 - 3, insole_x + insole_w + 2, y0 + INSOLE_H + 2],
            outline=color, width=4,
        )
        center_x = insole_x + insole_w // 2
        if not valid:
            badge, badge_color = "无效", C_GRAY
        elif contact_available:
            badge, badge_color = ("接触", C_RED) if on else ("无接触", C_GREEN)
        else:
            badge, badge_color = "无数据", C_GRAY
        draw_text(draw, (center_x, y0 + INSOLE_H + 20), badge, LABEL_FONT, fill=badge_color, anchor="mt")
        if not valid:
            sum_text, sum_color = "invalid", C_GRAY
        elif csv_available:
            sum_text = f"sum={s:.0f} > thr={CONTACT_SUM_THRESH:.0f}"
            sum_color = badge_color
        else:
            sum_text, sum_color = "CSV missing", C_GRAY
        draw_text(draw, (center_x, y0 + INSOLE_H + 52), sum_text, INFO_FONT, fill=sum_color, anchor="mt")

    if contact_available:
        rate_line = f"contact L {rate_l:.0%}  R {rate_r:.0%}"
    else:
        rate_line = "contact 无数据"
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 52), rate_line, INFO_FONT, fill=(180, 180, 190), anchor="mb")

    t_s = frame_idx / fps if fps else 0.0
    status = "valid" if valid else "invalid"
    info = f"{session_id}  t={t_s:.2f}s  {frame_idx}/{n_frames - 1}  {status}"
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 18), info, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_skeleton_contact_panel(joints, edges, foot_idx, left_on, right_on, valid, frame_idx, n_frames):
    canvas = Image.new("RGB", (PANEL_W, CANVAS_H), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (PANEL_W // 2, 12), "GT BVH Contact", TITLE_FONT, fill=C_BONE, anchor="mt")

    uv = project_joints(joints, PANEL_W, CANVAS_H - 40, margin=78)
    ground_y = int(CANVAS_H * 0.88)
    draw.ellipse([70, ground_y - 18, PANEL_W - 70, ground_y + 18], outline=(50, 54, 62), width=2)

    l_color = side_color(bool(left_on), valid)
    r_color = side_color(bool(right_on), valid)
    for i, j in edges:
        if i >= len(uv) or j >= len(uv):
            continue
        color = C_BONE
        if j in foot_idx["left"]:
            color = l_color
        elif j in foot_idx["right"]:
            color = r_color
        p0 = (int(uv[i, 0]), int(uv[i, 1]))
        p1 = (int(uv[j, 0]), int(uv[j, 1]))
        draw.line([p0, p1], fill=color, width=6)
    for idx, (u, v) in enumerate(uv):
        r = 7 if idx == 0 else 5
        if idx in foot_idx["left"]:
            fill = l_color
        elif idx in foot_idx["right"]:
            fill = r_color
        else:
            fill = (255, 220, 90) if idx == 0 else C_BONE
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill)

    footer = f"frame {frame_idx}/{n_frames - 1}"
    draw_text(draw, (PANEL_W // 2, CANVAS_H - 18), footer, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_skeleton_unavailable(message: str, frame_idx: int, n_frames: int):
    canvas = Image.new("RGB", (PANEL_W, CANVAS_H), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (PANEL_W // 2, 12), "GT BVH Contact", TITLE_FONT, fill=C_BONE, anchor="mt")
    draw_text(draw, (PANEL_W // 2, CANVAS_H // 2), message, INFO_FONT, fill=C_GRAY, anchor="mm")
    footer = f"frame {frame_idx}/{n_frames - 1}"
    draw_text(draw, (PANEL_W // 2, CANVAS_H - 18), footer, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_contact_timeline(left_contact, right_contact, fake, t, n_frames, contact_available):
    canvas = Image.new("RGB", (TOTAL_W, STRIP_H), (16, 16, 22))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, TOTAL_W - 1, STRIP_H - 1], outline=(48, 48, 56))
    draw_text(draw, (8, 4), "Contact Timeline", INFO_FONT, fill=(200, 200, 210), anchor="lt")

    legend_items = ((C_RED, "接触"), (C_GREEN, "无接触"), (C_YELLOW, "混合"))
    for k, (color, label) in enumerate(legend_items):
        x = 620 + k * 64
        draw.rectangle([x, 10, x + 8, 18], fill=color)
        draw_text(draw, (x + 12, 4), label, LEGEND_FONT, fill=(200, 200, 210), anchor="lt")

    bar_x0, bar_x1 = 24, TOTAL_W - 24  # 772 px span
    rows = (
        (left_contact, "L", (180, 210, 255), 34, 64),
        (right_contact, "R", (255, 190, 160), 70, 100),
    )
    span = bar_x1 - bar_x0
    n_blocks = max(1, min(n_frames, span))
    block_span = int(np.ceil(n_frames / n_blocks))
    block_w = span / n_blocks
    for contact, label, label_color, y0, y1 in rows:
        draw_text(draw, (8, (y0 + y1) // 2), label, LABEL_FONT, fill=label_color, anchor="mm")
        for b in range(n_blocks):
            lo = b * block_span
            hi = min(n_frames, lo + block_span)
            if not contact_available or fake[lo:hi].any():
                color = C_YELLOW
            elif contact[lo:hi].all():
                color = C_RED
            elif not contact[lo:hi].any():
                color = C_GREEN
            else:
                color = C_YELLOW
            bx0 = bar_x0 + int(round(b * block_w))
            bx1 = bar_x0 + int(round((b + 1) * block_w))
            draw.rectangle([bx0, y0, max(bx1 - 1, bx0), y1], fill=color)

    if n_frames > 1:
        px = bar_x0 + (t / (n_frames - 1)) * span
        draw.line([px, 30, px, 104], fill=(230, 230, 230), width=2)
        draw.rectangle([px - 2, 26, px + 2, 30], fill=(230, 230, 230))
    return np.asarray(canvas)


def compose_frame(
    left_block, right_block, left_on, right_on, left_sum, right_sum,
    joints, edges, foot_idx, fake_t, session_id, t, n_frames, fps,
    contact_available, csv_available, rate_l, rate_r, left_contact, right_contact, fake,
    gt_ok, gt_msg,
):
    valid = not bool(fake_t)
    tactile = render_tactile_panel(
        left_block, right_block, left_on, right_on, left_sum, right_sum,
        session_id, t, n_frames, fps, valid, contact_available, csv_available, rate_l, rate_r,
    )
    if gt_ok:
        skeleton = render_skeleton_contact_panel(joints[t], edges, foot_idx, left_on, right_on, valid, t, n_frames)
    else:
        skeleton = render_skeleton_unavailable(gt_msg, t, n_frames)
    top = np.concatenate([tactile, skeleton], axis=1)
    timeline = render_contact_timeline(left_contact, right_contact, fake, t, n_frames, contact_available)
    return np.concatenate([top, timeline], axis=0)


# --- Threshold analysis (candidate thresholds applied to the 48-cell CSV sums). ---

SUM_FIELDS = [
    "sum_mean_left", "sum_min_left", "sum_max_left",
    "sum_mean_right", "sum_min_right", "sum_max_right",
    "pct_sum_gt_100_left", "pct_sum_gt_100_right",
]
THR_FIELDS = [f"rate_left_{int(t):d}" for t in CANDIDATE_THRESHOLDS] + [
    f"rate_right_{int(t):d}" for t in CANDIDATE_THRESHOLDS
]
FIELD_NAMES = ["session_id", "n_frames", "contact_rate_left", "contact_rate_right"] + SUM_FIELDS + THR_FIELDS


def analyze_session(session_id: str, contact: np.ndarray, left48, right48, csv_available: bool) -> dict:
    row: dict = {
        "session_id": session_id,
        "n_frames": int(contact.shape[0]),
        "contact_rate_left": round(float(contact[:, 0].mean()), 4),
        "contact_rate_right": round(float(contact[:, 1].mean()), 4),
    }
    if csv_available:
        sums_l = np.asarray(left48, dtype=np.float64).sum(axis=1)
        sums_r = np.asarray(right48, dtype=np.float64).sum(axis=1)
        row.update({
            "sum_mean_left": round(float(sums_l.mean()), 1),
            "sum_min_left": round(float(sums_l.min()), 1),
            "sum_max_left": round(float(sums_l.max()), 1),
            "sum_mean_right": round(float(sums_r.mean()), 1),
            "sum_min_right": round(float(sums_r.min()), 1),
            "sum_max_right": round(float(sums_r.max()), 1),
            "pct_sum_gt_100_left": round(float((sums_l > CONTACT_SUM_THRESH).mean()), 4),
            "pct_sum_gt_100_right": round(float((sums_r > CONTACT_SUM_THRESH).mean()), 4),
        })
        for thr in CANDIDATE_THRESHOLDS:
            row[f"rate_left_{int(thr):d}"] = round(float((sums_l > thr).mean()), 4)
            row[f"rate_right_{int(thr):d}"] = round(float((sums_r > thr).mean()), 4)
        row["_sums_l"] = sums_l
        row["_sums_r"] = sums_r
    else:
        for key in SUM_FIELDS + THR_FIELDS:
            row[key] = ""
    return row


def write_summary_csv(out_dir: Path, rows: list[dict]):
    path = out_dir / "contact_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {path}")


def plot_threshold_analysis(out_dir: Path, rows: list[dict]):
    plot_rows = [row for row in rows if row.get("_sums_l") is not None]
    if not plot_rows:
        log.warning("No CSV sum data available; skipping threshold_analysis.png")
        return
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.isfile(cjk):
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    thresholds = np.asarray(CANDIDATE_THRESHOLDS, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, side in enumerate(("left", "right")):
        key = f"_sums_{side[0]}"
        sums_list = [row[key] for row in plot_rows]

        ax_hist = axes[0, col]
        ax_hist.hist(np.concatenate(sums_list), bins=64, color=("tab:blue" if col == 0 else "tab:orange"), alpha=0.75)
        for thr in thresholds:
            linestyle = "-" if thr == CONTACT_SUM_THRESH else "--"
            label = "thr=100（现行）" if thr == CONTACT_SUM_THRESH else None
            ax_hist.axvline(thr, color="black" if thr == CONTACT_SUM_THRESH else "gray",
                            linestyle=linestyle, linewidth=1.4, label=label)
        if col == 0:
            ax_hist.legend(loc="upper right", fontsize=9)
        ax_hist.set_title(f"{'左' if col == 0 else '右'}足 48格压力和分布")
        ax_hist.set_xlabel("帧压力和（48格求和）")
        ax_hist.set_ylabel("帧数")

        ax_rate = axes[1, col]
        for row in plot_rows:
            sums = row[key]
            rates = [(sums > thr).mean() for thr in thresholds]
            ax_rate.plot(thresholds, rates, color="tab:blue" if col == 0 else "tab:orange",
                         alpha=0.4, linewidth=1.0)
        mean_rates = np.mean(
            [[(row[key] > thr).mean() for thr in thresholds] for row in plot_rows], axis=0
        )
        ax_rate.plot(thresholds, mean_rates, color="black", linewidth=2.5, label="均值")
        ax_rate.set_xscale("log")
        ax_rate.set_xticks(thresholds)
        ax_rate.set_xticklabels([f"{int(t):d}" for t in thresholds], rotation=45)
        ax_rate.set_xlabel("候选阈值")
        ax_rate.set_ylabel("接触率")
        ax_rate.set_title(f"{'左' if col == 0 else '右'}足 接触率-阈值曲线（{len(plot_rows)} sessions）")
        ax_rate.legend(loc="lower left", fontsize=9)
        ax_rate.grid(True, which="both", alpha=0.25)

    fig.suptitle("Test5 contact 标签阈值分析", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "threshold_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {path}")


# --- Main flow ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test5: GT BVH + tactile insoles + contact indicator animation and threshold analysis.")
    cli_common.add_common_args(parser, seq_root=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test5_contact")
    return parser.parse_args()


def analysis_row(session_id: str, seq_dir: Path, fps: float) -> dict | None:
    """Compute the summary row without loading pressure.npz / BVH.

    Used for sessions whose animation already exists, so re-runs without
    ``--force`` still regenerate ``contact_summary.csv`` / ``threshold_analysis.png``.
    """
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    n = int(meta["n_frames"])
    contact_raw = None
    contact_path = Path(seq_dir) / "contact.npy"
    if contact_path.is_file():
        contact_raw = np.load(contact_path).astype(np.float32)
        n = min(n, contact_raw.shape[0])
    t_grid = float(meta["visual_start_s"]) + np.arange(n, dtype=np.float64) / float(fps)
    left48 = right48 = None
    csv_available = False
    try:
        out = load_session_pressure(meta, t_grid)
        left48, right48 = out["left48"], out["right48"]
        csv_available = True
        n = min(n, left48.shape[0])
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"{session_id}: no 48-cell CSV pressure ({exc}); sum/analysis columns will be empty")
    if csv_available:
        left48 = left48[:n]
        right48 = right48[:n]
    contact, _ = load_contact(seq_dir, n, left48, right48)
    return analyze_session(session_id, contact, left48, right48, csv_available)


def render_session(args: argparse.Namespace, session_id: str, out_dir: Path) -> tuple[str, dict | None]:
    gen_path = cli_common.media_path(out_dir, f"{session_id}_contact", args.gen)
    seq_dir = session_dir(args.seq_root, session_id)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}: outputs already exist under {out_dir}")
        return "skip", analysis_row(session_id, seq_dir, args.fps)

    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    pressure, fake = load_pressure(seq_dir)

    contact_raw = None
    contact_path = Path(seq_dir) / "contact.npy"
    if contact_path.is_file():
        contact_raw = np.load(contact_path).astype(np.float32)

    t_grid = float(meta["visual_start_s"]) + np.arange(pressure.shape[0], dtype=np.float64) / float(args.fps)
    csv_available = False
    left48 = right48 = None
    try:
        pressure_session = load_session_pressure(meta, t_grid)
        left48, right48 = pressure_session["left48"], pressure_session["right48"]
        csv_available = True
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"{session_id}: no 48-cell CSV pressure ({exc}); sum text/analysis columns will be empty")

    n = pressure.shape[0]
    if contact_raw is not None and contact_raw.shape[0] != n:
        log.warning(f"{session_id}: contact.npy has {contact_raw.shape[0]} frames vs pressure {n}")
    n = min(n, contact_raw.shape[0]) if contact_raw is not None else n
    if fake.shape[0] != n:
        log.warning(f"{session_id}: fake_mask has {fake.shape[0]} frames vs pressure {n}")
        n = min(n, fake.shape[0])
    if csv_available and left48.shape[0] != n:
        log.warning(f"{session_id}: CSV pressure has {left48.shape[0]} frames vs pressure {n}")
        n = min(n, left48.shape[0])
    pressure = pressure[:n]
    fake = fake[:n]
    contact, contact_available = load_contact(seq_dir, n, left48, right48)
    if csv_available:
        left48 = left48[:n]
        right48 = right48[:n]

    gt_ok, gt_msg = False, "GT BVH unavailable"
    joints = edges = None
    foot_idx = {"left": [], "right": []}
    try:
        joints, parents, names = load_gt(seq_dir, n, args.fps)
        edges = parents_to_edges(parents)
        foot_idx = foot_joint_indices(names)
        gt_ok, gt_msg = True, ""
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"{session_id}: GT BVH load failed ({exc}); rendering placeholder skeleton")

    row = analyze_session(session_id, contact, left48, right48, csv_available)
    if n < 2:
        log.warning(f"{session_id}: only {n} frame(s), skipping animation")
        return "row-only", row

    rate_l = float(contact[:, 0].mean())
    rate_r = float(contact[:, 1].mean())
    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    frame_fps = cli_common.viz_fps(args.fps, args.stride)

    frames = []
    for t in frame_ids:
        left_block = crop_foot(pressure[t], LEFT_FOOT_BOX)
        right_block = crop_foot(pressure[t], RIGHT_FOOT_BOX)
        left_sum = float(left48[t].sum()) if csv_available else None
        right_sum = float(right48[t].sum()) if csv_available else None
        frames.append(compose_frame(
            left_block, right_block, contact[t, 0], contact[t, 1], left_sum, right_sum,
            joints, edges, foot_idx, fake[t], session_id, t, n, args.fps,
            contact_available, csv_available, rate_l, rate_r,
            contact[:, 0], contact[:, 1], fake, gt_ok, gt_msg,
        ))
    if not frames:
        raise RuntimeError(f"No frames rendered for {session_id}")

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")
    log.info(
        f"{session_id}: rendered={len(frames)} frames, contact L={rate_l:.3f} R={rate_r:.3f}, "
        f"gt={'ok' if gt_ok else 'missing'}, csv={'ok' if csv_available else 'missing'}"
    )
    return "write", row


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

    collected = []
    for session_id in session_ids:
        try:
            status, row = render_session(args, session_id, out_dir)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"Skip {session_id}: {exc}")
            continue
        if row is not None:
            collected.append(row)
    if collected:
        write_summary_csv(out_dir, collected)
        plot_threshold_analysis(out_dir, collected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
