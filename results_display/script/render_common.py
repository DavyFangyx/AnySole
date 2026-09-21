"""Shared frame-rendering helpers for the results_display Test scripts.

Extracted from ``r_test1_visualize_motionpro.py`` so AnySole Test1/Test3 rendering no
longer imports the MotionPRO baseline tree (``Baselines/MotionPRO/lib``).
MotionPRO-specific model/checkpoint logic stays in ``r_test1_visualize_motionpro.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]
LEFT_FOOT_BOX = (slice(40, 120), slice(6, 54))
RIGHT_FOOT_BOX = (slice(40, 120), slice(66, 114))

PANEL_W = 460
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


def render_foot_panel(left_block, right_block, session_id, frame_idx, n_frames, fps, valid):
    canvas = Image.new("RGB", (FOOT_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, FOOT_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (FOOT_W // 2, 12), "Tactile", TITLE_FONT, fill=(230, 230, 235), anchor="mt")

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

    t_s = frame_idx / fps if fps else 0.0
    status = "valid" if valid else "invalid"
    info = f"{session_id}  t={t_s:.2f}s  {frame_idx}/{n_frames - 1}  {status}"
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 18), info, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


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


def render_skeleton_panel(joints, title, color, frame_idx, n_frames, edges=None, mpjpe_mm=None):
    canvas = Image.new("RGB", (PANEL_W, CANVAS_H), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (PANEL_W // 2, 12), title, TITLE_FONT, fill=color, anchor="mt")

    uv = project_joints(joints, PANEL_W, CANVAS_H - 40, margin=78)
    ground_y = int(CANVAS_H * 0.88)
    draw.ellipse([70, ground_y - 18, PANEL_W - 70, ground_y + 18], outline=(50, 54, 62), width=2)
    bone_edges = list(edges) if edges is not None else list(SMPL_EDGES)
    for i, j in bone_edges:
        if i >= len(uv) or j >= len(uv):
            continue
        p0 = (int(uv[i, 0]), int(uv[i, 1]))
        p1 = (int(uv[j, 0]), int(uv[j, 1]))
        draw.line([p0, p1], fill=color, width=6)
    for idx, (u, v) in enumerate(uv):
        r = 7 if idx in (0, 10, 11, 15, 20, 21) else 5
        fill = (255, 220, 90) if idx == 0 else color
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill)

    footer = f"frame {frame_idx}/{n_frames - 1}"
    if mpjpe_mm is not None and np.isfinite(mpjpe_mm):
        footer += f"  MPJPE {mpjpe_mm:.1f} mm"
    draw_text(draw, (PANEL_W // 2, CANVAS_H - 18), footer, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def compose_frame(left_block, right_block, pred_j, gt_j, gt_edges, session_id, frame_idx, n_frames, fps, valid, pred_edges=None):
    left = render_foot_panel(left_block, right_block, session_id, frame_idx, n_frames, fps, valid)
    mid = render_skeleton_panel(pred_j, "Predicted", (80, 200, 255), frame_idx, n_frames, edges=(pred_edges if pred_edges is not None else SMPL_EDGES))
    right = render_skeleton_panel(gt_j, "GT Motion", (255, 170, 80), frame_idx, n_frames, edges=gt_edges)
    return np.concatenate([left, mid, right], axis=1)
