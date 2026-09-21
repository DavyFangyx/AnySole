"""Render Step2Motion tactile input, generated BVH, and aligned raw BVH.

Usage (run from the repository root):
    python results_display/script/r_test1_visualize_step2motion.py
    python results_display/script/r_test1_visualize_step2motion.py \
        results://Step2Motion/predictions/gait_model/S7063_gen.bvh
    python results_display/script/r_test1_visualize_step2motion.py --self-test
    python results_display/script/r_test1_visualize_step2motion.py --mesh --max-frames 40

``--mesh`` renders the GT panel as the full SMPL surface (mesh + skeleton
overlay via ``utils.smpl_mesh``) when the session has an SMPL-24 archive and
no ``--raw`` override; otherwise the GT panel keeps the BVH skeleton.  The
generated panel stays a BVH skeleton (Step2Motion has no SMPL params).  The
default (and ``--only-bone``) renders skeleton panels only.

Outputs are written below ``results_display/r_test1_visualize/Step2Motion/gait_model`` (or the
``ANYSOLE_RESULTSDISPLAY`` override).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# The BVH-era Step2Motion export helper now lives in the archived source
# tree; Baselines/Step2Motion no longer ships bvh_export.py.
STEP2MOTION_SRC = REPO_ROOT / "Baselines_backup" / "Step2Motion" / "src"

# results_display ``utils`` must win over the Step2Motion tree's own
# ``utils.py``: import everything from it before pushing STEP2MOTION_SRC to
# sys.path[0] (its shadowing previously made ``from utils import cli_common``
# load Step2Motion's utils, which drags in pymotion and breaks on py3.8).
from utils import cli_common
import cv2
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as SciRotation
from utils.bvh_aligner_pose import parse_bvh_aligner
from utils.motion_io import load_motion, smpl_yup_to_display
from utils.smpl_mesh import mesh_from_archive, render_mesh_frame, smpl_faces
from anysole.data.smpl_io import resolve_smpl_path
from anysole.types import SMPL_ROOTS

if str(STEP2MOTION_SRC) not in sys.path:
    sys.path.insert(0, str(STEP2MOTION_SRC))

from bvh_export import write_bvh


PANEL_W = 460
FOOT_W = 360
INSOLE_W = 84
INSOLE_H = 252
CANVAS_H = 760

PRED_COLOR = (80, 200, 255)
GT_COLOR = (255, 170, 80)
ROOT_COLOR = (255, 220, 90)
HIGHLIGHT_JOINTS = (0, 3, 4, 7, 8, 13, 17, 21)

GAIT_ROOT = REPO_ROOT
SEQ_ROOT = cli_common.WORKSPACE_ROOT / "derived/MotionPRO/sequences/cam3"
DEFAULT_VIZ_DIR = cli_common.DISPLAY_ROOT / "r_test1_visualize/Step2Motion/gait_model"
MIN_FRAMES = 101
TARGET_HZ = 40.0
DEFAULT_PRED_DIR = cli_common.RESULTS_ROOT / "Step2Motion/predictions/gait_model"


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except Exception:
                continue
    return ImageFont.load_default()


TITLE_FONT = load_font(30)
LABEL_FONT = load_font(22)
INFO_FONT = load_font(18)


def parse_bvh(path: str | Path) -> dict:
    """Parse a BVH file and return world-space joints."""
    return parse_bvh_aligner(path)
    lines = Path(path).read_text(errors="replace").splitlines()
    names = []
    parents = []
    offsets = []
    n_channels = []
    rot_orders = []
    pos_orders = []
    parent_stack = []
    current = None
    end_site = False
    motion_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("MOTION"):
            motion_idx = i
            break
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            names.append(stripped.split()[1])
            parents.append(parent_stack[-1] if parent_stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            n_channels.append(0)
            rot_orders.append("xyz")
            pos_orders.append("xyz")
            current = len(names) - 1
            end_site = False
        elif stripped.startswith("End Site"):
            end_site = True
        elif stripped == "{":
            if current is not None and not end_site:
                parent_stack.append(current)
        elif stripped == "}":
            if end_site:
                end_site = False
            elif parent_stack:
                parent_stack.pop()
                current = parent_stack[-1] if parent_stack else None
        elif stripped.startswith("OFFSET") and current is not None and not end_site:
            offsets[current] = [float(v) for v in stripped.split()[1:4]]
        elif stripped.startswith("CHANNELS") and current is not None and not end_site:
            words = stripped.split()
            n_channels[current] = int(words[1])
            pos_axes = []
            rot_axes = []
            for word in words[2:]:
                if word.endswith("position"):
                    pos_axes.append(word[0].lower())
                elif word.endswith("rotation"):
                    rot_axes.append(word[0].lower())
            if pos_axes:
                pos_orders[current] = "".join(pos_axes)
            if rot_axes:
                rot_orders[current] = "".join(rot_axes)
    if motion_idx is None:
        raise ValueError("No MOTION section in %s" % path)
    n_frames = int(lines[motion_idx + 1].split(":")[1])
    frame_time = float(lines[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in lines[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape[0] != n_frames:
        raise ValueError("Expected %d frames in %s, got %d" % (n_frames, path, motion.shape[0]))
    n_joints = len(names)
    offsets_np = np.asarray(offsets, dtype=np.float64)
    local_pos = np.broadcast_to(offsets_np[None], (n_frames, n_joints, 3)).copy()
    eulers = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    expected = int(sum(n_channels))
    if motion.shape[1] != expected:
        raise ValueError("Expected %d channels in %s, got %d" % (expected, path, motion.shape[1]))
    cursor = 0
    for joint_i in range(n_joints):
        nch = n_channels[joint_i]
        chunk = motion[:, cursor : cursor + nch]
        if nch == 6:
            pos_vals = chunk[:, :3]
            rot_vals = chunk[:, 3:]
            for axis_i, axis in enumerate(pos_orders[joint_i]):
                local_pos[:, joint_i, "xyz".index(axis)] = pos_vals[:, axis_i]
            eulers[:, joint_i] = rot_vals
        elif nch == 3:
            eulers[:, joint_i] = chunk
        else:
            raise ValueError("Unsupported channel count %d for joint %s" % (nch, names[joint_i]))
        cursor += nch
    local_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)
    for joint_i in range(n_joints):
        local_rot[:, joint_i] = SciRotation.from_euler(
            rot_orders[joint_i], eulers[:, joint_i], degrees=True
        ).as_matrix()
    global_rot = np.zeros_like(local_rot)
    global_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    for joint_i, parent in enumerate(parents):
        if parent < 0:
            global_rot[:, joint_i] = local_rot[:, joint_i]
            global_pos[:, joint_i] = local_pos[:, joint_i]
        else:
            global_rot[:, joint_i] = global_rot[:, parent] @ local_rot[:, joint_i]
            offset = local_pos[:, joint_i, :, None]
            global_pos[:, joint_i] = global_pos[:, parent] + (global_rot[:, parent] @ offset)[..., 0]
    fps = 1.0 / frame_time if frame_time > 0 else 40.0
    return {
        "joints": global_pos.astype(np.float32),
        "parents": np.asarray(parents, dtype=np.int64),
        "names": names,
        "fps": float(fps),
        "frame_time": float(frame_time),
    }


def load_cs_pressures(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open() as handle:
        data = json.load(handle)
    left = np.asarray(data["l_pressures"], dtype=np.float32).reshape(-1, 16)
    right = np.asarray(data["r_pressures"], dtype=np.float32).reshape(-1, 16)
    return left, right


def moticon16_to_insole(cells16: np.ndarray) -> np.ndarray:
    """Map Moticon 16 channels onto a vertical insole, toe at the top.

    Official order is heel[8] then toe[8]. Each half is 4 medial-lateral rows
    by 2 length groups, matching process_gait.pool_48_to_16.
    """
    cells = np.asarray(cells16, dtype=np.float32).reshape(16)
    heel = cells[:8].reshape(4, 2)
    toe = cells[8:].reshape(4, 2)
    return np.concatenate([toe.T, heel.T], axis=0)


def pressure_to_heatmap(cells: np.ndarray, vmax: float) -> np.ndarray:
    cells = np.clip(np.asarray(cells, dtype=np.float32), 0.0, vmax) / max(vmax, 1e-6)
    rgb = (cm.inferno(cells)[..., :3] * 255.0).astype(np.uint8)
    rgb = cv2.resize(rgb, (INSOLE_W, INSOLE_H), interpolation=cv2.INTER_NEAREST)
    rows, cols = cells.shape[:2]
    cell_w = INSOLE_W / cols
    cell_h = INSOLE_H / rows
    for row in range(rows + 1):
        y = min(int(round(row * cell_h)), INSOLE_H - 1)
        rgb[y, :] = np.minimum(rgb[y, :] + 28, 255)
    for col in range(cols + 1):
        x = min(int(round(col * cell_w)), INSOLE_W - 1)
        rgb[:, x] = np.minimum(rgb[:, x] + 28, 255)
    return rgb


def draw_text(draw, xy, text, font, fill=(235, 235, 235), anchor="lt"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def render_foot_panel(left_block, right_block, clip_id, frame_idx, n_frames, fps, vmax):
    canvas = Image.new("RGB", (FOOT_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, FOOT_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (FOOT_W // 2, 12), "Tactile", TITLE_FONT, fill=(230, 230, 235), anchor="mt")

    left_img = Image.fromarray(pressure_to_heatmap(left_block, vmax))
    right_img = Image.fromarray(pressure_to_heatmap(right_block, vmax))
    gap = 24
    pair_w = left_img.width + gap + right_img.width
    x0 = (FOOT_W - pair_w) // 2
    y0 = 92
    canvas.paste(left_img, (x0, y0))
    canvas.paste(right_img, (x0 + left_img.width + gap, y0))
    draw_text(draw, (x0 + left_img.width // 2, 58), "Left", LABEL_FONT, fill=(180, 210, 255), anchor="mt")
    draw_text(
        draw,
        (x0 + left_img.width + gap + right_img.width // 2, 58),
        "Right",
        LABEL_FONT,
        fill=(255, 190, 160),
        anchor="mt",
    )

    t_s = frame_idx / fps if fps else 0.0
    info = "%s  t=%.2fs  %d/%d" % (clip_id, t_s, frame_idx, n_frames - 1)
    draw_text(draw, (FOOT_W // 2, CANVAS_H - 18), info, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def parents_to_edges(parents: np.ndarray):
    edges = []
    for child, parent in enumerate(np.asarray(parents).tolist()):
        if parent >= 0:
            edges.append((parent, child))
    return edges


def joints_to_meters(joints: np.ndarray) -> np.ndarray:
    pts = np.asarray(joints, dtype=np.float32)
    if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0:
        return pts * 0.01
    return pts


def interp_joints(joints: np.ndarray, frame_time: float, query_t: np.ndarray) -> np.ndarray:
    src_t = np.arange(joints.shape[0], dtype=np.float64) * float(frame_time)
    out = np.empty((query_t.size,) + joints.shape[1:], dtype=np.float64)
    clipped = np.clip(query_t, src_t[0], src_t[-1]) if src_t.size else query_t
    for joint_i in range(joints.shape[1]):
        for axis_i in range(3):
            out[:, joint_i, axis_i] = np.interp(clipped, src_t, joints[:, joint_i, axis_i])
    return out.astype(np.float32)


def split_fake_clips(fake_mask: np.ndarray, min_frames: int = MIN_FRAMES) -> list[tuple[int, int]]:
    clips = []
    n = len(fake_mask)
    i = 0
    while i < n:
        if fake_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not fake_mask[j]:
            j += 1
        if j - i >= min_frames:
            clips.append((i, j))
        i = j
    return clips


def parse_clip_id(clip_id: str) -> tuple[str, int]:
    name = str(clip_id or "").strip()
    if "_c" in name:
        base, _, suffix = name.rpartition("_c")
        if base.startswith("S") and suffix.isdigit():
            return base, int(suffix)
    return name, 0


def find_seq_dir(session_id: str) -> Path:
    matches = sorted(p for p in SEQ_ROOT.glob("*/*/%s" % session_id) if p.is_dir())
    if not matches:
        raise FileNotFoundError("No sequence dir for %s under %s" % (session_id, SEQ_ROOT))
    return matches[0]


def _gt_time_grid(clip_id: str, n_frames: int, fps: float) -> np.ndarray:
    """Mocap-source query grid for a clip (visual_start/offset + fake-split clip start)."""
    session_id, clip_index = parse_clip_id(clip_id)
    seq_dir = find_seq_dir(session_id)
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    fake_path = seq_dir / "fake_mask.npy"
    if fake_path.is_file():
        clips = split_fake_clips(np.load(fake_path).astype(bool), MIN_FRAMES)
    else:
        clips = []
    if clips:
        start, _end = clips[min(clip_index, len(clips) - 1)]
    else:
        start = 0
    t_grid = float(meta["visual_start_s"]) + (start + 1 + np.arange(n_frames, dtype=np.float64)) / float(fps)
    return t_grid - float(meta["offset_s"])


def load_original_gt(clip_id: str, n_frames: int, fps: float = TARGET_HZ) -> dict:
    session_id, _clip_index = parse_clip_id(clip_id)
    seq_dir = find_seq_dir(session_id)
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    parsed = parse_bvh(meta["bvh_path"])
    joints = interp_joints(parsed["joints"], parsed["frame_time"], _gt_time_grid(clip_id, n_frames, fps))
    return {
        "joints": joints_to_meters(joints),
        "parents": parsed["parents"],
        "names": parsed["names"],
    }


def load_original_gt_mesh(clip_id: str, n_frames: int, fps: float = TARGET_HZ) -> dict | None:
    """SMPL-24 GT mesh (verts + joints) on the clip's query grid.

    Returns None when the session has no SMPL archive; callers keep the BVH
    skeleton GT panel instead.
    """
    session_id, _clip_index = parse_clip_id(clip_id)
    seq_dir = find_seq_dir(session_id)
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    try:
        smpl = resolve_smpl_path(meta, tuple(SMPL_ROOTS))
    except FileNotFoundError:
        return None
    t_mocap = _gt_time_grid(clip_id, n_frames, fps)
    loaded = load_motion(smpl, query_t=t_mocap)
    mesh = mesh_from_archive(smpl, query_t=t_mocap)
    return {
        "joints": loaded["joints"],
        "parents": loaded["parents"],
        "names": loaded["names"],
        "verts": smpl_yup_to_display(mesh["verts"]),
        "faces": mesh["faces"],
    }


def load_raw_bvh(
    raw_bvh: str | Path | None,
    clip_id: str,
    n_frames: int,
    fps: float,
    raw_start_time: float | None,
) -> dict:
    if raw_bvh is None:
        return load_original_gt(clip_id, n_frames, fps)
    parsed = parse_bvh(raw_bvh)
    start = float(raw_start_time or 0.0)
    query_t = start + np.arange(n_frames, dtype=np.float64) / float(fps)
    joints = interp_joints(parsed["joints"], parsed["frame_time"], query_t)
    return {
        "joints": joints_to_meters(joints),
        "parents": parsed["parents"],
        "names": parsed["names"],
    }


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

def render_skeleton_panel(joints, parents, title, color, frame_idx, n_frames, mpjpe_mm=None):
    canvas = Image.new("RGB", (PANEL_W, CANVAS_H), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    draw_text(draw, (PANEL_W // 2, 12), title, TITLE_FONT, fill=color, anchor="mt")

    uv = project_joints(joints, PANEL_W, CANVAS_H - 40, margin=78)
    ground_y = int(CANVAS_H * 0.88)
    draw.ellipse([70, ground_y - 18, PANEL_W - 70, ground_y + 18], outline=(50, 54, 62), width=2)
    for i, j in parents_to_edges(parents):
        p0 = (int(uv[i, 0]), int(uv[i, 1]))
        p1 = (int(uv[j, 0]), int(uv[j, 1]))
        draw.line([p0, p1], fill=color, width=6)
    for idx, (u, v) in enumerate(uv):
        radius = 7 if idx in HIGHLIGHT_JOINTS else 5
        fill = ROOT_COLOR if idx == 0 else color
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=fill)

    footer = "frame %d/%d" % (frame_idx, n_frames - 1)
    if mpjpe_mm is not None and np.isfinite(mpjpe_mm):
        footer += "  MPJPE %.1f mm" % mpjpe_mm
    draw_text(draw, (PANEL_W // 2, CANVAS_H - 18), footer, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_mesh_panel(verts: np.ndarray, faces: np.ndarray, joints: np.ndarray,
                      parents: np.ndarray, title: str, color: tuple,
                      frame_idx: int, n_frames: int) -> np.ndarray:
    """Full-SMPL panel at the classic PANEL_W x CANVAS_H size (mesh + skeleton)."""
    return render_mesh_frame(
        verts, faces, joints=joints, edges=parents_to_edges(parents), title=title,
        frame_index=frame_idx, total=n_frames,
        mesh_color=tuple(c / 255.0 for c in color),
        bone_color="#C46A4A", joint_color="#{:02x}{:02x}{:02x}".format(*color),
        title_color="#{:02x}{:02x}{:02x}".format(*color),
        bg_color="#0A0C10", size=(PANEL_W, CANVAS_H),
    )


def compose_frame(left_block, right_block, pred_j, gt_j, pred_parents, gt_parents, clip_id, frame_idx, n_frames, fps, vmax):
    left = render_foot_panel(left_block, right_block, clip_id, frame_idx, n_frames, fps, vmax)
    mid = render_skeleton_panel(pred_j, pred_parents, "BVH GEN", PRED_COLOR, frame_idx, n_frames)
    right = render_skeleton_panel(gt_j, gt_parents, "BVH RAW", GT_COLOR, frame_idx, n_frames)
    return np.concatenate([left, mid, right], axis=1)


def discover_pred_files(pred_bvh):
    if pred_bvh.is_dir():
        pred_files = sorted(pred_bvh.glob("*_gen*.bvh"))
        if not pred_files:
            raise SystemExit("No *_gen.bvh files in %s" % pred_bvh)
        return pred_files
    if pred_bvh.is_file():
        return [pred_bvh]
    hint = (
        "Generated BVH not found: %s\n"
        "Default visualizes every test-set export:\n"
        "  python results_display/script/r_test1_visualize_step2motion.py\n"
        "One clip:\n"
        "  python results_display/script/r_test1_visualize_step2motion.py results://Step2Motion/predictions/gait_model/S7063_gen.bvh"
    ) % pred_bvh
    raise SystemExit(hint)


def visualize_from_exports(
    pred_bvh: str | Path,
    raw_bvh: str | Path | None,
    cs_json: str | Path,
    out_dir: str | Path,
    clip_id: str | None = None,
    fps: float | None = None,
    raw_start_time: float | None = None,
    stride: int = 2,
    max_frames: int = 0,
    gen: str = "gif",
    mesh: bool = False,
) -> dict:
    pred_bvh = Path(pred_bvh)
    cs_json = Path(cs_json)
    out_dir = Path(out_dir)
    clip_id = clip_id or export_clip_id(pred_bvh)

    pred = parse_bvh(pred_bvh)
    left, right = load_cs_pressures(cs_json)
    n = min(pred["joints"].shape[0], left.shape[0], right.shape[0])
    if n < 1:
        raise RuntimeError("No overlapping frames for %s" % clip_id)
    pred_j = pred["joints"][:n]
    left = left[:n]
    right = right[:n]
    fps = float(fps or pred["fps"] or TARGET_HZ)
    # SMPL surface GT when requested and no --raw override (the raw override
    # is an arbitrary BVH, which has no SMPL params).
    gt_mesh = load_original_gt_mesh(clip_id, n, fps) if mesh and raw_bvh is None else None
    if mesh and raw_bvh is None and gt_mesh is None:
        print("Warning: no SMPL archive for %s; mesh mode keeps the BVH skeleton GT panel." % clip_id)
    if gt_mesh is not None:
        gt_j = gt_mesh["joints"][:n]
        gt_parents = gt_mesh["parents"]
    else:
        try:
            gt = load_raw_bvh(raw_bvh, clip_id, n, fps, raw_start_time)
        except FileNotFoundError as exc:
            # Older exports may contain an absolute BVH path from a machine where
            # the original mocap archive was mounted.  Keep rendering predictions
            # and tactile input when that optional reference is unavailable.
            print("Warning: raw BVH unavailable (%s); rendering without a reference skeleton." % exc)
            gt = {
                "joints": pred["joints"].copy(),
                "parents": pred["parents"].copy(),
                "names": pred["names"],
            }
        gt_j = gt["joints"][:n]
        gt_parents = gt["parents"]
    pred_parents = pred["parents"]
    vmax = float(np.percentile(np.concatenate([left, right], axis=1), 99))
    vmax = max(vmax, 1.0)

    frame_ids = list(range(0, n, max(int(stride), 1)))
    if max_frames and max_frames > 0:
        frame_ids = frame_ids[: max_frames]
    frames = []
    for t in frame_ids:
        left_panel = render_foot_panel(
            moticon16_to_insole(left[t]),
            moticon16_to_insole(right[t]),
            clip_id, t, n, fps, vmax,
        )
        mid = render_skeleton_panel(pred_j[t], pred_parents, "BVH GEN", PRED_COLOR, t, n)
        if gt_mesh is not None:
            right_panel = render_mesh_panel(
                gt_mesh["verts"][t], gt_mesh["faces"], gt_j[t], gt_parents,
                "SMPL GT", GT_COLOR, t, n,
            )
        else:
            right_panel = render_skeleton_panel(gt_j[t], gt_parents, "BVH RAW", GT_COLOR, t, n)
        frames.append(np.concatenate([left_panel, mid, right_panel], axis=1))
    if not frames:
        raise RuntimeError("No frames rendered for %s" % clip_id)

    stem = "%s_compare%s" % (clip_id, "_mesh" if gt_mesh is not None else "")
    gen_path = cli_common.media_path(out_dir, stem, gen)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(fps, stride)
    if gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    print("Wrote %s" % gen_path)
    print("%s: rendered=%d frames=%d pred_joints=%d gt_joints=%d" % (
        clip_id, len(frames), n, pred_j.shape[1], gt_j.shape[1]
    ))
    return {"path": gen_path, "n": n, "rendered": len(frames)}


def infer_sidecar(pred_bvh: Path, suffix: str) -> Path:
    name = pred_bvh.name
    gen_name = re.sub(r"_gen(?:_s\d+)?(?=\.bvh$)", suffix, name)
    if gen_name != name:
        return pred_bvh.with_name(gen_name)
    if "_pred" in name:
        return pred_bvh.with_name(name.replace("_pred", suffix, 1))
    stem = pred_bvh.stem
    if stem.endswith("_pred"):
        return pred_bvh.with_name(stem[: -len("_pred")] + suffix + pred_bvh.suffix)
    return pred_bvh.with_name(stem + suffix + pred_bvh.suffix)


def export_clip_id(pred_bvh: Path) -> str:
    return re.sub(r"_gen(?:_s\d+)?$", "", pred_bvh.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tactile vs generated BVH vs raw BVH.")
    cli_common.add_common_args(
        parser,
        session=False,
        split_csv=False,
        split=False,
        seq_root=False,
        fps_default=0.0,
        out_dir_default=DEFAULT_VIZ_DIR,
    )
    parser.add_argument(
        "pred_bvh",
        nargs="?",
        default="",
        help="Generated BVH file or directory. Default: centralized Step2Motion predictions.",
    )
    parser.add_argument("--raw", type=str, default="", help="Raw mocap BVH path. Default reads *_meta.json or gait align_meta.json.")
    parser.add_argument("--raw-start-time", type=float, default=None, help="First generated frame time in the raw BVH, in seconds.")
    parser.add_argument("--cs", type=str, default="", help="Insole JSON path. Default infers *_cs.json.")
    parser.add_argument("--clip-id", type=str, default="")
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render the GT panel as the full SMPL surface (mesh + skeleton overlay) when the "
        "session has an SMPL-24 archive and no --raw override; generated panel stays a BVH skeleton.",
    )
    parser.add_argument(
        "--only-bone",
        action="store_true",
        help="Explicit skeleton-only panels (the default); conflicts with --mesh.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.mesh and args.only_bone:
        parser.error("--mesh and --only-bone are mutually exclusive")
    return args


def _tiny_bvh(n_frames: int = 2) -> str:
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "\tOFFSET 0 1 0",
        "\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation",
        "\tJOINT LeftUpLeg",
        "\t{",
        "\t\tOFFSET 0.1 -0.1 0",
        "\t\tCHANNELS 3 Xrotation Yrotation Zrotation",
        "\t\tEnd Site",
        "\t\t{",
        "\t\t\tOFFSET 0 -0.4 0",
        "\t\t}",
        "\t}",
        "}",
        "MOTION",
        "Frames: %d" % n_frames,
        "Frame Time: 0.025",
    ]
    for i in range(n_frames):
        lines.append("0.0 %.3f 0.0 0 0 0 0 0 0" % (1.0 + 0.01 * i))
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    cells = np.zeros(16, dtype=np.float32)
    cells[8] = 10.0  # first toe channel
    cells[7] = 20.0  # last heel channel
    grid = moticon16_to_insole(cells)
    assert grid.shape == (4, 4), grid.shape
    assert grid[0, 0] == 10.0, grid
    assert grid[-1, -1] == 20.0, grid

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bvh_path = tmp_path / "tiny.bvh"
        bvh_path.write_text(_tiny_bvh())
        parsed = parse_bvh(bvh_path)
        assert parsed["joints"].shape == (2, 3, 3), parsed["joints"].shape
        assert parsed["parents"].tolist() == [-1, 0, 1]
        np.testing.assert_allclose(parsed["joints"][0, 0], [0.0, 0.0, 2.0], atol=1e-6)
        generated_path = tmp_path / "generated.bvh"
        rotations = np.zeros((2, 2, 4), dtype=np.float64)
        rotations[..., 0] = 1.0
        write_bvh(
            generated_path,
            ["Hips", "LeftLeg"],
            np.asarray([-1, 0]),
            np.asarray([[0.0, 0.0, 0.0], [0.0, -0.4, 0.0]]),
            rotations,
            np.asarray([[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]]),
            0.025,
        )
        generated = parse_bvh(generated_path)
        assert generated["joints"].shape == (2, 3, 3)
        assert generated["parents"].tolist() == [-1, 0, 1]
        np.testing.assert_allclose(generated["joints"][1, 0], [0.1, 0.0, 1.0])
    print("visualize_compare self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    pred_bvh = Path(cli_common.resolve_path(args.pred_bvh)) if args.pred_bvh else DEFAULT_PRED_DIR
    pred_files = discover_pred_files(pred_bvh)
    out_dir = Path(cli_common.resolve_path(args.out_dir)) if args.out_dir else DEFAULT_VIZ_DIR
    if not args.pred_bvh:
        print("Visualizing all test-set exports in %s (%d clips)" % (pred_bvh, len(pred_files)))
    n_write = 0
    n_skip = 0
    for pred_path in pred_files:
        cs_json = Path(args.cs) if args.cs else infer_sidecar(pred_path, "_cs").with_suffix(".json")
        clip_id = args.clip_id or export_clip_id(pred_path)
        raw_bvh = args.raw or None
        raw_start_time = args.raw_start_time
        meta_path = infer_sidecar(pred_path, "_meta").with_suffix(".json")
        # In mesh mode the GT surface comes from the session's SMPL archive,
        # not the exported raw BVH; the meta raw_bvh_path is only the default
        # skeleton-GT source (keep it unless mesh is requested).
        if raw_bvh is None and meta_path.is_file() and not args.mesh:
            export_meta = json.loads(meta_path.read_text())
            raw_bvh = export_meta.get("raw_bvh_path") or None
            if raw_start_time is None:
                raw_start_time = export_meta.get("raw_start_time")
        if raw_bvh:
            raw_bvh = cli_common.resolve_path(raw_bvh)
        stem = "%s_compare%s" % (clip_id, "_mesh" if args.mesh else "")
        gen_path = cli_common.media_path(out_dir, stem, args.gen)
        if cli_common.outputs_ready([gen_path]) and not args.force:
            print("Skip %s: already exists" % clip_id)
            n_skip += 1
            continue
        visualize_from_exports(
            pred_path,
            raw_bvh,
            cs_json,
            out_dir,
            clip_id=clip_id,
            fps=args.fps or None,
            raw_start_time=raw_start_time,
            stride=args.stride,
            max_frames=args.max_frames,
            gen=args.gen,
            mesh=args.mesh,
        )
        n_write += 1
    print("visualize_compare done: wrote=%d skipped=%d total=%d" % (n_write, n_skip, len(pred_files)))


if __name__ == "__main__":
    main()
