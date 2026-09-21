"""Render MotionPRO tactile input, prediction, and GT as one animation.

Usage (run from the repository root):
    python results_display/script/visualize_motionpro.py
    python results_display/script/visualize_motionpro.py --session S14103
    python results_display/script/visualize_motionpro.py \
        --checkpoint results://MotionPRO/checkpoints/imagepressure2smpl/init/5e-05/imagepressure2smpl_best.pth

Outputs are written below ``results_display/Test1_visualization/MotionPRO`` (or the
``ANYSOLE_RESULTSDISPLAY`` override).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cli_common


REPO_ROOT = cli_common.REPO_ROOT
MOTIONPRO_ROOT = REPO_ROOT / "Baselines/MotionPRO"
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

import cv2
import numpy as np
import smplx
import torch
from loguru import logger as log
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as SciRotation

from lib.model.FRAPPE import FRAPPE
from lib.util.io import load_smpl_npy
from bvh_aligner_pose import parse_bvh_aligner


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
PRESSURE_HW = (96, 96)
WINDOW_LENGTH = 20
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


def parse_bvh(path):
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
        raise ValueError(f"No MOTION section in {path}")
    n_frames = int(lines[motion_idx + 1].split(":")[1])
    frame_time = float(lines[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in lines[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape[0] != n_frames:
        raise ValueError(f"Expected {n_frames} frames in {path}, got {motion.shape[0]}")
    n_joints = len(names)
    offsets_np = np.asarray(offsets, dtype=np.float64)
    local_pos = np.broadcast_to(offsets_np[None], (n_frames, n_joints, 3)).copy()
    eulers = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    expected = int(sum(n_channels))
    if motion.shape[1] != expected:
        raise ValueError(f"Expected {expected} channels in {path}, got {motion.shape[1]}")
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
            raise ValueError(f"Unsupported channel count {nch} for joint {names[joint_i]}")
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


def joints_to_meters(joints):
    pts = np.asarray(joints, dtype=np.float32)
    if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0:
        return pts * 0.01
    return pts


def smpl_to_aligner_zup(joints):
    """Convert SMPL x/y/z-up joints to MocapVideoAligner's z-up display axes."""
    pts = np.asarray(joints, dtype=np.float32)
    return np.stack((pts[..., 0], -pts[..., 2], pts[..., 1]), axis=-1)


def interp_joints(joints, frame_time, query_t):
    src_t = np.arange(joints.shape[0], dtype=np.float64) * float(frame_time)
    out = np.empty((query_t.size,) + joints.shape[1:], dtype=np.float64)
    clipped = np.clip(query_t, src_t[0], src_t[-1]) if src_t.size else query_t
    for joint_i in range(joints.shape[1]):
        for axis_i in range(3):
            out[:, joint_i, axis_i] = np.interp(clipped, src_t, joints[:, joint_i, axis_i])
    return out.astype(np.float32)


def load_original_gt(seq_dir, n_frames, fps=40.0):
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    parsed = parse_bvh(cli_common.resolve_path(meta["bvh_path"]))
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    joints = interp_joints(parsed["joints"], parsed["frame_time"], t_mocap)
    return joints_to_meters(joints), parsed["parents"]


def parents_to_edges(parents):
    edges = []
    for child, parent in enumerate(np.asarray(parents).tolist()):
        if parent >= 0:
            edges.append((parent, child))
    return edges


def load_session(seq_dir):
    pressure = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    feature = torch.load(seq_dir / "feature_hrnet.pth", map_location="cpu")
    if not torch.is_tensor(feature):
        feature = torch.as_tensor(feature)
    feature = feature.float()
    smpl_gt = load_smpl_npy(seq_dir / "smpl.npy")
    fake_path = seq_dir / "fake_mask.npy"
    if fake_path.is_file():
        fake = np.load(fake_path).astype(np.uint8).reshape(-1)
    else:
        fake = np.zeros((pressure.shape[0],), dtype=np.uint8)
    n = min(pressure.shape[0], feature.shape[0], smpl_gt["body_pose"].shape[0], fake.shape[0])
    return {
        "pressure": pressure[:n],
        "feature": feature[:n],
        "smpl": smpl_gt,
        "fake": fake[:n],
        "n": n,
    }


def resize_pressure_window(pressure_win):
    tensor = torch.from_numpy(np.ascontiguousarray(pressure_win)).float()
    if tuple(tensor.shape[-2:]) == PRESSURE_HW:
        return tensor
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(1),
        size=PRESSURE_HW,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def pad_window(tensor, length):
    if tensor.shape[0] >= length:
        return tensor
    pad = length - tensor.shape[0]
    zeros = torch.zeros((pad,) + tuple(tensor.shape[1:]), dtype=tensor.dtype)
    return torch.cat([tensor, zeros], dim=0)


def windows_for_session(n_frames, fake, window_length):
    n_windows = (n_frames + window_length - 1) // window_length
    out = []
    for window_idx in range(n_windows):
        left = window_idx * window_length
        right = min(n_frames, (window_idx + 1) * window_length)
        if fake[left:right].any():
            continue
        out.append((left, right))
    if not out:
        raise RuntimeError("No valid windows after dropping fake-marked frames")
    return out


def smpl_output(smpl, beta, theta, trans):
    pred_global_orient, pred_body_pose = torch.split(theta, [3, 69], dim=1)
    return smpl(
        betas=beta,
        body_pose=pred_body_pose,
        global_orient=pred_global_orient,
        transl=trans,
    )


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


def run_session(model, smpl, seq_dir, device, window_length):
    data = load_session(seq_dir)
    n = data["n"]
    pred = np.zeros((n, 22, 3), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    smpl_gt = data["smpl"]
    beta_full = torch.from_numpy(np.asarray(smpl_gt["betas"][:10], dtype=np.float32))

    model.eval()
    with torch.no_grad():
        for left, right in windows_for_session(n, data["fake"], window_length):
            n_real = right - left
            feature = pad_window(data["feature"][left:right], window_length).unsqueeze(0).to(device)
            pressure = pad_window(
                resize_pressure_window(data["pressure"][left:right]) / 255.0,
                window_length,
            ).unsqueeze(0).to(device)
            smpl_params = model(feature, pressure)
            if smpl_params.ndim == 3:
                smpl_params = smpl_params.reshape(-1, smpl_params.shape[-1])
            smpl_params = smpl_params[:n_real]
            _, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
            beta = beta_full.unsqueeze(0).repeat(n_real, 1).to(device)
            pred_smpl = smpl_output(smpl, beta, pred_theta, pred_trans)
            pred_joints = pred_smpl.joints[:, :22].detach().cpu().numpy()
            pred[left:right] = smpl_to_aligner_zup(pred_joints)
            valid[left:right] = True
    return data["pressure"], pred, valid, n


def discover_checkpoints(checkpoint_arg, orig_cwd):
    items = cli_common.split_csv_arg(checkpoint_arg)
    if not items:
        root = cli_common.RESULTS_ROOT / "MotionPRO/checkpoints"
        found = sorted(p for p in root.rglob("*.pth") if p.is_file())
        if not found:
            raise FileNotFoundError(f"No checkpoints under {root}")
        return found
    found = []
    for item in items:
        path = Path(cli_common.resolve_path(item, orig_cwd))
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p for p in path.rglob("*.pth") if p.is_file()))
        else:
            log.warning(f"Checkpoint not found, skip: {path}")
    uniq = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    if not uniq:
        raise FileNotFoundError("No checkpoint files resolved")
    return uniq


def checkpoint_tag(ckpt_path, orig_cwd):
    ckpt_path = Path(ckpt_path).resolve()
    root = (cli_common.RESULTS_ROOT / "MotionPRO/checkpoints").resolve()
    try:
        return str(ckpt_path.relative_to(root).with_suffix(""))
    except ValueError:
        return ckpt_path.stem


def load_model(checkpoint, device):
    model = FRAPPE()
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tactile vs predicted SMPL vs GT BVH.")
    cli_common.add_common_args(parser, seq_root=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test1_visualization" / "MotionPRO")
    parser.add_argument("--smpl-model", type=str, default=str(cli_common.WORKSPACE_ROOT / "dependencies/smpl/SMPL_NEUTRAL.pkl"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint file, directory, or comma-separated list. Empty scans the centralized results directory.",
    )
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def render_session(model, smpl, seq_dir, device, args, session_id, session_out):
    gen_path = cli_common.media_path(session_out, f"{session_id}_compare", args.gen)
    if cli_common.outputs_ready([gen_path]) and not args.force:
        log.info(f"Skip {session_id}: already exists under {session_out}")
        return "skip"
    pressure, pred, valid, n = run_session(model, smpl, seq_dir, device, args.window_length)
    gt, gt_parents = load_original_gt(seq_dir, n, args.fps)
    gt_edges = parents_to_edges(gt_parents)
    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames and args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    frames = []
    for t in frame_ids:
        keep = bool(valid[t])
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
                keep,
            )
        )
    if not frames:
        raise RuntimeError(f"No frames rendered for {session_id}")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")
    n_valid = int(valid.sum())
    log.info(f"{session_id}: rendered={len(frames)} valid={n_valid}/{n} gt_joints={gt.shape[1]}")
    return "write"


def main():
    args = parse_args()
    orig_cwd = os.getcwd()
    seq_root = cli_common.resolve_path(args.seq_root, orig_cwd)
    smpl_model = cli_common.resolve_path(args.smpl_model, orig_cwd)
    split_csv = cli_common.resolve_path(args.split_csv, orig_cwd)
    out_dir = Path(cli_common.resolve_path(args.out_dir, orig_cwd))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Sequence root: {seq_root}")

    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    if args.session:
        log.info(f"Sessions from --session: {session_ids}")
    else:
        log.info(f"Sessions from {args.split} split ({len(session_ids)}): {session_ids}")

    checkpoints = discover_checkpoints(args.checkpoint, orig_cwd)
    log.info(f"Checkpoints ({len(checkpoints)}): {[str(p) for p in checkpoints]}")
    smpl = None

    for ckpt_path in checkpoints:
        tag = checkpoint_tag(ckpt_path, orig_cwd)
        ckpt_out = out_dir / tag
        jobs = []
        for session_id in session_ids:
            try:
                seq_dir = session_dir(seq_root, session_id)
            except FileNotFoundError as exc:
                log.warning(str(exc))
                continue
            session_out = ckpt_out
            gen_path = cli_common.media_path(session_out, f"{session_id}_compare", args.gen)
            if cli_common.outputs_ready([gen_path]) and not args.force:
                log.info(f"Skip {tag}/{session_id}: already exists")
                continue
            jobs.append((session_id, seq_dir, session_out))
        if not jobs:
            log.info(f"Checkpoint {tag}: nothing to render")
            continue
        log.info(f"Checkpoint: {ckpt_path} -> {tag}")
        if smpl is None:
            smpl = smplx.create(smpl_model).to(device)
        model = load_model(str(ckpt_path), device)
        for session_id, seq_dir, session_out in jobs:
            log.info(f"Session {session_id} <- {seq_dir}")
            render_session(model, smpl, seq_dir, device, args, session_id, session_out)


if __name__ == "__main__":
    main()
