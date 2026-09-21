"""Render predicted vs GT root trajectories (Test3), protocol-agnostic.

SMPL-protocol models (AnySole) provide the standard ``<session>_<config>.npz``
files written by ``anysole.eval`` (``predictions/eval_motion/``); they contain
the exact ``pred_pelvis_trans`` / ``gt_pelvis_trans`` arrays used by
``traj_ATE``.  BVH-protocol models (e.g. Step2Motion) provide ``*.bvh`` motion
files whose root-joint path is the trajectory; their GT is loaded from the
session's SMPL/BVH motion.  The file format is detected automatically, no
protocol flag is needed.  No model inference happens here.

The npz arrays are in the raw mocap world frame (y-up, meters).  They are
converted to the same z-up display frame used by the Test1 skeleton panels:
display ``(x, -z, y)``, i.e. z is vertical and x/y form the ground plane.

Outputs are written below ``results_display/Test3_trajectory/AnySole/<modal>/
<config>/`` for AnySole models, and below ``Test3_trajectory/<model>/gen/``
for baseline models discovered with ``--auto`` (or the
``ANYSOLE_RESULTSDISPLAY`` override), with gif/mp4/png kept in separate
folders.

Usage (run from the repository root):
    python results_display/script/visualize_anysole_traj.py
    python results_display/script/visualize_anysole_traj.py --modal anysolev1 --config-id VT2M
    python results_display/script/visualize_anysole_traj.py --auto
    python results_display/script/visualize_anysole_traj.py --session S7013
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

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402
from motion_io import load_motion, load_session_gt  # noqa: E402
from render_common import (  # noqa: E402
    INFO_FONT,
    TITLE_FONT,
    draw_text,
    session_dir,
)

PRED_ROOT = cli_common.RESULTS_ROOT / "AnySole"

PANEL = 480
AZIMUTH_DEG = 45.0
ELEVATION_DEG = 25.0
WINDOW_LENGTH = 20

GT_COLOR = (255, 170, 80)
PRED_COLOR = (80, 200, 255)
# Static-figure palette: validated reference pair (light surface), slots 1/2.
PRED_HEX = "#2a78d6"
GT_HEX = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


def to_display(trans: np.ndarray) -> np.ndarray:
    """Convert motion y-up meters to the z-up display frame (x, -z, y)."""
    t = np.asarray(trans, dtype=np.float64)
    return np.stack((t[:, 0], -t[:, 2], t[:, 1]), axis=1)


def load_traj(path: Path, seq_dir: Path | None = None, fps: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Load pred/GT trajectories, auto-detecting the motion format.

    SMPL NPZ keeps the eval-embedded pelvis trajectories (the exact source of
    the traj_ATE metric).  BVH-protocol models carry no embedded trajectory:
    the root-joint path is used as the prediction, and the GT comes from the
    session's SMPL (BVH fallback) motion.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        pred_key = "pred_pelvis_trans" if "pred_pelvis_trans" in data else (
            "pred_trans_world" if "pred_trans_world" in data else "trans"
        )
        gt_key = "gt_pelvis_trans" if "gt_pelvis_trans" in data else (
            "gt_trans_world" if "gt_trans_world" in data else "gt_trans"
        )
        pred = to_display(data[pred_key])
        gt = to_display(data[gt_key])
        n = min(pred.shape[0], gt.shape[0])
        if pred.shape[0] != gt.shape[0]:
            log.warning(f"{path.name}: pred {pred.shape[0]} frames vs gt {gt.shape[0]}, using first {n}")
        return pred[:n], gt[:n]
    # BVH: load_motion returns z-up display meters for both protocols.
    pred = load_motion(path)["joints"][:, 0]
    if seq_dir is None:
        raise ValueError(f"BVH trajectory rendering needs the session dir for GT: {path}")
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    gt = load_session_gt(seq_dir, min(pred.shape[0], int(meta["n_frames"])), fps)["joints"][:, 0]
    n = min(pred.shape[0], gt.shape[0])
    return pred[:n], gt[:n]


def find_traj_file(pred_root: Path, session_id: str, config_id: str) -> Path | None:
    """Locate one model's motion file for a session, npz (SMPL) before bvh."""
    if config_id:
        for cand in (
            pred_root / "eval_motion" / f"{session_id}_{config_id}.npz",
            pred_root / "eval_bvh" / f"{session_id}_{config_id}.bvh",
        ):
            if cand.is_file():
                return cand
    # Baseline layouts (e.g. Step2Motion predictions/<run>/<session>_gen.bvh);
    # stats/traj sidecars are not motions.
    matches = sorted(
        p for p in pred_root.rglob(f"*{session_id}*")
        if p.suffix.lower() in (".npz", ".bvh")
        and "_stats" not in p.name and "_traj" not in p.name
    )
    return matches[0] if matches else None


def discover_baseline_dirs(results_root: Path) -> list[Path]:
    """Self-contained non-AnySole model dirs under results/ (Test2 口径).

    Archived ``*_backup*`` trees are excluded; AnySole models are addressed
    through ``--modal``.
    """
    found = []
    for root in sorted(p for p in results_root.rglob("*") if p.is_dir()):
        if root.name in {"checkpoints", "predictions", "metrics", "logs", "tensorboard"}:
            continue
        rel = root.relative_to(results_root)
        if any("backup" in part.lower() for part in rel.parts) or "AnySole" in rel.parts:
            continue
        if (root / "predictions").is_dir():
            found.append(rel)
    return found


class TrajProjector:
    """Fixed oblique orthographic camera (azimuth/elevation) fit to the data."""

    def __init__(self, pts_list: list[np.ndarray], size: int = PANEL, margin: int = 46):
        all_pts = np.concatenate(pts_list, axis=0)
        lo = all_pts.min(axis=0).copy()
        hi = all_pts.max(axis=0).copy()
        lo[2] = min(lo[2], 0.0)  # always include the ground plane
        # Minimum extent keeps near-stationary sessions readable.
        span = np.maximum(hi - lo, np.asarray([0.3, 0.3, 0.05], dtype=np.float64))
        hi = lo + span
        self.lo, self.hi = lo, hi
        corners = self._corners(lo, hi)
        uv = self._project(corners)
        extent_u = float(uv[:, 0].max() - uv[:, 0].min())
        extent_v = float(uv[:, 1].max() - uv[:, 1].min())
        self.scale = (size - 2 * margin) / max(extent_u, extent_v, 1e-6)
        # u = cx + sx * scale, v = cy - sy * scale; center the corner bbox.
        self.cx = size / 2.0 - float(uv[:, 0].min() + uv[:, 0].max()) / 2.0 * self.scale
        self.cy = size / 2.0 + float(uv[:, 1].min() + uv[:, 1].max()) / 2.0 * self.scale
        # Ground grid segments (drawn once, reused every frame).
        self.grid_segments = self._grid_segments(lo, hi)
        # Horizontal scale bar: 0.5 m for small paths, 1.0 m otherwise.
        self.bar_len = 0.5 if float(np.max(hi[:2] - lo[:2])) < 3.0 else 1.0
        self.size = size

    @staticmethod
    def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        pts = [
            (lo[0], lo[1], 0.0), (hi[0], lo[1], 0.0),
            (lo[0], hi[1], 0.0), (hi[0], hi[1], 0.0),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
            (lo[0], hi[1], hi[2]), (hi[0], hi[1], hi[2]),
        ]
        return np.asarray(pts, dtype=np.float64)

    @staticmethod
    def _project(pts: np.ndarray) -> np.ndarray:
        az = np.deg2rad(AZIMUTH_DEG)
        el = np.deg2rad(ELEVATION_DEG)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x1 = np.cos(az) * x + np.sin(az) * y
        y1 = -np.sin(az) * x + np.cos(az) * y
        sx = x1
        sy = np.cos(el) * z - np.sin(el) * y1
        return np.stack([sx, sy], axis=1)

    def uv(self, pts: np.ndarray) -> np.ndarray:
        sx, sy = self._project(np.asarray(pts, dtype=np.float64)).T
        u = self.cx + sx * self.scale
        v = self.cy - sy * self.scale
        return np.stack([u, v], axis=1)

    def _grid_segments(self, lo: np.ndarray, hi: np.ndarray) -> list[tuple[float, float, float, float]]:
        segments = []
        xs = np.arange(np.ceil(lo[0] / 0.5) * 0.5, hi[0] + 1e-9, 0.5)
        ys = np.arange(np.ceil(lo[1] / 0.5) * 0.5, hi[1] + 1e-9, 0.5)
        for x in xs:
            a = self.uv(np.asarray([[x, lo[1], 0.0], [x, hi[1], 0.0]]))
            segments.append((a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
        for y in ys:
            a = self.uv(np.asarray([[lo[0], y, 0.0], [hi[0], y, 0.0]]))
            segments.append((a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
        return segments


def render_traj_panel(
    projector: TrajProjector,
    pred: np.ndarray,
    gt: np.ndarray,
    frame_idx: int,
    session_id: str,
    config_id: str,
    n_frames: int,
    fps: float,
    ate_mean_mm: float,
) -> np.ndarray:
    """One animation frame: GT path (full) + predicted path growing to frame_idx."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (PANEL, PANEL), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(48, 48, 56))
    draw_text(draw, (PANEL // 2, 12), "%s  %s" % (session_id, config_id), TITLE_FONT, fill=(230, 230, 235), anchor="mt")

    for x0, y0, x1, y1 in projector.grid_segments:
        draw.line([(x0, y0), (x1, y1)], fill=(42, 46, 56), width=1)

    # Ground-plane scale bar, drawn at a fixed screen position.
    bar_len_px = projector.bar_len * projector.scale
    draw.line([(30, PANEL - 30), (30 + bar_len_px, PANEL - 30)], fill=(120, 126, 138), width=3)
    draw_text(draw, (34 + bar_len_px, PANEL - 42), "%.1f m" % projector.bar_len, INFO_FONT, fill=(150, 154, 166))

    # GT path: full sequence.
    gt_uv = projector.uv(gt)
    draw.line([tuple(p) for p in gt_uv], fill=GT_COLOR, width=3)
    # Predicted path: grows with the animation.
    pred_uv = projector.uv(pred[: frame_idx + 1])
    if len(pred_uv) >= 2:
        draw.line([tuple(p) for p in pred_uv], fill=PRED_COLOR, width=3)
    # Window-boundary ticks on the predicted path (re-anchored every window).
    for w in range(WINDOW_LENGTH, frame_idx + 1, WINDOW_LENGTH):
        u, v = pred_uv[w]
        draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(200, 210, 220))

    # Vertical stem from the current predicted point down to the ground plane.
    p_now = pred[frame_idx]
    stem = projector.uv(np.asarray([p_now, [p_now[0], p_now[1], 0.0]]))
    draw.line([tuple(stem[0]), tuple(stem[1])], fill=(46, 96, 128), width=1)

    # Start/end markers on the GT path.
    for idx, label in ((0, "start"), (len(gt) - 1, "end")):
        u, v = gt_uv[idx]
        if idx == 0:
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(235, 235, 235), width=2)
        else:
            draw.line([(u - 5, v - 5), (u + 5, v + 5)], fill=(235, 235, 235), width=2)
            draw.line([(u - 5, v + 5), (u + 5, v - 5)], fill=(235, 235, 235), width=2)
        draw_text(draw, (u + 8, v - 10), label, INFO_FONT, fill=(180, 180, 190))

    # Current positions.
    u, v = gt_uv[frame_idx]
    draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=GT_COLOR)
    u, v = pred_uv[frame_idx]
    draw.ellipse([u - 6, v - 6, u + 6, v + 6], fill=PRED_COLOR, outline=(235, 235, 235), width=2)

    # Legend.
    lx, ly = PANEL - 118, 46
    draw.line([(lx, ly), (lx + 22, ly)], fill=GT_COLOR, width=3)
    draw_text(draw, (lx + 28, ly), "GT", INFO_FONT, fill=(230, 230, 235), anchor="lm")
    draw.line([(lx, ly + 24), (lx + 22, ly + 24)], fill=PRED_COLOR, width=3)
    draw_text(draw, (lx + 28, ly + 24), "Pred", INFO_FONT, fill=(230, 230, 235), anchor="lm")

    ate_t_mm = float(np.linalg.norm(pred[frame_idx] - gt[frame_idx])) * 1000.0
    footer = "t=%.2fs  frame %d/%d  ATE@t %5.0f mm  ATE %5.0f mm" % (
        frame_idx / fps if fps else 0.0,
        frame_idx,
        n_frames - 1,
        ate_t_mm,
        ate_mean_mm,
    )
    draw_text(draw, (PANEL // 2, PANEL - 14), footer, INFO_FONT, fill=(180, 180, 190), anchor="mb")
    return np.asarray(canvas)


def render_traj_figure(
    pred: np.ndarray,
    gt: np.ndarray,
    session_id: str,
    config_id: str,
    ate_mean_mm: float,
    png_path: Path,
    fps: float = 40.0,
) -> None:
    """Static paper figure: 3D oblique view, top-down view, and height vs time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(gt.shape[0], dtype=np.float64) / float(fps)
    pts = np.concatenate([pred, gt], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lo[2] = min(lo[2], 0.0)
    span = np.maximum(hi - lo, np.asarray([0.3, 0.3, 0.05]))
    # Keep the 3D box tall enough that hip height stays legible on long paths.
    span[2] = max(span[2], 0.4 * max(span[0], span[1]))

    fig = plt.figure(figsize=(15.5, 5.4), dpi=100)
    fig.suptitle(
        "%s  %s   traj ATE %.0f mm" % (session_id, config_id, ate_mean_mm),
        fontsize=13, color=INK, y=1.0,
    )

    def style_2d(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#c8c8c4", linewidth=0.6, alpha=0.4)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c8c8c4")

    def start_end(ax, x, y, z=None):
        kw = dict(marker="o", s=46, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=5)
        if z is None:
            ax.scatter(x[:1], y[:1], **kw)
            ax.scatter(x[-1:], y[-1:], marker="X", s=60, color=INK, zorder=5)
        else:
            ax.scatter(x[:1], y[:1], z[:1], **kw)
            ax.scatter(x[-1:], y[-1:], z[-1:], marker="X", s=60, color=INK, zorder=5)

    # 3D oblique view.
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=GT_HEX, lw=2, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=PRED_HEX, lw=2, label="Pred")
    xs = np.arange(np.ceil(lo[0] / 0.5) * 0.5, hi[0] + 1e-9, 0.5)
    ys = np.arange(np.ceil(lo[1] / 0.5) * 0.5, hi[1] + 1e-9, 0.5)
    for x in xs:
        ax.plot([x, x], [lo[1], hi[1]], [0, 0], color="#c8c8c4", lw=0.6, alpha=0.5, zorder=0)
    for y in ys:
        ax.plot([lo[0], hi[0]], [y, y], [0, 0], color="#c8c8c4", lw=0.6, alpha=0.5, zorder=0)
    start_end(ax, gt[:, 0], gt[:, 1], gt[:, 2])
    ax.set_xlim(lo[0], lo[0] + span[0])
    ax.set_ylim(lo[1], lo[1] + span[1])
    ax.set_zlim(lo[2], lo[2] + span[2])
    ax.set_box_aspect((span[0], span[1], span[2]))
    ax.view_init(elev=ELEVATION_DEG, azim=AZIMUTH_DEG)
    ax.set_xlabel("x (m)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("y (m)", fontsize=9, color=INK_SECONDARY)
    ax.set_zlabel("z (m)", fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, pad=2)
    ax.set_title("3D view", fontsize=10, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)

    # Top-down view.
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gt[:, 0], gt[:, 1], color=GT_HEX, lw=2)
    ax.plot(pred[:, 0], pred[:, 1], color=PRED_HEX, lw=2)
    start_end(ax, gt[:, 0], gt[:, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("y (m)", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Top-down (x-y)", fontsize=10, color=INK)
    style_2d(ax)

    # Height over time.
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, gt[:, 2], color=GT_HEX, lw=2)
    ax.plot(t, pred[:, 2], color=PRED_HEX, lw=2)
    start_end(ax, t, gt[:, 2])
    ax.annotate("GT", (t[-1], gt[-1, 2]), xytext=(4, 6), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.annotate("Pred", (t[-1], pred[-1, 2]), xytext=(4, -12), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("t (s)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("z (m)", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Height vs time", fontsize=10, color=INK)
    style_2d(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_session(traj_path: Path, session_id: str, config_id: str, seq_dir: Path | None, args: argparse.Namespace, session_out: Path):
    stem = f"{session_id}_{config_id}" if config_id else session_id
    paths = {args.gen: cli_common.media_path(session_out, f"{stem}_traj", args.gen)}
    if not args.no_png:
        paths["png"] = session_out / "png" / f"{stem}_traj.png"
    if cli_common.outputs_ready(paths.values()) and not args.force:
        log.info(f"Skip {session_id}_{config_id}: already exists under {session_out}")
        return "skip"

    pred, gt = load_traj(traj_path, seq_dir, args.fps)
    n = pred.shape[0]
    ate_mm = np.linalg.norm(pred - gt, axis=1) * 1000.0
    ate_mean_mm = float(ate_mm.mean())

    frame_ids = list(range(0, n, max(args.stride, 1)))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]

    projector = TrajProjector([pred, gt])
    gen_path = paths[args.gen]
    frames = [
        render_traj_panel(projector, pred, gt, t, session_id, config_id, n, args.fps, ate_mean_mm)
        for t in frame_ids
    ]
    if not frames:
        raise RuntimeError(f"No frames rendered for {session_id}_{config_id}")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = cli_common.viz_fps(args.fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, gen_path, frame_fps)
    else:
        cli_common.write_mp4(frames, gen_path, frame_fps)
    log.info(f"Wrote {gen_path}")
    if not args.no_png:
        render_traj_figure(pred, gt, session_id, config_id, ate_mean_mm, paths["png"], args.fps)
        log.info(f"Wrote {paths['png']}")
    log.info(f"{session_id}_{config_id}: frames={n} traj_ATE={ate_mean_mm:.1f}mm")
    return "write"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize root trajectories (pred vs GT) as Test3 outputs, SMPL/BVH auto-detected.")
    cli_common.add_common_args(parser, seq_root=True, modal=True, contact_method=True, config_id=True, out_dir_default=cli_common.DISPLAY_ROOT / "Test3_trajectory" / "AnySole")
    parser.add_argument("--no-png", action="store_true", help="Skip the static per-session figure (not controlled by --gen).")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Also scan results/ for self-contained non-AnySole model dirs "
        "(e.g. Step2Motion predictions/<run>/<session>_gen.bvh); *_backup* trees are excluded.",
    )
    return parser.parse_args()


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

    # Jobs: (pred_root, session_out_root, config_id); config_id "" = baseline.
    jobs: list[tuple[Path, Path, str]] = []
    for modal in cli_common.split_csv_arg(args.modal):
        for contact_method in cli_common.split_csv_arg(args.contact_method):
            model_dir = cli_common.anysole_model_dir(modal, contact_method)
            for config_id in cli_common.split_csv_arg(args.config_id):
                jobs.append((PRED_ROOT / model_dir / "predictions", out_dir / model_dir / config_id, config_id))
    if args.auto:
        for rel in discover_baseline_dirs(cli_common.RESULTS_ROOT):
            jobs.append((cli_common.RESULTS_ROOT / rel / "predictions", out_dir.parent / rel / "gen", "gen"))
            log.info(f"Discovered baseline model: {rel}")

    for pred_root, session_out, config_id in jobs:
        if not pred_root.is_dir():
            log.warning(f"No prediction directory: {pred_root}")
            continue
        for session_id in session_ids:
            traj_path = find_traj_file(pred_root, session_id, config_id)
            if traj_path is None:
                log.warning(f"Skip {session_out.parent.name}/{session_out.name}/{session_id}: no SMPL/BVH motion file (re-run eval or the baseline)")
                continue
            seq_dir = None
            if traj_path.suffix.lower() == ".bvh":
                try:
                    seq_dir = session_dir(seq_root, session_id)
                except FileNotFoundError as exc:
                    log.warning(str(exc))
                    continue
            try:
                render_session(traj_path, session_id, config_id, seq_dir, args, session_out)
            except Exception as exc:  # one bad session must not stop the sweep
                log.error(f"{session_id}_{config_id}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
