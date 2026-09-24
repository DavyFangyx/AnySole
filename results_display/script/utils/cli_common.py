"""Shared CLI conventions and media helpers for results_display scripts.

Every script under ``results_display/script/`` registers its common arguments
through :func:`add_common_args` and resolves workspace paths through
:func:`resolve_path`.  This keeps the Test1-Test5 visualization scripts
aligned on one CLI surface (see ``results_display/README.md``, "统一 CLI 约定").

Path schemes: ``results://``, ``workspace://``, ``display://``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]  # utils/ -> script/ -> results_display/ -> gait repo root
WORKSPACE_ROOT = Path(os.environ.get("ANYSOLE_WORKSPACE", REPO_ROOT / "AnysoleWorkspace")).expanduser()
RESULTS_ROOT = Path(os.environ.get("ANYSOLE_RESULTS", REPO_ROOT / "results")).expanduser()
DISPLAY_ROOT = Path(os.environ.get("ANYSOLE_RESULTSDISPLAY", REPO_ROOT / "results_display")).expanduser()

DEFAULT_FPS = 40.0
DEFAULT_STRIDE = 2
DEFAULT_SPLIT_CSV = WORKSPACE_ROOT / "splits/default/splits.csv"
DEFAULT_MANIFEST = WORKSPACE_ROOT / "manifests/session_manifest.csv"
DEFAULT_SEQ_ROOT = WORKSPACE_ROOT / "derived/MotionPRO/sequences/cam3"

PREFIXES = {
    "workspace://": WORKSPACE_ROOT,
    "results://": RESULTS_ROOT,
    "display://": DISPLAY_ROOT,
}

GEN_CHOICES = ("gif", "mp4")
GEN_DEFAULT = "gif"


def _resolve_raw_bvh(path: Path) -> Path:
    """Resolve BVHs recorded before the workspace was reorganized."""
    marker = "/mocap_ori_bvh/"
    normalized = path.as_posix()
    if marker not in normalized:
        return path
    suffix = normalized.split(marker, 1)[1]
    matches = sorted((WORKSPACE_ROOT / "sources/raw").glob("*/mocap_ori_bvh/" + suffix))
    return matches[0] if matches else path


def resolve_path(value, base_dir=None) -> Path:
    """Resolve a path that may use a ``results://``-style scheme.

    Equivalent to the public subset of the Baselines workspace resolvers
    (without their legacy prefixes, which no results_display script uses).
    """
    text = str(value or "")
    for prefix, root in PREFIXES.items():
        if text.startswith(prefix):
            return Path(root) / text[len(prefix):]
    normalized = text.replace("\\", "/").rstrip("/")
    path = Path(os.path.expandvars(text)).expanduser()
    if path.is_absolute():
        return _resolve_raw_bvh(path)
    if base_dir is None:
        candidate = WORKSPACE_ROOT / path
        if candidate.exists() or "/mocap_ori_bvh/" in candidate.as_posix():
            return _resolve_raw_bvh(candidate)
        return path
    return _resolve_raw_bvh(Path(base_dir) / path)


def split_csv_arg(value) -> list:
    """Split a comma-separated list argument, tolerating braces and semicolons."""
    text = str(value or "").strip()
    if not text:
        return []
    text = text.replace("{", " ").replace("}", " ").replace(";", ",")
    out = []
    for chunk in text.replace(" ", ",").split(","):
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def anysole_model_dir(model_name: str, contact_method: str, variant: str | None = None) -> str:
    """Results dir name: <model_name>_<contact_method>, plus the
    optional stacked-hyperparameter subdir (2026-09-22 naming: tw40 / tw40_st20 /
    t0003_tw40_lr3e4 — model+contact+variant == the ckpt address)."""
    base = f"{model_name}_{contact_method}"
    return f"{base}/{variant}" if variant else base


def load_test_sessions(session_arg, split_csv, split="test") -> list:
    """Return session ids: explicit list when given, else the split column.

    Raises RuntimeError when the split column is empty (instead of silently
    rendering nothing).
    """
    sessions = split_csv_arg(session_arg)
    if sessions:
        return sessions
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = []
    seen = set()
    for row in rows:
        sid = (row.get(split) or "").strip()
        if sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    if not ids:
        raise RuntimeError(f"No {split} sessions in {split_csv}")
    return ids


def load_evaluable_sessions(session_arg, split_csv, split="test", manifest_path=None) -> list:
    """Select canonical split sessions with at least one usable frame.

    ``splits.csv`` remains the sole split authority.  The manifest is used
    only for the data-eligibility gate: sessions whose ``valid_frame_indices``
    is explicitly empty (for example S12102) are reported as no-window
    sessions and are not sent to downstream render/evaluation jobs.
    """
    requested = load_test_sessions(session_arg, split_csv, split)
    manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
    if not manifest_path.is_file():
        return requested
    rows = {}
    if manifest_path.suffix.lower() in (".jsonl", ".json"):
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["session_id"])] = row
    else:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = {str(row["session_id"]): row for row in csv.DictReader(handle)}

    selected = []
    for session_id in requested:
        row = rows.get(session_id)
        if row is None:
            selected.append(session_id)
            continue
        raw_valid = row.get("valid_frame_indices", None)
        if raw_valid is not None:
            if isinstance(raw_valid, str):
                try:
                    valid = json.loads(raw_valid) if raw_valid else []
                except json.JSONDecodeError:
                    valid = []
            else:
                valid = raw_valid
            if not valid:
                continue
        selected.append(session_id)
    return selected


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    session: bool = True,
    split_csv: bool = True,
    split: bool = True,
    seq_root: bool = False,
    modal: bool = False,
    modal_default: str = "anysolev1,anysolev1_insole_drift",
    variant: bool = False,
    contact_method: bool = False,
    contact_method_default: str = "tactile_abs",
    config_id: bool = False,
    config_default: str = "VT2M,V2M,T2M",
    gen: bool = True,
    fps: bool = True,
    fps_default: float = DEFAULT_FPS,
    stride: bool = True,
    stride_default: int = DEFAULT_STRIDE,
    max_frames: bool = True,
    force: bool = True,
    out_dir: bool = True,
    out_dir_default=None,
) -> None:
    """Register the common CLI surface shared by all results_display scripts."""
    if session:
        parser.add_argument("--session", type=str, default="", help="Session ids, e.g. S14103,S14023. Empty uses the split column.")
    if split_csv:
        parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    if split:
        parser.add_argument("--split", type=str, default="test", help="Split column used when --session is empty.")
    if seq_root:
        parser.add_argument("--seq-root", type=str, default=str(DEFAULT_SEQ_ROOT), help="Centralized sequence root.")
    if modal:
        parser.add_argument(
            "--model-name", "--modal", dest="modal", metavar="MODEL_NAME", type=str, default=modal_default,
            help="AnySole model name(s), comma-separated (legacy alias: --modal).",
        )
    if variant:
        parser.add_argument("--variant", type=str, default=None, help="Stacked hyperparameter subdir under the model dir (e.g. tw40); model+contact+variant == the ckpt address.")
    if contact_method:
        parser.add_argument("--contact-method", type=str, default=contact_method_default, help="Contact-label scheme(s), comma-separated; model dir is <model-name>_<contact-method>.")
    if config_id:
        parser.add_argument("--config-id", type=str, default=config_default, help="Generation configuration(s), comma-separated.")
    if gen:
        parser.add_argument(
            "--gen",
            choices=GEN_CHOICES,
            default=GEN_DEFAULT,
            help=f"Output animation format (single choice, default: {GEN_DEFAULT}).",
        )
    if fps:
        help_text = "Output animation FPS."
        if fps_default == 0.0:
            help_text = "Output FPS; 0 = auto (source BVH frame rate)."
        parser.add_argument("--fps", type=float, default=fps_default, help=help_text)
    if stride:
        parser.add_argument("--stride", type=int, default=stride_default, help="Render every Nth frame.")
    if max_frames:
        parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    if force:
        parser.add_argument("--force", action="store_true", help="Rebuild and overwrite existing outputs.")
    if out_dir:
        parser.add_argument("--out-dir", type=str, default=str(out_dir_default or DISPLAY_ROOT), help="Output directory.")


def media_path(base, stem, fmt) -> Path:
    """Animated output path under the ``{fmt}/`` subdirectory (the one output-directory rule)."""
    return Path(base) / fmt / f"{stem}.{fmt}"


def outputs_ready(paths) -> bool:
    """True when every path exists and is non-empty (used for skip-without---force)."""
    paths = list(paths)
    return bool(paths) and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)


def viz_fps(fps, stride) -> float:
    return max(fps / max(stride, 1), 1.0)


def write_gif(frames, path, fps) -> None:
    duration_ms = max(int(round(1000.0 / max(fps, 1e-6))), 20)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)


def write_mp4(frames, path, fps) -> None:
    """Encode frames to an H.264 MP4 via ffmpeg (widely playable).

    OpenCV's ``mp4v`` writer emits MPEG-4 Part 2, which many modern players
    cannot open, and this env's OpenCV FFmpeg has no H.264 encoder.  Frames
    are piped to the system ffmpeg (libx264) instead.
    """
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; cannot write H.264 mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    ]
    if w % 2 or h % 2:  # yuv420p requires even dimensions
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame[:, :, ::-1]))  # RGB -> BGR
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode: {path}")
