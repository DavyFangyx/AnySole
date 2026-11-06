"""Build contiguous visual-stream gap caches from an existing cam3 HRNet cache.

E5 temporal axis (``z_note/E5_视觉退化扩展说明.md`` §2.2): the model receives
the fixed VT input and never knows that the visual stream dropped, so the
degradation lives entirely in the visual feature cache.  No image is read and
no HRNet is run -- this is a pure cache rewrite.

Degradation = freeze: every gap frame copies the full 2051-d feature row
(HRNet 2048 + bbox_info 3) of the most recent live frame.  Freezing was chosen
over zeroing because a zeroed V row resembles the training-time null-V state
(the model can recognise it), while a frozen row is stale-but-plausible
evidence.

Gap pattern: cycle C = 100 frames (2.5 s at 40 Hz).  The last
round(rho * C) frames of each cycle are gap frames, the rest are live.
Each session draws a phase offset in 0..C-1 from
``np.random.default_rng((seed, crc32(session_id)))`` so the gaps do not align
with the 20-frame window grid; frame 0 is forced live so every session has an
anchor.  rho = 0 is an identity build (useful as a bitwise gate).
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from anysole.data.extract_hrnet import split_session_ids
from anysole.data.workspace_adapter import load_shared_session, write_cache_provenance
from anysole.types import HRNET_CACHE_ROOT, V_FEAT_DIM

CYCLE_DEFAULT = 100
SEED_DEFAULT = 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contiguous visual-stream gap caches (freeze) from cam3 HRNet features."
    )
    parser.add_argument("--cam-id", type=int, default=3)
    parser.add_argument("--rho", type=float, required=True,
                        help="Gap ratio in percent (0-100): the last round(rho/100*cycle) "
                             "frames of every cycle are frozen. E5 levels = 10/20/40/60/80.")
    parser.add_argument("--cycle", type=int, default=CYCLE_DEFAULT,
                        help="Cycle length in frames (default 100 = 2.5 s at 40 Hz).")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT,
                        help="Phase-offset seed (default 0; fixed by the E5 design).")
    parser.add_argument("--session", type=str, default=None, help="Only build this session id.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"],
                        help="Split column scope when --session is empty ('all' = every cached session).")
    parser.add_argument("--split-csv", type=str, default=None,
                        help="splits.csv path used by --split "
                             "(default = the same protocol splits.csv as extract_hrnet).")
    parser.add_argument("--limit-sessions", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions whose cache exists.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true",
                      help="Recompute and overwrite existing caches.")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="Input cam3 HRNet cache root (default: HRNET_CACHE_ROOT).")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Output cache root (default: <hrnet>/cam{id}_gap_r{rho:03d}).")
    return parser.parse_args(argv)


def gap_cache_dirname(cam_id: int, rho_percent: float) -> str:
    """Cache dir name for one gap level: cam{id}_gap_r{RRR} with
    RRR = round(rho_percent) zero-padded (20 -> cam3_gap_r020)."""
    return "cam%d_gap_r%03d" % (int(cam_id), int(round(rho_percent)))


def resolve_out_root(cam_id: int, rho_percent: float,
                     cache_root: Optional[Path], out_root: Optional[Path]) -> Path:
    if out_root is not None:
        return Path(out_root)
    base = (Path(cache_root) if cache_root is not None else HRNET_CACHE_ROOT).parent
    return base / gap_cache_dirname(cam_id, rho_percent)


def gap_length(rho_percent: float, cycle: int) -> int:
    if not 0.0 <= float(rho_percent) <= 100.0:
        raise ValueError("--rho must be in [0, 100], got %s" % rho_percent)
    if int(cycle) <= 0:
        raise ValueError("--cycle must be positive, got %s" % cycle)
    return int(round(float(rho_percent) / 100.0 * int(cycle)))


def session_phase(session_id: str, cycle: int, seed: int) -> int:
    """Deterministic per-session phase offset in 0..cycle-1.

    Python's builtin hash() is salted per process, so use crc32 for a stable
    session hash (same phase across every rebuild).
    """
    key = (int(seed), int(zlib.crc32(session_id.encode("utf-8"))))
    return int(np.random.default_rng(key).integers(0, int(cycle), dtype=np.int64))


def build_gap_array(feat: np.ndarray, rho_percent: float, cycle: int,
                    seed: int, session_id: str) -> np.ndarray:
    """Freeze the configured gap frames to their most recent live frame.

    Frame 0 is forced live.  Every frozen row is a bitwise copy of its anchor
    row (full 2051-d feature incl. bbox_info).
    """
    feat = np.asarray(feat)
    n_frames = int(feat.shape[0])
    if n_frames == 0:
        raise ValueError("%s: empty cache" % session_id)
    gap_len = gap_length(rho_percent, cycle)
    if gap_len == 0:
        return feat.copy()

    offset = session_phase(session_id, cycle, seed)
    pos = (np.arange(n_frames, dtype=np.int64) + offset) % int(cycle)
    is_gap = pos >= (int(cycle) - gap_len)
    is_gap[0] = False  # frame 0 forced live -> always has an anchor

    # anchor[i] = index of the most recent live frame at or before i
    anchors = np.where(~is_gap, np.arange(n_frames, dtype=np.int64), -1)
    np.maximum.accumulate(anchors, out=anchors)
    return feat[anchors]


def session_ids_for_scope(cache_root: Path, args: argparse.Namespace) -> List[str]:
    """Cached session ids for the requested scope (explicit --session wins)."""
    cached = {path.stem for path in Path(cache_root).glob("*.pt")}
    if args.session:
        if args.session not in cached:
            raise FileNotFoundError("Missing cam3 cache for session %s under %s"
                                    % (args.session, cache_root))
        return [args.session]
    if args.split == "all":
        return sorted(cached)
    if not args.split_csv:
        from anysole.types import WORKSPACE_ROOT
        split_csv = WORKSPACE_ROOT / "protocol/splits/default/splits.csv"
    else:
        split_csv = Path(args.split_csv)
    wanted = split_session_ids(split_csv, args.split)
    missing = sorted(wanted - cached)
    if missing:
        raise FileNotFoundError(
            "%d %s session(s) have no cam3 cache under %s, e.g. %s"
            % (len(missing), args.split, cache_root, missing[0]))
    return sorted(wanted)


def load_cache(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    feat = value.float().contiguous()
    if feat.ndim != 2 or feat.shape[1] != V_FEAT_DIM:
        raise ValueError("%s shape %s != (n, %d)" % (path, tuple(feat.shape), V_FEAT_DIM))
    return feat


def build_session(cache_root: Path, out_root: Path, session_id: str,
                  rho_percent: float, cycle: int, seed: int) -> dict:
    src_path = Path(cache_root) / ("%s.pt" % session_id)
    dst_path = Path(out_root) / ("%s.pt" % session_id)
    feat = load_cache(src_path)
    out = build_gap_array(feat.numpy(), rho_percent, cycle, seed, session_id)
    if out.shape != feat.shape:
        raise ValueError("%s output shape %s != input %s"
                         % (session_id, out.shape, tuple(feat.shape)))

    gap_len = gap_length(rho_percent, cycle)
    offset = session_phase(session_id, cycle, seed) if gap_len else 0
    pos = (np.arange(out.shape[0], dtype=np.int64) + offset) % int(cycle)
    is_gap = pos >= (int(cycle) - gap_len) if gap_len else np.zeros(out.shape[0], dtype=bool)
    is_gap[0] = False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(np.ascontiguousarray(out)), dst_path)
    write_cache_provenance(
        dst_path,
        session=load_shared_session(session_id),
        kind="stream_gap",
        shape=tuple(out.shape),
        parameters={
            "kind": "stream_gap",
            "rho_percent": float(rho_percent),
            "cycle": int(cycle),
            "seed": int(seed),
            "freeze": True,
            "phase_offset": int(offset),
            "source_cache": str(src_path),
        },
    )
    frozen_fraction = float(is_gap.mean()) if is_gap.size else 0.0
    print("saved %s %s frozen=%.2f%% (target %.2f%%) phase=%d"
          % (dst_path, tuple(out.shape), 100.0 * frozen_fraction,
             float(rho_percent), offset))
    return {"session_id": session_id, "shape": tuple(out.shape),
            "frozen_fraction": frozen_fraction, "phase_offset": offset}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.cache_root is not None:
        cache_root = Path(args.cache_root)
    elif int(args.cam_id) == 3:
        cache_root = HRNET_CACHE_ROOT
    else:
        cache_root = HRNET_CACHE_ROOT.parent / ("cam%d" % int(args.cam_id))
    if not cache_root.is_dir():
        raise FileNotFoundError("Missing cam3 cache root %s" % cache_root)
    out_root = resolve_out_root(args.cam_id, args.rho, args.cache_root, args.out_root)
    session_ids = session_ids_for_scope(cache_root, args)
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = session_ids[: args.limit_sessions]
    if not session_ids:
        raise FileNotFoundError("No sessions match split=%s under %s"
                                % (args.split, cache_root))
    if session_ids == sorted({p.stem for p in cache_root.glob("*.pt")}) \
            and args.session is None and args.limit_sessions is None:
        print("full cache scope: %d session(s)" % len(session_ids))

    built, skipped = 0, 0
    summaries = []
    for session_id in session_ids:
        dst_path = out_root / ("%s.pt" % session_id)
        if dst_path.is_file() and not args.force:
            skipped += 1
            continue
        summaries.append(build_session(cache_root, out_root, session_id,
                                       args.rho, args.cycle, args.seed))
        built += 1
    print("built %d session(s), skipped %d -> %s" % (built, skipped, out_root))
    if summaries:
        frac = float(np.mean([item["frozen_fraction"] for item in summaries]))
        print("mean frozen fraction %.2f%% (target %.2f%%)"
              % (100.0 * frac, float(args.rho)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
