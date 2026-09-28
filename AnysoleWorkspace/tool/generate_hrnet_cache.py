#!/usr/bin/env python3
"""Generate the AnySole HRNet visual cache through the model's own extractor.

Wraps ``python -m anysole.data.extract_hrnet`` with the AnySole input contract
of the workspace layout (2026-09-28):

    seq root    AnysoleWorkspace/shared/facts/sessions/cam3   (canonical rgb)
    bbox root   AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox
                (written by tool/adapters/AnySole/build_hrnet_bbox.py)
    cache root  AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/cam3
                -> <cache root>/<session>.pt, the path
                   anysole/data/dataset.hrnet_cache_path() reads

The extractor enumerates sessions itself with ``<seq root>/*/*/*`` (date /
subject / session), so one invocation covers many sessions and pays the CLIFF
checkpoint load once.  ``--shard I/N`` splits the selected sessions across
processes; it loops one session per subprocess (the extractor takes a single
``--session``), so every shard session costs one encoder load -- for parallel
runs prefer sharding by ``--split`` or by shard where the load cost is small
relative to the frames.

Examples:
    # whole canonical set on one idle GPU (resumable)
    CUDA_VISIBLE_DEVICES=5 python AnysoleWorkspace/tool/generate_hrnet_cache.py \
        --cam-id 3 --split all --skip-existing --device cuda

    # 4-way parallel sharding (one shard per idle GPU)
    CUDA_VISIBLE_DEVICES=5 python ... --cam-id 3 --split all --shard 0/4 --device cuda
    CUDA_VISIBLE_DEVICES=6 python ... --cam-id 3 --split all --shard 1/4 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# AnySole is the project root package after the repository layout migration.
ANYSOLE_ROOT = REPO_ROOT

WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
DEFAULT_SEQ_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
DEFAULT_BBOX_ROOT = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet/bbox"
DEFAULT_CACHE_ROOT = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet/cam3"
DEFAULT_SPLIT_CSV = WORKSPACE / "protocol/splits/default/splits.csv"


def split_session_ids(split_csv: Path, split: str) -> list:
    """Session ids of one splits.csv column, in file order (no duplicates)."""
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and split not in rows[0]:
        raise ValueError("%s missing column %s" % (split_csv, split))
    seen, ids = set(), []
    for row in rows:
        value = (row.get(split) or "").strip()
        if value and value not in seen:
            seen.add(value)
            ids.append(value)
    return ids


def all_session_ids(split_csv: Path) -> list:
    """Union of train/val/test, in splits.csv order (val == test today)."""
    out = []
    for split in ("train", "val", "test"):
        for session_id in split_session_ids(split_csv, split):
            if session_id not in out:
                out.append(session_id)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AnySole HRNet feature cache.")
    parser.add_argument("--cam-id", type=int, required=True, help="Camera id, for example 3.")
    parser.add_argument("--session", help="Process only one session.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"],
                        help="Split column scope when --session is empty ('all' = every session under the seq root).")
    parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT_CSV),
                        help="splits.csv path used by --split.")
    parser.add_argument("--shard", type=str, default=None, metavar="I/N",
                        help="Process only shard I of N selected sessions (I starts at 0).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument("--seq-root", type=Path, default=DEFAULT_SEQ_ROOT,
                        help="Shared-facts camera root (<date>/<subject>/<session>/rgb).")
    parser.add_argument("--bbox-root", type=Path, default=DEFAULT_BBOX_ROOT,
                        help="Per-session bbox.npy root (AnySole bbox shim).")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
                        help="Feature cache root; files land at <cache-root>/<session>.pt.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions whose cache exists.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true", help="Recompute and overwrite existing caches.")
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    return parser.parse_args()


def parse_shard(value: str):
    try:
        index, count = (int(part) for part in str(value).split("/", 1))
    except ValueError:
        raise SystemExit("--shard expects I/N, got %r" % value)
    if count < 1 or not 0 <= index < count:
        raise SystemExit("--shard I/N requires 0 <= I < N, got %r" % value)
    return index, count


def base_command(args: argparse.Namespace) -> list:
    return [
        sys.executable,
        "-m",
        "anysole.data.extract_hrnet",
        "--cam-id",
        str(args.cam_id),
        "--batch-size",
        str(args.batch_size),
        "--cache-root",
        str(args.cache_root),
        "--seq-root",
        str(args.seq_root),
        "--bbox-root",
        str(args.bbox_root),
    ]


def main() -> int:
    args = parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ANYSOLE_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    commands = []
    if args.shard is None:
        command = base_command(args)
        if args.session:
            command += ["--session", args.session]
        elif args.split != "all":
            command += ["--split", args.split, "--split-csv", str(args.split_csv)]
        if args.device:
            command += ["--device", args.device]
        command.append("--force" if args.force else "--skip-existing")
        if args.limit_sessions is not None:
            command += ["--limit-sessions", str(args.limit_sessions)]
        commands.append(command)
    else:
        # The extractor takes a single --session per run, so a shard is a
        # short loop; each session pays one CLIFF encoder load.
        index, count = parse_shard(args.shard)
        if args.session:
            session_ids = [args.session]
        elif args.split == "all":
            session_ids = all_session_ids(Path(args.split_csv))
        else:
            session_ids = split_session_ids(Path(args.split_csv), args.split)
        shard = [sid for position, sid in enumerate(session_ids) if position % count == index]
        if not shard:
            raise SystemExit("shard %s of %s is empty (%d sessions selected)"
                             % (args.shard, args.split, len(session_ids)))
        for session_id in shard:
            command = base_command(args) + ["--session", session_id]
            if args.device:
                command += ["--device", args.device]
            command.append("--force" if args.force else "--skip-existing")
            commands.append(command)
        print("shard %s: %d/%d sessions: %s" % (args.shard, len(shard), len(session_ids), " ".join(shard)))

    for command in commands:
        print("run:", " ".join(command))
        if args.dry_run:
            continue
        status = subprocess.call(command, cwd=ANYSOLE_ROOT, env=env)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
