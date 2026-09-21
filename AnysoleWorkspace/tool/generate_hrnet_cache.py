#!/usr/bin/env python3
"""Generate AnySole HRNet features through the model's existing extractor."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# AnySole is the project root package after the repository layout migration.
ANYSOLE_ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AnySole HRNet feature cache.")
    parser.add_argument("--cam-id", type=int, required=True, help="Camera id, for example 3.")
    parser.add_argument("--session", help="Process only one session.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions whose cache exists.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true", help="Recompute and overwrite existing caches.")
    parser.add_argument("--limit-sessions", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "anysole.data.extract_hrnet",
        "--cam-id",
        str(args.cam_id),
        "--batch-size",
        str(args.batch_size),
        "--cache-root",
        str(REPO_ROOT / "AnysoleWorkspace" / "derived" / "AnySole" / "hrnet_cache" / ("cam%d" % args.cam_id)),
        "--seq-root",
        str(REPO_ROOT / "AnysoleWorkspace" / "derived" / "MotionPRO" / "sequences" / ("cam%d" % args.cam_id)),
    ]
    if args.session:
        command += ["--session", args.session]
    if args.device:
        command += ["--device", args.device]
    command.append("--force" if args.force else "--skip-existing")
    if args.limit_sessions is not None:
        command += ["--limit-sessions", str(args.limit_sessions)]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ANYSOLE_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.call(command, cwd=ANYSOLE_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
