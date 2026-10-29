"""Operator CLI for the compact Fast SAM3DB baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import DATASET_ROOT, DEFAULT_FPS, RESULT_ROOT
from .pipeline import audit, parse_gpus, run, status
from .prepare import prepare


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("selection cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam3db-baseline", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="build deterministic hidden controls")
    prep.add_argument("--dataset-root", default=str(DATASET_ROOT))
    prep.add_argument("--result-root", default=str(RESULT_ROOT))
    prep.add_argument("--dates", type=_csv)
    prep.add_argument("--methods", type=_csv)
    prep.add_argument("--sessions", type=_csv)
    prep.add_argument("--cameras", type=_csv)
    prep.add_argument("--max-frames-per-unit", type=int)

    launch = commands.add_parser("run", help="run/resume autonomous workers")
    launch.add_argument("--result-root", default=str(RESULT_ROOT))
    launch.add_argument("--manifest")
    launch.add_argument("--gpus", type=parse_gpus, default=(0,))
    launch.add_argument("--workers-per-gpu", type=int, choices=(1, 2), default=1)
    launch.add_argument("--retries", type=int, default=1)
    launch.add_argument("--fps", type=float, default=DEFAULT_FPS)
    launch.add_argument("--full", action="store_true")
    launch.add_argument("--authorization")
    launch.add_argument("--force", action="store_true")
    launch.add_argument("--skip-preflight", action="store_true")

    for name in ("status", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--result-root", default=str(RESULT_ROOT))
        command.add_argument("--manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(dataset_root=args.dataset_root, result_root=args.result_root,
                         dates=args.dates, methods=args.methods, sessions=args.sessions,
                         cameras=args.cameras, max_frames_per_unit=args.max_frames_per_unit)
        output = {"ready": result["ready"], "selection": result["manifest"]["selection"]}
    elif args.command == "run":
        output = run(result_root=args.result_root, manifest_path=args.manifest, gpus=args.gpus,
                     workers_per_gpu=args.workers_per_gpu,
                     retries=args.retries, fps=args.fps, full=args.full,
                     authorization=args.authorization, force=args.force,
                     preflight=not args.skip_preflight)
    elif args.command == "status":
        output = status(result_root=args.result_root, manifest_path=args.manifest)
    else:
        output = audit(result_root=args.result_root, manifest_path=args.manifest)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    if args.command == "audit" and not output["clean"]:
        return 1
    if args.command == "run" and output["units"].get("succeeded", 0) != output["total_units"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
