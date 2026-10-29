"""Operator-facing command line for the WHAM baseline controller.

The CLI intentionally keeps the controller as a library first.  This makes
the dry-run plan and mock execution easy to exercise from focused tests while
keeping all WHAM-specific imports in ``runner.py``.

Examples (run with ``2_Code`` on ``PYTHONPATH``):

``python -m wham_baseline.cli run --manifest manifest.jsonl --result-root results --wham-root third_party/WHAM``
``python -m wham_baseline.cli plan --manifest manifest.jsonl --result-root results --wham-root third_party/WHAM --workers-per-gpu 2``
``python -m wham_baseline.cli calibrate --candidates 1,2,3``

``controller`` and ``launch`` are accepted as aliases for ``run``.  The
``controller run ...`` spelling is also accepted for compatibility with
operator notes that group commands under a controller namespace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .controller import (
    DEFAULT_GPU_IDS,
    ControllerError,
    calibration_plan,
    dry_run_plan,
    run_controller,
)


def _parse_gpus(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU list must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return values


def _parse_candidates(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidates must be comma-separated integers") from exc
    if not values or any(candidate < 1 for candidate in values):
        raise argparse.ArgumentTypeError("candidates must contain positive integers")
    return values


def _add_common_options(parser: argparse.ArgumentParser, *, plan: bool = False) -> None:
    parser.add_argument("--manifest", required=True, help="JSONL unit manifest")
    parser.add_argument(
        "--result-root",
        required=True,
        help="source-isolated output/status/log root",
    )
    parser.add_argument(
        "--wham-root",
        default=".",
        help="upstream WHAM checkout/root passed to each runner (default: .)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="independent slots per GPU (default: calibrated n=2; total slots = 7*n)",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=DEFAULT_GPU_IDS,
        help="comma-separated physical GPU IDs (default: 0,1,2,3,4,5,6)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="rerun units even when a terminal file exists",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="with resume enabled, rerun terminal failures",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="forward --mock to runner children (no model weights required)",
    )
    parser.add_argument(
        "--runner-dry-run",
        dest="dry_run_child",
        action="store_true",
        help="forward --dry-run to launched runner children",
    )
    parser.add_argument(
        "--runner-arg",
        dest="runner_flags",
        action="append",
        default=[],
        help="one additional runner argument; may be repeated",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="child Python executable (default: current interpreter)",
    )
    if plan:
        return
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a plan and do not create locks, status, or child processes",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="wait until all requested GPUs are idle before launching",
    )
    parser.add_argument(
        "--gpu-timeout",
        type=float,
        default=0.0,
        help="GPU preflight timeout in seconds (default: fail fast)",
    )
    parser.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=5.0,
        help="seconds between wait-for-GPU probes",
    )
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        help="force the read-only nvidia-smi preflight (default for real runs)",
    )
    preflight.add_argument(
        "--skip-preflight",
        dest="preflight",
        action="store_false",
        help="skip the GPU preflight (intended only for CPU/mock tests)",
    )
    parser.set_defaults(preflight=None)
    parser.add_argument(
        "--stop-grace",
        type=float,
        default=10.0,
        help="seconds to allow a stopped child to exit before kill",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wham-baseline",
        description="Resumable seven-GPU independent-process WHAM controller",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        aliases=("controller", "launch"),
        help="launch independent runner children or print --dry-run plan",
    )
    _add_common_options(run_parser)

    plan_parser = subparsers.add_parser(
        "plan",
        aliases=("dry-run",),
        help="build a side-effect-free deterministic slot plan",
    )
    _add_common_options(plan_parser, plan=True)

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="describe candidate n values without launching GPU/model work",
    )
    calibration_parser.add_argument(
        "--candidates",
        type=_parse_candidates,
        default=(1, 2, 3),
        help="comma-separated workers-per-GPU candidates (default: 1,2,3)",
    )
    calibration_parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=DEFAULT_GPU_IDS,
        help="comma-separated physical GPU IDs (default: 0,1,2,3,4,5,6)",
    )
    return parser


def _normalise_argv(argv: Optional[Sequence[str]]) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    # Permit ``controller run ...`` and ``controller ...`` forms while keeping
    # the regular argparse command surface compact.
    if values and values[0] == "controller":
        values.pop(0)
        if not values or values[0].startswith("-"):
            values.insert(0, "run")
    return values


def _print_json(value: object, *, stream: object = sys.stdout) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), file=stream)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning a process exit code."""

    parser = build_parser()
    values = _normalise_argv(argv)
    if not values:
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(values)
    try:
        if args.command in {"calibrate"}:
            _print_json(
                {
                    "status": "calibration_plan_only",
                    "candidates": list(calibration_plan(args.candidates, gpu_ids=args.gpus)),
                }
            )
            return 0

        if args.command in {"plan", "dry-run"}:
            plan = dry_run_plan(
                args.manifest,
                args.result_root,
                args.wham_root,
                workers_per_gpu=args.workers_per_gpu,
                gpu_ids=args.gpus,
                mock=args.mock,
                dry_run_child=True,
                runner_flags=args.runner_flags,
                resume=args.resume,
                retry_failed=args.retry_failed,
                python_executable=args.python_executable,
            )
            _print_json(plan)
            return 0

        if args.command in {"run", "controller", "launch"}:
            result = run_controller(
                args.manifest,
                args.result_root,
                args.wham_root,
                workers_per_gpu=args.workers_per_gpu,
                gpu_ids=args.gpus,
                dry_run=args.dry_run,
                mock=args.mock,
                dry_run_child=args.dry_run_child,
                runner_flags=args.runner_flags,
                resume=args.resume,
                retry_failed=args.retry_failed,
                wait_for_gpu=args.wait_for_gpu,
                gpu_timeout=args.gpu_timeout,
                gpu_poll_interval=args.gpu_poll_interval,
                preflight=args.preflight,
                python_executable=args.python_executable,
                stop_grace=args.stop_grace,
            )
            _print_json(result.to_dict())
            return result.exit_code
    except (ControllerError, OSError, ValueError) as exc:
        _print_json({"status": "error", "error": str(exc)}, stream=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess use
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
