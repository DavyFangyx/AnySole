"""Command-line entry points for the pinned GVHMR 2024 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_HMR2_CHECKPOINT,
    DEFAULT_PYTHON,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SMPL_MODEL,
    DEFAULT_SMPLX_MODEL,
    DEFAULT_SMPLX_TO_SMPL,
    DEFAULT_VITPOSE_CHECKPOINT,
    WORKSPACE_ROOT,
)
from .dataset import build_manifest, build_manifest_from_indexed_inventory, explicit_video_unit
from .evaluation import evaluate_camera_unit
from .io import atomic_json, read_state, sha256_file
from .pipeline import (
    DEFAULT_WORKERS_PER_GPU, build_slot_plan, pipeline_audit, pipeline_status,
    run_pipeline,
)
from .runtime import audit_manifest, audit_unit, list_status, read_manifest, run_unit, unit_output_dir


def _csv_args(values: list[str] | None) -> list[str] | None:
    return None if not values else [item for group in values for item in group.split(",") if item]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gvhmr_baseline", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build bounded RGB/target-box manifest")
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare.add_argument("--annotation-index", type=Path)
    prepare.add_argument("--indexed-inventory", type=Path, help="verified complete WHAM JSONL inventory; avoids re-decoding every source JPEG")
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    prepare.add_argument("--date", action="append")
    prepare.add_argument("--method", action="append")
    prepare.add_argument("--session", action="append")
    prepare.add_argument("--camera", action="append")
    prepare.add_argument("--max-units", type=int)
    prepare.add_argument("--max-frames", type=int, help="bound each unit while retaining full timing anchors")

    video = subparsers.add_parser("prepare-video", help="manifest an explicit video plus bbox .npy sidecar")
    video.add_argument("--video", type=Path, required=True)
    video.add_argument("--boxes-npy", type=Path, required=True, help="per-frame xyxy array [F,4]")
    video.add_argument("--fps", type=float, required=True, help="source video FPS used for timestamps")
    video.add_argument("--unit-id", default="explicit_video")
    video.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)

    run = subparsers.add_parser("run", help="infer and publish one bounded manifest unit")
    run.add_argument("--manifest", type=Path)
    run.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    run.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    run.add_argument("--unit-id", required=True)
    run.add_argument("--max-frames", type=int, required=True)
    run.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    run.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL_MODEL)
    run.add_argument("--mapper", type=Path, default=DEFAULT_SMPLX_TO_SMPL)
    run.add_argument("--hmr2-checkpoint", type=Path, default=DEFAULT_HMR2_CHECKPOINT)
    run.add_argument("--vitpose-checkpoint", type=Path, default=DEFAULT_VITPOSE_CHECKPOINT)
    run.add_argument("--moving-camera", action="store_true", help="run official DPVO path; unavailable extension fails clearly")
    run.add_argument("--retry", action="store_true", help="rerun a failed or interrupted unit from scratch")
    run.add_argument("--force", action="store_true", help="rerun even a succeeded unit and replace its outputs")

    status = subparsers.add_parser("status", help="show state for manifest units")
    status.add_argument("--manifest", type=Path)
    status.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    status.add_argument("--unit-id")

    audit = subparsers.add_parser("audit", help="verify results, state, and overlay checksums")
    audit.add_argument("--manifest", type=Path)
    audit.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)

    evaluate = subparsers.add_parser("evaluate", help="compute bounded camera-frame GT-prime metrics for one result")
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    evaluate.add_argument("--unit-id", required=True)
    evaluate.add_argument("--result", type=Path, help="defaults to <output-root>/<unit>/result.npz")
    evaluate.add_argument("--evaluation-output", type=Path, help="defaults to evaluation_camera.json beside result.npz")
    evaluate.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    evaluate.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL_MODEL)
    evaluate.add_argument("--device", default="cuda:0")

    plan = subparsers.add_parser("pipeline-plan", help="inspect deterministic seven-GPU shard allocation")
    plan.add_argument("--manifest", type=Path)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    plan.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)

    multi = subparsers.add_parser("pipeline-run", help="run seven-GPU inference, overlays and available GT-prime evaluation")
    multi.add_argument("--manifest", type=Path)
    multi.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    multi.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    multi.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    multi.add_argument("--moving-camera", action="store_true")
    multi.add_argument("--no-evaluate", action="store_true")
    multi.add_argument("--retries", type=int, default=1)
    multi.add_argument("--full", action="store_true", help="required for a complete image-dataset manifest")
    multi.add_argument("--authorization", choices=["user-approved"], help="explicit approval marker for a full run")
    multi.add_argument("--skip-gpu-preflight", action="store_true", help="only for controlled benchmarking")

    for command, description in (
        ("pipeline-status", "show terminal counts for the seven-GPU pipeline"),
        ("pipeline-audit", "verify every expected seven-GPU output and optional GT-prime report"),
    ):
        sub = subparsers.add_parser(command, help=description)
        sub.add_argument("--manifest", type=Path)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)

    return parser


def _manifest_path(args: Any) -> Path:
    if getattr(args, "manifest", None) is not None:
        return Path(args.manifest).expanduser().resolve()
    return Path(args.output_root).expanduser().resolve() / ".run" / "manifest.json"


def _index_path(output_root: Path, dates: list[str] | None) -> Path:
    token = "all_dates" if dates is None else "_".join(sorted(set(dates)))
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return output_root / ".run" / "annotation-index" / f"index_{digest}.sqlite"


def _prepare(args: Any) -> dict[str, Any]:
    from wham_baseline.constraints import build_annotation_index, read_index_metadata

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.indexed_inventory is not None:
        if args.annotation_index is None:
            raise ValueError("--indexed-inventory requires --annotation-index")
        if any((args.date, args.method, args.session, args.camera, args.max_units, args.max_frames)):
            raise ValueError("--indexed-inventory requires the complete, unfiltered corpus")
        manifest = build_manifest_from_indexed_inventory(dataset_root, args.indexed_inventory, args.annotation_index)
        destination = output_root / ".run" / "manifest.json"
        manifest["manifest_path"] = str(destination)
        atomic_json(destination, manifest)
        return {"manifest_path": str(destination), "summary": manifest["summary"], "source_inventory": manifest["source_inventory"], "annotation_index": manifest["annotation_index"]}
    dates = _csv_args(args.date)
    index_path = Path(args.annotation_index).expanduser().resolve() if args.annotation_index else _index_path(output_root, dates)
    if not index_path.is_file():
        build_annotation_index(dataset_root, index_path, dates=dates)
    else:
        metadata = read_index_metadata(index_path)
        needed = set(dates or [])
        if not needed.issubset(set(metadata.get("dates", []))):
            raise ValueError(
                f"annotation index lacks selected dates {sorted(needed - set(metadata.get('dates', [])))}: {index_path}"
            )
    manifest = build_manifest(
        dataset_root,
        index_path,
        dates=dates,
        methods=_csv_args(args.method),
        sessions=_csv_args(args.session),
        cameras=_csv_args(args.camera),
        max_units=args.max_units,
        max_frames_per_unit=args.max_frames,
    )
    destination = output_root / ".run" / "manifest.json"
    manifest["manifest_path"] = str(destination)
    atomic_json(destination, manifest)
    return {"manifest_path": str(destination), "summary": manifest["summary"], "annotation_index": manifest["annotation_index"]}


def _prepare_video(args: Any) -> dict[str, Any]:
    boxes = np.load(Path(args.boxes_npy).expanduser().resolve(), allow_pickle=False)
    unit = explicit_video_unit(args.video, boxes, fps=args.fps, unit_id=args.unit_id)
    manifest = {
        "schema_id": "tactile.gvhmr-baseline.inventory",
        "schema_version": 1,
        "dataset_root": str(Path(args.video).expanduser().resolve().parent),
        "source_tree": "explicit video with per-frame target boxes",
        "input_kind": "explicit_video",
        "selection": {"video": str(Path(args.video).expanduser().resolve()), "unit_id": args.unit_id},
        "complete": True,
        "annotation_index": None,
        "units": [unit],
        "summary": {"unit_count": 1, "frame_count": unit["frame_count"], "available_frame_count": unit["frame_count"]},
    }
    destination = Path(args.output_root).expanduser().resolve() / ".run" / "manifest.json"
    manifest["manifest_path"] = str(destination)
    atomic_json(destination, manifest)
    return {"manifest_path": str(destination), "summary": manifest["summary"]}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "prepare-video":
            result = _prepare_video(args)
        elif args.command == "run":
            manifest = read_manifest(_manifest_path(args))
            result = run_unit(
                manifest,
                args.unit_id,
                max_frames=args.max_frames,
                output_root=args.output_root,
                dataset_root=args.dataset_root,
                python_executable=args.python,
                checkpoint_path=args.checkpoint,
                smplx_model_path=args.smplx_model,
                smpl_model_path=args.smpl_model,
                mapper_path=args.mapper,
                hmr2_checkpoint_path=args.hmr2_checkpoint,
                vitpose_checkpoint_path=args.vitpose_checkpoint,
                static_camera=not args.moving_camera,
                retry=args.retry,
                force=args.force,
            )
        elif args.command == "status":
            manifest = read_manifest(_manifest_path(args))
            result = list_status(manifest, args.output_root)
            if args.unit_id:
                result = [row for row in result if str(row["unit_id"]) == args.unit_id]
                if not result:
                    raise ValueError(f"unit id not found in manifest: {args.unit_id}")
        elif args.command == "audit":
            manifest = read_manifest(_manifest_path(args))
            result = audit_manifest(manifest, args.output_root)
            if result["failure_count"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 2
        elif args.command == "evaluate":
            manifest = read_manifest(_manifest_path(args))
            matches = [unit for unit in manifest["units"] if str(unit["unit_id"]) == args.unit_id]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one manifest unit {args.unit_id!r}, found {len(matches)}")
            unit_directory = unit_output_dir(args.output_root, matches[0])
            state_path = unit_directory / "state.json"
            state = read_state(state_path)
            if state is None or state.get("status") != "succeeded":
                raise ValueError("camera evaluation requires a succeeded GVHMR unit; run and audit inference first")
            audit_unit(unit_directory)
            result_path = Path(args.result).expanduser().resolve() if args.result else unit_directory / "result.npz"
            evaluation_path = Path(args.evaluation_output).expanduser().resolve() if args.evaluation_output else result_path.with_name("evaluation_camera.json")
            if evaluation_path.parent == unit_directory and evaluation_path.name != "evaluation_camera.json":
                raise ValueError("an in-unit evaluation artifact must be named evaluation_camera.json; use an external path for diagnostics")
            result = evaluate_camera_unit(
                matches[0],
                result_path,
                evaluation_path,
                workspace_root=args.workspace_root,
                smpl_model_path=args.smpl_model,
                device=args.device,
            )
            if evaluation_path == unit_directory / "evaluation_camera.json":
                state.setdefault("artifacts", {})["evaluation_camera.json"] = {
                    "path": evaluation_path.name,
                    "size_bytes": evaluation_path.stat().st_size,
                    "sha256": sha256_file(evaluation_path),
                }
                atomic_json(state_path, state)
        elif args.command == "pipeline-plan":
            manifest = read_manifest(_manifest_path(args))
            slots = build_slot_plan(manifest["units"], workers_per_gpu=args.workers_per_gpu)
            result = {
                "workers_per_gpu": args.workers_per_gpu,
                "slot_count": len(slots),
                "unit_count": len(manifest["units"]),
                "estimated_frames": sum(int(unit["frame_count"]) for unit in manifest["units"]),
                "slots": slots,
            }
        elif args.command == "pipeline-run":
            result = run_pipeline(
                _manifest_path(args), output_root=args.output_root,
                workers_per_gpu=args.workers_per_gpu, smplx_model_path=args.smplx_model,
                static_camera=not args.moving_camera, evaluate=not args.no_evaluate,
                retries=args.retries, full=args.full, authorization=args.authorization,
                preflight=not args.skip_gpu_preflight,
            )
            if result["run_status"] not in {"completed", "already_complete"}:
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 2
        elif args.command == "pipeline-status":
            result = pipeline_status(_manifest_path(args), args.output_root)
        elif args.command == "pipeline-audit":
            result = pipeline_audit(_manifest_path(args), args.output_root)
            if not result["clean"]:
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 2
        else:  # pragma: no cover - argparse guards this
            raise ValueError(args.command)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
