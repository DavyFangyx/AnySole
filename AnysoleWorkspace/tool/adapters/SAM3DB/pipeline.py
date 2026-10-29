"""Autonomous multi-GPU controller, monitor, and final audit."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2

from wham_baseline.dataset import load_manifest, manifest_units

from .config import DEFAULT_FPS, DEFAULT_GPUS, RESULT_ROOT, runtime_environment
from .io import atomic_json, checked_result_root, compact_unit_valid, now, read_json, short_unit_id


def parse_gpus(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("GPUs must be comma-separated integers") from exc
    else:
        gpus = tuple(int(item) for item in value)
    if not gpus or any(item < 0 for item in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be non-empty, unique, non-negative integers")
    return gpus


def _load_control(result_root: str | Path, manifest_path: str | Path | None = None):
    root = checked_result_root(result_root)
    run_root = root / ".run"
    config = read_json(run_root / "config.json")
    ready = read_json(run_root / "ready.json")
    if not config or not ready or ready.get("status") != "ready":
        raise RuntimeError("pipeline is not prepared; run prepare first")
    path = Path(manifest_path or config["manifest"]).resolve()
    manifest = load_manifest(path)
    return root, run_root, config, manifest, path


def status(*, result_root: str | Path = RESULT_ROOT, manifest_path: str | Path | None = None) -> dict[str, Any]:
    root, run_root, config, manifest, path = _load_control(result_root, manifest_path)
    units = manifest_units(manifest)
    states = Counter()
    frames = Counter()
    for unit in units:
        if compact_unit_valid(root, unit, config["config_hash"]):
            states["succeeded"] += 1
            frames["succeeded"] += int(unit["frame_count"])
        elif (run_root / "failures" / f"{short_unit_id(unit)}.json").is_file():
            states["failed"] += 1
            frames["failed"] += int(unit["frame_count"])
        else:
            states["pending"] += 1
            frames["pending"] += int(unit["frame_count"])
    run = read_json(run_root / "run.json")
    return {
        "at": now(), "status": "completed" if states["succeeded"] == len(units) else "active",
        "config_hash": config["config_hash"], "manifest": str(path),
        "units": dict(states), "frames": dict(frames), "total_units": len(units),
        "total_frames": sum(int(unit["frame_count"]) for unit in units), "run": run,
    }


def _gpu_preflight(gpus: Sequence[int]) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True
    )
    rows = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(item.strip()) for item in line.split(","))
        rows[index] = (memory, utilization)
    missing = [gpu for gpu in gpus if gpu not in rows]
    busy = {gpu: rows[gpu] for gpu in gpus if gpu in rows and rows[gpu][0] > 512}
    if missing:
        raise RuntimeError(f"requested GPUs are absent: {missing}")
    if busy:
        raise RuntimeError(f"requested GPUs are not idle: {busy}")


def _pending(root: Path, units: Sequence[Mapping[str, Any]], config_hash: str) -> list[int]:
    return [index for index, unit in enumerate(units) if not compact_unit_valid(root, unit, config_hash)]


def _build_worker_plan(
    pending: Sequence[int], devices: Sequence[int], workers_per_gpu: int
) -> list[tuple[int, int, list[int]]]:
    """Build deterministic, disjoint GPU slots for the remaining units."""

    if workers_per_gpu not in (1, 2):
        raise ValueError("workers_per_gpu must be 1 or 2")
    slots = [
        (gpu, local_slot)
        for local_slot in range(workers_per_gpu)
        for gpu in devices
    ]
    shards = [list(pending[offset::len(slots)]) for offset in range(len(slots))]
    return [
        (gpu, local_slot, indices)
        for (gpu, local_slot), indices in zip(slots, shards)
        if indices
    ]


def run(
    *,
    result_root: str | Path = RESULT_ROOT,
    manifest_path: str | Path | None = None,
    gpus: Sequence[int] = DEFAULT_GPUS,
    workers_per_gpu: int = 1,
    retries: int = 1,
    fps: float = DEFAULT_FPS,
    full: bool = False,
    authorization: str | None = None,
    force: bool = False,
    preflight: bool = True,
) -> dict[str, Any]:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    devices = parse_gpus(gpus)
    if workers_per_gpu not in (1, 2):
        raise ValueError("workers_per_gpu must be 1 or 2")
    root, run_root, config, manifest, path = _load_control(result_root, manifest_path)
    units = manifest_units(manifest)
    if full:
        if authorization != "user-approved":
            raise RuntimeError("full execution requires --authorization user-approved")
        if not config.get("full_corpus"):
            raise RuntimeError("full execution requires the complete unfiltered corpus manifest")
    elif config.get("full_corpus"):
        raise RuntimeError("a complete manifest requires --full and explicit authorization")
    if preflight:
        _gpu_preflight(devices)

    run_record = {
        "schema_id": "tactile.sam3db-baseline.run", "schema_version": 1,
        "status": "running", "started_at": now(), "pid": os.getpid(), "pgid": os.getpgrp(),
        "config_hash": config["config_hash"], "manifest": str(path), "gpus": list(devices),
        "workers_per_gpu": workers_per_gpu,
        "slot_count": len(devices) * workers_per_gpu,
        "retries": retries, "fps": fps, "full": full,
    }
    atomic_json(run_root / "run.json", run_record)
    children: list[subprocess.Popen[bytes]] = []
    old_handlers: dict[int, Any] = {}
    stop_requested = False

    def stop_children(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        run_record.update(status="stopping", signal=signum, updated_at=now())
        atomic_json(run_root / "run.json", run_record)
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, stop_children)
    try:
        for attempt in range(retries + 1):
            pending = _pending(root, units, config["config_hash"])
            if not pending:
                break
            plan = _build_worker_plan(pending, devices, workers_per_gpu)
            children = []
            child_records = []
            for gpu, local_slot, indices in plan:
                command = [
                    sys.executable, "-m", "sam3db_baseline.worker",
                    "--manifest", str(path), "--indices", ",".join(str(index) for index in indices),
                    "--result-root", str(root), "--annotation-index", config["annotation_index"],
                    "--config-hash", config["config_hash"], "--fps", str(fps),
                ]
                if force:
                    command.append("--force")
                log_path = (
                    run_root / "logs"
                    / f"gpu-{gpu}-slot-{local_slot}-attempt-{attempt}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab")
                child = subprocess.Popen(
                    command, cwd=str(root.parents[1]), env=runtime_environment(gpu),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                children.append(child)
                child_records.append(
                    {"pid": child.pid, "gpu": gpu, "local_slot": local_slot,
                     "unit_count": len(indices), "log": str(log_path)}
                )
            run_record.update(
                attempt=attempt,
                worker_pids=[child.pid for child in children],
                workers=child_records,
                updated_at=now(),
            )
            atomic_json(run_root / "run.json", run_record)
            exits = [child.wait() for child in children]
            run_record.update(last_worker_exits=exits, updated_at=now())
            atomic_json(run_root / "run.json", run_record)
            if stop_requested:
                break
        final = status(result_root=root, manifest_path=path)
        succeeded = final["units"].get("succeeded", 0)
        run_record.update(
            status=("stopped" if stop_requested else
                    "completed" if succeeded == len(units) else "failed"),
            completed_at=now(), worker_pids=[], summary=final,
        )
        atomic_json(run_root / "run.json", run_record)
        atomic_json(run_root / "summary.json", final)
        return final
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def audit(*, result_root: str | Path = RESULT_ROOT, manifest_path: str | Path | None = None) -> dict[str, Any]:
    root, run_root, config, manifest, path = _load_control(result_root, manifest_path)
    units = manifest_units(manifest)
    issues: list[str] = []
    expected_dirs = {short_unit_id(unit) for unit in units}
    actual_dirs = {entry.name for entry in root.iterdir() if entry.is_dir() and entry.name != ".run"}
    for unexpected in sorted(actual_dirs - expected_dirs):
        issues.append(f"unexpected public directory: {unexpected}")
    succeeded = 0
    frames = 0
    for unit in units:
        if not compact_unit_valid(root, unit, config["config_hash"]):
            issues.append(f"invalid or missing compact unit: {short_unit_id(unit)}")
            continue
        unit_root = root / short_unit_id(unit)
        capture = cv2.VideoCapture(str(unit_root / "overlay.mp4"))
        video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if video_frames != int(unit["frame_count"]):
            issues.append(f"overlay frame mismatch: {short_unit_id(unit)}")
            continue
        succeeded += 1
        frames += int(unit["frame_count"])
    result = {
        "schema_id": "tactile.sam3db-baseline.audit", "schema_version": 1,
        "at": now(), "clean": not issues, "config_hash": config["config_hash"],
        "manifest": str(path), "expected_units": len(units), "succeeded_units": succeeded,
        "expected_frames": sum(int(unit["frame_count"]) for unit in units),
        "succeeded_frames": frames, "issues": issues,
    }
    atomic_json(run_root / "audit.json", result)
    return result
