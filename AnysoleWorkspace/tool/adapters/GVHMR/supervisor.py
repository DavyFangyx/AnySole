"""Detached, finite-retry supervisor for the authorized full RGB corpus run."""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .config import WORKSPACE_ROOT
from .io import atomic_json
from .pipeline import _cgroup_memory


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if root != WORKSPACE_ROOT / "3_Result" / "GVHMR_baseline":
        raise ValueError(f"unexpected full-run output root: {root}")
    run_root = root / ".run"
    manifest = run_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    state_path = run_root / "supervisor.json"
    state: dict[str, Any] = {
        "schema_id": "tactile.gvhmr-baseline.supervisor", "schema_version": 1,
        "pid": os.getpid(), "status": "running", "manifest": str(manifest),
        "attempts": [],
    }
    atomic_json(state_path, state)
    workers = 4
    ordinary_retries = 0
    for attempt in range(1, 7):
        memory = _cgroup_memory()
        if memory is not None:
            limit, current = memory
            while workers > 1 and current + (7 * workers * 6 + 16) * (1024 ** 3) > limit:
                workers -= 1
                print(f"cgroup preflight requires fewer workers; selecting 7x{workers}", flush=True)
        command = [
            sys.executable, "-m", "gvhmr_baseline.cli", "pipeline-run",
            "--manifest", str(manifest), "--output-root", str(root),
            "--workers-per-gpu", str(workers), "--retries", "1",
            "--full", "--authorization", "user-approved",
        ]
        print(f"supervisor attempt={attempt} workers_per_gpu={workers}", flush=True)
        completed = subprocess.run(command, check=False)
        run = _read(run_root / "pipeline_run.json") or {}
        summary = (run.get("summary") or {}).get("units") or {}
        record = {
            "attempt": attempt, "workers_per_gpu": workers,
            "exit_code": completed.returncode, "controller_status": run.get("status"),
            "succeeded_units": summary.get("succeeded"),
            "failed_units": summary.get("failed"),
            "memory_pressure": run.get("memory_pressure"),
        }
        state["attempts"].append(record)
        atomic_json(state_path, state)
        if completed.returncode == 0:
            state["status"] = "completed"
            atomic_json(state_path, state)
            print("supervisor completed full corpus", flush=True)
            return 0
        if run.get("memory_pressure") and workers > 1:
            workers -= 1
            print(f"memory-pressure guard stopped run; resuming safely at 7x{workers}", flush=True)
            continue
        if ordinary_retries < 1 and run.get("status") == "failed":
            ordinary_retries += 1
            print("controller reported unit failures; retrying incomplete units once", flush=True)
            continue
        state["status"] = "failed"
        atomic_json(state_path, state)
        print("supervisor stopped after bounded recovery attempts", flush=True)
        return 2
    state["status"] = "failed"
    atomic_json(state_path, state)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
