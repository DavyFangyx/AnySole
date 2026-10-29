"""Guarded seven-GPU GVHMR inference, overlay, evaluation and audit pipeline.

The controller stays in the Tactile environment. Each of its 7*n slot
processes sees exactly one physical GPU; the existing pinned GVHMR subprocess
inherits that CUDA visibility. A slot handles its assigned units sequentially.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

from .config import (
    CODE_ROOT, DEFAULT_CHECKPOINT, DEFAULT_DATASET_ROOT, DEFAULT_RESULT_ROOT,
    DEFAULT_SMPL_MODEL, DEFAULT_SMPLX_MODEL, WORKSPACE_ROOT,
)
from .evaluation import evaluate_camera_unit
from .io import atomic_json, read_state, sha256_file
from .runtime import audit_unit, read_manifest, run_unit, safe_unit_id, unit_output_dir


DEFAULT_GPUS = tuple(range(7))
# Bounded 16/64/128-frame seven-L20 calibration: n=4 is the fastest tested
# memory-safe setting; n=5 exceeds the 256-GiB cgroup RAM projection.
DEFAULT_WORKERS_PER_GPU = 4
MAX_WORKERS_PER_GPU = 4
PIPELINE_SCHEMA = "tactile.gvhmr-baseline.pipeline"


class PipelineError(RuntimeError):
    """Raised when a guarded multi-GPU run is unsafe or incomplete."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PipelineError(f"expected JSON object: {path}")
    return value


def gtprime_path(unit: Mapping[str, Any], workspace_root: str | Path = WORKSPACE_ROOT) -> Path | None:
    """Return a dated GT′ archive only when that unit can actually be scored."""

    if unit.get("input_kind") != "annotated_rgb_image_sequence":
        return None
    from smpl_evaluation.io import DATE_TO_MOCAP

    date = str(unit.get("date"))
    subdir = DATE_TO_MOCAP.get(date)
    if subdir is None:
        return None
    root = Path(workspace_root).resolve()
    archive = root / "Mocap" / subdir / "mocap_ori_c3d" / str(unit.get("session")) / "motion_neutral_smpl.npz"
    alignment = root / "1_Dataset" / date / "RGB_MoCap_time_alignment" / f"{unit.get('session')}.csv"
    calibration = root / "1_Dataset" / date / f"{date}.json"
    return archive if archive.is_file() and alignment.is_file() and calibration.is_file() else None


def evaluate_and_publish(
    unit: Mapping[str, Any], output_root: str | Path, *,
    workspace_root: str | Path = WORKSPACE_ROOT,
    smpl_model_path: str | Path = DEFAULT_SMPL_MODEL,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Evaluate a successful unit if GT′ exists and atomically declare its report."""

    directory = unit_output_dir(output_root, unit)
    state_path = directory / "state.json"
    state = read_state(state_path)
    if state is None or state.get("status") != "succeeded":
        raise PipelineError(f"inference is not succeeded: {unit['unit_id']}")
    audit_unit(directory)
    gt_path = gtprime_path(unit, workspace_root)
    if gt_path is None:
        return {"status": "skipped_no_gt_prime", "report": None}
    report_path = directory / "evaluation_camera.json"
    if "evaluation_camera.json" in state.get("artifacts", {}):
        return {"status": "already_audited", "report": str(report_path)}
    evaluate_camera_unit(
        unit, directory / "result.npz", report_path,
        workspace_root=workspace_root, smpl_model_path=smpl_model_path, device=device,
    )
    state = read_state(state_path)
    if state is None or state.get("status") != "succeeded":
        raise PipelineError(f"unit state changed during evaluation: {unit['unit_id']}")
    state.setdefault("artifacts", {})["evaluation_camera.json"] = {
        "path": report_path.name,
        "size_bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
    }
    atomic_json(state_path, state)
    audit_unit(directory)
    return {"status": "computed", "report": str(report_path)}


def build_slot_plan(
    units: Sequence[Mapping[str, Any]], *,
    gpus: Sequence[int] = DEFAULT_GPUS,
    workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
) -> list[dict[str, Any]]:
    """Create 7*n fixed-affinity slots with deterministic frame-weighted shards."""

    devices = tuple(int(gpu) for gpu in gpus)
    if not devices or len(set(devices)) != len(devices) or any(gpu < 0 for gpu in devices):
        raise PipelineError("GPU IDs must be non-empty, unique and non-negative")
    if (isinstance(workers_per_gpu, bool) or int(workers_per_gpu) != workers_per_gpu
            or not 1 <= workers_per_gpu <= MAX_WORKERS_PER_GPU):
        raise PipelineError(f"workers_per_gpu must be a positive integer <= {MAX_WORKERS_PER_GPU}")
    slots = [
        {"slot_id": f"gpu-{gpu}-slot-{local_slot}", "gpu": gpu,
         "local_slot": local_slot, "indices": [], "estimated_frames": 0}
        for local_slot in range(int(workers_per_gpu)) for gpu in devices
    ]
    for index in sorted(range(len(units)), key=lambda i: (-int(units[i]["frame_count"]), i)):
        slot = min(slots, key=lambda item: (item["estimated_frames"], len(item["indices"]), item["gpu"], item["local_slot"]))
        slot["indices"].append(index)
        slot["estimated_frames"] += int(units[index]["frame_count"])
    for slot in slots:
        slot["indices"].sort()
    return slots


def _gpu_sample(gpus: Sequence[int]) -> dict[int, dict[str, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    rows: dict[int, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        if index in gpus:
            rows[index] = {"memory_mib": memory, "utilization_pct": utilization}
    if set(rows) != set(gpus):
        raise PipelineError(f"requested GPUs are not all visible: {sorted(set(gpus) - set(rows))}")
    return rows


def _preflight_gpus(gpus: Sequence[int]) -> None:
    rows = _gpu_sample(gpus)
    busy = {gpu: row for gpu, row in rows.items() if row["memory_mib"] > 512 or row["utilization_pct"] > 20}
    if busy:
        raise PipelineError(f"requested GPUs are busy; refusing competing load: {busy}")


def _cgroup_memory() -> tuple[int, int] | None:
    limit_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if not limit_path.is_file() or not current_path.is_file():
        return None
    limit = limit_path.read_text(encoding="utf-8").strip()
    if limit == "max":
        return None
    return int(limit), int(current_path.read_text(encoding="utf-8").strip())


def _preflight_memory(workers_per_gpu: int) -> None:
    sample = _cgroup_memory()
    if sample is None:
        return
    limit, current = sample
    # Measured 16-frame workers reached ~6 GiB RSS each. Keep a 16-GiB
    # reserve for longer clips, the controller, evaluation, and other users.
    required = len(DEFAULT_GPUS) * workers_per_gpu * 6 * (1024 ** 3) + 16 * (1024 ** 3)
    if current + required > limit:
        raise PipelineError(
            f"insufficient cgroup RAM for 7x{workers_per_gpu}: "
            f"current={current}, required_with_reserve={required}, limit={limit}"
        )


@contextmanager
def _controller_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"another GVHMR controller holds {path}") from exc
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _terminal_path(output_root: Path, unit: Mapping[str, Any]) -> Path:
    return output_root / ".run" / "terminals" / f"{safe_unit_id(str(unit['unit_id']))}.json"


def _unit_complete(
    output_root: Path, unit: Mapping[str, Any], config_hash: str,
    *, evaluate: bool, workspace_root: Path,
) -> bool:
    terminal = _json(_terminal_path(output_root, unit))
    if not terminal or terminal.get("status") != "succeeded" or terminal.get("config_hash") != config_hash:
        return False
    try:
        audit_unit(unit_output_dir(output_root, unit))
        if evaluate and gtprime_path(unit, workspace_root) is not None:
            state = read_state(unit_output_dir(output_root, unit) / "state.json")
            return state is not None and "evaluation_camera.json" in state.get("artifacts", {})
        return True
    except Exception:
        return False


def _checked_output_root(output_root: str | Path, *, full: bool, authorization: str | None) -> Path:
    root = Path(output_root).expanduser().resolve()
    workspace = WORKSPACE_ROOT.resolve()
    if root in {workspace, workspace / "1_Dataset", workspace / "2_Code", workspace / "3_Result"}:
        raise PipelineError(f"output root is too broad: {root}")
    result_tree = workspace / "3_Result"
    if root.is_relative_to(result_tree) and not (full and authorization == "user-approved"):
        raise PipelineError("writing under 3_Result requires a separately authorized full run")
    return root


def _full_corpus(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("input_kind") == "annotated_rgb_image_sequence" and bool(manifest.get("complete"))


def _run_config(
    manifest_path: Path, *, static_camera: bool, evaluate: bool,
    smplx_model_path: Path, workspace_root: Path,
) -> dict[str, Any]:
    provenance = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
        "smplx_model": str(smplx_model_path),
        "smplx_sha256": sha256_file(smplx_model_path),
        "static_camera": bool(static_camera),
        "evaluate": bool(evaluate),
        "workspace_root": str(workspace_root),
        "pipeline_schema_version": 1,
    }
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
    return {"schema_id": PIPELINE_SCHEMA, "schema_version": 1, "config_hash": digest, **provenance}


def pipeline_status(
    manifest_path: str | Path, output_root: str | Path, *,
    config_hash: str | None = None,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    root = Path(output_root).expanduser().resolve()
    config = _json(root / ".run" / "pipeline_config.json")
    run = _json(root / ".run" / "pipeline_run.json")
    expected_hash = config_hash or (config or {}).get("config_hash")
    counts = {"succeeded": 0, "failed": 0, "pending": 0}
    frames = {key: 0 for key in counts}
    for unit in manifest["units"]:
        terminal = _json(_terminal_path(root, unit))
        status = terminal.get("status") if terminal and terminal.get("config_hash") == expected_hash else "pending"
        status = status if status in counts else "pending"
        counts[status] += 1
        frames[status] += int(unit["frame_count"])
    controller_status = (run or {}).get("status", "not_started")
    if controller_status in {"running", "stopping"} and run is not None:
        same_host = run.get("hostname") == socket.gethostname()
        try:
            if same_host:
                os.kill(int(run["pid"]), 0)
        except (OSError, ValueError, KeyError, TypeError):
            controller_status = "stale_running"
    return {
        "schema_id": PIPELINE_SCHEMA + ".status", "schema_version": 1,
        "manifest": str(Path(manifest_path).resolve()), "output_root": str(root),
        "config_hash": expected_hash, "controller_status": controller_status,
        "unit_count": len(manifest["units"]),
        "units": counts, "frames": frames,
    }


def pipeline_audit(manifest_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    root = Path(output_root).expanduser().resolve()
    config = _json(root / ".run" / "pipeline_config.json")
    if config is None:
        raise PipelineError("pipeline configuration is missing")
    failures: list[dict[str, str]] = []
    successes = 0
    for unit in manifest["units"]:
        try:
            if not _unit_complete(
                root, unit, config["config_hash"], evaluate=bool(config["evaluate"]),
                workspace_root=Path(config["workspace_root"]),
            ):
                raise PipelineError("terminal, artifact audit or required evaluation is incomplete")
            successes += 1
        except Exception as exc:
            failures.append({"unit_id": str(unit["unit_id"]), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema_id": PIPELINE_SCHEMA + ".audit", "schema_version": 1,
        "clean": not failures, "expected_units": len(manifest["units"]),
        "succeeded_units": successes, "failure_count": len(failures), "failures": failures,
    }


def run_pipeline(
    manifest_path: str | Path, *, output_root: str | Path = DEFAULT_RESULT_ROOT,
    workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
    gpus: Sequence[int] = DEFAULT_GPUS,
    smplx_model_path: str | Path = DEFAULT_SMPLX_MODEL,
    static_camera: bool = True, evaluate: bool = True,
    retries: int = 1, full: bool = False, authorization: str | None = None,
    preflight: bool = True,
) -> dict[str, Any]:
    """Execute one bounded manifest or an explicitly authorized complete one."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = read_manifest(path)
    units = manifest.get("units", [])
    if not units:
        raise PipelineError("manifest has no units")
    if len(gpus) != 7 or tuple(sorted(gpus)) != DEFAULT_GPUS:
        raise PipelineError("this calibrated pipeline requires physical GPUs 0 through 6")
    if retries < 0:
        raise PipelineError("retries must be non-negative")
    if isinstance(workers_per_gpu, bool) or not 1 <= workers_per_gpu <= MAX_WORKERS_PER_GPU:
        raise PipelineError(f"workers_per_gpu must be 1..{MAX_WORKERS_PER_GPU} on this calibrated host")
    if _full_corpus(manifest) != bool(full):
        raise PipelineError("complete manifests require --full; bounded manifests must omit --full")
    if full and authorization != "user-approved":
        raise PipelineError("full corpus execution requires --authorization user-approved")
    root = _checked_output_root(output_root, full=full, authorization=authorization)
    normalized_units = [dict(unit) for unit in units]
    names = [unit_output_dir(root, unit).name for unit in normalized_units]
    if len(names) != len(set(names)):
        raise PipelineError("multiple unit IDs map to the same result directory")
    model_path = Path(smplx_model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    workspace_root = WORKSPACE_ROOT.resolve()
    config = _run_config(
        path, static_camera=static_camera, evaluate=evaluate,
        smplx_model_path=model_path, workspace_root=workspace_root,
    )
    run_root = root / ".run"
    with _controller_lock(run_root / "pipeline.lock"):
        previous = _json(run_root / "pipeline_config.json")
        if previous and previous.get("config_hash") != config["config_hash"]:
            raise PipelineError("result root has a different manifest/model/config; use a new output root")
        atomic_json(run_root / "pipeline_config.json", config)
        pending = [
            index for index, unit in enumerate(normalized_units)
            if not _unit_complete(root, unit, config["config_hash"], evaluate=evaluate, workspace_root=workspace_root)
        ]
        slots = build_slot_plan([normalized_units[index] for index in pending], gpus=gpus, workers_per_gpu=workers_per_gpu)
        # Translate shard positions back to source-manifest indexes.
        for slot in slots:
            slot["indices"] = [pending[position] for position in slot["indices"]]
        plan_summary = {
            "schema_id": PIPELINE_SCHEMA + ".plan", "schema_version": 1,
            "config_hash": config["config_hash"], "manifest": str(path),
            "workers_per_gpu": workers_per_gpu, "slot_count": len(slots),
            "pending_units": len(pending), "slots": slots,
        }
        atomic_json(run_root / "pipeline_plan.json", plan_summary)
        if not pending:
            summary = pipeline_status(path, root, config_hash=config["config_hash"])
            return {**summary, "run_status": "already_complete", "gpu_metrics": {}}
        if preflight:
            _preflight_gpus(gpus)
            _preflight_memory(workers_per_gpu)
        plan_dir = run_root / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        for slot in slots:
            if not slot["indices"]:
                continue
            atomic_json(plan_dir / f"{slot['slot_id']}.json", {
                "schema_id": PIPELINE_SCHEMA + ".slot", "schema_version": 1,
                "config_hash": config["config_hash"], "gpu": slot["gpu"],
                "local_slot": slot["local_slot"], "output_root": str(root),
                "dataset_root": manifest.get("dataset_root", str(DEFAULT_DATASET_ROOT)),
                "workspace_root": str(workspace_root), "static_camera": static_camera,
                "evaluate": evaluate, "retries": retries, "smplx_model": str(model_path),
                "units": [normalized_units[index] for index in slot["indices"]],
            })
        run_record = {
            "schema_id": PIPELINE_SCHEMA + ".run", "schema_version": 1,
            "status": "running", "started_utc": _now(), "pid": os.getpid(),
            "hostname": socket.gethostname(), "config_hash": config["config_hash"],
            "workers_per_gpu": workers_per_gpu, "slot_count": len(slots),
            "pending_units": len(pending), "children": [],
        }
        run_path = run_root / "pipeline_run.json"
        atomic_json(run_path, run_record)
        children: list[tuple[subprocess.Popen[bytes], dict[str, Any]]] = []
        old_handlers: dict[int, Any] = {}
        stopped = False

        def stop_children(signum: int, _frame: Any) -> None:
            nonlocal stopped
            stopped = True
            run_record.update(status="stopping", signal=signum, updated_utc=_now())
            atomic_json(run_path, run_record)
            for child, _slot in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, stop_children)
        metrics = {gpu: {"peak_memory_mib": 0, "utilization_sum": 0, "samples": 0} for gpu in gpus}
        host_memory_peak_bytes = (_cgroup_memory() or (0, 0))[1]
        try:
            for slot in slots:
                if not slot["indices"]:
                    continue
                slot_id = slot["slot_id"]
                log_path = run_root / "logs" / f"{slot_id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(slot["gpu"])
                environment["OMP_NUM_THREADS"] = "2"
                environment["MKL_NUM_THREADS"] = "2"
                environment["PYTHONPATH"] = str(CODE_ROOT) + (
                    os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
                )
                command = [
                    sys.executable, "-m", "gvhmr_baseline.pipeline", "slot",
                    "--plan", str(plan_dir / f"{slot_id}.json"),
                ]
                with log_path.open("ab") as log:
                    child = subprocess.Popen(
                        command, cwd=str(CODE_ROOT), env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                children.append((child, slot))
                run_record["children"].append({
                    "pid": child.pid, "gpu": slot["gpu"], "local_slot": slot["local_slot"],
                    "unit_count": len(slot["indices"]), "log": str(log_path),
                })
            atomic_json(run_path, run_record)
            while any(child.poll() is None for child, _ in children):
                host_sample = _cgroup_memory()
                if host_sample is not None:
                    host_limit, host_current = host_sample
                    host_memory_peak_bytes = max(host_memory_peak_bytes, host_current)
                    if host_current > host_limit - 8 * (1024 ** 3) and not stopped:
                        run_record["memory_pressure"] = {
                            "current_bytes": host_current, "limit_bytes": host_limit,
                            "reason": "less than 8 GiB cgroup memory reserve remains",
                        }
                        stop_children(signal.SIGTERM, None)
                try:
                    sample = _gpu_sample(gpus)
                    for gpu, row in sample.items():
                        item = metrics[gpu]
                        item["peak_memory_mib"] = max(item["peak_memory_mib"], row["memory_mib"])
                        item["utilization_sum"] += row["utilization_pct"]
                        item["samples"] += 1
                except Exception as exc:
                    run_record["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(2)
            exits = {slot["slot_id"]: child.wait() for child, slot in children}
            summary = pipeline_status(path, root, config_hash=config["config_hash"])
            clean = summary["units"]["succeeded"] == len(units) and all(code == 0 for code in exits.values())
            final_status = "stopped" if stopped else "completed" if clean else "failed"
            summary["controller_status"] = final_status
            metric_summary = {
                str(gpu): {
                    "peak_memory_mib": values["peak_memory_mib"],
                    "mean_utilization_pct": round(values["utilization_sum"] / values["samples"], 2) if values["samples"] else None,
                    "samples": values["samples"],
                }
                for gpu, values in metrics.items()
            }
            run_record.update(
                status=final_status,
                finished_utc=_now(), child_exits=exits, gpu_metrics=metric_summary,
                host_memory_peak_bytes=host_memory_peak_bytes, summary=summary,
            )
            atomic_json(run_path, run_record)
            return {**summary, "run_status": run_record["status"], "gpu_metrics": metric_summary, "child_exits": exits}
        except BaseException as exc:
            for child, _slot in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for child, _slot in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            run_record.update(
                status="stopped" if stopped else "failed", finished_utc=_now(),
                error=f"{type(exc).__name__}: {exc}",
                child_exits={slot["slot_id"]: child.returncode for child, slot in children},
            )
            atomic_json(run_path, run_record)
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def run_slot(plan_path: str | Path) -> int:
    """Run one fixed-affinity shard; never hides an individual unit failure."""

    plan = _json(Path(plan_path).resolve())
    if plan is None or plan.get("schema_id") != PIPELINE_SCHEMA + ".slot":
        raise PipelineError("slot plan is missing or invalid")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(plan["gpu"]):
        raise PipelineError(f"slot GPU affinity mismatch: {visible!r} != {plan['gpu']}")
    root = Path(plan["output_root"])
    failures = 0
    for unit in plan["units"]:
        unit_id = str(unit["unit_id"])
        terminal_path = _terminal_path(root, unit)
        started = time.monotonic()
        stage = "inference"
        outcome: dict[str, Any] = {
            "schema_id": PIPELINE_SCHEMA + ".unit", "schema_version": 1,
            "config_hash": plan["config_hash"], "unit_id": unit_id,
            "gpu": plan["gpu"], "local_slot": plan["local_slot"],
            "frame_count": int(unit["frame_count"]), "started_utc": _now(),
        }
        try:
            attempts = int(plan["retries"]) + 1
            for attempt in range(attempts):
                try:
                    run_unit(
                        {"schema_id": "tactile.gvhmr-baseline.inventory", "units": [unit]},
                        unit_id, max_frames=int(unit["frame_count"]),
                        output_root=root, dataset_root=plan["dataset_root"],
                        smplx_model_path=plan["smplx_model"],
                        static_camera=bool(plan["static_camera"]), retry=True,
                    )
                    break
                except Exception:
                    if attempt + 1 == attempts:
                        raise
            stage = "evaluation"
            evaluation = (
                evaluate_and_publish(unit, root, workspace_root=plan["workspace_root"], device="cuda:0")
                if plan["evaluate"] else {"status": "disabled", "report": None}
            )
            stage = "audit"
            audit_unit(unit_output_dir(root, unit))
            outcome.update(status="succeeded", evaluation=evaluation, attempts=attempt + 1)
        except Exception as exc:
            failures += 1
            outcome.update(status="failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
        outcome.update(finished_utc=_now(), duration_seconds=round(time.monotonic() - started, 3))
        atomic_json(terminal_path, outcome)
        print(json.dumps({key: outcome[key] for key in ("unit_id", "status", "gpu", "duration_seconds")}), flush=True)
    return 1 if failures else 0


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Internal GVHMR fixed-GPU slot worker")
    parser.add_argument("operation", choices=["slot"])
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    return run_slot(args.plan)


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_GPUS", "DEFAULT_WORKERS_PER_GPU", "MAX_WORKERS_PER_GPU", "PipelineError", "build_slot_plan",
    "evaluate_and_publish", "gtprime_path", "pipeline_audit", "pipeline_status",
    "run_pipeline", "run_slot",
]
