"""Safe, resumable controller for independent WHAM worker processes.

The controller deliberately does not import WHAM (or torch).  It only owns
process scheduling, manifest bookkeeping, and run metadata.  Each runnable
manifest unit is dispatched to one of ``7 * workers_per_gpu`` independent
worker slots.  A slot runs its assigned units sequentially, while the
different slots run concurrently in independent child processes.

The public functions in this module are intentionally small enough to use
from tests and from an operator-facing CLI:

``load_manifest``
    Read one JSON object per line and assign stable line indexes.
``build_slot_plan`` / ``assign_shards``
    Build a deterministic, disjoint seven-GPU slot plan.
``build_runner_command``
    Form the narrow ``python -m wham_baseline.runner run`` invocation.
``wait_for_gpus``
    Perform an injectable, fail-fast or waiting GPU preflight gate.
``run_controller``
    Execute or dry-run a plan with atomic status and terminal records.

No distributed launcher is used.  ``CUDA_VISIBLE_DEVICES`` is set separately
for every child process, so each child sees exactly one selected device as
device zero from its own perspective.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


DEFAULT_GPU_IDS = tuple(range(7))
"""The seven physical GPU IDs managed by the baseline controller."""

TERMINAL_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "skipped",
        "interrupted",
        "cancelled",
    }
)

SUCCESS_STATUSES = frozenset({"succeeded", "skipped"})


class ControllerError(RuntimeError):
    """Base error for invalid controller configuration or execution state."""


class ManifestError(ControllerError):
    """Raised when a JSONL manifest cannot be consumed safely."""


class PreflightError(ControllerError):
    """Raised when required GPUs are not present and idle."""


class ControllerLockError(ControllerError):
    """Raised when another live controller owns the result root."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via same-directory temp file + ``os.replace``.

    Controller and unit terminal files are the recovery boundary.  A partial
    JSON file would make resume unsafe, so the replacement is made only after
    flushing the temporary file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestUnit:
    """One line in a JSONL manifest with a stable controller index."""

    index: int
    payload: Mapping[str, Any]
    unit_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "unit_id": self.unit_id,
            "payload": dict(self.payload),
        }


def _unit_id(payload: Mapping[str, Any], index: int) -> str:
    for key in ("unit_id", "id", "key", "name"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"unit-{index:06d}"


def load_manifest(path: os.PathLike[str] | str) -> tuple[ManifestUnit, ...]:
    """Load a deterministic JSONL manifest.

    Blank lines are ignored.  Each non-blank line must contain a JSON object.
    A single ``{"units": [...]}`` wrapper is accepted as a convenience for
    manifests generated by older inventory scripts, while preserving the
    resulting list order and indexes.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest is not a file: {manifest_path}")

    records: list[Mapping[str, Any]] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"invalid JSON on manifest line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ManifestError(
                        f"manifest line {line_number} must be a JSON object"
                    )
                records.append(value)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc

    if len(records) == 1 and isinstance(records[0].get("units"), list):
        wrapped = records[0]["units"]
        records = []
        for position, value in enumerate(wrapped):
            if not isinstance(value, Mapping):
                raise ManifestError(f"wrapped manifest unit {position} is not an object")
            records.append(value)

    units: list[ManifestUnit] = []
    for index, payload in enumerate(records):
        units.append(ManifestUnit(index=index, payload=payload, unit_id=_unit_id(payload, index)))
    return tuple(units)


def normalize_terminal_status(value: Any) -> Optional[str]:
    """Map common runner status spellings to controller terminal states."""

    if value is None:
        return None
    status = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ok": "succeeded",
        "success": "succeeded",
        "complete": "succeeded",
        "completed": "succeeded",
        "done": "succeeded",
        "failure": "failed",
        "error": "failed",
        "stopped": "interrupted",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "in_progress": "running",
        "inprogress": "running",
    }
    return aliases.get(status, status)


def _is_terminal_record(record: Optional[Mapping[str, Any]]) -> bool:
    if not record:
        return False
    status = normalize_terminal_status(
        record.get("status", record.get("state", record.get("result")))
    )
    if status in TERMINAL_STATUSES:
        return True
    return bool(record.get("terminal") is True)


def _safe_component(value: str) -> str:
    component = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return component[:160] or "unit"


def _runner_unit_slug(value: str, index: int) -> str:
    """Mirror the adapter's numeric-first unit directory convention."""

    slug = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    slug = slug.strip("-.") or "unit"
    return f"{index:06d}-{slug[:180]}"


def terminal_path_for_unit(result_root: os.PathLike[str] | str, unit_index: int) -> Path:
    """Return the controller-owned canonical terminal path for a unit."""

    return Path(result_root) / "units" / str(int(unit_index)) / "terminal.json"


def _terminal_candidates(
    result_root: Path, unit: ManifestUnit
) -> tuple[Path, ...]:
    primary = terminal_path_for_unit(result_root, unit.index)
    candidates = [
        primary,
        result_root / "units" / f"{unit.index:06d}" / "terminal.json",
        result_root / "units" / str(unit.index) / "status.json",
        result_root / "unit" / str(unit.index) / "terminal.json",
        result_root / "terminals" / f"{unit.index}.json",
        result_root / "terminal" / f"{unit.index}.json",
        result_root / f"unit-{unit.index:06d}" / "terminal.json",
        # Current runner.py writes ``units/<index>-<unit-id>/status.json``.
        result_root / "units" / _runner_unit_slug(unit.unit_id, unit.index) / "status.json",
        result_root / _safe_component(unit.unit_id) / "terminal.json",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def read_unit_terminal(
    result_root: os.PathLike[str] | str, unit: ManifestUnit
) -> tuple[Optional[Path], Optional[Mapping[str, Any]]]:
    """Find a valid terminal/status record for ``unit``.

    The canonical path is checked first.  A few read-only compatibility paths
    make resume work with an adapter that chose a legacy status layout.  The
    controller always writes its own canonical path after a child exits.
    """

    root = Path(result_root)
    for candidate in _terminal_candidates(root, unit):
        record = _read_json(candidate)
        if _is_terminal_record(record):
            return candidate, record
    return None, None


@dataclass(frozen=True)
class Slot:
    """One independent worker slot bound to exactly one physical GPU."""

    slot_index: int
    gpu_id: int
    worker_index: int
    unit_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def slot_id(self) -> str:
        return f"gpu-{self.gpu_id}-worker-{self.worker_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "slot_id": self.slot_id,
            "gpu_id": self.gpu_id,
            "worker_index": self.worker_index,
            "unit_indices": list(self.unit_indices),
            "cuda_visible_devices": str(self.gpu_id),
        }


def _normalize_gpu_ids(gpu_ids: Iterable[int]) -> tuple[int, ...]:
    values: list[int] = []
    for value in gpu_ids:
        try:
            gpu_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"invalid GPU id: {value!r}") from exc
        if gpu_id < 0:
            raise ControllerError(f"GPU id must be non-negative: {gpu_id}")
        if gpu_id in values:
            raise ControllerError(f"duplicate GPU id: {gpu_id}")
        values.append(gpu_id)
    if not values:
        raise ControllerError("at least one GPU id is required")
    return tuple(values)


def _normalize_unit_indices(
    unit_indices: Sequence[int] | Iterable[int] | int | None, *, num_units: Optional[int]
) -> tuple[int, ...]:
    if num_units is not None:
        if unit_indices is not None:
            raise ControllerError("provide unit_indices or num_units, not both")
        unit_indices = num_units
    if unit_indices is None:
        return tuple()
    if isinstance(unit_indices, int):
        if unit_indices < 0:
            raise ControllerError("num_units must be non-negative")
        values: Iterable[Any] = range(unit_indices)
    else:
        values = unit_indices
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if hasattr(value, "index") and not isinstance(value, (int, bool)):
            value = getattr(value, "index")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"invalid unit index: {value!r}") from exc
        if index < 0:
            raise ControllerError(f"unit index must be non-negative: {index}")
        if index in seen:
            raise ControllerError(f"duplicate unit index: {index}")
        seen.add(index)
        result.append(index)
    return tuple(result)


def build_slot_plan(
    unit_indices: Sequence[int] | Iterable[int] | int | None = None,
    workers_per_gpu: int = 1,
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
    *,
    num_units: Optional[int] = None,
) -> tuple[Slot, ...]:
    """Build exactly ``len(gpu_ids) * workers_per_gpu`` deterministic slots.

    Units are assigned round-robin over GPU-major slot order.  Consequently,
    every unit appears exactly once, shard order is stable, and every slot is
    pinned to one GPU.  Passing an integer as ``unit_indices`` means
    ``range(unit_indices)``; ``num_units=`` is an explicit equivalent.
    """

    try:
        workers = int(workers_per_gpu)
    except (TypeError, ValueError) as exc:
        raise ControllerError("workers_per_gpu must be an integer") from exc
    if workers < 1:
        raise ControllerError("workers_per_gpu must be >= 1")
    normalized_gpus = _normalize_gpu_ids(gpu_ids)
    indexes = _normalize_unit_indices(unit_indices, num_units=num_units)

    slots: list[Slot] = []
    for gpu_id in normalized_gpus:
        for worker_index in range(workers):
            slots.append(
                Slot(
                    slot_index=len(slots),
                    gpu_id=gpu_id,
                    worker_index=worker_index,
                    unit_indices=tuple(),
                )
            )
    buckets: list[list[int]] = [[] for _ in slots]
    for position, unit_index in enumerate(indexes):
        buckets[position % len(slots)].append(unit_index)
    return tuple(
        Slot(
            slot_index=slot.slot_index,
            gpu_id=slot.gpu_id,
            worker_index=slot.worker_index,
            unit_indices=tuple(buckets[slot.slot_index]),
        )
        for slot in slots
    )


def _retain_runnable_units(
    slots: Sequence[Slot], runnable_indexes: Iterable[int]
) -> tuple[Slot, ...]:
    """Remove resumed terminal units without changing their original slots."""

    allowed = set(runnable_indexes)
    return tuple(
        Slot(
            slot_index=slot.slot_index,
            gpu_id=slot.gpu_id,
            worker_index=slot.worker_index,
            unit_indices=tuple(index for index in slot.unit_indices if index in allowed),
        )
        for slot in slots
    )


def assign_shards(
    unit_indices: Sequence[int] | Iterable[int] | int,
    workers_per_gpu: int = 1,
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
) -> dict[int, tuple[int, ...]]:
    """Return ``slot_index -> disjoint unit indexes`` for inspection/tests."""

    return {
        slot.slot_index: slot.unit_indices
        for slot in build_slot_plan(unit_indices, workers_per_gpu, gpu_ids)
    }


# Friendly aliases for callers that used the earlier design vocabulary.
slot_plan = build_slot_plan
shard_units = assign_shards


def _normalize_runner_flags(flags: Optional[Sequence[str] | str]) -> tuple[str, ...]:
    if flags is None:
        return tuple()
    if isinstance(flags, str):
        return tuple(shlex.split(flags))
    return tuple(str(flag) for flag in flags)


def build_runner_command(
    manifest_path: os.PathLike[str] | str,
    unit_index: int,
    result_root: os.PathLike[str] | str,
    wham_root: os.PathLike[str] | str,
    *,
    dry_run: bool = False,
    mock: bool = False,
    runner_flags: Optional[Sequence[str] | str] = None,
    python_executable: Optional[str] = None,
) -> list[str]:
    """Construct the exact narrow runner invocation for one unit."""

    executable = python_executable or sys.executable
    command = [
        str(executable),
        "-m",
        "wham_baseline.runner",
        "run",
        "--manifest",
        str(manifest_path),
        "--unit-index",
        str(int(unit_index)),
        "--result-root",
        str(result_root),
        "--wham-root",
        str(wham_root),
    ]
    if dry_run:
        command.append("--dry-run")
    if mock:
        command.append("--mock")
    command.extend(_normalize_runner_flags(runner_flags))
    return command


def worker_environment(
    gpu_id: int,
    *,
    base_environment: Optional[Mapping[str, str]] = None,
    run_id: Optional[str] = None,
    slot_index: Optional[int] = None,
    config_root: Optional[os.PathLike[str] | str] = None,
) -> dict[str, str]:
    """Return a child environment pinned to one GPU.

    ``base_environment`` is copied; the caller's process environment is never
    mutated.  The explicit controller metadata is useful in logs and makes it
    easy to audit that no child inherited a multi-GPU visibility list.
    """

    environment = dict(os.environ if base_environment is None else base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    if run_id is not None:
        environment["WHAM_CONTROLLER_RUN_ID"] = str(run_id)
    if slot_index is not None:
        environment["WHAM_CONTROLLER_SLOT"] = str(int(slot_index))
    if config_root is not None:
        if slot_index is None:
            raise ControllerError("slot_index is required with config_root")
        # Ultralytics 8.0 writes settings.yaml at import time without an
        # inter-process lock.  A slot-local directory prevents concurrent
        # workers from observing a partially written shared YAML file.
        environment["YOLO_CONFIG_DIR"] = str(
            Path(config_root) / f"slot-{int(slot_index):03d}"
        )
    return environment


# Alias used by a few integrations.
build_worker_environment = worker_environment


def _parse_csv_row(row: str) -> list[str]:
    return [part.strip() for part in row.strip().split(",")]


def _default_gpu_probe(gpu_ids: Sequence[int]) -> Mapping[int, Mapping[str, Any]]:
    """Inspect presence and compute-process occupancy with ``nvidia-smi``.

    This is read-only.  It intentionally does not import torch or initialize a
    CUDA context.  A GPU with a running compute process is considered busy,
    which protects the active Fast-MHR job from an accidental WHAM launch.
    """

    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreflightError("nvidia-smi is not available for GPU preflight") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise PreflightError(f"nvidia-smi GPU query failed: {detail}") from exc

    records: dict[int, dict[str, Any]] = {}
    uuid_to_gpu: dict[str, int] = {}
    for raw_row in result.stdout.splitlines():
        if not raw_row.strip():
            continue
        parts = _parse_csv_row(raw_row)
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        uuid = parts[1]
        records[index] = {
            "available": True,
            "idle": True,
            "uuid": uuid,
            "memory_used_mib": parts[2] if len(parts) > 2 else None,
            "compute_pids": [],
        }
        uuid_to_gpu[uuid] = index

    apps_query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        apps = subprocess.run(
            apps_query,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Presence is still useful if the optional process query is not
        # supported by a driver; callers can choose a stricter custom probe.
        apps = None
    if apps is not None:
        for raw_row in apps.stdout.splitlines():
            if not raw_row.strip():
                continue
            parts = _parse_csv_row(raw_row)
            if not parts:
                continue
            gpu_index = uuid_to_gpu.get(parts[0])
            if gpu_index is None:
                continue
            record = records[gpu_index]
            record["available"] = False
            record["idle"] = False
            if len(parts) > 1:
                record.setdefault("compute_pids", []).append(parts[1])

    required = set(int(gpu_id) for gpu_id in gpu_ids)
    return {
        gpu_id: records.get(
            gpu_id,
            {"available": False, "idle": False, "reason": "GPU not reported"},
        )
        for gpu_id in required
    }


def _probe_result_available(result: Any, gpu_ids: Sequence[int]) -> set[int]:
    """Normalize common injected probe result shapes to available GPU IDs."""

    required = set(gpu_ids)
    if result is True:
        return required
    if result is False or result is None:
        return set()
    if isinstance(result, Mapping):
        available: set[int] = set()
        for raw_gpu, value in result.items():
            try:
                gpu_id = int(raw_gpu)
            except (TypeError, ValueError):
                continue
            if gpu_id not in required:
                continue
            if isinstance(value, Mapping):
                ready = value.get("available", value.get("idle", value.get("free", False)))
                if bool(ready):
                    available.add(gpu_id)
            elif bool(value):
                available.add(gpu_id)
        return available
    if isinstance(result, (str, bytes)):
        return set()
    try:
        values = {int(value) for value in result}
    except (TypeError, ValueError):
        return set()
    return values & required


def _call_probe(
    probe: Callable[..., Any], gpu_ids: Sequence[int]
) -> Any:
    # The one-argument form is the documented shape.  Supporting a no-arg
    # lambda keeps tests concise without making subprocess probing mandatory.
    try:
        return probe(tuple(gpu_ids))
    except TypeError as first_error:
        try:
            return probe()
        except TypeError:
            raise first_error


def wait_for_gpus(
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
    *,
    timeout: Optional[float] = 0.0,
    poll_interval: float = 5.0,
    probe: Optional[Callable[..., Any]] = None,
) -> Mapping[int, Any]:
    """Wait until every requested GPU is present and idle.

    ``probe`` may return a mapping of GPU ID to booleans/records, an iterable
    of available IDs, or one boolean for all IDs.  With ``timeout=0`` the
    check is fail-fast; ``None`` waits indefinitely.  The returned mapping is
    the final raw probe result and is intended for status evidence.
    """

    normalized = _normalize_gpu_ids(gpu_ids)
    try:
        wait_seconds = None if timeout is None else float(timeout)
    except (TypeError, ValueError) as exc:
        raise PreflightError("GPU timeout must be a number or null") from exc
    if wait_seconds is not None and wait_seconds < 0:
        wait_seconds = None
    try:
        interval = float(poll_interval)
    except (TypeError, ValueError) as exc:
        raise PreflightError("GPU poll interval must be a number") from exc
    if interval < 0:
        raise PreflightError("GPU poll interval must be non-negative")

    probe_fn: Callable[..., Any] = probe or _default_gpu_probe
    started = time.monotonic()
    last_result: Any = None
    while True:
        last_result = _call_probe(probe_fn, normalized)
        available = _probe_result_available(last_result, normalized)
        missing = [gpu_id for gpu_id in normalized if gpu_id not in available]
        if not missing:
            return last_result if isinstance(last_result, Mapping) else {
                str(gpu_id): True for gpu_id in normalized
            }
        if wait_seconds is not None and time.monotonic() - started >= wait_seconds:
            raise PreflightError(
                "GPU preflight failed; unavailable/busy devices: "
                + ", ".join(str(gpu_id) for gpu_id in missing)
            )
        if interval:
            time.sleep(interval)
        else:
            # Avoid a hot loop when a test intentionally uses a zero interval.
            time.sleep(0)


preflight_gpus = wait_for_gpus


def _pid_is_alive(pid: Any) -> bool:
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return False
    if process_id <= 0:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_live(record: Optional[Mapping[str, Any]], path: Path, stale_after: float) -> bool:
    if record is None:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        return age < stale_after
    host = str(record.get("hostname", ""))
    pid = record.get("pid")
    if host and host != socket.gethostname():
        # A remote host may still be running even though its PID is meaningless
        # locally.  Only an explicitly old lock can be recovered automatically.
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        return age < stale_after
    return _pid_is_alive(pid)


class ControllerLock:
    """Atomic result-root lock with live/stale owner handling."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        run_id: Optional[str] = None,
        stale_after: float = 3600.0,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id or str(uuid.uuid4())
        self.stale_after = float(stale_after)
        self.acquired = False
        self.metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _utc_now(),
            "started_epoch": time.time(),
        }

    def acquire(self) -> "ControllerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(5):
            try:
                descriptor = os.open(
                    str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
                )
            except FileExistsError:
                existing = _read_json(self.path)
                if _lock_is_live(existing, self.path, self.stale_after):
                    owner = "unknown"
                    if existing:
                        owner = f"pid={existing.get('pid')} host={existing.get('hostname')}"
                    raise ControllerLockError(
                        f"controller lock is held by a live process ({owner}): {self.path}"
                    )
                stale_path = self.path.with_name(
                    f"{self.path.name}.stale-{int(time.time() * 1000000)}-{os.getpid()}"
                )
                try:
                    os.replace(str(self.path), str(stale_path))
                except FileNotFoundError:
                    continue
                continue
            except OSError as exc:
                raise ControllerLockError(f"cannot create controller lock {self.path}: {exc}") from exc
            try:
                encoded = (json.dumps(self.metadata, sort_keys=True) + "\n").encode("utf-8")
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                raise
            self.acquired = True
            return self
        raise ControllerLockError(f"could not acquire controller lock after stale recovery: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        current = _read_json(self.path)
        if current and current.get("run_id") != self.run_id:
            self.acquired = False
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False

    def __enter__(self) -> "ControllerLock":
        return self.acquire()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()


@dataclass
class UnitOutcome:
    index: int
    unit_id: str
    slot_index: Optional[int]
    gpu_id: Optional[int]
    status: str
    returncode: Optional[int] = None
    error: Optional[str] = None
    command: tuple[str, ...] = field(default_factory=tuple)
    terminal_path: Optional[str] = None
    log_path: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "unit_id": self.unit_id,
            "slot_index": self.slot_index,
            "gpu_id": self.gpu_id,
            "status": self.status,
            "returncode": self.returncode,
            "error": self.error,
            "command": list(self.command),
            "terminal_path": self.terminal_path,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class ControllerResult:
    """Structured result returned by :func:`run_controller`."""

    status: str
    exit_code: int
    run_id: str
    summary: Mapping[str, Any]
    plan: Optional[Mapping[str, Any]] = None
    outcomes: tuple[UnitOutcome, ...] = field(default_factory=tuple)
    status_path: Optional[str] = None

    @property
    def returncode(self) -> int:
        return self.exit_code

    @property
    def failures(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.summary.get("failures", ()))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": self.status,
            "exit_code": self.exit_code,
            "run_id": self.run_id,
            "summary": dict(self.summary),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }
        if self.plan is not None:
            value["plan"] = dict(self.plan)
        if self.status_path is not None:
            value["status_path"] = self.status_path
        return value


class _StatusBook:
    """Thread-safe atomic controller status writer."""

    def __init__(self, path: Path, initial: Mapping[str, Any]) -> None:
        self.path = path
        self._payload = dict(initial)
        self._lock = threading.Lock()

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._payload.update(fields)
            self._payload["updated_at"] = _utc_now()
            _write_atomic_json(self.path, self._payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._payload)


class _SignalStop:
    """Install signal handlers that only set an event, then restore them."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_SignalStop":
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                self.previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, self._handle)
            except (ValueError, OSError):
                # Only the main thread can install handlers.  Tests may run a
                # controller from a worker thread; process polling remains safe.
                continue
        return self

    def _handle(self, _signum: int, _frame: Any) -> None:
        self.stop_event.set()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for signal_number, previous in self.previous.items():
            try:
                signal.signal(signal_number, previous)
            except (ValueError, OSError):
                pass


def _process_returncode(process: Any) -> Optional[int]:
    value = process.poll()
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _terminate_process(process: Any, grace_seconds: float) -> None:
    try:
        process.terminate()
    except (AttributeError, OSError):
        return
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while _process_returncode(process) is None and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if _process_returncode(process) is None:
        try:
            process.kill()
        except (AttributeError, OSError):
            pass
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
    except (AttributeError, OSError, subprocess.TimeoutExpired):
        pass


def _wait_process(
    process: Any,
    stop_event: threading.Event,
    stop_grace: float,
) -> tuple[int, bool]:
    while True:
        returncode = _process_returncode(process)
        if returncode is not None:
            return returncode, False
        if stop_event.is_set():
            _terminate_process(process, stop_grace)
            returncode = _process_returncode(process)
            return (130 if returncode is None else returncode), True
        stop_event.wait(0.05)


def _summary_for(
    outcomes: Iterable[UnitOutcome],
    *,
    total_units: int,
    slot_count: int,
    interrupted: bool = False,
    extra_failures: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    rows = list(outcomes)
    counts = {
        "total": total_units,
        "succeeded": sum(row.status == "succeeded" for row in rows),
        "failed": sum(row.status == "failed" for row in rows),
        "skipped": sum(row.status == "skipped" for row in rows),
        "interrupted": sum(row.status == "interrupted" for row in rows),
        "pending": max(0, total_units - len(rows)),
    }
    failures: list[dict[str, Any]] = [
        row.to_dict() for row in rows if row.status == "failed"
    ]
    if extra_failures:
        failures.extend(dict(failure) for failure in extra_failures)
    counts["failure_count"] = len(failures)
    counts["slot_count"] = slot_count
    counts["interrupted_run"] = bool(interrupted)
    counts["failures"] = failures
    return counts


def _plan_for(
    units: Sequence[ManifestUnit],
    runnable_units: Sequence[ManifestUnit],
    slots: Sequence[Slot],
    *,
    manifest_path: Path,
    result_root: Path,
    wham_root: Path,
    dry_run_child: bool,
    mock: bool,
    runner_flags: Sequence[str],
    python_executable: Optional[str],
    terminal_records: Mapping[int, tuple[Optional[Path], Optional[Mapping[str, Any]]]],
) -> dict[str, Any]:
    commands: dict[int, list[str]] = {}
    for unit in runnable_units:
        commands[unit.index] = build_runner_command(
            manifest_path,
            unit.index,
            result_root,
            wham_root,
            dry_run=dry_run_child,
            mock=mock,
            runner_flags=runner_flags,
            python_executable=python_executable,
        )
    return {
        "schema_version": 1,
        "controller": "wham-independent-process",
        "distributed_launcher": None,
        "uses_ddp": False,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "result_root": str(result_root),
        "wham_root": str(wham_root),
        "dry_run": True,
        "workers_per_gpu": len(slots) // len({slot.gpu_id for slot in slots}),
        "gpu_ids": sorted({slot.gpu_id for slot in slots}),
        "slot_count": len(slots),
        "total_units": len(units),
        "runnable_units": len(runnable_units),
        "skipped_units": len(units) - len(runnable_units),
        "runner_flags": list(runner_flags),
        "command_template": commands[runnable_units[0].index]
        if runnable_units
        else build_runner_command(
            manifest_path,
            0,
            result_root,
            wham_root,
            dry_run=dry_run_child,
            mock=mock,
            runner_flags=runner_flags,
            python_executable=python_executable,
        ),
        "units": [
            {
                "index": unit.index,
                "unit_id": unit.unit_id,
                "status": normalize_terminal_status(
                    (terminal_records.get(unit.index) or (None, None))[1].get("status")
                    if (terminal_records.get(unit.index) or (None, None))[1]
                    else None
                )
                or ("planned" if unit in runnable_units else "skipped"),
                "terminal_path": str(
                    (terminal_records.get(unit.index) or (None, None))[0]
                    or terminal_path_for_unit(result_root, unit.index)
                ),
                "command": commands.get(unit.index),
            }
            for unit in units
        ],
        "slots": [
            {
                **slot.to_dict(),
                "commands": [commands[index] for index in slot.unit_indices],
            }
            for slot in slots
        ],
    }


def dry_run_plan(
    manifest_path: os.PathLike[str] | str,
    result_root: os.PathLike[str] | str,
    wham_root: os.PathLike[str] | str,
    *,
    workers_per_gpu: int = 1,
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
    mock: bool = False,
    dry_run_child: bool = True,
    runner_flags: Optional[Sequence[str] | str] = None,
    resume: bool = True,
    retry_failed: bool = False,
    python_executable: Optional[str] = None,
) -> dict[str, Any]:
    """Build a side-effect-free execution plan for operator review."""

    manifest = Path(manifest_path)
    result = Path(result_root)
    wham = Path(wham_root)
    units = load_manifest(manifest)
    terminal_records = {
        unit.index: read_unit_terminal(result, unit) for unit in units
    }
    runnable: list[ManifestUnit] = []
    for unit in units:
        _path, record = terminal_records[unit.index]
        status = normalize_terminal_status(
            record.get("status", record.get("state")) if record else None
        )
        terminal = _is_terminal_record(record)
        if resume and terminal and not (retry_failed and status == "failed"):
            continue
        runnable.append(unit)
    slots = _retain_runnable_units(
        build_slot_plan(
            tuple(unit.index for unit in units),
            workers_per_gpu=workers_per_gpu,
            gpu_ids=gpu_ids,
        ),
        (unit.index for unit in runnable),
    )
    return _plan_for(
        units,
        runnable,
        slots,
        manifest_path=manifest,
        result_root=result,
        wham_root=wham,
        dry_run_child=dry_run_child,
        mock=mock,
        runner_flags=_normalize_runner_flags(runner_flags),
        python_executable=python_executable,
        terminal_records=terminal_records,
    )


def _write_unit_terminal(
    result_root: Path,
    unit: ManifestUnit,
    outcome: UnitOutcome,
    *,
    existing: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Path:
    path = terminal_path_for_unit(result_root, unit.index)
    payload: dict[str, Any] = dict(existing or {})
    payload.update(
        {
            "schema_version": payload.get("schema_version", 1),
            "unit_index": unit.index,
            "unit_id": unit.unit_id,
            "status": outcome.status,
            "terminal": True,
            "returncode": outcome.returncode,
            "error": outcome.error,
            "controller_run_id": run_id or payload.get("controller_run_id"),
            "controller_finished_at": outcome.finished_at,
            "controller_slot_index": outcome.slot_index,
            "controller_gpu_id": outcome.gpu_id,
            "command": list(outcome.command),
        }
    )
    _write_atomic_json(path, payload)
    return path


def _make_failure_outcome(
    unit: ManifestUnit,
    slot: Slot,
    command: Sequence[str],
    log_path: Path,
    error: str,
    *,
    returncode: Optional[int] = None,
    status: str = "failed",
    started_at: Optional[str] = None,
) -> UnitOutcome:
    return UnitOutcome(
        index=unit.index,
        unit_id=unit.unit_id,
        slot_index=slot.slot_index,
        gpu_id=slot.gpu_id,
        status=status,
        returncode=returncode,
        error=error,
        command=tuple(command),
        log_path=str(log_path),
        started_at=started_at,
        finished_at=_utc_now(),
    )


def _run_slot(
    slot: Slot,
    units_by_index: Mapping[int, ManifestUnit],
    *,
    manifest_path: Path,
    result_root: Path,
    wham_root: Path,
    run_id: str,
    mock: bool,
    dry_run_child: bool,
    runner_flags: Sequence[str],
    python_executable: Optional[str],
    stop_event: threading.Event,
    stop_grace: float,
    popen_factory: Callable[..., Any],
    active_processes: MutableMapping[int, Any],
    active_processes_lock: threading.Lock,
) -> list[UnitOutcome]:
    outcomes: list[UnitOutcome] = []
    for unit_index in slot.unit_indices:
        unit = units_by_index[unit_index]
        command = build_runner_command(
            manifest_path,
            unit.index,
            result_root,
            wham_root,
            dry_run=dry_run_child,
            mock=mock,
            runner_flags=runner_flags,
            python_executable=python_executable,
        )
        unit_log = result_root / "logs" / f"slot-{slot.slot_index:03d}" / f"unit-{unit.index:06d}.log"
        unit_log.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        if stop_event.is_set():
            outcome = _make_failure_outcome(
                unit,
                slot,
                command,
                unit_log,
                "controller stop requested before launch",
                returncode=130,
                status="interrupted",
                started_at=started_at,
            )
            outcome.terminal_path = str(
                _write_unit_terminal(result_root, unit, outcome, run_id=run_id)
            )
            outcomes.append(outcome)
            continue

        environment = worker_environment(
            slot.gpu_id,
            run_id=run_id,
            slot_index=slot.slot_index,
            config_root=result_root / ".worker-config",
        )
        process: Any = None
        try:
            with unit_log.open("w", encoding="utf-8") as log_stream:
                log_stream.write(
                    json.dumps(
                        {
                            "command": command,
                            "env": {
                                "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
                                "YOLO_CONFIG_DIR": environment["YOLO_CONFIG_DIR"],
                            },
                        }
                    )
                    + "\n"
                )
                log_stream.flush()
                process = popen_factory(
                    command,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with active_processes_lock:
                    active_processes[unit.index] = process
                returncode, was_interrupted = _wait_process(process, stop_event, stop_grace)
        except Exception as exc:  # child launch failures belong to this unit
            outcome = _make_failure_outcome(
                unit,
                slot,
                command,
                unit_log,
                f"worker launch/wait failed: {exc}",
                started_at=started_at,
            )
            returncode = outcome.returncode
            was_interrupted = False
        finally:
            with active_processes_lock:
                active_processes.pop(unit.index, None)

        # A retry leaves the controller-owned canonical terminal in place
        # while the child runs.  Prefer the runner's freshly written status
        # after the child exits; otherwise the stale canonical failure would
        # mask a successful retry.  Fake/custom runners may not create this
        # path, so retain the compatibility search as a fallback.
        runner_terminal_path = (
            result_root
            / "units"
            / _runner_unit_slug(unit.unit_id, unit.index)
            / "status.json"
        )
        fresh_runner_record = _read_json(runner_terminal_path)
        if _is_terminal_record(fresh_runner_record):
            terminal_path, terminal_record = runner_terminal_path, fresh_runner_record
        else:
            terminal_path, terminal_record = read_unit_terminal(result_root, unit)
        runner_status = normalize_terminal_status(
            terminal_record.get("status", terminal_record.get("state"))
            if terminal_record
            else None
        )
        if was_interrupted:
            status = "interrupted"
            error = "controller stop requested"
        elif runner_status == "failed":
            status = "failed"
            error = str(terminal_record.get("error") or f"runner terminal status: {runner_status}")
        elif returncode != 0:
            status = "failed"
            error = f"runner exited with status {returncode}"
        else:
            status = "succeeded"
            error = None
        outcome = UnitOutcome(
            index=unit.index,
            unit_id=unit.unit_id,
            slot_index=slot.slot_index,
            gpu_id=slot.gpu_id,
            status=status,
            returncode=returncode,
            error=error,
            command=tuple(command),
            terminal_path=str(terminal_path) if terminal_path else None,
            log_path=str(unit_log),
            started_at=started_at,
            finished_at=_utc_now(),
        )
        outcome.terminal_path = str(
            _write_unit_terminal(
                result_root,
                unit,
                outcome,
                existing=terminal_record,
                run_id=run_id,
            )
        )
        outcomes.append(outcome)
        # A failed unit does not cancel its siblings.  Independent slots keep
        # progressing so the final summary accounts for every unit.
    return outcomes


def _skip_outcome(
    unit: ManifestUnit,
    terminal_path: Optional[Path],
    terminal_record: Mapping[str, Any],
) -> UnitOutcome:
    status = normalize_terminal_status(
        terminal_record.get("status", terminal_record.get("state"))
    ) or "skipped"
    if status not in TERMINAL_STATUSES:
        status = "skipped"
    # A resumed run must not hide an earlier terminal failure merely because
    # retry_failed=False.  Keep it in the failure summary while avoiding a
    # duplicate child claim; callers can opt into a new attempt explicitly.
    outcome_status = "failed" if status in {"failed", "interrupted"} else "skipped"
    return UnitOutcome(
        index=unit.index,
        unit_id=unit.unit_id,
        slot_index=None,
        gpu_id=None,
        status=outcome_status,
        returncode=terminal_record.get("returncode")
        if isinstance(terminal_record.get("returncode"), int)
        else None,
        error=(
            str(terminal_record.get("error"))
            if outcome_status == "failed" and terminal_record.get("error")
            else (
                "existing terminal failure; pass retry_failed=True to retry"
                if outcome_status == "failed"
                else None
            )
        ),
        command=tuple(),
        terminal_path=str(terminal_path) if terminal_path else None,
        started_at=None,
        finished_at=None,
    )


def run_controller(
    manifest_path: os.PathLike[str] | str,
    result_root: os.PathLike[str] | str,
    wham_root: os.PathLike[str] | str,
    *,
    workers_per_gpu: int = 1,
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
    dry_run: bool = False,
    mock: bool = False,
    dry_run_child: bool = False,
    runner_flags: Optional[Sequence[str] | str] = None,
    resume: bool = True,
    retry_failed: bool = False,
    wait_for_gpu: bool = False,
    gpu_timeout: Optional[float] = 0.0,
    gpu_poll_interval: float = 5.0,
    preflight: Optional[bool] = None,
    gpu_probe: Optional[Callable[..., Any]] = None,
    popen_factory: Optional[Callable[..., Any]] = None,
    python_executable: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
    stop_grace: float = 10.0,
    lock_path: Optional[os.PathLike[str] | str] = None,
    status_path: Optional[os.PathLike[str] | str] = None,
    run_id: Optional[str] = None,
    stale_lock_after: float = 3600.0,
) -> ControllerResult:
    """Run the seven-GPU controller or return a side-effect-free dry-run plan.

    Real execution requires a successful GPU preflight unless ``preflight`` is
    explicitly set false.  The default automatically disables that gate for
    ``mock`` runs, because mock tests intentionally do not require hardware.
    ``dry_run`` never creates a lock, status file, child process, or result
    directory.
    """

    manifest = Path(manifest_path)
    result = Path(result_root)
    wham = Path(wham_root)
    units = load_manifest(manifest)
    normalized_gpus = _normalize_gpu_ids(gpu_ids)
    flags = _normalize_runner_flags(runner_flags)
    generated_run_id = run_id or str(uuid.uuid4())

    if dry_run:
        plan = dry_run_plan(
            manifest,
            result,
            wham,
            workers_per_gpu=workers_per_gpu,
            gpu_ids=normalized_gpus,
            mock=mock,
            dry_run_child=True if not dry_run_child else dry_run_child,
            runner_flags=flags,
            resume=resume,
            retry_failed=retry_failed,
            python_executable=python_executable,
        )
        summary = {
            "total": len(units),
            "runnable": plan["runnable_units"],
            "skipped": plan["skipped_units"],
            "slot_count": plan["slot_count"],
            "failures": [],
        }
        return ControllerResult(
            status="dry_run",
            exit_code=0,
            run_id=generated_run_id,
            summary=summary,
            plan=plan,
            outcomes=tuple(),
            status_path=None,
        )

    result.mkdir(parents=True, exist_ok=True)
    lock = ControllerLock(
        lock_path or result / "controller.lock",
        run_id=generated_run_id,
        stale_after=stale_lock_after,
    )
    lock.acquire()
    final_status_path = Path(status_path) if status_path else result / "controller_status.json"
    terminal_records = {unit.index: read_unit_terminal(result, unit) for unit in units}
    runnable: list[ManifestUnit] = []
    skipped: list[UnitOutcome] = []
    for unit in units:
        terminal_path, record = terminal_records[unit.index]
        status = normalize_terminal_status(
            record.get("status", record.get("state")) if record else None
        )
        if resume and _is_terminal_record(record) and not (
            retry_failed and status == "failed"
        ):
            skipped.append(_skip_outcome(unit, terminal_path, record or {}))
        else:
            runnable.append(unit)

    slots = _retain_runnable_units(
        build_slot_plan(
            tuple(unit.index for unit in units),
            workers_per_gpu=workers_per_gpu,
            gpu_ids=normalized_gpus,
        ),
        (unit.index for unit in runnable),
    )
    plan = _plan_for(
        units,
        runnable,
        slots,
        manifest_path=manifest,
        result_root=result,
        wham_root=wham,
        dry_run_child=dry_run_child,
        mock=mock,
        runner_flags=flags,
        python_executable=python_executable,
        terminal_records=terminal_records,
    )

    status_book = _StatusBook(
        final_status_path,
        {
            "schema_version": 1,
            "controller": "wham-independent-process",
            "run_id": generated_run_id,
            "state": "planned",
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "result_root": str(result),
            "wham_root": str(wham),
            "workers_per_gpu": int(workers_per_gpu),
            "gpu_ids": list(normalized_gpus),
            "slot_count": len(slots),
            "plan": plan,
            "outcomes": [outcome.to_dict() for outcome in skipped],
        },
    )
    status_book.update(state="preflight")

    if preflight is None:
        preflight = not mock
    if wait_for_gpu:
        preflight = True

    try:
        if preflight and runnable:
            probe_result = wait_for_gpus(
                normalized_gpus,
                timeout=None if wait_for_gpu and gpu_timeout is None else gpu_timeout,
                poll_interval=gpu_poll_interval,
                probe=gpu_probe,
            )
            status_book.update(preflight={"status": "passed", "probe": probe_result})
        elif not runnable:
            status_book.update(
                preflight={"status": "skipped", "reason": "no runnable units"}
            )
        else:
            status_book.update(preflight={"status": "skipped", "reason": "disabled"})
    except Exception as exc:
        failure = {
            "type": "preflight",
            "error": str(exc),
            "status": "failed",
        }
        summary = _summary_for(
            skipped,
            total_units=len(units),
            slot_count=len(slots),
            extra_failures=[failure],
        )
        status_book.update(
            state="failed",
            finished_at=_utc_now(),
            summary=summary,
            outcomes=[outcome.to_dict() for outcome in skipped],
        )
        lock.release()
        return ControllerResult(
            status="failed",
            exit_code=2,
            run_id=generated_run_id,
            summary=summary,
            plan=plan,
            outcomes=tuple(skipped),
            status_path=str(final_status_path),
        )

    if not runnable:
        summary = _summary_for(
            skipped,
            total_units=len(units),
            slot_count=len(slots),
        )
        final_state = "failed" if summary["failure_count"] else "succeeded"
        final_exit_code = 1 if summary["failure_count"] else 0
        status_book.update(
            state=final_state,
            finished_at=_utc_now(),
            summary=summary,
            outcomes=[outcome.to_dict() for outcome in skipped],
        )
        lock.release()
        return ControllerResult(
            status=final_state,
            exit_code=final_exit_code,
            run_id=generated_run_id,
            summary=summary,
            plan=plan,
            outcomes=tuple(skipped),
            status_path=str(final_status_path),
        )

    stop = stop_event or threading.Event()
    process_factory = popen_factory or subprocess.Popen
    active_processes: dict[int, Any] = {}
    active_processes_lock = threading.Lock()
    outcomes: list[UnitOutcome] = list(skipped)
    units_by_index = {unit.index: unit for unit in runnable}
    status_book.update(state="running")

    try:
        with _SignalStop(stop):
            with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="wham-slot") as executor:
                futures: dict[Future[list[UnitOutcome]], Slot] = {
                    executor.submit(
                        _run_slot,
                        slot,
                        units_by_index,
                        manifest_path=manifest,
                        result_root=result,
                        wham_root=wham,
                        run_id=generated_run_id,
                        mock=mock,
                        dry_run_child=dry_run_child,
                        runner_flags=flags,
                        python_executable=python_executable,
                        stop_event=stop,
                        stop_grace=stop_grace,
                        popen_factory=process_factory,
                        active_processes=active_processes,
                        active_processes_lock=active_processes_lock,
                    ): slot
                    for slot in slots
                    if slot.unit_indices
                }
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        rows = []
                        for unit_index in slot.unit_indices:
                            unit = units_by_index[unit_index]
                            command = build_runner_command(
                                manifest,
                                unit.index,
                                result,
                                wham,
                                dry_run=dry_run_child,
                                mock=mock,
                                runner_flags=flags,
                                python_executable=python_executable,
                            )
                            log_path = result / "logs" / f"slot-{slot.slot_index:03d}" / f"unit-{unit.index:06d}.log"
                            outcome = _make_failure_outcome(
                                unit,
                                slot,
                                command,
                                log_path,
                                f"slot worker failed: {exc}",
                            )
                            outcome.terminal_path = str(
                                _write_unit_terminal(
                                    result, unit, outcome, run_id=generated_run_id
                                )
                            )
                            rows.append(outcome)
                    outcomes.extend(rows)
                    outcomes.sort(key=lambda outcome: outcome.index)
                    status_book.update(
                        state="stopping" if stop.is_set() else "running",
                        outcomes=[outcome.to_dict() for outcome in outcomes],
                        summary=_summary_for(
                            outcomes,
                            total_units=len(units),
                            slot_count=len(slots),
                            interrupted=stop.is_set(),
                        ),
                    )
    except KeyboardInterrupt:
        stop.set()
        # Worker polling handles child termination; status below records the
        # interruption even if Python raised while collecting futures.
    finally:
        stop_requested = stop.is_set()
        summary = _summary_for(
            outcomes,
            total_units=len(units),
            slot_count=len(slots),
            interrupted=stop_requested,
        )
        if stop_requested:
            final_state = "interrupted"
            exit_code = 130
        elif summary["failure_count"]:
            final_state = "failed"
            exit_code = 1
        else:
            final_state = "succeeded"
            exit_code = 0
        outcomes.sort(key=lambda outcome: outcome.index)
        status_book.update(
            state=final_state,
            finished_at=_utc_now(),
            summary=summary,
            outcomes=[outcome.to_dict() for outcome in outcomes],
        )
        lock.release()

    return ControllerResult(
        status=final_state,
        exit_code=exit_code,
        run_id=generated_run_id,
        summary=summary,
        plan=plan,
        outcomes=tuple(outcomes),
        status_path=str(final_status_path),
    )


def calibration_plan(
    candidates: Iterable[int] = (1, 2, 3),
    *,
    gpu_ids: Iterable[int] = DEFAULT_GPU_IDS,
) -> tuple[dict[str, Any], ...]:
    """Describe safe candidate slot counts without launching any worker.

    Actual memory/throughput calibration is intentionally an integration step
    after the active Fast-MHR task releases all GPUs.  This helper only emits
    the candidate shape and can therefore be used safely during development.
    """

    normalized_gpus = _normalize_gpu_ids(gpu_ids)
    values: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            workers = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"invalid calibration candidate: {candidate!r}") from exc
        if workers < 1:
            raise ControllerError("calibration candidates must be >= 1")
        values.append(
            {
                "workers_per_gpu": workers,
                "gpu_ids": list(normalized_gpus),
                "slot_count": len(normalized_gpus) * workers,
                "launch_allowed": False,
            }
        )
    return tuple(values)


__all__ = [
    "DEFAULT_GPU_IDS",
    "TERMINAL_STATUSES",
    "ControllerError",
    "ManifestError",
    "PreflightError",
    "ControllerLockError",
    "ManifestUnit",
    "Slot",
    "UnitOutcome",
    "ControllerResult",
    "ControllerLock",
    "load_manifest",
    "normalize_terminal_status",
    "terminal_path_for_unit",
    "read_unit_terminal",
    "build_slot_plan",
    "assign_shards",
    "slot_plan",
    "shard_units",
    "build_runner_command",
    "worker_environment",
    "build_worker_environment",
    "wait_for_gpus",
    "preflight_gpus",
    "dry_run_plan",
    "run_controller",
    "calibration_plan",
]
