"""Resumable, source-isolated single-unit runner for the WHAM demo.

The upstream repository exposes ``demo.py --video ... --output_pth ...``.  It
does not expose a light-weight import boundary: importing ``wham_api`` loads
Torch, OpenCV, detector models, and the body model immediately.  This module
therefore prepares a video under the run result root and invokes the upstream
CLI lazily with :mod:`subprocess`.  Inventory and mock runs remain usable when
none of the WHAM dependencies or model weights are installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from .dataset import (
    InventoryError,
    build_inventory,
    load_manifest,
    manifest_sha256,
    unit_at,
    unit_frame_paths,
    write_manifest,
)
from .constraints import prepare_unit_constraints


RUN_SCHEMA_ID = "tactile.wham_baseline.unit-run"
RUN_SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
_UNIT_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


class RunnerError(RuntimeError):
    """Raised for invalid run configuration before a terminal status exists."""


@dataclass(frozen=True)
class RunResult:
    """Machine-readable result of one run attempt."""

    status: str
    unit_index: int
    unit_id: str
    result_root: Path
    unit_root: Path
    status_path: Path | None
    command_path: Path | None
    exit_code: int | None
    mode: str
    resumed: bool = False
    error: str | None = None
    command: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("result_root", "unit_root", "status_path", "command_path"):
            if value[field] is not None:
                value[field] = str(value[field])
        value["command"] = list(value["command"])
        value["succeeded"] = self.succeeded
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_value(manifest: Mapping[str, Any] | str | Path, unit_index: int | None = None) -> tuple[dict[str, Any], Path | None]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        from .manifest_index import index_path, unit_manifest
        if unit_index is not None and index_path(path).exists():
            return unit_manifest(path, unit_index), path.resolve()
        return load_manifest(path), path.expanduser().resolve()
    if not isinstance(manifest, Mapping):
        raise RunnerError("manifest must be a mapping or JSON path")
    # ``load_manifest`` checks the schema and required shape.  It accepts a
    # path, so perform the same top-level checks here without serialising to a
    # temporary file (important for dry-run).
    try:
        units = manifest["units"]
        schema_id = manifest["schema_id"]
        schema_version = manifest["schema_version"]
    except KeyError as exc:
        raise RunnerError(f"manifest is missing {exc.args[0]!r}") from exc
    if schema_id != "tactile.wham_baseline.inventory" or schema_version != 1 or not isinstance(units, list):
        raise RunnerError("unsupported WHAM inventory manifest")
    return dict(manifest), None


def _safe_unit_slug(unit_id: str, unit_index: int) -> str:
    slug = _UNIT_SLUG_RE.sub("-", unit_id).strip("-.") or "unit"
    # Keep paths manageable even if a future source introduces verbose names;
    # the numeric index remains the collision-resistant component.
    return f"{unit_index:06d}-{slug[:180]}"


def unit_root_for(result_root: str | Path, unit: Mapping[str, Any], unit_index: int) -> Path:
    """Return the output directory for a manifest unit."""

    root = Path(result_root).expanduser().resolve()
    marker = root / 'layout.json'
    if root.name == '.run' and marker.is_file() and json.loads(marker.read_text()).get('layout') == 'compact-v1':
        from .compact import short_name
        return root / 'units' / short_name(unit)
    unit_id = str(unit.get("unit_id", "unit"))
    return root / "units" / _safe_unit_slug(unit_id, unit_index)


def status_path_for(result_root: str | Path, unit: Mapping[str, Any], unit_index: int) -> Path:
    return unit_root_for(result_root, unit, unit_index) / "status.json"


def read_status(path: str | Path) -> dict[str, Any] | None:
    """Read an existing status record, returning ``None`` if absent."""

    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read status {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"status {source} must contain a JSON object")
    return value


load_status = read_status


def _validate_result_isolation(manifest: Mapping[str, Any], result_root: Path) -> None:
    dataset_value = manifest.get("dataset_root")
    if not dataset_value:
        raise RunnerError("manifest does not specify dataset_root")
    dataset_root = Path(str(dataset_value)).expanduser().resolve()
    result_root = result_root.resolve()
    if result_root == dataset_root or dataset_root in result_root.parents:
        raise RunnerError(
            f"result root {result_root} is inside source dataset {dataset_root}; output must be source-isolated"
        )


def _frame_list_text(paths: Sequence[Path]) -> str:
    """Encode paths for ffmpeg's concat demuxer."""

    lines: list[str] = []
    for path in paths:
        # concat files use single-quoted paths with backslash escapes.  Source
        # names are controlled, but escaping keeps the boundary safe for
        # synthetic tests and future exports.
        encoded = str(path.resolve()).replace("\\", "\\\\").replace("'", "'\\''")
        lines.append(f"file '{encoded}'")
    return "\n".join(lines) + ("\n" if lines else "")


def build_video_command(
    frames_file: str | Path,
    output_video: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    fps: float = 30.0,
) -> list[str]:
    """Build the image-sequence-to-video command used by the adapter."""

    if fps <= 0:
        raise RunnerError("fps must be positive")
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        # Declare the concat image stream rate on the input side.  Applying
        # ``-r`` only after ``-i`` converts ffmpeg's default 25 fps timeline
        # and duplicates 60 source frames into 72 frames at 30 fps.
        "-r",
        f"{fps:g}",
        "-i",
        str(Path(frames_file).resolve()),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(Path(output_video).resolve()),
    ]


def build_wham_command(
    wham_root: str | Path,
    video: str | Path,
    output_dir: str | Path,
    *,
    estimate_local_only: bool = True,
    save_pkl: bool = True,
    extra_args: Iterable[str] = (),
) -> list[str]:
    """Build the actual upstream ``demo.py`` invocation.

    The command intentionally targets the pinned checkout's public CLI rather
    than reproducing its detector/network internals in local code.
    """

    root = Path(wham_root).expanduser().resolve()
    command = [
        sys.executable,
        "demo.py",
        "--video",
        str(Path(video).resolve()),
        "--output_pth",
        str(Path(output_dir).resolve()),
    ]
    if estimate_local_only:
        command.append("--estimate_local_only")
    if save_pkl:
        command.append("--save_pkl")
    command.extend(str(item) for item in extra_args)
    # Keep the root argument observable for callers while running with cwd set
    # to that root.  ``root`` is otherwise used by run_unit's subprocess call.
    _ = root
    return command


def prepare_wham_video(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    unit_root: str | Path,
    *,
    fps: float = 30.0,
    ffmpeg_bin: str = "ffmpeg",
    log_path: str | Path | None = None,
) -> tuple[Path, list[str]]:
    """Materialise one source image sequence as a video below ``unit_root``.

    The source images are never copied or renamed.  The concat list and video
    are result-local preparation artifacts, so reruns cannot contaminate
    ``1_Dataset`` or another source/method.
    """

    root = Path(unit_root).expanduser().resolve()
    input_root = root / "input"
    frames_file = input_root / "frames.txt"
    output_video = input_root / "sequence.mp4"
    paths = unit_frame_paths(manifest, unit)
    if not paths:
        raise RunnerError(f"unit {unit.get('unit_id', '<unknown>')} has no frames")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RunnerError(f"manifest references missing source frame(s), first: {missing[0]}")
    _atomic_write_bytes(frames_file, _frame_list_text(paths).encode("utf-8"))
    # Keep the media suffix on the temporary path.  ffmpeg selects the output
    # muxer from the final extension, so a name ending only in ``.partial``
    # fails with "Unable to find a suitable output format".
    temporary_video = output_video.with_name(
        f".{output_video.stem}.{os.getpid()}.partial{output_video.suffix}"
    )
    try:
        temporary_video.unlink()
    except FileNotFoundError:
        pass
    command = build_video_command(frames_file, temporary_video, ffmpeg_bin=ffmpeg_bin, fps=fps)
    if log_path is None:
        completed = subprocess.run(command, cwd=root, check=False)
    else:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {shlex.join(command)}\n")
            handle.flush()
            completed = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RunnerError(f"video preparation failed with exit code {completed.returncode}")
    if not temporary_video.is_file():
        raise RunnerError("video preparation exited successfully but produced no video")
    os.replace(temporary_video, output_video)
    return output_video, command


def _source_metadata(manifest: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("unit_id", "date", "method", "session", "camera", "source_relpath", "frame_count", "available_frame_count")
    return {field: unit.get(field) for field in fields}


def _status_record(
    *,
    unit: Mapping[str, Any],
    unit_index: int,
    manifest: Mapping[str, Any],
    status: str,
    mode: str,
    unit_root: Path,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int | None = None,
    error: str | None = None,
    command: Sequence[str] | None = None,
    previous_status: str | None = None,
    attempt: int = 1,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_id": RUN_SCHEMA_ID,
        "schema_version": RUN_SCHEMA_VERSION,
        "status": status,
        "terminal": status in TERMINAL_STATUSES,
        "mode": mode,
        "unit_index": unit_index,
        "unit_id": unit.get("unit_id"),
        "source": _source_metadata(manifest, unit),
        "manifest_sha256": manifest.get('indexed_manifest_sha256') or manifest_sha256(manifest),
        "unit_root": str(unit_root),
        "started_at": started_at,
        "attempt": attempt,
    }
    if finished_at is not None:
        value["finished_at"] = finished_at
    if exit_code is not None:
        value["exit_code"] = exit_code
    if error is not None:
        value["error"] = error
    if command is not None:
        value["command"] = [str(item) for item in command]
    if previous_status is not None:
        value["previous_status"] = previous_status
    if artifacts is not None:
        value["artifacts"] = dict(artifacts)
    return value


def _result(
    *,
    status: str,
    unit_index: int,
    unit: Mapping[str, Any],
    result_root: Path,
    unit_root: Path,
    status_path: Path | None,
    command_path: Path | None,
    exit_code: int | None,
    mode: str,
    resumed: bool = False,
    error: str | None = None,
    command: Sequence[str] = (),
) -> RunResult:
    return RunResult(
        status=status,
        unit_index=unit_index,
        unit_id=str(unit.get("unit_id", "")),
        result_root=result_root,
        unit_root=unit_root,
        status_path=status_path,
        command_path=command_path,
        exit_code=exit_code,
        mode=mode,
        resumed=resumed,
        error=error,
        command=tuple(str(item) for item in command),
    )


def run_unit(
    manifest: Mapping[str, Any] | str | Path,
    unit_index: int,
    result_root: str | Path,
    wham_root: str | Path | None = None,
    *,
    dry_run: bool = False,
    mock: bool = False,
    mock_fail: bool = False,
    force: bool = False,
    fps: float = 30.0,
    ffmpeg_bin: str = "ffmpeg",
    estimate_local_only: bool = True,
    extra_args: Iterable[str] = (),
    annotation_index: str | Path | None = None,
    use_target_bbox: bool = False,
    validate_target_output: bool = False,
    config_hash: str | None = None,
) -> RunResult:
    """Execute, mock, or dry-run one manifest unit.

    A successful terminal status is resumable: later calls return it without
    re-running preparation or WHAM unless ``force=True``.  Failures are also
    terminal and retain their error; a subsequent call starts a new attempt so
    interrupted or repaired units can resume safely.
    """

    parsed_manifest, _manifest_path = _manifest_value(manifest, unit_index)
    try:
        unit = unit_at(parsed_manifest, unit_index)
    except InventoryError as exc:
        raise RunnerError(str(exc)) from exc
    result_path = Path(result_root).expanduser().resolve()
    _validate_result_isolation(parsed_manifest, result_path)
    unit_root = unit_root_for(result_path, unit, unit_index)
    status_path = unit_root / "status.json"
    command_path = unit_root / "command.json"
    existing = read_status(status_path)
    previous_status = str(existing.get("status")) if existing else None
    if existing and previous_status == "succeeded" and not force:
        existing_command = existing.get("command", ())
        if not isinstance(existing_command, (list, tuple)):
            existing_command = ()
        return _result(
            status="succeeded",
            unit_index=unit_index,
            unit=unit,
            result_root=result_path,
            unit_root=unit_root,
            status_path=status_path,
            command_path=command_path if command_path.exists() else None,
            exit_code=existing.get("exit_code"),
            mode=str(existing.get("mode", "unknown")),
            resumed=True,
            command=existing_command,
        )
    if dry_run:
        # Dry-run is intentionally read-only: no status, command, input, or
        # output artifact is created.  The returned unit_root shows where a
        # real attempt would be isolated.
        mode = "mock" if mock else "wham"
        planned_command: list[str] = []
        if not mock:
            if wham_root is None:
                raise RunnerError("wham_root is required for a WHAM dry-run")
            planned_extra_args = list(extra_args)
            if use_target_bbox:
                if annotation_index is None:
                    raise RunnerError("--use-target-bbox requires --annotation-index")
                planned_extra_args.extend(
                    ("--target_bboxes", str(unit_root / "input" / "target_bboxes.json"))
                )
            planned_command = build_wham_command(
                wham_root,
                unit_root / "input" / "sequence.mp4",
                unit_root / "wham_output",
                estimate_local_only=estimate_local_only,
                extra_args=planned_extra_args,
            )
        return _result(
            status="dry_run",
            unit_index=unit_index,
            unit=unit,
            result_root=result_path,
            unit_root=unit_root,
            status_path=None,
            command_path=None,
            exit_code=0,
            mode=mode,
            resumed=False,
            command=planned_command,
        )

    mode = "mock" if mock else "wham"
    started_at = _utc_now()
    attempt = int(existing.get("attempt", 0)) + 1 if existing else 1
    unit_root.mkdir(parents=True, exist_ok=True)
    running = _status_record(
        unit=unit,
        unit_index=unit_index,
        manifest=parsed_manifest,
        status="running",
        mode=mode,
        unit_root=unit_root,
        started_at=started_at,
        previous_status=previous_status,
        attempt=attempt,
    )
    running['owner_pid'] = os.getpid()
    if config_hash:
        running['config_hash'] = config_hash
    _atomic_write_json(status_path, running)
    log_path = unit_root / "run.log"
    _append_log(log_path, f"[{started_at}] start unit={unit.get('unit_id')} mode={mode} attempt={attempt}")

    command: list[str] = []
    exit_code: int | None = None
    artifacts: dict[str, Any] = {"log": str(log_path)}
    try:
        if force and (unit_root / 'wham_output').exists():
            # Do not reuse truncated/stale upstream preprocessing after an
            # interrupted or invalid inference. Preserve it as attempt evidence.
            archive = unit_root / 'attempts' / f'wham_output-before-attempt-{attempt}'
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise RunnerError(f'output evidence archive already exists: {archive}')
            os.replace(unit_root / 'wham_output', archive)
            artifacts['previous_output_archive'] = str(archive)
        if mock:
            command = ["mock", "--unit-index", str(unit_index), "--unit-id", str(unit.get("unit_id", ""))]
            _atomic_write_json(
                command_path,
                {
                    "mode": "mock",
                    "working_directory": str(unit_root),
                    "command": command,
                },
            )
            artifacts["command"] = str(command_path)
            if mock_fail:
                raise RunnerError("mock failure requested")
            mock_output = unit_root / "wham_output" / "mock_result.json"
            _atomic_write_json(
                mock_output,
                {
                    "schema_id": "tactile.wham_baseline.mock-result",
                    "unit_id": unit.get("unit_id"),
                    "unit_index": unit_index,
                    "frame_count": unit.get("frame_count"),
                    "source_relpath": unit.get("source_relpath"),
                    "status": "succeeded",
                },
            )
            artifacts["mock_result"] = str(mock_output)
            _append_log(log_path, f"[{_utc_now()}] mock success")
        else:
            if wham_root is None:
                raise RunnerError("wham_root is required for a WHAM run")
            wham_path = Path(wham_root).expanduser().resolve()
            demo_path = wham_path / "demo.py"
            if not wham_path.is_dir() or not demo_path.is_file():
                raise RunnerError(f"WHAM root does not contain demo.py: {wham_path}")
            video, preparation_command = prepare_wham_video(
                parsed_manifest,
                unit,
                unit_root,
                fps=fps,
                ffmpeg_bin=ffmpeg_bin,
                log_path=log_path,
            )
            artifacts["frames_file"] = str(unit_root / "input" / "frames.txt")
            artifacts["video"] = str(video)
            constraint_artifacts = prepare_unit_constraints(
                parsed_manifest,
                unit,
                unit_root / "input",
                annotation_index=annotation_index,
                use_target_bbox=use_target_bbox,
            )
            artifacts.update({key: str(path) for key, path in constraint_artifacts.items()})
            command_extra_args = list(extra_args)
            if use_target_bbox:
                if "--target_bboxes" in command_extra_args:
                    raise RunnerError("do not combine --use-target-bbox with --extra-arg=--target_bboxes")
                command_extra_args.extend(
                    ("--target_bboxes", str(constraint_artifacts["target_bboxes"]))
                )
            command = build_wham_command(
                wham_path,
                video,
                unit_root / "wham_output",
                estimate_local_only=estimate_local_only,
                extra_args=command_extra_args,
            )
            _atomic_write_json(
                command_path,
                {
                    "wham_root": str(wham_path),
                    "working_directory": str(wham_path),
                    "video_preparation_command": preparation_command,
                    "command": command,
                },
            )
            artifacts["command"] = str(command_path)
            _append_log(log_path, f"$ (cd {shlex.quote(str(wham_path))} && {shlex.join(command)})")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.flush()
                completed = subprocess.run(
                    command,
                    cwd=wham_path,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            exit_code = completed.returncode
            if exit_code != 0:
                raise RunnerError(f"WHAM demo failed with exit code {exit_code}")
            _append_log(log_path, f"[{_utc_now()}] WHAM success")
        if validate_target_output and not mock:
            from .validation import prediction_path, validate_prediction, validate_tracking
            validate_prediction(prediction_path(unit_root), int(unit['frame_count']))
            validate_tracking(unit_root, int(unit['frame_count']))
        terminal = _status_record(
            unit=unit,
            unit_index=unit_index,
            manifest=parsed_manifest,
            status="succeeded",
            mode=mode,
            unit_root=unit_root,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code if exit_code is not None else 0,
            command=command or None,
            previous_status=previous_status,
            attempt=attempt,
            artifacts=artifacts,
        )
        if config_hash:
            terminal['config_hash'] = config_hash
        terminal['owner_pid'] = os.getpid()
        _atomic_write_json(status_path, terminal)
        return _result(
            status="succeeded",
            unit_index=unit_index,
            unit=unit,
            result_root=result_path,
            unit_root=unit_root,
            status_path=status_path,
            command_path=command_path if command_path.exists() else None,
            exit_code=terminal.get("exit_code"),
            mode=mode,
            command=command,
        )
    except Exception as exc:  # terminal failure is an artifact, not an orphaned process
        error = str(exc)
        if not error:
            error = exc.__class__.__name__
        _append_log(log_path, f"[{_utc_now()}] failure: {error}")
        terminal = _status_record(
            unit=unit,
            unit_index=unit_index,
            manifest=parsed_manifest,
            status="failed",
            mode=mode,
            unit_root=unit_root,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code if exit_code is not None else 1,
            error=error,
            command=command or None,
            previous_status=previous_status,
            attempt=attempt,
            artifacts=artifacts,
        )
        terminal['owner_pid'] = os.getpid()
        if config_hash:
            terminal['config_hash'] = config_hash
        _atomic_write_json(status_path, terminal)
        return _result(
            status="failed",
            unit_index=unit_index,
            unit=unit,
            result_root=result_path,
            unit_root=unit_root,
            status_path=status_path,
            command_path=command_path if command_path.exists() else None,
            exit_code=terminal.get("exit_code"),
            mode=mode,
            error=error,
            command=command,
        )


run_manifest_unit = run_unit


def _comma_or_repeat(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory and run one WHAM input unit")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="build a deterministic RGB source manifest")
    inventory.add_argument("--dataset-root", default="1_Dataset", type=Path)
    inventory.add_argument("--manifest", "--output", "-o", default=Path("wham_manifest.jsonl"), type=Path)
    inventory.add_argument("--method", "--methods", dest="methods", action="append")
    inventory.add_argument("--date", "--dates", dest="dates", action="append")
    inventory.add_argument("--max-units", type=int, default=None)
    inventory.add_argument("--max-frames-per-unit", type=int, default=None)
    inventory.add_argument("--strict", action="store_true", help="fail on malformed source image names")

    run = commands.add_parser("run", help="run, mock, or dry-run one manifest unit")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--validate-target-output", action="store_true")
    run.add_argument("--config-hash")
    run.add_argument("--unit-index", required=True, type=int)
    run.add_argument("--result-root", required=True, type=Path)
    run.add_argument("--wham-root", default=Path("third_party/WHAM"), type=Path)
    modes = run.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    run.add_argument("--mock-fail", "--mock-failure", action="store_true")
    run.add_argument("--force", action="store_true", help="rerun even when status is succeeded")
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument("--ffmpeg-bin", default="ffmpeg")
    run.add_argument(
        "--estimate-global",
        action="store_true",
        help="allow the upstream global SLAM path (local-only is the safe default)",
    )
    run.add_argument(
        "--annotation-index",
        type=Path,
        default=None,
        help="SQLite index built by wham_baseline.constraints (required for target bbox)",
    )
    run.add_argument(
        "--use-target-bbox",
        action="store_true",
        help="bypass generic multi-person detection with the annotated target bbox",
    )
    run.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            manifest = build_inventory(
                args.dataset_root,
                methods=_comma_or_repeat(args.methods),
                dates=_comma_or_repeat(args.dates),
                max_units=args.max_units,
                max_frames_per_unit=args.max_frames_per_unit,
                strict=args.strict,
            )
            destination = write_manifest(args.manifest, manifest)
            print(
                json.dumps(
                    {
                        "status": "succeeded",
                        "manifest": str(destination.resolve()),
                        "manifest_sha256": manifest_sha256(manifest),
                        "summary": manifest["summary"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = run_unit(
            args.manifest,
            args.unit_index,
            args.result_root,
            args.wham_root,
            dry_run=args.dry_run,
            mock=args.mock,
            mock_fail=args.mock_fail,
            force=args.force,
            fps=args.fps,
            ffmpeg_bin=args.ffmpeg_bin,
            estimate_local_only=not args.estimate_global,
            extra_args=args.extra_args,
            annotation_index=args.annotation_index,
            use_target_bbox=args.use_target_bbox,
            validate_target_output=args.validate_target_output,
            config_hash=args.config_hash,
        )
        print(json.dumps(result.as_dict(), sort_keys=True))
        return 0 if result.status in {"succeeded", "dry_run"} else 1
    except (InventoryError, RunnerError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


__all__ = [
    "RUN_SCHEMA_ID",
    "RUN_SCHEMA_VERSION",
    "RunResult",
    "RunnerError",
    "TERMINAL_STATUSES",
    "build_video_command",
    "build_wham_command",
    "load_status",
    "main",
    "prepare_wham_video",
    "read_status",
    "run_manifest_unit",
    "run_unit",
    "status_path_for",
    "unit_root_for",
]


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
