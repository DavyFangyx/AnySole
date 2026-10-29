"""Source inventory and manifest helpers for the local WHAM baseline.

The WHAM demo consumes a *video*, while the Tactile source is stored as dated
RGB image trees.  This module deliberately stops at a small, dependency-free
description of those trees.  :mod:`wham_baseline.runner` is responsible for
preparing a video below a result directory before invoking the upstream demo.

No model package (and no image/video package) is imported here.  Inventory can
therefore run in the normal CPU environment and never writes below
``1_Dataset``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_ID = "tactile.wham_baseline.inventory"
SCHEMA_VERSION = 1
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_DATE_RE = re.compile(r"^\d{8}$")
_FRAME_RE = re.compile(
    r"^frame(?P<frame>\d+)_(?P<stream>\d+)_(?P<timestamp>\d{6}(?:\.\d+)?)\.(?P<suffix>[A-Za-z0-9]+)$",
    re.IGNORECASE,
)
_GENERIC_FRAME_RE = re.compile(r"^frame[-_]?(\d+)", re.IGNORECASE)


class InventoryError(ValueError):
    """Raised when a source tree or manifest cannot satisfy the contract."""


@dataclass(frozen=True)
class FrameName:
    """Filename-derived provenance for one RGB frame."""

    frame: int | None
    stream: int | None
    timestamp_seconds: float | None


@dataclass(frozen=True)
class FrameRecord:
    """A source frame represented with paths relative to ``dataset_root``."""

    date: str
    method: str
    session: str
    camera: str
    frame_index: int
    frame: int | None
    filename: str
    relative_path: str
    timestamp_seconds: float | None
    stream: int | None
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetUnit:
    """One coherent WHAM input unit (one date/method/session/camera tree)."""

    unit_id: str
    date: str
    method: str
    session: str
    camera: str
    source_relpath: str
    frame_count: int
    available_frame_count: int
    source_image_count: int
    truncated: bool
    frame_start: int | None
    frame_end: int | None
    frames: tuple[FrameRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["frames"] = [frame.as_dict() for frame in self.frames]
        return value


# Names used by callers that prefer the term used in the task brief.
InputUnit = DatasetUnit
ManifestUnit = DatasetUnit


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    """Return a comparable key that orders embedded numbers naturally."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def parse_frame_filename(name: str) -> FrameName:
    """Parse common Tactile frame names without touching the file.

    Current source files look like ``frame406_1_162902.060.jpg``.  A small
    fallback accepts names such as ``frame0001.jpg`` so synthetic fixtures and
    future camera exports can still be inventoried while retaining the exact
    filename as provenance.  A name without a frame token is rejected because
    assigning an arbitrary frame number would make resume/sharding ambiguous.
    """

    basename = Path(name).name
    match = _FRAME_RE.fullmatch(basename)
    if match is not None:
        frame = int(match.group("frame"))
        stream = int(match.group("stream"))
        token = match.group("timestamp")
        hours = int(token[:2])
        minutes = int(token[2:4])
        seconds = float(token[4:])
        if hours >= 24 or minutes >= 60 or seconds >= 60:
            raise InventoryError(f"invalid HHMMSS timestamp in {basename!r}")
        return FrameName(
            frame=frame,
            stream=stream,
            timestamp_seconds=hours * 3600.0 + minutes * 60.0 + seconds,
        )

    fallback = _GENERIC_FRAME_RE.match(Path(basename).stem)
    if fallback is None:
        raise InventoryError(
            f"unexpected RGB filename {basename!r}; expected frame<N>_<stream>_<HHMMSS.fraction>.jpg"
        )
    return FrameName(frame=int(fallback.group(1)), stream=None, timestamp_seconds=None)


# Compatibility alias for code that uses the longer spelling.
parse_frame_name = parse_frame_filename


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without adding a self-referential field to it."""

    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


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


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest atomically in stable, human-readable JSON form."""

    destination = Path(path)
    if destination.suffix.casefold() == ".jsonl":
        return write_manifest_jsonl(destination, manifest)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_bytes(destination, payload.encode("utf-8"))
    return destination


def write_manifest_jsonl(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write one self-contained unit per line for queue/controller consumers.

    Each line keeps the metadata needed to resolve source paths, while the
    loader below reconstructs the equivalent object-style manifest.  This
    avoids forcing a controller to load a potentially large JSON array just to
    inspect or shard units.
    """

    destination = Path(path)
    units = manifest_units(manifest)
    metadata = {
        "schema_id": manifest.get("schema_id"),
        "schema_version": manifest.get("schema_version"),
        "dataset_root": manifest.get("dataset_root"),
        "source_tree": manifest.get("source_tree"),
        "selection": manifest.get("selection"),
        "complete": manifest.get("complete"),
        "summary": manifest.get("summary"),
        "issues": manifest.get("issues", []),
    }
    lines: list[str] = []
    for unit in units:
        record = dict(metadata)
        record["record_type"] = "unit"
        record.update(unit)
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write_bytes(destination, payload)
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an inventory manifest."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise InventoryError(f"cannot read manifest {source}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _load_manifest_jsonl(source, text)
    else:
        # A one-unit JSONL file is also valid JSON syntax.  Distinguish it
        # from the object-style manifest by its missing top-level ``units``
        # field and normalize it through the JSONL path.
        if isinstance(value, dict) and "units" not in value and (
            value.get("record_type") == "unit" or "unit_id" in value
        ):
            value = _load_manifest_jsonl(source, text)
    if not isinstance(value, dict):
        raise InventoryError(f"manifest {source} must contain a JSON object")
    if value.get("schema_id") != SCHEMA_ID:
        raise InventoryError(
            f"manifest {source} has schema_id {value.get('schema_id')!r}, expected {SCHEMA_ID!r}"
        )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise InventoryError(
            f"manifest {source} has schema_version {value.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    units = value.get("units")
    if not isinstance(units, list):
        raise InventoryError(f"manifest {source} must contain a units list")
    for position, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise InventoryError(f"manifest unit {position} must be an object")
        for field in ("unit_id", "date", "method", "session", "camera", "frames"):
            if field not in unit:
                raise InventoryError(f"manifest unit {position} is missing {field!r}")
        if not isinstance(unit["frames"], list):
            raise InventoryError(f"manifest unit {position} frames must be a list")
    return value


def _load_manifest_jsonl(source: Path, text: str) -> dict[str, Any]:
    """Load direct-unit JSONL records and normalize them to object form."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InventoryError(f"invalid JSONL manifest {source} line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise InventoryError(f"JSONL manifest {source} line {line_number} must be an object")
        records.append(record)
    if not records:
        raise InventoryError(f"JSONL manifest {source} is empty")
    first = records[0]
    units: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_type") not in (None, "unit"):
            continue
        nested = record.get("unit")
        unit = dict(nested) if isinstance(nested, dict) else dict(record)
        unit.pop("record_type", None)
        # Metadata keys are deliberately retained only at the top level.
        for key in ("schema_id", "schema_version", "dataset_root", "source_tree", "selection", "complete", "summary", "issues"):
            unit.pop(key, None)
        units.append(unit)
    if not units:
        raise InventoryError(f"JSONL manifest {source} contains no unit records")
    frame_count = sum(int(unit.get("frame_count", len(unit.get("frames", [])))) for unit in units)
    available_frame_count = sum(int(unit.get("available_frame_count", unit.get("frame_count", 0))) for unit in units)
    source_image_count = sum(int(unit.get("source_image_count", unit.get("frame_count", 0))) for unit in units)
    summary = first.get("summary")
    if not isinstance(summary, dict):
        summary = {
            "unit_count": len(units),
            "frame_count": frame_count,
            "available_frame_count": available_frame_count,
            "source_image_count": source_image_count,
            "issue_count": 0,
        }
    issues = first.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return {
        "schema_id": first.get("schema_id"),
        "schema_version": first.get("schema_version"),
        "dataset_root": first.get("dataset_root"),
        "source_tree": first.get("source_tree", "dated/RGB/<method>/<session>/<camera>"),
        "selection": first.get("selection"),
        "complete": first.get("complete", True),
        "units": units,
        "summary": summary,
        "issues": issues,
    }


# Compatibility alias used by some controllers.
read_manifest = load_manifest


def _selected_values(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    selected = {str(value) for value in values}
    return selected


def _iter_directories(path: Path) -> list[Path]:
    try:
        return sorted((entry for entry in path.iterdir() if entry.is_dir()), key=lambda item: natural_key(item.name))
    except OSError as exc:
        raise InventoryError(f"cannot inspect directory {path}: {exc}") from exc


def _image_files(path: Path) -> list[Path]:
    try:
        return sorted(
            (
                entry
                for entry in path.iterdir()
                if entry.is_file() and entry.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda item: natural_key(item.name),
        )
    except OSError as exc:
        raise InventoryError(f"cannot inspect camera directory {path}: {exc}") from exc


def _frame_sort_key(record: tuple[Path, FrameName]) -> tuple[Any, ...]:
    path, parsed = record
    # Parsed frame IDs are primary.  Timestamps and names break ties without
    # depending on directory enumeration order or filesystem mtimes.
    return (
        0 if parsed.frame is not None else 1,
        parsed.frame if parsed.frame is not None else 0,
        parsed.timestamp_seconds if parsed.timestamp_seconds is not None else float("inf"),
        natural_key(path.name),
    )


def _make_unit(
    *,
    dataset_root: Path,
    date: str,
    method: str,
    session: str,
    camera: str,
    camera_root: Path,
    max_frames_per_unit: int | None,
    issues: list[dict[str, str]],
    strict: bool,
) -> DatasetUnit | None:
    image_paths = _image_files(camera_root)
    if not image_paths:
        return None

    parsed: list[tuple[Path, FrameName]] = []
    for path in image_paths:
        try:
            frame_name = parse_frame_filename(path.name)
        except InventoryError as exc:
            issue = {"path": path.relative_to(dataset_root).as_posix(), "reason": str(exc)}
            issues.append(issue)
            if strict:
                raise
            continue
        parsed.append((path, frame_name))
    if not parsed:
        return None

    parsed.sort(key=_frame_sort_key)
    available_frame_count = len(parsed)
    if max_frames_per_unit is None:
        selected = parsed
    else:
        selected = parsed[:max_frames_per_unit]
    frames: list[FrameRecord] = []
    source_relpath = camera_root.relative_to(dataset_root).as_posix()
    for frame_index, (path, frame_name) in enumerate(selected):
        frames.append(
            FrameRecord(
                date=date,
                method=method,
                session=session,
                camera=camera,
                frame_index=frame_index,
                frame=frame_name.frame,
                filename=path.name,
                relative_path=path.relative_to(dataset_root).as_posix(),
                timestamp_seconds=frame_name.timestamp_seconds,
                stream=frame_name.stream,
                size_bytes=path.stat().st_size,
            )
        )
    if not frames:
        return None
    unit_id = "/".join((date, method, session, camera))
    return DatasetUnit(
        unit_id=unit_id,
        date=date,
        method=method,
        session=session,
        camera=camera,
        source_relpath=source_relpath,
        frame_count=len(frames),
        available_frame_count=available_frame_count,
        source_image_count=len(image_paths),
        truncated=len(frames) < available_frame_count,
        frame_start=frames[0].frame,
        frame_end=frames[-1].frame,
        frames=tuple(frames),
    )


def build_inventory(
    dataset_root: str | Path,
    *,
    methods: Iterable[str] | None = None,
    dates: Iterable[str] | None = None,
    max_units: int | None = None,
    max_frames_per_unit: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Build a deterministic inventory from dated ``RGB`` trees.

    Paths in the manifest are relative to the resolved dataset root.  The root
    itself is retained so a runner can resolve a manifest without a second
    source-selection argument.  ``max_units`` and ``max_frames_per_unit`` take
    the first items *after* deterministic date/method/session/camera/frame
    ordering, making them suitable for bounded smoke runs.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise InventoryError(f"dataset root is not a directory: {root}")
    if max_units is not None and max_units < 0:
        raise InventoryError("max_units must be non-negative")
    if max_frames_per_unit is not None and max_frames_per_unit < 0:
        raise InventoryError("max_frames_per_unit must be non-negative")
    selected_methods = _selected_values(methods)
    selected_dates = _selected_values(dates)
    issues: list[dict[str, str]] = []
    units: list[DatasetUnit] = []

    try:
        date_roots = sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.is_dir() and _DATE_RE.fullmatch(entry.name) and (selected_dates is None or entry.name in selected_dates)
            ),
            key=lambda item: item.name,
        )
    except OSError as exc:
        raise InventoryError(f"cannot inspect dataset root {root}: {exc}") from exc

    stop = False
    for date_root in date_roots:
        rgb_root = date_root / "RGB"
        if not rgb_root.is_dir():
            issues.append({"path": date_root.relative_to(root).as_posix(), "reason": "missing RGB directory"})
            continue
        for method_root in _iter_directories(rgb_root):
            if selected_methods is not None and method_root.name not in selected_methods:
                continue
            for session_root in _iter_directories(method_root):
                for camera_root in _iter_directories(session_root):
                    unit = _make_unit(
                        dataset_root=root,
                        date=date_root.name,
                        method=method_root.name,
                        session=session_root.name,
                        camera=camera_root.name,
                        camera_root=camera_root,
                        max_frames_per_unit=max_frames_per_unit,
                        issues=issues,
                        strict=strict,
                    )
                    if unit is None:
                        continue
                    units.append(unit)
                    if max_units is not None and len(units) >= max_units:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    unit_dicts = [unit.as_dict() for unit in units]
    frame_count = sum(unit.frame_count for unit in units)
    available_frame_count = sum(unit.available_frame_count for unit in units)
    source_image_count = sum(unit.source_image_count for unit in units)
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "source_tree": "dated/RGB/<method>/<session>/<camera>",
        "selection": {
            "dates": sorted(selected_dates) if selected_dates is not None else None,
            "methods": sorted(selected_methods) if selected_methods is not None else None,
            "max_units": max_units,
            "max_frames_per_unit": max_frames_per_unit,
            "strict": strict,
        },
        "complete": max_units is None and max_frames_per_unit is None,
        "units": unit_dicts,
        "summary": {
            "unit_count": len(unit_dicts),
            "frame_count": frame_count,
            "available_frame_count": available_frame_count,
            "source_image_count": source_image_count,
            "issue_count": len(issues),
        },
        "issues": issues,
    }


# Descriptive aliases retained as a stable controller-facing surface.
build_manifest = build_inventory
build_source_inventory = build_inventory


def manifest_units(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return manifest units after validating the top-level schema."""

    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("schema_version") != SCHEMA_VERSION:
        raise InventoryError("unsupported WHAM inventory manifest")
    units = manifest.get("units")
    if not isinstance(units, list):
        raise InventoryError("manifest units must be a list")
    return units


def unit_at(manifest: Mapping[str, Any], unit_index: int) -> Mapping[str, Any]:
    """Resolve one unit index with a useful bounds error."""

    if isinstance(unit_index, bool) or not isinstance(unit_index, int):
        raise InventoryError("unit_index must be an integer")
    units = manifest_units(manifest)
    if unit_index < 0 or unit_index >= len(units):
        raise InventoryError(f"unit_index {unit_index} outside manifest range [0, {len(units)})")
    return units[unit_index]


def unit_frame_paths(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any] | None = None,
    *,
    dataset_root: str | Path | None = None,
) -> list[Path]:
    """Resolve one unit's relative frame paths without creating any files."""

    root_value = dataset_root if dataset_root is not None else manifest.get("dataset_root")
    if not root_value:
        raise InventoryError("manifest does not specify dataset_root")
    root = Path(root_value).expanduser().resolve()
    chosen = unit if unit is not None else unit_at(manifest, 0)
    frames = chosen.get("frames")
    if not isinstance(frames, list):
        raise InventoryError("unit frames must be a list")
    paths: list[Path] = []
    for frame in frames:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("relative_path"), str):
            raise InventoryError("each frame must contain a relative_path")
        relative = Path(frame["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise InventoryError(f"unsafe frame relative_path {relative}")
        paths.append(root / relative)
    return paths


__all__ = [
    "DatasetUnit",
    "FrameName",
    "FrameRecord",
    "IMAGE_SUFFIXES",
    "InputUnit",
    "InventoryError",
    "ManifestUnit",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "build_inventory",
    "build_manifest",
    "build_source_inventory",
    "load_manifest",
    "manifest_sha256",
    "manifest_units",
    "natural_key",
    "parse_frame_filename",
    "parse_frame_name",
    "read_manifest",
    "unit_at",
    "unit_frame_paths",
    "write_manifest",
]
