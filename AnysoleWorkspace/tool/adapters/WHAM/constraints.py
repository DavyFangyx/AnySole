"""Target-person constraint preparation and camera-metadata audit for WHAM.

The source annotation tables are Parquet and are indexed once with the Tactile
environment.  GPU workers then query a compact SQLite file using only the
Python standard library, keeping PyArrow out of the legacy WHAM environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence


INDEX_SCHEMA_ID = "tactile.wham_baseline.annotation-index"
INDEX_SCHEMA_VERSION = 1
TARGET_BOX_SCHEMA_ID = "tactile.wham_baseline.target-bboxes"
TARGET_BOX_SCHEMA_VERSION = 1
_DATE_RE = re.compile(r"^\d{8}$")
_CAMERA_RE = re.compile(r"^(?:camera_|cam)?(\d+)$", re.IGNORECASE)


class ConstraintError(RuntimeError):
    """Raised when calibration or target annotations cannot be joined safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _camera_number(value: Any) -> int:
    match = _CAMERA_RE.fullmatch(str(value).strip())
    if match is None:
        raise ConstraintError(f"invalid camera identifier {value!r}")
    number = int(match.group(1))
    if number < 1:
        raise ConstraintError(f"camera number must be positive, got {number}")
    return number


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ConstraintError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ConstraintError(f"{name} contains a non-finite value")
    return result


def load_camera_calibration(dataset_root: str | Path, date: str, camera: str) -> dict[str, Any]:
    """Load and validate one date/camera calibration without changing pixels."""

    root = Path(dataset_root).expanduser().resolve()
    if not _DATE_RE.fullmatch(str(date)):
        raise ConstraintError(f"invalid date {date!r}")
    path = root / str(date) / f"{date}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstraintError(f"cannot read calibration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConstraintError(f"calibration {path} must contain an object")
    image_size = _finite_vector(value.get("image_size_px"), 2, "image_size_px")
    width, height = (int(item) for item in image_size)
    if width <= 0 or height <= 0:
        raise ConstraintError("image_size_px values must be positive")
    cameras = value.get("cameras")
    if not isinstance(cameras, Mapping):
        raise ConstraintError(f"calibration {path} has no cameras object")
    number = _camera_number(camera)
    camera_key = f"cam{number}"
    raw = cameras.get(camera_key)
    if not isinstance(raw, Mapping):
        raise ConstraintError(f"calibration {path} has no {camera_key}")
    matrix = raw.get("K")
    if not isinstance(matrix, Sequence) or len(matrix) != 3:
        raise ConstraintError(f"{camera_key}.K must be 3x3")
    K = [_finite_vector(row, 3, f"{camera_key}.K") for row in matrix]
    if K[0][0] <= 0 or K[1][1] <= 0:
        raise ConstraintError(f"{camera_key}.K focal lengths must be positive")
    if any(abs(left - right) > 1e-9 for left, right in zip(K[2], (0.0, 0.0, 1.0))):
        raise ConstraintError(f"{camera_key}.K final row must be [0, 0, 1]")
    D_raw = raw.get("D")
    if not isinstance(D_raw, Sequence) or isinstance(D_raw, (str, bytes)) or len(D_raw) < 4:
        raise ConstraintError(f"{camera_key}.D must contain at least four coefficients")
    D = [float(item) for item in D_raw]
    if not all(math.isfinite(item) for item in D):
        raise ConstraintError(f"{camera_key}.D contains a non-finite value")
    R_raw = raw.get("R")
    if not isinstance(R_raw, Sequence) or len(R_raw) != 3:
        raise ConstraintError(f"{camera_key}.R must be 3x3")
    R = [_finite_vector(row, 3, f"{camera_key}.R") for row in R_raw]
    t = _finite_vector(raw.get("t"), 3, f"{camera_key}.t")
    pixel_coordinates = str(value.get("pixel_coordinates", "")).strip()
    if pixel_coordinates != "raw_distorted":
        raise ConstraintError(
            f"unsupported pixel_coordinates {pixel_coordinates!r}; expected 'raw_distorted'"
        )
    return {
        "schema_id": "tactile.wham_baseline.camera-calibration",
        "schema_version": 1,
        "source": str(path),
        "source_sha256": _sha256(path),
        "date": str(date),
        "camera": f"camera_{number}",
        "image_size_px": [width, height],
        "pixel_coordinates": pixel_coordinates,
        "K": K,
        "D": D,
        "R": R,
        "t": t,
        "world_to_camera": value.get("world_to_camera"),
        "mocap_raw_to_checkerboard_world": value.get("mocap_raw_to_checkerboard_world"),
        "inference_policy": {
            "camera_parameters": "audit only; K/D/R/t are not passed to WHAM",
            "intrinsics": "WHAM retains its image-size focal/principal-point heuristic",
            "distortion": "raw_distorted pixels are unchanged; D is not applied",
            "extrinsics": "R/t are not applied; WHAM output is not checkerboard-anchored",
        },
    }


def build_annotation_index(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Atomically build a read-only worker index from date-local Parquet files."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConstraintError("building the annotation index requires pyarrow in Tactile") from exc

    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    requested = None if dates is None else {str(item) for item in dates}
    available = sorted(
        path.name for path in root.iterdir() if path.is_dir() and _DATE_RE.fullmatch(path.name)
    )
    selected = [date for date in available if requested is None or date in requested]
    missing = sorted((requested or set()) - set(selected))
    if missing:
        raise ConstraintError(f"requested dates are absent: {missing}")
    if not selected:
        raise ConstraintError("no date roots selected for annotation indexing")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    row_count = 0
    sources: list[dict[str, Any]] = []
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE annotations (
                date TEXT NOT NULL,
                session TEXT NOT NULL,
                camera INTEGER NOT NULL,
                frame INTEGER NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                segmentation_confidence REAL,
                mask_relpath TEXT NOT NULL,
                PRIMARY KEY (date, session, camera, frame)
            ) WITHOUT ROWID;
            """
        )
        insert_sql = """
            INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for date in selected:
            parquet_path = root / date / "annotations" / "records.parquet"
            if not parquet_path.is_file():
                raise ConstraintError(f"missing annotation table {parquet_path}")
            parquet = pq.ParquetFile(parquet_path)
            date_rows = 0
            for batch in parquet.iter_batches(
                columns=(
                    "date", "session", "camera", "frame", "bbox_xyxy_px",
                    "segmentation_confidence", "mask_relpath",
                ),
                batch_size=65536,
            ):
                records: list[tuple[Any, ...]] = []
                for row in batch.to_pylist():
                    bbox = _finite_vector(row["bbox_xyxy_px"], 4, "bbox_xyxy_px")
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        raise ConstraintError(f"invalid bbox for {date}/{row['session']}/{row['frame']}")
                    confidence = row.get("segmentation_confidence")
                    confidence = None if confidence is None else float(confidence)
                    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
                        raise ConstraintError("segmentation_confidence must be in [0,1]")
                    row_date = str(row["date"])
                    if row_date != date:
                        raise ConstraintError(f"annotation date {row_date} does not match {date}")
                    records.append(
                        (
                            date,
                            str(row["session"]),
                            _camera_number(row["camera"]),
                            int(row["frame"]),
                            *bbox,
                            confidence,
                            str(row["mask_relpath"]),
                        )
                    )
                connection.executemany(insert_sql, records)
                date_rows += len(records)
            row_count += date_rows
            sources.append(
                {
                    "date": date,
                    "path": str(parquet_path),
                    "sha256": _sha256(parquet_path),
                    "rows": date_rows,
                }
            )
        metadata = {
            "schema_id": INDEX_SCHEMA_ID,
            "schema_version": INDEX_SCHEMA_VERSION,
            "dataset_root": str(root),
            "dates": selected,
            "row_count": row_count,
            "sources": sources,
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise ConstraintError(f"SQLite integrity check failed: {check}")
    except Exception:
        connection.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        connection.close()
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    result = dict(metadata)
    result["output"] = str(destination)
    result["output_sha256"] = _sha256(destination)
    return result


def read_index_metadata(index_path: str | Path) -> dict[str, Any]:
    source = Path(index_path).expanduser().resolve()
    if not source.is_file():
        raise ConstraintError(f"annotation index is not a file: {source}")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    metadata = {str(key): json.loads(value) for key, value in rows}
    if metadata.get("schema_id") != INDEX_SCHEMA_ID or metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ConstraintError(f"unsupported annotation index schema in {source}")
    return metadata


def write_target_bbox_sidecar(
    index_path: str | Path,
    unit: Mapping[str, Any],
    input_root: str | Path,
) -> Path:
    """Join one manifest unit's selected frames to target-person annotations."""

    source = Path(index_path).expanduser().resolve()
    metadata = read_index_metadata(source)
    date = str(unit.get("date", ""))
    session = str(unit.get("session", ""))
    camera = _camera_number(unit.get("camera", ""))
    frames = unit.get("frames")
    if not isinstance(frames, Sequence) or not frames:
        raise ConstraintError(f"unit {unit.get('unit_id')} has no frames")
    joined: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        for video_index, frame_record in enumerate(frames):
            if not isinstance(frame_record, Mapping) or frame_record.get("frame") is None:
                raise ConstraintError(f"unit {unit.get('unit_id')} has a frame without a source frame number")
            source_frame = int(frame_record["frame"])
            row = connection.execute(
                """
                SELECT bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       segmentation_confidence, mask_relpath
                FROM annotations
                WHERE date=? AND session=? AND camera=? AND frame=?
                """,
                (date, session, camera, source_frame),
            ).fetchone()
            if row is None:
                raise ConstraintError(
                    f"no target annotation for {date}/{session}/camera_{camera}/frame{source_frame}"
                )
            joined.append(
                {
                    "video_frame_index": video_index,
                    "source_frame": source_frame,
                    "source_filename": frame_record.get("filename"),
                    "bbox_xyxy_px": [float(item) for item in row[:4]],
                    "segmentation_confidence": None if row[4] is None else float(row[4]),
                    "mask_relpath": str(row[5]),
                }
            )
    payload = {
        "schema_id": TARGET_BOX_SCHEMA_ID,
        "schema_version": TARGET_BOX_SCHEMA_VERSION,
        "unit_id": unit.get("unit_id"),
        "date": date,
        "session": session,
        "camera": f"camera_{camera}",
        "coordinate_space": "raw_distorted_pixels_xyxy",
        "frame_count": len(joined),
        "index": {"path": str(source), "sha256": _sha256(source), "metadata": metadata},
        "frames": joined,
    }
    destination = Path(input_root).expanduser().resolve() / "target_bboxes.json"
    _atomic_write(destination, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return destination


def prepare_unit_constraints(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    input_root: str | Path,
    *,
    annotation_index: str | Path | None = None,
    use_target_bbox: bool = False,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    _ = manifest
    if use_target_bbox:
        if annotation_index is None:
            raise ConstraintError("use_target_bbox requires annotation_index")
        artifacts["target_bboxes"] = write_target_bbox_sidecar(annotation_index, unit, input_root)
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-index", help="build the SQLite annotation index")
    build.add_argument("--dataset-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--date", dest="dates", action="append")
    inspect = subparsers.add_parser("inspect-index", help="print index metadata")
    inspect.add_argument("--index", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-index":
        result = build_annotation_index(args.dataset_root, args.output, dates=args.dates)
    else:
        result = read_index_metadata(args.index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
