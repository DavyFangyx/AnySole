"""Rebuild tactile pressure CSVs into segmented, uniform-time files."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable


from pwlib.paths import DEFAULT_RAW_ROOT_DIR, RECONSTRUCTED_OUTPUTS_DIR

DEFAULT_ROOT_DIR = DEFAULT_RAW_ROOT_DIR
OUTPUT_PARENT_DIR = RECONSTRUCTED_OUTPUTS_DIR
EXPECTED_CHANNELS = 48
EXPECTED_LEFT_NAMES = ("pressure_left.csv", "Pressure_left.csv")
EXPECTED_RIGHT_NAMES = ("pressure_right.csv", "Pressure_right.csv")
DATE_DIR_RE = re.compile(r"^\d{8}$")
SUBJECT_DIR_RE = re.compile(r"^S\d+$", re.IGNORECASE)
EXCLUDED_DATE_DIRS = {"20260419", "20260422"}
HEARTBEAT_PERIOD_US = 15_625.0
DEFAULT_FRAME_PERIOD_US = 25_000.0
RESAMPLED_FRAME_PERIOD_US = 25_000.0


@dataclass
class PressureRow:
    source_frame_idx: int
    source_t_us: float
    values: list[float]


@dataclass
class ManifestRow:
    block_type: str
    side: str
    name: str
    frame_idx_start: int
    frame_idx_end: int
    t_us_start: float
    t_us_end: float
    source_rows: int
    inserted_rows: int
    dt_hat_us: float
    source_t_us_start: float
    source_t_us_end: float
    prev_name: str = ""
    next_name: str = ""
    contact_transition_mode: str = ""


class SegmentGapError(ValueError):
    def __init__(self, boundary_t_us: float, message: str) -> None:
        super().__init__(message)
        self.boundary_t_us = boundary_t_us


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if resume:
            return path
        if not overwrite:
            raise SystemExit(f"Output directory already exists and is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_side_from_name(name: str) -> str:
    lower = name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return ""


def find_column(header: list[str], names: set[str]) -> int | None:
    lookup = {cell.strip().lower(): idx for idx, cell in enumerate(header)}
    for name in names:
        idx = lookup.get(name.lower())
        if idx is not None:
            return idx
    return None


def session_output_dir_for(output_root: Path, session_dir: Path) -> Path:
    return output_root / session_dir.parent.parent.name / session_dir.parent.name / session_dir.name


def session_is_complete(session_output_dir: Path) -> bool:
    return (session_output_dir / "reconstruction_manifest.csv").is_file()


def discover_session_dirs(root_dir: Path, excluded_date_dirs: set[str] | None = None) -> list[Path]:
    session_dirs: list[Path] = []
    excluded = excluded_date_dirs or set()
    if not root_dir.exists():
        return session_dirs
    for date_dir in sorted(root_dir.iterdir()):
        if not date_dir.is_dir() or not DATE_DIR_RE.fullmatch(date_dir.name):
            continue
        if date_dir.name in excluded:
            continue
        for subject_dir in sorted(date_dir.iterdir()):
            if not subject_dir.is_dir() or not SUBJECT_DIR_RE.fullmatch(subject_dir.name):
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                if find_pressure_file(session_dir, "left") and find_pressure_file(session_dir, "right"):
                    session_dirs.append(session_dir)
    return session_dirs


def find_pressure_file(session_dir: Path, side: str) -> Path | None:
    names = EXPECTED_LEFT_NAMES if side == "left" else EXPECTED_RIGHT_NAMES
    for name in names:
        candidate = session_dir / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(session_dir.glob(f"*{side}*.csv")):
        if candidate.is_file() and get_side_from_name(candidate.name) == side:
            return candidate
    return None


def parse_pressure_csv(path: Path) -> list[PressureRow]:
    rows: list[PressureRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty pressure CSV: {path}") from exc

        frame_idx_col = find_column(header, {"frame_idx", "frame_index"})
        time_col = find_column(header, {"t_us", "captured_at"})
        heartbeat_col = find_column(header, {"heartbeat_id", "heartbeat"})
        payload_cols = [idx for idx in range(len(header)) if idx not in {frame_idx_col, time_col, heartbeat_col}]

        for row_idx, row in enumerate(reader):
            if not row or all(not cell.strip() for cell in row):
                continue
            if frame_idx_col is not None and frame_idx_col < len(row) and row[frame_idx_col].strip():
                try:
                    source_frame_idx = int(float(row[frame_idx_col]))
                except ValueError:
                    source_frame_idx = row_idx
            else:
                source_frame_idx = row_idx

            if time_col is not None and time_col < len(row) and row[time_col].strip():
                try:
                    source_t_us = float(row[time_col])
                except ValueError as exc:
                    raise ValueError(f"Bad t_us value in {path}: {row[time_col]!r}") from exc
            elif frame_idx_col is not None and frame_idx_col < len(row) and row[frame_idx_col].strip():
                try:
                    source_t_us = float(row[frame_idx_col]) * DEFAULT_FRAME_PERIOD_US
                except ValueError as exc:
                    raise ValueError(f"Bad frame_idx value in {path}: {row[frame_idx_col]!r}") from exc
            elif heartbeat_col is not None and heartbeat_col < len(row) and row[heartbeat_col].strip():
                try:
                    source_t_us = float(row[heartbeat_col]) * HEARTBEAT_PERIOD_US
                except ValueError as exc:
                    raise ValueError(f"Bad heartbeat_id value in {path}: {row[heartbeat_col]!r}") from exc
            else:
                source_t_us = float(row_idx) * DEFAULT_FRAME_PERIOD_US

            values: list[float] = []
            for idx in payload_cols:
                if idx >= len(row) or not row[idx].strip():
                    raise ValueError(f"Missing pressure value in {path} row {row_idx + 2}")
                try:
                    values.append(float(row[idx]))
                except ValueError as exc:
                    raise ValueError(f"Bad pressure value in {path}: {row[idx]!r}") from exc

            if len(values) != EXPECTED_CHANNELS:
                raise ValueError(
                    f"Expected {EXPECTED_CHANNELS} channels in {path}, got {len(values)}"
                )
            rows.append(PressureRow(source_frame_idx=source_frame_idx, source_t_us=source_t_us, values=values))
    return rows


def drop_exact_duplicate_rows(rows: list[PressureRow]) -> list[PressureRow]:
    if not rows:
        return []
    cleaned = [rows[0]]
    for row in rows[1:]:
        prev = cleaned[-1]
        if row.source_t_us == prev.source_t_us and row.values == prev.values:
            continue
        cleaned.append(row)
    return cleaned


def regression_xs(rows: list[PressureRow]) -> list[float]:
    if not rows:
        return []
    base_frame_idx = rows[0].source_frame_idx
    xs = [float(row.source_frame_idx - base_frame_idx) for row in rows]
    if any(xs[idx] <= xs[idx - 1] for idx in range(1, len(xs))):
        return [float(idx) for idx in range(len(rows))]
    return xs


def weighted_linear_regression(xs: list[float], ys: list[float], weights: list[float]) -> tuple[float, float]:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return 0.0, ys[0] if ys else 0.0
    x_bar = sum(w * x for w, x in zip(weights, xs)) / weight_sum
    y_bar = sum(w * y for w, y in zip(weights, ys)) / weight_sum
    denom = sum(w * (x - x_bar) ** 2 for w, x in zip(weights, xs))
    if denom <= 0:
        return 0.0, y_bar
    numer = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(weights, xs, ys))
    slope = numer / denom
    intercept = y_bar - slope * x_bar
    return slope, intercept


def ordinary_linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    weights = [1.0] * len(xs)
    return weighted_linear_regression(xs, ys, weights)


def robust_linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return 0.0, ys[0]

    slope, intercept = ordinary_linear_regression(xs, ys)
    for _ in range(12):
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
        scale = median([abs(r) for r in residuals])
        if scale <= 1e-9:
            break
        scale *= 1.4826
        cutoff = 1.345 * scale
        weights = [1.0 if abs(r) <= cutoff else cutoff / abs(r) for r in residuals]
        new_slope, new_intercept = weighted_linear_regression(xs, ys, weights)
        if not math.isfinite(new_slope) or not math.isfinite(new_intercept):
            break
        if abs(new_slope - slope) < 1e-6 and abs(new_intercept - intercept) < 1e-3:
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept = new_slope, new_intercept

    if not math.isfinite(slope) or slope <= 0:
        positive_diffs = [ys[idx + 1] - ys[idx] for idx in range(len(ys) - 1) if ys[idx + 1] > ys[idx]]
        fallback = median(positive_diffs) if positive_diffs else 25_000.0
        slope = float(fallback) if fallback and fallback > 0 else 25_000.0
        intercept = ys[0] - slope * xs[0]
    return slope, intercept


def estimate_period_us(rows: list[PressureRow]) -> float:
    if len(rows) < 2:
        return DEFAULT_FRAME_PERIOD_US
    xs = regression_xs(rows)
    ys = [row.source_t_us for row in rows]
    slope, _ = robust_linear_regression(xs, ys)
    return slope if slope > 0 else DEFAULT_FRAME_PERIOD_US


def fit_segment_axis(rows: list[PressureRow], dt_fallback_us: float) -> tuple[float, float]:
    if not rows:
        dt_hat_us = dt_fallback_us if dt_fallback_us > 0 else DEFAULT_FRAME_PERIOD_US
        return dt_hat_us, 0.0
    xs = regression_xs(rows)
    ys = [row.source_t_us for row in rows]
    dt_hat_us, t0_us = robust_linear_regression(xs, ys)
    if not math.isfinite(dt_hat_us) or dt_hat_us <= 0:
        dt_hat_us = dt_fallback_us if dt_fallback_us > 0 else DEFAULT_FRAME_PERIOD_US
        t0_us = ys[0] - dt_hat_us * xs[0]
    return dt_hat_us, t0_us


def assign_uniform_slots(rows: list[PressureRow], dt_hat_us: float, t0_us: float) -> list[int]:
    if not rows:
        return []
    if dt_hat_us <= 0:
        return list(range(len(rows)))

    slots: list[int] = []
    for idx, row in enumerate(rows):
        if idx == 0:
            slots.append(0)
            continue
        prev = rows[idx - 1]
        if abs(row.source_t_us - prev.source_t_us) <= 1e-6:
            slots.append(slots[-1] + 1)
            continue
        target_slot = int(round((row.source_t_us - t0_us) / dt_hat_us))
        slots.append(max(slots[-1] + 1, target_slot))
    return slots


def rebuild_rows_on_frame_axis(rows: list[PressureRow]) -> list[PressureRow]:
    if not rows:
        return []
    base_frame_idx = rows[0].source_frame_idx
    return [
        PressureRow(
            source_frame_idx=row.source_frame_idx,
            source_t_us=float(row.source_frame_idx - base_frame_idx) * DEFAULT_FRAME_PERIOD_US,
            values=row.values,
        )
        for row in rows
    ]


def split_rows_by_boundaries(rows: list[PressureRow], boundaries: list[float]) -> list[list[PressureRow]]:
    segments: list[list[PressureRow]] = [[] for _ in range(len(boundaries) + 1)]
    for row in rows:
        idx = bisect_right(boundaries, row.source_t_us)
        segments[idx].append(row)
    return segments


def add_boundary(boundaries: list[float], value: float, tolerance: float = 1e-6) -> bool:
    for existing in boundaries:
        if abs(existing - value) <= tolerance:
            return False
    boundaries.append(value)
    boundaries.sort()
    return True


def infer_session_boundaries(sides_rows: dict[str, list[PressureRow]]) -> list[float]:
    boundaries: list[float] = []
    for _ in range(8):
        changed = False
        for rows in sides_rows.values():
            for segment in split_rows_by_boundaries(rows, boundaries):
                if len(segment) < 2:
                    continue
                dt_hat, t0_us = fit_segment_axis(segment, DEFAULT_FRAME_PERIOD_US)
                slots = assign_uniform_slots(segment, dt_hat, t0_us)
                for idx in range(len(segment) - 1):
                    missing = slots[idx + 1] - slots[idx] - 1
                    if missing > 2:
                        boundary = 0.5 * (segment[idx].source_t_us + segment[idx + 1].source_t_us)
                        changed = add_boundary(boundaries, boundary) or changed
        if not changed:
            break
    return boundaries


def interpolate_values(left: list[float], right: list[float], alpha: float) -> list[float]:
    return [(1.0 - alpha) * l + alpha * r for l, r in zip(left, right)]


def is_contact_frame(values: list[float]) -> bool:
    return any(value > 0.0 for value in values)


def interpolate_or_snap_values(
    left: list[float],
    right: list[float],
    alpha: float,
    snap_on_contact_transition: bool,
) -> list[float]:
    if snap_on_contact_transition and is_contact_frame(left) != is_contact_frame(right):
        return list(left) if alpha <= 0.5 else list(right)
    return interpolate_values(left, right, alpha)


def offset_reconstructed_rows(
    rows: list[dict[str, object]],
    frame_offset: int,
    time_offset_us: float,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in rows:
        output_row = dict(row)
        output_row["frame_idx"] = int(row["frame_idx"]) + frame_offset
        output_row["t_us"] = float(row["t_us"]) + time_offset_us
        output_rows.append(output_row)
    return output_rows


def trim_negative_leading_rows(
    rows: list[dict[str, object]],
    manifest_rows: list[ManifestRow],
    tolerance_us: float = 0.0,
) -> None:
    first_keep_idx = 0
    while first_keep_idx < len(rows) and float(rows[first_keep_idx]["t_us"]) < tolerance_us:
        first_keep_idx += 1
    if first_keep_idx == 0:
        return

    first_kept_t_us = float(rows[first_keep_idx]["t_us"])
    del rows[:first_keep_idx]
    for frame_idx, row in enumerate(rows):
        row["frame_idx"] = frame_idx

    kept_manifest_rows: list[ManifestRow] = []
    for manifest_row in manifest_rows:
        if manifest_row.frame_idx_end < first_keep_idx:
            continue
        if manifest_row.frame_idx_start < first_keep_idx:
            manifest_row.frame_idx_start = 0
            manifest_row.t_us_start = max(manifest_row.t_us_start, first_kept_t_us)
        else:
            manifest_row.frame_idx_start -= first_keep_idx
        manifest_row.frame_idx_end -= first_keep_idx
        kept_manifest_rows.append(manifest_row)
    manifest_rows[:] = kept_manifest_rows


def make_segment_manifest_row(
    side: str,
    name: str,
    source_rows: list[PressureRow],
    output_rows: list[dict[str, object]],
    inserted_rows: int,
    dt_hat_us: float,
) -> ManifestRow:
    if not source_rows or not output_rows:
        raise ValueError(f"Cannot build manifest row for empty segment {side}:{name}")
    return ManifestRow(
        block_type="segment",
        side=side,
        name=name,
        frame_idx_start=int(output_rows[0]["frame_idx"]),
        frame_idx_end=int(output_rows[-1]["frame_idx"]),
        t_us_start=float(output_rows[0]["t_us"]),
        t_us_end=float(output_rows[-1]["t_us"]),
        source_rows=len(source_rows),
        inserted_rows=inserted_rows,
        dt_hat_us=dt_hat_us,
        source_t_us_start=source_rows[0].source_t_us,
        source_t_us_end=source_rows[-1].source_t_us,
    )


def make_bridge_rows(
    prev_source_row: PressureRow,
    next_source_row: PressureRow,
    missing_count: int,
    start_frame_idx: int,
    start_t_us: float,
    dt_hat_us: float,
    snap_on_contact_transition: bool,
) -> list[dict[str, object]]:
    bridge_rows: list[dict[str, object]] = []
    for gap_idx in range(1, missing_count + 1):
        alpha = gap_idx / float(missing_count + 1)
        values = interpolate_or_snap_values(
            prev_source_row.values,
            next_source_row.values,
            alpha,
            snap_on_contact_transition=snap_on_contact_transition,
        )
        bridge_rows.append(
            make_output_row(
                frame_idx=start_frame_idx + gap_idx - 1,
                t_us=start_t_us + (gap_idx - 1) * dt_hat_us,
                valid_mask=0,
                source_row=None,
                values=values,
            )
        )
    return bridge_rows


def make_bridge_manifest_row(
    side: str,
    prev_name: str,
    next_name: str,
    prev_source_row: PressureRow,
    next_source_row: PressureRow,
    start_frame_idx: int,
    start_t_us: float,
    inserted_rows: int,
    dt_hat_us: float,
    snap_on_contact_transition: bool,
) -> ManifestRow:
    mode = "snap" if snap_on_contact_transition and is_contact_frame(prev_source_row.values) != is_contact_frame(next_source_row.values) else "interpolate"
    return ManifestRow(
        block_type="bridge",
        side=side,
        name=f"{prev_name}->{next_name}",
        prev_name=prev_name,
        next_name=next_name,
        frame_idx_start=start_frame_idx,
        frame_idx_end=start_frame_idx + inserted_rows - 1,
        t_us_start=start_t_us,
        t_us_end=start_t_us + (inserted_rows - 1) * dt_hat_us,
        source_rows=2,
        inserted_rows=inserted_rows,
        dt_hat_us=dt_hat_us,
        source_t_us_start=prev_source_row.source_t_us,
        source_t_us_end=next_source_row.source_t_us,
        contact_transition_mode=mode,
    )


def make_output_row(
    frame_idx: int,
    t_us: float,
    valid_mask: int,
    source_row: PressureRow | None,
    values: list[float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_idx": frame_idx,
        "t_us": t_us,
        "valid_mask": valid_mask,
        "source_frame_idx": "" if source_row is None else source_row.source_frame_idx,
        "source_t_us": "" if source_row is None else source_row.source_t_us,
    }
    for idx, value in enumerate(values, start=1):
        row[str(idx)] = value
    return row


def reconstruct_segment(
    rows: list[PressureRow],
    dt_fallback_us: float,
    snap_on_contact_transition: bool,
) -> tuple[list[dict[str, object]], float, int]:
    if not rows:
        return [], dt_fallback_us, 0

    dt_hat_us, t0_us = fit_segment_axis(rows, dt_fallback_us)
    slots = assign_uniform_slots(rows, dt_hat_us, t0_us)
    output_rows: list[dict[str, object]] = []
    inserted_count = 0

    for idx, row in enumerate(rows):
        if idx > 0:
            prev = rows[idx - 1]
            if row.source_t_us < prev.source_t_us - 1e-6:
                raise ValueError("Pressure CSV rows are not monotonic in time.")
            missing = slots[idx] - slots[idx - 1] - 1
            if missing > 2:
                raise SegmentGapError(
                    boundary_t_us=0.5 * (prev.source_t_us + row.source_t_us),
                    message=(
                        f"Unexpected large gap after segmentation: "
                        f"slots={slots[idx - 1]}->{slots[idx]}, dt_hat={dt_hat_us:.3f}us"
                    ),
                )
            for gap_idx in range(1, missing + 1):
                alpha = gap_idx / float(missing + 1)
                values = interpolate_or_snap_values(
                    prev.values,
                    row.values,
                    alpha,
                    snap_on_contact_transition=snap_on_contact_transition,
                )
                frame_idx = slots[idx - 1] + gap_idx
                output_rows.append(
                    make_output_row(
                        frame_idx=frame_idx,
                        t_us=t0_us + frame_idx * dt_hat_us,
                        valid_mask=1,
                        source_row=None,
                        values=values,
                    )
                )
                inserted_count += 1

        frame_idx = slots[idx]
        output_rows.append(
            make_output_row(
                frame_idx=frame_idx,
                t_us=t0_us + frame_idx * dt_hat_us,
                valid_mask=1,
                source_row=row,
                values=row.values,
            )
        )

    return output_rows, dt_hat_us, inserted_count


def extract_output_values(row: dict[str, object]) -> list[float]:
    return [float(row[str(idx)]) for idx in range(1, EXPECTED_CHANNELS + 1)]


def make_output_row_from_existing(
    frame_idx: int,
    t_us: float,
    valid_mask: int,
    source_row: dict[str, object] | None,
    values: list[float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_idx": frame_idx,
        "t_us": t_us,
        "valid_mask": valid_mask,
        "source_frame_idx": "" if source_row is None else source_row["source_frame_idx"],
        "source_t_us": "" if source_row is None else source_row["source_t_us"],
    }
    for idx, value in enumerate(values, start=1):
        row[str(idx)] = value
    return row


def resample_output_rows_to_fixed_rate(
    rows: list[dict[str, object]],
    target_period_us: float,
    snap_on_contact_transition: bool,
) -> tuple[list[dict[str, object]], int]:
    if not rows:
        return [], 0
    if len(rows) == 1:
        row = rows[0]
        valid_mask = int(row["valid_mask"])
        return [
            make_output_row_from_existing(
                frame_idx=0,
                t_us=float(row["t_us"]),
                valid_mask=valid_mask,
                source_row=row if valid_mask == 1 else None,
                values=extract_output_values(row),
            )
        ], 0 if valid_mask == 1 else 1

    source_times = [float(row["t_us"]) for row in rows]
    start_t_us = source_times[0]
    end_t_us = source_times[-1]
    frame_count = max(1, int(math.floor((end_t_us - start_t_us) / target_period_us + 1e-6)) + 1)
    tolerance_us = 1.0
    resampled_rows: list[dict[str, object]] = []
    interpolated_count = 0

    for frame_idx in range(frame_count):
        target_t_us = start_t_us + frame_idx * target_period_us
        insert_idx = bisect_right(source_times, target_t_us)

        exact_row: dict[str, object] | None = None
        if insert_idx > 0 and abs(source_times[insert_idx - 1] - target_t_us) <= tolerance_us:
            exact_row = rows[insert_idx - 1]
        elif insert_idx < len(rows) and abs(source_times[insert_idx] - target_t_us) <= tolerance_us:
            exact_row = rows[insert_idx]

        if exact_row is not None:
            valid_mask = int(exact_row["valid_mask"])
            resampled_rows.append(
                make_output_row_from_existing(
                    frame_idx=frame_idx,
                    t_us=target_t_us,
                    valid_mask=valid_mask,
                    source_row=exact_row if valid_mask == 1 else None,
                    values=extract_output_values(exact_row),
                )
            )
            if valid_mask == 0:
                interpolated_count += 1
            continue

        if insert_idx <= 0:
            left_row = rows[0]
            right_row = rows[1]
        elif insert_idx >= len(rows):
            left_row = rows[-2]
            right_row = rows[-1]
        else:
            left_row = rows[insert_idx - 1]
            right_row = rows[insert_idx]

        left_t_us = float(left_row["t_us"])
        right_t_us = float(right_row["t_us"])
        if right_t_us <= left_t_us + 1e-6:
            values = extract_output_values(left_row)
        else:
            alpha = (target_t_us - left_t_us) / (right_t_us - left_t_us)
            values = interpolate_or_snap_values(
                extract_output_values(left_row),
                extract_output_values(right_row),
                alpha,
                snap_on_contact_transition=snap_on_contact_transition,
            )
        resampled_rows.append(
            make_output_row_from_existing(
                frame_idx=frame_idx,
                t_us=target_t_us,
                valid_mask=int(left_row["valid_mask"]) and int(right_row["valid_mask"]),
                source_row=None,
                values=values,
            )
        )
        interpolated_count += 1

    return resampled_rows, interpolated_count


def write_reconstructed_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["frame_idx", "t_us", "valid_mask", "source_frame_idx", "source_t_us"] + [
        str(idx) for idx in range(1, EXPECTED_CHANNELS + 1)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    fieldnames = [
        "block_type",
        "side",
        "name",
        "prev_name",
        "next_name",
        "frame_idx_start",
        "frame_idx_end",
        "t_us_start",
        "t_us_end",
        "source_rows",
        "inserted_rows",
        "dt_hat_us",
        "source_t_us_start",
        "source_t_us_end",
        "contact_transition_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "block_type": row.block_type,
                    "side": row.side,
                    "name": row.name,
                    "prev_name": row.prev_name,
                    "next_name": row.next_name,
                    "frame_idx_start": row.frame_idx_start,
                    "frame_idx_end": row.frame_idx_end,
                    "t_us_start": row.t_us_start,
                    "t_us_end": row.t_us_end,
                    "source_rows": row.source_rows,
                    "inserted_rows": row.inserted_rows,
                    "dt_hat_us": row.dt_hat_us,
                    "source_t_us_start": row.source_t_us_start,
                    "source_t_us_end": row.source_t_us_end,
                    "contact_transition_mode": row.contact_transition_mode,
                }
            )


def reconstruct_side_output(
    side: str,
    rows: list[PressureRow],
    boundaries: list[float],
    dt_fallback_us: float,
    snap_on_contact_transition: bool,
) -> tuple[list[dict[str, object]], list[ManifestRow]]:
    segment_rows = split_rows_by_boundaries(rows, boundaries)
    output_rows: list[dict[str, object]] = []
    manifest_rows: list[ManifestRow] = []
    prev_nonempty_name = ""
    prev_nonempty_rows: list[PressureRow] | None = None
    prev_nonempty_dt_hat_us = dt_fallback_us

    for segment_idx, source_segment in enumerate(segment_rows):
        if not source_segment:
            continue

        segment_name = f"t{segment_idx}"
        segment_output, segment_dt_hat_us, segment_inserted = reconstruct_segment(
            source_segment,
            dt_fallback_us,
            snap_on_contact_transition,
        )

        if not output_rows:
            frame_offset = 0
            time_offset_us = 0.0
        else:
            prev_last_t_us = float(output_rows[-1]["t_us"])
            bridge_dt_us = median([prev_nonempty_dt_hat_us, segment_dt_hat_us, dt_fallback_us])
            if not math.isfinite(bridge_dt_us) or bridge_dt_us <= 0:
                bridge_dt_us = dt_fallback_us if dt_fallback_us > 0 else DEFAULT_FRAME_PERIOD_US
            prev_last_source = prev_nonempty_rows[-1] if prev_nonempty_rows else None
            next_first_source = source_segment[0]
            if prev_last_source is not None:
                missing_count = max(0, int(round((next_first_source.source_t_us - prev_last_source.source_t_us) / bridge_dt_us)) - 1)
            else:
                missing_count = 0

            if missing_count > 0 and prev_last_source is not None:
                bridge_start_frame = len(output_rows)
                bridge_start_t_us = prev_last_t_us + bridge_dt_us
                bridge_rows = make_bridge_rows(
                    prev_last_source,
                    next_first_source,
                    missing_count,
                    bridge_start_frame,
                    bridge_start_t_us,
                    bridge_dt_us,
                    snap_on_contact_transition,
                )
                output_rows.extend(bridge_rows)
                manifest_rows.append(
                    make_bridge_manifest_row(
                        side=side,
                        prev_name=prev_nonempty_name,
                        next_name=segment_name,
                        prev_source_row=prev_last_source,
                        next_source_row=next_first_source,
                        start_frame_idx=bridge_start_frame,
                        start_t_us=bridge_start_t_us,
                        inserted_rows=missing_count,
                        dt_hat_us=bridge_dt_us,
                        snap_on_contact_transition=snap_on_contact_transition,
                    )
                )
                prev_last_t_us = float(output_rows[-1]["t_us"])

            frame_offset = len(output_rows)
            time_offset_us = prev_last_t_us + bridge_dt_us - float(segment_output[0]["t_us"])

        transformed_segment = offset_reconstructed_rows(segment_output, frame_offset, time_offset_us)
        output_rows.extend(transformed_segment)
        manifest_rows.append(
            make_segment_manifest_row(
                side=side,
                name=segment_name,
                source_rows=source_segment,
                output_rows=transformed_segment,
                inserted_rows=segment_inserted,
                dt_hat_us=segment_dt_hat_us,
            )
        )

        prev_nonempty_name = segment_name
        prev_nonempty_rows = source_segment
        prev_nonempty_dt_hat_us = segment_dt_hat_us

    return output_rows, manifest_rows


def process_session(
    session_dir: Path,
    output_root: Path,
    resample_to_40hz: bool,
    snap_on_contact_transition: bool,
) -> list[ManifestRow]:
    left_path = find_pressure_file(session_dir, "left")
    right_path = find_pressure_file(session_dir, "right")
    if left_path is None or right_path is None:
        return []

    left_rows = drop_exact_duplicate_rows(parse_pressure_csv(left_path))
    right_rows = drop_exact_duplicate_rows(parse_pressure_csv(right_path))
    if not left_rows and not right_rows:
        return []

    sides_rows = {"left": left_rows, "right": right_rows}
    boundaries = infer_session_boundaries(sides_rows)
    fallback_periods = [estimate_period_us(rows) for rows in sides_rows.values() if len(rows) >= 2]
    dt_fallback_us = median(fallback_periods) if fallback_periods else DEFAULT_FRAME_PERIOD_US

    left_output: list[dict[str, object]] = []
    right_output: list[dict[str, object]] = []
    left_manifest_rows: list[ManifestRow] = []
    right_manifest_rows: list[ManifestRow] = []

    for _ in range(32):
        try:
            left_output, left_manifest_rows = reconstruct_side_output(
                "left",
                left_rows,
                boundaries,
                dt_fallback_us,
                snap_on_contact_transition,
            )
            right_output, right_manifest_rows = reconstruct_side_output(
                "right",
                right_rows,
                boundaries,
                dt_fallback_us,
                snap_on_contact_transition,
            )
            break
        except SegmentGapError as exc:
            if add_boundary(boundaries, exc.boundary_t_us):
                continue
            raise

    session_output_dir = session_output_dir_for(output_root, session_dir)
    if session_output_dir.exists():
        shutil.rmtree(session_output_dir)
    ensure_dir(session_output_dir)

    if resample_to_40hz:
        left_input_len = len(left_output)
        right_input_len = len(right_output)
        left_output, left_inserted = resample_output_rows_to_fixed_rate(
            left_output,
            RESAMPLED_FRAME_PERIOD_US,
            snap_on_contact_transition=snap_on_contact_transition,
        )
        right_output, right_inserted = resample_output_rows_to_fixed_rate(
            right_output,
            RESAMPLED_FRAME_PERIOD_US,
            snap_on_contact_transition=snap_on_contact_transition,
        )
        left_manifest_rows.append(
            ManifestRow(
                block_type="resample",
                side="left",
                name="40hz",
                frame_idx_start=0,
                frame_idx_end=max(0, len(left_output) - 1),
                t_us_start=float(left_output[0]["t_us"]) if left_output else 0.0,
                t_us_end=float(left_output[-1]["t_us"]) if left_output else 0.0,
                source_rows=left_input_len,
                inserted_rows=left_inserted,
                dt_hat_us=RESAMPLED_FRAME_PERIOD_US,
                source_t_us_start=float(left_output[0]["t_us"]) if left_output else 0.0,
                source_t_us_end=float(left_output[-1]["t_us"]) if left_output else 0.0,
            )
        )
        right_manifest_rows.append(
            ManifestRow(
                block_type="resample",
                side="right",
                name="40hz",
                frame_idx_start=0,
                frame_idx_end=max(0, len(right_output) - 1),
                t_us_start=float(right_output[0]["t_us"]) if right_output else 0.0,
                t_us_end=float(right_output[-1]["t_us"]) if right_output else 0.0,
                source_rows=right_input_len,
                inserted_rows=right_inserted,
                dt_hat_us=RESAMPLED_FRAME_PERIOD_US,
                source_t_us_start=float(right_output[0]["t_us"]) if right_output else 0.0,
                source_t_us_end=float(right_output[-1]["t_us"]) if right_output else 0.0,
            )
        )

    trim_negative_leading_rows(left_output, left_manifest_rows)
    trim_negative_leading_rows(right_output, right_manifest_rows)

    write_reconstructed_csv(session_output_dir / "pressure_left.csv", left_output)
    write_reconstructed_csv(session_output_dir / "pressure_right.csv", right_output)
    write_manifest(session_output_dir / "reconstruction_manifest.csv", left_manifest_rows + right_manifest_rows)
    return left_manifest_rows + right_manifest_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild tactile pressure sessions into uniform-time segments.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR, help="Raw tactile dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for this reconstruction run. Default: outputs/reconstructed/reconstruction_<timestamp>/",
    )
    parser.add_argument("--date", type=str, default=None, help="Process only one date directory.")
    parser.add_argument("--subject", type=str, default=None, help="Process only one subject directory.")
    parser.add_argument("--session-name", type=str, default=None, help="Process only one session directory.")
    parser.add_argument(
        "--exclude-date",
        action="append",
        default=None,
        help=(
            "Exclude one date directory. Can be passed multiple times. "
            "Default excludes: 20260419, 20260422."
        ),
    )
    parser.add_argument(
        "--include-excluded-dates",
        action="store_true",
        help="Disable the default date exclusions and reconstruct all dates unless --exclude-date is provided.",
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Process at most N sessions.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing output directory and skip completed sessions.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed session.")
    parser.add_argument(
        "-40hz",
        "--resample-40hz",
        action="store_true",
        dest="resample_40hz",
        help="Resample each reconstructed segment to a fixed 40Hz timeline by interpolation.",
    )
    parser.add_argument(
        "--snap-contact-transition-interpolation",
        action="store_true",
        help=(
            "When an interpolated point lies between a contact frame and a non-contact frame, "
            "copy the nearest frame instead of linearly interpolating."
        ),
    )
    return parser


def session_matches_filters(session_dir: Path, date: str | None, subject: str | None, session_name: str | None) -> bool:
    if date is not None and session_dir.parent.parent.name != date:
        return False
    if subject is not None and session_dir.parent.name != subject.upper():
        return False
    if session_name is not None and session_dir.name != session_name:
        return False
    return True


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive.")
    excluded_date_dirs = set(args.exclude_date or [])
    if not args.include_excluded_dates:
        excluded_date_dirs.update(EXCLUDED_DATE_DIRS)
    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_root = OUTPUT_PARENT_DIR / f"reconstruction_{stamp}"
    output_root = prepare_output_dir(output_root, args.overwrite, args.resume)

    session_dirs = [
        session_dir
        for session_dir in discover_session_dirs(args.root_dir, excluded_date_dirs=excluded_date_dirs)
        if session_matches_filters(session_dir, args.date, args.subject, args.session_name)
    ]
    if args.max_sessions is not None:
        session_dirs = session_dirs[: args.max_sessions]

    if not session_dirs:
        raise SystemExit("No valid pressure sessions were found.")

    skipped = 0
    if args.resume:
        original_count = len(session_dirs)
        session_dirs = [
            session_dir
            for session_dir in session_dirs
            if not session_is_complete(session_output_dir_for(output_root, session_dir))
        ]
        skipped = original_count - len(session_dirs)
        if not session_dirs:
            print(f"Done. Nothing to do; skipped {skipped} completed sessions in {output_root}")
            return 0

    processed = 0
    failed: list[dict[str, str]] = []
    for session_dir in session_dirs:
        try:
            process_session(
                session_dir,
                output_root,
                args.resample_40hz,
                args.snap_contact_transition_interpolation,
            )
        except Exception as exc:
            failed.append(
                {
                    "session_dir": str(session_dir),
                    "error": str(exc),
                }
            )
            print(f"Failed {session_dir}: {exc}")
            if args.strict:
                raise
            continue
        processed += 1
        print(f"Rebuilt {session_dir}")

    print(f"Done. Rebuilt {processed} sessions into {output_root} (skipped {skipped})")
    failures_path = output_root / "reconstruction_failures.csv"
    if failed:
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["session_dir", "error"])
            writer.writeheader()
            writer.writerows(failed)
        print(f"Failed {len(failed)} sessions; see {failures_path}")
    elif failures_path.exists():
        failures_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
