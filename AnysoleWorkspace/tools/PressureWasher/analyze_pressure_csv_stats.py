"""Analyze raw tactile pressure CSV files as per-session, per-second frame counts.

Outputs:
    outputs/stats/pressure_stats_<timestamp>/
    ├── overall/
    │   ├── frame_per_second.csv
    │   └── frame_per_second.png
    └── <date>/<Si>/<session_name>/
        ├── frame_per_second.csv
        └── frame_per_second.png

Each CSV merges left/right statistics in one table and includes:
- actual frame count per second
- expected frame count for start-edge seconds estimated from the average frame interval
- end-edge seconds are only marked, not estimated

Each session also adds:
- t_us_diff_histogram.png
- t_us_diff_histogram.csv
- gap_list.csv
- gap_positions.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/data/fangyuxuan/.tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwlib.paths import DEFAULT_RAW_ROOT_DIR, STATS_OUTPUTS_DIR


NEW_DATA_ROOT = DEFAULT_RAW_ROOT_DIR
DEFAULT_ROOT_DIRS = [
    NEW_DATA_ROOT,
    Path(r"F:\3_Experiment_Data\FourVideosData\Experiment\S1"),
    Path(r"F:\3_Experiment_Data\FourVideosData\Experiment\S2"),
    Path(r"F:\3_Experiment_Data\FourVideosData\Experiment\S3"),
    Path(r"F:\3_Experiment_Data\FourVideosData\Experiment\S4"),
]
DEFAULT_FILE_PATTERNS = (
    "Pressure_left_*.csv",
    "Pressure_right_*.csv",
    "pressure_left*.csv",
    "pressure_right*.csv",
)
EXCLUDED_DATE_DIRS = {"20260419", "20260422"}
DEFAULT_VIDEO_FRAME_RATE = 40
EXPECTED_ACTIONS_PER_SUBJECT = 11
EXPECTED_GROUPS_PER_ACTION = 3
OUTPUT_PARENT_DIR = STATS_OUTPUTS_DIR

DATE_DIR_RE = re.compile(r"^\d{8}$")
SUBJECT_DIR_RE = re.compile(r"^S\d+$", re.IGNORECASE)


@dataclass
class FileSeries:
    date: str
    subject: str
    session_name: str
    side: str
    counts_by_second: dict[int, int]
    expected_counts_by_second: dict[int, float]
    min_observed_second: int | None
    max_observed_second: int | None
    frame_idx_values: list[int]
    time_us_values: list[float]
    frame_idx_values: list[int]
    time_us_values: list[float]


def discover_pressure_files(roots: Iterable[Path], file_patterns: Iterable[str]) -> list[Path]:
    file_map: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for pattern in file_patterns:
            for path in root.rglob(pattern):
                if path.is_file() and not any(part in EXCLUDED_DATE_DIRS for part in path.parts):
                    file_map[str(path.resolve())] = path.resolve()
    return sorted(file_map.values(), key=lambda path: str(path))


def discover_subject_dirs(roots: Iterable[Path]) -> dict[tuple[str, str], Path]:
    subject_dirs: dict[tuple[str, str], Path] = {}
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for date_dir in root.iterdir():
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            if not DATE_DIR_RE.fullmatch(date) or date in EXCLUDED_DATE_DIRS:
                continue
            for subject_dir in date_dir.iterdir():
                if not subject_dir.is_dir() or not SUBJECT_DIR_RE.fullmatch(subject_dir.name):
                    continue
                subject_dirs[(date, subject_dir.name.upper())] = subject_dir.resolve()
    return dict(sorted(subject_dirs.items()))


def get_date_subject_and_session(path: Path) -> tuple[str, str, str]:
    date = ""
    subject = ""
    session_name = ""
    parts = path.parts
    for idx, part in enumerate(parts):
        if DATE_DIR_RE.fullmatch(part):
            date = part
            for next_part in parts[idx + 1 :]:
                if SUBJECT_DIR_RE.fullmatch(next_part):
                    subject = next_part.upper()
                    session_name = path.parent.name
                    return date, subject, session_name
    return date, subject, session_name


def get_subject_number(path: Path) -> int | None:
    for part in path.parts:
        match = SUBJECT_DIR_RE.fullmatch(part)
        if match:
            try:
                return int(part[1:])
            except ValueError:
                return None
    return None


def get_subject_number_from_name(subject: str) -> int | None:
    if SUBJECT_DIR_RE.fullmatch(subject):
        try:
            return int(subject[1:])
        except ValueError:
            return None
    return None


def subject_in_range(subject: str, start_s: int | None, end_s: int | None) -> bool:
    subject_number = get_subject_number_from_name(subject)
    if subject_number is None:
        return False
    if start_s is not None and subject_number < start_s:
        return False
    if end_s is not None and subject_number > end_s:
        return False
    return True


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


def extract_time_us(
    row: list[str],
    time_col: int | None,
    frame_idx_col: int | None,
    frame_rate: int,
) -> float | None:
    if time_col is not None and time_col < len(row):
        raw = row[time_col].strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = float("nan")
            if math.isfinite(value):
                return value

    if frame_idx_col is not None and frame_idx_col < len(row):
        raw = row[frame_idx_col].strip()
        if raw:
            try:
                frame_idx = float(raw)
            except ValueError:
                frame_idx = float("nan")
            if math.isfinite(frame_idx):
                return frame_idx * 1_000_000.0 / max(1, frame_rate)

    return None


def extract_frame_idx(row: list[str], frame_idx_col: int | None) -> int | None:
    if frame_idx_col is not None and frame_idx_col < len(row):
        raw = row[frame_idx_col].strip()
        if raw:
            try:
                value = int(float(raw))
            except ValueError:
                value = None
            return value
    return None


def parse_second_from_time_us(time_us: float | None, row_idx: int, frame_rate: int) -> int:
    if time_us is not None and math.isfinite(time_us):
        return max(0, int(time_us // 1_000_000.0))
    return max(0, row_idx // max(1, frame_rate))


def estimate_expected_counts_by_second(
    times_us: list[float],
    counts_by_second: dict[int, int],
) -> dict[int, float]:
    if not counts_by_second:
        return {}

    max_second = max(counts_by_second)
    if len(times_us) < 2:
        return {second: float(counts_by_second.get(second, 0)) for second in range(max_second + 1)}

    min_t = min(times_us)
    max_t = max(times_us)
    span_us = max(1.0, max_t - min_t)
    avg_interval_us = span_us / float(len(times_us) - 1)

    expected: dict[int, float] = {}
    for second in range(max_second + 1):
        start_us = second * 1_000_000.0
        end_us = (second + 1) * 1_000_000.0
        overlap_us = max(0.0, min(max_t, end_us) - max(min_t, start_us))
        expected[second] = overlap_us / avg_interval_us if avg_interval_us > 0 else 0.0
    return expected


def analyze_pressure_csv(path: Path, frame_rate: int) -> FileSeries:
    side = get_side_from_name(path.name)
    if not side:
        raise ValueError(f"Cannot infer side from file name: {path.name}")

    counts_by_second: dict[int, int] = defaultdict(int)
    times_us: list[float] = []
    frame_idx_values: list[int] = []
    min_observed_second: int | None = None
    max_observed_second: int | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc

        time_col = find_column(header, {"t_us", "captured_at"})
        frame_idx_col = find_column(header, {"frame_idx", "frame_index"})

        for row_idx, row in enumerate(reader):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                continue

            time_us = extract_time_us(row, time_col, frame_idx_col, frame_rate)
            frame_idx = extract_frame_idx(row, frame_idx_col)
            second = parse_second_from_time_us(time_us, row_idx, frame_rate)
            counts_by_second[second] += 1

            if time_us is not None and math.isfinite(time_us):
                times_us.append(time_us)
            if frame_idx is not None:
                frame_idx_values.append(frame_idx)

            if min_observed_second is None or second < min_observed_second:
                min_observed_second = second
            if max_observed_second is None or second > max_observed_second:
                max_observed_second = second

    date, subject, session_name = get_date_subject_and_session(path)
    return FileSeries(
        date=date,
        subject=subject,
        session_name=session_name,
        side=side,
        counts_by_second=dict(counts_by_second),
        expected_counts_by_second=estimate_expected_counts_by_second(times_us, dict(counts_by_second)),
        min_observed_second=min_observed_second,
        max_observed_second=max_observed_second,
        frame_idx_values=frame_idx_values,
        time_us_values=times_us,
    )


def merge_group_series(series_list: list[FileSeries]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "left_actual": 0.0,
            "right_actual": 0.0,
            "left_expected": 0.0,
            "right_expected": 0.0,
        }
    )
    for series in series_list:
        for second, count in series.counts_by_second.items():
            merged[second][f"{series.side}_actual"] += float(count)
        for second, count in series.expected_counts_by_second.items():
            merged[second][f"{series.side}_expected"] += float(count)
    return dict(sorted(merged.items()))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists and is not empty: {path}\n"
                "Use a new timestamped output dir, or pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "second",
        "left_frame_count",
        "right_frame_count",
        "total_frame_count",
        "left_expected_frame_count",
        "right_expected_frame_count",
        "total_expected_frame_count",
        "is_edge_second",
        "edge_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_png(path: Path, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        return

    seconds = [int(row["second"]) for row in rows]
    left = [float(row["left_frame_count"]) for row in rows]
    right = [float(row["right_frame_count"]) for row in rows]
    left_expected = [
        float(row["left_expected_frame_count"]) if row["left_expected_frame_count"] != "" else math.nan
        for row in rows
    ]
    right_expected = [
        float(row["right_expected_frame_count"]) if row["right_expected_frame_count"] != "" else math.nan
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    ax.plot(seconds, left, color="#1f77b4", marker="o", linewidth=2, label="left actual")
    ax.plot(seconds, right, color="#d62728", marker="o", linewidth=2, label="right actual")
    ax.plot(seconds, left_expected, color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.55, label="left expected")
    ax.plot(seconds, right_expected, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.55, label="right expected")
    ax.set_title(title)
    ax.set_xlabel("second")
    ax.set_ylabel("frame count")
    ax.set_xticks(seconds)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4), dpi=160)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_histogram_edges(values: list[float], desired_bins: int = 50) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        return [min_value, max_value + 1.0]
    bin_count = max(1, min(desired_bins, len(values)))
    width = (max_value - min_value) / float(bin_count)
    if width <= 0:
        width = 1.0
    return [min_value + idx * width for idx in range(bin_count)] + [max_value]


def histogram_counts(values: list[float], edges: list[float]) -> list[int]:
    if len(edges) < 2:
        return []
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        if value <= edges[0]:
            counts[0] += 1
            continue
        if value >= edges[-1]:
            counts[-1] += 1
            continue
        for idx in range(len(edges) - 1):
            left = edges[idx]
            right = edges[idx + 1]
            if left <= value < right:
                counts[idx] += 1
                break
    return counts


def write_t_us_diff_histogram_csv(path: Path, left_diffs: list[float], right_diffs: list[float]) -> None:
    all_diffs = left_diffs + right_diffs
    fieldnames = [
        "bin_left",
        "bin_right",
        "left_count",
        "left_ratio",
        "right_count",
        "right_ratio",
        "total_count",
        "total_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if not all_diffs:
            return
        edges = build_histogram_edges(all_diffs, desired_bins=50)
        left_counts = histogram_counts(left_diffs, edges)
        right_counts = histogram_counts(right_diffs, edges)
        left_total = float(len(left_diffs))
        right_total = float(len(right_diffs))
        total_total = float(len(all_diffs))
        for idx in range(len(edges) - 1):
            total_count = left_counts[idx] + right_counts[idx]
            if total_count == 0:
                continue
            writer.writerow(
                {
                    "bin_left": edges[idx],
                    "bin_right": edges[idx + 1],
                    "left_count": left_counts[idx],
                    "left_ratio": (left_counts[idx] / left_total) if left_total > 0 else 0.0,
                    "right_count": right_counts[idx],
                    "right_ratio": (right_counts[idx] / right_total) if right_total > 0 else 0.0,
                    "total_count": total_count,
                    "total_ratio": (total_count / total_total) if total_total > 0 else 0.0,
                }
            )


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def compute_gap_rows(series_list: list[FileSeries]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series in series_list:
        if len(series.time_us_values) < 2:
            continue
        diffs = [
            series.time_us_values[idx + 1] - series.time_us_values[idx]
            for idx in range(len(series.time_us_values) - 1)
        ]
        positive_diffs = [diff for diff in diffs if diff > 0]
        dt0 = median(positive_diffs)
        if dt0 is None or dt0 <= 0:
            continue
        threshold = 1.5 * dt0
        for idx, diff in enumerate(diffs):
            if diff <= threshold:
                continue
            prev_frame_idx = series.frame_idx_values[idx] if idx < len(series.frame_idx_values) else ""
            next_frame_idx = series.frame_idx_values[idx + 1] if idx + 1 < len(series.frame_idx_values) else ""
            prev_t_us = series.time_us_values[idx]
            next_t_us = series.time_us_values[idx + 1]
            rows.append(
                {
                    "side": series.side,
                    "gap_idx": idx,
                    "prev_frame_idx": prev_frame_idx,
                    "next_frame_idx": next_frame_idx,
                    "prev_t_us": prev_t_us,
                    "next_t_us": next_t_us,
                    "diff_t_us": diff,
                    "dt0_us": dt0,
                    "threshold_us": threshold,
                    "ratio_to_dt0": diff / dt0,
                    "gap_mid_second": (prev_t_us + next_t_us) / 2.0 / 1_000_000.0,
                }
            )
    return rows


def write_gap_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "side",
        "gap_idx",
        "prev_frame_idx",
        "next_frame_idx",
        "prev_t_us",
        "next_t_us",
        "diff_t_us",
        "dt0_us",
        "threshold_us",
        "ratio_to_dt0",
        "gap_mid_second",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_t_us_diff_histogram(path: Path, series_list: list[FileSeries], title: str) -> None:
    diff_map: dict[str, list[float]] = {"left": [], "right": []}
    for series in series_list:
        if len(series.time_us_values) < 2:
            continue
        diffs = [
            series.time_us_values[idx + 1] - series.time_us_values[idx]
            for idx in range(len(series.time_us_values) - 1)
            if series.time_us_values[idx + 1] - series.time_us_values[idx] >= 0
        ]
        diff_map[series.side].extend(diffs)

    if not diff_map["left"] and not diff_map["right"]:
        make_empty_plot(path, title, "No t_us diff data")
        return

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    if diff_map["left"]:
        ax.hist(diff_map["left"], bins=50, alpha=0.45, color="#1f77b4", label="left")
    if diff_map["right"]:
        ax.hist(diff_map["right"], bins=50, alpha=0.45, color="#d62728", label="right")
    ax.set_title(title)
    ax.set_xlabel("t_us diff")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def collect_t_us_diffs(series_list: list[FileSeries]) -> dict[str, list[float]]:
    diff_map: dict[str, list[float]] = {"left": [], "right": []}
    for series in series_list:
        if len(series.time_us_values) < 2:
            continue
        diffs = [
            series.time_us_values[idx + 1] - series.time_us_values[idx]
            for idx in range(len(series.time_us_values) - 1)
            if series.time_us_values[idx + 1] - series.time_us_values[idx] >= 0
        ]
        diff_map[series.side].extend(diffs)
    return diff_map


def write_gap_positions_plot(path: Path, gap_rows: list[dict[str, object]], title: str) -> None:
    if not gap_rows:
        make_empty_plot(path, title, "No gaps over 1.5 x dt0")
        return

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    for side, color in (("left", "#1f77b4"), ("right", "#d62728")):
        side_rows = [row for row in gap_rows if row["side"] == side]
        if not side_rows:
            continue
        ax.scatter(
            [float(row["gap_mid_second"]) for row in side_rows],
            [float(row["ratio_to_dt0"]) for row in side_rows],
            color=color,
            s=36,
            alpha=0.8,
            label=side,
        )
    ax.axhline(1.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("diff / dt0")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def group_rows(series_list: list[FileSeries]) -> list[dict[str, object]]:
    merged = merge_group_series(series_list)
    if not merged:
        return []

    min_observed_second = min(
        series.min_observed_second
        for series in series_list
        if series.min_observed_second is not None
    )
    max_observed_second = max(
        series.max_observed_second
        for series in series_list
        if series.max_observed_second is not None
    )

    max_second = max(merged)
    rows: list[dict[str, object]] = []
    for second in range(max_second + 1):
        counts = merged.get(
            second,
            {
                "left_actual": 0.0,
                "right_actual": 0.0,
                "left_expected": 0.0,
                "right_expected": 0.0,
            },
        )
        left = int(round(counts["left_actual"]))
        right = int(round(counts["right_actual"]))
        edge_parts: list[str] = []
        if second == min_observed_second:
            edge_parts.append("start")
        if second == max_observed_second:
            edge_parts.append("end")
        left_expected: float | str = float(counts["left_expected"])
        right_expected: float | str = float(counts["right_expected"])
        if "end" in edge_parts:
            left_expected = ""
            right_expected = ""
        rows.append(
            {
                "second": second,
                "left_frame_count": left,
                "right_frame_count": right,
                "total_frame_count": left + right,
                "left_expected_frame_count": left_expected,
                "right_expected_frame_count": right_expected,
                "total_expected_frame_count": (
                    "" if "end" in edge_parts else float(left_expected) + float(right_expected)
                ),
                "is_edge_second": "true" if edge_parts else "",
                "edge_type": ",".join(edge_parts),
            }
        )
    return rows


def build_overall_frame_count_distribution_rows(
    grouped: dict[tuple[str, str, str], list[FileSeries]]
) -> list[dict[str, object]]:
    bucket_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "non_end_left": 0,
            "non_end_right": 0,
            "end_left": 0,
            "end_right": 0,
        }
    )

    for series_list in grouped.values():
        rows = group_rows(series_list)
        for row in rows:
            edge_type = str(row["edge_type"])
            is_end = "end" in {part.strip() for part in edge_type.split(",") if part.strip()}
            left_frame_count = int(row["left_frame_count"])
            right_frame_count = int(row["right_frame_count"])

            left_bucket = bucket_counts[left_frame_count]
            right_bucket = bucket_counts[right_frame_count]
            if is_end:
                left_bucket["end_left"] += 1
                right_bucket["end_right"] += 1
            else:
                left_bucket["non_end_left"] += 1
                right_bucket["non_end_right"] += 1

    if not bucket_counts:
        return []

    total_samples = float(
        sum(
            bucket["non_end_left"] + bucket["non_end_right"] + bucket["end_left"] + bucket["end_right"]
            for bucket in bucket_counts.values()
        )
    )
    rows: list[dict[str, object]] = []
    for frame_count in sorted(bucket_counts):
        bucket = bucket_counts[frame_count]
        non_end_left_count = bucket["non_end_left"]
        non_end_right_count = bucket["non_end_right"]
        end_left_count = bucket["end_left"]
        end_right_count = bucket["end_right"]
        non_end_count = non_end_left_count + non_end_right_count
        end_count = end_left_count + end_right_count
        total_count = non_end_count + end_count
        if total_count == 0:
            continue
        rows.append(
            {
                "frame_count": frame_count,
                "non_end_left_count": non_end_left_count,
                "non_end_right_count": non_end_right_count,
                "non_end_count": non_end_count,
                "end_left_count": end_left_count,
                "end_right_count": end_right_count,
                "end_count": end_count,
                "total_count": total_count,
                "non_end_ratio": (non_end_count / total_samples) if total_samples > 0 else 0.0,
                "end_ratio": (end_count / total_samples) if total_samples > 0 else 0.0,
                "total_ratio": (total_count / total_samples) if total_samples > 0 else 0.0,
            }
        )
    return rows


def write_frame_count_distribution_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame_count",
        "non_end_left_count",
        "non_end_right_count",
        "non_end_count",
        "end_left_count",
        "end_right_count",
        "end_count",
        "total_count",
        "non_end_ratio",
        "end_ratio",
        "total_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_frame_count_distribution_png(path: Path, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        make_empty_plot(path, title, "No frame-count distribution data")
        return

    frame_counts = [int(row["frame_count"]) for row in rows]
    non_end_counts = [int(row["non_end_count"]) for row in rows]
    end_counts = [int(row["end_count"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    ax.bar(frame_counts, non_end_counts, color="#1f77b4", width=0.9, label="non-end")
    ax.bar(frame_counts, end_counts, bottom=non_end_counts, color="#d62728", width=0.9, label="end")
    ax.set_title(title)
    ax.set_xlabel("frame count per second")
    ax.set_ylabel("sample count")
    if len(frame_counts) <= 25:
        ax.set_xticks(frame_counts)
    else:
        step = max(1, len(frame_counts) // 20)
        ax.set_xticks(frame_counts[::step])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_dataset_object_id(session_name: str, side: str) -> str:
    match = re.search(r"(S\d+)_(\d+)$", session_name)
    if match:
        return f"{match.group(1)}{match.group(2)}-{side[:1].upper()}"
    return f"{session_name}-{side[:1].upper()}"


def build_dataset_id(session_name: str) -> str:
    match = re.search(r"(S\d+)_(\d+)$", session_name)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return session_name


def build_expected_dataset_core(subject: str, action_index: int) -> str:
    return f"{subject}{action_index:02d}"


def build_expected_dataset_id(subject: str, action_index: int, group_index: int) -> str:
    return f"{build_expected_dataset_core(subject, action_index)}{group_index}"


def build_expected_session_suffix(subject: str, action_index: int, group_index: int) -> str:
    return f"{build_expected_dataset_core(subject, action_index)}_{group_index}"


def parse_session_action_and_group(session_name: str, subject: str) -> tuple[int, int] | None:
    match = re.search(rf"{re.escape(subject)}(\d{{2}})_(\d+)$", session_name)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def discover_session_name_map(
    subject_dirs: dict[tuple[str, str], Path]
) -> dict[tuple[str, str, int, int], str]:
    session_name_map: dict[tuple[str, str, int, int], str] = {}
    for (date, subject), subject_dir in subject_dirs.items():
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            parsed = parse_session_action_and_group(session_dir.name, subject)
            if parsed is None:
                continue
            action_index, group_index = parsed
            session_name_map[(date, subject, action_index, group_index)] = session_dir.name
    return session_name_map


def compute_session_max_gap_s(series_list: list[FileSeries]) -> float:
    max_gap_s = 0.0
    for series in series_list:
        if len(series.time_us_values) < 2:
            continue
        for idx in range(len(series.time_us_values) - 1):
            diff_us = series.time_us_values[idx + 1] - series.time_us_values[idx]
            if diff_us > 0:
                max_gap_s = max(max_gap_s, diff_us / 1_000_000.0)
    return max_gap_s


def build_quality_classification_rows(
    grouped: dict[tuple[str, str, str], list[FileSeries]],
    subject_dirs: dict[tuple[str, str], Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session_name_map = discover_session_name_map(subject_dirs)
    grouped_lookup = {(date, subject, session_name): series_list for (date, subject, session_name), series_list in grouped.items()}

    for (date, subject), _subject_dir in sorted(subject_dirs.items()):
        for action_index in range(1, EXPECTED_ACTIONS_PER_SUBJECT + 1):
            for group_index in range(1, EXPECTED_GROUPS_PER_ACTION + 1):
                dataset_id = build_expected_dataset_id(subject, action_index, group_index)
                session_name = session_name_map.get((date, subject, action_index, group_index), "")

                if not session_name:
                    continue

                series_list = grouped_lookup.get((date, subject, session_name))
                if not series_list:
                    continue

                session_rows = group_rows(series_list)
                left_total = sum(int(row["left_frame_count"]) for row in session_rows)
                right_total = sum(int(row["right_frame_count"]) for row in session_rows)
                zero_frame_second_count = sum(
                    1
                    for row in session_rows
                    if int(row["left_frame_count"]) == 0 or int(row["right_frame_count"]) == 0
                )
                max_gap_s = compute_session_max_gap_s(series_list)

                missing_sides: list[str] = []
                if left_total == 0:
                    missing_sides.append("L")
                if right_total == 0:
                    missing_sides.append("R")

                if missing_sides:
                    quality_class = "C"
                    reason = f"single-foot missing: {','.join(missing_sides)}"
                elif zero_frame_second_count == 0 and max_gap_s < 0.3:
                    quality_class = "A"
                    reason = "both feet complete, zero-frame seconds = 0, max gap < 0.3s"
                else:
                    quality_class = "B"
                    reason_parts: list[str] = []
                    if zero_frame_second_count > 0:
                        reason_parts.append(f"zero-frame seconds = {zero_frame_second_count}")
                    if 0.3 <= max_gap_s <= 1.0:
                        reason_parts.append(f"max gap = {max_gap_s:.6f}s")
                    elif max_gap_s > 1.0:
                        reason_parts.append(f"max gap > 1s ({max_gap_s:.6f}s)")
                    reason = "; ".join(reason_parts) if reason_parts else "review needed"

                rows.append(
                    {
                        "quality_class": quality_class,
                        "date": date,
                        "subject": subject,
                        "session_name": session_name,
                        "dataset_id": dataset_id,
                        "session_status": "session_present",
                        "missing_side_objects": (
                            ",".join(
                                build_dataset_object_id(session_name, "left" if side == "L" else "right")
                                for side in missing_sides
                            )
                            if missing_sides
                            else ""
                        ),
                        "zero_frame_second_count": zero_frame_second_count,
                        "max_gap_s": max_gap_s,
                        "left_total_frames": left_total,
                        "right_total_frames": right_total,
                        "reason": reason,
                    }
                )
    return rows


def write_quality_classification_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "quality_class",
        "date",
        "subject",
        "session_name",
        "dataset_id",
        "session_status",
        "missing_side_objects",
        "zero_frame_second_count",
        "max_gap_s",
        "left_total_frames",
        "right_total_frames",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_session_extra_outputs(group_dir: Path, series_list: list[FileSeries], title_prefix: str) -> None:
    diff_map = collect_t_us_diffs(series_list)
    write_t_us_diff_histogram_csv(
        group_dir / "t_us_diff_histogram.csv",
        diff_map["left"],
        diff_map["right"],
    )
    write_t_us_diff_histogram(
        group_dir / "t_us_diff_histogram.png",
        series_list,
        title=f"{title_prefix} t_us first-difference histogram",
    )
    gap_rows = compute_gap_rows(series_list)
    write_gap_csv(group_dir / "gap_list.csv", gap_rows)
    write_gap_positions_plot(
        group_dir / "gap_positions.png",
        gap_rows,
        title=f"{title_prefix} gaps over 1.5 x dt0",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze raw tactile pressure CSV files and write per-session per-second frame-count outputs."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Raw data roots to scan. Defaults to the new 1_Data root plus legacy experiment roots.",
    )
    parser.add_argument("--output-dir", type=Path, help="Explicit output run directory.")
    parser.add_argument(
        "--file-pattern",
        dest="file_patterns",
        action="append",
        help="Additional filename pattern to scan.",
    )
    parser.add_argument("--video-frame-rate", type=int, default=DEFAULT_VIDEO_FRAME_RATE)
    parser.add_argument("--max-files", type=int, help="Analyze only the first N discovered files.")
    parser.add_argument("--start-s", type=int, help="Only include subjects from this S number, inclusive. Example: 5 for S5.")
    parser.add_argument("--end-s", type=int, help="Only include subjects up to this S number, inclusive. Example: 14 for S14.")
    parser.add_argument("--overwrite", action="store_true", help="Clear an existing output directory before writing.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    roots = args.roots or DEFAULT_ROOT_DIRS
    file_patterns = tuple(args.file_patterns) if args.file_patterns else DEFAULT_FILE_PATTERNS
    subject_dirs = {
        key: path
        for key, path in discover_subject_dirs(roots).items()
        if subject_in_range(key[1], args.start_s, args.end_s)
    }

    files = discover_pressure_files(roots, file_patterns)
    if args.start_s is not None or args.end_s is not None:
        filtered_files: list[Path] = []
        for file_path in files:
            subject_number = get_subject_number(file_path)
            if subject_number is None:
                continue
            if args.start_s is not None and subject_number < args.start_s:
                continue
            if args.end_s is not None and subject_number > args.end_s:
                continue
            filtered_files.append(file_path)
        files = filtered_files
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files and not subject_dirs:
        raise SystemExit("No pressure CSV files or subject directories found.")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_PARENT_DIR / f"pressure_stats_{stamp}"
    output_dir = prepare_output_dir(output_dir, overwrite=args.overwrite)

    grouped: dict[tuple[str, str, str], list[FileSeries]] = defaultdict(list)
    all_series: list[FileSeries] = []

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path}")
        series = analyze_pressure_csv(file_path, frame_rate=args.video_frame_rate)
        if not series.date or not series.subject or not series.session_name:
            continue
        grouped[(series.date, series.subject, series.session_name)].append(series)
        all_series.append(series)

    if not all_series:
        raise SystemExit("No valid pressure CSV files were analyzed.")

    for (date, subject, session_name), series_list in sorted(grouped.items(), key=lambda item: item[0]):
        group_dir = ensure_dir(output_dir / date / subject / session_name)
        rows = group_rows(series_list)
        write_csv(group_dir / "frame_per_second.csv", rows)
        write_png(
            group_dir / "frame_per_second.png",
            rows,
            title=f"{date}/{subject}/{session_name} frame count per second",
        )
        write_session_extra_outputs(
            group_dir,
            series_list,
            title_prefix=f"{date}/{subject}/{session_name}",
        )

    overall_dir = ensure_dir(output_dir / "overall")
    overall_rows = build_overall_frame_count_distribution_rows(grouped)
    write_frame_count_distribution_csv(overall_dir / "frame_per_second.csv", overall_rows)
    write_frame_count_distribution_png(
        overall_dir / "frame_per_second.png",
        overall_rows,
        title="overall frame-count distribution per second",
    )
    write_quality_classification_csv(
        overall_dir / "missing_pressure_objects.csv",
        build_quality_classification_rows(grouped, subject_dirs),
    )
    write_session_extra_outputs(
        overall_dir,
        all_series,
        title_prefix="overall",
    )

    print("")
    if args.max_files is not None:
        print(f"Partial run: only the first {args.max_files} discovered files were analyzed.")
    print(f"Done. Results saved to: {output_dir}")
    print("Output structure:")
    print(f"  {output_dir}/overall")
    print(f"  {output_dir}/<date>/<Si>/<session_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
