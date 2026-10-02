"""Mark fake frames in reconstructed tactile CSVs.

This script post-processes a reconstructed tactile dataset directory.
For each session:
- add a `fake` column immediately after `valid_mask`
- remove `source_frame_idx` but keep `source_t_us` (raw physical sample
  timestamps; build_shared needs them for absolute-time resampling, per the
  S2 clock-scan ruling, Baselines/决策/05_R1-R3_风险裁定.md §三 证据 C)
- add a `raw_t_us` column: the reconstructed physical wall-clock time of
  every shipped row, computed from the per-segment (t_us, source_t_us)
  anchors in `reconstruction_manifest.csv`.  The shipped 40 Hz grid is
  linear in t_us; each segment/bridge block maps its t_us range onto the raw
  clock linearly (two-point slope), absorbing the recorder-clock drift.
  Rows that must never be selected as a pressure source (valid_mask=0 or
  fake=1) get an empty raw_t_us.
- mark `fake=1` when reconstructed `frame_idx` is listed in
  `PressureWasher/configs/fake_frames/*_fake_frames.csv` for that side

The output directory preserves the same date/subject/session structure and
copies `reconstruction_manifest.csv` unchanged.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


from pwlib.paths import FAKE_FRAME_RULES_DIR, FAKE_MARKED_OUTPUTS_DIR

DEFAULT_FAKE_DIR = FAKE_FRAME_RULES_DIR
PRESSURE_FILES = {
    "left": "pressure_left.csv",
    "right": "pressure_right.csv",
}
SESSION_CODE_RE = re.compile(r"(S\d+)_(\d+)$", re.IGNORECASE)


@dataclass
class FakeFrameSets:
    left: set[int]
    right: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark fake frames in a reconstructed tactile dataset.")
    parser.add_argument(
        "-input",
        "--input-reconstruction-dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input reconstructed dataset directory, e.g. reconstruction_20260817_161459",
    )
    parser.add_argument(
        "-output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/fake_marked/<input>_fake_marked",
    )
    parser.add_argument(
        "--fake-csv-dir",
        type=Path,
        default=DEFAULT_FAKE_DIR,
        help="Directory containing *_fake_frames.csv files.",
    )
    parser.add_argument("--date", type=str, default=None, help="Process only one date directory.")
    parser.add_argument("--subject", type=str, default=None, help="Process only one subject directory.")
    parser.add_argument("--session-name", type=str, default=None, help="Process only one session directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_matches_filters(session_dir: Path, date: str | None, subject: str | None, session_name: str | None) -> bool:
    if date is not None and session_dir.parent.parent.name != date:
        return False
    if subject is not None and session_dir.parent.name.upper() != subject.upper():
        return False
    if session_name is not None and session_dir.name != session_name:
        return False
    return True


def discover_session_dirs(input_dir: Path, date: str | None, subject: str | None, session_name: str | None) -> list[Path]:
    session_dirs: list[Path] = []
    for pressure_left in sorted(input_dir.glob("*/*/*/pressure_left.csv")):
        session_dir = pressure_left.parent
        if not (session_dir / PRESSURE_FILES["right"]).is_file():
            continue
        if not session_matches_filters(session_dir, date, subject, session_name):
            continue
        session_dirs.append(session_dir)
    return session_dirs


def session_code_from_name(session_name: str) -> str | None:
    match = SESSION_CODE_RE.search(session_name)
    if match is None:
        return None
    return f"{match.group(1).upper()}{match.group(2)}"


def load_fake_frame_sets(fake_csv_dir: Path) -> dict[str, FakeFrameSets]:
    fake_sets: dict[str, FakeFrameSets] = {}
    for csv_path in sorted(fake_csv_dir.glob("*_fake_frames.csv")):
        session_code = csv_path.name.removesuffix("_fake_frames.csv").upper()
        left_frames: set[int] = set()
        right_frames: set[int] = set()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                left_value = row.get("left_frame_idx", "").strip()
                right_value = row.get("right_frame_idx", "").strip()
                if left_value:
                    left_frames.add(int(float(left_value)))
                if right_value:
                    right_frames.add(int(float(right_value)))
        fake_sets[session_code] = FakeFrameSets(left=left_frames, right=right_frames)
    return fake_sets


def transformed_fieldnames(fieldnames: list[str]) -> list[str]:
    output_fields: list[str] = []
    for field in fieldnames:
        if field == "source_frame_idx":
            continue
        output_fields.append(field)
        if field == "valid_mask":
            output_fields.append("fake")
    if "valid_mask" not in fieldnames:
        raise ValueError("Input pressure CSV is missing required field `valid_mask`.")
    if "source_t_us" in fieldnames:
        output_fields.append("raw_t_us")
    return output_fields


def load_clock_blocks(manifest_path: Path, side: str) -> list[tuple[float, float, float, float]]:
    """(t_start, t_end, raw_start, raw_end) per segment/bridge block, by t_us range."""
    blocks: list[tuple[float, float, float, float]] = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("side") != side or row.get("block_type") not in {"segment", "bridge"}:
                continue
            t_start, t_end = float(row["t_us_start"]), float(row["t_us_end"])
            raw_start, raw_end = float(row["source_t_us_start"]), float(row["source_t_us_end"])
            if t_end > t_start:
                blocks.append((t_start, t_end, raw_start, raw_end))
    return sorted(blocks)


def raw_time_for_t(blocks: list[tuple[float, float, float, float]], t: float) -> float:
    """Raw-clock time of a shipped row at synthetic time t (linear two-point map)."""
    if not blocks:
        return float("nan")
    if t <= blocks[0][0]:
        t_start, t_end, raw_start, raw_end = blocks[0]
    elif t >= blocks[-1][1]:
        t_start, t_end, raw_start, raw_end = blocks[-1]
    else:
        lo, hi = 0, len(blocks) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if blocks[mid][1] < t:
                lo = mid + 1
            else:
                hi = mid
        t_start, t_end, raw_start, raw_end = blocks[lo]
    return raw_start + (t - t_start) * (raw_end - raw_start) / (t_end - t_start)


def process_pressure_csv(input_path: Path, output_path: Path, fake_frame_idxs: set[int]) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty pressure CSV: {input_path}")
        output_fields = transformed_fieldnames(reader.fieldnames)
        with_clock = "raw_t_us" in output_fields
        side = "left" if input_path.name.startswith("pressure_left") else "right"
        blocks = load_clock_blocks(input_path.parent / "reconstruction_manifest.csv", side) if with_clock else []
        rows_written = 0
        fake_rows = 0
        with output_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                frame_idx = int(float(row["frame_idx"]))
                fake_value = 1 if frame_idx in fake_frame_idxs else 0
                output_row: dict[str, object] = {}
                for field in reader.fieldnames:
                    if field == "source_frame_idx":
                        continue
                    output_row[field] = row[field]
                    if field == "valid_mask":
                        output_row["fake"] = fake_value
                if with_clock:
                    if fake_value or int(float(row.get("valid_mask", 1))) != 1:
                        output_row["raw_t_us"] = ""
                    else:
                        raw = raw_time_for_t(blocks, float(row["t_us"]))
                        output_row["raw_t_us"] = "" if raw != raw else repr(raw)
                writer.writerow(output_row)
                rows_written += 1
                fake_rows += fake_value
    return rows_written, fake_rows


def copy_manifest_if_present(session_dir: Path, output_session_dir: Path) -> None:
    manifest_path = session_dir / "reconstruction_manifest.csv"
    if manifest_path.is_file():
        shutil.copy2(manifest_path, output_session_dir / manifest_path.name)


def output_session_dir_for(output_root: Path, session_dir: Path) -> Path:
    return output_root / session_dir.parent.parent.name / session_dir.parent.name / session_dir.name


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input reconstruction directory does not exist: {args.input_dir}")
    if not args.fake_csv_dir.is_dir():
        raise SystemExit(f"Fake frame CSV directory does not exist: {args.fake_csv_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = FAKE_MARKED_OUTPUTS_DIR / f"{args.input_dir.name}_fake_marked"
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    fake_sets = load_fake_frame_sets(args.fake_csv_dir)
    session_dirs = discover_session_dirs(args.input_dir, args.date, args.subject, args.session_name)
    if not session_dirs:
        raise SystemExit("No matching reconstructed sessions were found.")

    processed_sessions = 0
    processed_rows = 0
    marked_fake_rows = 0
    sessions_with_rules = 0

    for session_dir in session_dirs:
        session_code = session_code_from_name(session_dir.name)
        session_fake_sets = fake_sets.get(session_code or "", FakeFrameSets(left=set(), right=set()))
        if session_code in fake_sets:
            sessions_with_rules += 1

        output_session_dir = output_session_dir_for(output_dir, session_dir)
        output_session_dir.mkdir(parents=True, exist_ok=True)

        for side, file_name in PRESSURE_FILES.items():
            input_csv = session_dir / file_name
            output_csv = output_session_dir / file_name
            fake_frame_idxs = session_fake_sets.left if side == "left" else session_fake_sets.right
            row_count, fake_count = process_pressure_csv(input_csv, output_csv, fake_frame_idxs)
            processed_rows += row_count
            marked_fake_rows += fake_count

        copy_manifest_if_present(session_dir, output_session_dir)
        processed_sessions += 1

    print(f"Processed {processed_sessions} sessions into {output_dir}")
    print(f"Sessions with fake-frame rules: {sessions_with_rules}")
    print(f"Processed pressure rows: {processed_rows}")
    print(f"Marked fake rows: {marked_fake_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
