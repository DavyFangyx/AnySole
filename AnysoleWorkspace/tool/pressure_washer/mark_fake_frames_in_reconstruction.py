"""Mark fake frames in reconstructed tactile CSVs.

This script post-processes a reconstructed tactile dataset directory.
For each session:
- add a `fake` column immediately after `valid_mask`
- remove `source_frame_idx` and `source_t_us`
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
        if field in {"source_frame_idx", "source_t_us"}:
            continue
        output_fields.append(field)
        if field == "valid_mask":
            output_fields.append("fake")
    if "valid_mask" not in fieldnames:
        raise ValueError("Input pressure CSV is missing required field `valid_mask`.")
    return output_fields


def process_pressure_csv(input_path: Path, output_path: Path, fake_frame_idxs: set[int]) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty pressure CSV: {input_path}")
        output_fields = transformed_fieldnames(reader.fieldnames)
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
                    if field in {"source_frame_idx", "source_t_us"}:
                        continue
                    output_row[field] = row[field]
                    if field == "valid_mask":
                        output_row["fake"] = fake_value
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
