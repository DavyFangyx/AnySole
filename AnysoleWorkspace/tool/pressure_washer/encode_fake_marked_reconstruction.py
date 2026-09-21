"""Encode fake-marked reconstructed tactile CSVs into per-side mask files."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from pwlib.paths import ENCODED_OUTPUTS_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
PRESSURE_FILES = {
    "left": "pressure_left.csv",
    "right": "pressure_right.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode fake-marked reconstructed tactile data.")
    parser.add_argument(
        "-input",
        "--input-dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input fake-marked reconstruction directory or a single pressure CSV.",
    )
    parser.add_argument(
        "-output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/encoded/<input>_encoded for directories, outputs/encoded/<file>_encoded.npy for files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    return parser.parse_args()


def prepare_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def transform_fieldnames(fieldnames: list[str]) -> list[str]:
    if "fake" not in fieldnames:
        raise ValueError("Input CSV must contain a `fake` column.")
    return [name for name in fieldnames if name != "fake"]


def encode_pressure_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty pressure CSV: {input_path}")
        fake_mask: list[int] = []
        rows = 0
        for row in reader:
            fake_mask.append(int(float(row["fake"])))
            rows += 1
    np.save(output_path, np.asarray(fake_mask, dtype=np.uint8))
    return rows, int(sum(fake_mask))


def session_output_dir(output_root: Path, session_dir: Path, input_root: Path) -> Path:
    try:
        rel = session_dir.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"Session directory is not inside input root: {session_dir}") from exc
    return output_root / rel


def copy_manifest_if_present(session_dir: Path, output_session_dir: Path) -> None:
    manifest_path = session_dir / "reconstruction_manifest.csv"
    if manifest_path.is_file():
        shutil.copy2(manifest_path, output_session_dir / manifest_path.name)


def process_session(session_dir: Path, output_root: Path, input_root: Path) -> tuple[int, int]:
    output_session_dir = session_output_dir(output_root, session_dir, input_root)
    output_session_dir.mkdir(parents=True, exist_ok=True)

    rows_total = 0
    fake_total = 0
    for side, file_name in PRESSURE_FILES.items():
        input_csv = session_dir / file_name
        if not input_csv.is_file():
            raise FileNotFoundError(f"Missing pressure CSV: {input_csv}")
        rows, fake_rows = encode_pressure_csv(input_csv, output_session_dir / f"fake_mask_{side}.npy")
        rows_total += rows
        fake_total += fake_rows

    copy_manifest_if_present(session_dir, output_session_dir)
    return rows_total, fake_total


def process_single_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return encode_pressure_csv(input_path, output_path)


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input path does not exist: {args.input_dir}")

    if args.input_dir.is_file():
        output_path = args.output_dir
        if output_path is None:
            output_path = ENCODED_OUTPUTS_DIR / f"{args.input_dir.stem}_encoded.npy"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output file already exists: {output_path}")
        rows, fake_rows = process_single_csv(args.input_dir, output_path)
        print(f"Encoded 1 CSV into {output_path} ({rows} rows, fake={fake_rows})")
        return 0

    input_root = args.input_dir
    output_root = args.output_dir
    if output_root is None:
        output_root = ENCODED_OUTPUTS_DIR / f"{input_root.name}_encoded"
    prepare_dir(output_root, args.overwrite)

    session_dirs = [
        path.parent
        for path in sorted(input_root.rglob("pressure_left.csv"))
        if (path.parent / "pressure_right.csv").is_file()
    ]
    if not session_dirs:
        raise SystemExit("No reconstructed sessions were found.")

    processed_sessions = 0
    processed_rows = 0
    processed_fake = 0
    for session_dir in session_dirs:
        rows, fake_rows = process_session(session_dir, output_root, input_root)
        processed_sessions += 1
        processed_rows += rows
        processed_fake += fake_rows

    print(f"Encoded {processed_sessions} sessions into {output_root}")
    print(f"Processed pressure rows: {processed_rows}")
    print(f"Fake rows: {processed_fake}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
