#!/usr/bin/env python3
"""Run one stage of the PressureWasher preparation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRESSURE_ROOT = REPO_ROOT / "AnysoleWorkspace" / "tools" / "PressureWasher"
PRESSURE_DATA_ROOT = REPO_ROOT / "AnysoleWorkspace" / "sources" / "PressureWasher"
PYTHON = sys.executable

STAGES = {
    "inspect": "analyze_pressure_csv_stats.py",
    "reconstruct": "reconstruct_pressure_dataset.py",
    "mark-fake": "mark_fake_frames_in_reconstruction.py",
    "encode": "encode_fake_marked_reconstruction.py",
}
STAGE_OUTPUTS = {
    "inspect": "stats",
    "reconstruct": "reconstructed",
    "mark-fake": "fake_marked",
    "encode": "encoded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AnySole pressure dependencies.")
    parser.add_argument("stage", choices=tuple(STAGES), help="Pipeline stage to run.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip when this stage already has output.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true", help="Run the stage and overwrite fixed output paths.")
    parser.add_argument("--input", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_output = PRESSURE_DATA_ROOT / STAGE_OUTPUTS[args.stage]
    if args.skip_existing and stage_output.is_dir() and any(stage_output.iterdir()):
        print("skip %s (%s already contains output)" % (args.stage, stage_output))
        return 0
    script = PRESSURE_ROOT / STAGES[args.stage]
    command = [PYTHON, str(script)]
    if args.stage == "mark-fake" and args.input:
        command += ["-input", str(args.input)]
    elif args.stage == "encode" and args.input:
        command += ["-input", str(args.input)]
    elif args.stage in ("mark-fake", "encode"):
        output_root = PRESSURE_DATA_ROOT / ("reconstructed" if args.stage == "mark-fake" else "fake_marked")
        candidates = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.is_dir() else []
        if not candidates:
            raise FileNotFoundError("No input dataset found under %s" % output_root)
        command += ["-input", str(candidates[-1])]
    if args.force and args.stage in ("mark-fake", "encode"):
        command.append("--overwrite")
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
