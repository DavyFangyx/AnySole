"""Central path definitions for the PressureWasher pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
CONFIGS_DIR = PROJECT_DIR / "configs"

# Runtime data belongs to the workspace sources area; this directory contains
# generated tactile datasets and review artifacts, not executable code.
PRESSURE_SOURCE_DIR = WORKSPACE_DIR / "sources" / "PressureWasher"
OUTPUTS_DIR = PRESSURE_SOURCE_DIR

STATS_OUTPUTS_DIR = OUTPUTS_DIR / "stats"
RECONSTRUCTED_OUTPUTS_DIR = OUTPUTS_DIR / "reconstructed"
FAKE_MARKED_OUTPUTS_DIR = OUTPUTS_DIR / "fake_marked"
ENCODED_OUTPUTS_DIR = OUTPUTS_DIR / "encoded"

FAKE_FRAME_RULES_DIR = CONFIGS_DIR / "fake_frames"
DEFAULT_RAW_ROOT_DIR = Path(
    os.environ.get("PRESSUREWASHER_RAW_ROOT", "/data/lizhe/projects/Tactile/1_Data")
).expanduser()


def stage_output_dir(parent: Path, name: str) -> Path:
    return parent / name
