#!/usr/bin/env python3
"""Frozen public schema constants for shared session artifacts."""

from __future__ import annotations

from typing import Final

SESSION_SCHEMA_VERSION: Final = "shared.session.v1"
FRAME_KEYS: Final = ("frame_id", "visual_time_s", "mocap_time_s", "valid", "fake")
PRESSURE_KEYS: Final = (
    "frame_id",
    "left48",
    "right48",
    "valid",
    "fake",
    "unit",
    "source_file_left",
    "source_file_right",
)
CANONICAL_SPLIT_COUNTS: Final = {"train": 92, "val": 12, "test": 36}
