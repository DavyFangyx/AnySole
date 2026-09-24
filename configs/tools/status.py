#!/usr/bin/env python3
"""Shared-queue status (configs/queue + running/done/failed)."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]  # configs/
values = []
for state in ("queue", "running", "done", "failed"):
    values.append("%s=%d" % (state, len(list((root / state).glob("*.conf")))))
print("shared queue:", "  ".join(values))
