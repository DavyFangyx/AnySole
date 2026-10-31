#!/usr/bin/env python3
"""Shared-queue status (configs/queue + running/done/failed)."""

from pathlib import Path
import os

root = Path(os.environ.get("ANYSOLE_QUEUE_STATE_ROOT", Path(__file__).resolve().parents[1]))
values = []
for state in ("queue", "running", "done", "failed"):
    values.append("%s=%d" % (state, len(list((root / state).glob("*.conf")))))
print("shared queue:", "  ".join(values))
