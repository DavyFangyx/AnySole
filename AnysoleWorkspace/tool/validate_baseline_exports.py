#!/usr/bin/env python3
"""Validate the unified baseline eval_motion export contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
RESULTS = ROOT / "results"
MODEL_DIRS = {
    "motionpro": "MotionPRO",
    "pressure_toolkit": "pressure_tookit",
    "vp_mocap": "VP-MoCap",
}


def sessions(split: str) -> list[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    result = []
    with (WORKSPACE / "splits" / "default" / "splits.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    result.append(value)
    return sorted(set(result))


def manifest() -> dict[str, dict]:
    result = {}
    with (WORKSPACE / "manifests" / "session_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["session_id"]] = row
    return result


def validate_file(path: Path, expected_frames: int) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"joint_xyz_world", "joint_names", "valid_mask"}
            missing = sorted(required - set(data.files))
            if missing:
                return f"missing_keys={missing}"
            joints = np.asarray(data["joint_xyz_world"])
            names = np.asarray(data["joint_names"]).reshape(-1)
            valid = np.asarray(data["valid_mask"]).reshape(-1)
            if joints.shape != (expected_frames, 24, 3):
                return f"joint_shape={joints.shape}, expected=({expected_frames},24,3)"
            if len(names) != 24 or len(valid) != expected_frames:
                return f"name/mask_shape=({len(names)},{len(valid)})"
            if not np.isfinite(joints[valid.astype(bool)]).all():
                return "nonfinite_valid_joints"
    except Exception as exc:  # noqa: BLE001
        return f"read_error={type(exc).__name__}:{exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--model", choices=tuple(MODEL_DIRS) + ("all",), default="all")
    args = parser.parse_args()
    manifest_rows = manifest()
    selected_models = tuple(MODEL_DIRS) if args.model == "all" else (args.model,)
    selected_sessions = sessions(args.split)
    failures = 0
    for model in selected_models:
        root = RESULTS / MODEL_DIRS[model] / "predictions" / "eval_motion"
        counts = {"ok": 0, "missing": 0, "invalid": 0}
        for sid in selected_sessions:
            if sid not in manifest_rows:
                print(f"{model} {sid}: manifest_missing")
                failures += 1
                continue
            reason = validate_file(root / f"{sid}.npz", int(manifest_rows[sid]["n_frames"]))
            if reason is None:
                counts["ok"] += 1
            elif reason == "missing":
                counts["missing"] += 1
            else:
                counts["invalid"] += 1
                print(f"{model} {sid}: {reason}")
        failures += counts["missing"] + counts["invalid"]
        print(f"{model}: {counts['ok']}/{len(selected_sessions)} valid, "
              f"missing={counts['missing']} invalid={counts['invalid']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
