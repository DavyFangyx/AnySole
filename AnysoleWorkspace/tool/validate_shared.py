#!/usr/bin/env python3
"""Validate shared facts, public representations and dependency boundaries."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from shared_schema import FRAME_KEYS, PRESSURE_KEYS


def validate_session(path: Path) -> list[str]:
    errors: list[str] = []
    sid = path.name
    for name in ("session.json", "frames.npz", "pressure_48.npz"):
        if not (path / name).is_file():
            errors.append(f"{sid}: missing {name}")
    if errors:
        return errors
    try:
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
        frames = np.load(path / "frames.npz", allow_pickle=True)
        pressure = np.load(path / "pressure_48.npz", allow_pickle=True)
        if set(FRAME_KEYS) - set(frames.files):
            errors.append(f"{sid}: frames missing {sorted(set(FRAME_KEYS) - set(frames.files))}")
        if set(PRESSURE_KEYS) - set(pressure.files):
            errors.append(f"{sid}: pressure missing {sorted(set(PRESSURE_KEYS) - set(pressure.files))}")
        if errors:
            return errors
        n = len(frames["frame_id"])
        if meta.get("frame_count") != n:
            errors.append(f"{sid}: session frame_count mismatch")
        frame_id = frames["frame_id"]
        if frame_id.dtype != np.int64 or not np.array_equal(frame_id, np.arange(n, dtype=np.int64)):
            errors.append(f"{sid}: frame_id is not int64 monotonic 0..T-1")
        for key in ("visual_time_s", "mocap_time_s"):
            if frames[key].dtype != np.float64 or len(frames[key]) != n:
                errors.append(f"{sid}: bad frame array {key}")
        for key in ("valid", "fake"):
            if frames[key].dtype != np.uint8 or len(frames[key]) != n:
                errors.append(f"{sid}: bad frame array {key}")
        if np.any(frames["valid"] & frames["fake"]):
            errors.append(f"{sid}: valid/fake overlap")
        if not np.array_equal(pressure["frame_id"], frame_id):
            errors.append(f"{sid}: pressure frame ids differ")
        for key in ("left48", "right48"):
            if pressure[key].dtype != np.float32 or pressure[key].shape != (n, 48):
                errors.append(f"{sid}: {key} must be float32[T,48]")
        for key in ("valid", "fake"):
            if pressure[key].dtype != np.uint8 or pressure[key].shape != (n,):
                errors.append(f"{sid}: pressure {key} must be uint8[T]")
        if np.any(pressure["valid"] & pressure["fake"]):
            errors.append(f"{sid}: pressure valid/fake overlap")
        if not np.array_equal(pressure["valid"], frames["valid"]) or not np.array_equal(pressure["fake"], frames["fake"]):
            errors.append(f"{sid}: frame and pressure quality flags differ")
        if not (path / "rgb").is_dir() or len(list((path / "rgb").iterdir())) != n:
            errors.append(f"{sid}: RGB frame count mismatch")
    except Exception as exc:
        errors.append(f"{sid}: unreadable shared session: {exc}")
    return errors


def validate_boundaries(workspace: Path) -> list[str]:
    errors: list[str] = []
    shared = workspace / "shared"
    for path in shared.rglob("*"):
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {}
            except json.JSONDecodeError:
                payload = {}
            sources = payload.get("source_artifacts", []) if isinstance(payload, dict) else []
            if any(str(source).startswith(("model-input://", "model-input://MotionPRO")) for source in sources):
                errors.append(f"shared artifact references forbidden private path: {path}")
    for path in (workspace / "tool").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"invalid public tool syntax {path}: {exc}")
            continue
        imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        imports_display = any(
            (isinstance(node, ast.ImportFrom) and "results_display" in (node.module or ""))
            or any("results_display" in (alias.name or "") for alias in getattr(node, "names", []))
            for node in imported
        )
        if imports_display:
            errors.append(f"public tool imports results_display: {path}")
    return errors


def validate_mmvp(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"missing MMVP representation root: {root}"]
    if not (root / "artifact.json").is_file():
        errors.append(f"MMVP root missing artifact.json: {root}")
    for session in (path for path in root.glob("*/*/*") if path.is_dir()):
        files = sorted((session / "insole").glob("*.npy"))
        if not files:
            errors.append(f"MMVP session has no frames: {session.name}")
            continue
        frame_ids = session / "frame_id.npy"
        if not frame_ids.is_file():
            errors.append(f"MMVP session missing frame_id.npy: {session.name}")
        elif np.asarray(np.load(frame_ids)).dtype != np.int64 or len(np.load(frame_ids)) != len(files):
            errors.append(f"MMVP frame_id mismatch: {session.name}")
        for path in files:
            try:
                array = np.load(path)
                if array.dtype != np.float32 or array.shape != (2, 31, 11):
                    errors.append(f"bad MMVP frame: {path}")
            except Exception as exc:
                errors.append(f"unreadable MMVP frame {path}: {exc}")
    return errors


def validate_human_masks(root: Path, expected_sessions: int | None = None) -> list[str]:
    errors: list[str] = []
    sessions = [path for path in root.glob("*/*/*") if path.is_dir()]
    if expected_sessions is not None and len(sessions) != expected_sessions:
        errors.append(f"human-mask session count: expected {expected_sessions}, got {len(sessions)}")
    for session in sessions:
        path = session / "bbox.npz"
        if not path.is_file():
            errors.append(f"human-mask bbox missing: {session}")
            continue
        try:
            with np.load(path) as data:
                n = len(data["frame_id"])
                if not np.array_equal(data["frame_id"], np.arange(n, dtype=np.int64)):
                    errors.append(f"human-mask frame ids are not canonical: {session}")
                if data["bbox_xyxy_norm"].shape != (n, 4):
                    errors.append(f"human-mask bbox shape mismatch: {session}")
                if np.asarray(data["source_mode"]).reshape(-1)[0] != "direct_csv_bbox":
                    errors.append(f"human-mask source mode mismatch: {session}")
        except Exception as exc:
            errors.append(f"unreadable human-mask bbox {path}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--mmvp", type=Path, default=None)
    parser.add_argument("--human-masks", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sessions = [path for path in args.facts.glob("cam*/*/*/*") if path.is_dir()]
    errors = [error for session in sessions for error in validate_session(session)]
    errors.extend(validate_boundaries(args.workspace))
    if args.mmvp:
        errors.extend(validate_mmvp(args.mmvp))
    if args.human_masks:
        errors.extend(validate_human_masks(args.human_masks, expected_sessions=140))
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print(f"shared ok: {len(sessions)} sessions; quality flags disjoint; boundaries clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
