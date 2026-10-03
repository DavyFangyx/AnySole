#!/usr/bin/env python3
"""Export FPP-Net V2T session archives (native press2Cont semantics).

Reads the FPP-Net work predictions (one sidecar per temporal-5 center
frame, each carrying its shared frame id) and writes the standard V2T
archive:

    results/baselines/FPP-Net/predictions/v2t/<session>.npz
    ├── pressure_pred    float32[n,31,22]   (zero-padded outside available)
    ├── pressure_gt      float32[n,31,22]
    ├── contact_smpl_pred float16[n,2,96]   network contact head (continuous)
    ├── contact_smpl_gt  float16[n,2,96]    native binary vertex contact GT
    ├── contact_pred/gt  uint8[n,2]         diagnostic mean-threshold flags
    ├── valid_mask       bool[n]            available & shared-valid
    └── provenance fields (contact_gt_source, contact_gt_threshold,
        pixel_weight_revision, ...)

The retired f6_soft broadcast GT is gone: ``contact_gt_source`` is now the
native press2Cont binary vertex contact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
RESULTS = REPO_ROOT / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

FPP_PREDICTIONS = WORKSPACE / "work/VP-MoCap/v1/fpp_predictions"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
OUTPUT_ROOT = RESULTS / "baselines/FPP-Net/predictions/v2t"

# E3-A: the archive carries the contact-GT binarisation threshold and the
# pixel-weight revision, so the "silent drift" between the on-disk GT and the
# documented recipe cannot recur (04_E §4.4 step 3).  The revision id names
# the build generation of the per-date sub_info.npy weights (R3-cleaned raw
# insole + absolute-nearest-neighbour resampling, 2026-10-02); the adapter's
# build_metadata report carries no separate revision identifier.
CONTACT_GT_THRESHOLD = 0.5
PIXEL_WEIGHT_REVISION = "R3_cleaned_20261002"


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def shared_valid(row: dict) -> np.ndarray:
    date, subject, sid = session_parts(row)
    frames = np.load(FACTS_ROOT / date / subject / sid / "frames.npz")
    valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
    return valid & ~fake


def export_session(row: dict, force: bool) -> str:
    sid = row["session_id"]
    date, subject, _ = session_parts(row)
    n = int(row["n_frames"])
    output = OUTPUT_ROOT / f"{sid}.npz"
    if output.is_file() and not force:
        return f"skip existing {output}"
    source = FPP_PREDICTIONS / date / subject / sid / "pred_contact_smpl"
    if not source.is_dir():
        raise FileNotFoundError(f"FPP-Net predictions missing for {sid}: {source}")

    pressure_pred = np.zeros((n, 31, 22), dtype=np.float32)
    pressure_gt = np.zeros_like(pressure_pred)
    contact_pred = np.zeros((n, 2), dtype=np.uint8)
    contact_gt = np.zeros_like(contact_pred)
    contact_smpl_pred = np.zeros((n, 2, 96), dtype=np.float16)
    contact_smpl_gt = np.zeros((n, 2, 96), dtype=np.float16)
    available = np.zeros(n, dtype=bool)
    for frame in range(n):
        path = source / f"{frame:06d}.npy"
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True).item()
        if int(payload.get("frame_id", -1)) != frame:
            raise ValueError(f"{path}: stored frame id != filename frame")
        p_pred = np.asarray(payload["pressure"]["pred"], dtype=np.float32)
        p_gt = np.asarray(payload["pressure"]["gt"], dtype=np.float32)
        if p_pred.shape != (31, 22) or p_gt.shape != (31, 22):
            raise ValueError(f"{path}: expected (31,22) pressure maps")
        cp = np.asarray(payload["contact_smpl"]["pred"], dtype=np.float32).reshape(2, -1)
        cg = np.asarray(payload["contact_smpl"]["gt"], dtype=np.float32).reshape(2, -1)
        if cp.shape != (2, 96) or cg.shape != (2, 96):
            raise ValueError(f"{path}: expected (2,96) vertex contact")
        pressure_pred[frame] = p_pred
        pressure_gt[frame] = p_gt
        contact_smpl_pred[frame] = cp
        contact_smpl_gt[frame] = cg
        contact_pred[frame] = cp.mean(axis=1) > 0.5
        contact_gt[frame] = cg.mean(axis=1) > 0.5
        available[frame] = True
    valid = shared_valid(row) & available
    if not valid.any():
        raise FileNotFoundError(f"no FPP-Net V2T frames for {sid}: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pressure_pred=pressure_pred,
        pressure_gt=pressure_gt,
        contact_smpl_pred=contact_smpl_pred,
        contact_smpl_gt=contact_smpl_gt,
        contact_pred=contact_pred,
        contact_gt=contact_gt,
        valid_mask=valid,
        frame_indices=np.arange(n, dtype=np.int64),
        target_fps=np.asarray(float(row.get("target_fps") or 40.0), dtype=np.float32),
        mode=np.asarray("V2T"),
        pressure_source_grid=np.asarray("31x11_per_foot"),
        comparison_pressure_grid=np.asarray("31x11_per_foot"),
        source_native_output=np.asarray(str(source)),
        provenance_source_type=np.asarray("model_prediction"),
        contact_gt_source=np.asarray("press2Cont_binary_vertex"),
        contact_gt_threshold=np.asarray(CONTACT_GT_THRESHOLD, dtype=np.float32),
        pixel_weight_revision=np.asarray(PIXEL_WEIGHT_REVISION),
        formal_contact_metrics=np.asarray("contact_smpl_mse/contact_smpl_bce"),
    )
    return f"wrote {output} V2T frames={int(valid.sum())}/{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_manifest()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    for sid in sessions:
        print(export_session(rows[sid], args.force))


if __name__ == "__main__":
    main()
