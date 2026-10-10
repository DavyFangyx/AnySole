#!/usr/bin/env python3
"""Reevaluate local WHAM/SAM3DB/GVHMR predictions with AnySole formulas.

Run from AnySole, in the Tactile environment:
  python AnysoleWorkspace/tool/eval_rgb_smpl.py --workers 7
  python AnysoleWorkspace/tool/eval_rgb_smpl.py --workers 2 --limit 8 --output <pilot>

Uses locally adapted I/O, alignment, calibration and SMPL helpers under
anysole/data/rgb_smpl; importing or reading 2_Code source is unnecessary.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anysole.utils.vision_smpl_metrics import (  # noqa: E402
    REQUESTED_METRICS, evaluate_motion,
)

BASELINES = ("WHAM", "SAM3DB", "GVHMR")
VERSIONS = ("anysole_aligned",)
REFERENCE = "2_code_aligned"


def helpers(workspace: Path | None = None):
    return tuple(importlib.import_module(f"anysole.data.rgb_smpl.{part}")
                 for part in ("io", "alignment", "coordinates", "model"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def target_geometry(workspace, unit, gt_path, model, calibration, io, alignment, coords):
    source = io.load_gt(gt_path)
    times = io.unit_mocap_times(unit, workspace / "1_Dataset")
    brackets = alignment.bracket_times(source["source_frame_times_s"], times)
    if not brackets.valid.all():
        raise ValueError(f"GT timestamps lack brackets: {unit.unit_id}")
    poses = alignment.interpolate_axis_angle(source["poses"].reshape(-1, 24, 3), brackets).reshape(-1, 72)
    trans = alignment.interpolate_linear(source["trans"], brackets)
    vertices, joints = model.forward(poses, source["betas"], trans)
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    rotations[:, 0] = coords.gt_root_rotation_to_checkerboard(rotations[:, 0], calibration)
    repaired = io.repaired_frames(gt_path)
    safe = np.asarray([int(lo) not in repaired and int(hi) not in repaired
                       for lo, hi in zip(brackets.lower, brackets.upper)])
    tverts, tjoints = model.tpose(source["betas"])
    return {"tpose_joints": tjoints, "tpose_vertices": tverts,
            "joints": coords.gt_smpl_to_checkerboard(joints, calibration),
            "vertices": coords.gt_smpl_to_checkerboard(vertices, calibration),
            "rotations": rotations, "poses": poses}, times, safe


def predicted_geometry(workspace, baseline, unit, model, camera, io, coords):
    path = workspace / "3_Result" / f"{baseline}_baseline" / unit.short_id
    if baseline == "WHAM":
        raw = io.load_wham(path / "result.pkl", unit)
        vertices, joints = model.forward(raw["pose"], raw["betas"], raw["trans"])
        replay = float(np.sqrt(np.mean((vertices - raw["verts"]) ** 2)) * 1000)
        # Historical accepted corpus: 0.088--0.104 mm RMS (WHAM LBS vs
        # smplx LBS). Require replay below 0.2 mm, not exact bit identity.
        if not np.isfinite(replay) or replay > 0.2:
            raise ValueError(f"WHAM vertex replay RMS {replay} mm")
        # Native protocol uses FK joint centres, not J_regressor @ posed mesh.
        poses = raw["pose"]
    elif baseline == "SAM3DB":
        raw = io.load_sam(path / "result.pkl", unit)
        poses = np.zeros((unit.frame_numbers.size, 72), dtype=np.float32)
        poses[:, :3], poses[:, 3:66] = raw["global_orient"], raw["body_pose"]
        vertices, joints = model.forward(poses, raw["betas"])
        rotation = raw["similarity_rotation"].astype(np.float64)
        scale = raw["similarity_scale"].astype(np.float64)
        translation = raw["similarity_translation"].astype(np.float64) + raw["camera_translation"].astype(np.float64)
        vertices = scale[:, None, None] * np.einsum("tvi,tij->tvj", vertices, rotation) + translation[:, None]
        joints = scale[:, None, None] * np.einsum("tji,tik->tjk", joints, rotation) + translation[:, None]
        replay = 0.0
    else:
        with np.load(path / "result.npz", allow_pickle=False) as archive:
            raw = {key: np.asarray(archive[key]) for key in
                   ("pose", "betas", "trans", "source_frame_numbers", "physical_times_s", "joints_camera")}
        if not np.array_equal(raw["source_frame_numbers"], unit.frame_numbers):
            raise ValueError("GVHMR source frame mismatch")
        if not np.array_equal(raw["physical_times_s"], unit.physical_times):
            raise ValueError("GVHMR timestamp mismatch")
        poses = raw["pose"]
        vertices, joints = model.forward(poses, raw["betas"], raw["trans"])
        replay = float(np.sqrt(np.mean((joints - raw["joints_camera"]) ** 2)) * 1000)
        if not np.isfinite(replay) or replay > 0.001:
            raise ValueError(f"GVHMR joint replay RMS {replay} mm")
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    if baseline == "SAM3DB":
        rotations[:, 0] = np.einsum("tji,tjk->tik", rotation, rotations[:, 0])
    rotations[:, 0] = coords.camera_root_rotation_to_checkerboard(rotations[:, 0], camera)
    native = {"joints": coords.camera_to_checkerboard(joints, camera),
              "vertices": coords.camera_to_checkerboard(vertices, camera),
              "rotations": rotations, "poses": poses}
    tverts, tjoints = model.tpose(raw["betas"])
    native.update(tpose_joints=tjoints, tpose_vertices=tverts)
    return native, replay


def shard(workspace: Path, units, index: int, batch_size: int, device: str, checkpoint: Path):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    io, alignment, coords, model_module = helpers(workspace)
    model = model_module.SMPLForward(workspace / "models/smpl/SMPL_NEUTRAL.pkl",
                                    device=device, batch_size=batch_size)
    inventory = io.gt_inventory(workspace / "Mocap")
    calibrations, rows, failures = {}, [], []
    for count, unit in enumerate(units, 1):
        try:
            if unit.date not in calibrations:
                calibrations[unit.date] = json.loads((workspace / "1_Dataset" / unit.date / f"{unit.date}.json").read_text())
            calibration = calibrations[unit.date]
            camera = calibration["cameras"][f"cam{unit.camera_index}"]
            target, times, safe = target_geometry(workspace, unit, inventory[(unit.date, unit.session)],
                                                 model, calibration, io, alignment, coords)
            for baseline in BASELINES:
                native, replay = predicted_geometry(workspace, baseline, unit, model, camera, io, coords)
                measured = evaluate_motion(native, target, times)
                if any(not np.isfinite(value) for value in measured.values()):
                    raise ValueError(f"nonfinite metric {baseline}/{VERSIONS[0]}")
                rows.append({"baseline": baseline, "metric_version": VERSIONS[0],
                             "unit_id": unit.unit_id, "short_id": unit.short_id,
                             "date": unit.date, "session": unit.session,
                             "method": unit.method, "camera": unit.camera,
                             "frame_count": int(unit.frame_numbers.size),
                             "repaired_reference_frames": int((~safe).sum()),
                             "geometry_replay_rms_mm": replay, **measured})
                del native
        except Exception as exc:
            failures.append({"unit_id": unit.unit_id, "error": f"{type(exc).__name__}: {exc}"})
        if count % 25 == 0 or count == len(units):
            write_json(checkpoint, {"worker": index, "processed_units": count, "total_units": len(units),
                                    "rows": len(rows), "failures": failures})
            print(json.dumps({"worker": index, "units": count, "total": len(units),
                              "rows": len(rows), "failures": len(failures)}), flush=True)
    forbidden = [name for name in sys.modules if name in
                 ("smpl_evaluation.metrics", "smpl_evaluation.runner", "gvhmr_baseline.evaluation")
                 or name == "_anysole_tactile_helpers.metrics"]
    if forbidden:
        raise RuntimeError(f"forbidden metric modules loaded: {forbidden}")
    return rows, failures


def aggregate(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["date"], row["session"])].append(row)
    sessions = [{"baseline": rows[0]["baseline"], "date": date, "session": session,
                 "unit_count": len(group), **{key: float(np.mean([row[key] for row in group])) for key in REQUESTED_METRICS}}
                for (date, session), group in sorted(groups.items())]
    summaries = []
    for stratum, field in (("overall", None), ("date", "date"), ("method", "method"), ("camera", "camera")):
        values = [None] if field is None else sorted({row[field] for row in rows})
        for value in values:
            subset = rows if field is None else [row for row in rows if row[field] == value]
            subgroup = defaultdict(list)
            for row in subset:
                subgroup[(row["date"], row["session"])].append(row)
            summary = {"stratum": stratum, "baseline": rows[0]["baseline"],
                       "session_count": len(subgroup), "unit_count": len(subset),
                       "frame_count": sum(row["frame_count"] for row in subset)}
            if field:
                summary[field] = value
            for metric in REQUESTED_METRICS:
                summary[metric] = float(np.mean([
                    np.mean([row[metric] for row in group]) for group in subgroup.values()]))
            summaries.append(summary)
    return sessions, summaries


def run(args):
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    io, _, _, _ = helpers(workspace)
    inventory = io.gt_inventory(workspace / "Mocap")
    manifest = workspace / "3_Result/SAM3DB_baseline/.run/manifest.jsonl"
    units = sorted(io.matched_units(manifest, set(inventory)), key=lambda unit: unit.unit_id)
    historical = workspace / "3_Result/SMPL_evaluation"
    if (len(units) != 2056 or len({(u.date, u.session) for u in units}) != 257
            or sum(u.frame_numbers.size for u in units) != 785088):
        raise ValueError("matched corpus changed")
    if args.limit:
        # Evenly cover dates/methods/cameras instead of taking adjacent units.
        units = [units[int(index)] for index in np.linspace(0, len(units) - 1, args.limit)]
    if output != historical and historical not in output.parents:
        if not args.limit:
            raise ValueError("full-run output must be below 3_Result/SMPL_evaluation")
    targets = [output / baseline / version for baseline in BASELINES for version in VERSIONS]
    if any(path.exists() for path in targets):
        raise FileExistsError("version destination exists; choose a new output root")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f".rgb_smpl_run_{os.getpid()}"
    run_dir.mkdir()
    source_paths = [REPO / "anysole/utils/metrics.py", REPO / "anysole/utils/vision_smpl_metrics.py", Path(__file__)]
    source_paths += [REPO / "anysole/data/rgb_smpl" / f"{part}.py" for part in
                     ("io", "alignment", "coordinates", "model")]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "anysole_base_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                  "anysole_branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
                  "source_sha256": {str(path): digest(path) for path in source_paths},
                  "manifest_sha256": digest(manifest), "model_sha256": digest(workspace / "models/smpl/SMPL_NEUTRAL.pkl"),
                  "gt_sha256": {str(path): digest(path) for path in inventory.values()},
                  "calibration_sha256": {date: digest(workspace / "1_Dataset" / date / f"{date}.json") for date in {u.date for u in units}},
                  "command": sys.argv, "workers": args.workers, "batch_size": args.batch_size,
                  "units": len(units), "sessions": len({(u.date, u.session) for u in units}),
                  "frames_per_baseline": sum(u.frame_numbers.size for u in units),
                  "aggregation": "unit means -> 8 equal method-camera units per physical session -> session macro mean",
                  "sampling": "matched RGB frames; CSV-anchored actual mocap times, SO(3)/translation GT interpolation; no resampling to 40Hz",
                  "native_protocol": "native SMPL24 FK; body21 local rotations; pelvis-local temporal all paired frames; own beta PVE/PVE-T; joint feet",
                  "forbidden_metric_imports": [], "status": "running"}
    write_json(run_dir / "run.json", provenance)
    rows, failures = [], []
    if args.workers == 1:
        rows, failures = shard(workspace, units, 0, args.batch_size, args.device, run_dir / "worker_0.json")
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=helpers, initargs=(workspace,)) as pool:
            futures = [pool.submit(shard, workspace, units[i::args.workers], i, args.batch_size,
                                   f"cuda:{i}", run_dir / f"worker_{i}.json") for i in range(args.workers)]
            for future in as_completed(futures):
                part, failed = future.result()
                rows.extend(part)
                failures.extend(failed)
    write_json(run_dir / "failures.json", failures)
    if failures or len(rows) != len(units) * len(BASELINES) * len(VERSIONS):
        raise RuntimeError(f"evaluation incomplete: {len(rows)} rows; failures {failures[:5]}")
    write_csv(run_dir / "all_unit_metrics.csv", rows)
    for baseline in BASELINES:
        for version in VERSIONS:
            selected = sorted([row for row in rows if row["baseline"] == baseline and row["metric_version"] == version],
                              key=lambda row: row["unit_id"])
            sessions, summaries = aggregate(selected)
            destination = output / baseline / version
            destination.mkdir(parents=True)
            write_csv(destination / "unit_metrics.csv", selected)
            write_csv(destination / "session_metrics.csv", sessions)
            write_csv(destination / "aggregate_metrics.csv", summaries)
            write_json(destination / "run.json", {**provenance, "status": "completed", "baseline": baseline,
                                                  "metric_version": version, "failures": 0})
    provenance["status"] = "completed"
    write_json(run_dir / "run.json", provenance)
    print(json.dumps({"status": "completed", "rows": len(rows), "output": str(output)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--output", type=Path, default=REPO.parent / "3_Result/SMPL_evaluation")
    parser.add_argument("--workers", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0", help="single-worker device")
    parser.add_argument("--limit", type=int, default=0, help="bounded pilot only")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.limit <= 2056:
        parser.error("positive batch size; limit 0..2056")
    run(args)


if __name__ == "__main__":
    main()
