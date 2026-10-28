#!/usr/bin/env python3
"""Frozen B1 evaluation: native RGB windows, one fused teacher stream.

The training evaluator remains unchanged. This adapter reuses its strict model
loader, SO(3) overlap reducer and once-per-segment velocity integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
for search in (REPO, WORKSPACE / "2_Code"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from anysole.data.dataset import grouped_collate
from anysole.data.workspace_adapter import sha256_file
from anysole.utils import eval_protocol as protocol
from anysole.utils.b1_training import predictions_to_world
from anysole.utils.geometry import rot6d_to_rotmat

SCHEMA = "anysole.b1.native-evaluation.v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def inside_workspace(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("evaluation output must remain in the workspace")
    return path


def collect_batch(model, batch, device, *, teacher=False):
    """Forward receives only RGB, observed-valid masks, time and geometry.

    GT stays on CPU. Teacher geometry is static calibration, never root/shape
    truth. Padding and source gaps do not acquire predictions through biases.
    """
    cpu_visual, labels, masks, meta = (batch[k] for k in ("visual", "labels", "masks", "meta"))
    visual = {name: value.to(device) if torch.is_tensor(value) else value
              for name, value in cpu_visual.items()}
    cameras = array(meta["camera_id"])
    if teacher:
        if any(not len(row[row >= 0]) or tuple(row[row >= 0]) != tuple(sorted(set(row[row >= 0]))) or
               not set(row[row >= 0]).issubset({1, 2, 4}) for row in cameras):
            raise ValueError("teacher inputs must be ordered subsets of cameras 1/2/4 only")
        out = model.forward_multiview(visual["V_feat"], visual["view_valid"],
                                     visual["target_time_s"], visual["camera_geometry"])
        selected = out["selected_view_valid"]
        valid = out["frame_valid"]
        if not torch.equal(valid, selected.any(1)) or (selected & ~visual["view_valid"]).any():
            raise ValueError("teacher fused validity disagrees with actual selected inputs")
        out = predictions_to_world(out, visual["reference_R_cw"])
        edges = (selected[:, :, :-1] & selected[:, :, 1:] & visual["view_edges"]).any(1)
        selected_np = array(selected)
        active = [(g, 0, 1, g) for g in range(len(cameras))]
    else:
        if any(len(row[row >= 0]) != 1 for row in cameras):
            raise ValueError("student evaluation requires exactly one camera per RGB sample")
        g, c = visual["view_valid"].any(-1).nonzero(as_tuple=True)
        active_map = {int(row): flat for flat, row in enumerate(array(g))}
        if len(g):
            out = model.forward_visual(visual["V_feat"][g, c], visual["view_valid"][g, c],
                                       visual["target_time_s"][g])
            out = predictions_to_world(out, visual["camera_R_cw"][g, c])
        else:
            out = None
        active = [(row, 0, int(camera[0]), active_map.get(row)) for row, camera in enumerate(cameras)]
    rotation = None if out is None else array(rot6d_to_rotmat(out["x0_hat"].float().reshape(-1, visual["target_time_s"].shape[1], 24, 6)))
    result = []
    for g, c, camera, flat in active:
        length = visual["target_time_s"].shape[1]
        frame_valid = array(valid[g] if teacher else visual["view_valid"][g, c]).astype(bool)
        item = {"recording_id": meta["recording_id"][g], "camera_id": camera,
                **{name: meta.get(name, ["unknown"] * len(cameras))[g] for name in ("subject", "action", "date")},
                "frame_id": array(meta["target_frame_id"][g]),
                "time_s": array(cpu_visual["target_time_s"][g]), "valid": frame_valid,
                "view_edges": array(edges[g] if teacher else visual["view_edges"][g, c]),
                **{name: array(masks[name][g]) for name in ("motion_valid", "pressure_valid", "motion_edges", "pressure_edges")},
                **{"gt_" + dst: array(labels[src][g]) for src, dst in
                   (("kp_gt", "joints"), ("pressure", "pressure"), ("pose6d", "pose6d"), ("root", "root"), ("betas", "betas"))}}
        for name in ("source_frame_id", "source_time_s", "reuse", "time_residual_ms"):
            if name in meta:
                item[name] = array(meta[name][g, c])
        if teacher:
            # Dynamic continuity follows the actual contributing camera set.
            # The reference axis is a native cam1 timeline, independent of
            # whether cam1 itself contributes an observed feature at a frame.
            item.update(source_frame_id=item["frame_id"].copy(), source_time_s=item["time_s"].copy(),
                        reuse=np.zeros(length, bool), time_residual_ms=np.zeros(length),
                        teacher_eligible=selected_np[g].sum(0) >= 2,
                        available_allowed_view_count=selected_np[g].sum(0))
        if flat is None:
            item.update(rotation=np.zeros((length, 24, 3, 3)), pressure=np.zeros((length, 96)), velocity=np.zeros((length, 3)))
        else:
            item.update(rotation=rotation[flat], pressure=array(out["pressure_hat"][flat].float()),
                        velocity=array(out["v_hat"][flat].float()))
        result.append(item)
    return result


def collect_recording(model, dataset, recording_id, device, *, teacher=False, batch_size=32):
    """Retain a full native axis, including frames without RGB or labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    original_windows = dataset.windows
    selected = [w for w in original_windows if dataset.records[w[0]]["recording_id"] == recording_id]
    from anysole.train import capture_training_rng, restore_training_rng
    state, was_training = capture_training_rng(device), model.training
    windows = []
    model.eval()
    try:
        dataset.windows = selected
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                batch = grouped_collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
                windows.extend(collect_batch(model, batch, device, teacher=teacher))
        windows.extend(axis_templates(dataset, recording_id, teacher=teacher))
        series = protocol.aggregate_visual_windows(windows)
        for item in series.values():
            item["coordinate_frame"] = "world"
            record = next(r for r in dataset.records if r["recording_id"] == recording_id)
            if not np.array_equal(item["frame_id"], np.asarray(record["target_frame_id"])):
                raise ValueError("aggregated prediction axis differs from the complete native record axis")
            if "available_allowed_view_count" in record:
                item["available_allowed_view_count"] = np.asarray(record["available_allowed_view_count"])
            if teacher:
                available = np.zeros(len(item["frame_id"]), np.int64)
                lookup = {int(frame): i for i, frame in enumerate(item["frame_id"])}
                for window in windows:
                    if "available_allowed_view_count" in window:
                        for frame, count in zip(window["frame_id"], window["available_allowed_view_count"]):
                            if int(frame) >= 0:
                                available[lookup[int(frame)]] = int(count)
                item["teacher_input_view_count"] = available
                if "available_allowed_view_count" not in item:
                    item["available_allowed_view_count"] = available
                view = next(v for v in record["views"] if v["camera_id"] == 1)
                reference_valid = np.asarray(dataset._load_features(record, view)[1], bool)
                item["reference_valid"] = reference_valid
                item["reference_edges"] = (np.asarray(view["view_edges"], bool) &
                                            reference_valid[:-1] & reference_valid[1:])
        return series
    finally:
        dataset.windows = original_windows
        model.train(was_training)
        restore_training_rng(state, device)


def axis_templates(dataset, recording_id, *, teacher=False):
    result = []
    for ri, record in enumerate(dataset.records):
        if record["recording_id"] != recording_id:
            continue
        labels = dataset.load_labels(ri)
        n = len(record["target_frame_id"])
        views = record["views"][:1] if teacher else record["views"]
        for view in views:
            item = {"recording_id": recording_id, "camera_id": 1 if teacher else int(view["camera_id"]),
                    **{name: record.get(name, "unknown") for name in ("subject", "action", "date")},
                    "frame_id": record["target_frame_id"], "time_s": record["target_time_s"],
                    "valid": np.zeros(n, bool), "view_edges": np.ones(max(n - 1, 0), bool),
                    "motion_valid": np.asarray(record["motion"]["valid"], bool),
                    "pressure_valid": np.repeat(np.asarray(record["pressure"]["foot_valid"], bool), 48, axis=1),
                    "motion_edges": record["motion_edges"],
                    "pressure_edges": np.repeat(np.asarray(record["pressure_foot_edges"], bool), 48, axis=1),
                    **{"gt_" + dst: labels[src] for src, dst in
                       (("kp_gt", "joints"), ("pressure", "pressure"), ("pose6d", "pose6d"), ("root", "root"), ("betas", "betas"))}}
            for name in ("source_frame_id", "source_time_s", "reuse", "time_residual_ms"):
                item[name] = view[name]
            if teacher:
                item.update(source_frame_id=np.asarray(item["frame_id"]), source_time_s=np.asarray(item["time_s"]),
                            reuse=np.zeros(n, bool), time_residual_ms=np.zeros(n))
            result.append(item)
    return result


def export_series(series, output, config, provenance):
    """B1 native-time variant of export_visual_series; prediction arrays only."""
    from anysole.data.smpl_io import _rest_joints, smpl24_pose6d_to_poses
    output = inside_workspace(output)
    output.mkdir(parents=True, exist_ok=True)
    rest = _rest_joints({"betas": np.zeros(10)}, model_path=Path(config["smpl_model_path"]))[0]
    paths = []
    for (recording, camera), item in sorted(series.items()):
        if not re.fullmatch(r"\d{8}/S\d+|[A-Za-z0-9][A-Za-z0-9_.-]*", recording):
            raise ValueError("unsafe recording identifier")
        path = output / f"{recording.replace('/', '_')}_cam{camera}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        valid = item["valid"]
        pose = np.full((len(valid), 144), np.nan, np.float32)
        rotation = item["rotation"][valid]
        pose[valid] = np.concatenate((rotation[..., 0], rotation[..., 1]), -1).reshape(-1, 144)
        poses = np.full((len(valid), 72), np.nan, np.float32)
        if valid.any():
            poses[valid] = smpl24_pose6d_to_poses(pose[valid])
        meta = {**provenance, "protocol": SCHEMA, "recording_id": recording, "camera_id": camera,
                "pred_beta_source": "zero", "coordinate_system": "native SMPL +Y up, metres; MoCap world orientation",
                "pose_layout": "root world, joints1..23 parent-local", "absolute_world_origin": None,
                "trajectory_origin": "p0=0 independently per continuous source-valid segment",
                "velocity_convention": "v[i] ends at target i; aggregate windows before integration",
                "timebase": "native per-camera RGB timestamps; teacher target axis is native cam1",
                "duplicate_policy": "uniform_SO3_chordal_projection_pressure_velocity_mean",
                "pressure_main": "unclamped_linear", "pressure_display": "clip_0_1_for_display_only",
                "pressure_unit": "normalized_sensor_reading; uncalibrated; not Newtons",
                "smpl_trans": "pelvis_root_translation minus beta0 unrotated rest pelvis",
                "effective_config": config}
        names = ("frame_id", "time_s", "valid", "overlap_count", "rotation", "pressure", "velocity", "trajectory",
                 "segment_id", "edges", "source_frame_id", "source_time_s", "reuse", "time_residual_ms", "teacher_eligible",
                 "available_allowed_view_count", "teacher_input_view_count")
        names += ("reference_valid", "reference_edges")
        np.savez_compressed(path, **{name: item[name] for name in names if name in item},
                            pose6d=pose, poses=poses, trans=item["trajectory"] - rest,
                            betas=np.zeros(10, np.float32), pressure_display=np.clip(item["pressure"], 0, 1),
                            metadata_json=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)))
        paths.append({"recording_id": recording, "camera_id": camera, "path": str(path), "sha256": sha256_file(path)})
    return paths


def load_frozen_entry(plan_path, run_id, which):
    """Read frozen metadata before any dataset construction or test access."""
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    if plan.get("frozen") is not True or plan.get("test_access_authorized") is not True:
        raise ValueError("evaluation plan is not frozen")
    matches = [entry for entry in plan.get("runs", []) if entry.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError("entry must uniquely name a predeclared checkpoint")
    run = matches[0]
    if which not in ("best", "last") or not str(plan.get("checkpoint_roles", {}).get(which, "")).startswith("primary" if which == "best" else "diagnostic"):
        raise ValueError("only predeclared primary best or diagnostic last checkpoints are permitted")
    entry = {**run, "entry_id": f"{run_id}/{which}", "which": which,
             "checkpoint": run["checkpoints"][which], "checkpoint_sha256": run["checkpoint_hashes"][which]}
    path = Path(entry["checkpoint"])
    if entry.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError("frozen checkpoint fingerprint changed")
    return plan, entry


def verify_execution_authorization(path, plan_path, device, smpl_device):
    """Root seals actual implementation bytes before any heldout forward."""
    receipt = read_json(path)
    if (receipt.get("test_inference_authorized") is not True or
            receipt.get("evaluation_plan_sha256") != sha256_file(Path(plan_path)) or
            receipt.get("evaluator_sha256") != sha256_file(Path(__file__))):
        raise ValueError("root inference authorization or evaluator source seal mismatch")
    for current in (device, smpl_device):
        if str(current) not in receipt.get("allowed_devices", []):
            raise ValueError("execution device absent from root allocation")
    for filename, expected in receipt.get("source_files", {}).items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(f"sealed evaluation source changed: {filename}")
    return receipt


def export_scoring_inputs(series, output, provenance):
    """Explicit evaluator-only GT/support archive for independent rescoring."""
    output = inside_workspace(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for (recording, camera), item in sorted(series.items()):
        path = output / f"{recording.replace('/', '_')}_cam{camera}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        meta = {**provenance, **{name: item.get(name, "unknown") for name in
                                ("recording_id", "camera_id", "subject", "action", "date", "coordinate_frame")},
                "purpose": "evaluator-only prediction, GT and support; never a model input"}
        arrays = {name: value for name, value in item.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)))
        paths.append({"recording_id": recording, "camera_id": camera, "path": str(path), "sha256": sha256_file(path)})
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--which", choices=("best", "last"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, help="Defaults to data-dir/coverage.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smpl-device", default="cpu")
    parser.add_argument("--smpl-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--recordings", nargs="+")
    parser.add_argument("--resume", action="store_true", help="Verify and retain completed recording units with identical provenance")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.smpl_batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("batch sizes and CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    plan, entry = load_frozen_entry(args.plan, args.run_id, args.which)
    entry_id = entry["entry_id"]
    output = inside_workspace(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("evaluation output must be new or empty")
    execution = verify_execution_authorization(args.execution_authorization, args.plan, args.device, args.smpl_device)
    from anysole.data.b1_evaluation import NativeRGBEvaluationDataset, verify_test_authorization
    authorization = verify_test_authorization(args.authorization)
    if authorization["evaluation_plan_sha256"] != sha256_file(args.plan):
        raise ValueError("data authorization refers to a different evaluation plan")
    from smpl_evaluation.anysole_b1 import score_series
    from smpl_evaluation.model import SMPLForward
    model, config, stats, checkpoint = protocol.load_visual_student(entry["checkpoint"], args.device)
    if config.get("b1_version") != "anysole.b1.v1" or int(config.get("tw", -1)) != 20:
        raise ValueError("checkpoint must be a B1 T20 model")
    experiment = entry.get("experiment", re.sub(r"-s\d+$", "", entry["run_id"]))
    if config.get("b1_run_id") != experiment or int(config["seed"]) != int(entry["seed"]):
        raise ValueError("plan/checkpoint run or seed identity differs")
    teacher = config["b1_role"] == "teacher"
    preparation = args.preparation or args.data_dir / "coverage.json"
    manifests = read_json(preparation)
    names = ["teacher"] if teacher else ["cam1", "cam2", "cam3", "cam4"]
    rows, predictions, scoring_inputs, timing, expected = [], [], [], [], {}
    provenance = {"evaluation_plan_sha256": sha256_file(args.plan), "checkpoint_sha256": entry["checkpoint_sha256"],
                  "entry_id": entry_id, "which": entry["which"], "run_id": entry["run_id"], "seed": entry["seed"],
                  "teacher_quality_qualification": entry.get("teacher_quality_qualification"),
                  "student_source_hash": config.get("source_hash"), "structure_hash": config["structure_hash"],
                  "stats_sha256": stats.get("stats_sha256"), "evaluator_sha256": sha256_file(Path(__file__)),
                  "training_eval_protocol_sha256": sha256_file(Path(protocol.__file__)),
                  "training_manifest_sha256": config.get("manifest_hash"), "training_cache_index_sha256": config.get("cache_index_hash"),
                  "dataset_preparation_sha256": sha256_file(preparation),
                  "cache_index_sha256": sha256_file(Path(manifests["cache_index"])),
                  "execution_authorization_sha256": sha256_file(args.execution_authorization),
                  "implementation_sources": execution.get("source_files", {}),
                  "smpl_runtime_model_hash": sha256_file(Path(config["smpl_model_path"]))}
    if args.resume and (output / "eval.json").exists():
        prior = read_json(output / "eval.json")
        if any(prior.get(key) != value for key, value in provenance.items()):
            raise ValueError("completed report provenance differs from resume request")
        if prior.get("explicit_recording_subset") != args.recordings:
            raise ValueError("completed report recording scope differs from resume request")
        for item in prior["predictions"] + prior["scoring_inputs"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("completed report artifact fingerprint changed")
        print(json.dumps({"event": "already_complete_verified", "entry_id": entry_id}), flush=True)
        return 0
    smpl = SMPLForward(config["smpl_model_path"], device=args.smpl_device, batch_size=args.smpl_batch_size)
    for name in names:
        manifest = Path(manifests["manifests"][name])
        dataset = NativeRGBEvaluationDataset(manifest, Path(config["dataset_root"]), Path(config["feature_cache_root"]),
            Path(config["smpl_model_path"]), cache_index=Path(manifests["cache_index"]), authorization=args.authorization,
            window_length=20, stride=10)
        ids = sorted(record["recording_id"] for record in dataset.records)
        expected[name] = ids
        selected = ids if args.recordings is None else [rid for rid in ids if rid in args.recordings]
        if args.recordings is not None and not set(args.recordings).issubset(ids):
            raise ValueError("requested recording absent from frozen test membership")
        for recording in selected:
            unit_path = output / "recordings" / f"{name}_{recording.replace('/', '_')}.json"
            record = next(record for record in dataset.records if record["recording_id"] == recording)
            sources = {name: {key: record[name][key] for key in ("path", "sha256")}
                       for name in ("motion", "pressure", "calibration")}
            for source in sources.values():
                if sha256_file(Path(config["dataset_root"]) / source["path"]) != source["sha256"]:
                    raise ValueError("evaluation recording source fingerprint changed")
            unit_provenance = {**provenance, "manifest_sha256": sha256_file(manifest), "recording_sources": sources}
            if args.resume and unit_path.exists():
                prior = read_json(unit_path)
                if prior.get("provenance") != unit_provenance:
                    raise ValueError(f"resume provenance mismatch: {unit_path}")
                for prediction in prior["predictions"] + prior["scoring_inputs"]:
                    if sha256_file(Path(prediction["path"])) != prediction["sha256"]:
                        raise ValueError("resume prediction archive fingerprint changed")
                rows.extend(prior["rows"])
                predictions.extend(prior["predictions"])
                scoring_inputs.extend(prior["scoring_inputs"])
                timing.append(prior["cost"])
                continue
            started = time.perf_counter()
            series = collect_recording(model, dataset, recording, torch.device(args.device), teacher=teacher, batch_size=args.batch_size)
            forward_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            new_rows = score_series(series, smpl=smpl, batch_size=args.smpl_batch_size,
                                    frozen_stats={**stats, **{k: config[k] for k in config if k.startswith("pressure_")}})
            new_rows = [{**row, "entry_id": entry_id, "run_id": entry["run_id"], "seed": entry["seed"],
                         "which": entry["which"], "teacher_quality_qualification": entry.get("teacher_quality_qualification")} for row in new_rows]
            rows.extend(new_rows)
            new_predictions = export_series(series, output / "predictions", config, unit_provenance)
            predictions.extend(new_predictions)
            new_scoring = export_scoring_inputs(series, output / "scoring_inputs", unit_provenance)
            scoring_inputs.extend(new_scoring)
            timing.append({"manifest": name, "recording_id": recording, "cached_inference_seconds": forward_elapsed,
                           "scoring_and_export_seconds": time.perf_counter() - started})
            write_json(unit_path, {"provenance": unit_provenance, "rows": new_rows, "predictions": new_predictions,
                                   "scoring_inputs": new_scoring, "cost": timing[-1]})
            for cache_name in ("_features", "_labels"):
                getattr(dataset, cache_name, {}).clear()
            print(json.dumps({"event": "recording_complete", "entry_id": entry_id, "manifest": name, "recording": recording}), flush=True)
    report = {"schema_version": SCHEMA, **provenance, "rows": rows, "predictions": predictions,
              "scoring_inputs": scoring_inputs,
              "prediction_hash": protocol.visual_hash([p["sha256"] for p in predictions]),
              "expected_recordings": expected, "explicit_recording_subset": args.recordings,
              "cost": timing, "effective_config": config,
              "teacher_inputs": [1, 2, 4] if teacher else None,
              "student_inputs": "one independent RGB stream per camera" if not teacher else None,
              "coverage_policy": "full native RGB axis; labels never decide prediction extent; metric-specific valid support",
              "frontend_limit": "verified frozen HRNet feature cache; raw RGB frontend was not rerun"}
    write_json(output / "eval.json", report)
    write_json(output / "session_camera_rows.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
