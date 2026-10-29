"""Bounded camera-frame evaluation against the existing GT-prime corpus."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .io import atomic_json, load_result


class EvaluationError(RuntimeError):
    """Raised when a prediction cannot be aligned to a GT-prime camera unit."""


def evaluate_camera_unit(
    manifest_unit: Mapping[str, Any],
    result_path: str | Path,
    output_path: str | Path,
    *,
    workspace_root: str | Path,
    smpl_model_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Compute local/camera metrics after physical-time alignment.

    GVHMR's `smpl_params_incam` is compared in its estimated pinhole-camera
    frame to GT-prime SMPL geometry transformed through the dated mocap and
    camera calibration. Global/world metrics are deliberately not computed:
    GVHMR's gravity-view gauge and the dataset checkerboard gauge have not
    been explicitly aligned.
    """

    required = {"date", "method", "session", "camera", "frames"}
    if not required.issubset(manifest_unit) or manifest_unit.get("input_kind") != "annotated_rgb_image_sequence":
        raise EvaluationError("GT-prime evaluation requires a dated annotated RGB image-sequence unit")

    # Import evaluation primitives only when this adapter is used, preserving
    # the separate Tactile/GVHMR environment boundary during RGB inference.
    from smpl_evaluation.alignment import bracket_times, interpolate_axis_angle, interpolate_linear
    from smpl_evaluation.coordinates import checkerboard_to_camera, gt_smpl_to_checkerboard
    from smpl_evaluation.io import Unit, gt_inventory, load_gt, repaired_frames, unit_mocap_times
    from smpl_evaluation.metrics import (
        mean_point_error,
        pa_mpjpe,
        pelvis_align,
        rotation_error_degrees,
        shape_vertex_std,
        temporal_metrics,
    )
    from smpl_evaluation.model import SMPLForward

    root = Path(workspace_root).expanduser().resolve()
    dataset_root = root / "1_Dataset"
    mocap_root = root / "Mocap"
    date = str(manifest_unit["date"])
    method = str(manifest_unit["method"])
    session = str(manifest_unit["session"])
    camera = str(manifest_unit["camera"])
    try:
        camera_index = int(camera.removeprefix("camera_"))
    except ValueError as exc:
        raise EvaluationError(f"invalid camera id {camera!r}") from exc

    full_frames = manifest_unit.get("alignment_frames") or manifest_unit["frames"]
    if not isinstance(full_frames, list) or len(full_frames) < 2:
        raise EvaluationError("full alignment_frames are required for CSV endpoint mapping")
    frame_numbers = np.asarray([frame["frame"] for frame in full_frames], dtype=np.int64)
    physical_times = np.asarray([frame["timestamp_seconds"] for frame in full_frames], dtype=np.float64)
    if np.any(np.diff(frame_numbers) <= 0):
        raise EvaluationError("alignment frame numbers must be ordered")
    full_unit = Unit(
        unit_id=str(manifest_unit["unit_id"]),
        date=date,
        method=method,
        session=session,
        camera=camera,
        camera_index=camera_index,
        frame_numbers=frame_numbers,
        physical_times=physical_times,
    )

    arrays = load_result(result_path)
    positions = {int(number): index for index, number in enumerate(frame_numbers)}
    try:
        selected = np.asarray([positions[int(number)] for number in arrays["source_frame_numbers"]], dtype=np.int64)
    except KeyError as exc:
        raise EvaluationError(f"prediction frame {exc.args[0]} is absent from full RGB alignment unit") from exc

    gt_paths = gt_inventory(mocap_root)
    gt_path = gt_paths.get((date, session))
    if gt_path is None:
        raise EvaluationError(f"no GT-prime sequence for {date}/{session}")
    calibration_path = dataset_root / date / f"{date}.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        camera_calibration = calibration["cameras"][f"cam{camera_index}"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise EvaluationError(f"cannot load camera calibration {calibration_path}: {exc}") from exc

    target_times_full = unit_mocap_times(full_unit, dataset_root)
    target_times = target_times_full[selected]
    gt = load_gt(gt_path)
    brackets = bracket_times(gt["source_frame_times_s"], target_times)
    if not brackets.valid.all():
        raise EvaluationError(
            f"{int((~brackets.valid).sum())} prediction timestamps fall outside GT-prime coverage"
        )
    gt_pose = interpolate_axis_angle(gt["poses"].reshape(-1, 24, 3), brackets).reshape(-1, 72)
    gt_translation = interpolate_linear(gt["trans"], brackets)
    gt_betas = np.broadcast_to(gt["betas"], (target_times.size, 10)).copy()

    model = SMPLForward(smpl_model_path, device=device, batch_size=64)
    pred_vertices, pred_joints = model.forward(arrays["pose"], arrays["betas"], arrays["trans"])
    gt_vertices, gt_joints = model.forward(gt_pose, gt_betas, gt_translation)
    gt_vertices_world = gt_smpl_to_checkerboard(gt_vertices, calibration)
    gt_joints_world = gt_smpl_to_checkerboard(gt_joints, calibration)
    target_vertices = checkerboard_to_camera(gt_vertices_world, camera_calibration)
    target_joints = checkerboard_to_camera(gt_joints_world, camera_calibration)

    pred_local_joints, pred_local_vertices = pelvis_align(pred_joints, pred_vertices)
    gt_local_joints, gt_local_vertices = pelvis_align(target_joints, target_vertices)
    assert pred_local_vertices is not None and gt_local_vertices is not None
    metrics: dict[str, float | int | None] = {
        "mpjpe_mm": float(mean_point_error(pred_local_joints, gt_local_joints, scale=1000.0).mean()),
        "pa_mpjpe_mm": float(pa_mpjpe(pred_joints, target_joints).mean()),
        "pve_mm": float(mean_point_error(pred_local_vertices, gt_local_vertices, scale=1000.0).mean()),
        "mpjae_deg": float(
            rotation_error_degrees(
                arrays["pose"][:, 3:66].reshape(-1, 21, 3),
                gt_pose[:, 3:66].reshape(-1, 21, 3),
            ).mean()
        ),
    }
    pred_tverts, pred_tjoints = model.tpose(arrays["betas"])
    target_tverts_single, target_tjoints_single = model.tpose(gt["betas"])
    pred_tjoints_local, pred_tverts_local = pelvis_align(pred_tjoints, pred_tverts)
    target_tjoints_local, target_tverts_local = pelvis_align(
        np.broadcast_to(target_tjoints_single[:1], pred_tjoints.shape),
        np.broadcast_to(target_tverts_single[:1], pred_tverts.shape),
    )
    assert pred_tverts_local is not None and target_tverts_local is not None
    metrics["pve_t_mm"] = float(
        mean_point_error(pred_tverts_local, target_tverts_local, scale=1000.0).mean()
    )
    metrics["shape_vertex_std_mm"] = shape_vertex_std(pred_tverts_local)

    repaired = repaired_frames(gt_path)
    repair_mask = np.asarray(
        [int(lo) in repaired or int(hi) in repaired for lo, hi in zip(brackets.lower, brackets.upper)],
        dtype=bool,
    )
    pred_temp, _ = pelvis_align(pred_joints)
    gt_temp, _ = pelvis_align(target_joints)
    metrics.update(temporal_metrics(pred_temp, gt_temp, target_times))

    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    temporal_samples = int(metrics.get("temporal_unique_frames", 0))
    temporal_status = "computed" if temporal_samples >= 3 else "not_enough_unique_timestamps"
    report = {
        "schema_id": "tactile.gvhmr-baseline.camera-evaluation",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "unit_id": str(manifest_unit["unit_id"]),
        "test_fixture_only": bool(manifest_unit.get("test_fixture_only", False)),
        "baseline": "GVHMR_SIGGRAPH_Asia_2024",
        "frame_count": int(target_times.size),
        "gt_prime": str(gt_path),
        "calibration": str(calibration_path),
        "smpl_model": str(Path(smpl_model_path).resolve()),
        "smpl_model_sha256": model.model_sha256,
        "camera_frame_metrics": metrics,
        "temporal_metric_status": temporal_status,
        "repair_masked_frame_count": 0,
        "repaired_reference_frames": int(repair_mask.sum()),
        "world_metrics": {
            "status": "not_computed",
            "reason": "GVHMR gravity-view global output gauge/axes have not been aligned to checkerboard world",
        },
        "coordinate_semantics": {
            "prediction": "GVHMR estimated pinhole-camera SMPL fit; metres; SMPL local y-up geometry with model camera convention",
            "ground_truth": "GT-prime SMPL transformed by mocap-to-checkerboard and current camera extrinsics; metres",
            "comparison": "camera-frame geometry; pelvis-relative pose/shape metrics; absolute/world metrics omitted",
            "intrinsics": "GVHMR estimated focal length; recorded in result; current calibration used for GT coordinate transform",
            "camera_axis_validation": "camera-frame interpretation follows the pinned GVHMR model convention; not a checkerboard-world alignment",
        },
    }
    report = json_safe(report)
    atomic_json(output_path, report)
    return report


__all__ = ["EvaluationError", "evaluate_camera_unit"]
