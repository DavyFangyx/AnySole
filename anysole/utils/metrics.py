"""Canonical numerical metrics shared by AnySole and all baselines.

These definitions are migrated from the repository-level ``metrics.py``.
Keep formulas, units, and field names stable: model-specific evaluators may
add diagnostics, but must not redefine a canonical metric locally.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


MOTION_METRIC_NAMES = (
    "mpjpe_mm",
    "pa_mpjpe_mm",
    "mpjae_deg",
    "pve_mm",
    "shape_vertex_std_mm",
    "root_ate_mm",
    "root_rte_percent",
    "w_mpjpe100_mm",
    "wa_mpjpe100_mm",
    "accel_error_m_s2",
    "jitter_pred_m_s3",
    "jitter_gt_m_s3",
    "foot_sliding_mm",
)


def pelvis_align(joints: np.ndarray, vertices: np.ndarray | None = None,
                 *, pelvis_indices: tuple[int, int] = (1, 2)) -> tuple[np.ndarray, np.ndarray | None]:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape [T,J,3]")
    pelvis = joints[:, pelvis_indices].mean(axis=1, keepdims=True)
    aligned_joints = joints - pelvis
    aligned_vertices = None
    if vertices is not None:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[0] != joints.shape[0] or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [T,V,3] and match joints")
        aligned_vertices = vertices - pelvis
    return aligned_joints, aligned_vertices


def mean_point_error(predicted: np.ndarray, target: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("point arrays must have matching [T,N,3] shapes")
    return np.linalg.norm(predicted - target, axis=-1).mean(axis=-1) * scale


def similarity_align(predicted: np.ndarray, target: np.ndarray, *, fixed_scale: bool = False) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 3 or predicted.ndim < 2:
        raise ValueError("alignment inputs must have matching [...,N,3] shapes")
    original_shape = predicted.shape
    x = predicted.reshape(-1, original_shape[-2], 3)
    y = target.reshape(-1, original_shape[-2], 3)
    mx = x.mean(axis=1, keepdims=True)
    my = y.mean(axis=1, keepdims=True)
    x0 = x - mx
    y0 = y - my
    covariance = np.einsum("bni,bnj->bij", y0, x0) / x.shape[1]
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.ones((x.shape[0], 3), dtype=np.float64)
    correction[:, -1] = np.where(np.linalg.det(u) * np.linalg.det(vh) < 0.0, -1.0, 1.0)
    rotation = np.einsum("bij,bjk,bkl->bil", u, np.eye(3)[None] * correction[:, None, :], vh)
    if fixed_scale:
        scale = np.ones(x.shape[0], dtype=np.float64)
    else:
        variance = np.sum(x0 * x0, axis=(1, 2)) / x.shape[1]
        if np.any(variance <= 1e-12):
            raise ValueError("degenerate predicted point set")
        scale = np.sum(singular * correction, axis=1) / variance
    translation = my[:, 0] - scale[:, None] * np.einsum("bij,bj->bi", rotation, mx[:, 0])
    aligned = scale[:, None, None] * np.einsum("bij,bnj->bni", rotation, x) + translation[:, None]
    return aligned.reshape(original_shape)


def pa_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return mean_point_error(similarity_align(predicted, target), target, scale=1000.0)


def rotation_error_degrees(predicted_aa: np.ndarray, target_aa: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_aa, dtype=np.float64)
    target = np.asarray(target_aa, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("axis-angle arrays must have matching [...,3] shapes")
    rp = Rotation.from_rotvec(predicted.reshape(-1, 3)).as_matrix()
    rt = Rotation.from_rotvec(target.reshape(-1, 3)).as_matrix()
    relative = np.einsum("bij,bjk->bik", np.transpose(rp, (0, 2, 1)), rt)
    angles = Rotation.from_matrix(relative).magnitude()
    return np.degrees(angles).reshape(predicted.shape[:-1])


def matrix_rotation_error_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have matching [...,3,3] shapes")
    relative = np.einsum("...ji,...jk->...ik", predicted, target)
    angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return np.degrees(angles).reshape(predicted.shape[:-2])


def collapse_duplicate_times(times: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0.0):
        raise ValueError("times must be finite and non-decreasing")
    checked = [np.asarray(array, dtype=np.float64) for array in arrays]
    if any(array.shape[0] != times.size for array in checked):
        raise ValueError("time and sample lengths differ")
    unique, inverse = np.unique(times, return_inverse=True)
    outputs: list[np.ndarray] = [unique]
    counts = np.bincount(inverse).astype(np.float64)
    for array in checked:
        accumulated = np.zeros((unique.size,) + array.shape[1:], dtype=np.float64)
        np.add.at(accumulated, inverse, array)
        outputs.append(accumulated / counts.reshape((unique.size,) + (1,) * (array.ndim - 1)))
    return tuple(outputs)


def _segments(times: np.ndarray, *, gap_factor: float = 3.0) -> list[slice]:
    if times.size == 0:
        return []
    delta = np.diff(times)
    if np.any(delta <= 0.0):
        raise ValueError("derivative times must be strictly increasing")
    median = float(np.median(delta)) if delta.size else 0.0
    breaks = np.flatnonzero(delta > gap_factor * median) + 1 if median > 0.0 else np.array([], dtype=int)
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, times.size]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def _differentiate(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.diff(times)
    if np.any(delta <= 0.0):
        raise ValueError("derivative times must be strictly increasing")
    shape = (delta.size,) + (1,) * (values.ndim - 1)
    derivative = np.diff(values, axis=0) / delta.reshape(shape)
    return derivative, (times[1:] + times[:-1]) * 0.5


def temporal_metrics(predicted: np.ndarray, target: np.ndarray, times: np.ndarray,
                     *, gap_factor: float = 3.0) -> dict[str, float | int]:
    times, predicted, target = collapse_duplicate_times(times, predicted, target)
    acceleration_errors: list[np.ndarray] = []
    pred_jerks: list[np.ndarray] = []
    target_jerks: list[np.ndarray] = []
    accel_samples = jerk_samples = 0
    for segment in _segments(times, gap_factor=gap_factor):
        ts, pp, tt = times[segment], predicted[segment], target[segment]
        if ts.size < 3:
            continue
        pv, vt = _differentiate(pp, ts)
        tv, _ = _differentiate(tt, ts)
        pa, at = _differentiate(pv, vt)
        ta, _ = _differentiate(tv, vt)
        acceleration_errors.append(np.linalg.norm(pa - ta, axis=-1).mean(axis=-1))
        accel_samples += pa.shape[0]
        if ts.size >= 4:
            pj, _ = _differentiate(pa, at)
            tj, _ = _differentiate(ta, at)
            pred_jerks.append(np.linalg.norm(pj, axis=-1).mean(axis=-1))
            target_jerks.append(np.linalg.norm(tj, axis=-1).mean(axis=-1))
            jerk_samples += pj.shape[0]

    def average(parts: list[np.ndarray]) -> float:
        return float(np.concatenate(parts).mean()) if parts else float("nan")

    return {
        "accel_error_m_s2": average(acceleration_errors),
        "jitter_pred_m_s3": average(pred_jerks),
        "jitter_gt_m_s3": average(target_jerks),
        "temporal_unique_frames": int(times.size),
        "acceleration_samples": int(accel_samples),
        "jerk_samples": int(jerk_samples),
    }


def _fit_similarity(source: np.ndarray, target: np.ndarray,
                    *, fixed_scale: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    x0, y0 = x - mx, y - my
    covariance = y0.T @ x0 / x.shape[0]
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    if fixed_scale:
        scale = 1.0
    else:
        variance = float(np.sum(x0 * x0) / x.shape[0])
        if variance <= 1e-12:
            raise ValueError("degenerate source points")
        scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = my - scale * (rotation @ mx)
    return scale, rotation, translation


def _apply_fitted_similarity(source_fit: np.ndarray, target_fit: np.ndarray,
                             values: np.ndarray) -> np.ndarray:
    scale, rotation, translation = _fit_similarity(source_fit, target_fit)
    return scale * np.einsum("ij,...j->...i", rotation, values) + translation


def windowed_world_mpjpe(predicted: np.ndarray, target: np.ndarray,
                         *, window: int = 100) -> tuple[float, float]:
    first_errors: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    for start in range(0, predicted.shape[0], window):
        pred = predicted[start:start + window]
        gt = target[start:start + window]
        if pred.shape[0] < 2:
            continue
        transformed = _apply_fitted_similarity(pred[:2].reshape(-1, 3), gt[:2].reshape(-1, 3), pred)
        global_aligned = similarity_align(pred.reshape(1, -1, 3), gt.reshape(1, -1, 3))[0].reshape(pred.shape)
        first_errors.append(mean_point_error(transformed, gt, scale=1000.0))
        all_errors.append(mean_point_error(global_aligned, gt, scale=1000.0))
    if not first_errors:
        return float("nan"), float("nan")
    return float(np.concatenate(first_errors).mean()), float(np.concatenate(all_errors).mean())


def root_trajectory_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    absolute = np.linalg.norm(predicted - target, axis=-1)
    _, rotation, translation = _fit_similarity(predicted, target, fixed_scale=True)
    aligned = np.einsum("ij,tj->ti", rotation, predicted) + translation
    residual = np.linalg.norm(aligned - target, axis=-1)
    path = float(np.linalg.norm(np.diff(target, axis=0), axis=-1).sum())
    return {
        "root_ate_mm": float(absolute.mean() * 1000.0),
        "root_rte_percent": float(residual.mean() / path * 100.0) if path > 1e-9 else float("nan"),
        "root_path_length_m": path,
    }


def foot_sliding(predicted_vertices: np.ndarray, target_vertices: np.ndarray,
                 times: np.ndarray, *,
                 foot_indices: Iterable[int] = (3216, 3387, 6617, 6787),
                 contact_speed_m_s: float = 0.3) -> tuple[float, int]:
    ids = np.asarray(tuple(foot_indices), dtype=np.int64)
    times, predicted, target = collapse_duplicate_times(
        times, predicted_vertices[:, ids], target_vertices[:, ids]
    )
    if times.size < 2:
        return float("nan"), 0
    dt = np.diff(times)
    gt_displacement = np.linalg.norm(np.diff(target, axis=0), axis=-1)
    pred_displacement = np.linalg.norm(np.diff(predicted, axis=0), axis=-1)
    contact = gt_displacement / dt[:, None] < contact_speed_m_s
    count = int(contact.sum())
    return (float(pred_displacement[contact].mean() * 1000.0), count) if count else (float("nan"), 0)


def shape_vertex_std(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    center = vertices.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(vertices - center, axis=-1).mean() * 1000.0)
