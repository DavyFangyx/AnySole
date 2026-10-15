"""Physical-time alignment and manifold interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BracketMap:
    target_times: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray
    exact: np.ndarray


def unwrap_seconds_of_day(values: np.ndarray) -> np.ndarray:
    """Unwrap a monotonically captured seconds-of-day series across midnight."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("physical timestamps must be a finite non-empty vector")
    result = source.copy()
    offset = 0.0
    for index in range(1, source.size):
        if source[index] < source[index - 1] - 12.0 * 3600.0:
            offset += 24.0 * 3600.0
        result[index] += offset
    if np.any(np.diff(result) < 0.0):
        raise ValueError("physical timestamps are non-monotone after midnight unwrapping")
    return result


def map_rgb_to_mocap(
    frame_numbers: np.ndarray,
    physical_times: np.ndarray,
    *,
    visual_start_frame: int,
    visual_end_frame: int,
    mocap_start_time: float,
    mocap_end_time: float,
) -> np.ndarray:
    """Map per-camera JPEG times onto mocap time through the two CSV anchors."""

    frames = np.asarray(frame_numbers, dtype=np.int64)
    times = unwrap_seconds_of_day(physical_times)
    if frames.ndim != 1 or frames.shape != times.shape:
        raise ValueError("frame numbers and physical timestamps must be equal vectors")
    if frames.size == 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("visual frame numbers must be strictly increasing")
    positions = {int(frame): index for index, frame in enumerate(frames)}
    try:
        start_index = positions[int(visual_start_frame)]
        end_index = positions[int(visual_end_frame)]
    except KeyError as exc:
        raise ValueError(f"CSV endpoint frame is absent from the RGB unit: {exc.args[0]}") from exc
    if start_index >= end_index:
        raise ValueError("CSV endpoint frames are reversed or degenerate")
    denominator = times[end_index] - times[start_index]
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("CSV endpoint physical-time span must be positive")
    mocap_span = float(mocap_end_time) - float(mocap_start_time)
    if not np.isfinite(mocap_span) or mocap_span <= 0.0:
        raise ValueError("CSV endpoint mocap-time span must be positive")
    return float(mocap_start_time) + (times - times[start_index]) * mocap_span / denominator


def bracket_times(source_times: np.ndarray, target_times: np.ndarray, *, atol: float = 1e-9) -> BracketMap:
    """Bracket each target by source samples without extrapolation."""

    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("source times must be a finite non-empty vector")
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("source times must be strictly increasing")
    if target.ndim != 1:
        raise ValueError("target times must be a vector")

    upper = np.searchsorted(source, target, side="left")
    lower = upper - 1
    finite = np.isfinite(target)
    valid = finite & (target >= source[0] - atol) & (target <= source[-1] + atol)

    clipped_upper = np.clip(upper, 0, source.size - 1)
    exact = valid & np.isclose(target, source[clipped_upper], rtol=0.0, atol=atol)
    lower = np.where(exact, clipped_upper, lower)
    upper = np.where(exact, clipped_upper, upper)
    valid &= (lower >= 0) & (upper < source.size)

    lower_safe = np.clip(lower, 0, source.size - 1)
    upper_safe = np.clip(upper, 0, source.size - 1)
    denominator = source[upper_safe] - source[lower_safe]
    alpha = np.zeros(target.shape, dtype=np.float64)
    interpolated = valid & ~exact
    valid &= exact | (denominator > 0.0)
    alpha[interpolated] = (
        target[interpolated] - source[lower_safe[interpolated]]
    ) / denominator[interpolated]
    alpha = np.clip(alpha, 0.0, 1.0)
    return BracketMap(target, lower_safe, upper_safe, alpha, valid, exact)


def _slerp_quaternions(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP for scipy's xyzw convention."""

    first = np.asarray(q0, dtype=np.float64)
    second = np.asarray(q1, dtype=np.float64).copy()
    weights = np.asarray(alpha, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("quaternion endpoints must have matching [...,4] shapes")
    if weights.shape != first.shape[:-1]:
        weights = np.broadcast_to(weights, first.shape[:-1])
    dots = np.sum(first * second, axis=-1)
    negative = dots < 0.0
    second[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    close = dots > 1.0 - 1e-8
    result = np.empty_like(first)
    if np.any(close):
        result[close] = (1.0 - weights[close, None]) * first[close] + weights[close, None] * second[close]
    far = ~close
    if np.any(far):
        theta = np.arccos(dots[far])
        sine = np.sin(theta)
        result[far] = (
            (np.sin((1.0 - weights[far]) * theta) / sine)[:, None] * first[far]
            + (np.sin(weights[far] * theta) / sine)[:, None] * second[far]
        )
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("SLERP produced a degenerate quaternion")
    return result / norms


def interpolate_axis_angle(values: np.ndarray, brackets: BracketMap) -> np.ndarray:
    """Interpolate an axis-angle array shaped ``[T,...,3]`` on SO(3)."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError("axis-angle source must have shape [T,...,3]")
    if source.shape[0] <= int(max(brackets.lower.max(initial=0), brackets.upper.max(initial=0))):
        raise ValueError("bracket index exceeds the pose sequence")
    if not brackets.valid.all():
        raise ValueError("cannot interpolate invalid brackets")
    flat_shape = (source.shape[0], -1, 3)
    flat = source.reshape(flat_shape)
    lo = flat[brackets.lower]
    hi = flat[brackets.upper]
    q0 = Rotation.from_rotvec(lo.reshape(-1, 3)).as_quat().reshape(*lo.shape[:-1], 4)
    q1 = Rotation.from_rotvec(hi.reshape(-1, 3)).as_quat().reshape(*hi.shape[:-1], 4)
    weights = np.broadcast_to(brackets.alpha[:, None], lo.shape[:-1])
    interpolated = _slerp_quaternions(q0.reshape(-1, 4), q1.reshape(-1, 4), weights.reshape(-1))
    rotations = Rotation.from_quat(interpolated).as_rotvec().reshape(lo.shape)
    return rotations.reshape((brackets.target_times.size,) + source.shape[1:])


def interpolate_linear(values: np.ndarray, brackets: BracketMap) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape[0] <= int(max(brackets.lower.max(initial=0), brackets.upper.max(initial=0))):
        raise ValueError("bracket index exceeds the sequence")
    if not brackets.valid.all():
        raise ValueError("cannot interpolate invalid brackets")
    weight_shape = (brackets.alpha.size,) + (1,) * (source.ndim - 1)
    alpha = brackets.alpha.reshape(weight_shape)
    return (1.0 - alpha) * source[brackets.lower] + alpha * source[brackets.upper]
