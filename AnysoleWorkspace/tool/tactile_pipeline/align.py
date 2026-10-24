"""触觉行数据 → 视觉帧网格的对齐（绝对墙钟最近行语义，2026-09-27 起）。

数值源与质量源分开传入：值（清洗产物 processed48 + wall_time_day_s）按
``nearest_indices`` 直接映射；valid/fake 掩码用 ``resample_quality`` 从
管线 quality_rows 行级标记映射（含 dropout 跨段 fake 传播）。两路都落在同一
帧网格上。
"""

from __future__ import annotations

import numpy as np

def nearest_indices(times_s: np.ndarray, targets_s: np.ndarray) -> np.ndarray:
    """Index of the nearest element of ``times_s`` (ascending) for each target."""
    idx = np.clip(np.searchsorted(times_s, targets_s), 1, len(times_s) - 1)
    left = idx - 1
    return np.where(np.abs(times_s[left] - targets_s) <= np.abs(times_s[idx] - targets_s), left, idx)


def fake_gap_frame_marks(raw_rows: np.ndarray, row_fake: np.ndarray,
                         t_sub: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Frames whose nearest pressure row is a fake row, nearest-value semantics.

    The absolute nearest-row selection only sees rows with a wall-clock
    timestamp, so fake rows inside a dropout span never land on any frame.
    Each run of such rows sits between two real rows (or before the first /
    after the last real row); every frame whose wall time falls inside that
    run's span is marked fake.  No timeline reconstruction, no interpolation:
    the span is bounded by the two neighbouring real rows, and boundary frames
    stay with their real row (conservative at ±1 frame).
    """
    marks = np.zeros(len(targets), dtype=bool)
    if raw_rows.size == 0:
        return marks
    for k in range(len(raw_rows) - 1):
        lo, hi = raw_rows[k], raw_rows[k + 1]
        if hi - lo > 1 and row_fake[lo + 1:hi].any():
            marks |= (targets > t_sub[k]) & (targets < t_sub[k + 1])
    if raw_rows[0] > 0 and row_fake[:raw_rows[0]].any():
        marks |= targets < t_sub[0]
    if raw_rows[-1] < len(row_fake) - 1 and row_fake[raw_rows[-1] + 1:].any():
        marks |= targets > t_sub[-1]
    return marks


def resample_quality(left: tuple, right: tuple, n_frames: int, *,
                     video_start_day_s: float | None = None,
                     pressure_epoch_day_s: float | None = None,
                     fps: float = 40.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Map quality rows to the video frame grid; values are mapped separately
    by the caller from the cleaning pipeline's rows (``nearest_indices``).

    Pack = ``(t_us, raw_t_us, valid, fake)`` per foot (pipeline quality_rows).

    Absolute mode (default when the wall-clock anchors are available and the
    rows carry raw timestamps): video frame ``i`` has wall time
    ``video_start_day_s + i/fps`` (the source camera's jpeg clock); row ``j``
    has wall time ``pressure_epoch_day_s + raw_t_us[j]/1e6`` (``started_at_iso``
    + raw ``time.monotonic`` bridge -- the same clock, S2 §三 证据 C).
    Quality flags are taken from the nearest row; no interpolation.  Video
    frames outside the source span clamp to the edge row and are counted.
    Fake rows that carry no raw timestamp (dropout spans) are invisible to
    that selection; their flags are propagated onto the frame span between the
    neighbouring real rows (see ``fake_gap_frame_marks``).

    Legacy mode (no anchors or no raw timestamps): the previous normalized
    grid, with nearest-row flags instead of linear interpolation.
    """
    target = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    absolute = (video_start_day_s is not None and pressure_epoch_day_s is not None
                and np.isfinite(left[1]).any() and np.isfinite(right[1]).any())
    out = {}

    def map_foot(pack: tuple) -> tuple[np.ndarray, np.ndarray, int, int]:
        time, raw_time, valid, fake = pack
        if absolute:
            # Map onto real raw samples only: rows without a raw timestamp
            # (bridge/interpolated rows) are never selected as a source row.
            has_raw = np.isfinite(raw_time)
            raw_rows = np.flatnonzero(has_raw)
            t_sub = raw_time[raw_rows] / 1e6 + pressure_epoch_day_s
            targets = video_start_day_s + np.arange(n_frames, dtype=np.float64) / fps
            idx = raw_rows[nearest_indices(t_sub, targets)]
            n_clamp_start = int((targets < t_sub[0]).sum())
            n_clamp_end = int((targets > t_sub[-1]).sum())
            frame_fake = fake[idx].astype(np.uint8)
            frame_fake |= fake_gap_frame_marks(raw_rows, fake, t_sub, targets).astype(np.uint8)
        else:
            idx = np.rint(target * max(len(time) - 1, 0)).astype(int)
            n_clamp_start = n_clamp_end = 0
            frame_fake = fake[idx].astype(np.uint8)
        return valid[idx], frame_fake, n_clamp_start, n_clamp_end

    left_valid, left_fake, lc0, lc1 = map_foot(left)
    right_valid, right_fake, rc0, rc1 = map_foot(right)
    fake = np.maximum(left_fake, right_fake).astype(np.uint8)
    valid = np.minimum(left_valid, right_valid).astype(np.uint8)
    valid = np.minimum(valid, 1 - fake).astype(np.uint8)
    out.update(
        alignment_method="nearest_values_absolute_t_us" if absolute else "nearest_values_normalized_t_us",
        absolute=absolute,
        video_frame0_wall_s=video_start_day_s, pressure_epoch_wall_s=pressure_epoch_day_s,
        out_of_span_frames_left=(lc0, lc1), out_of_span_frames_right=(rc0, rc1),
    )
    if absolute:
        frame_times = video_start_day_s + np.arange(n_frames, dtype=np.float64) / fps
    else:
        frame_times = target
    return valid, fake, frame_times, out


