"""Tests for the build_shared absolute-time nearest-neighbour pressure resampling.

Self-contained: run directly —

    python tests/test_build_shared_resample.py     # touch_gait env

Implements the S2 clock-scan ruling (Baselines/决策/05_R1-R3_风险裁定.md §三
证据 C): values and quality flags are taken from the nearest real raw sample
by absolute wall-clock time; linear interpolation and the normalized-grid
compression are gone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "AnysoleWorkspace" / "tool")):
    if p not in sys.path:
        sys.path.insert(0, p)

from AnysoleWorkspace.tool.build_shared import (  # noqa: E402
    day_seconds_from_iso,
    day_seconds_from_jpeg_name,
    nearest_indices,
    resample_pressure,
)


def pack(times_us, values, valid=None, fake=None, raw_us=None):
    times = np.asarray(times_us, dtype=np.float64)
    rows = len(times)
    if values is None:
        values = np.zeros((rows, 48), dtype=np.float32)
    else:
        values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.ones(rows, dtype=np.uint8)
    if fake is None:
        fake = np.zeros(rows, dtype=np.uint8)
    if raw_us is None:
        raw_us = np.full(rows, np.nan)
    return (times, np.asarray(raw_us, dtype=np.float64), values, np.asarray(valid, np.uint8), np.asarray(fake, np.uint8))


def test_day_seconds_parsers():
    assert day_seconds_from_iso("2026-08-04T19:51:59.117557") == 19 * 3600 + 51 * 60 + 59.117557
    assert day_seconds_from_jpeg_name("3_195159.136.jpg") == 19 * 3600 + 51 * 60 + 59.136
    assert day_seconds_from_jpeg_name("3_000001.000.jpg") == 1.0


def test_nearest_indices():
    times = np.array([0.18, 0.33, 0.48, 0.63])
    idx = nearest_indices(times, np.array([0.25, 0.25, 0.10, 0.90, 0.40]))
    assert list(idx) == [0, 0, 0, 3, 1]  # 0.25 -> 0.18 (user's example)


def test_absolute_mode_picks_real_samples():
    # user's example: tactile at 0.18s / 0.33s; video frame at 0.25s takes 0.18s
    epoch = 1000.0
    left = pack([180000, 330000], raw_us=[180000, 330000],
                values=np.array([[1.0] * 48, [2.0] * 48], dtype=np.float32))
    right = pack([180000, 330000], raw_us=[180000, 330000],
                 values=np.array([[10.0] * 48, [20.0] * 48], dtype=np.float32))
    # video frames at 1000.20 / 1000.225 / 1000.25 -> nearest raw samples 0.18, 0.18, 0.18
    l48, r48, valid, fake, times, meta = resample_pressure(
        left, right, 3, video_start_day_s=epoch + 0.20, pressure_epoch_day_s=epoch, fps=40.0)
    assert meta["absolute"] and meta["alignment_method"] == "nearest_values_absolute_t_us"
    assert np.allclose(times, [epoch + 0.20, epoch + 0.225, epoch + 0.25])
    assert np.allclose(l48[:, 0], [1.0, 1.0, 1.0])
    assert np.allclose(r48[:, 0], [10.0, 10.0, 10.0])
    assert meta["out_of_span_frames_right"] == (0, 0)


def test_absolute_mode_clamps_and_counts_out_of_span():
    epoch = 1000.0
    foot = pack([180000, 330000], raw_us=[180000, 330000],
                values=np.array([[1.0] * 48, [2.0] * 48], dtype=np.float32))
    # frames span 1000.10 .. 1000.375 -> 4 clamp at start (before 0.18), 2 at end (after 0.33)
    l48, r48, valid, fake, times, meta = resample_pressure(
        foot, foot, 12, video_start_day_s=epoch + 0.10, pressure_epoch_day_s=epoch, fps=40.0)
    assert meta["out_of_span_frames_right"][0] == 4 and meta["out_of_span_frames_right"][1] == 2
    assert r48[0, 0] == 1.0 and r48[-1, 0] == 2.0  # clamped to edge samples


def test_quality_flags_combine_across_feet():
    epoch = 1000.0
    left = pack([180000, 330000], None, raw_us=[180000, 330000],
                valid=np.array([1, 1], np.uint8), fake=np.array([0, 1], np.uint8))
    right = pack([180000, 330000], None, raw_us=[180000, 330000],
                 valid=np.array([1, 1], np.uint8), fake=np.array([0, 0], np.uint8))
    l48, r48, valid, fake, times, meta = resample_pressure(
        left, right, 4, video_start_day_s=epoch + 0.20, pressure_epoch_day_s=epoch, fps=40.0)
    # frames at 0.20/0.225/0.25 -> row 0; 0.275 -> row 1 (left fake=1 -> combined fake)
    assert list(fake) == [0, 0, 0, 1] and list(valid) == [1, 1, 1, 0]


def test_legacy_fallback_without_raw_timestamps():
    left = pack([0, 25000, 50000], values=np.array([[i] * 48 for i in (1, 2, 3)], dtype=np.float32))
    right = pack([0, 25000, 50000], values=np.array([[i] * 48 for i in (1, 2, 3)], dtype=np.float32))
    l48, r48, valid, fake, times, meta = resample_pressure(
        left, right, 2, video_start_day_s=None, pressure_epoch_day_s=None, fps=40.0)
    assert not meta["absolute"] and meta["alignment_method"] == "nearest_values_normalized_t_us"
    assert np.allclose(r48[:, 0], [1.0, 3.0])  # normalized nearest rows 0, 2


def test_bridge_rows_never_selected():
    epoch = 1000.0
    # row 1 is a bridge row (no raw timestamp); its value must never be selected
    foot = pack([0, 25000, 50000], raw_us=[0.0, np.nan, 50000.0],
                values=np.array([[1.0] * 48, [99.0] * 48, [3.0] * 48], dtype=np.float32))
    l48, r48, valid, fake, times, meta = resample_pressure(
        foot, foot, 5, video_start_day_s=epoch, pressure_epoch_day_s=epoch, fps=40.0)
    assert not (r48[:, 0] == 99.0).any()
    assert np.allclose(r48[:, 0], [1.0, 1.0, 3.0, 3.0, 3.0])


if __name__ == "__main__":
    test_day_seconds_parsers()
    test_nearest_indices()
    test_absolute_mode_picks_real_samples()
    test_absolute_mode_clamps_and_counts_out_of_span()
    test_quality_flags_combine_across_feet()
    test_legacy_fallback_without_raw_timestamps()
    test_bridge_rows_never_selected()
    print("build_shared resample tests: 7/7 passed")
