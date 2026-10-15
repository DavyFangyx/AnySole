"""Fixed calibration transforms for GT-prime and camera-space predictions."""

from __future__ import annotations

import numpy as np


def gt_smpl_to_checkerboard(points_m: np.ndarray, calibration: dict) -> np.ndarray:
    """Map GT' SMPL-world metres into checkerboard-world metres.

    The date calibration was fitted on the same normalized mocap axes retained
    by the published GT' SMPL file. Its documented ``input_scale=0.1`` consumes
    source millimetres, hence SMPL metres become checkerboard centimetres by
    multiplying by 100 before applying the recorded rigid transform.
    """

    points = np.asarray(points_m, dtype=np.float64)
    transform = calibration["mocap_raw_to_checkerboard_world"]
    rotation = np.asarray(transform["R"], dtype=np.float64)
    translation_cm = np.asarray(transform["t"], dtype=np.float64)
    return (np.einsum("ij,...j->...i", rotation, points * 100.0) + translation_cm) / 100.0


def checkerboard_to_camera(points_m: np.ndarray, camera: dict) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation_cm = np.asarray(camera["t"], dtype=np.float64)
    return (np.einsum("ij,...j->...i", rotation, points * 100.0) + translation_cm) / 100.0


def camera_to_checkerboard(points_m: np.ndarray, camera: dict) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation_cm = np.asarray(camera["t"], dtype=np.float64)
    return np.einsum("ji,...j->...i", rotation, points * 100.0 - translation_cm) / 100.0


def gt_root_rotation_to_checkerboard(rotation: np.ndarray, calibration: dict) -> np.ndarray:
    fixed = np.asarray(calibration["mocap_raw_to_checkerboard_world"]["R"], dtype=np.float64)
    return np.einsum("ij,...jk->...ik", fixed, np.asarray(rotation, dtype=np.float64))


def camera_root_rotation_to_checkerboard(rotation: np.ndarray, camera: dict) -> np.ndarray:
    fixed = np.asarray(camera["R"], dtype=np.float64)
    return np.einsum("ji,...jk->...ik", fixed, np.asarray(rotation, dtype=np.float64))

