"""Small CPU renderer for target-bbox and fitted-SMPL joint overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


SMPL_EDGES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
)


class VisualizationError(RuntimeError):
    """Raised when an overlay cannot be rendered or verified."""


def project_joints(joints: Any, intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame [T,24,3] SMPL joints using per-frame 3x3 K."""

    points = np.asarray(joints, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (24, 3):
        raise VisualizationError(f"joints must be [T,24,3], got {points.shape}")
    if matrices.shape == (3, 3):
        matrices = np.broadcast_to(matrices, (points.shape[0], 3, 3))
    if matrices.shape != (points.shape[0], 3, 3):
        raise VisualizationError(f"intrinsics must be [3,3] or [T,3,3], got {matrices.shape}")
    if not np.isfinite(points).all() or not np.isfinite(matrices).all():
        raise VisualizationError("joints or intrinsics contain non-finite values")
    depth = points[..., 2]
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)
    normalized = points / safe_depth[..., None]
    projected = np.einsum("tij,tkj->tki", matrices, normalized)[..., :2]
    valid &= np.isfinite(projected).all(axis=-1)
    return projected, valid


def render_overlay(
    video_path: str | Path,
    output_path: str | Path,
    boxes_xyxy: Any,
    joints_camera: Any,
    intrinsics: Any,
    *,
    fps: float = 30.0,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Draw the annotated target box and fitted neutral-SMPL skeleton per frame."""

    source = Path(video_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VisualizationError(f"cannot open source video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    joints = np.asarray(joints_camera, dtype=np.float64)
    if boxes.shape != (reported_count, 4) or joints.shape != (reported_count, 24, 3):
        capture.release()
        raise VisualizationError(
            f"video/overlay lengths disagree: video={reported_count}, boxes={boxes.shape}, joints={joints.shape}"
        )
    if not np.isfinite(boxes).all() or not np.isfinite(joints).all():
        capture.release()
        raise VisualizationError("overlay inputs contain non-finite values")
    projected, visible = project_joints(joints, intrinsics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise VisualizationError(f"cannot create overlay video: {destination}")
    edges_drawn = 0
    count = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            box = np.rint(boxes[count]).astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 220, 255), 2)
            for first, second in SMPL_EDGES:
                if visible[count, first] and visible[count, second]:
                    a = tuple(np.rint(projected[count, first]).astype(int))
                    b = tuple(np.rint(projected[count, second]).astype(int))
                    cv2.line(image, a, b, (40, 230, 40), 2, cv2.LINE_AA)
                    edges_drawn += 1
            for joint_index, point in enumerate(projected[count]):
                if visible[count, joint_index]:
                    center = tuple(np.rint(point).astype(int))
                    cv2.circle(image, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
            label = labels[count] if labels is not None else "GVHMR 2024 -> neutral SMPL fit"
            cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
            count += 1
    finally:
        capture.release()
        writer.release()
    if count != reported_count or count == 0:
        raise VisualizationError(f"overlay wrote {count} frames; source reported {reported_count}")
    check = cv2.VideoCapture(str(destination))
    if not check.isOpened():
        raise VisualizationError(f"overlay is not readable after writing: {destination}")
    output_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check.release()
    if (output_count, output_width, output_height) != (count, width, height):
        raise VisualizationError(
            f"overlay verification mismatch: {(output_count, output_width, output_height)} != {(count, width, height)}"
        )
    return {
        "path": str(destination),
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": float(fps),
        "projected_joints": int(visible.sum()),
        "skeleton_edges_drawn": int(edges_drawn),
    }


__all__ = ["SMPL_EDGES", "VisualizationError", "project_joints", "render_overlay"]
