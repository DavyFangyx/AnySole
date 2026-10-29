"""Render review artifacts from a saved WHAM prediction.

This CPU-only post-processor projects saved camera-coordinate SMPL vertices
with the same focal-length heuristic used by WHAM. It writes a side-by-side
video and contact sheet without PyTorch3D or GPU use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import cv2
import joblib
import numpy as np


COLORS_BGR = ((72, 220, 72), (32, 150, 255), (220, 80, 220), (220, 210, 40), (80, 80, 255))
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


class VisualizationError(RuntimeError):
    """Raised when saved prediction artifacts cannot be rendered safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_smpl_geometry(path: str | Path) -> tuple[np.ndarray, Any]:
    """Load only faces and the joint regressor from a licensed SMPL pickle."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    try:
        faces = np.asarray(model["f"], dtype=np.int32)
        joint_regressor = model["J_regressor"]
    except (KeyError, TypeError) as exc:
        raise VisualizationError(f"SMPL model {source} lacks faces/J_regressor") from exc
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise VisualizationError(f"invalid SMPL faces shape {faces.shape}")
    return faces, joint_regressor


def project_vertices(
    vertices: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-coordinate vertices with WHAM's demo intrinsics."""

    values = np.asarray(vertices, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise VisualizationError(f"expected vertices [N,3], got {values.shape}")
    focal_x = focal_y = math.hypot(width, height)
    center_x, center_y = width / 2.0, height / 2.0
    valid = np.isfinite(values).all(axis=1) & (values[:, 2] > 1e-6)
    pixels = np.full((len(values), 2), np.nan, dtype=np.float32)
    pixels[valid, 0] = focal_x * values[valid, 0] / values[valid, 2] + center_x
    pixels[valid, 1] = focal_y * values[valid, 1] / values[valid, 2] + center_y
    return pixels, valid


def _joint_positions(joint_regressor: Any, vertices: np.ndarray) -> np.ndarray:
    return np.asarray(joint_regressor.dot(vertices), dtype=np.float32)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.65) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _frame_vertices(track: Mapping[str, Any], frame_index: int) -> np.ndarray | None:
    frame_ids = np.asarray(track.get("frame_ids", ()), dtype=np.int64)
    positions = np.flatnonzero(frame_ids == frame_index)
    if not len(positions):
        return None
    return np.asarray(track["verts"][int(positions[0])], dtype=np.float32)


def render_overlay(
    frame: np.ndarray,
    predictions: Mapping[Any, Mapping[str, Any]],
    frame_index: int,
    faces: np.ndarray,
    joint_regressor: Any,
    *,
    mesh_alpha: float = 0.35,
    wire_faces: int = 260,
    joints_only: bool = False,
) -> tuple[np.ndarray, list[Any]]:
    """Overlay all tracks that contain ``frame_index`` on one BGR frame."""

    if not 0.0 <= mesh_alpha <= 1.0:
        raise VisualizationError("mesh_alpha must be in [0, 1]")
    height, width = frame.shape[:2]
    output = frame.copy()
    visible_tracks: list[Any] = []

    for order, (track_id, track) in enumerate(sorted(predictions.items(), key=lambda item: str(item[0]))):
        vertices = _frame_vertices(track, frame_index)
        if vertices is None:
            continue
        if vertices.shape != (6890, 3):
            raise VisualizationError(f"track {track_id!r} has vertices shape {vertices.shape}, expected (6890, 3)")
        pixels, valid_vertices = project_vertices(vertices, width, height)
        color = np.asarray(COLORS_BGR[order % len(COLORS_BGR)], dtype=np.float32)

        mask = np.zeros((height, width), dtype=np.uint8)
        if not joints_only:
            valid_faces = valid_vertices[faces].all(axis=1)
            triangles = np.rint(pixels[faces[valid_faces]]).astype(np.int32)
            if len(triangles):
                cv2.fillPoly(mask, triangles, 255, lineType=cv2.LINE_AA)
                selected = np.flatnonzero(valid_faces)
                if wire_faces > 0 and len(selected):
                    step = max(1, len(selected) // wire_faces)
                    wire = np.rint(pixels[faces[selected[::step]]]).astype(np.int32)
                    cv2.polylines(output, wire, True, tuple(int(x) for x in color), 1, cv2.LINE_AA)
        inside = mask.astype(bool)
        if inside.any():
            blended = output[inside].astype(np.float32) * (1.0 - mesh_alpha) + color * mesh_alpha
            output[inside] = np.clip(blended, 0, 255).astype(np.uint8)

        joints = _joint_positions(joint_regressor, vertices)
        joint_pixels, joint_valid = project_vertices(joints, width, height)
        joint_pixels_int = np.rint(np.nan_to_num(joint_pixels, nan=-10000.0)).astype(np.int32)
        draw_color = tuple(int(x) for x in color)
        for child, parent in enumerate(SMPL_PARENTS):
            if parent >= 0 and child < len(joint_valid) and joint_valid[child] and joint_valid[parent]:
                cv2.line(output, tuple(joint_pixels_int[parent]), tuple(joint_pixels_int[child]), draw_color, 3, cv2.LINE_AA)
        for point, valid in zip(joint_pixels_int, joint_valid):
            if valid and -8 <= point[0] < width + 8 and -8 <= point[1] < height + 8:
                cv2.circle(output, tuple(point), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(output, tuple(point), 3, draw_color, -1, cv2.LINE_AA)

        in_frame = valid_vertices & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        if in_frame.any():
            anchor = np.nanmedian(pixels[in_frame], axis=0).astype(int)
            _put_label(output, f"track {track_id}", (int(anchor[0]), max(24, int(anchor[1]) - 12)))
        visible_tracks.append(track_id)

    _put_label(output, f"WHAM | frame {frame_index:04d} | {len(visible_tracks)} tracks", (20, 36), 0.8)
    return output, visible_tracks


def _side_by_side(original: np.ndarray, overlay: np.ndarray, scale: float) -> np.ndarray:
    if not 0.0 < scale <= 1.0:
        raise VisualizationError("review_scale must be in (0, 1]")
    height, width = original.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    left = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    _put_label(left, "ORIGINAL", (14, 30), 0.75)
    _put_label(right, "WHAM OVERLAY", (14, 30), 0.75)
    return np.concatenate((left, right), axis=1)


def _parse_contact_frames(value: str, frame_count: int) -> list[int]:
    if value.strip():
        frames = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    else:
        frames = sorted({0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1})
    if not frames or frames[0] < 0 or frames[-1] >= frame_count:
        raise VisualizationError(f"contact frames {frames} outside [0, {frame_count})")
    return frames


def render_review(
    video_path: str | Path,
    prediction_path: str | Path,
    smpl_model_path: str | Path,
    output_dir: str | Path,
    *,
    label: str = "",
    contact_frames: str = "",
    review_scale: float = 0.5,
    mesh_alpha: float = 0.35,
    joints_only: bool = False,
) -> dict[str, Any]:
    """Render a side-by-side MP4, selected PNGs, contact sheet, and metadata."""

    video = Path(video_path).expanduser().resolve()
    prediction = Path(prediction_path).expanduser().resolve()
    smpl_model = Path(smpl_model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    for source in (video, prediction, smpl_model):
        if not source.is_file():
            raise VisualizationError(f"required input does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    loaded = joblib.load(prediction)
    if not isinstance(loaded, Mapping) or not loaded:
        raise VisualizationError(f"prediction {prediction} must contain at least one track")
    predictions: Mapping[Any, Mapping[str, Any]] = loaded
    faces, joint_regressor = load_smpl_geometry(smpl_model)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VisualizationError(f"cannot open video {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise VisualizationError(f"invalid video geometry/count for {video}")
    selected = _parse_contact_frames(contact_frames, frame_count)

    review_width = max(1, round(width * review_scale)) * 2
    review_height = max(1, round(height * review_scale))
    video_output = destination / "wham-review-side-by-side.mp4"
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (review_width, review_height))
    if not writer.isOpened():
        capture.release()
        raise VisualizationError(f"cannot create video {video_output}")

    contact_images: list[np.ndarray] = []
    tracks_by_frame: dict[str, list[str]] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            overlay, visible = render_overlay(
                frame,
                predictions,
                frame_index,
                faces,
                joint_regressor,
                mesh_alpha=mesh_alpha,
                joints_only=joints_only,
            )
            combined = _side_by_side(frame, overlay, review_scale)
            if label:
                _put_label(combined, label, (20, review_height - 18), 0.6)
            writer.write(combined)
            if frame_index in selected:
                contact_images.append(combined.copy())
                cv2.imwrite(str(destination / f"frame-{frame_index:04d}-comparison.png"), combined)
                tracks_by_frame[str(frame_index)] = [str(item) for item in visible]
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index != frame_count:
        raise VisualizationError(f"decoded {frame_index} frames but video metadata reported {frame_count}")
    if len(contact_images) != len(selected):
        raise VisualizationError("not all requested contact frames were decoded")

    separator = np.full((12, review_width, 3), 245, dtype=np.uint8)
    sheet_parts: list[np.ndarray] = []
    for position, image in enumerate(contact_images):
        if position:
            sheet_parts.append(separator)
        sheet_parts.append(image)
    contact_sheet = np.concatenate(sheet_parts, axis=0)
    contact_output = destination / "wham-review-contact-sheet.jpg"
    cv2.imwrite(str(contact_output), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])

    metadata = {
        "schema_id": "tactile.wham_baseline.review-visualization",
        "schema_version": 1,
        "label": label,
        "inputs": {
            "video": str(video), "video_sha256": _sha256(video),
            "prediction": str(prediction), "prediction_sha256": _sha256(prediction),
            "smpl_model": str(smpl_model), "smpl_model_sha256": _sha256(smpl_model),
        },
        "video": {"frames": frame_count, "fps": fps, "width": width, "height": height},
        "prediction": {"track_count": len(predictions), "track_ids": [str(key) for key in sorted(predictions, key=str)]},
        "render": {
            "projection": "f=sqrt(width^2+height^2), principal_point=(width/2,height/2)",
            "calibration": None,
            "joints_only": joints_only,
            "mesh_alpha": mesh_alpha, "review_scale": review_scale, "contact_frames": selected,
            "tracks_by_contact_frame": tracks_by_frame,
        },
        "outputs": {
            "side_by_side_video": str(video_output), "contact_sheet": str(contact_output),
            "comparison_frames": [str(destination / f"frame-{index:04d}-comparison.png") for index in selected],
        },
    }
    metadata_output = destination / "metadata.json"
    metadata["outputs"]["metadata"] = str(metadata_output)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--smpl-model", default="/private/Tactile/models/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--contact-frames", default="")
    parser.add_argument("--review-scale", type=float, default=0.5)
    parser.add_argument("--mesh-alpha", type=float, default=0.35)
    parser.add_argument("--joints-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = render_review(
        args.video, args.prediction, args.smpl_model, args.output_dir,
        label=args.label, contact_frames=args.contact_frames,
        review_scale=args.review_scale, mesh_alpha=args.mesh_alpha,
        joints_only=args.joints_only,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
