"""Persistent local Fast SAM3DB → neural SMPL runtime."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .config import (
    CHECKPOINT_ROOT,
    MHR2SMPL_MODEL_DIR,
    MHR_FACE_MESH,
    MHR_MAPPING,
    REQUIRED_ASSETS,
    SMPL_MODEL,
    UPSTREAM_ROOT,
)


SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares similarity mapping row-vector ``source`` onto ``target``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("similarity inputs must have matching [N,3] shapes")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1.0
    rotation = (u * correction[None, :]) @ vt
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        raise ValueError("degenerate SMPL vertices")
    scale = float(np.sum(singular * correction) / denominator)
    translation = target_mean - scale * (source_mean @ rotation)
    aligned = scale * (source @ rotation) + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return scale, rotation.astype(np.float32), translation.astype(np.float32), rmse


class Engine:
    """One persistent model stack per GPU worker."""

    def __init__(self, device: str = "cuda") -> None:
        for name, path in REQUIRED_ASSETS.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"missing required asset {name}: {path}")
        if os.environ.get("SAM3DB_OFFLINE") != "1":
            raise RuntimeError("SAM3DB_OFFLINE=1 is required")
        if str(UPSTREAM_ROOT) not in sys.path:
            sys.path.insert(0, str(UPSTREAM_ROOT))

        import torch
        from notebook.utils import setup_sam_3d_body
        from mocap.core.multiview_mhr2smpl import (
            MultiViewFusionRunner,
            rot6d_to_aa,
            split_output,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Fast SAM3DB inference")
        self.torch = torch
        self._rot6d_to_aa = rot6d_to_aa
        self._split_output = split_output
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        self.estimator = setup_sam_3d_body(
            detector_name=None,
            segmentor_name=None,
            fov_name="moge2",
            device=device,
            local_checkpoint_path=str(CHECKPOINT_ROOT),
        )
        self.converter = MultiViewFusionRunner(
            smpl_model_path=SMPL_MODEL,
            model_dir=MHR2SMPL_MODEL_DIR,
            mapping_path=MHR_MAPPING,
            mhr_mesh_path=MHR_FACE_MESH,
            device=device,
        )

    def close(self) -> None:
        self._devnull.close()

    def infer(self, image_path: str | Path, bbox: np.ndarray, image_size: tuple[int, int]) -> dict[str, Any]:
        torch = self.torch
        box = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        with contextlib.redirect_stdout(self._devnull):
            outputs = self.estimator.process_one_image(
                str(image_path), bboxes=box, inference_type="body"
            )
        if len(outputs) != 1:
            raise RuntimeError(f"annotation-constrained inference returned {len(outputs)} people")
        pred = outputs[0]
        features = self.converter._preprocess_view(pred["pred_vertices"], pred["pred_cam_t"])
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.converter.device)
        with torch.no_grad():
            params = self.converter.model.forward_single_view(tensor)
            go6d, body_pose, betas = self._split_output(params)
            global_orient = self._rot6d_to_aa(go6d)
            body_full = torch.zeros((1, 69), dtype=torch.float32, device=self.converter.device)
            body_full[:, :63] = body_pose
            smpl = self.converter._smpl(
                global_orient=global_orient, body_pose=body_full, betas=betas
            )

        canonical_vertices = smpl.vertices[0].detach().cpu().numpy().astype(np.float32)
        canonical_joints = smpl.joints[0, :24].detach().cpu().numpy().astype(np.float32)
        mhr_vertices = np.asarray(pred["pred_vertices"], dtype=np.float32)
        mapped_vertices = (
            mhr_vertices[self.converter._mhr_vert_ids]
            * self.converter._baryc[:, :, None]
        ).sum(axis=1)
        scale, rotation, translation, rmse = similarity_transform(
            canonical_vertices, mapped_vertices
        )
        joints_local = scale * (canonical_joints @ rotation) + translation
        camera_translation = np.asarray(pred["pred_cam_t"], dtype=np.float32)
        joints_camera = joints_local + camera_translation[None, :]
        focal = float(np.asarray(pred["focal_length"]).reshape(-1)[0])
        width, height = image_size
        depth = joints_camera[:, 2]
        if not np.isfinite(depth).all() or np.any(depth <= 1e-5):
            raise RuntimeError("projected SMPL joints have invalid depth")
        joints_2d = np.stack(
            (
                focal * joints_camera[:, 0] / depth + width / 2.0,
                focal * joints_camera[:, 1] / depth + height / 2.0,
            ),
            axis=1,
        ).astype(np.float32)
        result = {
            "bbox_xyxy": box[0],
            "global_orient": global_orient[0].detach().cpu().numpy().astype(np.float32),
            "body_pose": body_pose[0].detach().cpu().numpy().astype(np.float32),
            "betas": betas[0].detach().cpu().numpy().astype(np.float32),
            "camera_translation": camera_translation,
            "focal_length": np.float32(focal),
            "similarity_scale": np.float32(scale),
            "similarity_rotation": rotation,
            "similarity_translation": translation,
            "smpl_joints_2d": joints_2d,
            "alignment_rmse": np.float32(rmse),
        }
        for name, value in result.items():
            if not np.isfinite(np.asarray(value)).all():
                raise RuntimeError(f"non-finite runtime output: {name}")
        return result


def draw_overlay(image: np.ndarray, joints: np.ndarray, bbox: np.ndarray, label: str) -> np.ndarray:
    import cv2

    output = image.copy()
    box = np.rint(bbox).astype(int)
    cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), (70, 220, 70), 2)
    points = np.rint(joints).astype(int)
    height, width = output.shape[:2]
    for child, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        a, b = points[parent], points[child]
        if all((-width <= p[0] < 2 * width and -height <= p[1] < 2 * height) for p in (a, b)):
            cv2.line(output, tuple(a), tuple(b), (0, 190, 255), 3, cv2.LINE_AA)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(output, (int(x), int(y)), 4, (255, 120, 40), -1, cv2.LINE_AA)
    cv2.putText(output, label, (max(8, box[0]), max(24, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 230, 255), 2, cv2.LINE_AA)
    return output
