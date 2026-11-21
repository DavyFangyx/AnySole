"""Build SMPL-X training labels from the neutral SMPL ground truth.

The label direction is deliberately the inverse of the evaluation adapter:
the neutral SMPL GT surface is used as the target, a neutral SMPL-X mesh is
fitted through the official ``smplx2smpl_sparse.pt`` correspondence, and the
fitted root is then transformed into the physical cam3 frame.  No per-date
``*_native_to_mocap`` offset is applied because those offsets absorb the
pretrained model's depth bias and must not be baked into training labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import common
from .. import config as _config

SMPL_JOINTS = 22
_MODEL_CACHE: dict[str, tuple[Any, Any, Any, Any]] = {}


class LabelError(RuntimeError):
    """Raised when the SMPL-to-SMPL-X refit violates its quality gates."""


@dataclass(frozen=True)
class FitResult:
    body_pose: np.ndarray
    betas: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    vertex_rmse_mm: np.ndarray
    joint_rmse_mm: np.ndarray
    metadata: dict[str, Any]


def _load_body_models():
    import smplx
    import torch

    if not _config.DEFAULT_SMPL_MODEL.is_file() or not _config.DEFAULT_SMPLX_MODEL.is_file():
        raise LabelError("licensed SMPL / SMPL-X body models are missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = str(device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    smpl = smplx.create(
        str(_config.DEFAULT_SMPL_MODEL), "smpl", gender="neutral", num_betas=10
    ).to(device).eval()
    smplx_model = smplx.create(
        str(_config.DEFAULT_SMPLX_MODEL.parent.parent),
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=10,
    ).to(device).eval()
    mapper = torch.load(
        str(_config.DEFAULT_SMPLX_TO_SMPL), map_location="cpu"
    ).to(device).coalesce()
    _MODEL_CACHE[key] = (smpl, smplx_model, mapper, device)
    return _MODEL_CACHE[key]


def _map_vertices(vertices, mapper):
    import torch

    flat = vertices.permute(1, 0, 2).reshape(vertices.shape[1], -1)
    return torch.sparse.mm(mapper, flat).reshape(
        mapper.shape[0], vertices.shape[0], 3
    ).permute(1, 0, 2)


def _smplx_forward(smplx_model, body_pose, betas, global_orient, transl):
    """Call the 55-joint SMPL-X model with explicit full-batch defaults."""
    import torch

    batch = body_pose.shape[0]
    device = body_pose.device
    zeros = lambda width: torch.zeros(batch, width, dtype=torch.float32, device=device)
    return smplx_model(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        jaw_pose=zeros(3),
        leye_pose=zeros(3),
        reye_pose=zeros(3),
        left_hand_pose=zeros(45),
        right_hand_pose=zeros(45),
        expression=zeros(int(smplx_model.num_expression_coeffs)),
    )


def _smpl_forward(smpl_model, poses: np.ndarray, betas: np.ndarray, trans: np.ndarray):
    import torch

    device = next(smpl_model.parameters()).device
    with torch.no_grad():
        output = smpl_model(
            global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(poses[:, 3:], dtype=torch.float32, device=device),
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device)[None].expand(
                len(poses), -1
            ),
            transl=torch.as_tensor(trans, dtype=torch.float32, device=device),
        )
    return output.vertices, output.joints[:, :SMPL_JOINTS]


def fit_smplx(
    target_vertices,
    seed_poses: np.ndarray,
    seed_betas: np.ndarray,
    seed_trans: np.ndarray,
    *,
    steps: int = 300,
    batch_size: int = 24,
    learning_rate: float = 0.02,
    joint_weight: float = 0.0,
    pose_prior: float = 0.0,
    beta_prior: float = 0.0,
    translation_prior: float = 0.0,
    smooth_body: float = 0.005,
    smooth_root: float = 0.001,
    smooth_trans: float = 0.0,
    max_vertex_rmse_mm: float = 50.0,
    seed: int = 0,
    frame_times: np.ndarray | None = None,
    models: tuple[Any, Any, Any, Any] | None = None,
) -> FitResult:
    """Fit one shared SMPL-X shape and per-frame pose/root to SMPL surfaces."""
    import torch
    import torch.nn.functional as F

    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise LabelError("fit steps, batch size and learning rate must be positive")
    torch.manual_seed(int(seed))
    smpl, smplx_model, mapper, device = models if models is not None else _load_body_models()
    count = len(seed_poses)
    if target_vertices.shape[0] != count:
        raise LabelError("target vertices and seed poses disagree on frame count")
    target = torch.as_tensor(np.asarray(target_vertices), dtype=torch.float32, device=device)
    poses = torch.as_tensor(np.asarray(seed_poses), dtype=torch.float32, device=device)
    trans = torch.as_tensor(np.asarray(seed_trans), dtype=torch.float32, device=device)
    betas = torch.as_tensor(np.asarray(seed_betas), dtype=torch.float32, device=device)

    body = torch.nn.Parameter(poses[:, 3:66].clone())
    root = torch.nn.Parameter(poses[:, :3].clone())
    translation = torch.nn.Parameter(trans.clone())
    shape = torch.nn.Parameter(betas[None].clone())
    optimizer = torch.optim.Adam([body, root, translation, shape], lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(steps), eta_min=float(learning_rate) * 0.1
    )

    # SMPL joints of the target surface are the correspondence anchor beyond
    # the sparse vertex map; they are computed once and reused every epoch.
    with torch.no_grad():
        output = smpl(
            global_orient=poses[:, :3],
            body_pose=poses[:, 3:],
            betas=betas[None].expand(count, -1),
            transl=trans,
        )
        target_joints = output.joints[:, :SMPL_JOINTS].clone()
    if not torch.isfinite(target).all() or not torch.isfinite(target_joints).all():
        raise LabelError("target SMPL surface contains non-finite values")

    for epoch in range(int(steps)):
        order = torch.randperm(count, device=device)
        for start in range(0, count, int(batch_size)):
            index = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            output = _smplx_forward(
                smplx_model,
                body[index],
                shape.expand(len(index), -1),
                root[index],
                translation[index],
            )
            mapped = _map_vertices(output.vertices, mapper)
            vertex_loss = F.mse_loss(mapped, target[index])
            joint_loss = F.mse_loss(output.joints[:, :SMPL_JOINTS], target_joints[index])
            prior = (
                pose_prior * (body[index] - poses[index, 3:66]).square().mean()
                + pose_prior * (root[index] - poses[index, :3]).square().mean()
                + translation_prior * (translation[index] - trans[index]).square().mean()
                + beta_prior * (shape - betas[None]).square().mean()
            )
            smooth = (
                smooth_body * (body[1:] - body[:-1]).square().mean()
                + smooth_root * (root[1:] - root[:-1]).square().mean()
                + smooth_trans * (translation[1:] - translation[:-1]).square().mean()
            )
            (vertex_loss + float(joint_weight) * joint_loss + prior + smooth).backward()
            optimizer.step()
        scheduler.step()

    # Duplicate RGB stamps describe the same instant and target; share the
    # fitted parameters instead of letting the batched solver diverge on one
    # copy of an identical frame.
    duplicate_frames = 0
    if frame_times is not None:
        times = np.asarray(frame_times, dtype=np.float64)
        if times.shape != (count,):
            raise LabelError("frame_times must match the fitted frame count")
        duplicates = np.flatnonzero(np.diff(times) <= 1e-9) + 1
        for index_value in duplicates:
            body.data[index_value] = body.data[index_value - 1]
            root.data[index_value] = root.data[index_value - 1]
            translation.data[index_value] = translation.data[index_value - 1]
        duplicate_frames = int(len(duplicates))

    def residuals():
        vertex_values = []
        joint_values = []
        with torch.no_grad():
            for start in range(0, count, int(batch_size)):
                end = min(count, start + int(batch_size))
                output = _smplx_forward(
                    smplx_model,
                    body[start:end],
                    shape.expand(end - start, -1),
                    root[start:end],
                    translation[start:end],
                )
                mapped = _map_vertices(output.vertices, mapper)
                error = torch.linalg.vector_norm(mapped - target[start:end], dim=-1)
                joint_error = torch.linalg.vector_norm(
                    output.joints[:, :SMPL_JOINTS] - target_joints[start:end], dim=-1
                )
                vertex_values.append(
                    torch.sqrt(error.square().mean(dim=1)).cpu().numpy() * 1000.0
                )
                joint_values.append(
                    torch.sqrt(joint_error.square().mean(dim=1)).cpu().numpy() * 1000.0
                )
        return (
            np.concatenate(vertex_values).astype(np.float32),
            np.concatenate(joint_values).astype(np.float32),
        )

    vertex_rmse, joint_rmse = residuals()
    failed = vertex_rmse > float(max_vertex_rmse_mm)
    refined = 0
    if failed.any():
        # Fix outliers from a neighbour's fitted pose (local minima), never by
        # dropping frames.  Shape stays fixed to the session-wide value.
        for frame in np.flatnonzero(failed):
            source = None
            if frame > 0 and not failed[frame - 1]:
                source = frame - 1
            elif frame + 1 < count and not failed[frame + 1]:
                source = frame + 1
            if source is not None:
                init_body = body.data[source : source + 1].clone()
                init_root = root.data[source : source + 1].clone()
                init_trans = translation.data[source : source + 1].clone()
            else:
                init_body = poses[frame : frame + 1, 3:66].clone()
                init_root = poses[frame : frame + 1, :3].clone()
                init_trans = trans[frame : frame + 1].clone()
            local_body = torch.nn.Parameter(init_body)
            local_root = torch.nn.Parameter(init_root)
            local_trans = torch.nn.Parameter(init_trans)
            local_opt = torch.optim.Adam(
                [local_body, local_root, local_trans], lr=0.03
            )
            for _ in range(300):
                local_opt.zero_grad(set_to_none=True)
                local_out = _smplx_forward(
                    smplx_model,
                    local_body,
                    shape.expand(1, -1),
                    local_root,
                    local_trans,
                )
                local_loss = F.mse_loss(
                    _map_vertices(local_out.vertices, mapper),
                    target[frame : frame + 1],
                )
                local_loss.backward()
                local_opt.step()
            body.data[frame] = local_body.data[0]
            root.data[frame] = local_root.data[0]
            translation.data[frame] = local_trans.data[0]
            refined += 1
        vertex_rmse, joint_rmse = residuals()
        failed = vertex_rmse > float(max_vertex_rmse_mm)
    if failed.any():
        bad = np.flatnonzero(failed)
        raise LabelError(
            f"SMPL-X refit exceeds {max_vertex_rmse_mm:.0f} mm on {len(bad)} frame(s): "
            f"indices={bad[:10].tolist()} max={float(vertex_rmse.max()):.3f} mm"
        )
    if not np.isfinite(vertex_rmse).all() or not np.isfinite(joint_rmse).all():
        raise LabelError("SMPL-X refit produced non-finite residuals")
    return FitResult(
        body_pose=body.detach().cpu().numpy().astype(np.float32),
        betas=shape.detach().cpu().numpy().reshape(-1).astype(np.float32),
        global_orient=root.detach().cpu().numpy().astype(np.float32),
        transl=translation.detach().cpu().numpy().astype(np.float32),
        vertex_rmse_mm=vertex_rmse,
        joint_rmse_mm=joint_rmse,
        metadata={
            "target": "neutral SMPL GT surface",
            "correspondence": "official smplx2smpl_sparse.pt",
            "optimizer": "Adam + cosine",
            "seed": int(seed),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "joint_weight": float(joint_weight),
            "pose_prior": float(pose_prior),
            "beta_prior": float(beta_prior),
            "translation_prior": float(translation_prior),
            "smooth_body": float(smooth_body),
            "smooth_root": float(smooth_root),
            "smooth_trans": float(smooth_trans),
            "shape_shared_across_frames": True,
            "vertex_rmse_mm_mean": float(vertex_rmse.mean()),
            "vertex_rmse_mm_p95": float(np.percentile(vertex_rmse, 95)),
            "vertex_rmse_mm_max": float(vertex_rmse.max()),
            "joint_rmse_mm_mean": float(joint_rmse.mean()),
            "joint_rmse_mm_p95": float(np.percentile(joint_rmse, 95)),
            "frames_over_gate": 0,
            "duplicate_frames": int(duplicate_frames),
            "refined_frames": int(refined),
            "vertex_gate_mm": float(max_vertex_rmse_mm),
        },
    )


def _root_center_smpl(smpl_model, betas: np.ndarray) -> np.ndarray:
    """Shape-dependent rest root joint of the neutral SMPL model."""
    import torch

    device = next(smpl_model.parameters()).device
    with torch.no_grad():
        output = smpl_model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device)[None],
            transl=torch.zeros(1, 3, device=device),
        )
    return output.joints[0, 0].cpu().numpy().astype(np.float64)


def _root_center_smplx(smplx_model, betas: np.ndarray) -> np.ndarray:
    """Shape-dependent rest root joint of the neutral SMPL-X model."""
    import torch

    device = next(smplx_model.parameters()).device
    with torch.no_grad():
        output = _smplx_forward(
            smplx_model,
            torch.zeros(1, 63, device=device),
            torch.as_tensor(betas, dtype=torch.float32, device=device)[None],
            torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device),
        )
    return output.joints[0, 0].cpu().numpy().astype(np.float64)


def _projection_check(
    camera_joints: np.ndarray,
    boxes: np.ndarray,
    date: str,
    *,
    cam: int = 3,
    sample: int = 64,
) -> dict[str, Any]:
    """Project final camera-space label joints with the *training* intrinsics."""
    value = common.calibration(date)
    K = np.asarray(value["cameras"][f"cam{cam}"]["K"], dtype=np.float64)
    width, height = [int(item) for item in value["image_size_px"]]
    depth = camera_joints[..., 2]
    pixel = np.stack(
        [
            camera_joints[..., 0] / depth * K[0, 0] + K[0, 2],
            camera_joints[..., 1] / depth * K[1, 1] + K[1, 2],
        ],
        axis=-1,
    )
    skeleton_height = pixel[..., 1].max(axis=1) - pixel[..., 1].min(axis=1)
    box_height = boxes[:, 3] - boxes[:, 1]
    box_width = boxes[:, 2] - boxes[:, 0]
    ratio = skeleton_height / np.maximum(box_height, 1.0)
    projected_center = (pixel.min(axis=1) + pixel.max(axis=1)) / 2.0
    target_center = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=-1)
    offset_x = np.abs(projected_center[:, 0] - target_center[:, 0]) / np.maximum(box_width, 1.0)
    offset_y = np.abs(projected_center[:, 1] - target_center[:, 1]) / np.maximum(box_height, 1.0)
    index = np.linspace(0, len(ratio) - 1, min(sample, len(ratio))).round().astype(int)
    return {
        "frames": int(len(ratio)),
        "intrinsics": f"calibration cam{cam} K (physical-camera label diagnostic)",
        "focal_px": [float(K[0, 0]), float(K[1, 1])],
        "image_size_px": [width, height],
        "projected_skeleton_over_bbox_height_median": float(np.median(ratio[index])),
        "projected_skeleton_over_bbox_height_p95": float(np.percentile(ratio[index], 95)),
        "bbox_center_offset_fraction_x_median": float(np.median(offset_x[index])),
        "bbox_center_offset_fraction_y_median": float(np.median(offset_y[index])),
        "median_person_depth_m": float(np.median(depth[index])),
    }


def prepare_session_labels(
    row: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    cam: int = 3,
    fit_steps: int = 300,
    fit_batch_size: int = 24,
    fit_learning_rate: float = 0.02,
    max_vertex_rmse_mm: float = 50.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate camera-space SMPL-X labels for one RGB unit."""
    session = str(row["session_id"])
    date, _subject = common.session_date_subject(row)
    canonical = common.canonical_indices(row, unit, cam)
    query = common.label_times(row, unit, cam)
    smpl = common.resample_smpl(common.resolve_uri(row["smpl_path"]), query)
    valid = np.asarray(smpl["query_valid"], dtype=bool)
    if not valid.any():
        raise LabelError(f"{session}: no unit frame falls inside the SMPL GT support")
    boxes = common.bboxes_for_unit(row, unit, cam)
    timestamps = np.asarray(
        [float(value["timestamp_seconds"]) for value in unit["frames"]], dtype=np.float64
    )

    models = _load_body_models()
    smpl_model, smplx_model, mapper, device = models
    import torch

    index = np.flatnonzero(valid)
    poses = np.asarray(smpl["poses"], dtype=np.float32)[index]
    trans = np.asarray(smpl["trans"], dtype=np.float32)[index]
    betas = np.asarray(smpl["betas"], dtype=np.float32)
    with torch.no_grad():
        target_vertices, target_joints = _smpl_forward(smpl_model, poses, betas, trans)
    fit = fit_smplx(
        target_vertices.cpu().numpy(),
        poses,
        betas,
        trans,
        steps=fit_steps,
        batch_size=fit_batch_size,
        learning_rate=fit_learning_rate,
        max_vertex_rmse_mm=max_vertex_rmse_mm,
        frame_times=timestamps[index],
        models=models,
    )
    label_valid = valid.copy()

    rotation, translation, calibration = common.camera_transform(date, cam)
    j_smpl = _root_center_smpl(smpl_model, betas)
    j_smplx = _root_center_smplx(smplx_model, fit.betas)

    def to_camera(params: dict[str, np.ndarray], root_center: np.ndarray) -> dict[str, np.ndarray]:
        from scipy.spatial.transform import Rotation as SciRotation

        root = SciRotation.from_rotvec(params["global_orient"].astype(np.float64)).as_matrix()
        root_cam = np.einsum("ij,...jk->...ik", rotation, root)
        transl_world = np.asarray(params["transl"], dtype=np.float64)
        # SMPL/SMPL-X rotate about the shape-dependent rest root joint, so the
        # model translation transforms as R(t + J) + T - J, not R t + T.
        transl_cam = np.einsum(
            "ij,tj->ti", rotation, transl_world + root_center
        ) + translation - root_center
        return {
            "global_orient": SciRotation.from_matrix(root_cam).as_rotvec().astype(np.float32),
            "body_pose": params["body_pose"].astype(np.float32),
            "betas": params["betas"].astype(np.float32),
            "transl": transl_cam.astype(np.float32),
        }

    fitted = {
        "global_orient": fit.global_orient,
        "body_pose": fit.body_pose,
        "betas": np.broadcast_to(fit.betas[None], (len(index), 10)).copy(),
        "transl": fit.transl,
    }
    smplx_camera = to_camera(fitted, j_smplx)
    smpl_camera = to_camera(
        {
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3:],
            "betas": np.broadcast_to(betas[None], (len(poses), 10)).copy(),
            "transl": trans,
        },
        j_smpl,
    )

    def reconstruction_error(
        world_params: dict[str, np.ndarray],
        camera_params: dict[str, np.ndarray],
        model_kind: str,
    ) -> dict[str, float]:
        with torch.no_grad():
            if model_kind == "smplx":
                world_vertices = _smplx_forward(
                    smplx_model,
                    torch.as_tensor(world_params["body_pose"], dtype=torch.float32, device=device),
                    torch.as_tensor(world_params["betas"], dtype=torch.float32, device=device),
                    torch.as_tensor(world_params["global_orient"], dtype=torch.float32, device=device),
                    torch.as_tensor(world_params["transl"], dtype=torch.float32, device=device),
                ).vertices
                camera_vertices = _smplx_forward(
                    smplx_model,
                    torch.as_tensor(camera_params["body_pose"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["betas"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["global_orient"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["transl"], dtype=torch.float32, device=device),
                ).vertices
                camera_joints = _smplx_forward(
                    smplx_model,
                    torch.as_tensor(camera_params["body_pose"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["betas"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["global_orient"], dtype=torch.float32, device=device),
                    torch.as_tensor(camera_params["transl"], dtype=torch.float32, device=device),
                ).joints[:, :22]
            else:
                world_vertices = smpl_model(
                    global_orient=torch.as_tensor(world_params["global_orient"], dtype=torch.float32, device=device),
                    body_pose=torch.as_tensor(world_params["body_pose"], dtype=torch.float32, device=device),
                    betas=torch.as_tensor(world_params["betas"], dtype=torch.float32, device=device),
                    transl=torch.as_tensor(world_params["transl"], dtype=torch.float32, device=device),
                ).vertices
                camera_out = smpl_model(
                    global_orient=torch.as_tensor(camera_params["global_orient"], dtype=torch.float32, device=device),
                    body_pose=torch.as_tensor(camera_params["body_pose"], dtype=torch.float32, device=device),
                    betas=torch.as_tensor(camera_params["betas"], dtype=torch.float32, device=device),
                    transl=torch.as_tensor(camera_params["transl"], dtype=torch.float32, device=device),
                )
                camera_vertices = camera_out.vertices
                camera_joints = camera_out.joints[:, :22]
            expected = torch.einsum(
                "ij,tvj->tvi",
                torch.as_tensor(rotation, dtype=torch.float32, device=device),
                world_vertices,
            ) + torch.as_tensor(translation, dtype=torch.float32, device=device)
            error = torch.linalg.vector_norm(camera_vertices - expected, dim=-1)
        return {
            "vertex_max_mm": float(error.max() * 1000.0),
            "vertex_mean_mm": float(error.mean() * 1000.0),
            "camera_joints": camera_joints.cpu().numpy(),
        }

    smplx_check = reconstruction_error(
        {
            "body_pose": fit.body_pose,
            "betas": fitted["betas"],
            "global_orient": fit.global_orient,
            "transl": fit.transl,
        },
        smplx_camera,
        "smplx",
    )
    smpl_check = reconstruction_error(
        {
            "body_pose": poses[:, 3:],
            "betas": np.broadcast_to(betas[None], (len(poses), 10)).copy(),
            "global_orient": poses[:, :3],
            "transl": trans,
        },
        smpl_camera,
        "smpl",
    )
    for name, check in (("SMPL-X", smplx_check), ("SMPL", smpl_check)):
        if check["vertex_max_mm"] > 0.1:
            raise LabelError(
                f"{session}: {name} camera-label forward check failed: "
                f"max {check['vertex_max_mm']:.4f} mm"
            )

    # Non-blocking physical-camera diagnostic.  Training/inference use
    # estimate_K, so this projection is not a label validity gate.
    projection = _projection_check(
        smplx_check["camera_joints"], boxes[index], date, cam=cam
    )

    count = len(canonical)
    output: dict[str, np.ndarray] = {
        "valid": label_valid,
        "query_valid": valid,
        "canonical_index": canonical,
        "source_frame_number": np.asarray(
            [int(value["frame"]) for value in unit["frames"]], dtype=np.int64
        ),
        "timestamp_seconds": timestamps,
        "bbox_xyxy_px": boxes.astype(np.float32),
        "smplx_body_pose": np.zeros((count, 63), dtype=np.float32),
        "smplx_global_orient": np.zeros((count, 3), dtype=np.float32),
        "smplx_transl": np.zeros((count, 3), dtype=np.float32),
        "smplx_betas": np.zeros((count, 10), dtype=np.float32),
        "smplx_body_pose_world": np.zeros((count, 63), dtype=np.float32),
        "smplx_global_orient_world": np.zeros((count, 3), dtype=np.float32),
        "smplx_transl_world": np.zeros((count, 3), dtype=np.float32),
        "smpl_body_pose": np.zeros((count, 69), dtype=np.float32),
        "smpl_global_orient": np.zeros((count, 3), dtype=np.float32),
        "smpl_transl": np.zeros((count, 3), dtype=np.float32),
        "smpl_betas": np.broadcast_to(betas[None], (count, 10)).copy(),
        "fit_vertex_rmse_mm": np.full(count, np.nan, dtype=np.float32),
        "fit_joint_rmse_mm": np.full(count, np.nan, dtype=np.float32),
    }
    output["smplx_body_pose"][index] = smplx_camera["body_pose"]
    output["smplx_global_orient"][index] = smplx_camera["global_orient"]
    output["smplx_transl"][index] = smplx_camera["transl"]
    output["smplx_betas"][index] = smplx_camera["betas"]
    output["smplx_body_pose_world"][index] = fitted["body_pose"]
    output["smplx_global_orient_world"][index] = fitted["global_orient"]
    output["smplx_transl_world"][index] = fitted["transl"]
    output["smpl_body_pose"][index] = smpl_camera["body_pose"]
    output["smpl_global_orient"][index] = smpl_camera["global_orient"]
    output["smpl_transl"][index] = smpl_camera["transl"]
    output["fit_vertex_rmse_mm"][index] = fit.vertex_rmse_mm
    output["fit_joint_rmse_mm"][index] = fit.joint_rmse_mm

    metadata = {
        "session_id": session,
        "date": date,
        "camera": f"camera_{cam}",
        "unit_id": unit["unit_id"],
        "frame_count": int(count),
        "valid_frames": int(label_valid.sum()),
        "query_invalid_frames": int((~valid).sum()),
        "gate_invalid_frames": 0,
        "fit": fit.metadata,
        "projection_check": projection,
        "camera_forward_check": {
            "smplx_vertex_max_mm": smplx_check["vertex_max_mm"],
            "smplx_vertex_mean_mm": smplx_check["vertex_mean_mm"],
            "smpl_vertex_max_mm": smpl_check["vertex_max_mm"],
            "smpl_vertex_mean_mm": smpl_check["vertex_mean_mm"],
        },
        "root_center_m": {
            "smpl": j_smpl.tolist(),
            "smplx": j_smplx.tolist(),
        },
        "camera_transform": {
            "source": str(common.PROTOCOL / "calibration" / f"{date}.json"),
            "per_date_native_to_mocap_translation_applied": False,
            "rotation_det": float(np.linalg.det(rotation)),
            "translation_m": translation.tolist(),
            "root_center_correction": "t_c = R(t_w + J_beta) + T - J_beta",
        },
        "smpl_gt_path": str(common.resolve_uri(row["smpl_path"])),
        "smpl_gt_sha256": common.sha256_file(common.resolve_uri(row["smpl_path"])),
        "smplx_model": str(_config.DEFAULT_SMPLX_MODEL),
        "calibration_image_size_px": calibration["image_size_px"],
        "source_time_range_s": list(smpl["source_time_range_s"]),
        "source_fps": float(smpl["source_fps"]),
        "coordinate_system": (
            "physical cam3 metres; body-local pose; root orientation/translation "
            "expressed in the GVHMR pinhole camera frame"
        ),
    }
    return output, metadata


def write_labels(
    output_root: str | Path,
    session: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    root = Path(output_root).expanduser().resolve()
    path = root / "labels" / f"{session}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    common.json_dump(path.with_suffix(".json"), metadata)
    return path


def load_labels(
    output_root: str | Path, session: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    npz = root / "labels" / f"{session}.npz"
    meta = root / "labels" / f"{session}.json"
    if not npz.is_file() or not meta.is_file():
        raise LabelError(f"prepared labels are missing for {session}: {npz}")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return arrays, json.loads(meta.read_text(encoding="utf-8"))
