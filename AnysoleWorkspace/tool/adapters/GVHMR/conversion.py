"""Fit native GVHMR SMPL-X surfaces to the licensed neutral SMPL model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .config import UPSTREAM_ROOT, required_assets
from .io import sha256_file


class ConversionError(RuntimeError):
    """Raised when the native GVHMR body output cannot be converted to SMPL."""


@dataclass(frozen=True)
class SMPLFit:
    pose: np.ndarray
    betas: np.ndarray
    translation: np.ndarray
    vertices: np.ndarray
    joints: np.ndarray
    vertex_rmse_mm: np.ndarray
    vertex_p95_mm: np.ndarray
    joint_rmse_mm: np.ndarray
    metadata: dict[str, Any]


def pad_native_pose(global_orient: Any, body_pose: Any) -> np.ndarray:
    """Expand GVHMR's 66D root+21-joint pose to SMPL's 72D contract seed.

    The six trailing zeros are only an initialization for the two SMPL body
    joints that GVHMR's native SMPL-X skeleton does not predict. The returned
    vector is not presented as the fitted SMPL result.
    """

    global_value = np.asarray(global_orient, dtype=np.float32)
    body_value = np.asarray(body_pose, dtype=np.float32)
    if global_value.ndim != 2 or global_value.shape[1] != 3:
        raise ConversionError(f"global_orient must be [T,3], got {global_value.shape}")
    if body_value.shape != (global_value.shape[0], 63):
        raise ConversionError(f"body_pose must be [T,63], got {body_value.shape}")
    if not np.isfinite(global_value).all() or not np.isfinite(body_value).all():
        raise ConversionError("native pose contains non-finite values")
    return np.concatenate(
        [global_value, body_value, np.zeros((global_value.shape[0], 6), dtype=np.float32)], axis=1
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def fit_smpl_vertices(
    target_vertices: Any,
    seed_pose: Any,
    seed_betas: Any,
    seed_translation: Any,
    smpl_model: Any,
    *,
    steps: int = 60,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    pose_prior: float = 2.0e-4,
    beta_prior: float = 2.0e-3,
    translation_prior: float = 2.0e-4,
    max_vertex_rmse_mm: float | None = 50.0,
) -> SMPLFit:
    """Fit shared neutral-SMPL shape and per-frame pose/root to target meshes.

    Fitting uses a sequence-shared beta vector, frame-specific root/body pose
    and translation, a robust vertex loss, and weak priors to the native 2024
    pose and shape. `smpl_model` is injectable so solver behavior can be tested
    without redistributing the licensed model assets.
    """

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - target runtime has torch
        raise ConversionError("PyTorch is required for GVHMR-to-SMPL fitting") from exc

    target_np = np.asarray(target_vertices, dtype=np.float32)
    pose_np = np.asarray(seed_pose, dtype=np.float32)
    betas_np = np.asarray(seed_betas, dtype=np.float32)
    translation_np = np.asarray(seed_translation, dtype=np.float32)
    if target_np.ndim != 3 or target_np.shape[-1] != 3 or target_np.shape[1] == 0:
        raise ConversionError(f"target vertices must be [T,V,3], got {target_np.shape}")
    count = target_np.shape[0]
    if pose_np.shape != (count, 72):
        raise ConversionError(f"seed_pose must be [{count},72], got {pose_np.shape}")
    if betas_np.shape == (10,):
        betas_np = np.broadcast_to(betas_np, (count, 10)).copy()
    if betas_np.shape != (count, 10):
        raise ConversionError(f"seed_betas must be [{count},10], got {betas_np.shape}")
    if translation_np.shape != (count, 3):
        raise ConversionError(f"seed_translation must be [{count},3], got {translation_np.shape}")
    if not all(np.isfinite(item).all() for item in (target_np, pose_np, betas_np, translation_np)):
        raise ConversionError("SMPL fitting input contains non-finite values")
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ConversionError("steps must be a non-negative integer")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ConversionError("batch_size must be a positive integer")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ConversionError("learning_rate must be finite and positive")
    if not all(np.isfinite(value) and value >= 0 for value in (pose_prior, beta_prior, translation_prior)):
        raise ConversionError("fit prior weights must be finite and non-negative")

    try:
        parameter = next(smpl_model.parameters())
        device = parameter.device
    except (AttributeError, StopIteration, TypeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    pose_seed = torch.as_tensor(pose_np, dtype=dtype, device=device)
    translation_seed = torch.as_tensor(translation_np, dtype=dtype, device=device)
    betas_seed = torch.as_tensor(betas_np, dtype=dtype, device=device)
    targets = torch.as_tensor(target_np, dtype=dtype, device=device)
    pose_var = torch.nn.Parameter(pose_seed.clone())
    translation_var = torch.nn.Parameter(translation_seed.clone())
    beta_var = torch.nn.Parameter(betas_seed.mean(dim=0, keepdim=True))
    optimizer = torch.optim.Adam([pose_var, translation_var, beta_var], lr=float(learning_rate))

    if steps:
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, count, batch_size):
                end = min(count, start + batch_size)
                output = smpl_model(
                    global_orient=pose_var[start:end, :3],
                    body_pose=pose_var[start:end, 3:],
                    betas=beta_var.expand(end - start, -1),
                    transl=translation_var[start:end],
                )
                predicted = output.vertices
                # A 1 cm Huber transition limits the influence of mapped
                # boundary vertices while preserving useful meter-scale grads.
                vertex_loss = F.smooth_l1_loss(predicted, targets[start:end], beta=0.01)
                frame_weight = (end - start) / count
                prior = (
                    pose_prior * (pose_var[start:end] - pose_seed[start:end]).square().mean()
                    + translation_prior
                    * (translation_var[start:end] - translation_seed[start:end]).square().mean()
                    + beta_prior * (beta_var - betas_seed[start:end].mean(dim=0, keepdim=True)).square().mean()
                )
                (frame_weight * vertex_loss + frame_weight * prior).backward()
            optimizer.step()

    poses: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    rmses: list[np.ndarray] = []
    p95s: list[np.ndarray] = []
    joint_rmses: list[np.ndarray] = []
    regressor = getattr(smpl_model, "J_regressor", None)
    if regressor is None:
        raise ConversionError("neutral SMPL model must expose J_regressor for joint-fit residuals")
    regressor = torch.as_tensor(regressor, dtype=dtype, device=device)[:24]
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            output = smpl_model(
                global_orient=pose_var[start:end, :3],
                body_pose=pose_var[start:end, 3:],
                betas=beta_var.expand(end - start, -1),
                transl=translation_var[start:end],
            )
            mesh = output.vertices
            error = torch.linalg.vector_norm(mesh - targets[start:end], dim=-1)
            joints_value = output.joints[:, :24]
            target_joints = torch.einsum("jv,tvc->tjc", regressor, targets[start:end])
            joint_error = torch.linalg.vector_norm(joints_value - target_joints, dim=-1)
            poses.append(_as_numpy(pose_var[start:end]).astype(np.float32))
            translations.append(_as_numpy(translation_var[start:end]).astype(np.float32))
            betas.append(_as_numpy(beta_var.expand(end - start, -1)).astype(np.float32))
            vertices.append(_as_numpy(mesh).astype(np.float32))
            joints.append(_as_numpy(joints_value).astype(np.float32))
            rmses.append(torch.sqrt(error.square().mean(dim=1)).cpu().numpy().astype(np.float32))
            p95s.append(torch.quantile(error, 0.95, dim=1).cpu().numpy().astype(np.float32))
            joint_rmses.append(torch.sqrt(joint_error.square().mean(dim=1)).cpu().numpy().astype(np.float32))
    pose_result = np.concatenate(poses)
    beta_result = np.concatenate(betas)
    translation_result = np.concatenate(translations)
    vertex_result = np.concatenate(vertices)
    joint_result = np.concatenate(joints)
    rmse_result = np.concatenate(rmses) * 1000.0
    p95_result = np.concatenate(p95s) * 1000.0
    joint_rmse_result = np.concatenate(joint_rmses) * 1000.0
    if not all(np.isfinite(item).all() for item in (pose_result, beta_result, translation_result, rmse_result, p95_result, joint_rmse_result)):
        raise ConversionError("SMPL fitting produced non-finite parameters or residuals")
    if max_vertex_rmse_mm is not None and (
        not np.isfinite(max_vertex_rmse_mm)
        or max_vertex_rmse_mm <= 0
        or float(np.max(rmse_result)) > float(max_vertex_rmse_mm)
    ):
        raise ConversionError(
            f"neutral-SMPL vertex fit failed quality gate: max RMSE {float(np.max(rmse_result)):.3f} mm "
            f"exceeds {max_vertex_rmse_mm!r} mm"
        )
    return SMPLFit(
        pose=pose_result,
        betas=beta_result,
        translation=translation_result,
        vertices=vertex_result,
        joints=joint_result,
        vertex_rmse_mm=rmse_result,
        vertex_p95_mm=p95_result,
        joint_rmse_mm=joint_rmse_result,
        metadata={
            "method": "per-frame pose/translation and sequence-shared neutral-SMPL betas fitted to 2024 SMPL-X-to-SMPL mapped vertices",
            "optimizer": "Adam",
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "pose_prior_weight": float(pose_prior),
            "beta_prior_weight": float(beta_prior),
            "translation_prior_weight": float(translation_prior),
            "max_vertex_rmse_mm": max_vertex_rmse_mm,
            "vertex_fit_quality_gate_passed": True,
            "joint_rmse_mm_mean": float(joint_rmse_result.mean()),
            "joint_rmse_mm_p95": float(np.percentile(joint_rmse_result, 95)),
            "vertex_rmse_mm_mean": float(rmse_result.mean()),
            "vertex_rmse_mm_p95": float(np.percentile(rmse_result, 95)),
            "betas_shared_across_sequence": True,
        },
    )


def map_vertices_to_smpl(smplx_vertices: Any, mapper: Any) -> Any:
    """Apply GVHMR's official 2024 SMPL-X→SMPL sparse correspondence map."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ConversionError("PyTorch is required for the GVHMR vertex mapper") from exc
    vertices = torch.as_tensor(smplx_vertices)
    mapping = mapper.to(device=vertices.device) if torch.is_tensor(mapper) else torch.as_tensor(mapper, device=vertices.device)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or mapping.ndim != 2:
        raise ConversionError(f"invalid mapper inputs: vertices={tuple(vertices.shape)} mapping={tuple(mapping.shape)}")
    if vertices.shape[1] != mapping.shape[1]:
        raise ConversionError(f"mapper expects {mapping.shape[1]} SMPL-X vertices, received {vertices.shape[1]}")
    if mapping.layout == torch.strided:
        converted = torch.einsum("sv,tvc->tsc", mapping, vertices)
    else:
        flat = vertices.permute(1, 0, 2).reshape(vertices.shape[1], -1)
        converted = torch.sparse.mm(mapping, flat).reshape(mapping.shape[0], vertices.shape[0], 3).permute(1, 0, 2)
    return converted


def convert_native_sequence(
    native_params: Mapping[str, Any],
    smplx_model: Any,
    mapper: Any,
    smpl_model: Any,
    *,
    fit_steps: int = 60,
    fit_batch_size: int = 16,
    max_vertex_rmse_mm: float | None = 50.0,
) -> SMPLFit:
    """Generate official mapped vertices, then fit valid neutral-SMPL parameters."""

    required = {"global_orient", "body_pose", "betas", "transl"}
    if not required.issubset(native_params):
        raise ConversionError(f"native GVHMR parameters are missing {sorted(required - set(native_params))}")
    global_orient = _as_numpy(native_params["global_orient"]).astype(np.float32)
    body_pose = _as_numpy(native_params["body_pose"]).astype(np.float32)
    native_betas = _as_numpy(native_params["betas"]).astype(np.float32)
    translation = _as_numpy(native_params["transl"]).astype(np.float32)
    pose_seed = pad_native_pose(global_orient, body_pose)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ConversionError("PyTorch is required for GVHMR body conversion") from exc
    try:
        parameter = next(smplx_model.parameters())
        device = parameter.device
    except (AttributeError, StopIteration, TypeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = {
        "global_orient": torch.as_tensor(global_orient, dtype=torch.float32, device=device),
        "body_pose": torch.as_tensor(body_pose, dtype=torch.float32, device=device),
        "betas": torch.as_tensor(native_betas, dtype=torch.float32, device=device),
        "transl": torch.as_tensor(translation, dtype=torch.float32, device=device),
    }
    smplx_model.eval()
    with torch.no_grad():
        output = smplx_model(**tensors)
        mapped_vertices = map_vertices_to_smpl(output.vertices, mapper)
    fit = fit_smpl_vertices(
        _as_numpy(mapped_vertices), pose_seed, native_betas, translation, smpl_model,
        steps=fit_steps, batch_size=fit_batch_size,
        max_vertex_rmse_mm=max_vertex_rmse_mm,
    )
    return SMPLFit(
        pose=fit.pose,
        betas=fit.betas,
        translation=fit.translation,
        vertices=fit.vertices,
        joints=fit.joints,
        vertex_rmse_mm=fit.vertex_rmse_mm,
        vertex_p95_mm=fit.vertex_p95_mm,
        joint_rmse_mm=fit.joint_rmse_mm,
        metadata={
            **fit.metadata,
            "source_model_family": "GVHMR SIGGRAPH Asia 2024 SMPL-X supermotion neutral",
            "source_pose_parameters": "global_orient [T,3] + body_pose [T,63]; 21 body joints",
            "smpl_conversion": "official sparse SMPL-X-to-SMPL vertex correspondence followed by differentiable neutral-SMPL fit",
            "native_pose_dimensions": 66,
            "smpl_pose_dimensions": 72,
            "source_beta_basis": "SMPL-X; used only as a weak fit initialization/prior",
        },
    )


def load_conversion_models(
    *,
    upstream_root: str | Path = UPSTREAM_ROOT,
    checkpoint_path: str | Path | None = None,
    smplx_model_path: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    mapper_path: str | Path | None = None,
    device: str = "cuda",
) -> tuple[Any, Any, Any, dict[str, str]]:
    """Load licensed GVHMR SMPL-X, official mapper, and neutral SMPL models."""

    upstream = Path(upstream_root).expanduser().resolve()
    from .config import DEFAULT_CHECKPOINT, DEFAULT_SMPL_MODEL, DEFAULT_SMPLX_MODEL, DEFAULT_SMPLX_TO_SMPL

    assets = required_assets(
        checkpoint=checkpoint_path or DEFAULT_CHECKPOINT,
        smplx_model=smplx_model_path or DEFAULT_SMPLX_MODEL,
        smpl_model=smpl_model_path or DEFAULT_SMPL_MODEL,
        mapper=mapper_path or DEFAULT_SMPLX_TO_SMPL,
    )
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    try:
        import smplx
        import torch
        from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX
    except Exception as exc:
        raise ConversionError(f"cannot import GVHMR conversion dependencies from {upstream}: {exc}") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ConversionError(f"CUDA conversion requested but unavailable: {device}")
    try:
        # Match the pinned 2024 make_smplx("supermotion") configuration. That
        # factory hardcodes PROJ_ROOT and cannot accept model_path as an extra
        # kwarg, so construct its own BodyModelSMPLX class with our licensed
        # body-model root instead. smplx.create appends `/smplx` to this root.
        smplx_model = BodyModelSMPLX(
            model_path=assets["smplx_model"].parent.parent,
            model_type="smplx",
            gender="neutral",
            num_pca_comps=12,
            flat_hand_mean=False,
        ).to(device).eval()
        smpl_model = smplx.SMPL(model_path=str(assets["smpl_model"]), gender="neutral").to(device).eval()
        mapper = torch.load(assets["smplx_to_smpl_mapper"], map_location=device, weights_only=False)
        if tuple(mapper.shape) != (6890, 10475):
            raise ConversionError(f"official mapper has unexpected shape {tuple(mapper.shape)}")
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(f"cannot construct GVHMR/SMPL conversion models: {exc}") from exc
    hashes = {
        "checkpoint_sha256": sha256_file(assets["checkpoint"]),
        "smplx_model_sha256": sha256_file(assets["smplx_model"]),
        "smpl_model_sha256": sha256_file(assets["smpl_model"]),
        "smplx_to_smpl_mapper_sha256": sha256_file(assets["smplx_to_smpl_mapper"]),
    }
    return smplx_model, mapper, smpl_model, hashes


__all__ = [
    "ConversionError",
    "SMPLFit",
    "convert_native_sequence",
    "fit_smpl_vertices",
    "load_conversion_models",
    "map_vertices_to_smpl",
    "pad_native_pose",
]
