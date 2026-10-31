"""GT-free E3 G1 adaptation of a frozen V7 model.

The manuscript specifies per-view tokens, a virtual view, L2D/Lcon/Lreg,
and a fixed tactile contact prediction weighting a shared free ground height.
It does not specify virtual-view parameterization. Here its minimal extension
is one additional motion token, initialized by the weighted real-view tokens.
Lcon compares real-view FK joints to that virtual body. The exported virtual
reconstruction is a legal-rotation weighted fusion of refined real views.

V7 predicts training-world (+Y up, metres) motion, not camera-frame HMR.
We retain that coordinate system and fit world-to-camera transforms. No
per-view body rotation, scale, test shape, mocap projection or GT anchor is
an optimizer input. Camera optical depth is constrained positive by a shared
per-camera translation parameterization; there is no added depth loss.

All windows of ONE recording are optimized together: cameras/ground are
shared, root velocity is integrated once on unique frames, and gaps start
explicit independent segments. This is offline adaptation, not online TTA.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from anysole.types import CONFIG_V, CONFIG_VT, FPS, FOOT_JOINTS, N_JOINTS
from anysole.utils.geometry import fk_pose6d, rot6d_to_rotmat, rotmat_to_6d

# Exact anatomical correspondences only. COCO nose/eyes/ears are not SMPL
# head joints, and COCO ankles are not the terminal SMPL foot joints.
COCO17_SMPL24 = ((5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21),
                (11, 1), (12, 2), (13, 4), (14, 5), (15, 7), (16, 8))


@dataclass(frozen=True)
class TrainTemplate:
    offsets: torch.Tensor
    parents: torch.Tensor
    sha256: str
    split: str = "train"
    support_offsets: Optional[torch.Tensor] = None

    def validate(self):
        if self.split != "train" or len(self.sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError("E3 requires an identified frozen train-only template")
        if self.offsets.shape != (N_JOINTS, 3) or self.parents.shape != (N_JOINTS,):
            raise ValueError("train template must contain SMPL24 offsets/parents")
        if not torch.isfinite(self.offsets).all():
            raise ValueError("non-finite train template")
        if self.support_offsets is not None and (
                self.support_offsets.shape != (2, 3) or
                not torch.isfinite(self.support_offsets).all()):
            raise ValueError("support_offsets must be two fixed world-Y-up offsets")


@dataclass(frozen=True)
class E3Observations:
    """Deployment inputs only; strict construction rejects unknown GT fields.

    Features are [views, windows, time, channels]; pressure is one physical
    [windows,time,channels] stream, never duplicated as independent evidence.
    Detected points/confidence are [views,windows,time,keypoints,(xy)].
    frame_ids/times identify the original recording grid, including overlap.
    Intrinsics are declared deployment-known. No extrinsics are accepted.
    """
    V_feat: torch.Tensor
    frame_ids: torch.Tensor
    detected_2d: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor
    joint_indices: torch.Tensor
    visual_condition: str = "clean"
    detection_condition: str = "clean"
    detector_source: str = "image_detection"
    T_raw: Optional[torch.Tensor] = None
    T_phys: Optional[torch.Tensor] = None
    tactile_source: Optional[str] = None
    frame_times: Optional[torch.Tensor] = None
    view_weights: Optional[torch.Tensor] = None

    def validate(self):
        if self.V_feat.ndim != 4 or self.V_feat.shape[0] < 1:
            raise ValueError("V_feat must be [views,windows,time,channels]")
        views, windows, time, _ = self.V_feat.shape
        if windows < 1 or time < 1 or self.frame_ids.shape != (windows, time):
            raise ValueError("complete real windows and matching frame_ids are required")
        if self.frame_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("frame_ids must be integer source-frame identities")
        if (self.frame_ids < 0).any() or (self.frame_ids[:, 1:] - self.frame_ids[:, :-1] != 1).any():
            raise ValueError("windows must be consecutive real frames; padding is unsupported")
        if self.frame_times is not None and self.frame_times.shape != (windows, time):
            raise ValueError("frame_times must match physical windows")
        points = len(self.joint_indices)
        if self.joint_indices.ndim != 1 or self.joint_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("joint_indices must be a detector-to-SMPL24 integer map")
        if ((self.joint_indices < 0) | (self.joint_indices >= N_JOINTS)).any():
            raise ValueError("detector joint mapping outside SMPL24")
        if self.detected_2d.shape != (views, windows, time, points, 2) or self.confidence.shape != (views, windows, time, points):
            raise ValueError("detected_2d/confidence shapes do not match")
        if self.intrinsics.shape != (views, 3, 3) or not torch.isfinite(self.intrinsics).all():
            raise ValueError("deployment-known intrinsics must be [views,3,3]")
        if (self.intrinsics[:, (0, 1), (0, 1)] <= 0).any():
            raise ValueError("camera focal lengths must be positive")
        if not torch.allclose(self.intrinsics[:, 2], self.intrinsics.new_tensor([0., 0, 1]).expand(views, -1)):
            raise ValueError("intrinsics must use the standard optical homogeneous convention")
        if not torch.isfinite(self.confidence).all() or ((self.confidence < 0) | (self.confidence > 1)).any():
            raise ValueError("detector confidence must lie in [0,1]")
        if self.visual_condition not in ("clean", "occ010", "occ030") or self.detection_condition != self.visual_condition:
            raise ValueError("features and detections must use the same declared degradation")
        if self.detector_source != "image_detection":
            raise ValueError("only same-image detector points are allowed, never mocap projections")
        if not torch.isfinite(self.V_feat).all():
            raise ValueError("non-finite visual features")
        if (self.T_raw is None) != (self.T_phys is None):
            raise ValueError("physical tactile input requires both T_raw and T_phys")
        if self.T_raw is not None:
            if not self.tactile_source:
                raise ValueError("actual tactile input requires explicit provenance")
            for value in (self.T_raw, self.T_phys):
                if value.ndim != 3 or value.shape[:2] != (windows, time) or not torch.isfinite(value).all():
                    raise ValueError("tactile data must be one finite physical window stream")
        if self.view_weights is not None and (
                self.view_weights.shape != (views,) or not torch.isfinite(self.view_weights).all() or
                (self.view_weights < 0).any() or self.view_weights.sum() <= 0):
            raise ValueError("view weights must be fixed nonnegative evidence weights")


@dataclass(frozen=True)
class TTAConfig:
    steps: int = 30
    lr: float = 5e-3
    lambda_2d: float = 1.
    lambda_con: float = 1.
    lambda_reg: float = 1.
    lambda_contact: float = 1.
    use_contact: bool = False
    initialization: str = "V"
    recipe: str = "U0"
    path: str = "C0"
    fps: float = FPS
    min_depth: float = .05

    def validate(self, observations):
        if self.recipe not in ("U0", "UP_G1", "UC_G1"):
            raise ValueError("G1 recipe must be U0, UP_G1 or UC_G1")
        if self.path not in ("direct", "0step", "C0", "C_original"):
            raise ValueError("G1 path must be direct, 0step, C0 or C_original")
        if self.steps < 0 or self.lr <= 0 or self.fps <= 0 or self.min_depth <= 0:
            raise ValueError("invalid optimization budget")
        if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in (
                self.lambda_2d, self.lambda_con, self.lambda_reg, self.lambda_contact)):
            raise ValueError("loss weights must be finite and nonnegative")
        if self.path in ("direct", "0step") and self.steps != 0:
            raise ValueError("direct/0step require exactly zero optimization steps")
        if self.recipe == "UP_G1" and (self.path != "direct" or self.steps != 0 or self.use_contact):
            raise ValueError("UP is direct only: no TTA optimizer may be constructed")
        if self.initialization not in ("V", "VT"):
            raise ValueError("initialization must be V or VT")
        if self.initialization == "VT" and observations.T_raw is None:
            raise ValueError("VT initialization requires actual test T")
        if self.use_contact and (self.path != "C_original" or observations.T_raw is None):
            raise ValueError("C_original requires actual T and its own declared path")


@contextmanager
def _frozen(model):
    parameters = list(model.parameters())
    required = [p.requires_grad for p in parameters]
    training = [(module, module.training) for module in model.modules()]
    model.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, value in zip(parameters, required):
            parameter.requires_grad_(value)
        for module, value in training:
            module.training = value


def _proper_rotations(d6):
    """Gram-Schmidt with a deterministic fallback at degenerate fusion sums."""
    a = d6[..., :3]
    identity_x = torch.zeros_like(a)
    identity_x[..., 0] = 1
    a = torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, identity_x)
    a = F.normalize(a, dim=-1)
    b = d6[..., 3:] - (d6[..., 3:] * a).sum(-1, keepdim=True) * a
    axis = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    fallback = axis - (axis * a).sum(-1, keepdim=True) * a
    b = torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fallback)
    b = F.normalize(b, dim=-1)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), dim=-1)


def _timeline(observations, fps):
    ids, inverse, counts = torch.unique(observations.frame_ids.flatten(), sorted=True,
                                       return_inverse=True, return_counts=True)
    if observations.frame_times is None:
        times = ids.to(torch.float64) / fps
    else:
        flat = observations.frame_times.flatten().to(device=observations.V_feat.device, dtype=torch.float64)
        times = flat.new_zeros(len(ids)).index_add(0, inverse, flat) / counts
        if not torch.isfinite(flat).all() or not torch.allclose(times[inverse], flat, atol=1e-6, rtol=0):
            raise ValueError("overlapping source-frame timestamps disagree")
    if (times[1:] <= times[:-1]).any():
        raise ValueError("source times must be strictly increasing")
    starts = torch.cat([torch.ones(1, device=ids.device, dtype=torch.bool), ids[1:] != ids[:-1] + 1])
    dt = torch.cat([times.new_tensor([1. / fps]), times[1:] - times[:-1]])
    dt = torch.where(starts, dt.new_tensor(1. / fps), dt)
    return ids, inverse, counts, dt.to(observations.V_feat), starts


def _continuous_motion(out, inverse, counts, dt, starts):
    """Fold overlapping rotations/velocities and integrate each source frame once."""
    pose = out["x0_hat"].reshape(-1, N_JOINTS, 6)
    pose = rotmat_to_6d(_proper_rotations(pose))
    pose = pose.new_zeros(len(counts), N_JOINTS, 6).index_add(0, inverse, pose) / counts[:, None, None]
    pose = rotmat_to_6d(_proper_rotations(pose)).flatten(-2)
    velocity = out["v_hat"].reshape(-1, 3)
    velocity = velocity.new_zeros(len(counts), 3).index_add(0, inverse, velocity) / counts[:, None]
    raw = torch.cumsum(velocity * dt[:, None], dim=0)
    # Gaps have no observations to integrate. Reset the local origin for each
    # explicit segment instead of reading GT or inventing motion in the gap.
    segment = starts.long().cumsum(0) - 1
    previous = torch.cat([raw.new_zeros(1, 3), raw[:-1]])[starts]
    trans = raw - previous[segment]
    return {"pose": pose, "velocity": velocity, "trans": trans}


def _fuse(motions, weights):
    poses = torch.stack([m["pose"] for m in motions]).reshape(len(motions), -1, N_JOINTS, 6)
    pose = (poses * weights[:, None, None, None]).sum(0)
    return {"pose": rotmat_to_6d(_proper_rotations(pose)).flatten(-2),
            "velocity": (torch.stack([m["velocity"] for m in motions]) * weights[:, None, None]).sum(0),
            "trans": (torch.stack([m["trans"] for m in motions]) * weights[:, None, None]).sum(0)}


def _joints(motion, template):
    return fk_pose6d(motion["pose"][None], motion["trans"][None],
                     template.offsets, template.parents)[0]


def _camera_transform(joints, rotation_6d, xy, depth, min_depth):
    rotations = _proper_rotations(rotation_6d)
    rotated = torch.einsum("vij,vfkj->vfki", rotations, joints)
    # One translation per recording/camera. Reparameterizing its depth using
    # the minimum of ALL frame/joint depths enforces positive optical depth
    # without a frame-dependent camera or an extra objective term.
    tz = F.softplus(depth) + min_depth - rotated[..., 2].amin(dim=(1, 2))
    translation = torch.cat([xy, tz[:, None]], dim=-1)
    return rotated + translation[:, None, None], rotations, translation


def _initialize_cameras(real_joints, observations, inverse, min_depth):
    """Image-only PnP from the frozen model's initial body, never GT cameras.

    A bounded deterministic point sample initializes static cameras. Empty or
    underdetermined detections retain the declared identity/depth fallback.
    PnP supplies only a starting point; all cameras still enter the same TTA.
    """
    rotations = real_joints.new_tensor([1., 0, 0, 0, 1, 0]).repeat(len(real_joints), 1)
    xy = real_joints.new_zeros(len(real_joints), 2)
    depth = real_joints.new_full((len(real_joints),), 1.)
    methods = ["identity_positive_depth_fallback"] * len(real_joints)
    if len(observations.joint_indices) == 0:
        return rotations, xy, depth, methods
    import cv2
    import numpy as np
    for view in range(len(real_joints)):
        xyz = real_joints[view, inverse][:, observations.joint_indices].reshape(-1, 3).detach().cpu().double().numpy()
        uv = observations.detected_2d[view].reshape(-1, 2).detach().cpu().double().numpy()
        conf = observations.confidence[view].flatten().detach().cpu().numpy()
        valid = (conf > .05) & np.isfinite(uv).all(-1) & np.isfinite(xyz).all(-1)
        selected = np.flatnonzero(valid)
        if len(selected) < 6 or np.linalg.matrix_rank(xyz[selected] - xyz[selected].mean(0)) < 3:
            continue
        if len(selected) > 256:
            selected = selected[np.linspace(0, len(selected) - 1, 256).astype(int)]
        k = observations.intrinsics[view].detach().cpu().double().numpy()
        try:
            success, rvec, tvec = cv2.solvePnP(xyz[selected], uv[selected], k, None, flags=cv2.SOLVEPNP_EPNP)
            if not success:
                continue
            success, rvec, tvec = cv2.solvePnP(xyz[selected], uv[selected], k, None, rvec, tvec,
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            rotation = cv2.Rodrigues(rvec)[0]
            translation = tvec.reshape(3)
            if not success or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue
            r = real_joints.new_tensor(rotation)
            t = real_joints.new_tensor(translation)
            minimum = torch.einsum("ij,fkj->fki", r, real_joints[view])[..., 2].amin()
            gap = (t[2] + minimum - min_depth).clamp_min(1e-4)
            rotations[view] = rotmat_to_6d(r)
            xy[view] = t[:2]
            depth[view] = gap + torch.log(-torch.expm1(-gap))
            methods[view] = "image_detection_model_initial_body_PnP"
        except cv2.error:
            continue
    return rotations, xy, depth, methods


def adapt_multiview(model, observations: E3Observations, template: TrainTemplate,
                    config: TTAConfig = TTAConfig()):
    """Return unique-frame continuous motion without accessing scoring targets.

    C0 and contact-disabled C_original execute the identical optimizer loop.
    Contact is predicted from the actual T stream ONCE before that loop and
    detached. G1 uses its raw squared height sum divided only by the fixed
    count of unique physical frames (lambda_raw=lambda_contact/frame_count).
    It never normalizes by confidence mass, scale, or number of views.
    """
    observations.validate()
    template.validate()
    config.validate(observations)
    device = observations.V_feat.device
    dtype = observations.V_feat.dtype
    template = TrainTemplate(template.offsets.detach().to(device=device, dtype=dtype),
                             template.parents.detach().to(device=device), template.sha256,
                             template.split, None if template.support_offsets is None else
                             template.support_offsets.detach().to(device=device, dtype=dtype))
    views, windows, time, _ = observations.V_feat.shape
    ids, inverse, counts, dt, starts = _timeline(observations, config.fps)
    weights = observations.V_feat.new_ones(views) if observations.view_weights is None else observations.view_weights.detach().to(observations.V_feat)
    weights = weights / weights.sum()
    actual_t = observations.T_raw is not None
    # V7 FootConv raw108 schema. Missing T is explicitly absent, not a contact
    # prediction source; its null head is never used for contact adaptation.
    raw = observations.V_feat.new_zeros(windows, time, 96) if not actual_t else observations.T_raw
    phys = observations.V_feat.new_zeros(windows, time, 12) if not actual_t else observations.T_phys
    cid = torch.full((windows,), CONFIG_V if config.initialization == "V" else CONFIG_VT,
                     device=device, dtype=torch.long)
    trace = []
    with torch.inference_mode(False), _frozen(model), torch.enable_grad():
        with torch.no_grad():
            outputs = [model(observations.V_feat[v], raw if config.initialization == "VT" else torch.zeros_like(raw),
                             phys if config.initialization == "VT" else torch.zeros_like(phys), cid)
                       for v in range(views)]
            z0 = torch.stack([out["z"] for out in outputs]).detach().clone()
            virtual0 = (z0 * weights[:, None, None, None, None]).sum(0)
            contact = None
            if config.use_contact:
                # The original tactile head is view-independent. One forward
                # with ACTUAL T is equivalent to averaging duplicated views.
                t_cid = torch.full_like(cid, CONFIG_VT)
                tactile_out = outputs[0] if config.initialization == "VT" else model(
                    observations.V_feat[0], raw, phys, t_cid)
                c = tactile_out["contact_logits"].sigmoid().reshape(-1, 2)
                contact = c.new_zeros(len(ids), 2).index_add(0, inverse, c) / counts[:, None]
                contact = contact.detach()
        z = torch.nn.Parameter(z0.clone())
        virtual_z = torch.nn.Parameter(virtual0.clone())
        rotation_6d = torch.nn.Parameter(observations.V_feat.new_tensor([1., 0, 0, 0, 1, 0]).repeat(views, 1))
        xy = torch.nn.Parameter(observations.V_feat.new_zeros(views, 2))
        depth = torch.nn.Parameter(observations.V_feat.new_full((views,), 1.))

        def decode():
            real = [_continuous_motion(model.decode_z(z[v]), inverse, counts, dt, starts) for v in range(views)]
            virtual = _continuous_motion(model.decode_z(virtual_z), inverse, counts, dt, starts)
            return real, virtual

        with torch.no_grad():
            real, virtual = decode()
            initial = _fuse(real, weights)
            camera_initialization = ["unused_for_zero_step_prediction"] * views
            if config.steps:
                r0, xy0, d0, camera_initialization = _initialize_cameras(
                    torch.stack([_joints(m, template) for m in real]), observations, inverse, config.min_depth)
                rotation_6d.copy_(r0)
                xy.copy_(xy0)
                depth.copy_(d0)
            feet0 = _joints(virtual, template)[:, FOOT_JOINTS, 1]
            if template.support_offsets is not None:
                feet0 = feet0 + template.support_offsets[:, 1]
            ground0 = feet0.flatten().median()
        ground = torch.nn.Parameter(ground0.clone())
        # Deliberately no optimizer at all for UP/direct/0step.
        optimizer = None if config.steps == 0 else torch.optim.Adam(
            [z, virtual_z, rotation_6d, xy, depth, ground], lr=config.lr)
        detected = observations.detected_2d.reshape(views, windows * time, len(observations.joint_indices), 2)
        confidence = observations.confidence.reshape(views, windows * time, len(observations.joint_indices)).detach()
        confidence = confidence * torch.isfinite(detected).all(-1)
        confidence = confidence / counts[inverse][None, :, None]
        detected = torch.nan_to_num(detected, nan=0., posinf=0., neginf=0.).detach()
        # Normalize image residuals by declared focal lengths. This changes
        # coordinate units only, not the contact formula or a fitted scale.
        focal = observations.intrinsics[:, (0, 1), (0, 1)]
        gradient_norms = {}
        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            real, virtual = decode()
            real_joints = torch.stack([_joints(m, template) for m in real])
            virtual_joints = _joints(virtual, template)
            camera, _, _ = _camera_transform(real_joints, rotation_6d, xy, depth, config.min_depth)
            selected = camera[:, inverse][:, :, observations.joint_indices]
            homogeneous = torch.einsum("vij,vfkj->vfki", observations.intrinsics, selected)
            pixels = homogeneous[..., :2] / homogeneous[..., 2:]
            residual = (pixels - detected) / focal[:, None, None]
            l2d = (residual.square().sum(-1) * confidence).sum() / confidence.sum().clamp_min(1)
            lcon = ((real_joints - virtual_joints[None]).square().mean(dim=(1, 2, 3)) * weights).sum()
            lreg = (z - z0).square().mean() + (virtual_z - virtual0).square().mean()
            raw_contact = z.sum() * 0
            if contact is not None:
                foot_y = virtual_joints[:, FOOT_JOINTS, 1]
                if template.support_offsets is not None:
                    foot_y = foot_y + template.support_offsets[:, 1]
                raw_contact = (contact * (foot_y - ground).square()).sum()
            lcontact = raw_contact / len(ids)
            loss = config.lambda_2d * l2d + config.lambda_con * lcon + config.lambda_reg * lreg + config.lambda_contact * lcontact
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite G1 TTA objective")
            loss.backward()
            gradient_norms = {}
            for name, parameter in (("z", z), ("virtual_z", virtual_z), ("camera_rotation", rotation_6d),
                                    ("camera_xy", xy), ("camera_depth", depth), ("ground", ground)):
                grad = parameter.grad
                if grad is not None and not torch.isfinite(grad).all():
                    raise FloatingPointError("non-finite TTA gradient: " + name)
                gradient_norms[name] = 0. if grad is None else float(grad.norm().detach())
            optimizer.step()
            trace.append({"step": step + 1, "loss": float(loss.detach()), "L2D": float(l2d.detach()),
                          "Lcon": float(lcon.detach()), "Lreg": float(lreg.detach()),
                          "Lcontact_raw_sum": float(raw_contact.detach()), "Lcontact": float(lcontact.detach()),
                          "ground": float(ground.detach()), "gradient_norms": gradient_norms})
        with torch.no_grad():
            real, virtual = decode()
            final = initial if config.steps == 0 else _fuse(real, weights)
            final_joints = _joints(final, template)
            virtual_joints = _joints(virtual, template)
            camera, rotations, translation = _camera_transform(torch.stack([_joints(m, template) for m in real]),
                                                               rotation_6d, xy, depth, config.min_depth)
    return {**{key: value.detach() for key, value in final.items()}, "joints": final_joints.detach(),
            "frame_ids": ids.detach(), "segment_ids": (starts.long().cumsum(0) - 1).detach(),
            "initial_pose": initial["pose"].detach(), "initial_trans": initial["trans"].detach(),
            "camera_rotation": rotations.detach(), "camera_translation": translation.detach(),
            "minimum_depth": camera[..., 2].amin().detach(), "ground": ground.detach(),
            "initial_ground": ground0.detach(), "contact_weights": contact,
            "camera_initialization": camera_initialization,
            "virtual_joints": virtual_joints.detach(),
            "fusion": "weighted_real_view_SO3_motion_fusion; virtual_contact_transfers_through_Lcon",
            "contact_reduction": "raw_sum / unique_physical_frame_count",
            "lambda_contact_raw": config.lambda_contact / len(ids),
            "lambda_contact_config": config.lambda_contact,
            "trace": trace, "optimizer_steps": len(trace), "template_sha256": template.sha256,
            "coordinates": "training_world_Y_up_metres; no GT anchors; no fitted body scale",
            "contact_source": observations.tactile_source if config.use_contact else None,
            "support_definition": "SMPL24_foot_joints_10_11" if template.support_offsets is None else
                                  "SMPL24_foot_joints_10_11_plus_fixed_train_world_offsets"}


@dataclass(frozen=True)
class CalibratedReferenceObservations:
    """Separate calibrated/oracle reference API; never accepted by normal TTA.

    Frames are unique, full-image detected points are undistorted, and all
    translations use training-world metres. Extrinsics must be audited to
    transform the training-world SMPL frame into optical camera coordinates.
    """
    detected_2d: torch.Tensor  # [views, frames, keypoints, 2]
    confidence: torch.Tensor  # [views, frames, keypoints]
    intrinsics: torch.Tensor
    rotation: torch.Tensor  # [views,3,3], world -> camera
    translation: torch.Tensor  # [views,3], metres
    joint_indices: torch.Tensor
    frame_ids: torch.Tensor
    calibration_source: str
    visual_condition: str = "clean"
    detection_condition: str = "clean"
    detector_source: str = "image_detection"

    def validate(self):
        if self.detected_2d.ndim != 4 or self.detected_2d.shape[0] < 2:
            raise ValueError("calibrated triangulation needs at least two real views")
        views, frames, points, xy = self.detected_2d.shape
        if xy != 2 or self.confidence.shape != (views, frames, points) or self.frame_ids.shape != (frames,):
            raise ValueError("invalid calibrated reference observation shapes")
        if (self.frame_ids[1:] <= self.frame_ids[:-1]).any() or frames == 0:
            raise ValueError("reference frame identities must be unique and sorted")
        if self.joint_indices.shape != (points,) or ((self.joint_indices < 0) | (self.joint_indices >= N_JOINTS)).any():
            raise ValueError("explicit detector-to-SMPL mapping required")
        if not self.calibration_source or self.rotation.shape != (views, 3, 3) or self.translation.shape != (views, 3):
            raise ValueError("explicit calibrated/oracle extrinsics provenance is required")
        eye = torch.eye(3, device=self.rotation.device, dtype=self.rotation.dtype).expand(views, -1, -1)
        if not torch.allclose(self.rotation.transpose(-1, -2) @ self.rotation, eye, atol=1e-4, rtol=1e-4) or not torch.allclose(
                torch.det(self.rotation), eye[:, 0, 0], atol=1e-4, rtol=1e-4):
            raise ValueError("calibrated camera rotations must lie in SO(3)")
        if self.intrinsics.shape != (views, 3, 3) or not torch.isfinite(self.intrinsics).all() or not torch.isfinite(self.translation).all():
            raise ValueError("invalid calibrated camera matrices")
        if not torch.isfinite(self.confidence).all() or ((self.confidence < 0) | (self.confidence > 1)).any():
            raise ValueError("invalid reference detector confidence")
        if self.visual_condition not in ("clean", "occ010", "occ030") or self.detection_condition != self.visual_condition or self.detector_source != "image_detection":
            raise ValueError("reference must use same-image, same-degradation detections")


def triangulate_calibrated_reference(observations: CalibratedReferenceObservations):
    """Confidence-weighted homogeneous DLT; no learned or mocap 3D targets."""
    observations.validate()
    uv = torch.nan_to_num(observations.detected_2d, nan=0., posinf=0., neginf=0.).detach()
    confidence = observations.confidence.detach() * torch.isfinite(observations.detected_2d).all(-1)
    pixel = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
    rays = torch.einsum("vij,vfkj->vfki", torch.linalg.inv(observations.intrinsics), pixel)
    xy = rays[..., :2] / rays[..., 2:]
    projection = torch.cat([observations.rotation, observations.translation[..., None]], dim=-1)
    a0 = xy[..., 0, None] * projection[:, None, None, 2] - projection[:, None, None, 0]
    a1 = xy[..., 1, None] * projection[:, None, None, 2] - projection[:, None, None, 1]
    # Sqrt confidence corresponds to confidence-weighted squared residuals.
    matrix = torch.stack([a0, a1], dim=-2) * confidence.sqrt()[..., None, None]
    matrix = matrix.permute(1, 2, 0, 3, 4).flatten(2, 3)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    homogeneous = vh[..., -1, :]
    denominator = homogeneous[..., 3:]
    finite_w = denominator[..., 0].abs() > 1e-8
    points = homogeneous[..., :3] / torch.where(finite_w[..., None], denominator, torch.ones_like(denominator))
    camera_z = torch.einsum("vij,fkj->vfki", observations.rotation, points)[..., 2] + observations.translation[:, None, None, 2]
    positive = ((camera_z > 0) | (confidence <= 0)).all(0)
    valid = (confidence > 0).sum(0) >= 2
    valid = valid & finite_w & positive & torch.isfinite(points).all(-1) & (singular[..., -2] > 1e-8)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return {"points": points.detach(), "valid": valid.detach(),
            "weights": (confidence.mean(0) * valid).detach(),
            "method": "calibrated_confidence_weighted_DLT; reference_only"}


def fit_calibrated_reference(observations: CalibratedReferenceObservations,
                             template: TrainTemplate, *, steps=100, lr=.03,
                             lambda_initial_pose=1e-3):
    """DLT + fixed-template SMPL24 FK fit, initialized from canonical rest pose.

    This is an independent pose-only calibrated reference, not EasyMocap.
    Shape stays at the train mean. Missing joints retain a small fixed rest
    pose prior. No AnySole checkpoint or test GT motion/shape enters the fit.
    Hyperparameters must be frozen on train/val before test evaluation.
    """
    template.validate()
    if steps < 0 or lr <= 0 or lambda_initial_pose < 0:
        raise ValueError("invalid calibrated fitting budget")
    triangulated = triangulate_calibrated_reference(observations)
    target, valid, weights = triangulated["points"], triangulated["valid"], triangulated["weights"]
    if not valid.any():
        raise ValueError("no triangulable calibrated reference observations")
    # Frames with fewer than three triangulated joints cannot support a body
    # fit. Preserve explicit coverage instead of inventing a scored body.
    supported = valid.sum(-1) >= 3
    if not supported.any():
        raise ValueError("fewer than three triangulated joints per supported frame")
    device, dtype = target.device, target.dtype
    template = TrainTemplate(template.offsets.to(device=device, dtype=dtype), template.parents.to(device),
                             template.sha256, template.split, template.support_offsets)
    initial_pose = target.new_tensor([1., 0, 0, 0, 1, 0]).repeat(1, len(target), N_JOINTS)
    # Mid-hip initializes translation when both hips were detected; otherwise
    # a confidence-weighted visible-joint centroid is the declared fallback.
    translation0 = (target * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1e-8)[:, None]
    hip_indices = [torch.where(observations.joint_indices == j)[0] for j in (1, 2)]
    if all(len(i) == 1 for i in hip_indices):
        left, right = (int(i[0]) for i in hip_indices)
        hips_valid = valid[:, left] & valid[:, right]
        translation0 = torch.where(hips_valid[:, None], (target[:, left] + target[:, right]) * .5, translation0)
    pose = torch.nn.Parameter(initial_pose.clone())
    translation = torch.nn.Parameter(translation0[None].clone())
    weights = weights * supported[:, None]
    trace = []
    with torch.inference_mode(False), torch.enable_grad():
        optimizer = torch.optim.Adam([pose, translation], lr=lr) if steps else None
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            legal_pose = rotmat_to_6d(_proper_rotations(pose.reshape(1, -1, N_JOINTS, 6))).flatten(-2)
            points = fk_pose6d(legal_pose, translation, template.offsets, template.parents)[0]
            error = (points[:, observations.joint_indices] - target).square().sum(-1)
            data_loss = (error * weights).sum() / weights.sum().clamp_min(1)
            prior = (pose - initial_pose).square().mean()
            loss = data_loss + lambda_initial_pose * prior
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite calibrated FK fitting loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in (pose, translation)):
                raise FloatingPointError("non-finite calibrated FK fitting gradient")
            optimizer.step()
            trace.append({"step": step + 1, "loss": float(loss.detach()), "triangulation_fit": float(data_loss.detach())})
        with torch.no_grad():
            legal_pose = rotmat_to_6d(_proper_rotations(pose.reshape(1, -1, N_JOINTS, 6))).flatten(-2)
            joints = fk_pose6d(legal_pose, translation, template.offsets, template.parents)[0]
    frame_ids = observations.frame_ids[supported]
    starts = torch.cat([torch.ones(1, dtype=torch.bool, device=device), frame_ids[1:] != frame_ids[:-1] + 1])
    return {"pose": legal_pose[0, supported].detach(), "trans": translation[0, supported].detach(),
            "joints": joints[supported].detach(), "frame_ids": frame_ids.detach(),
            "segment_ids": starts.long().cumsum(0) - 1,
            "triangulated_points": target, "triangulation_valid": valid,
            "trace": trace, "optimizer_steps": len(trace), "template_sha256": template.sha256,
            "method": "calibrated_DLT_fixed_train_template_SMPL24_FK_fit; not_EasyMocap",
            "calibration_source": observations.calibration_source,
            "initialization": "canonical_rest_pose; triangulated_midhip_or_visible_joint_centroid",
            "coverage": {"observed_frames": len(target), "supported_frames": int(supported.sum())}}
