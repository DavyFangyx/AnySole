"""SMPL full-mesh forward and pyrender mesh renderer for results_display.

The skinning is ported from the ReferenceWorks SMPL forwards (MDM
``model/rotation2xyz.py`` / PhysPT ``models/smpl_phys.py``): the canonical
SMPL formulation in pure numpy, straight from ``SMPL_NEUTRAL.pkl``::

    v_shaped = v_template + shapedirs @ beta
    J        = J_regressor @ v_shaped
    v_posed  = v_shaped + posedirs @ lrotmin(pose_body)
    v_world  = sum_j w_j * (R_glob_j (v_posed - J_j) + J_glob_j) + trans

with R_glob/J_glob accumulated along the SMPL parent chain.  ``trans`` here
is AnySole's pelvis-rooted translation (``trans_m`` from
``anysole.data.smpl_io.load_smpl``), so the rest pelvis J_0 is subtracted.
Validated against an independent 4x4-transform LBS reference (max diff
4.4e-16 m).

The rendering follows the original ReferenceWorks pipeline (CLIFF
``common/renderer_pyrd.py`` / pressure_tookit ``render_utils.py``): pyrender
EGL offscreen renderer, MetallicRoughness material, two camera-relative
directional lights, pinhole camera.  The camera framing (distance/focal from
the body radius, elev/azim view angle) and the skeleton overlay are the
results_display adaptation; the scene/material/light/render call itself is
unchanged from the source.

Coordinates: vertices are computed in native SMPL (x right, y up, z forward,
metres).  Callers convert with ``utils.motion_io.smpl_yup_to_display`` to the
renderer's z-up convention before drawing.
"""
from __future__ import annotations

import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # utils/ -> script/ -> gait repo root
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from anysole.data.smpl_io import load_smpl, smpl24_pose6d_to_poses  # noqa: E402
from anysole.types import JOINT_PARENTS, SMPL_MODEL_PATH  # noqa: E402
from utils.motion_io import smpl_yup_to_display  # noqa: E402

N_VERTS = 6890
_RENDERERS = {}  # (width, height) -> cached pyrender.OffscreenRenderer


@lru_cache(maxsize=1)
def _body_model() -> dict:
    """Extract the skinning tensors AnySole's SMPL_NEUTRAL.pkl (cached)."""
    model_path = Path(os.environ.get("ANYSOLE_SMPL_MODEL", str(SMPL_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL_NEUTRAL.pkl not found: {model_path}")
    with open(model_path, "rb") as handle:
        model = pickle.load(handle, encoding="latin1")

    def dense(key: str) -> np.ndarray:
        value = model[key]
        return np.asarray(value.toarray(), dtype=np.float64) if hasattr(value, "toarray") else np.asarray(value, dtype=np.float64)

    model = {
        "v_template": dense("v_template"),
        "shapedirs": dense("shapedirs"),
        "posedirs": dense("posedirs"),
        "weights": dense("weights"),
        "regressor": dense("J_regressor"),
        "faces": np.asarray(model["f"], dtype=np.int64),
    }
    if model["v_template"].shape != (N_VERTS, 3) or model["weights"].shape != (N_VERTS, 24):
        raise ValueError(f"unexpected SMPL model shapes: v_template {model['v_template'].shape}, weights {model['weights'].shape}")
    if model["faces"].shape[1] != 3 or model["faces"].max() >= N_VERTS:
        raise ValueError(f"unexpected SMPL face table shape: {model['faces'].shape}")
    return model


def smpl_faces() -> np.ndarray:
    """SMPL face table (13776, 3)."""
    return _body_model()["faces"]


def smpl_vertices(pose_rotvec: np.ndarray, trans_m: np.ndarray, betas: np.ndarray,
                  joint_rest_m: np.ndarray) -> np.ndarray:
    """Skin the full SMPL mesh for a sequence of poses.

    Args:
        pose_rotvec: (T, 24, 3) axis-angle joint rotations (SMPL order).
        trans_m: (T, 3) pelvis-rooted world translation, native y-up metres
            (AnySole ``trans_m`` = model ``trans`` + rest pelvis J_0).
        betas: (<=10,) shape coefficients.
        joint_rest_m: (24, 3) rest joint positions for these betas
            (``load_smpl`` returns it as ``joint_rest_m``).

    Returns:
        (T, 6890, 3) world vertices, native SMPL y-up metres.
    """
    pose = np.asarray(pose_rotvec, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (24, 3):
        raise ValueError(f"pose_rotvec must be (T,24,3), got {pose.shape}")
    trans = np.asarray(trans_m, dtype=np.float64).reshape(-1, 3)
    if trans.shape[0] != pose.shape[0]:
        raise ValueError("pose and trans frame counts differ")
    rest = np.asarray(joint_rest_m, dtype=np.float64)
    if rest.shape != (24, 3):
        raise ValueError(f"joint_rest_m must be (24,3), got {rest.shape}")

    body = _body_model()
    n_shape = min(10, body["shapedirs"].shape[-1])
    beta = np.zeros(n_shape, dtype=np.float64)
    beta[: min(n_shape, np.asarray(betas).reshape(-1).size)] = np.asarray(betas, dtype=np.float64).reshape(-1)[:n_shape]
    v_shaped = body["v_template"] + np.einsum("vkc,c->vk", body["shapedirs"][:, :, :n_shape], beta)

    rotmats = SciRotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    # Pose blend shapes: 23 body joints' (R - I) flattened -> 207 coefficients.
    lrotmin = (rotmats[:, 1:] - np.eye(3, dtype=np.float64)).reshape(pose.shape[0], 207)
    v_posed = v_shaped[None] + np.einsum("vck,tk->tvc", body["posedirs"], lrotmin)

    # Global rotations and joint positions along the SMPL parent chain.
    parents = tuple(JOINT_PARENTS)
    r_glob = np.empty_like(rotmats)
    j_glob = np.empty((pose.shape[0], 24, 3), dtype=np.float64)
    for t in range(pose.shape[0]):
        r_glob[t, 0] = rotmats[t, 0]
        j_glob[t, 0] = rest[0]
        for joint in range(1, 24):
            parent = parents[joint]
            r_glob[t, joint] = r_glob[t, parent] @ rotmats[t, joint]
            j_glob[t, joint] = r_glob[t, parent] @ (rest[joint] - rest[parent]) + j_glob[t, parent]

    # LBS: vertices live in model-origin space (pelvis rest J_0 at origin);
    # AnySole trans_m is pelvis-rooted, so J_0 is subtracted when translating.
    verts = np.empty((pose.shape[0], N_VERTS, 3), dtype=np.float64)
    for t in range(pose.shape[0]):
        rel = v_posed[t][None] - rest[:, None, :]  # (24, 6890, 3)
        world = np.einsum("jab,jvb->jva", r_glob[t], rel) + j_glob[t, :, None, :]
        verts[t] = np.einsum("vj,jvx->vx", body["weights"], world) + (trans[t] - rest[0])
    return verts


def mesh_from_archive(path: Path, query_t: Optional[np.ndarray] = None) -> dict:
    """Full-mesh vertices for a standard SMPL archive (AnySole NPZ contract).

    Args:
        path: SMPL NPZ archive (poses/root_orient+pose_body, trans, betas).
        query_t: optional time grid (seconds, archive source-frame clock) for
            slerp resampling, the same grid callers pass to ``load_motion``.

    Returns:
        dict with ``verts`` (T, 6890, 3) native y-up metres, ``faces``
        (13776, 3), ``fps``.
    """
    loaded = load_smpl(Path(path), query_t=query_t)
    rotvec = smpl24_pose6d_to_poses(loaded["pose_6d"]).reshape(-1, 24, 3)
    verts = smpl_vertices(rotvec, loaded["trans_m"], loaded["betas"], loaded["joint_rest_m"])
    return {"verts": verts, "faces": smpl_faces(), "fps": loaded["fps"]}


def _renderer(width: int, height: int):
    """Cached pyrender EGL offscreen renderer per viewport size."""
    key = (width, height)
    if key not in _RENDERERS:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender

        _RENDERERS[key] = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height, point_size=1.0
        )
    return _RENDERERS[key]


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame camera pose whose -z axis points from ``eye`` to ``target``."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right_norm = np.linalg.norm(right)
    right = right / right_norm if right_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _project_points(points: np.ndarray, camera_pose: np.ndarray, focal: float,
                    width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image (u, v) pixels plus view-space depth."""
    cam = (np.asarray(points) - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    depth = -cam[:, 2]
    u = width * 0.5 + focal * cam[:, 0] / depth
    v = height * 0.5 - focal * cam[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def _hex_to_rgb01(color: str) -> list:
    text = color.lstrip("#")
    return [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _depth_visible(u: float, v: float, point_depth: float, depth_image: np.ndarray) -> bool:
    """Depth-test one projected point against the rendered surface.

    Joints sit a few cm inside the body, so a 6cm tolerance keeps them
    visible on the camera-facing side while hiding the far-side skeleton.
    """
    row, col = int(round(v)), int(round(u))
    if row < 0 or row >= depth_image.shape[0] or col < 0 or col >= depth_image.shape[1]:
        return True  # outside the viewport: nothing can occlude it
    surface = depth_image[row, col]
    return surface <= 0.0 or point_depth <= surface + 0.06


def _overlay_2d(image: np.ndarray, uv: np.ndarray, edges: list, bone_color: str,
                joint_color: str, title: str, title_color: str,
                depth_image: np.ndarray, point_depth: np.ndarray) -> np.ndarray:
    """Draw skeleton + title onto the rendered frame with the same camera
    projection, depth-tested against the mesh surface so far-side bones stay
    hidden behind the body."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape[:2]
    for parent, child in edges:
        p0, p1 = uv[parent], uv[child]
        if not (_depth_visible(p0[0], p0[1], point_depth[parent], depth_image)
                or _depth_visible(p1[0], p1[1], point_depth[child], depth_image)):
            continue
        draw.line([(p0[0], p0[1]), (p1[0], p1[1])], fill=bone_color, width=4)
    for (u, v), depth in zip(uv, point_depth):
        if _depth_visible(u, v, depth, depth_image):
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=joint_color)
    if title:
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 8), title, fill=title_color, font=font)
    return np.asarray(canvas)


def render_mesh_frame(verts: np.ndarray, faces: np.ndarray, *, joints: Optional[np.ndarray] = None,
                      edges: Optional[list] = None, title: str = "", frame_index: int = 0,
                      total: int = 1, mesh_color: tuple = (0.52, 0.63, 0.72),
                      bone_color: str = "#AF5A3B", joint_color: str = "#355166",
                      title_color: str = "#333333", bg_color: str = "#F3E6D7",
                      size: tuple = (512, 512), elev: float = 18.0,
                      azim: float = -62.0) -> np.ndarray:
    """Render one full-mesh frame, CLIFF/pressure_tookit-style (RGB array).

    Scene/material/lights/render call follow the original pyrender pipeline;
    the camera is framed on the body (distance and focal derived from the
    vertex radius, view angle elev/azim as in the skeleton renderer).
    ``verts`` must already be in display z-up metres
    (``smpl_yup_to_display``).  When ``joints``/``edges`` are given the
    skeleton is drawn on top using the same camera projection.
    """
    import trimesh
    import pyrender
    from trimesh.transformations import rotation_matrix

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    width, height = size
    center = np.asarray(joints if joints is not None else verts).mean(axis=0)
    radius = max(float(np.max(np.ptp(verts, axis=0))), 1.0) * 0.65
    distance = radius * 2.2
    focal = (min(width, height) * 0.5) * distance / radius
    azimuth, elevation = np.radians(azim), np.radians(elev)
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    camera_pose = _look_at(center + direction * distance, center)

    scene = pyrender.Scene(bg_color=_hex_to_rgb01(bg_color), ambient_light=np.zeros(3))
    scene.add(
        pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=width * 0.5, cy=height * 0.5),
        pose=camera_pose,
    )
    for angle, axis in ((np.radians(-45), [1.0, 0.0, 0.0]), (np.radians(45), [0.0, 1.0, 0.0])):
        # CLIFF's light poses are camera-relative; compose them with the camera.
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=camera_pose @ rotation_matrix(angle, axis),
        )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.5,
        baseColorFactor=[*mesh_color, 1.0],
        alphaMode="OPAQUE",
    )
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts, faces=faces), material=material))
    color, depth = _renderer(width, height).render(scene)
    image = color[..., :3].copy()

    label = f"{title}  frame {frame_index}/{total - 1}" if title else ""
    if joints is not None and edges:
        uv, point_depth = _project_points(joints, camera_pose, focal, width, height)
        image = _overlay_2d(image, uv, edges, bone_color, joint_color, label, title_color,
                            depth, point_depth)
    elif label:
        image = _overlay_2d(image, np.zeros((0, 2)), [], bone_color, joint_color, label,
                            title_color, depth, np.zeros(0))
    return image
