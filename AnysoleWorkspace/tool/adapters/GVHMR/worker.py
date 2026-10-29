"""Isolated official GVHMR 2024 inference worker (run in the GVHMR Conda env)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .config import (
    DEFAULT_HMR2_CHECKPOINT,
    DEFAULT_VITPOSE_CHECKPOINT,
    UPSTREAM_ROOT,
    required_assets,
    required_preprocess_assets,
)
from .conversion import convert_native_sequence, load_conversion_models
from .io import atomic_json


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _licensed_smplx_archive(source: Path, work_dir: Path) -> Path:
    """Expose any user-provided licensed archive under the name expected upstream."""

    source = source.expanduser().resolve()
    if source.is_dir():
        candidate = source / "SMPLX_NEUTRAL.npz"
        if not candidate.is_file():
            candidate = source / "smplx" / "SMPLX_NEUTRAL.npz"
        if not candidate.is_file():
            raise FileNotFoundError(f"licensed SMPL-X directory lacks SMPLX_NEUTRAL.npz: {source}")
        source = candidate.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"licensed SMPL-X model archive does not exist: {source}")
    if source.name == "SMPLX_NEUTRAL.npz" and source.parent.name == "smplx":
        return source
    alias_dir = work_dir / "licensed_body_models" / "smplx"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / "SMPLX_NEUTRAL.npz"
    alias.symlink_to(source)
    return alias


def _native_params(prediction: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    params = prediction.get(key)
    expected = {"global_orient", "body_pose", "betas", "transl"}
    if not isinstance(params, dict) or not expected.issubset(params):
        raise RuntimeError(f"GVHMR output {key} missing fields: {expected - set(params or {})}")
    native = {name: _numpy(params[name]) for name in expected}
    count = native["global_orient"].shape[0]
    shapes = {
        "global_orient": (count, 3),
        "body_pose": (count, 63),
        "betas": (count, 10),
        "transl": (count, 3),
    }
    for name, shape in shapes.items():
        if native[name].shape != shape or not np.isfinite(native[name]).all():
            raise RuntimeError(f"invalid native {key}.{name}: {native[name].shape}, expected {shape}")
    return native


def run_request(request_path: str | Path, output_path: str | Path) -> None:
    request_file = Path(request_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    request = json.loads(request_file.read_text(encoding="utf-8"))
    work_dir = Path(request["work_dir"]).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    upstream = Path(request.get("upstream_root", UPSTREAM_ROOT)).expanduser().resolve()
    checkpoint = Path(request["checkpoint_path"]).expanduser().resolve()
    smplx_archive = _licensed_smplx_archive(Path(request["smplx_model_path"]), work_dir)
    smpl_model = Path(request["smpl_model_path"]).expanduser().resolve()
    mapper = Path(request["mapper_path"]).expanduser().resolve()
    required_assets(
        checkpoint=checkpoint,
        smplx_model=smplx_archive,
        smpl_model=smpl_model,
        mapper=mapper,
    )
    hmr2_checkpoint = Path(request.get("hmr2_checkpoint_path", DEFAULT_HMR2_CHECKPOINT)).expanduser().resolve()
    vitpose_checkpoint = Path(request.get("vitpose_checkpoint_path", DEFAULT_VITPOSE_CHECKPOINT)).expanduser().resolve()
    preprocessing_assets = required_preprocess_assets(
        hmr2_checkpoint=hmr2_checkpoint,
        vitpose_checkpoint=vitpose_checkpoint,
    )

    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import cv2
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GVHMR official DemoPL.predict requires CUDA")
    torch.cuda.set_device(int(request.get("cuda_device", 0)))
    torch.backends.cudnn.benchmark = False
    os_cwd = Path.cwd()
    if os_cwd != upstream:
        # Pinned extractors use PROJ_ROOT for checkpoints, and one ViTPose
        # checkpoint is referenced relative to the official source root.
        os.chdir(upstream)

    source_video = Path(request["video_path"]).expanduser().resolve()
    count = int(request["frame_count"])
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open staged inference video {source_video}")
    actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count < 2 or actual_count != count or min(width, height) <= 0:
        raise RuntimeError(f"invalid official video geometry/count: {(actual_count, width, height)} != {count}")

    import hydra
    from hydra import compose, initialize_config_module

    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.body_model import smplx_lite
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    # The 2024 demo registers its Hydra model group as an import side effect;
    # register_store_gvhmr() alone imports the training PL, not DemoPL.
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401

    # The upstream extractor constructors hard-code their release checkpoint
    # paths. Keep the classes/forward path official, substituting only the
    # caller-validated local weight filenames.
    import hmr4d.utils.preproc.vitfeat_extractor as feature_module
    import hmr4d.utils.preproc.vitpose as pose_module
    from hmr4d.network.hmr2 import load_hmr2

    feature_module.load_hmr2 = lambda: load_hmr2(str(hmr2_checkpoint))
    original_build_model = pose_module.build_model
    pose_module.build_model = lambda name, ckpt_path=None: original_build_model(name, str(vitpose_checkpoint))

    # DemoPL's EnDecoder constructs SmplxLiteV437Coco17 with its repository
    # default. Override only this in-memory constructor default so a licensed
    # user file can be used without editing or copying any upstream source.
    original_smplx_lite_init = smplx_lite.SmplxLite.__init__

    def init_with_licensed_archive(self, *args, **kwargs):
        kwargs.setdefault("model_path", smplx_archive)
        return original_smplx_lite_init(self, *args, **kwargs)

    smplx_lite.SmplxLite.__init__ = init_with_licensed_archive

    boxes = np.asarray(request["bbox_xyxy_px"], dtype=np.float32)
    if boxes.shape != (count, 4) or not np.isfinite(boxes).all():
        raise RuntimeError(f"invalid target bbox shape: {boxes.shape}")
    bbx_xyxy = torch.as_tensor(boxes, dtype=torch.float32)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    if bbx_xys.shape != (count, 3) or not torch.isfinite(bbx_xys).all():
        raise RuntimeError("official target-box conversion returned invalid `bbx_xys`")

    # These are the official pinned ViTPose/HMR2 preprocessors, called with
    # our target's boxes rather than the upstream auto-tracker.
    vitpose_extractor = VitPoseExtractor(tqdm_leave=False)
    keypoints = vitpose_extractor.extract(str(source_video), bbx_xys)
    feature_extractor = Extractor(tqdm_leave=False)
    image_features = feature_extractor.extract_video_features(str(source_video), bbx_xys)
    del vitpose_extractor, feature_extractor
    torch.cuda.empty_cache()
    if keypoints.shape != (count, 17, 3) or image_features.shape[0] != count:
        raise RuntimeError(f"official preprocessing shapes invalid: {keypoints.shape}, {image_features.shape}")

    K_fullimg = estimate_K(width, height).repeat(count, 1, 1)
    if bool(request.get("static_camera", True)):
        identity_w2c = torch.eye(3).repeat(count, 1, 1)
        cam_angvel = compute_cam_angvel(identity_w2c)
    else:
        try:
            from dpvo.config import cfg as _dpvo_cfg  # noqa: F401
            from dpvo.dpvo import DPVO as _DPVO  # noqa: F401
            from dpvo.utils import Timer as _Timer  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "moving-camera inference needs the pinned DPVO submodule/extension and checkpoint; "
                "use static-camera mode for fixed-camera dataset clips"
            ) from exc
        dpvo_config = upstream / "third-party" / "DPVO" / "config" / "default.yaml"
        dpvo_checkpoint = upstream / "inputs" / "checkpoints" / "dpvo" / "dpvo.pth"
        if not dpvo_config.is_file() or not dpvo_checkpoint.is_file():
            raise RuntimeError(
                f"moving-camera inference needs DPVO config/checkpoint: {dpvo_config}, {dpvo_checkpoint}"
            )
        from pytorch3d.transforms import quaternion_to_matrix
        from hmr4d.utils.geo.hmr_cam import convert_K_to_K4
        from hmr4d.utils.preproc.slam import SLAMModel

        dynamic_slam = SLAMModel(
            str(source_video), width, height, convert_K_to_K4(K_fullimg[0]), buffer=4000, resize=0.5
        )
        while dynamic_slam.track():
            pass
        trajectory = dynamic_slam.process()
        if trajectory.shape[0] != count:
            raise RuntimeError(f"DPVO returned {trajectory.shape[0]} camera poses for {count} frames")
        rotations = quaternion_to_matrix(torch.from_numpy(trajectory[:, [6, 3, 4, 5]]).float()).mT
        cam_angvel = compute_cam_angvel(rotations)
    data = {
        "length": torch.tensor(count),
        "bbx_xys": bbx_xys,
        "kp2d": keypoints,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": image_features,
    }
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo")
    model_cls = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model_cls.load_pretrained_model(str(checkpoint))
    model_cls = model_cls.eval().cuda()
    prediction = model_cls.predict(data, static_cam=bool(request.get("static_camera", True)))
    native_camera = _native_params(prediction, "smpl_params_incam")
    native_global = _native_params(prediction, "smpl_params_global")
    K_prediction = _numpy(prediction["K_fullimg"])
    del model_cls, prediction, data
    torch.cuda.empty_cache()

    smplx_model, mapper_value, smpl_model_value, hashes = load_conversion_models(
        upstream_root=upstream,
        checkpoint_path=checkpoint,
        smplx_model_path=smplx_archive,
        smpl_model_path=smpl_model,
        mapper_path=mapper,
        device=f"cuda:{int(request.get('cuda_device', 0))}",
    )
    camera_fit = convert_native_sequence(native_camera, smplx_model, mapper_value, smpl_model_value)
    global_fit = convert_native_sequence(native_global, smplx_model, mapper_value, smpl_model_value)
    if K_prediction.shape != (count, 3, 3) or not np.isfinite(K_prediction).all():
        raise RuntimeError(f"invalid model intrinsics output: {K_prediction.shape}")

    result_arrays = {
        "native_pose_camera": np.concatenate([native_camera["global_orient"], native_camera["body_pose"]], axis=1),
        "native_pose_global": np.concatenate([native_global["global_orient"], native_global["body_pose"]], axis=1),
        "native_betas_camera": native_camera["betas"],
        "native_betas_global": native_global["betas"],
        "native_trans_camera": native_camera["transl"],
        "native_trans_global": native_global["transl"],
        "pose": camera_fit.pose,
        "betas": camera_fit.betas,
        "trans": camera_fit.translation,
        "pose_global": global_fit.pose,
        "betas_global": global_fit.betas,
        "trans_global": global_fit.translation,
        "fit_vertex_rmse_mm": camera_fit.vertex_rmse_mm,
        "fit_vertex_p95_mm": camera_fit.vertex_p95_mm,
        "fit_joint_rmse_mm": camera_fit.joint_rmse_mm,
        "fit_global_vertex_rmse_mm": global_fit.vertex_rmse_mm,
        "fit_global_vertex_p95_mm": global_fit.vertex_p95_mm,
        "fit_global_joint_rmse_mm": global_fit.joint_rmse_mm,
        "joints_camera": camera_fit.joints,
        "joints_global": global_fit.joints,
        "K_fullimg": K_prediction,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **result_arrays)
    atomic_json(
        output_file.with_suffix(".metadata.json"),
        {
            "upstream_revision": "44fbc1999dd57d2180e882c62c78d356d4c840d0",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "smplx_licensed_archive": str(smplx_archive),
            "preprocess_assets": {name: str(path) for name, path in preprocessing_assets.items()},
            "conversion_asset_hashes": hashes,
            "camera_frame_coordinate_semantics": "GVHMR SIGGRAPH Asia 2024 `smpl_params_incam`; trained estimated pinhole camera; metres; not checkerboard/world aligned",
            "global_frame_coordinate_semantics": "GVHMR SIGGRAPH Asia 2024 gravity-view/global gauge; metres; not aligned to Tactile checkerboard world",
            "native_model_family": "SMPL-X supermotion; 22 joints / 66 axis-angle values",
            "smpl_conversion": {
                "camera": camera_fit.metadata,
                "global": global_fit.metadata,
            },
            "static_camera": bool(request.get("static_camera", True)),
            "model_video_fps": 30,
        },
    )
    if os_cwd != upstream:
        os.chdir(os_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_request(arguments.request, arguments.output)


if __name__ == "__main__":
    main()
