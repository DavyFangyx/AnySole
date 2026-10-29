"""Stable paths and runtime policy for the SAM3DB baseline."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("VB_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
DATASET_ROOT = WORKSPACE_ROOT / "1_Dataset"
CODE_ROOT = WORKSPACE_ROOT / "2_Code"
UPSTREAM_ROOT = WORKSPACE_ROOT / "third_party" / "fast-sam-3d-body"
DINOV3_ROOT = WORKSPACE_ROOT / "third_party" / "dinov3"
RESULT_ROOT = WORKSPACE_ROOT / "3_Result" / "SAM3DB_baseline"
CHECKPOINT_ROOT = UPSTREAM_ROOT / "checkpoints" / "sam-3d-body-dinov3"
MOGE_MODEL = WORKSPACE_ROOT / "models" / "moge-2-vits-normal" / "model.pt"
SMPL_MODEL = WORKSPACE_ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
MHR2SMPL_MODEL_DIR = (
    UPSTREAM_ROOT / "mhr2smpl" / "experiments" / "multiview_n30000_e500"
)
MHR_MAPPING = (
    WORKSPACE_ROOT
    / "third_party"
    / "mhr"
    / "tools"
    / "mhr_smpl_conversion"
    / "assets"
    / "mhr2smpl_mapping.npz"
)
MHR_FACE_MESH = MHR_MAPPING.with_name("mhr_face_mask.ply")
DEFAULT_FPS = 30.0
DEFAULT_GPUS = tuple(range(7))


REQUIRED_ASSETS = {
    "sam3db_checkpoint": CHECKPOINT_ROOT / "model.ckpt",
    "mhr_model": CHECKPOINT_ROOT / "assets" / "mhr_model.pt",
    "dinov3_hubconf": DINOV3_ROOT / "hubconf.py",
    "moge_model": MOGE_MODEL,
    "smpl_model": SMPL_MODEL,
    "mhr2smpl_model": MHR2SMPL_MODEL_DIR / "best_model.pth",
    "mhr2smpl_sample_indices": MHR2SMPL_MODEL_DIR / "sample_idx.npy",
    "mhr2smpl_mapping": MHR_MAPPING,
    "mhr_face_mesh": MHR_FACE_MESH,
}


def runtime_environment(gpu: int | None = None) -> dict[str, str]:
    """Return the fail-closed local runtime environment."""

    env = dict(os.environ)
    pythonpath = [str(UPSTREAM_ROOT), str(CODE_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    prefix_lib = str(Path(os.environ["VB_TACTILE_PREFIX"]) / "lib")
    library_path = prefix_lib
    if env.get("LD_LIBRARY_PATH"):
        library_path += ":" + env["LD_LIBRARY_PATH"]
    env.update(
        PYTHONPATH=":".join(pythonpath),
        LD_LIBRARY_PATH=library_path,
        PYOPENGL_PLATFORM="egl",
        PYGLET_HEADLESS="true",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SAM3DB_OFFLINE="1",
        DINOV3_REPO_PATH=str(DINOV3_ROOT),
        FOV_MODEL_PATH=str(MOGE_MODEL),
        FOV_TRT="0",
        FOV_MODEL="s",
        FOV_LEVEL="0",
        FOV_FAST="1",
        USE_TRT_BACKBONE="0",
        USE_COMPILE="0",
        USE_COMPILE_BACKBONE="0",
        DECODER_COMPILE="0",
        MHR_USE_CUDA_GRAPH="0",
        LAYER_DTYPE="fp32",
        MHR_NO_CORRECTIVES="1",
        SKIP_KEYPOINT_PROMPT="1",
        KEYPOINT_PROMPT_INTERM_INTERVAL="999",
        BODY_INTERM_PRED_LAYERS="0,1,2",
        HAND_INTERM_PRED_LAYERS="0,1",
        GPU_HAND_PREP="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        OPENCV_FOR_THREADS_NUM="1",
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env
