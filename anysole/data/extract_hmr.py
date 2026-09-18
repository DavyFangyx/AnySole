"""F1: extract per-frame GVHMR SMPL parameters into AnySole hmr caches.

Offline prep, run BEFORE any ``v_input="hmr_gvhmr"`` training/eval.  For each
cam3 session: CLIFF bbox (bbox.npy) -> bbx_xys -> ViTPose COCO-17 kp2d +
HMR2 ViT features (f_imgseq) -> GVHMR DemoPL.predict (static camera) ->
per-frame SMPL params (axis-angle body_pose, betas, global_orient, transl,
both camera and gravity-aligned world space) + 22 SMPL-X body joints via
EnDecoder.fk_v2 (for the zero-training probe).

Cache layout (per session):
  AnysoleWorkspace/derived/AnySole/hmr_cache/<model>/cam3/<session>.pt
  keys: body_pose (N,63) aa | global_orient (N,3) aa | betas (N,10)
        transl (N,3) | kp2d (N,17,3) | f_imgseq (N,1024) | bbx_xys (N,3)
        joints_hmr (N,22,3) global-space FK | q_v (N,2)
  All float32.  ``body_pose``/``global_orient``/``transl`` are the
  post-processed global-space (gravity-aligned) params; ``*_incam`` variants
  are the camera-space ones.

REQUIREMENTS (none shipped in the repo — download first):
  Baselines/Video2Motion/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
  Baselines/Video2Motion/GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
  Baselines/Video2Motion/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
  Baselines/Video2Motion/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
  plus the repo's own python deps (hydra, pytorch3d, smplx, ... — its
  requirements.txt).  The GVHMR repo dir must be on sys.path (this script
  inserts it automatically).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

GAIT_ROOT = Path("/data/fangyuxuan/projects/gait")
GVHMR_ROOT = GAIT_ROOT / "Baselines" / "Video2Motion" / "GVHMR"
sys.path.insert(0, str(GVHMR_ROOT))

from anysole.data.dataset import find_session_dir  # noqa: E402
from anysole.types import HMR_CACHE_ROOT, SEQ_ROOT  # noqa: E402

REQUIRED_FILES = (
    GVHMR_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    GVHMR_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    GVHMR_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    GVHMR_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-frame GVHMR SMPL params into AnySole hmr caches."
    )
    parser.add_argument("--cam-id", type=int, default=3)
    parser.add_argument("--model", type=str, default="gvhmr", help="Cache subdir name.")
    parser.add_argument("--session", type=str, default=None, help="Only extract this session id.")
    mode = parser.add_argument("--skip-existing", action="store_true",
                               help="Skip sessions whose cache exists.")
    parser.add_argument("--force", "--overwrite", dest="force", action="store_true",
                        help="Recompute and overwrite existing caches.")
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def check_requirements() -> None:
    missing = [str(p) for p in REQUIRED_FILES if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "GVHMR checkpoints/body models not found (see GVHMR docs/INSTALL.md for the "
            "Google Drive links):\n  %s" % "\n  ".join(missing)
        )


def load_model(device: str):
    from hydra import initialize_config_module, compose
    import hydra
    from hmr4d.configs import register_store_gvhmr
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=[
            "static_cam=True", "verbose=False", "use_dpvo=False",
        ])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().to(device)
    return model


def load_frames(seq_dir: Path, n_frames: int) -> np.ndarray:
    """(L, H, W, 3) RGB uint8 frames, L = n_frames."""
    import cv2
    color_dir = seq_dir / "color"
    frames = []
    for i in range(n_frames):
        path = color_dir / ("%06d.jpg" % i)
        if not path.is_file():
            raise FileNotFoundError("missing frame %s" % path)
        frames.append(cv2.imread(str(path))[..., ::-1])  # BGR -> RGB
    return np.stack(frames, axis=0)


def load_bbox_xyxy(seq_dir: Path, n_frames: int) -> torch.Tensor:
    bbox = np.load(seq_dir / "bbox.npy")
    # rows: [frame, x1, y1, x2, y2, ...] (CLIFF output, see extract_hrnet.py).
    xyxy = np.zeros((n_frames, 4), dtype=np.float32)
    for row in bbox:
        frame = int(row[0])
        if 0 <= frame < n_frames:
            xyxy[frame] = row[1:5]
    return torch.from_numpy(xyxy)


def bbox_trunc_ratio(bbx_xys: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    """Fraction of each bbox lying outside the image (quality feature q_V)."""
    cx, cy, size = bbx_xys[:, 0], bbx_xys[:, 1], bbx_xys[:, 2]
    x1 = (cx - size / 2).clamp(min=0)
    x2 = (cx + size / 2).clamp(max=img_w)
    y1 = (cy - size / 2).clamp(min=0)
    y2 = (cy + size / 2).clamp(max=img_h)
    inside = ((x2 - x1) * (y2 - y1)) / (size * size + 1e-6)
    return (1.0 - inside.clamp(min=0)).numpy().astype(np.float32)


def extract_session(model, session_id: str, device: str) -> dict:
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor

    seq_dir = find_session_dir(SEQ_ROOT, session_id)
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    n = int(meta["n_frames"])
    frames = load_frames(seq_dir, n)
    img_h, img_w = frames.shape[1], frames.shape[2]
    bbx_xys = get_bbx_xys_from_xyxy(load_bbox_xyxy(seq_dir, n), base_enlarge=1.2)  # (L,3)

    # Shared cropped batch (L,3,256,256) — ViTPose and the HMR2 backbone both
    # consume the get_batch output tensor.
    imgs, bbx_xys = get_batch(frames, bbx_xys, img_ds=1.0, path_type="np")
    kp2d = VitPoseExtractor(tqdm_leave=False).extract(imgs.to(device), bbx_xys)  # (L,17,3)
    f_imgseq = Extractor(tqdm_leave=False).extract_video_features(imgs, bbx_xys)  # (L,1024)

    K_fullimg = estimate_K(img_w, img_h).repeat(n, 1, 1)
    cam_angvel = compute_cam_angvel(torch.eye(3).repeat(n, 1, 1))
    data = {
        "length": torch.tensor(n),
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": f_imgseq,
    }
    pred = model.predict(data, static_cam=True)
    global_params = pred["smpl_params_global"]
    incam_params = pred["smpl_params_incam"]
    joints = model.pipeline.endecoder.fk_v2(**global_params)[0]  # (L,22,3)

    def to_np(x) -> np.ndarray:
        return torch.as_tensor(x).detach().cpu().numpy().astype(np.float32)

    kp_conf_mean = kp2d[:, :, 2].mean(dim=1).numpy().astype(np.float32)
    trunc = bbox_trunc_ratio(bbx_xys, img_w, img_h)
    out = {
        "body_pose": to_np(global_params["body_pose"]),
        "global_orient": to_np(global_params["global_orient"]),
        "betas": to_np(global_params["betas"]),
        "transl": to_np(global_params["transl"]),
        "body_pose_incam": to_np(incam_params["body_pose"]),
        "global_orient_incam": to_np(incam_params["global_orient"]),
        "transl_incam": to_np(incam_params["transl"]),
        "kp2d": to_np(kp2d),
        "f_imgseq": to_np(f_imgseq),
        "bbx_xys": to_np(bbx_xys),
        "joints_hmr": to_np(joints),
        "q_v": np.stack([kp_conf_mean, trunc], axis=1).astype(np.float32),
        # Image dims for the dataset-side bbox normalization (misc block).
        "img_w": int(img_w),
        "img_h": int(img_h),
    }
    for key, value in out.items():
        if value.shape[0] != n:
            raise ValueError("%s length %d != n_frames %d" % (key, value.shape[0], n))
    return out


def cache_path(model: str, session_id: str) -> Path:
    return HMR_CACHE_ROOT / model / "cam3" / ("%s.pt" % session_id)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    check_requirements()
    from anysole.data.dataset import load_split_ids
    from anysole.types import SPLIT_CSV
    sessions = [args.session] if args.session else load_split_ids(SPLIT_CSV, "train")
    if args.limit_sessions is not None:
        sessions = sessions[: args.limit_sessions]
    model = load_model(args.device)
    for session_id in sessions:
        out_path = cache_path(args.model, session_id)
        if out_path.is_file() and args.skip_existing and not args.force:
            print("skip %s (cache exists)" % session_id)
            continue
        print("extracting %s ..." % session_id)
        out = extract_session(model, session_id, args.device)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, out_path)
        print("  -> %s (%d frames)" % (out_path, out["body_pose"].shape[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
