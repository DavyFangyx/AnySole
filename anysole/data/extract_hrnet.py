"""Extract per-frame CLIFF-HRNet features and bbox info into AnySole caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from anysole.data.crop import CROP_IMG_HEIGHT, CROP_IMG_WIDTH, process_image
from anysole.data.dataset import find_session_dir, hrnet_cache_path
from anysole.types import (
    CLIFF_CKPT,
    HRNET_YAML,
    MOTIONPRO_ROOT,
    HRNET_CACHE_ROOT,
    SEQ_ROOT,
    V_FEAT_DIM,
)

HRNET_FEAT_DIM = 2048
BBOX_INFO_DIM = 3
CLIFF_FOCAL_LENGTH = 608.0

ENCODER_PREFIX = "module.encoder."


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2051-d AnySole visual features (HRNet 2048 + CLIFF bbox_info 3)."
    )
    parser.add_argument("--cam-id", type=int, required=True)
    parser.add_argument("--session", type=str, default=None, help="Only extract this session id.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--seq-root", type=Path, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    return parser.parse_args(argv)


def resolve_roots(
    cam_id: int,
    cache_root: Optional[Path] = None,
    seq_root: Optional[Path] = None,
) -> tuple:
    resolved_seq_root = seq_root or SEQ_ROOT.parent / ("cam%d" % int(cam_id))
    resolved_cache_root = cache_root or HRNET_CACHE_ROOT.parent / ("cam%d" % int(cam_id))
    return resolved_seq_root, resolved_cache_root


def list_session_dirs(seq_root: Path, session_id: Optional[str]) -> List[Path]:
    if session_id:
        return [find_session_dir(seq_root, session_id)]
    dirs = sorted(path for path in seq_root.glob("*/*/*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError("No sequence dirs under %s" % seq_root)
    return dirs


def load_encoder(device: torch.device) -> torch.nn.Module:
    motionpro_root = str(MOTIONPRO_ROOT)
    if motionpro_root not in sys.path:
        sys.path.insert(0, motionpro_root)

    from lib.model.backbone.hrnet.cls_hrnet import HighResolutionNet
    from lib.model.backbone.hrnet.hrnet_config import cfg
    from lib.model.backbone.hrnet.hrnet_config import update_config

    update_config(cfg, str(HRNET_YAML))
    model = HighResolutionNet(cfg)
    ckpt = torch.load(str(CLIFF_CKPT), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    enc_state = {}
    for key, value in state.items():
        if key.startswith(ENCODER_PREFIX):
            enc_state[key[len(ENCODER_PREFIX):]] = value
    if not enc_state:
        raise KeyError("CLIFF checkpoint missing %s weights: %s" % (ENCODER_PREFIX, CLIFF_CKPT))
    model.load_state_dict(enc_state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Failed to read image %s" % path)
    return bgr[:, :, ::-1]


def session_frame_paths(seq_dir: Path, n_frames: int) -> List[Path]:
    color_dir = seq_dir / "color"
    paths = [color_dir / ("%06d.jpg" % idx) for idx in range(n_frames)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "%s expected %d JPGs under color/; missing e.g. %s" % (seq_dir, n_frames, missing[0])
        )
    return paths


def extract_session(
    seq_dir: Path,
    cache_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> None:
    meta = json.loads((seq_dir / "align_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    image_paths = session_frame_paths(seq_dir, n_frames)
    bbox = np.load(seq_dir / "bbox.npy")
    if bbox.shape[0] != n_frames:
        raise ValueError("%s bbox.npy length %s != n_frames %d" % (seq_dir, bbox.shape, n_frames))

    chunks = []
    batch = []
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            rgb = read_rgb(image_path)
            box = bbox[idx][1:5]
            norm_img, center, scale, _, _, _ = process_image(
                rgb, box, crop_height=CROP_IMG_HEIGHT, crop_width=CROP_IMG_WIDTH
            )
            norm_img = np.asarray(norm_img, dtype=np.float32)
            if tuple(norm_img.shape) != (3, CROP_IMG_HEIGHT, CROP_IMG_WIDTH):
                raise ValueError(
                    "%s frame %d crop shape %s != (3, %d, %d)"
                    % (seq_dir.name, idx, norm_img.shape, CROP_IMG_HEIGHT, CROP_IMG_WIDTH)
                )
            img_height, img_width = rgb.shape[:2]
            center = np.asarray(center, dtype=np.float32)
            bbox_size = np.float32(scale) * 200.0
            bbox_info = np.asarray(
                [
                    (center[0] - img_width / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                    (center[1] - img_height / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                    (bbox_size - 0.24 * CLIFF_FOCAL_LENGTH)
                    / (0.06 * CLIFF_FOCAL_LENGTH),
                ],
                dtype=np.float32,
            )
            batch.append((norm_img, bbox_info))
            if len(batch) < batch_size and idx + 1 < n_frames:
                continue
            x = torch.from_numpy(np.stack([item[0] for item in batch], axis=0)).to(
                device=device, dtype=torch.float32
            )
            xf = model(x)
            if xf.ndim != 2 or xf.shape[1] != HRNET_FEAT_DIM:
                raise ValueError(
                    "HRNet output shape %s != (B, %d)" % (tuple(xf.shape), HRNET_FEAT_DIM)
                )
            bbox_info = torch.from_numpy(np.stack([item[1] for item in batch], axis=0))
            features = torch.cat(
                [xf.detach().cpu().to(dtype=torch.float32), bbox_info.float()], dim=1
            )
            if features.shape[1] != V_FEAT_DIM:
                raise ValueError(
                    "AnySole visual feature shape %s != (B, %d)"
                    % (tuple(features.shape), V_FEAT_DIM)
                )
            chunks.append(features)
            batch = []

    feat = torch.cat(chunks, dim=0)
    if tuple(feat.shape) != (n_frames, V_FEAT_DIM):
        raise ValueError(
            "%s feature shape %s != (%d, %d)" % (seq_dir.name, tuple(feat.shape), n_frames, V_FEAT_DIM)
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat.contiguous(), cache_path)
    print("saved %s %s" % (cache_path, tuple(feat.shape)))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    seq_root, cache_root = resolve_roots(args.cam_id, args.cache_root, args.seq_root)
    session_dirs = list_session_dirs(seq_root, args.session)
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_dirs = session_dirs[: args.limit_sessions]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    model = None
    for seq_dir in session_dirs:
        cache_path = hrnet_cache_path(seq_dir.name, cache_root)
        if cache_path.is_file() and not args.overwrite:
            print("skip %s (%s exists)" % (seq_dir.name, cache_path))
            continue
        if model is None:
            print("load encoder on %s" % device)
            model = load_encoder(device)
        print("extract %s" % seq_dir)
        extract_session(seq_dir, cache_path, model, device, int(args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
