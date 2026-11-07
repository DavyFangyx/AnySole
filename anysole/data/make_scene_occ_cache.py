"""Build a pilot HRNet cache with a calibrated static scene occluder.

For every frame marked active in the manifest, composite the column into the
RGB frame and rerun the CLIFF-HRNet encoder.  Inactive frames copy the clean
``cam3`` feature row bit-for-bit, so the only changed rows are those whose
crop contains the column.

Usage:
  python -m anysole.data.make_scene_occ_cache \
      --manifest-dir results/diagnostics/20261107_e5_sceneocc_pilot/manifests \
      --out AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/cam3_occ_scene_pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))

from anysole.data.dataset import find_session_dir, hrnet_cache_path  # noqa: E402
from anysole.data.extract_hrnet import (  # noqa: E402
    BBOX_INFO_DIM, CLIFF_FOCAL_LENGTH, HRNET_FEAT_DIM, V_FEAT_DIM,
    load_encoder, process_image, read_rgb, session_frame_paths)
from anysole.data.workspace_adapter import load_shared_session  # noqa: E402
from anysole.types import HRNET_CACHE_ROOT, SEQ_ROOT, V_FEAT_DIM as _V  # noqa: E402

CROP_H = 256
CROP_W = 192
BBOX_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox_cam3"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--demo-dir", type=Path, default=None,
                   help="Optional directory for a few composited preview JPGs.")
    return p.parse_args(argv)


def column_texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(82.0, 132.0, h, dtype=np.float32)[:, None]
    tex = np.repeat(base[:, :, None], w, axis=1)
    tex = np.repeat(tex, 3, axis=2)
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


def polygon_mask(shape, poly) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(m, poly.astype(np.int32), 255)
    return m > 0


def composite(rgb: np.ndarray, person: np.ndarray, poly: np.ndarray,
              texture: np.ndarray, behind: bool) -> np.ndarray:
    out = rgb.copy()
    col = polygon_mask(rgb.shape, poly)
    if behind:
        paint = col
    else:
        paint = col & ~person
    out[paint] = texture[paint]
    return out


def build_session(manifest_path: Path, out_root: Path, model, device,
                  batch_size: int, demo_dir: Path | None) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sid = manifest["session_id"]
    sess = load_shared_session(sid)
    seq_dir = find_session_dir(SEQ_ROOT, sid)
    n_frames = int(manifest["n_frames"])
    paths = session_frame_paths(seq_dir, n_frames)
    bbox = np.load(BBOX_ROOT / ("%s.npy" % sid))
    if bbox.shape[0] != n_frames:
        raise ValueError("%s bbox length mismatch" % sid)
    freq = manifest["frames"]
    active = np.asarray(freq["active"], dtype=bool)
    behind = np.asarray(freq["behind"], dtype=bool)
    vis = np.asarray(freq["vis"], dtype=np.float64)
    poly = np.asarray(manifest["column_polygon_xy"], dtype=np.float32)

    clean = torch.load(hrnet_cache_path(sid, HRNET_CACHE_ROOT), map_location="cpu")
    if tuple(clean.shape) != (n_frames, V_FEAT_DIM):
        raise ValueError("%s clean cache shape %s" % (sid, tuple(clean.shape)))
    feat = clean.clone()

    # Person masks are only needed where the column enters the crop.  Load them
    # from the same SAM3.1 index used by the manifest builder.
    index = {}
    for line in open(ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"):
        row = json.loads(line)
        if row.get("session") == sid and row.get("camera") == "cam3":
            index[int(row["frame_id"])] = row["mask_path"]

    H, W = cv2.imread(str(paths[0])).shape[:2]
    texture = column_texture(H, W, seed=0)
    active_ids = np.where(active)[0].tolist()
    print("%s: %d/%d active frames" % (sid, len(active_ids), n_frames))

    batch_imgs, batch_info, batch_ids = [], [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device=device, dtype=torch.float32)
        xf = model(x)
        if xf.ndim != 2 or xf.shape[1] != HRNET_FEAT_DIM:
            raise ValueError("HRNet output %s" % (tuple(xf.shape),))
        info = torch.from_numpy(np.stack(batch_info, axis=0))
        rows = torch.cat([xf.detach().cpu().to(torch.float32), info.float()], dim=1)
        for idx, row in zip(batch_ids, rows):
            feat[idx] = row
        batch_imgs.clear(); batch_info.clear(); batch_ids.clear()

    with torch.no_grad():
        for idx in active_ids:
            rgb = read_rgb(paths[idx])
            mpath = index.get(idx)
            person = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 0
                      if mpath and Path(mpath).is_file() else np.zeros(rgb.shape[:2], bool))
            rgb = composite(rgb, person, poly, texture, bool(behind[idx]))
            if demo_dir is not None and idx in demo_frames(manifest):
                demo_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(demo_dir / ("%s_%04d_vis%.2f.jpg" % (
                    sid, idx, vis[idx]))), rgb)
            norm_img, center, scale, _, _, _ = process_image(
                rgb, bbox[idx][1:5], crop_height=CROP_H, crop_width=CROP_W)
            norm_img = np.asarray(norm_img, dtype=np.float32)
            img_h, img_w = rgb.shape[:2]
            center = np.asarray(center, dtype=np.float32)
            bbox_size = np.float32(scale) * 200.0
            info = np.asarray([
                (center[0] - img_w / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                (center[1] - img_h / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                (bbox_size - 0.24 * CLIFF_FOCAL_LENGTH) / (0.06 * CLIFF_FOCAL_LENGTH),
            ], dtype=np.float32)
            batch_imgs.append(norm_img)
            batch_info.append(info)
            batch_ids.append(idx)
            if len(batch_imgs) >= batch_size:
                flush()
        flush()

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / ("%s.pt" % sid)
    torch.save(feat.contiguous(), out_path)
    changed = int((feat != clean).any(dim=1).sum())
    print("saved %s %s changed_rows=%d" % (out_path, tuple(feat.shape), changed))


def demo_frames(manifest: dict) -> set:
    """One preview per full event plus the partial frames around it."""
    picks = set()
    vis = np.asarray(manifest["frames"]["vis"], dtype=np.float64)
    for ev in manifest.get("full_events", []):
        s, e = int(ev["start"]), int(ev["end"])
        for t in (s - 5, s, (s + e) // 2, e, e + 5):
            if 0 <= t < len(vis):
                picks.add(t)
    return picks


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_encoder(device)
    manifests = sorted(args.manifest_dir.glob("*.json"))
    for mpath in manifests:
        build_session(mpath, args.out, model, device, int(args.batch_size), args.demo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
