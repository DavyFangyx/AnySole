#!/usr/bin/env python3
"""Build MotionPRO E5 degradation input trees (occ / gap / scene).

Degradation only touches the visual frontend (``feature_hrnet.pth``, 2048-d);
pressure / smpl / contact / keypoints stay bit-identical to the clean
``adapter_v3_bvh24`` tree (symlinked).  The degradation recipes are shared
with the V8 E5 axes:

  occ   ``anysole.data.extract_hrnet.occlude_lower_strip`` (bbox-bottom strip,
        black fill; bbox untouched), MotionPRO-native CLIFF-HRNet48 crop.
  gap   ``anysole.data.make_gap_cache.build_gap_array`` freeze schedule
        (cycle=100, seed=0, per-session crc32 phase) applied to the 2048-d rows.
  scene ``anysole.data.make_scene_occ_cache`` column composite on the
        full-run manifests (active frames rerun, others copy clean rows).

Output trees (one per level)::

  model_inputs/MotionPRO/adapter_v3_bvh24_occNNN/cam3/<date>/<sub>/<sid>/
  model_inputs/MotionPRO/adapter_v3_bvh24_gap_rNNN/cam3/...
  model_inputs/MotionPRO/adapter_v3_bvh24_scene/cam3/...

Sessions = the four-fold test union (98 sessions, E5 评估会话集) for occ/gap;
the sceneocc split session set for scene.  Session paths that are already
built are skipped unless --force.

Usage (touch_gait env, one GPU):
  python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_e5_variants --build occ
  python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_e5_variants --build gap
  python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_e5_variants --build scene
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT / "AnysoleWorkspace"
for path in (ROOT, ROOT / "Baselines/MotionPRO"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# MotionPRO-native frontend (import chdirs to Baselines/MotionPRO — fine, the
# rest of this script uses absolute paths).
from lib.util.gen_image_feature import (  # noqa: E402
    cliff_hr48, process_image,
)
from anysole.data.extract_hrnet import occlude_lower_strip  # noqa: E402
from anysole.data.make_gap_cache import (  # noqa: E402
    CYCLE_DEFAULT, SEED_DEFAULT, build_gap_array,
)
from anysole.data.make_scene_occ_cache import (  # noqa: E402
    column_texture, composite, polygon_mask,
)

V3_BASE = WORKSPACE / "model_inputs/MotionPRO/adapter_v3_bvh24/cam3"
V4_BASE = WORKSPACE / "model_inputs/MotionPRO/adapter_v4_anysolebbox/cam3"
BASE = V3_BASE
V1 = WORKSPACE / "model_inputs/MotionPRO/adapter_v1/cam3"
# 2026-11-12 裁定：遮挡条带锚到 V8/WHAM/GVHMR 共用的 AnySole bbox_cam3，
# 保证退化画面逐位一致；v4 基座连 crop 也用 AnySole bbox（--crop-anysole）。
ANYSOLE_BBOX = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet/bbox_cam3"
CROP_ANYSOLE = False  # v4: crop also uses AnySole bbox
OUT = WORKSPACE / "model_inputs/MotionPRO"
OCC_LEVELS = ("010", "020", "030", "040", "050", "060", "070", "080", "090")
GAP_LEVELS = ("010", "020", "040", "060")
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
FOLD_SPLITS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12",
    "fS5S11S12": "splits_test_S5S11S12_val_S13",
    "fS6S7": "splits_test_S6S7_val_S14",
    "fS5S6S8": "splits_test_S5S6S8_val_S7",
}
SCENE_MANIFESTS = ROOT / "results/diagnostics/20261107_e5_sceneocc_full/manifests"
SAM_INDEX = ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"
CKPT = WORKSPACE / "assets/third_party/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
SMPL_MEAN = WORKSPACE / "assets/third_party/smpl/smpl_mean_params.npz"
# MotionPRO native bbox_info normalization (ImageDataset: focal_length = 608).
FOCAL = 608.0

UNION_SESSIONS: list[str] = []


def union_sessions() -> list[str]:
    global UNION_SESSIONS
    if UNION_SESSIONS:
        return UNION_SESSIONS
    seen, out = set(), []
    for fold in FOLDS:
        path = WORKSPACE / "protocol/splits/default" / f"{FOLD_SPLITS[fold]}.csv"
        for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
            sid = (row.get("test") or "").strip()
            if sid and sid != "S12102" and sid not in seen:
                seen.add(sid)
                out.append(sid)
    UNION_SESSIONS = out
    return out


def scene_sessions() -> list[str]:
    seen, out = set(), []
    for fold in FOLDS:
        path = WORKSPACE / "protocol/splits/default" / f"splits_sceneocc_{fold}.csv"
        for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
            sid = (row.get("test") or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def session_dirs(sid: str) -> tuple[Path, Path, Path]:
    """(base_dir, v1_dir, out_date_sub) for one session."""
    matches = sorted(path.parent for path in BASE.glob(f"*/*/{sid}/feature_hrnet.pth"))
    if not matches:
        raise FileNotFoundError(sid)
    base_dir = matches[0]
    date, subj = base_dir.parts[-3], base_dir.parts[-2]
    v1_dir = V1 / date / subj / sid
    return base_dir, v1_dir, Path(date) / subj / sid


def link_inputs(base_dir: Path, out_dir: Path, feature: torch.Tensor,
                force: bool) -> None:
    """Level tree: feature + symlinks for every other input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "feature_hrnet.pth"
    if feat_path.is_file() and not force:
        raise FileExistsError(feat_path)
    torch.save(feature, feat_path)
    for name in ("pressure.npz", "smpl.npy", "contact.npy", "keypoints.npy",
                 "frame_id.npy", "bbox.npy"):
        src = base_dir / name
        if name == "bbox.npy" and not src.is_file():
            src = V1 / base_dir.parts[-3] / base_dir.parts[-2] / base_dir.name / name
        if not src.is_file():
            raise FileNotFoundError(src)
        link = out_dir / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(os.path.relpath(src, out_dir))
    color_src = base_dir / "color"
    color_link = out_dir / "color"
    if color_src.is_dir() and not (color_link.exists() or color_link.is_symlink()):
        color_link.symlink_to(os.path.relpath(color_src, out_dir))


def load_encoder(device: torch.device):
    model = torch.nn.DataParallel(
        cliff_hr48(smpl_mean_params=SMPL_MEAN)).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def read_rgb(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[:, :, ::-1]


def color_frames(base_dir: Path, n: int) -> list[Path]:
    color_dir = base_dir / "color"
    paths = [color_dir / f"{i:06d}.jpg" for i in range(n)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing[:3])
    return paths


def encode_frames(device, model, base_dir, bbox, paths, mask_fn=None,
                  frame_ids=None, batch: int = 16, strip_bbox=None):
    """MotionPRO-native crop + CLIFF encode.

    ``bbox`` drives the native crop (MotionPRO private bbox); ``strip_bbox``
    (AnySole bbox_cam3) drives the occlusion strip when ``mask_fn`` is given,
    so the degraded picture matches the V8 E5 axis.
    mask_fn(rgb, strip_box, frame_id) -> rgb.
    """
    frames = []
    info_rows = []
    with torch.no_grad():
        for pos, path in enumerate(paths):
            fid = int(frame_ids[pos]) if frame_ids is not None else pos
            rgb = read_rgb(path)
            box = bbox[pos][1:5]
            if mask_fn is not None:
                strip_box = (strip_bbox[pos][1:5] if strip_bbox is not None else box)
                rgb = mask_fn(rgb, strip_box, fid)
            norm, center, scale, _, _, _ = process_image(rgb, box)
            norm = np.asarray(norm, dtype=np.float32)
            img_h, img_w = rgb.shape[:2]
            center = np.asarray(center, dtype=np.float32)
            b = float(np.asarray(scale, dtype=np.float32)) * 200.0
            info = np.asarray([
                (center[0] - img_w / 2.0) / FOCAL * 2.8,
                (center[1] - img_h / 2.0) / FOCAL * 2.8,
                (b - 0.24 * FOCAL) / (0.06 * FOCAL),
            ], dtype=np.float32)
            frames.append(norm)
            info_rows.append(info)
    feats = []
    for start in range(0, len(frames), batch):
        x = torch.from_numpy(np.stack(frames[start:start + batch])).to(device)
        info = torch.from_numpy(np.stack(info_rows[start:start + batch])).to(device)
        feats.append(model(x, info).detach().cpu())
    return torch.cat(feats, dim=0)


def build_occ(device, model, force: bool, level: str = "") -> None:
    levels = [level] if level else OCC_LEVELS
    for lv in levels:
        frac = int(lv) / 100.0
        tree = OUT / f"{BASE.parent.name}_occ{lv}/cam3"
        for sid in union_sessions():
            base_dir, v1_dir, rel = session_dirs(sid)
            out_dir = tree / rel
            if (out_dir / "feature_hrnet.pth").is_file() and not force:
                continue
            clean = torch.load(base_dir / "feature_hrnet.pth", map_location="cpu")
            bbox = np.load(ANYSOLE_BBOX / f"{sid}.npy") if CROP_ANYSOLE \
                else np.load(v1_dir / "bbox.npy")
            anysole_bbox = np.load(ANYSOLE_BBOX / f"{sid}.npy")
            paths = color_frames(base_dir, len(bbox))
            feat = encode_frames(
                device, model, base_dir, bbox, paths,
                mask_fn=lambda rgb, box, idx: occlude_lower_strip(
                    rgb, box, frac, "black"),
                strip_bbox=anysole_bbox)
            if tuple(feat.shape) != tuple(clean.shape):
                raise ValueError(f"{sid} occ{lv}: {tuple(feat.shape)} vs clean {tuple(clean.shape)}")
            link_inputs(base_dir, out_dir, feat, force)
            print(f"[occ{lv}] {sid} {tuple(feat.shape)}", flush=True)


def build_gap(force: bool, level: str = "") -> None:
    levels = [level] if level else GAP_LEVELS
    for lv in levels:
        rho = int(lv)
        tree = OUT / f"{BASE.parent.name}_gap_r{lv}/cam3"
        for sid in union_sessions():
            base_dir, v1_dir, rel = session_dirs(sid)
            out_dir = tree / rel
            if (out_dir / "feature_hrnet.pth").is_file() and not force:
                continue
            clean = torch.load(base_dir / "feature_hrnet.pth", map_location="cpu")
            gapped = build_gap_array(clean.numpy(), rho, CYCLE_DEFAULT, SEED_DEFAULT, sid)
            link_inputs(base_dir, out_dir, torch.from_numpy(gapped), force)
            print(f"[gap{lv}] {sid}", flush=True)


def build_scene(device, model, force: bool) -> None:
    tree = OUT / f"{BASE.parent.name}_scene/cam3"
    sam = {}
    if SAM_INDEX.is_file():
        for line in SAM_INDEX.read_text().splitlines():
            row = json.loads(line)
            if row.get("camera") == "cam3":
                sam.setdefault(row["session"], {})[int(row["frame_id"])] = row["mask_path"]
    for sid in scene_sessions():
        manifest_path = SCENE_MANIFESTS / f"{sid}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_dir, v1_dir, rel = session_dirs(sid)
        out_dir = tree / rel
        if (out_dir / "feature_hrnet.pth").is_file() and not force:
            continue
        clean = torch.load(base_dir / "feature_hrnet.pth", map_location="cpu")
        n = int(manifest["n_frames"])
        bbox = np.load(ANYSOLE_BBOX / f"{sid}.npy") if CROP_ANYSOLE \
            else np.load(v1_dir / "bbox.npy")
        paths = color_frames(base_dir, n)
        freq = manifest["frames"]
        active = np.asarray(freq["active"], dtype=bool)
        behind = np.asarray(freq["behind"], dtype=bool)
        poly = np.asarray(manifest["column_polygon_xy"], dtype=np.float32)
        import cv2
        h, w = cv2.imread(str(paths[0])).shape[:2]
        texture = column_texture(h, w, seed=0)
        index = sam.get(sid, {})
        active_ids = np.where(active)[0].tolist()

        def mask_fn(rgb, box, idx):
            mpath = index.get(int(idx))
            person = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 0
                      if mpath and Path(mpath).is_file()
                      else np.zeros(rgb.shape[:2], bool))
            return composite(rgb, person, poly, texture, bool(behind[idx]))

        feat = clean.clone()
        if active_ids:
            sub_bbox = bbox[active_ids]
            sub_paths = [paths[i] for i in active_ids]
            rows = encode_frames(device, model, base_dir, sub_bbox, sub_paths,
                                 mask_fn, frame_ids=active_ids)
            for pos, idx in enumerate(active_ids):
                feat[idx] = rows[pos]
        link_inputs(base_dir, out_dir, feat, force)
        print(f"[scene] {sid} active={len(active_ids)}/{n}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", choices=("occ", "gap", "scene", "all"), default="all")
    parser.add_argument("--base", choices=("v3", "v4"), default="v3",
                        help="clean feature base tree (v4 = AnySole-bbox features)")
    parser.add_argument("--crop-anysole", action="store_true",
                        help="crop with the AnySole bbox (v4 pipeline)")
    parser.add_argument("--level", default="",
                        help="occ: single level (e.g. 010); gap: single rho")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--session", default="", help="smoke: single session id")
    args = parser.parse_args()

    global UNION_SESSIONS
    if args.session:
        UNION_SESSIONS = [args.session]

    global BASE, CROP_ANYSOLE
    BASE = V4_BASE if args.base == "v4" else V3_BASE
    CROP_ANYSOLE = args.crop_anysole
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.build in ("occ", "scene", "all"):
        model = load_encoder(device)
    if args.build in ("occ", "all"):
        build_occ(device, model, args.force, args.level)
    if args.build in ("gap", "all"):
        build_gap(args.force, args.level)
    if args.build in ("scene", "all"):
        build_scene(device, model, args.force)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
