#!/usr/bin/env python3
"""Produce the single-person CLIFF NPZ for PoseTransOpt (V2 contract).

One NPZ per canonical session under
``model-input://PoseTransOpt/adapter_v1/<date>/<subject>/<session>/``:

    CLIFF_results.npz
    ├── frame_id        int64[n_frames]      unique, sorted, one row per frame
    ├── pose            float32[n,72]        axis-angle (rotvec)
    ├── shape           float32[n,10]        SMPL betas
    ├── global_t        float32[n,3]         full-frame camera translation
    ├── pred_joints     float32[n,J,3]
    ├── focal_l         float32[n]
    ├── valid           uint8[n]
    ├── bbox_xyxy_px    float32[n,4]         mask-recomputed single-person bbox
    ├── time_error_s    float64[n]           |visual_time - mask csv time|
    ├── mask_area       int64[n]
    ├── conf            float32[n]           SAM3.1 confidence from the csv
    ├── mask_path       <U[n]
    ├── imgname         <U[n]
    └── provenance strings / source hashes

Per the single-person frontend contract, the bbox of every canonical frame
is recomputed from the actual SAM3.1 binary mask (the csv bbox column is
audit-only and may be zero for ``skipped_existing`` rows); mask rows are
joined to canonical frames by ``visual_time_s`` nearest-neighbour with a
20ms tolerance, never by row index.  A frame whose mask file is missing or
empty, or whose time error exceeds 20ms, is marked ``valid=0`` (zero-filled
row) -- there is no fallback to any multi-person detector, so downstream
betas averaging/smoothing never mixes bystander shapes into the subject.

Existing NPZ files are validated (row count, unique frame ids) and rebuilt
when they violate the contract; existence alone never skips a rebuild.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

CLIFF_ROOT = WORKSPACE / "assets/third_party/CLIFF"
CKPT_DEFAULT = CLIFF_ROOT / "data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
MASK_ROOT = Path("/data/lizhe/projects/Tactile/3_Result/processed/rgb_human_masks")
SMPL_NEUTRAL = WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl"
OUTPUT_ROOT = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1"
TOLERANCE_S = 0.020


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_contract_ok(path: Path, n_frames: int) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        data = np.load(path, allow_pickle=False)
        frame_id = np.asarray(data["frame_id"])
        if len(frame_id) != n_frames:
            return f"row_count={len(frame_id)}!={n_frames}"
        if not np.array_equal(frame_id, np.arange(n_frames, dtype=np.int64)):
            return "frame_ids_not_canonical"
        for key in ("pose", "shape", "global_t", "valid"):
            if np.asarray(data[key]).shape[0] != n_frames:
                return f"{key}_row_count_mismatch"
        return None
    except Exception as exc:  # noqa: BLE001
        return f"read_error={type(exc).__name__}"


def mask_rows(session: str) -> tuple[list[dict], dict[str, str]]:
    root = MASK_ROOT / session
    index_path = root / "masks_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing mask index: {index_path}")
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if str(row.get("camera", "")) in {"3", "cam3"}]
    rows.sort(key=lambda row: float(row["visual_time_s"]))
    sources = {"masks_index": sha256_file(index_path)}
    for name in ("segmentation_manifest.json", "failures.csv"):
        path = root / name
        if path.is_file():
            sources[name] = sha256_file(path)
    return rows, sources


def match_masks(rows: list[dict], visual_times: np.ndarray) -> tuple[list[dict | None], np.ndarray]:
    """Nearest-neighbour join on visual_time_s with a 20ms tolerance."""
    if not rows:
        raise ValueError("no camera-3 mask rows")
    mask_times = np.asarray([float(r["visual_time_s"]) for r in rows],
                            dtype=np.float64)
    order = np.searchsorted(mask_times, visual_times)
    order = np.clip(order, 1, len(mask_times) - 1)
    best = np.where(
        np.abs(mask_times[order] - visual_times)
        < np.abs(mask_times[order - 1] - visual_times),
        order, order - 1)
    error = np.abs(mask_times[best] - visual_times)
    matched = [rows[int(i)] if error[k] <= TOLERANCE_S else None
               for k, i in enumerate(best)]
    return matched, error


def mask_bbox(mask_path: Path) -> np.ndarray | None:
    """Bbox of the nonzero binary mask, in mask pixel coordinates."""
    from PIL import Image
    if not mask_path.is_file():
        return None
    image = np.asarray(Image.open(mask_path))
    ys, xs = np.nonzero(image > 0)
    if xs.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                      dtype=np.float32)


def run_session(session: str, row: dict, ckpt: str, backbone: str,
                batch_size: int, device: str, force: bool) -> dict:
    import cv2
    import smplx
    import torch
    from scipy.spatial.transform import Rotation as SciRotation
    from torch.utils.data import DataLoader

    if str(CLIFF_ROOT) not in sys.path:
        sys.path.insert(0, str(CLIFF_ROOT))
    from common import constants
    from common.mocap_dataset import MocapDataset
    from common.utils import cam_crop2full, strip_prefix_if_present
    if backbone == "hr48":
        from models.cliff_hr48.cliff import CLIFF as CliffModel
    else:
        from models.cliff_res50.cliff import CLIFF as CliffModel

    date, subject, sid = session_parts(row)
    n = int(row["n_frames"])
    out_dir = OUTPUT_ROOT / date / subject / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "CLIFF_results.npz"
    reason = npz_contract_ok(output, n)
    if reason is None and not force:
        return {"session": sid, "status": "ok_existing", "frames": n}

    frames = np.load(FACTS_ROOT / date / subject / sid / "frames.npz")
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    visual = np.asarray(frames["visual_time_s"], dtype=np.float64)
    shared_valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    if len(frame_ids) != n or not np.array_equal(
            frame_ids, np.arange(n, dtype=np.int64)):
        raise ValueError(f"{sid}: shared frame ids != canonical frames")

    csv_rows, sources = mask_rows(sid)
    matched, time_error = match_masks(csv_rows, visual)

    rgb_dir = FACTS_ROOT / date / subject / sid / "rgb"
    images_by_frame = sorted(
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(images_by_frame) != n:
        raise ValueError(f"{sid}: RGB count {len(images_by_frame)} != {n}")

    # per-frame mask-recomputed bbox; validity = shared valid AND mask
    # present AND non-empty AND within the 20ms tolerance
    bboxes = np.zeros((n, 4), dtype=np.float32)
    mask_area = np.zeros(n, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float32)
    mask_paths = np.asarray(["" for _ in range(n)])
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        item = matched[i]
        if item is None or not shared_valid[i]:
            continue
        path = Path(str(item.get("mask_path", "") or ""))
        if not path.is_file():
            continue
        bbox = mask_bbox(path)
        if bbox is None:
            continue
        bboxes[i] = bbox
        mask_area[i] = int(float(item.get("mask_area", 0) or 0))
        conf[i] = float(item.get("conf", 0) or 0)
        mask_paths[i] = str(path)
        valid[i] = True

    # Single-person contract: every valid canonical frame contributes
    # exactly one detection row (its mask-recomputed bbox); invalid frames
    # are never sent to a multi-person detector.
    valid_indices = np.flatnonzero(valid).tolist()
    images = [cv2.imread(str(images_by_frame[i])) for i in valid_indices]
    if any(image is None for image in images):
        raise ValueError(f"{sid}: failed to read an RGB frame")
    detections = np.zeros((len(images), 8), dtype=np.float32)
    for row_i, frame_i in enumerate(valid_indices):
        x1, y1, x2, y2 = bboxes[frame_i]
        detections[row_i] = [row_i, x1, y1, x2, y2, 1.0, 1.0, 0.0]

    torch_device = torch.device(device if device else
                                ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CliffModel(constants.SMPL_MEAN_PARAMS).to(torch_device).eval()
    state = torch.load(ckpt, map_location=torch_device)["model"]
    model.load_state_dict(strip_prefix_if_present(state, prefix="module."), strict=True)
    smpl_model = smplx.create(str(SMPL_NEUTRAL), "smpl", gender="neutral").to(torch_device)
    dataset = MocapDataset(images, detections)
    loader = DataLoader(dataset, batch_size=min(batch_size, max(len(dataset), 1)),
                        num_workers=0)
    poses, betas, translations, joints, focals = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            norm_img = batch["norm_img"].to(torch_device).float()
            center = batch["center"].to(torch_device).float()
            scale = batch["scale"].to(torch_device).float()
            img_h = batch["img_h"].to(torch_device).float()
            img_w = batch["img_w"].to(torch_device).float()
            focal = batch["focal_length"].to(torch_device).float()
            cx, cy, box = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., box], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal.unsqueeze(-1) * 2.8
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal) / (0.06 * focal)
            pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info)
            full_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_shape, focal)
            out = smpl_model(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                             global_orient=pred_rotmat[:, [0]], pose2rot=False,
                             transl=pred_cam_full)
            pose = SciRotation.from_matrix(
                pred_rotmat.detach().cpu().numpy().reshape(-1, 3, 3)
            ).as_rotvec().reshape(-1, 72)
            poses.extend(pose)
            betas.extend(pred_betas.cpu().numpy())
            translations.extend(pred_cam_full.cpu().numpy())
            joints.extend(out.joints.cpu().numpy())
            focals.extend(focal.cpu().numpy())

    # scatter back to one row per canonical frame; invalid rows are
    # zero-filled with valid=0
    pose_full = np.zeros((n, 72), dtype=np.float32)
    shape_full = np.zeros((n, 10), dtype=np.float32)
    trans_full = np.zeros((n, 3), dtype=np.float32)
    joint_shape = np.asarray(joints).shape[1:] if joints else (45, 3)
    joints_full = np.zeros((n,) + joint_shape, dtype=np.float32)
    focal_full = np.zeros(n, dtype=np.float32)
    if valid_indices:
        pose_full[valid_indices] = np.asarray(poses)
        shape_full[valid_indices] = np.asarray(betas)
        trans_full[valid_indices] = np.asarray(translations)
        joints_full[valid_indices] = np.asarray(joints)
        focal_full[valid_indices] = np.asarray(focals)

    np.savez(
        output,
        frame_id=np.arange(n, dtype=np.int64),
        pose=pose_full, shape=shape_full, global_t=trans_full,
        pred_joints=joints_full, focal_l=focal_full,
        valid=valid.astype(np.uint8),
        bbox_xyxy_px=bboxes, time_error_s=time_error,
        mask_area=mask_area, conf=conf, mask_path=mask_paths,
        imgname=np.asarray([str(p) for p in images_by_frame]),
        bbox_source=np.asarray("SAM3.1 unique-subject binary mask; bbox "
                               "recomputed from the mask PNG per frame; "
                               "mask rows joined by visual_time_s (<=20ms)"),
        mask_source_hashes=np.asarray(json.dumps(sources)),
        detection_persons=np.asarray("single-person; no multi-person detector rows"),
        ckpt=np.asarray(str(ckpt)),
    )
    return {"session": sid, "status": "rebuilt" if reason else "built",
            "frames": n, "mask_valid": int(valid.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    parser.add_argument("--backbone", choices=("hr48", "res50"), default="hr48")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    rows = read_manifest()
    sessions = [args.session] if args.session else split_sessions(args.split)
    sessions = sessions[args.shard::args.shards]
    result = [run_session(sid, rows[sid], str(ckpt.resolve()), args.backbone,
                          args.batch_size, args.device, args.force)
              for sid in sessions]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
