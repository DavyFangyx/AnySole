#!/usr/bin/env python3
"""Run the CLIFF HPS stage for MMVP sessions and fan out its NPZ result.

The current workspace lacks CLIFF's bundled YOLOv3 detector.  The supported
fallback consumes MotionPRO's RGB-only YOLOX ``bbox.npy`` and records that
substitution in the NPZ metadata; those outputs are adapted smoke results,
not strict official-detector reproduction results.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
CLIFF_ROOT = WORKSPACE / "dependencies/CLIFF"
CKPT_DEFAULT = WORKSPACE / "dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"


def rows() -> dict[str, dict]:
    result = {}
    for line in (WORKSPACE / "manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["session_id"]] = row
    return result


def sessions(split: str) -> list[str]:
    cols = ("train", "val", "test") if split == "all" else (split,)
    result = []
    with (WORKSPACE / "splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(value for value in result if value))


def info(row: dict) -> tuple[str, str, str]:
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    return tuple(parts[cam + 1 : cam + 4])


def run_from_existing_bboxes(session: str, row: dict, work: Path, ckpt: str,
                             backbone: str, batch_size: int,
                             max_frames: int = 0, output_path: Path | None = None) -> Path:
    """Run CLIFF directly with MotionPRO's existing detector boxes.

    The released CLIFF folder does not contain its optional YOLO submodule in
    this workspace.  MotionPRO already has the same per-frame person boxes,
    so use those as the detector interface and keep CLIFF's HPS model intact.
    """
    import cv2
    import numpy as np
    import smplx
    import torch
    from scipy.spatial.transform import Rotation as SciRotation
    from torch.utils.data import DataLoader

    if str(CLIFF_ROOT) not in sys.path:
        sys.path.insert(0, str(CLIFF_ROOT))
    from common import constants
    from common.mocap_dataset import MocapDataset
    from common.utils import cam_crop2full, estimate_focal_length, strip_prefix_if_present
    if backbone == "hr48":
        from models.cliff_hr48.cliff import CLIFF as CliffModel
    else:
        from models.cliff_res50.cliff import CLIFF as CliffModel

    date, subject, _ = info(row)
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    seq_dir = WORKSPACE / "derived/MotionPRO/sequences/cam3" / date / subject / session
    bbox_path = seq_dir / "bbox.npy"
    if not bbox_path.is_file():
        raise FileNotFoundError(f"missing existing detector boxes: {bbox_path}")
    all_image_paths = sorted(p for p in (work / "imgs").iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    image_paths = all_image_paths
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    images = [cv2.imread(str(path)) for path in image_paths]
    if any(image is None for image in images):
        raise ValueError(f"failed to read an RGB frame for {session}")
    detections = np.asarray(np.load(bbox_path), dtype=np.float32)
    if detections.ndim != 2 or detections.shape[0] != len(all_image_paths) or detections.shape[1] < 5:
        raise ValueError(f"bbox/RGB mismatch for {session}: {detections.shape} vs {len(all_image_paths)}")
    detections = detections[:len(images), :8]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CliffModel(constants.SMPL_MEAN_PARAMS).to(device).eval()
    state = torch.load(ckpt, map_location=device)["model"]
    model.load_state_dict(strip_prefix_if_present(state, prefix="module."), strict=True)
    smpl_file = WORKSPACE / "dependencies/smpl/SMPL_NEUTRAL.pkl"
    smpl_model = smplx.create(str(smpl_file), "smpl", gender="neutral").to(device)
    dataset = MocapDataset(images, detections)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), num_workers=0)
    poses, betas, translations, joints, focals = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            norm_img = batch["norm_img"].to(device).float()
            center = batch["center"].to(device).float()
            scale = batch["scale"].to(device).float()
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()
            focal = batch["focal_length"].to(device).float()
            cx, cy, box = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., box], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal.unsqueeze(-1) * 2.8
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal) / (0.06 * focal)
            pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info)
            full_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_shape, focal)
            output = smpl_model(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, [0]], pose2rot=False,
                                transl=pred_cam_full)
            # torchgeometry's legacy conversion uses bool subtraction and
            # breaks on torch 2.4; SciPy performs the same rotmat->axis-angle
            # conversion without changing the CLIFF model output.
            pose = SciRotation.from_matrix(
                pred_rotmat.detach().cpu().numpy().reshape(-1, 3, 3)
            ).as_rotvec().reshape(-1, 72)
            poses.extend(pose)
            betas.extend(pred_betas.cpu().numpy())
            translations.extend(pred_cam_full.cpu().numpy())
            joints.extend(output.joints.cpu().numpy())
            focals.extend(focal.cpu().numpy())
    output = output_path or work / f"{session}_cliff_{backbone}.npz"
    np.savez(output, imgname=np.asarray([str(p) for p in image_paths]),
             pose=np.asarray(poses), shape=np.asarray(betas),
             global_t=np.asarray(translations), pred_joints=np.asarray(joints),
             focal_l=np.asarray(focals), detection_all=detections,
             bbox_source=np.asarray("MotionPRO bbox.npy generated by MMDetection YOLOX"),
             reproduction_status=np.asarray("adapted_detector_smoke"))
    return output


def run_official_demo(work: Path, ckpt: str, backbone: str, batch_size: int) -> Path:
    """Run the original CLIFF demo, including its YOLOv3 detector."""
    session = work.name
    output = work / f"{session}_cliff_{backbone}.npz"
    command = [
        sys.executable, str(CLIFF_ROOT / "demo.py"),
        "--input_type", "folder", "--input_path", str(work),
        "--ckpt", str(ckpt), "--backbone", backbone,
        "--batch_size", str(batch_size), "--save_results", "--pose_format", "aa",
    ]
    subprocess.run(command, cwd=CLIFF_ROOT, check=True)
    return output


def run_one(session: str, row: dict, args: argparse.Namespace) -> dict:
    date, subject, _ = info(row)
    fpp_session = WORKSPACE / "derived/VP-MoCap" / date / subject / session
    color = fpp_session / "color"
    if not color.is_dir() or not any(color.iterdir()):
        raise FileNotFoundError(f"{session}: missing prepared RGB directory {color}")
    work = WORKSPACE / "derived/VP-MoCap/_cliff_inputs" / date / subject / session
    work.mkdir(parents=True, exist_ok=True)
    imgs = work / "imgs"
    if imgs.is_symlink() or imgs.exists():
        if imgs.is_symlink() and imgs.resolve() == color.resolve():
            pass
        elif args.force:
            imgs.unlink() if imgs.is_symlink() or imgs.is_file() else shutil.rmtree(imgs)
        else:
            raise FileExistsError(f"CLIFF input exists: {imgs}; use --force")
    if not imgs.exists():
        imgs.symlink_to(color, target_is_directory=True)
    if args.max_frames > 0:
        output = work / f"{session}_cliff_{args.backbone}_smoke.npz"
    else:
        output = work / f"{session}_cliff_{args.backbone}.npz"
    if not output.is_file() or args.force:
        if args.detector == "official_yolov3":
            if args.max_frames > 0:
                raise ValueError("--max-frames is only supported by the adapted bbox smoke path")
            run_official_demo(work, args.ckpt, args.backbone, args.batch_size)
        else:
            run_from_existing_bboxes(session, row, args.ckpt,
                                     args.backbone, args.batch_size,
                                     max_frames=args.max_frames, output_path=output)
    if not output.is_file():
        raise FileNotFoundError(f"CLIFF did not write {output}")
    if args.max_frames > 0:
        return {"session": session, "smoke": str(output), "frames": args.max_frames}
    shutil.copy2(output, fpp_session / "CLIFF_results.npz")
    toolkit_root = ROOT / "results/offline/pressure_toolkit" / session / date / subject / session
    toolkit_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, toolkit_root / f"{session}_cliff_{args.backbone}.npz")
    return {"session": session, "cliff": str(output), "fpp": str(fpp_session / "CLIFF_results.npz"), "toolkit": str(toolkit_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    parser.add_argument("--backbone", choices=("hr48", "res50"), default="hr48")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--detector", choices=("official_yolov3", "motionpro_yolox"),
                        default="official_yolov3")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="CPU smoke limit; partial output is never copied to canonical inputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    args.ckpt = str(ckpt.resolve())
    if not ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    manifest = rows()
    selected = [args.session] if args.session else sessions(args.split)
    print(json.dumps([run_one(sid, manifest[sid], args) for sid in selected], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
