"""Prepare WHAM fine-tuning labels and image features.

Run in the WHAM environment (``/data/fangyuxuan/miniconda3/envs/wham``):

    python -m GVHMR.finetune.wham_prepare --fold fS5S6S11 --split train,val --gpu 2

The fold label files follow the native BEDLAMDataset schema.  SMPL is already
the target model, so no SMPL-X conversion is needed.  Camera labels are the
physical cal3 camera-space SMPL parameters from :mod:`labels`; image features
come from WHAM's own HMR2a extractor on the same staged 40 Hz frame stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import common

ROOT = common.ROOT
WORKSPACE = common.WORKSPACE
WHAM_ROOT = ROOT / "Baselines" / "WHAM"
DEFAULT_OUTPUT = WORKSPACE / "work" / "gvhmr_wham"
SEQLEN = 81
N_FRAMES = SEQLEN + 1
MAX_STEP_S = 0.05


def _ensure_wham_path() -> None:
    from .wham_env import activate

    activate()


def _root_center(betas: np.ndarray) -> np.ndarray:
    import smplx
    import torch

    from .. import config as _config

    model = smplx.create(
        str(_config.DEFAULT_SMPL_MODEL), "smpl", gender="neutral", batch_size=1, num_betas=10
    ).to("cpu")
    with torch.no_grad():
        output = model(
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 69),
            betas=torch.as_tensor(betas, dtype=torch.float32)[None],
            transl=torch.zeros(1, 3),
        )
    return output.joints[0, 0].cpu().numpy().astype(np.float64)


def _video_path(root: Path, row: dict, unit: dict, cam: int) -> Path:
    path = root / "staging" / f"{row['session_id']}_c{cam}.mp4"
    if path.is_file():
        return path
    frames = common.unit_frame_paths(row, unit, cam)
    return common.stage_video(frames, path, fps=common.MODEL_FPS)


def _extract_features(
    root: Path, row: dict, unit: dict, labels: dict, cam: int, gpu: int
) -> dict:
    import torch

    import cv2
    from lib.models.preproc.backbone.utils import process_image
    from lib.models.preproc.extractor import FeatureExtractor

    session = str(row["session_id"])
    cache = root / "wham_features" / f"{session}.pt"
    if cache.is_file():
        cached = torch.load(cache, map_location="cpu")
        if cached.get("schema") == 2:
            return cached
    features_source = torch.load(
        root / "features" / f"{session}.pt", map_location="cpu"
    )
    keypoints = features_source["kp2d"].numpy()
    boxes = np.asarray(labels["bbox_xyxy_px"], dtype=np.float64)
    width = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    bbox = np.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0,
            width * 1.05 / 200.0,
        ],
        axis=-1,
    ).astype(np.float32)
    video = _video_path(root, row, unit, cam)
    torch.cuda.set_device(0)
    extractor = FeatureExtractor("cuda", flip_eval=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"{session}: cannot open staged WHAM video")
    features = []
    init_global = []
    init_body = []
    init_betas = []
    frame_id = 0
    with torch.no_grad():
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_id >= len(bbox):
                raise RuntimeError(f"{session}: video has more frames than bbox labels")
            cx, cy, scale = bbox[frame_id]
            norm_img, _crop = process_image(
                image[..., ::-1], [cx, cy], scale, 256, 256
            )
            tensor = torch.from_numpy(norm_img).unsqueeze(0).to(extractor.device)
            features.append(extractor.model(tensor, encode=True).cpu())
            g, body, beta, _ = extractor.model(tensor, encode=False)
            init_global.append(g.detach().cpu())
            init_body.append(body.detach().cpu())
            init_betas.append(beta.detach().cpu())
            frame_id += 1
    capture.release()
    if frame_id != len(bbox):
        raise RuntimeError(f"{session}: video frame count {frame_id} != bbox {len(bbox)}")
    features = torch.cat(features, dim=0).numpy().astype(np.float32)
    init_global = torch.cat(init_global, dim=0)
    init_body = torch.cat(init_body, dim=0)
    init_betas = torch.cat(init_betas, dim=0)
    if features.shape != (len(bbox), 1024) or not np.isfinite(features).all():
        raise RuntimeError(f"{session}: WHAM features invalid {features.shape}")
    payload = {
        "schema": 2,
        "session_id": session,
        "features": torch.from_numpy(features).half(),
        "bbox_cxcys": torch.from_numpy(np.asarray(bbox, dtype=np.float32)),
        "keypoints": torch.from_numpy(np.asarray(keypoints, dtype=np.float32)),
        # Per-frame HMR2a init: later validation windows use their own first frame.
        "init_global_orient": init_global,
        "init_body_pose": init_body,
        "init_betas": init_betas,
        "video": str(video),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def prepare_session(
    root: Path, session: str, cam: int, gpu: int
) -> dict:
    rows = common.load_session_manifest()
    row = rows[session]
    date, _subject = common.session_date_subject(row)
    unit = common.load_unit(date, session, cam)
    labels = np.load(root / "labels" / f"{session}.npz", allow_pickle=False)
    valid_label = _manifest_valid(
        row,
        np.asarray(labels["canonical_index"], dtype=np.int64),
        np.asarray(labels["valid"], dtype=bool),
    )
    timestamps = np.asarray(labels["timestamp_seconds"], dtype=np.float64)
    query = common.label_times(row, unit, cam)
    world = common.resample_smpl(common.resolve_uri(row["smpl_path"]), query)
    pose_camera = np.concatenate(
        [labels["smpl_global_orient"], labels["smpl_body_pose"]], axis=1
    ).astype(np.float32)
    c_trans = np.asarray(labels["smpl_transl"], dtype=np.float32)
    betas = np.asarray(labels["smpl_betas"], dtype=np.float32)
    world_poses = np.asarray(world["poses"], dtype=np.float32)
    world_trans = np.asarray(world["trans"], dtype=np.float32)
    j_smpl = _root_center(np.asarray(world["betas"], dtype=np.float32))
    pelvis_world = world_trans + j_smpl[None]
    rotation, translation, calibration = common.camera_transform(date, cam)
    extrinsics = np.zeros((len(timestamps), 4, 4), dtype=np.float32)
    extrinsics[:, :3, :3] = rotation.astype(np.float32)
    extrinsics[:, :3, 3] = translation.astype(np.float32)
    extrinsics[:, 3, 3] = 1.0
    width, height = [int(value) for value in calibration["image_size_px"]]
    focal = (width * width + height * height) ** 0.5
    K = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsics = np.broadcast_to(K[None], (len(timestamps), 3, 3)).copy()
    res = np.broadcast_to(
        np.asarray(calibration["image_size_px"], dtype=np.float32)[None],
        (len(timestamps), 2),
    ).copy()
    wham = _extract_features(root, row, unit, labels, cam, gpu)
    features = wham["features"].float().numpy()
    bbox = wham["bbox_cxcys"].numpy()
    keypoints = wham["keypoints"].numpy()
    import torch

    init_global = wham.get("init_global_orient")
    init_body = wham.get("init_body_pose")
    init_betas = wham.get("init_betas")
    if init_global is None or init_body is None or init_betas is None:
        raise RuntimeError(f"{session}: WHAM visual init is missing")
    visual_init = torch.cat(
        [torch.as_tensor(init_global).float(), torch.as_tensor(init_body).float()], dim=1
    ).numpy()
    if len(visual_init) != len(timestamps) or len(init_betas) != len(timestamps):
        raise RuntimeError(f"{session}: visual init frame count drift")
    if not (len(features) == len(timestamps) == len(keypoints)):
        raise RuntimeError(f"{session}: WHAM feature/frame count drift")
    return {
        "row": row,
        "pose_camera": pose_camera,
        "c_trans": c_trans,
        "betas": betas,
        "w_trans": pelvis_world,
        "root": world_poses[:, :3],
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "res": res,
        "features": features,
        "kp2d": keypoints,
        "bbox": bbox,
        "timestamps": timestamps,
        "valid": valid_label,
        "visual_init_pose": visual_init,
        "visual_init_betas": torch.as_tensor(init_betas).float().numpy(),
    }


def _manifest_valid(row: dict, canonical: np.ndarray, label_valid: np.ndarray) -> np.ndarray:
    import json

    allowed = np.zeros(len(canonical), dtype=bool)
    if "valid_frame_indices" in row:
        raw = row.get("valid_frame_indices", [])
        if isinstance(raw, str):
            raw = json.loads(raw)
        if raw:
            allowed[np.isin(canonical, np.asarray(raw, dtype=np.int64))] = True
    else:
        allowed[:] = True
    raw = row.get("fake_frame_indices", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    rejected = np.zeros(len(canonical), dtype=bool)
    if raw:
        rejected[np.isin(canonical, np.asarray(raw, dtype=np.int64))] = True
    return label_valid & allowed & ~rejected


def _runs(timestamps: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, good in enumerate(valid):
        continues = (
            start is not None
            and index > 0
            and timestamps[index] - timestamps[index - 1] <= MAX_STEP_S
        )
        if good and (start is None or not continues):
            if start is not None:
                runs.append((start, index))
            start = index
        elif not good and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def build_fold_labels(
    root: Path, sessions: list[str], destination: Path, cam: int, gpu: int
) -> dict:
    arrays = {
        name: [] for name in (
            "vid", "pose", "c_trans", "betas", "w_trans", "root", "extrinsics",
            "intrinsics", "res", "features", "kp2d", "bbox", "frame_id",
            "visual_init_pose", "visual_init_betas",
        )
    }
    prepared = {}
    vid_counter = 0
    skipped = []
    short_runs = []
    for session in sessions:
        record = prepare_session(root, session, cam, gpu)
        prepared[session] = record
        runs = [
            (start, end) for start, end in _runs(record["timestamps"], record["valid"])
            if end - start >= N_FRAMES
        ]
        all_runs = _runs(record["timestamps"], record["valid"])
        dropped = [end - start for start, end in all_runs if end - start < N_FRAMES]
        if dropped:
            short_runs.append({"session_id": session, "runs": dropped})
        if not runs:
            skipped.append({
                "session_id": session,
                "valid_frames": int(record["valid"].sum()),
                "max_run": max([end - start for start, end in all_runs], default=0),
            })
            continue
        for start, end in runs:
            vid = vid_counter
            vid_counter += 1
            length = end - start
            arrays["vid"].append(np.full(length, vid, dtype=np.int64))
            arrays["pose"].append(record["pose_camera"][start:end])
            arrays["c_trans"].append(record["c_trans"][start:end])
            arrays["betas"].append(record["betas"][start:end])
            arrays["w_trans"].append(record["w_trans"][start:end])
            arrays["root"].append(record["root"][start:end])
            arrays["extrinsics"].append(record["extrinsics"][start:end])
            arrays["intrinsics"].append(record["intrinsics"][start:end])
            arrays["res"].append(record["res"][start:end])
            arrays["features"].append(record["features"][start:end])
            arrays["kp2d"].append(record["kp2d"][start:end])
            arrays["bbox"].append(record["bbox"][start:end])
            arrays["frame_id"].append(np.arange(start, end, dtype=np.int64))
            arrays["visual_init_pose"].append(record["visual_init_pose"][start:end])
            arrays["visual_init_betas"].append(record["visual_init_betas"][start:end])
    if not arrays["vid"]:
        raise RuntimeError(f"no session has a run >= {N_FRAMES} frames")
    import torch

    payload = {
        name: torch.from_numpy(np.concatenate(value, axis=0))
        for name, value in arrays.items()
    }
    for name in ("pose", "c_trans", "betas", "w_trans", "root", "extrinsics",
                 "intrinsics", "res", "features", "kp2d", "bbox",
                 "visual_init_pose", "visual_init_betas"):
        payload[name] = payload[name].float()
    import joblib

    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination)
    metadata = {
        "sessions": sessions,
        "runs": int(len(np.unique(payload["vid"]))),
        "frames": int(len(payload["vid"])),
        "destination": str(destination),
        "camera": f"camera_{cam}",
        "seqlen": SEQLEN,
        "max_step_s": MAX_STEP_S,
        "intrinsics": "estimate_K(diagonal FOV), original WHAM inference path",
        "source": "physical camera SMPL labels; world pelvis/root for trajectory fields",
        "skipped_sessions": skipped,
        "short_run_fragments": short_runs,
    }
    common.json_dump(destination.with_suffix(".json"), metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", required=True, choices=sorted(common.FOLD_SPLITS))
    parser.add_argument("--split", default="train,val")
    parser.add_argument("--train-sessions", default="")
    parser.add_argument("--val-sessions", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--support-root", default=str(common.DEFAULT_OUTPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = common.check_gpu(args.gpu)
    support = Path(args.support_root).expanduser()
    if not support.is_absolute():
        support = common.ROOT / support
    support = support.resolve()
    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        root = common.ROOT / root
    root = root.resolve()
    _ensure_wham_path()
    pools = common.fold_sessions(args.fold)
    train = (
        [value.strip() for value in args.train_sessions.split(",") if value.strip()]
        if args.train_sessions else pools["train"]
    )
    val = (
        [value.strip() for value in args.val_sessions.split(",") if value.strip()]
        if args.val_sessions else pools["val"]
    )
    tag = args.tag or args.fold
    metadata = {}
    for name, sessions in (("train", train), ("val", val)):
        if not sessions:
            continue
        destination = root / "labels" / f"{tag}_{name}.pkl"
        metadata[name] = build_fold_labels(support, sessions, destination, args.cam, int(args.gpu))
    common.json_dump(root / "labels" / f"{tag}_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
