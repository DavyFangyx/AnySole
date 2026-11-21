"""Prepare per-session GVHMR fine-tuning support: labels, then RGB features.

Run inside the ``vb_gvhmr`` environment:

    python -m GVHMR.finetune.prepare --stage all --fold fS5S6S11 --gpu 2

Labels are written to ``<root>/labels/<session>.npz`` and image features to
``<root>/features/<session>.pt``.  Both stages are resumable; ``--force``
recomputes selected sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import common, labels as label_module


def _apply_shard(values: list[str], args) -> list[str]:
    if args.num_shards <= 0 or args.shard < 0 or args.shard >= args.num_shards:
        raise SystemExit("--shard must lie in [0, --num-shards)")
    return values[args.shard :: args.num_shards]


def _select(args) -> list[str]:
    if args.sessions:
        return _apply_shard(
            common.select_sessions(
                sessions=args.sessions.split(","), limit=args.limit
            ),
            args,
        )
    if args.all_folds:
        selected = set()
        for fold in common.FOLD_SPLITS:
            pools = common.fold_sessions(fold)
            names = [name.strip() for name in args.split.split(",") if name.strip()]
            for name in names:
                selected.update(pools[name])
        values = sorted(selected)
        values = values[: args.limit] if args.limit else values
        return _apply_shard(values, args)
    return _apply_shard(
        common.select_sessions(fold=args.fold, split=args.split, limit=args.limit),
        args,
    )


def _label_path(root: Path, session: str) -> Path:
    return root / "labels" / f"{session}.npz"


def _feature_path(root: Path, session: str) -> Path:
    return root / "features" / f"{session}.pt"


def prepare_label(root: Path, row: dict, unit: dict, args, cam: int) -> str:
    path = _label_path(root, row["session_id"])
    if path.is_file() and not args.force:
        return "skip"
    arrays, metadata = label_module.prepare_session_labels(
        row,
        unit,
        cam=cam,
        fit_steps=args.fit_steps,
        fit_batch_size=args.fit_batch_size,
        fit_learning_rate=args.fit_learning_rate,
        max_vertex_rmse_mm=args.max_vertex_rmse_mm,
    )
    label_module.write_labels(root, row["session_id"], arrays, metadata)
    return "ok"


class _Extractors:
    """Official HMR2 / ViTPose front end with local licensed weights."""

    def __init__(self) -> None:
        import torch

        from .. import config as _config

        torch.cuda.set_device(0)
        import hmr4d.utils.preproc.vitfeat_extractor as feature_module
        import hmr4d.utils.preproc.vitpose as pose_module
        from hmr4d.network.hmr2 import load_hmr2
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor

        feature_module.load_hmr2 = lambda: load_hmr2(str(_config.DEFAULT_HMR2_CHECKPOINT))
        original_build_model = pose_module.build_model
        pose_module.build_model = lambda name, ckpt_path=None: original_build_model(
            name, str(_config.DEFAULT_VITPOSE_CHECKPOINT)
        )
        self.features = Extractor(tqdm_leave=False)
        self.pose = VitPoseExtractor(tqdm_leave=False)

    def run(self, video: Path, boxes_px: np.ndarray):
        import torch

        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

        boxes = torch.as_tensor(np.asarray(boxes_px), dtype=torch.float32)
        bbx_xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
        keypoints = self.pose.extract(str(video), bbx_xys)
        features = self.features.extract_video_features(str(video), bbx_xys)
        return bbx_xys, keypoints, features


def _video_geometry(video: Path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise common.FinetuneDataError(f"cannot open staged video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0:
        raise common.FinetuneDataError(f"invalid staged video geometry: {video}")
    return width, height


def prepare_features(
    root: Path,
    row: dict,
    unit: dict,
    extractors: _Extractors,
    args,
    cam: int,
) -> str:
    import torch

    from hmr4d.utils.geo.hmr_cam import estimate_K
    from hmr4d.utils.geo_transform import compute_cam_angvel

    session = str(row["session_id"])
    path = _feature_path(root, session)
    if path.is_file() and not args.force:
        return "skip"
    date, _subject = common.session_date_subject(row)
    frame_paths = common.unit_frame_paths(row, unit, cam)
    video = common.stage_video(
        frame_paths, root / "staging" / f"{session}_c{cam}.mp4", fps=common.MODEL_FPS
    )
    width, height = _video_geometry(video)
    expected = [int(value) for value in common.calibration(date)["image_size_px"]]
    if [width, height] != expected:
        raise common.FinetuneDataError(
            f"{session}: staged video {width}x{height} != calibration {expected}"
        )
    boxes = common.bboxes_for_unit(row, unit, cam)
    if len(boxes) != len(unit["frames"]):
        raise common.FinetuneDataError(f"{session}: bbox/frame count drift")
    bbx_xys, keypoints, features = extractors.run(video, boxes)
    count = len(unit["frames"])
    if keypoints.shape != (count, 17, 3) or features.shape[0] != count:
        raise common.FinetuneDataError(
            f"{session}: extractor shapes {keypoints.shape}, {features.shape}"
        )
    if not torch.isfinite(bbx_xys).all() or not torch.isfinite(keypoints).all():
        raise common.FinetuneDataError(f"{session}: non-finite bbox or keypoints")
    if not torch.isfinite(features).all():
        raise common.FinetuneDataError(f"{session}: non-finite HMR2 features")
    canonical = common.canonical_indices(row, unit, cam)
    calibration = common.calibration(date)
    K = estimate_K(width, height).repeat(count, 1, 1)
    K_value = K[0].tolist()
    intrinsics_label = "estimate_K(diagonal FOV), original inference path"
    cam_angvel = compute_cam_angvel(torch.eye(3).repeat(count, 1, 1))
    payload = {
        "session_id": session,
        "unit_id": unit["unit_id"],
        "camera": f"camera_{cam}",
        "length": count,
        "image_size_px": [width, height],
        "bbx_xys": bbx_xys.cpu().float(),
        "kp2d": keypoints.cpu().float(),
        "f_imgseq": features.cpu().half(),
        "K_fullimg": K.cpu().float(),
        "cam_angvel": cam_angvel.cpu().float(),
        "canonical_index": torch.as_tensor(canonical, dtype=torch.long),
        "source_frame_number": torch.as_tensor(
            [int(value["frame"]) for value in unit["frames"]], dtype=torch.long
        ),
        "timestamp_seconds": torch.as_tensor(
            [float(value["timestamp_seconds"]) for value in unit["frames"]], dtype=torch.float64
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    common.json_dump(
        path.with_suffix(".json"),
        {
            "session_id": session,
            "unit_id": unit["unit_id"],
            "video": str(video),
            "frame_count": count,
            "image_size_px": [width, height],
            "bbox_source": str(common.FRONTEND_BBOX / date / _subject / session / "bbox.npz"),
            "feature_frontend": "official HMR2 + ViTPose (pinned local weights)",
            "intrinsics": intrinsics_label,
            "intrinsics_matrix": K_value,
            "model_fps": common.MODEL_FPS,
            "timing": (
                "unit wall-clock frames are fed in order; the 30 fps ffmpeg container "
                "rate is metadata only, and training clips are split at gaps > 50 ms"
            ),
            "video_sha256": common.sha256_file(video),
        },
    )
    return "ok"


def load_features(root: Path, session: str) -> dict[str, Any]:
    import torch

    path = _feature_path(Path(root).expanduser().resolve(), session)
    if not path.is_file():
        raise common.FinetuneDataError(f"prepared features are missing for {session}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("labels", "features", "all"), default="all")
    parser.add_argument("--fold", default="")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--split", default="train,val")
    parser.add_argument("--sessions", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--gpu", default="2", help="physical GPU index used by this process")
    parser.add_argument("--output-root", default=str(common.DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-batch-size", type=int, default=24)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--max-vertex-rmse-mm", type=float, default=50.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = common.check_gpu(args.gpu)
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sessions = _select(args)
    manifest = common.load_session_manifest()
    missing = [session for session in sessions if session not in manifest]
    if missing:
        raise SystemExit(f"sessions absent from the manifest: {missing}")
    extractors = None
    summary = {"labels": {}, "features": {}}
    failures = []
    for position, session in enumerate(sessions, 1):
        row = manifest[session]
        date, _subject = common.session_date_subject(row)
        unit = common.load_unit(date, session, args.cam)
        label_ok = True
        if args.stage in ("labels", "all"):
            try:
                status = prepare_label(root, row, unit, args, args.cam)
            except Exception as exc:  # noqa: BLE001 — report and continue the shard
                status = f"FAILED: {type(exc).__name__}: {exc}"
                label_ok = False
                failures.append(("labels", session, str(exc)))
            summary["labels"][session] = status
            print(f"[{position}/{len(sessions)}] labels {session}: {status}", flush=True)
        if args.stage in ("features", "all") and label_ok:
            try:
                if extractors is None:
                    original = Path.cwd()
                    os.chdir(common.ROOT / "Baselines" / "GVHMR")
                    try:
                        extractors = _Extractors()
                    finally:
                        os.chdir(original)
                status = prepare_features(root, row, unit, extractors, args, args.cam)
            except Exception as exc:  # noqa: BLE001 — report and continue the shard
                status = f"FAILED: {type(exc).__name__}: {exc}"
                failures.append(("features", session, str(exc)))
            summary["features"][session] = status
            print(f"[{position}/{len(sessions)}] features {session}: {status}", flush=True)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard}"
    output = root / f"prepare_{args.stage}{suffix}.json"
    common.json_dump(output, {"sessions": sessions, "status": summary, "failures": failures})
    print(json.dumps({key: {"ok": sum(v == "ok" for v in value.values()),
                            "skip": sum(v == "skip" for v in value.values()),
                            "failed": sum(v.startswith("FAILED") for v in value.values())}
                      for key, value in summary.items()}, sort_keys=True))
    print(f"summary: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
