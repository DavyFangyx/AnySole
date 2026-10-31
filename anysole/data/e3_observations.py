"""Create GT-free E3 observations from the same clean/occluded source images.

The image transformation and HRNet crop reuse E5. The original observed bbox
is retained for all conditions, while YOLO keypoints are rerun on each actual
transformed image. No motion, test shape, ground, or extrinsics enter outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from anysole.data.crop import process_image
from anysole.data.e3_grouped import E3GroupedDataset
from anysole.data.extract_hrnet import load_encoder, occlude_lower_strip, read_rgb
from anysole.data.workspace_adapter import sha256_file

COCO_JOINTS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
SMPL_JOINTS = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)


def _person_index(boxes: np.ndarray, confidence: np.ndarray, reference: np.ndarray) -> int | None:
    if not len(boxes):
        return None
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    intersection = np.maximum(np.minimum(boxes[:, 2:], reference[2:]) -
                              np.maximum(boxes[:, :2], reference[:2]), 0).prod(1)
    union = area + np.maximum(reference[2:] - reference[:2], 0).prod() - intersection
    iou = intersection / np.maximum(union, 1e-6)
    return min(range(len(boxes)), key=lambda i: (-iou[i], -confidence[i], *boxes[i].tolist()))


def extract_window(dataset: E3GroupedDataset, index: int, *, dataset_root: Path, encoder,
                   detector, device: torch.device, condition: str, batch_size: int = 8) -> tuple[dict, dict]:
    """One continuous complete window; safe inputs contain no GT motion fields."""
    fractions = {"clean": 0., "occ010": .1, "occ030": .3}
    if condition not in fractions:
        raise ValueError("unknown E3 degradation")
    fraction = fractions[condition]
    item = dataset[index]
    ri, start, end, cameras = dataset.windows[index]
    record = dataset.records[ri]
    calibration = json.loads((dataset_root / record["calibration"]["path"]).read_text())
    all_features, all_points, all_confidence, intrinsics, pixel_hashes = [], [], [], [], []
    clean_delta = 0.
    for cam_i, cam in enumerate(cameras):
        view = next(v for v in record["views"] if v["camera_id"] == cam)
        entry = dataset.cache_index["entries"][f"{record['recording_id']}/cam{cam}"]
        rows = np.asarray(view["source_row"][start:end], np.int64)
        unique = sorted(set(rows.tolist()))
        result_by_row = {}
        k = np.asarray(calibration["cameras"][f"cam{cam}"]["K"], np.float64)
        distortion = np.asarray(calibration["cameras"][f"cam{cam}"]["D"], np.float64)
        intrinsics.append(k)
        for offset in range(0, len(unique), batch_size):
            batch_rows = unique[offset:offset + batch_size]
            images, prepared, bbox_info, batch_hashes, references = [], [], [], [], []
            for row in batch_rows:
                source = view["sources"][row]
                path = dataset_root / view["directory"] / source["file"]
                if sha256_file(path) != entry["provenance"]["source_image_sha256"][row]:
                    raise ValueError(f"E3 observation RGB source hash mismatch: {path}")
                rgb = read_rgb(path)
                reference = np.asarray(entry["provenance"]["source_geometry"][row]["bbox_xyxy"], np.float32)
                if fraction:
                    rgb = occlude_lower_strip(rgb, reference, fraction, "black")
                # Both feature crop and keypoint detector consume this exact
                # transformed pixel array; the cache pins its content hash.
                batch_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                norm, center, scale, _, _, _ = process_image(rgb, reference)
                center = np.asarray(center, np.float32)
                height, width = rgb.shape[:2]
                box_size = np.float32(scale) * 200
                bbox_info.append(np.asarray([(center[0] - width / 2) / 608 * 2.8,
                    (center[1] - height / 2) / 608 * 2.8, (box_size - .24 * 608) / (.06 * 608)], np.float32))
                prepared.append(np.asarray(norm, np.float32))
                images.append(np.ascontiguousarray(rgb[:, :, ::-1]))
                references.append(reference)
            with torch.inference_mode():
                encoded = encoder(torch.from_numpy(np.stack(prepared)).to(device)).cpu().float()
                predictions = detector.predict(source=images, classes=[0], conf=.25,
                    verbose=False, device=str(device), save=False)
            features = torch.cat((encoded, torch.from_numpy(np.stack(bbox_info))), 1)
            for bi, (row, prediction) in enumerate(zip(batch_rows, predictions)):
                points = np.zeros((len(COCO_JOINTS), 2), np.float32)
                confidence = np.zeros(len(COCO_JOINTS), np.float32)
                boxes = prediction.boxes.xyxy.cpu().numpy()
                which = _person_index(boxes, prediction.boxes.conf.cpu().numpy(), references[bi])
                if which is not None and prediction.keypoints is not None:
                    raw_points = prediction.keypoints.xy[which].cpu().numpy()[list(COCO_JOINTS)]
                    confidence = prediction.keypoints.conf[which].cpu().numpy()[list(COCO_JOINTS)]
                    points = cv2.undistortPoints(raw_points[:, None].astype(np.float64), k,
                                                distortion, P=k)[:, 0].astype(np.float32)
                result_by_row[row] = (features[bi], points, confidence, batch_hashes[bi])
        values = torch.stack([result_by_row[row][0] for row in rows])
        if condition == "clean":
            clean_delta = max(clean_delta, float((values - item["views"][cam_i]["V_feat"]).abs().max()))
        all_features.append(values)
        all_points.append(np.stack([result_by_row[row][1] for row in rows]))
        all_confidence.append(np.stack([result_by_row[row][2] for row in rows]))
        pixel_hashes.append([result_by_row[row][3] for row in rows])
    output = {"V_feat": torch.stack(all_features)[:, None],
              "frame_ids": item["meta"]["target_frame_id"][None],
              "target_time_s": item["meta"]["target_time_s"][None],
              "detected_2d": torch.from_numpy(np.stack(all_points))[:, None],
              "confidence": torch.from_numpy(np.stack(all_confidence))[:, None],
              "intrinsics": torch.tensor(np.stack(intrinsics), dtype=torch.float32),
              "joint_indices": torch.tensor(SMPL_JOINTS, dtype=torch.long),
              "T_raw": item["physical"]["T_raw"][None],
              "T_phys": item["physical"]["T_phys"][None],
              "camera_ids": torch.tensor(cameras, dtype=torch.long),
              "valid": torch.ones((len(cameras), 1, end - start), dtype=torch.bool),
              "source_frame_ids": item["meta"]["source_frame_id"][:, None],
              "source_time_s": item["meta"]["source_time_s"][:, None],
              "degradation": condition}
    receipt = {"schema": "anysole.e3.observations.v1", "split": dataset.split,
               "sample_key": dataset.sample_key(index), "data_hash": dataset.manifest["data_hash"],
               "degradation": condition, "fraction": fraction,
               "transform": "existing E5 occlude_lower_strip, black full-width rows from bbox y1+(1-frac)*(y2-y1)",
               "feature_bbox": "same frozen RGB-only bbox used by original clean HRNet; bbox_info unchanged across conditions",
               "detection_source": "YOLO pose rerun on identical transformed full-image pixels",
               "pixel_coordinates": "undistorted full-image pixels; cv2.undistortPoints(originalK,D,P=originalK)",
               "same_image_pixel_sha256": pixel_hashes,
               "stats_sha256": sha256_file(dataset.root / "stats.json"),
               "clean_native_feature_max_abs_diff": clean_delta if condition == "clean" else None,
               "confident_keypoints": int((output["confidence"] > .25).sum()),
               "keypoints_total": output["confidence"].numel()}
    return output, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hrnet-module-root", type=Path, required=True)
    parser.add_argument("--hrnet-config", type=Path, required=True)
    parser.add_argument("--hrnet-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / ".ultralytics")
    from ultralytics import YOLO
    device = torch.device(args.device)
    torch.set_num_threads(2)
    encoder = load_encoder(device, args.hrnet_module_root, args.hrnet_config, args.hrnet_checkpoint)
    detector = YOLO(str(args.detector_checkpoint))
    receipts = []
    for split in ("train", "val"):
        dataset = E3GroupedDataset(args.data_root, split=split)
        for condition in ("clean", "occ010", "occ030"):
            output, receipt = extract_window(dataset, 0, dataset_root=args.dataset_root,
                encoder=encoder, detector=detector, device=device, condition=condition)
            path = args.output / f"{split}_{condition}.pt"
            torch.save(output, path)
            receipt.update(path=str(path), sha256=sha256_file(path))
            receipts.append(receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != "same_image_pixel_sha256"}), flush=True)
    manifest = {"schema": "anysole.e3.observation-smoke.v1", "status": "PASS", "receipts": receipts,
                "assets": {str(p): sha256_file(p) for p in (args.hrnet_config, args.hrnet_checkpoint, args.detector_checkpoint)},
                "full_evaluation_cache": "not extracted; these are train/val smoke windows only"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
