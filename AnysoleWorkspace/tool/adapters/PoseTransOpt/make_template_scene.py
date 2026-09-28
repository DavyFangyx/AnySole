#!/usr/bin/env python3
"""Produce PoseTransOpt template_scene_rgbd.npy (first-valid-frame RGB-D).

The scene template is the PoseTransOpt static observation: an RGB image and
a metric depth map of the walking plane, stored as
``{'rgb': uint8[H,W,3], 'depth': float32[H,W]}`` (metres) for the first
canonical frame that is ``valid=1`` and ``fake=0`` in shared facts.

Depth comes from Apple Depth Pro (transformers), inferred at the RGB
resolution so the PoseTransOpt unprojection (image_width/image_height,
focal 1394) stays self-consistent.  Runs in the ``depthpro`` environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

INPUT_ROOT = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
DEPTHPRO_MODEL = WORKSPACE / "assets/third_party/pressure_toolkit/depthpro"
MAX_M = 65.535


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


def run_sessions(sessions: list[str], rows: dict[str, dict], proc, model,
                 device, dtype, force: bool) -> list[dict]:
    import torch
    from PIL import Image

    reports = []
    for sid in sessions:
        row = rows[sid]
        date, subject, _ = session_parts(row)
        out_dir = INPUT_ROOT / date / subject / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "template_scene_rgbd.npy"
        if output.is_file() and not force:
            reports.append({"session": sid, "status": "ok_existing"})
            continue
        frames = np.load(FACTS_ROOT / date / subject / sid / "frames.npz")
        valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
        fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
        candidates = np.flatnonzero(valid & ~fake)
        # fake/invalid marks the pressure stream; the static RGB-D scene
        # template only needs a real camera frame, so fall back to frame 0
        # when no valid frame exists (S12102 is all-fake).
        frame = int(candidates[0]) if candidates.size else 0
        rgb_path = FACTS_ROOT / date / subject / sid / "rgb" / f"{frame:06d}.jpg"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"{sid}: RGB frame missing: {rgb_path}")
        image = Image.open(rgb_path).convert("RGB")
        inputs = proc(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if dtype == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model(**inputs)
        result = proc.post_process_depth_estimation(
            out, target_sizes=[(image.height, image.width)])[0]
        depth_m = result["predicted_depth"].float().cpu().numpy()
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, MAX_M)
        rgb = np.asarray(image)
        image.close()
        # npz container (no pickle) so numpy 1.x consumers can read files
        # produced under the depthpro env (numpy 2.x); the PoseTransOpt
        # loader accepts both the npz and the legacy dict-in-npy form.
        # np.savez appends ".npz" unless given a file object, so write
        # through an explicit handle to keep the fixed .npy contract name.
        if output.is_file():
            output.unlink()
        stale = Path(str(output) + ".npz")
        if stale.is_file():
            stale.unlink()
        with open(output, "wb") as handle:
            np.savez(handle, rgb=rgb, depth=depth_m.astype(np.float32))
        reports.append({"session": sid, "status": "built", "frame": frame,
                        "shape": [int(rgb.shape[0]), int(rgb.shape[1])]})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = DepthProImageProcessorFast.from_pretrained(str(DEPTHPRO_MODEL))
    model = DepthProForDepthEstimation.from_pretrained(
        str(DEPTHPRO_MODEL), torch_dtype=dtype).to(device).eval()
    rows = read_manifest()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    sessions = sessions[args.shard::args.shards]
    reports = run_sessions(sessions, rows, proc, model, device, dtype, args.force)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
