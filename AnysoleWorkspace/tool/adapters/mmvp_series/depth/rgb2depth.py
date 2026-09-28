#!/usr/bin/env python3
"""用 Apple Depth Pro 从 shared facts RGB 生成毫米刻度深度 PNG（MMVP 数据端）。

深度是 pressure_tookit 的模型输入（人体深度 mask、行走地面和深度损失
的观测源），也是 PoseTransOpt 模板场景的来源。T1-B（11 号文档 §5）把深度
producer 归入 ``mmvp_series/depth/``，一次生产、消费端符号链接采用：

    shared/frontends/depthpro/v1/<date>/<subject>/<session>/depth/
    ├── <frame_id:06d>.png     16 位毫米刻度（SCALE=1000，MAX_M=65.535m 截断）
    └── meta.json              逐帧 src/focal_px/fov_deg/w/h/scale/unit/depth_min/max

相对 ``Baselines_old/pressure_tookit/data_prep/rgb2depth.py`` 的改动：

1. **输出位置**：不再直接写 pressure_toolkit 的 model_inputs 树，改写成
   ``shared/frontends/depthpro/v1`` 前端树（消费端以符号链接映射）。
2. **输入来源**：不再读 toolkit 树的 ``session_dir/color`` + ``frame_ids.npy``，
   改为直接读 shared facts 的 ``frames.npz``（取 ``valid & ~fake`` 帧）
   与 ``rgb/<frame_id:06d>.jpg``——因此**深度可以在 toolkit 树落盘之前先跑**。
3. **session 发现**：不再扫描 ``frame_ids.npy``，改为 canonical manifest +
   ``protocol/splits/default/splits.csv``（``--split`` / ``--sessions`` / ``--session``）。
4. **依赖**：``lib.utils.workspace`` 换成 gait 侧
   ``AnysoleWorkspace/tool/workspace``（``resolve_uri``）与仓库根推导。

保持不变的语义：DepthPro 逐帧 fp16 CUDA 推理、16 位毫米 PNG、``SCALE=1000``、
``MAX_M=65.535`` 截断、``meta.json`` 字段与增量保存（每 20 帧落盘一次）、
``--device/--worker-id/--num-workers/--skip-existing/--limit/--batch-size``。

被移除的上游参数：``--sessions-root``（扫描 ``frame_ids.npy`` 的旧适配器树入口）
与遗留的位置参数 ``paths``（原实现即为报错占位）；改由
``--split/--sessions/--session/--output-root`` 选择 collection。

用法（depthpro 环境）：

    python -m AnysoleWorkspace.tool.adapters.mmvp_series.depth.rgb2depth --session S10101
    python -m AnysoleWorkspace.tool.adapters.mmvp_series.depth.rgb2depth --split all --skip-existing
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

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

SCALE = 1000.0  # 1 单位 = 1 毫米
MAX_M = 65.535  # 16 位在毫米刻度下的上限，超出一律截断
DEFAULT_MODEL = WORKSPACE / "assets/third_party/pressure_toolkit/depthpro"
OUTPUT_ROOT = WORKSPACE / "shared/frontends/depthpro/v1"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate metric depth PNGs with Apple Depth Pro into the "
                    "shared depth frontend (valid, non-fake frames only).")
    parser.add_argument("--split", choices=("train", "val", "test", "all"),
                        default="all")
    parser.add_argument("--sessions", default="",
                        help="comma-separated session ids")
    parser.add_argument("--session", default=None,
                        help="process only this session id")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT),
                        help="frontend root (default: shared/frontends/depthpro/v1)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Local model path, canonical URI, or a Hugging Face model id")
    parser.add_argument("--device", default=None,
                        help="cuda, cuda:0, or cpu. Default: cuda if available")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true",
                      help="Skip frames whose depth PNG already exists")
    mode.add_argument("--force", action="store_true",
                      help="Recompute and overwrite every depth PNG")
    parser.add_argument("--limit", type=int, default=0,
                        help="Per session, process only the first N frames. 0 means all")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="Process only the first N sessions. 0 means all")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per GPU forward; start with 2 or 4")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Session shard index for multi-process runs")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of session shards; each process loads one model copy")
    return parser.parse_args()


def fitting_frames(date: str, subject: str, session: str) -> list[int]:
    """shared facts 的 fitting frame ids（valid 且非 fake）。"""
    frames = np.load(FACTS_ROOT / date / subject / session / "frames.npz")
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
    if frame_ids.shape != valid.shape or valid.shape != fake.shape:
        raise ValueError(f"{session}: frames.npz arrays differ in length")
    return [int(f) for f in frame_ids[valid & ~fake]]


def session_rgb(date: str, subject: str, session: str, frame_id: int) -> Path:
    return FACTS_ROOT / date / subject / session / "rgb" / f"{frame_id:06d}.jpg"


def load_meta(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_meta(path, meta):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def resolve_model(model: str) -> str:
    """Local path, canonical URI, or a Hugging Face model id."""
    if "://" in model:
        return str(resolve_uri(model, must_exist=True))
    candidate = Path(model).expanduser()
    return str(candidate if candidate.exists() else model)


def load_model(model_path, device):
    # Import the heavyweight optional dependency only after argument parsing;
    # ``--help`` and static smoke tests must work without depthpro installed.
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    use_fp16 = device.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    proc = DepthProImageProcessorFast.from_pretrained(model_path)
    model = DepthProForDepthEstimation.from_pretrained(
        model_path, torch_dtype=dtype).to(device).eval()
    return proc, model, use_fp16, dtype


def process_session(date: str, subject: str, session: str, output_root: Path,
                    proc, model, device, use_fp16, limit,
                    skip_existing, batch_size):
    from PIL import Image
    from tqdm import tqdm

    frame_ids = fitting_frames(date, subject, session)
    if limit > 0:
        frame_ids = frame_ids[:limit]
    dst_path = output_root / date / subject / session / "depth"
    dst_path.mkdir(parents=True, exist_ok=True)
    meta_path = dst_path / "meta.json"
    meta = load_meta(str(meta_path))
    pending = []
    n_ok = n_skip = n_failed = 0
    for frame_id in frame_ids:
        source = session_rgb(date, subject, session, frame_id)
        if not source.is_file():
            raise FileNotFoundError(f"{session}: shared RGB frame missing: {source}")
        out_png = dst_path / f"{frame_id:06d}.png"
        if skip_existing and out_png.is_file():
            n_skip += 1
        else:
            pending.append((source, out_png))

    for start in tqdm(range(0, len(pending), batch_size),
                      desc=session, leave=False, unit="batch"):
        batch = pending[start:start + batch_size]
        images = []
        try:
            images = [Image.open(path).convert("RGB") for path, _ in batch]
            inputs = proc(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if use_fp16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.no_grad():
                out = model(**inputs)
            results = proc.post_process_depth_estimation(
                out, target_sizes=[(img.height, img.width) for img in images])
            for (path, out_png), img, result in zip(batch, images, results):
                name = Path(path).stem
                depth_m = result["predicted_depth"].float().cpu().numpy()
                depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
                depth_m = np.clip(depth_m, 0.0, MAX_M)
                Image.fromarray((depth_m * SCALE).round().astype(np.uint16)).save(out_png)
                meta[name] = {
                    "src": Path(path).name,
                    "focal_px": to_float(result["focal_length"]),
                    "fov_deg": to_float(result["field_of_view"]),
                    "w": img.width, "h": img.height, "scale": SCALE, "unit": "mm",
                    "depth_min_m": float(depth_m.min()), "depth_max_m": float(depth_m.max()),
                }
                n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_failed += len(batch)
            print(f"failed batch starting at {start} ({len(batch)} frames): {exc}")
        finally:
            for image in images:
                image.close()
        if n_ok and n_ok % 20 == 0:
            save_meta(str(meta_path), meta)
    save_meta(str(meta_path), meta)
    return n_ok, n_skip, n_failed


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise SystemExit("require 0 <= --worker-id < --num-workers and --num-workers >= 1")
    rows = read_manifest()
    if args.session:
        sessions = [args.session]
    elif args.sessions:
        sessions = [s for s in args.sessions.split(",") if s]
    else:
        sessions = split_sessions(args.split)
    for session in sessions:
        if session not in rows:
            raise KeyError(f"session not in manifest: {session}")
    if args.limit_sessions > 0:
        sessions = sessions[:args.limit_sessions]
    sessions = sessions[args.worker_id::args.num_workers]
    if not sessions:
        raise FileNotFoundError("No sessions matched the requested filter")

    model_path = resolve_model(args.model)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    proc, model, use_fp16, dtype = load_model(model_path, device)
    print("sessions=%d device=%s dtype=%s model=%s" % (
        len(sessions), device, dtype, model_path))
    if device.type == "cuda":
        print("gpu=%s" % torch.cuda.get_device_name(device))
    totals = [0, 0, 0]
    failed_sessions = []
    for session in sessions:
        date, subject, sid = session_parts(rows[session])
        try:
            counts = process_session(date, subject, sid, output_root, proc, model,
                                     device, use_fp16, args.limit,
                                     args.skip_existing, args.batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"failed session {sid}: {exc}")
            counts = (0, 0, 1)
            failed_sessions.append(sid)
        totals = [a + b for a, b in zip(totals, counts)]
    print("wrote %d depth maps, skipped %d, failed %d" % tuple(totals))
    if failed_sessions:
        print("failed sessions:")
        for session in failed_sessions:
            print("  " + session)
    return 1 if totals[2] else 0


if __name__ == "__main__":
    raise SystemExit(main())
