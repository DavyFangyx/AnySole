#!/usr/bin/env python3
"""构建 pressure_toolkit 上游原生目录契约（model_inputs/pressure_toolkit/v1）。

只读源（06 任务书 §4 完整来源链）：

    shared://facts/sessions/cam3/<date>/<subject>/<session>
        ├── frames.npz          frame id / valid / fake
        ├── rgb/<frame_id>.jpg
        └── pressure_48.npz     只作 provenance，数值经公共 31×11 表示消费
    shared://representations/tactile/mmvp_31x11/v1    （逐帧 insole）
    shared://frontends/human_masks/sam31/v1           （单受试者 mask 索引）
    shared://frontends/depthpro/v1                    （深度，若已生成）
    shared://frontends/rtmpose_halpe26/v1             （HALPE-26 关键点，若已生成）
    shared://frontends/cliff_hr48/v1                  （Agent E 单受试者 CLIFF，若已生成）
    protocol://calibration/<date>.json                （cam3 标定，只做相机变换）

写出目录：

    model_inputs/pressure_toolkit/v1/
    ├── images/<date>/<subject>/<session>/
    │   ├── color/<frame_id:06d>.jpg       → shared rgb 符号链接
    │   ├── depth/<frame_id:06d>.png       DepthPro 毫米刻度
    │   │                                  (mmvp_series/depth/rgb2depth.py 写入
    │   │                                   shared/frontends/depthpro/v1，本树符号链接)
    │   ├── depth_mask/<frame_id:06d>.png  human ∩ finite ∩ [0.4m,5m]
    │   ├── insole/<frame_id:06d>.npy      {'insole':[L,R],'frame_id'} 上游包装
    │   ├── calibration.npy                上游 loadCalibrationFromNpy 契约
    │   └── frame_ids.npy                  拟合帧（valid 且非 fake）的 shared frame id
    ├── annotations/<date>/floor_info/floor_<subject>.npy + floor_artifact_<date>.json
    ├── input/<subject>/<session>/keypoints/<frame_id:06d>.npy
    ├── initialization/<date>/<subject>/<session>/<session>_cliff_hr48.npz
    ├── session_reports/<date>/<subject>/<session>.json
    └── artifact.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from AnysoleWorkspace.tool.adapters.mmvp_series.common import common
from AnysoleWorkspace.tool.adapters.mmvp_series.masks import masks
from AnysoleWorkspace.tool.adapters.mmvp_series.calibration.calibration import (
    build_calibration_npy,
    load_calibration,
)
from AnysoleWorkspace.tool.adapters.mmvp_series.insole_31x11 import insole as insole_mod
from AnysoleWorkspace.tool.adapters.mmvp_series.pressure_tookit import floor as floor_mod

MAX_TIME_ERROR_S = 0.020  # 总控 §2.2：sam31 匹配容差上限 20ms


def default_floor_roi(depth_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """底部中央行走带：行 55%~90%，列 30%~70%。"""
    rows, cols = int(depth_shape[0]), int(depth_shape[1])
    return (int(rows * 0.55), int(rows * 0.90), int(cols * 0.30), int(cols * 0.70))


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(item) for item in value.split(",")]
    if len(parts) != 4 or parts[0] >= parts[1] or parts[2] >= parts[3]:
        raise ValueError(f"bad ROI {value!r}; expected r0,r1,c0,c1")
    return tuple(parts)  # type: ignore[return-value]


def link_color_frames(session_facts: dict, session_dir: Path,
                      fitting_ids: np.ndarray, force: bool) -> int:
    """shared rgb → color/<frame_id:06d>.jpg 符号链接（只对拟合帧）。"""
    color_dir = session_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in fitting_ids:
        target = color_dir / f"{int(frame_id):06d}.jpg"
        if target.is_symlink() and not force:
            continue
        source = session_facts["dir"] / "rgb" / f"{int(frame_id):06d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"missing shared rgb frame: {source}")
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(os.path.relpath(source, target.parent))
    return int(len(fitting_ids))


def session_report_path(date: str, subject: str, session: str) -> Path:
    return common.ADAPTER_ROOT / "session_reports" / date / subject / f"{session}.json"


def link_frontend_dir(source_dir: Path, link: Path, force: bool) -> str:
    """消费端目录指向共享前端（幂等）；源不存在时保持缺失，不建悬空链接。"""
    if not source_dir.is_dir():
        return "no_frontend"
    if link.is_symlink():
        if os.readlink(link) == os.path.relpath(source_dir, link.parent):
            return "kept"
        link.unlink()
    elif link.exists():
        return "physical_dir_kept"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(source_dir, link.parent))
    return "linked"


def prepare_session(row: dict, args: argparse.Namespace,
                    rtmpose_present: bool, cliff_present: bool,
                    sam31: dict[str, list[dict]]) -> dict:
    date, subject, session = common.session_parts(row)
    session_facts = common.load_session_facts(date, subject, session)
    frames = session_facts["frames"]
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    valid = np.asarray(frames["valid"], dtype=np.uint8).astype(bool)
    fake = np.asarray(frames["fake"], dtype=np.uint8).astype(bool)
    if frame_ids.shape != valid.shape or valid.shape != fake.shape:
        raise ValueError(f"{session}: frames.npz arrays differ in length")

    fitting_ids = frame_ids[valid & ~fake]
    if fitting_ids.size == 0:
        # 全 fake session（如 S12102）不产生伪正式结果：跳过并登记，不中断批量
        return {
            "session_id": session,
            "date": date,
            "subject": subject,
            "n_manifest": int(row["n_frames"]),
            "n_shared_frames": int(len(frame_ids)),
            "n_fitting": 0,
            "n_skipped_fake": int(np.count_nonzero(fake)),
            "n_skipped_invalid": int(np.count_nonzero(~valid & ~fake)),
            "fitting_frame_ids": [],
            "errors": ["every frame is fake/invalid; no formal fitting task"],
            "skipped": True,
        }

    session_dir = common.ADAPTER_ROOT / "images" / date / subject / session
    session_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "session_id": session,
        "date": date,
        "subject": subject,
        "n_manifest": int(row["n_frames"]),
        "n_shared_frames": int(len(frame_ids)),
        "n_fitting": int(len(fitting_ids)),
        "n_skipped_fake": int(np.count_nonzero(fake)),
        "n_skipped_invalid": int(np.count_nonzero(~valid & ~fake)),
        "fitting_frame_ids": fitting_ids.tolist(),
        "depth_source": None,
        "sam31": {},
        "floor": None,
        "insole": {},
        "keypoints": None,
        "cliff": None,
        "errors": [],
    }

    # 1. frame_ids.npy：拟合任务只包含 valid 且非 fake 的 shared frame id
    np.save(session_dir / "frame_ids.npy", fitting_ids)

    # 2. color 符号链接（shared facts rgb，frame-id 命名）
    report["n_color_linked"] = link_color_frames(
        session_facts, session_dir, fitting_ids, args.force)

    # 3. calibration.npy：棋盘格标定只用于相机坐标变换
    cal = load_calibration(common.CALIBRATION_DIR / f"{date}.json")
    np.save(session_dir / "calibration.npy", build_calibration_npy(cal))

    # 4. depth：DepthPro 毫米刻度 PNG，由 mmvp_series/depth/rgb2depth.py 写入
    # shared://frontends/depthpro/v1，本树以目录符号链接采用（前端缺该 session
    # 时保持缺失并记录错误；depth gate 见 §main）
    depth_dir = session_dir / "depth"
    report["depth_link"] = link_frontend_dir(
        common.DEPTHPRO_ROOT / date / subject / session / "depth",
        depth_dir, args.force)
    depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.is_dir() else []
    if depth_files:
        report["depth_source"] = (
            "shared://frontends/depthpro/v1/<date>/<subject>/<session>/depth "
            "(DepthPro, generated by "
            "AnysoleWorkspace/tool/adapters/mmvp_series/depth/rgb2depth.py)")
        report["n_depth"] = len(depth_files)
    else:
        report["errors"].append(
            "depth missing: run "
            "AnysoleWorkspace/tool/adapters/mmvp_series/depth/rgb2depth.py")

    # 5. 地面：上游 calculateFloorNormal 于行走深度帧（本项为必做，P1）
    depth_png = None
    if depth_files:
        by_id = {int(p.stem): p for p in depth_files}
        floor_frame = args.floor_frame
        if floor_frame is None:
            floor_frame = int(fitting_ids[len(fitting_ids) // 2])
        depth_png = by_id.get(floor_frame)
        if depth_png is None:
            report["errors"].append(
                f"floor frame {floor_frame} has no depth; floor artifact skipped")
    if depth_png is not None:
        K = cal["K"]
        depth_png_filled = depth_png
        if args.floor_roi:
            roi = parse_roi(args.floor_roi)
        else:
            # ROI 缺省时按深度图尺寸取底部中央带（行走地面候选）
            from PIL import Image

            with Image.open(depth_png_filled) as img:
                roi = default_floor_roi((img.height, img.width))
        try:
            estimate = floor_mod.estimate_floor(
                depth_png_filled, roi, float(K[0, 0]), float(K[1, 1]),
                float(K[0, 2]), float(K[1, 2]))
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"floor estimation failed: {exc}")
            estimate = None
        if estimate is not None:
            floor_path = floor_mod.write_floor_artifact(
                common.ADAPTER_ROOT, date, subject, estimate,
                depth_png=depth_png_filled, xy=roi,
                depth_source=report["depth_source"],
                depth_hash=masks.sha256_file(depth_png_filled),
                frame_id=floor_frame)
            slices = floor_mod.slice_points(
                depth_png_filled, roi, float(K[0, 0]), float(K[1, 1]),
                float(K[0, 2]), float(K[1, 2]))
            checkerboard = floor_mod.checkerboard_floor_mapping(
                common.CALIBRATION_DIR / f"{date}.json")
            report["floor"] = {
                "path": str(floor_path),
                "frame_id": int(floor_frame),
                "roi": list(roi),
                "residual_mm": estimate["residual_mm"],
                "audit": floor_mod.floor_audit(estimate, checkerboard, slices),
            }

    # 6. insole：公共 31×11 表示的逐帧上游包装（数值 parity）
    mmvp = insole_mod.load_mmvp_session(date, subject, session)
    report["insole"] = insole_mod.build_session_insoles(
        session_facts, mmvp, session_dir, args.force)

    # 7. SAM3.1 单受试者 depth mask：human ∩ finite ∩ [0.4m,5m]
    sam31_rows = {int(item["frame_id"]): item for item in sam31.get(session, [])}
    mask_stats = []
    mask_dir = session_dir / "depth_mask"
    for frame_id in fitting_ids:
        row = sam31_rows.get(int(frame_id))
        if row is None or not int(row.get("valid", 0)):
            report["errors"].append(
                f"frame {int(frame_id)}: no valid SAM3.1 mask index row")
            continue
        time_error = row.get("time_error_s")
        if time_error is None or abs(float(time_error)) > MAX_TIME_ERROR_S:
            report["errors"].append(
                f"frame {int(frame_id)}: SAM3.1 time error "
                f"{time_error}s > {MAX_TIME_ERROR_S}s")
            continue
        mask_png = Path(row["mask_path"])
        depth_frame = depth_dir / f"{int(frame_id):06d}.png"
        if not depth_frame.is_file():
            continue  # 深度未生成，mask 留待 rgb2depth 完成后重建
        out_png = mask_dir / f"{int(frame_id):06d}.png"
        if out_png.is_file() and not args.force:
            continue
        stat = masks.build_depth_mask(depth_frame, mask_png, out_png)
        stat["frame_id"] = int(frame_id)
        stat["time_error_s"] = float(row.get("time_error_s", 0.0))
        mask_stats.append(stat)
    report["sam31"] = {
        "indexed_frames": len(sam31_rows),
        "masks_built": len(mask_stats),
        "empty_masks": [item["frame_id"] for item in mask_stats if item["empty"]],
        "matching_frames": len(mask_stats),
        "area_before_intersection_total": sum(
            item["human_mask_area"] for item in mask_stats),
        "area_after_intersection_total": sum(
            item["intersected_area"] for item in mask_stats),
    }

    # 8. HALPE-26 关键点：只消费 rtmpose 前端（若已生成）
    if rtmpose_present:
        kp_root = common.RTMPOSE_ROOT / date / subject / session / "keypoints"
        kp_out = common.ADAPTER_ROOT / "input" / subject / session / "keypoints"
        kp_out.mkdir(parents=True, exist_ok=True)
        n_kp = 0
        for frame_id in fitting_ids:
            source = kp_root / f"{int(frame_id):06d}.npy"
            if not source.is_file():
                continue
            target = kp_out / f"{int(frame_id):06d}.npy"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.path.relpath(source, target.parent))
            n_kp += 1
        report["keypoints"] = (
            f"shared://frontends/rtmpose_halpe26/v1: {n_kp}/"
            f"{len(fitting_ids)} frames staged")

    # 9. CLIFF 单受试者初始化：只消费 Agent E 前端（若已生成）
    if cliff_present:
        cliff_src = (common.CLIFF_ROOT / date / subject / session /
                     f"{session}_cliff_hr48.npz")
        if cliff_src.is_file():
            init_dir = common.ADAPTER_ROOT / "initialization" / date / subject / session
            init_dir.mkdir(parents=True, exist_ok=True)
            target = init_dir / f"{session}_cliff_hr48.npz"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.path.relpath(cliff_src, target.parent))
            report["cliff"] = str(cliff_src)
        else:
            report["cliff"] = None

    if report["errors"]:
        report["errors"].sort()
    report_path = session_report_path(date, subject, session)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"),
                        default="all")
    parser.add_argument("--sessions", default="",
                        help="comma-separated session ids")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--floor-frame", type=int, default=None,
                        help="frame id of the depth frame used for floor estimation")
    parser.add_argument("--floor-roi", default=None,
                        help="r0,r1,c0,c1 ROI for floor estimation")
    args = parser.parse_args()

    rows = common.manifest_rows()
    sam31 = common.load_sam31_index()
    sessions = ([s for s in args.sessions.split(",") if s]
                if args.sessions else common.split_sessions(args.split))
    rtmpose_present = (common.RTMPOSE_ROOT / "artifact.json").is_file()
    cliff_present = (common.CLIFF_ROOT / "artifact.json").is_file()

    reports = []
    for session in sessions:
        if session not in rows:
            raise KeyError(f"session not in manifest: {session}")
        reports.append(prepare_session(rows[session], args,
                                       rtmpose_present, cliff_present, sam31))

    artifact = common.adapter_artifact(
        [f"shared://facts/sessions/{s['date']}/{s['subject']}/{s['session_id']}"
         for s in reports],
        {"adapter_version": common.ADAPTER_VERSION,
         "sessions": len(reports)},
        frame_count=None)
    (common.ADAPTER_ROOT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {"sessions": len(reports),
         "with_errors": [r["session_id"] for r in reports if r["errors"]],
         "reports": reports},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
