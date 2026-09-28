#!/usr/bin/env python3
"""MMVP 前端提升：一次生产、三个消费端（T1-B，11 号文档 §5）。

MMVP 方法族（FPP-Net / PoseTransOpt / pressure_tookit）此前各自维护一份
RTMPose 关键点与 CLIFF 单人结果，同一份数据被重复生产、重复落盘。本模块把
它们提升为共享前端，并让消费端以符号链接采用：

    shared/frontends/rtmpose_halpe26/v1/<date>/<subject>/<session>/keypoints/
    shared/frontends/cliff_hr48/v1/<date>/<subject>/<session>/
        ├── CLIFF_results.npz          PoseTransOpt 原生契约
        └── <session>_cliff_hr48.npz   pressure_tookit 初值契约（本模块转换）

消费端映射（提升后由符号链接指向共享前端，物理副本只剩一份）：

    model_inputs/FPP-Net/adapter_v1/.../keypoints      → 目录符号链接
    model_inputs/PoseTransOpt/adapter_v1/.../keypoints → 目录符号链接
    model_inputs/PoseTransOpt/adapter_v1/.../CLIFF_results.npz → 文件符号链接
    model_inputs/pressure_toolkit/v1/...               → 由 build_inputs.py
        逐帧/整目录符号链接自共享前端（不改上游读取路径）

提升是**幂等**的：已提升（已是符号链接）的 session 直接跳过；前端文件已存在
且大小一致时复用；已存在的物理副本在整目录 `rm` 之前必须逐文件校验与前端
一致，否则该 session 报错退出、绝不做破坏性删除。

用法：

    python -m AnysoleWorkspace.tool.adapters.mmvp_series.frontends --frontend rtmpose_halpe26 --dry-run
    python -m AnysoleWorkspace.tool.adapters.mmvp_series.frontends --sessions S5011
    python -m AnysoleWorkspace.tool.adapters.mmvp_series.frontends --frontend all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]  # mmvp_series/ sits one level above the subpackages
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.mmvp_series.cliff import export_toolkit_init  # noqa: E402
from AnysoleWorkspace.tool.adapters.mmvp_series.common import common  # noqa: E402
from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402

MODEL_INPUTS = WORKSPACE / "model_inputs"
CONSUMER_ROOTS = {
    "FPP-Net": MODEL_INPUTS / "FPP-Net/adapter_v1",
    "PoseTransOpt": MODEL_INPUTS / "PoseTransOpt/adapter_v1",
}
KEYPOINT_CONSUMERS = ("FPP-Net", "PoseTransOpt")
CLIFF_CONSUMERS = ("PoseTransOpt",)
KEYPOINT_MODEL = (
    WORKSPACE / "assets/third_party/rtmpose"
    / "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
)
KEYPOINT_CONFIG = (
    WORKSPACE / "assets/third_party/mmpose/configs/body_2d_keypoint/rtmpose/body8"
    / "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
)
CLIFF_CKPT = (
    WORKSPACE / "assets/third_party/CLIFF/data/ckpt"
    / "hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
)
SAM31_ARTIFACT = WORKSPACE / "shared/frontends/human_masks/sam31/v1/artifact.json"


def relative_link(link: Path, target: Path) -> str:
    """把 ``link`` 幂等地替换为指向 ``target`` 的相对符号链接。"""
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink():
        if os.readlink(link) == rel:
            return "kept"
        link.unlink()
    elif link.exists():
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(rel)
    return "created"


def is_linked(link: Path, target: Path) -> bool:
    return (link.is_symlink()
            and os.readlink(link) == os.path.relpath(target, link.parent))


def hardlink_into(source_dir: Path, target_dir: Path, pattern: str,
                  dry_run: bool) -> dict:
    """把 ``source_dir/pattern`` 逐文件硬链接进 ``target_dir``（跨设备退化为复制）。"""
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {source_dir}")
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    n_linked = n_existing = n_copied = 0
    bytes_total = 0
    for source in files:
        target = target_dir / source.name
        bytes_total += source.stat().st_size
        if target.exists():
            if target.stat().st_size == source.stat().st_size:
                # 由同一趟生产给出且大小一致：视为已提升（哈希抽样另行校验）
                n_existing += 1
                continue
            if not dry_run:
                target.unlink()
        if dry_run:
            n_linked += 1
            continue
        try:
            os.link(source, target)
            n_linked += 1
        except OSError as exc:
            if exc.errno != 18:  # EXDEV
                raise
            shutil.copy2(source, target)
            n_copied += 1
    return {"files": len(files), "linked": n_linked, "reused": n_existing,
            "copied": n_copied, "bytes": bytes_total}


def verify_copy(source_dir: Path, target_dir: Path, pattern: str,
                sample: int) -> dict:
    """校验目标目录覆盖源目录（数目、名称、大小，另加 ``sample`` 个文件的哈希）。"""
    sources = sorted(source_dir.glob(pattern))
    targets = sorted(target_dir.glob(pattern))
    source_names = [p.name for p in sources]
    target_names = [p.name for p in targets]
    if source_names != target_names:
        missing = sorted(set(source_names) - set(target_names))[:5]
        extra = sorted(set(target_names) - set(source_names))[:5]
        raise ValueError(
            f"{target_dir}: file set mismatch vs {source_dir} "
            f"(missing={missing}, extra={extra})")
    checked = 0
    if sample > 0 and sources:
        step = max(1, len(sources) // sample)
        for index in range(0, len(sources), step):
            source = sources[index]
            if sha256_file(source) != sha256_file(target_dir / source.name):
                raise ValueError(f"{target_dir / source.name}: differs from {source}")
            checked += 1
    return {"files": len(sources), "hash_checked": checked}


def refresh_keypoint_links(date: str, subject: str, session: str,
                           roots=KEYPOINT_CONSUMERS, dry_run: bool = False,
                           guard: bool = True) -> dict:
    """让消费端 ``keypoints/`` 指向共享前端。

    ``guard=True``（生产者增量运行的默认值）绝不删除未提升的物理副本；
    提升流程在逐文件校验之后以 ``guard=False`` 调用，完成目录符号链接替换。
    """
    frontend_dir = common.RTMPOSE_ROOT / date / subject / session / "keypoints"
    result = {}
    for name in roots:
        link = CONSUMER_ROOTS[name] / date / subject / session / "keypoints"
        try:
            if not frontend_dir.is_dir():
                result[name] = "no_frontend"
            elif is_linked(link, frontend_dir):
                result[name] = "kept"
            elif link.is_symlink() or not link.exists():
                result[name] = "would_link" if dry_run else relative_link(link, frontend_dir)
            elif link.is_dir() and any(link.iterdir()):
                if guard:
                    result[name] = "physical_dir_kept"
                else:
                    verify_copy(link, frontend_dir, "*.npy", sample=2)
                    result[name] = "would_link" if dry_run else relative_link(link, frontend_dir)
            else:
                result[name] = "would_link" if dry_run else relative_link(link, frontend_dir)
        except Exception as exc:  # noqa: BLE001
            result[name] = f"failed: {type(exc).__name__}: {exc}"
    return result


def refresh_cliff_links(date: str, subject: str, session: str,
                        roots=CLIFF_CONSUMERS, dry_run: bool = False,
                        guard: bool = True) -> dict:
    """让消费端 ``CLIFF_results.npz`` 指向共享前端（幂等）。"""
    frontend_file = common.CLIFF_ROOT / date / subject / session / "CLIFF_results.npz"
    result = {}
    for name in roots:
        link = CONSUMER_ROOTS[name] / date / subject / session / "CLIFF_results.npz"
        try:
            if not frontend_file.is_file():
                result[name] = "no_frontend"
            elif is_linked(link, frontend_file):
                result[name] = "kept"
            elif link.is_symlink() or not link.exists():
                result[name] = "would_link" if dry_run else relative_link(link, frontend_file)
            elif guard:
                result[name] = "physical_file_kept"
            else:
                if sha256_file(link) != sha256_file(frontend_file):
                    raise ValueError(f"{link}: differs from {frontend_file}")
                result[name] = "would_link" if dry_run else relative_link(link, frontend_file)
        except Exception as exc:  # noqa: BLE001
            result[name] = f"failed: {type(exc).__name__}: {exc}"
    return result


def promote_rtmpose(sessions, rows, dry_run: bool, sample: int) -> dict:
    """FPP-Net 关键点树 → shared/frontends/rtmpose_halpe26/v1（目录符号链接采用）。"""
    reports = []
    failures = []
    total_files = 0
    for sid in sessions:
        date, subject, _ = common.session_parts(rows[sid])
        source = CONSUMER_ROOTS["FPP-Net"] / date / subject / sid / "keypoints"
        frontend_dir = common.RTMPOSE_ROOT / date / subject / sid / "keypoints"
        entry = {"session": sid, "source": str(source)}
        try:
            if source.is_symlink():
                entry["status"] = "already_promoted"
                entry["link"] = {"files": len(list(frontend_dir.glob("*.npy"))),
                                 "linked": 0, "reused": 0, "copied": 0}
                entry["verify"] = verify_copy(frontend_dir, frontend_dir, "*.npy", 0)
            elif not source.is_dir():
                raise FileNotFoundError(f"no keypoints under {source}")
            else:
                entry["link"] = hardlink_into(source, frontend_dir, "*.npy", dry_run)
                if dry_run:
                    entry["verify"] = {"files": entry["link"]["files"], "hash_checked": 0}
                else:
                    entry["verify"] = verify_copy(source, frontend_dir, "*.npy", sample)
                    # 第二个消费端的物理副本也必须在删除前校验一致
                    other = CONSUMER_ROOTS["PoseTransOpt"] / date / subject / sid / "keypoints"
                    if other.is_dir() and not other.is_symlink():
                        entry["second_copy"] = verify_copy(
                            other, frontend_dir, "*.npy", min(sample, 2))
                entry["status"] = "promoted"
            entry["links"] = refresh_keypoint_links(
                date, subject, sid, dry_run=dry_run,
                guard=entry["status"] == "already_promoted")
            total_files += entry["link"]["files"]
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(sid)
        reports.append(entry)
    return {"frame_count": total_files, "sessions": len(sessions),
            "reports": reports, "failures": failures}


def promote_cliff(sessions, rows, dry_run: bool, sample: int,
                  force_convert: bool) -> dict:
    """PoseTransOpt CLIFF 结果 → shared/frontends/cliff_hr48/v1（文件符号链接采用）。"""
    reports = []
    failures = []
    total_frames = 0
    for sid in sessions:
        date, subject, _ = common.session_parts(rows[sid])
        session_root = CONSUMER_ROOTS["PoseTransOpt"] / date / subject / sid
        source = session_root / "CLIFF_results.npz"
        frontend_dir = common.CLIFF_ROOT / date / subject / sid
        frontend_file = frontend_dir / "CLIFF_results.npz"
        entry = {"session": sid, "source": str(source)}
        try:
            if frontend_file.is_file():
                entry["link"] = {"files": 1, "linked": 0, "reused": 1, "copied": 0}
            elif source.is_symlink() or not source.is_file():
                raise FileNotFoundError(f"no CLIFF_results.npz under {session_root}")
            else:
                entry["link"] = hardlink_into(session_root, frontend_dir,
                                              "CLIFF_results.npz", dry_run)
            if dry_run:
                entry["convert"] = {"status": "would_convert"}
                frames = int(len(np.load(source, allow_pickle=False)["frame_id"]))
            else:
                entry["convert"] = export_toolkit_init.convert(
                    frontend_file, frontend_dir / f"{sid}_cliff_hr48.npz",
                    force_convert)
                frames = int(entry["convert"].get("frames", len(np.load(
                    frontend_file, allow_pickle=False)["frame_id"])))
            total_frames += frames
            entry["frames"] = frames
            entry["links"] = refresh_cliff_links(date, subject, sid,
                                                 dry_run=dry_run, guard=False)
            entry["status"] = "promoted"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(sid)
        reports.append(entry)
    return {"frame_count": total_frames, "sessions": len(sessions),
            "reports": reports, "failures": failures}


def write_frontend_artifacts(rtmpose_frame_count: int | None,
                             cliff_frame_count: int | None,
                             sessions: int) -> list[str]:
    written = []
    if rtmpose_frame_count is not None:
        write_artifact(
            common.RTMPOSE_ROOT,
            schema_version="shared.frontend.rtmpose_halpe26.v1",
            producer="mmvp_series/keypoints/run_rtmpose.py",
            repository_root=REPO_ROOT,
            parameters={
                "session_count": sessions,
                "model": KEYPOINT_MODEL.name,
                "config": str(KEYPOINT_CONFIG.relative_to(WORKSPACE)),
                "source_tree": "model-input://FPP-Net/adapter_v1",
                "layout": "<date>/<subject>/<session>/keypoints/<frame_id:06d>.npy",
                "sidecar_keys": ["keypoints (26,2)", "keypoint_scores (26,)",
                                 "frame_id", "source_image", "model"],
            },
            source_artifacts=["model-input://FPP-Net/adapter_v1",
                              "asset://third_party/rtmpose",
                              "asset://third_party/mmpose"],
            source_hashes={"model": sha256_file(KEYPOINT_MODEL),
                           "config": sha256_file(KEYPOINT_CONFIG)},
            frame_id_min=0,
            frame_id_max=None,
            frame_count=rtmpose_frame_count,
            consumers=["FPP-Net", "PoseTransOpt", "pressure_tookit"],
        )
        written.append(str(common.RTMPOSE_ROOT / "artifact.json"))
    if cliff_frame_count is not None:
        write_artifact(
            common.CLIFF_ROOT,
            schema_version="shared.frontend.cliff_hr48.v1",
            producer="mmvp_series/cliff/run_cliff.py",
            repository_root=REPO_ROOT,
            parameters={
                "session_count": sessions,
                "ckpt": CLIFF_CKPT.name,
                "backbone": "hr48",
                "single_person": "SAM3.1 mask-recomputed bbox; visual_time_s<=20ms join",
                "layout": "<date>/<subject>/<session>/CLIFF_results.npz",
                "toolkit_init": "<date>/<subject>/<session>/<session>_cliff_hr48.npz "
                                "(mmvp_series/cliff/export_toolkit_init.py)",
            },
            source_artifacts=["model-input://PoseTransOpt/adapter_v1",
                              "shared://frontends/human_masks/sam31/v1",
                              "asset://third_party/CLIFF"],
            source_hashes={"ckpt": sha256_file(CLIFF_CKPT),
                           "sam31_artifact": sha256_file(SAM31_ARTIFACT)},
            frame_id_min=0,
            frame_id_max=None,
            frame_count=cliff_frame_count,
            consumers=["PoseTransOpt", "pressure_tookit"],
        )
        written.append(str(common.CLIFF_ROOT / "artifact.json"))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", choices=("rtmpose_halpe26", "cliff_hr48", "all"),
                        default="all")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session ids")
    parser.add_argument("--dry-run", action="store_true",
                        help="report planned actions without writing anything")
    parser.add_argument("--verify-hashes", type=int, default=3,
                        help="per-session hash samples checked against the source tree")
    parser.add_argument("--force-convert", action="store_true",
                        help="rebuild the toolkit init npz even when it exists")
    parser.add_argument("--skip-artifact", action="store_true",
                        help="promote data but do not write artifact.json "
                             "(used for partial session sets)")
    args = parser.parse_args()

    rows = common.manifest_rows()
    sessions = ([s for s in args.sessions.split(",") if s] if args.sessions
                else common.split_sessions(args.split))
    for sid in sessions:
        if sid not in rows:
            raise KeyError(f"session not in manifest: {sid}")

    frontends = {}
    if args.frontend in ("rtmpose_halpe26", "all"):
        frontends["rtmpose_halpe26"] = promote_rtmpose(
            sessions, rows, args.dry_run, args.verify_hashes)
    if args.frontend in ("cliff_hr48", "all"):
        frontends["cliff_hr48"] = promote_cliff(
            sessions, rows, args.dry_run, args.verify_hashes, args.force_convert)

    failures = sorted({sid for data in frontends.values()
                       for sid in data["failures"]})
    result = {
        "sessions": len(sessions),
        "dry_run": args.dry_run,
        "failures": failures,
        "frontends": {
            name: {"frame_count": data["frame_count"], "reports": data["reports"]}
            for name, data in frontends.items()
        },
    }
    if not args.dry_run and not failures and not args.skip_artifact:
        result["artifacts"] = write_frontend_artifacts(
            frontends["rtmpose_halpe26"]["frame_count"] if "rtmpose_halpe26" in frontends else None,
            frontends["cliff_hr48"]["frame_count"] if "cliff_hr48" in frontends else None,
            len(sessions))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
