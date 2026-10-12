#!/usr/bin/env python3
"""Materialise the FPP-Net model_inputs tree (adapter_v1).

Layout per session:

    model-input://FPP-Net/adapter_v1/<date>/<subject>/<session>/
    ├── color/            symlinks to the canonical shared RGB frames
    ├── keypoints/        RTMPose HALPE-26 sidecars with frame ids
    │                     (produced by run_rtmpose.py)
    └── adapter_manifest.json

The 31x11 insole data is NOT copied here: FPP-Net reads the single public
representation ``shared://representations/tactile/mmvp_31x11/v1`` read-only
at train time (see PED_tempKPCont.tactile_root).  The manifest records the
link/verification result per session.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

FPP_ROOT = WORKSPACE / "model_inputs/FPP-Net/adapter_v1"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions/cam3"
TACTILE_ROOT = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
RTMPOSE_ROOT = WORKSPACE / "shared/frontends/rtmpose_halpe26/v1"
ADAPTER_VERSION = "adapter_v1"


def facts_root(cam: int = 3) -> Path:
    """shared facts camera root (cam3 = the frozen public tree)."""
    return FACTS_ROOT.parent / f"cam{cam}"


def adapter_root(cam: int = 3) -> Path:
    """FPP-Net adapter root; cam3 keeps the frozen public path, other cameras
    get a ``cam{N}`` sibling subtree."""
    return FPP_ROOT if cam == 3 else FPP_ROOT / f"cam{cam}"


def _cam_list(value: str) -> tuple[int, ...]:
    if value.strip() == "all":
        return (1, 2, 3, 4)
    try:
        cams = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError:
        raise SystemExit(f"--cam expects a comma list or 'all', got {value!r}")
    unknown = set(cams) - {1, 2, 3, 4}
    if unknown:
        raise SystemExit(f"unknown camera(s): {sorted(unknown)}")
    if not cams:
        raise SystemExit("--cam list is empty")
    return cams


def link_dir(source_dir: Path, link: Path, force: bool = False) -> str:
    """目录符号链接指向共享前端（幂等）；源不存在时保持缺失。"""
    if not source_dir.is_dir():
        return "no_frontend"
    if link.is_symlink():
        if os.readlink(link) == os.path.relpath(source_dir, link.parent):
            return "kept"
        if not force:
            raise FileExistsError(f"refusing to replace existing link: {link}")
        link.unlink()
    elif link.exists():
        return "physical_dir_kept"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(source_dir, link.parent))
    return "linked"


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


def link_one(source: Path, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"refusing to replace existing adapter file: {target}")
        target.unlink()
    target.symlink_to(os.path.relpath(source, target.parent))


def build_session(row: dict, cam: int, force: bool) -> dict:
    date, subject, sid = session_parts(row)
    session_root = adapter_root(cam) / date / subject / sid
    n = int(row["n_frames"])
    color_dir = session_root / "color"
    frames = sorted(p for p in (facts_root(cam) / date / subject / sid / "rgb").iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(frames) != n:
        raise ValueError(f"{sid}: shared RGB count {len(frames)} != manifest {n}")
    for index, source in enumerate(frames):
        link_one(source, color_dir / f"{index:06d}{source.suffix.lower()}", force)

    # verify the public insole representation covers every canonical frame
    insole_dir = TACTILE_ROOT / date / subject / sid / "insole"
    if not insole_dir.is_dir():
        raise FileNotFoundError(f"{sid}: public insole representation missing: {insole_dir}")
    missing_insole = [i for i in range(n) if not (insole_dir / f"{i:06d}.npy").is_file()]
    if missing_insole:
        raise FileNotFoundError(f"{sid}: public insole missing frames {missing_insole[:10]}")

    # keypoints：cam3 沿用前端符号链接刷新（run_rtmpose 冻结行为）；cam!=3
    # 由本工具直接把 per-camera 前端目录链接进适配器树
    keypoints_dir = session_root / "keypoints"
    keypoints_link = ""
    if cam != 3:
        keypoints_link = link_dir(
            RTMPOSE_ROOT / date / subject / sid / f"keypoints_cam{cam}",
            keypoints_dir, force)
    n_keypoints = len(list(keypoints_dir.glob("*.npy"))) if keypoints_dir.is_dir() else 0
    report = {
        "adapter_version": ADAPTER_VERSION,
        "session_id": sid, "date": date, "subject": subject,
        "camera": f"cam{cam}",
        "n_frames": n, "n_color": len(frames),
        "n_keypoints": n_keypoints,
        "keypoints_link": keypoints_link or None,
        "insole_source": str(insole_dir),
        "insole_private_copy": False,
    }
    (session_root / "adapter_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--cam", default="3",
                        help="camera(s) to build adapter inputs for: 'all' or "
                             "comma list (1,2,3,4), default 3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_manifest()
    sessions = [s for s in args.sessions.split(",") if s] if args.sessions else split_sessions(args.split)
    reports = [build_session(rows[sid], cam, args.force)
               for cam in _cam_list(args.cam)
               for sid in sessions]
    print(json.dumps({"sessions": len(reports), "reports": reports},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
