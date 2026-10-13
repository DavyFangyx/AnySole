#!/usr/bin/env python3
"""cam3 命名统一软链迁移（四相机扩展 D5，2026-10-13，幂等可重跑）。

历史冻结造成「cam3 旧名 + 新相机 _cam{N}」的不对称索引。D5 裁定：一律
``{name}_cam{id}``（含 cam3）；本脚本为 cam3 旧名建 ``_cam3`` 软链（新名→旧名，
零数据移动、文件 mtime 不变），使消费端/输出端可删掉 ``cam == 3`` 分支。
工具输出端统一写 ``_cam{id}`` 后，cam3 输出路径穿过软链落在旧名目录上；
--force 重建会先删软链另建真目录（旧名目录成为历史遗留，无害）。

覆盖：
  1. AnySole bbox：  visual/hrnet/bbox                    -> bbox_cam3
  2. RTMPose：        rtmpose_halpe26/v1/<d>/<S#>/<sid>/keypoints -> keypoints_cam3
  3. DepthPro：       depthpro/v1/<d>/<S#>/<sid>/depth           -> depth_cam3
  4. CLIFF：          cliff_hr48/v1/<d>/<S#>/<sid>/CLIFF_results.npz
                     -> CLIFF_results_cam3.npz
  5. pressure_toolkit 初值：model_inputs/pressure_toolkit/v1/initialization/
                     <d>/<S#>/<sid>/<sid>_cliff_hr48.npz -> <sid>_cliff_hr48_cam3.npz

Usage: python3 AnysoleWorkspace/tool/normalize_cam_names.py [--apply]
       （无 --apply 仅列出将建/已建的链接）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
HRNET = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet"
FRONTENDS = WORKSPACE / "shared/frontends"
PTK = WORKSPACE / "model_inputs/pressure_toolkit/v1"


def link_new_to_old(new: Path, old: Path) -> str:
    """Symlink new -> old (relative). Returns action taken ("" = nothing)."""
    if new.is_symlink() or new.exists():
        return ""
    if not old.exists():
        return f"skip (source missing): {new}"
    if args.apply:
        new.symlink_to(Path(".") / old.name if new.parent == old.parent else old.relative_to(new.parent))
        return f"linked: {new} -> {old.name}"
    return f"would link: {new} -> {old.name}"


def session_dirs(root: Path) -> list[Path]:
    # <date>/<S#>/<sid> three levels below the version root; the glob hits are
    # the sid dirs themselves.
    return sorted(p for p in root.glob("*/*/*") if p.is_dir())


def main() -> int:
    actions: list[str] = []
    # 1. AnySole bbox
    actions.append(link_new_to_old(HRNET / "bbox_cam3", HRNET / "bbox"))
    # 2. RTMPose keypoints
    for d in session_dirs(FRONTENDS / "rtmpose_halpe26/v1"):
        actions.append(link_new_to_old(d / "keypoints_cam3", d / "keypoints"))
    # 3. DepthPro depth
    for d in session_dirs(FRONTENDS / "depthpro/v1"):
        actions.append(link_new_to_old(d / "depth_cam3", d / "depth"))
    # 4. CLIFF results
    for d in session_dirs(FRONTENDS / "cliff_hr48/v1"):
        actions.append(link_new_to_old(d / "CLIFF_results_cam3.npz", d / "CLIFF_results.npz"))
    # 5. pressure_toolkit CLIFF 初值
    init_root = PTK / "initialization"
    for d in sorted(init_root.glob("*/*/*")) if init_root.is_dir() else []:
        actions.append(link_new_to_old(
            d / (d.name + "_cliff_hr48_cam3.npz"), d / (d.name + "_cliff_hr48.npz")))
    done = [a for a in actions if a]
    print(f"checked {len(actions)} link sites; {len(done)} action(s)" + ("" if args.apply else " (dry-run)"))
    for a in done:
        print(" ", a)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main())
