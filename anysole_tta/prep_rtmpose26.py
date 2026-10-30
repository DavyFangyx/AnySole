"""AnySole TTA 用 RTMPose-Halpe26 检测器缓存组装（推理时可得 2D 证据）。

源 = AnysoleWorkspace/shared/frontends/rtmpose_halpe26/v1/<date>/<subject>/<sid>/
     keypoints_cam{N}/{frame_id:06d}.npy——RTMPose-m HALPE-26 真实检测器输出，
     独立于被优化模型；图像 = facts 原始帧（1624×1240，含畸变像素空间）。
输出 = results/baselines/HMR2.0/data/rtmpose26/cam{N}/{sid}.npz：
     kp (N,26,2) px float32 / score (N,26) float32 / frame_id (N,) int64

用法（base env，仅 numpy）：
  python anysole_tta/prep_rtmpose26.py --sessions S13011,S14011
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（anysole/anysole_tta/hmr2_tta）
from hmr2_tta.paths import RESULTS_ROOT  # noqa: E402

FRONTEND_ROOT = Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/"
                     "shared/frontends/rtmpose_halpe26/v1")
CAM_NAMES = {1: "keypoints_cam1", 2: "keypoints_cam2",
             3: "keypoints_cam3", 4: "keypoints_cam4"}


def find_session(sid: str) -> Path:
    hits = sorted(FRONTEND_ROOT.glob(f"*/*/{sid}"))
    if not hits:
        raise FileNotFoundError(f"{sid}: no frontend dir under {FRONTEND_ROOT}")
    return hits[0]


def assemble(sid: str) -> None:
    root = find_session(sid)
    for cam, name in CAM_NAMES.items():
        files = sorted((root / name).glob("*.npy"))
        n = len(files)
        kp = np.empty((n, 26, 2), dtype=np.float32)
        score = np.empty((n, 26), dtype=np.float32)
        frame_id = np.empty(n, dtype=np.int64)
        for i, f in enumerate(files):
            d = np.load(f, allow_pickle=True).item()
            kp[i] = d["keypoints"]
            score[i] = d["keypoint_scores"]
            frame_id[i] = int(d["frame_id"])
        if not (np.diff(frame_id) == 1).all():
            raise ValueError(f"{sid} cam{cam}: frame_id not contiguous "
                             f"{frame_id[:5]}..")
        out = RESULTS_ROOT / "data/rtmpose26" / f"cam{cam}" / f"{sid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, kp=kp, score=score, frame_id=frame_id)
        print(f"[ok] {sid} cam{cam}: {n} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="S13011")
    args = ap.parse_args()
    for sid in args.sessions.split(","):
        assemble(sid.strip())


if __name__ == "__main__":
    main()
