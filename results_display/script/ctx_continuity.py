"""续写链连续性诊断（2026-11-01）：T2M 滚转/航向稳定性的量化判据。

读 eval_motion 的会话 npz（joint_xyz_world，z-up），输出每会话三组量：
  1. flip 计数 —— 逐帧世界系根旋转测地角 >30° 的单帧跳变次数；
  2. seam |Δyaw| —— 发射窗口边界（首窗全发、后续每 stride 帧新发）处
     的航向跳变分布（mean/p50/p80/p95/max）——「窗间 |Δyaw| 收紧」判据；
  3. 全帧 |Δyaw| —— 逐帧航向变化的 mean/p99（基线对照）。

用法：
    python results_display/script/ctx_continuity.py \
        --npz-dirs results/AnySole/<model>/<variant>/predictions/eval_motion \
        --mode T2M [--seam 10] [--flip-deg 30] [--out continuity.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def body_frame(j: np.ndarray, names: list) -> np.ndarray:
    """(T,24,3) → (T,3,3) 世界系身体标架（列 = [up, lateral, forward]，z-up）。"""
    up = j[:, names.index("spine3")] - j[:, names.index("pelvis")]
    lateral = j[:, names.index("left_hip")] - j[:, names.index("right_hip")]
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    lateral = lateral - (lateral * up).sum(1, keepdims=True) * up
    lateral = lateral / np.linalg.norm(lateral, axis=1, keepdims=True)
    forward = np.cross(up, lateral)
    return np.stack([up, lateral, forward], axis=-1)  # (T,3,3) 正交右手系


def heading_deg(forward: np.ndarray) -> np.ndarray:
    """世界系航向（z-up：y 为正前参考，atan2 连续化）。"""
    h = np.degrees(np.arctan2(forward[:, 1], forward[:, 0]))
    return h


def geodesic_deg(r: np.ndarray) -> np.ndarray:
    """相邻帧根旋转测地角（度）。"""
    d = np.einsum("tij,tkj->tik", r[1:], r[:-1])  # R_{t+1}^T R_t
    tr = np.trace(d, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def session_stats(path: Path, seam: int, flip_deg: float) -> dict:
    d = np.load(path)
    j = d["joint_xyz_world"][d["valid_mask"]]
    names = [str(n) for n in d["joint_names"]]
    if "spine3" not in names or "left_hip" not in names:
        raise ValueError(f"{path.name}: missing spine3/left_hip joints")
    r = body_frame(j, names)
    geo = geodesic_deg(r)
    fwd = r[:, :, 2]
    h = heading_deg(fwd)
    dh = np.abs(np.diff(np.unwrap(h, period=360.0)))
    seams = np.arange(seam, len(h), seam) - 1 if seam > 0 else np.array([])
    seam_dh = dh[seams] if len(seams) else np.array([])
    return {
        "session": Path(path.stem).stem.split("_")[0],
        "n_frames": int(len(h)),
        "flips": int((geo > flip_deg).sum()),
        "max_geodesic": float(geo.max()) if len(geo) else 0.0,
        "dh_mean": float(dh.mean()) if len(dh) else 0.0,
        "dh_p99": float(np.percentile(dh, 99)) if len(dh) else 0.0,
        "seam_n": int(len(seam_dh)),
        "seam_mean": float(seam_dh.mean()) if len(seam_dh) else 0.0,
        "seam_p50": float(np.percentile(seam_dh, 50)) if len(seam_dh) else 0.0,
        "seam_p80": float(np.percentile(seam_dh, 80)) if len(seam_dh) else 0.0,
        "seam_p95": float(np.percentile(seam_dh, 95)) if len(seam_dh) else 0.0,
        "seam_max": float(seam_dh.max()) if len(seam_dh) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dirs", nargs="+", required=True,
                    help="eval_motion 目录（可多个）")
    ap.add_argument("--mode", default="T2M",
                    help="会话 npz 后缀模式（默认 T2M）")
    ap.add_argument("--seam", type=int, default=10,
                    help="发射窗口步长（链式 10 / 非重叠 20）")
    ap.add_argument("--flip-deg", type=float, default=30.0,
                    help="单帧跳变计数阈值（度）")
    ap.add_argument("--out", default=None, help="CSV 输出路径（可选）")
    args = ap.parse_args()

    rows = []
    for d in args.npz_dirs:
        files = sorted(glob.glob(str(Path(d) / f"*_{args.mode}.npz")))
        if not files:
            # 基线（如 Step2Motion）注册口径 = "{session_id}.npz" 无模式后缀。
            files = sorted(glob.glob(str(Path(d) / "*.npz")))
        for p in files:
            rows.append(session_stats(Path(p), args.seam, args.flip_deg))

    if not rows:
        raise SystemExit("no npz found")
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))

    if args.out is not None:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
