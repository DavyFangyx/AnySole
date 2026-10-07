#!/usr/bin/env python3
"""D_Test6 收口查看器：读生产 floor npy + 深度帧，验证"人站在地面上"。

输入（默认自动定位）：
- floor npy：``model_inputs/pressure_toolkit/v1/annotations/<date>/floor_info/
  floor_<subject>.npy``（keys: trans / normal / depth2floor）
- 深度帧与 ROI：同日期 ``floor_artifact_<date>.json`` 记录的 depth_file 与
  roi_rows/roi_cols（可用 --depth-png / --roi 覆盖）

输出四联图 ``results_display/DataTest/D6Test_floor/<date>_<subject>_floor_view.png``：
- 左上：全帧深度点云经 depth2floor 到 floor 系，按高度着色（<10cm 脚底区红色），
  y=0 平面 + 坐标轴 —— 人应立在地面之上（y>0）；
- 右上：侧视图（floor-z vs height y），红线 y=0 —— 人体轮廓应从 0 升起；
- 左下：RGB 帧 + 拟合用 ROI 矩形；
- 右下：全帧点云高度直方图 + 关键统计（ROI 残差、上部图像区高度中位等）。

只读，不写任何模型输入。环境：touch_gait。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(REPO_ROOT / "Baselines" / "pressure_tookit"))

from AnysoleWorkspace.tool.adapters.mmvp_series.calibration.calibration import (  # noqa: E402
    load_calibration,
)
from AnysoleWorkspace.tool.adapters.mmvp_series.pressure_tookit import floor as floor_mod  # noqa: E402
from lib.utils.depth_utils import depth2PointCloud  # noqa: E402

OUT_DIR = REPO_ROOT / "results_display" / "DataTest" / "D6Test_floor"
ADAPTER_ROOT = WORKSPACE / "model_inputs" / "pressure_toolkit" / "v1"


def load_floor(date: str, subject: str) -> dict:
    path = ADAPTER_ROOT / "annotations" / date / "floor_info" / f"floor_{subject}.npy"
    info = np.load(path, allow_pickle=True).item()
    return {
        "trans": np.asarray(info["trans"], dtype=np.float64),
        "normal": np.asarray(info["normal"], dtype=np.float64),
        "depth2floor": np.asarray(info["depth2floor"], dtype=np.float64),
        "path": path,
    }


def artifact_defaults(date: str) -> tuple[Path, tuple[int, int, int, int], int]:
    art = ADAPTER_ROOT / "annotations" / date / "floor_info" / f"floor_artifact_{date}.json"
    data = json.loads(art.read_text(encoding="utf-8"))
    roi = (data["roi_rows"][0], data["roi_rows"][1],
           data["roi_cols"][0], data["roi_cols"][1])
    return Path(data["depth_file"]), roi, int(data["frame_id"])


def sample(points: np.ndarray, n: int = 6000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def plot_view(date: str, subject: str, floor: dict, depth_png: Path,
              roi: tuple[int, int, int, int], depth_map: np.ndarray,
              fx: float, fy: float, cx: float, cy: float,
              tag: str = "") -> Path:
    pc = depth2PointCloud(depth_map, fx, fy, cx, cy)
    rows, cols = depth_map.shape
    full = pc[np.isfinite(pc).all(axis=2)].reshape(-1, 3)
    full_floor = floor_mod.transform_points(floor["depth2floor"], full)

    # ROI 点（拟合所用区域）
    roi_pts = pc[roi[0]:roi[1], roi[2]:roi[3], :].reshape(-1, 3)
    roi_pts = roi_pts[np.isfinite(roi_pts).all(axis=1)]
    roi_floor = floor_mod.transform_points(floor["depth2floor"], roi_pts)
    residual = float(np.sqrt(np.mean(roi_floor[:, 1] ** 2)) * 1000.0)

    # 上部图像区（人/背景，应在地面之上）
    upper = pc[0:int(rows * 0.30), :, :].reshape(-1, 3)
    upper = upper[np.isfinite(upper).all(axis=1)]
    upper_floor = floor_mod.transform_points(floor["depth2floor"], upper)

    pts = sample(full_floor)
    heights = pts[:, 1]
    colors = np.zeros((len(pts), 3))
    feet = heights < 0.10
    below = heights < -0.02
    colors[below] = (0.45, 0.45, 0.45)          # 地面之下（噪声）
    colors[feet] = (0.85, 0.15, 0.15)           # 脚底区
    rest = ~(feet | below)
    colors[rest] = np.stack([
        np.zeros(rest.sum()),
        np.clip(0.25 + heights[rest] / 1.5, 0, 1),
        np.clip(1.0 - heights[rest] / 1.5, 0, 1),
    ], axis=1)

    fig = plt.figure(figsize=(15.5, 11.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    # 左上：floor 系 3D 点云
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=0.5, c=colors)
    px = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 5)
    pz = np.linspace(pts[:, 2].min(), pts[:, 2].max(), 5)
    gx, gz = np.meshgrid(px, pz)
    ax3.plot_surface(gx, np.zeros_like(gx), gz, alpha=0.18, color="0.35")
    center = pts.mean(axis=0)
    scale = float(np.ptp(pts[:, :2])) * 0.35
    ax3.quiver(center[0], center[2], center[1], 0.0, 0.0, 0.35,
               color="black", linewidth=2, label="floor normal [0,1,0]")
    ax3.quiver(center[0], center[2], center[1], scale, 0, 0,
               color="red", label="x")
    ax3.quiver(center[0], center[2], center[1], 0, 0, scale,
               color="green", label="z")
    ax3.set_xlabel("floor x (m)")
    ax3.set_ylabel("floor z (m)")
    ax3.set_zlabel("floor y = height (m)")
    ax3.set_title(f"{date} {subject}  full-frame depth in floor frame\n"
                  f"(red = foot zone <10cm, gray = below floor)",
                  fontsize=10)

    # 右上：侧视轮廓（z vs 高度）
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.scatter(pts[:, 2], pts[:, 1], s=0.5, c=colors)
    ax_side.axhline(0.0, color="crimson", linewidth=2,
                    label="floor plane y=0")
    ax_side.set_xlabel("floor z (m)")
    ax_side.set_ylabel("height y (m)")
    ax_side.set_ylim(max(-0.4, np.percentile(heights, 0.5) - 0.2),
                     min(2.6, np.percentile(heights, 99.5) + 0.2))
    ax_side.legend(loc="upper right", fontsize=8)
    ax_side.set_title("side view: person rises from y=0", fontsize=10)

    # 左下：RGB + ROI
    ax_img = fig.add_subplot(gs[1, 0])
    rgb_path = Path(str(depth_png).replace("/depth/", "/color/")).with_suffix(".jpg")
    if rgb_path.is_file():
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        ax_img.imshow(rgb)
        from matplotlib.patches import Rectangle
        ax_img.add_patch(Rectangle((roi[2], roi[0]), roi[3] - roi[2],
                                   roi[1] - roi[0], fill=False,
                                   edgecolor="lime", linewidth=2))
    ax_img.set_title(f"RGB frame (green = floor-fit ROI)\n"
                     f"depth: {depth_png.parent.name}/{depth_png.name}",
                     fontsize=10)
    ax_img.axis("off")

    # 右下：高度直方图 + 统计
    ax_hist = fig.add_subplot(gs[1, 1])
    band = (heights > -0.3) & (heights < 2.6)
    ax_hist.hist(heights[band], bins=120, color="0.55")
    ax_hist.axvline(0.0, color="crimson", linewidth=2)
    ax_hist.set_xlabel("height y in floor frame (m)")
    ax_hist.set_ylabel("points")
    stats = (f"floor npy: {floor['path'].name}\n"
             f"ROI residual = {residual:.2f} mm\n"
             f"ROI pts y: median {np.median(roi_floor[:, 1]):.3f} m\n"
             f"upper-image pts y (rows 0-30%):\n"
             f"  median {np.median(upper_floor[:, 1]):.3f} m | "
             f"p25 {np.percentile(upper_floor[:, 1], 25):.2f} | "
             f"p75 {np.percentile(upper_floor[:, 1], 75):.2f}\n"
             f"points below y=0: {(heights < 0).mean() * 100:.2f}%")
    print(stats)
    ax_hist.text(0.98, 0.97, stats, transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox={"facecolor": "white", "alpha": 0.85})
    ax_hist.set_title("full-frame height histogram (floor frame)",
                      fontsize=10)

    fig.suptitle(f"D_Test6 floor closure view - {date} {subject} "
                 f"(person standing on the floor)", fontsize=13)
    tag = f"_{tag}" if tag else ""
    out_path = OUT_DIR / f"{date}_{subject}_floor_view{tag}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--depth-png", default=None,
                        help="override depth frame; default from floor artifact")
    parser.add_argument("--roi", default=None,
                        help="override ROI r0,r1,c0,c1; default from artifact")
    parser.add_argument("--tag", default="",
                        help="suffix for the output png name")
    args = parser.parse_args()

    floor = load_floor(args.date, args.subject)
    if args.depth_png and args.roi:
        depth_png = Path(args.depth_png)
        parts = [int(item) for item in args.roi.split(",")]
        roi = (parts[0], parts[1], parts[2], parts[3])
    else:
        depth_png, roi, _ = artifact_defaults(args.date)

    import cv2
    depth_map = cv2.imread(str(depth_png), -1).astype(np.float32) / 1000.0
    cal = load_calibration(WORKSPACE / "protocol/calibration" / f"{args.date}.json")
    K = np.array(cal["K"])

    out = plot_view(args.date, args.subject, floor, depth_png, roi, depth_map,
                    float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                    tag=args.tag)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
