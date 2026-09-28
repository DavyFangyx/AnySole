#!/usr/bin/env python3
"""D_Test6: pressure_toolkit 地面估计与坐标系审计。

对四个采集日期各取一个代表 session，生成候选地面帧/ROI，用上游
``calculateFloorNormal`` 估计行走地面并输出可视化：

    results_display/DataTest/D6Test_floor/
    ├── <date>_floor_audit.png        每个日期一张 2×2 审计图
    ├── floor_candidates.csv          全部候选的残差/棋盘格偏移
    └── README.md                     四日期用户确认清单

图内元素（全英文）：
- 候选 ROI 深度点云（按离地高度着色，脚底区红色）+ 拟合平面 + 法向 + 坐标轴；
- RGB 帧 + ROI 矩形 + SAM3.1 单受试者 mask 轮廓叠加；
- 全部候选残差条形图；
- 估计地面 vs 棋盘格标定平面（旧适配器错误口径）的偏移对比。

只读外部 SAM3.1 mask 源与 shared 树；不写任何模型输入。
"""

from __future__ import annotations

import argparse
import csv
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
TOOLKIT = REPO_ROOT / "Baselines" / "pressure_tookit"
for path in (REPO_ROOT, TOOLKIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AnysoleWorkspace.tool.adapters.mmvp_series.calibration.calibration import (  # noqa: E402
    load_calibration,
)
from AnysoleWorkspace.tool.adapters.mmvp_series.pressure_tookit import floor as floor_mod  # noqa: E402

OUT_DIR = REPO_ROOT / "results_display" / "DataTest" / "D6Test_floor"
MASK_SOURCE_ROOT = Path("/data/lizhe/projects/Tactile/3_Result/processed/rgb_human_masks")

# 每日期代表 session（date -> session id）
REPRESENTATIVE = {
    "20260804": "S5011",
    "20260807": "S8011",
    "20260808": "S10101",
    "20260810": "S13011",
}

CANDIDATE_FRAME_IDS = (0, 1, 2)


def candidate_rois(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    rows, cols = shape
    return {
        "bottom_band_default": (int(rows * 0.55), int(rows * 0.90),
                                int(cols * 0.30), int(cols * 0.70)),
        "wider_band": (int(rows * 0.45), int(rows * 0.95),
                       int(cols * 0.15), int(cols * 0.85)),
        "lower_narrow": (int(rows * 0.70), int(rows * 0.95),
                         int(cols * 0.35), int(cols * 0.65)),
    }


def sample_points(points: np.ndarray, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def mask_path_for_frame(session: str, color_file: Path) -> Path | None:
    """外部 SAM3.1 单受试者 mask（只读，显示用；正式输入走 Agent A 索引）。"""
    mask = MASK_SOURCE_ROOT / session / "3" / (color_file.stem + ".png")
    return mask if mask.is_file() else None


def plot_floor_audit(date: str, records: list[dict], best: dict, rows: dict) -> Path:
    fig = plt.figure(figsize=(15.5, 11.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    # ---- 左：最佳候选的点云 / 平面 / 法向 / 坐标轴 ----
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    slices_floor = best["slices_floor"]
    points = sample_points(slices_floor)
    heights = points[:, 1]
    colors = np.zeros((len(points), 3))
    # 脚底区（离地 < 10cm）红色，其余按高度蓝→绿
    feet = heights < 0.10
    colors[feet] = (0.85, 0.15, 0.15)
    colors[~feet] = np.stack([
        np.zeros((~feet).sum()),
        np.clip(0.25 + heights[~feet] / 1.5, 0, 1),
        np.clip(1.0 - heights[~feet] / 1.5, 0, 1),
    ], axis=1)
    ax3.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.6, c=colors)
    plane_x = np.linspace(points[:, 0].min(), points[:, 0].max(), 5)
    plane_z = np.linspace(points[:, 2].min(), points[:, 2].max(), 5)
    px, pz = np.meshgrid(plane_x, plane_z)
    ax3.plot_surface(px, np.zeros_like(px), pz, alpha=0.18, color="0.35")
    center = points.mean(axis=0)
    ax3.quiver(center[0], center[2], center[1], 0.0, 0.0, 0.35,
               color="black", linewidth=2, label="floor normal [0,1,0]")
    scale = float(np.ptp(points[:, :2])) * 0.35
    ax3.quiver(center[0], center[2], center[1], scale, 0, 0, color="red", label="x")
    ax3.quiver(center[0], center[2], center[1], 0, 0, scale, color="green", label="z")
    ax3.set_xlabel("floor x (m)")
    ax3.set_ylabel("floor z (m)")
    ax3.set_zlabel("floor y = height (m)")
    ax3.set_title(f"{date}  best candidate: depth {best['frame_id']} "
                  f"{best['roi_name']}\nplane-fit RMS residual = "
                  f"{best['residual_mm']:.1f} mm",
                  fontsize=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax3.legend(
        handles=[Line2D([0], [0], color="black", lw=2),
                 Line2D([0], [0], color="red"),
                 Line2D([0], [0], color="green"),
                 Patch(facecolor="0.35", alpha=0.4)],
        labels=["floor normal [0,1,0]", "x", "z", "fitted floor plane"],
        fontsize=8, loc="upper right")

    # ---- 右上：RGB + ROI + mask 轮廓 ----
    ax_img = fig.add_subplot(gs[0, 1])
    color_file = best["color_file"]
    rgb = np.asarray(Image.open(color_file).convert("RGB"))
    ax_img.imshow(rgb)
    r0, r1, c0, c1 = best["roi"]
    from matplotlib.patches import Rectangle

    ax_img.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                               edgecolor="lime", linewidth=2))
    mask_file = best["mask_file"]
    if mask_file is not None:
        mask = np.asarray(Image.open(mask_file).convert("L")) > 0
        ax_img.contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
        ax_img.text(0.02, 0.98, "cyan = SAM3.1 single-subject mask contour",
                    transform=ax_img.transAxes, fontsize=8, va="top",
                    color="white", bbox={"facecolor": "black", "alpha": 0.5})
    ax_img.set_title(f"{date}  depth frame {best['frame_id']}  "
                     f"(green = floor ROI)", fontsize=10)
    ax_img.axis("off")

    # ---- 左下：候选残差 ----
    ax_bar = fig.add_subplot(gs[1, 0])
    labels = [f"f{r['frame_id']}\n{r['roi_name']}" for r in records]
    values = [r["residual_mm"] for r in records]
    bars = ax_bar.bar(range(len(records)), values, color="0.6")
    best_index = records.index(best)
    bars[best_index].set_color("crimson")
    ax_bar.set_xticks(range(len(records)))
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylabel("plane-fit RMS residual (mm)")
    ax_bar.set_title("candidate floor frames x ROIs (red = selected)",
                     fontsize=10)
    ax_bar.axhline(30.0, color="orange", linestyle="--", linewidth=1)
    ax_bar.text(len(records) - 0.4, 30.0, "30 mm", fontsize=8, va="bottom",
                color="orange")

    # ---- 右下：估计地面 vs 棋盘格平面 ----
    ax_cmp = fig.add_subplot(gs[1, 1], projection="3d")
    chk = best["checkerboard"]
    chk_points = (chk["depth2floor"][:3, :3] @ best["slices"].T +
                  chk["depth2floor"][:3, 3].reshape(3, 1)).T
    points2 = sample_points(chk_points)
    ax_cmp.scatter(points2[:, 0], points2[:, 2], points2[:, 1], s=0.5,
                   color="0.45")
    ax_cmp.plot_surface(px, np.zeros_like(px), pz, alpha=0.25, color="crimson")
    ax_cmp.set_title(
        f"checkerboard-plane view: mean offset = "
        f"{best['checkerboard_offset_m']:.3f} m\n"
        f"(old adapter error, task P1)", fontsize=10)
    ax_cmp.set_xlabel("x (m)")
    ax_cmp.set_ylabel("z (m)")
    ax_cmp.set_zlabel("y (m)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax_cmp.legend(
        handles=[Line2D([0], [0], color="0.45", marker="o", lw=0),
                 Patch(facecolor="crimson", alpha=0.25)],
        labels=["same depth points", "checkerboard plane (old wrong floor)"],
        fontsize=8)

    fig.suptitle(f"D_Test6 floor estimation & coordinate audit - {date}",
                 fontsize=13)
    out_path = OUT_DIR / f"{date}_floor_audit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def audit_date(date: str, session: str) -> list[dict]:
    rows_map = {}
    with (WORKSPACE / "protocol/manifests/session_manifest.jsonl").open(
            encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_map[row["session_id"]] = row
    row = rows_map[session]
    subject = row["subject_id"]
    session_dir = (WORKSPACE / "model_inputs/pressure_toolkit/v1/images"
                   / date / subject / session)
    cal = load_calibration(WORKSPACE / "protocol/calibration" / f"{date}.json")
    K = cal["K"]
    records = []
    for frame_id in CANDIDATE_FRAME_IDS:
        depth_png = session_dir / "depth" / f"{frame_id:06d}.png"
        if not depth_png.is_file():
            continue
        with Image.open(depth_png) as img:
            shape = (img.height, img.width)
        for roi_name, roi in candidate_rois(shape).items():
            estimate = floor_mod.estimate_floor(
                depth_png, roi, float(K[0, 0]), float(K[1, 1]),
                float(K[0, 2]), float(K[1, 2]))
            slices = floor_mod.slice_points(
                depth_png, roi, float(K[0, 0]), float(K[1, 1]),
                float(K[0, 2]), float(K[1, 2]))
            checkerboard = floor_mod.checkerboard_floor_mapping(
                WORKSPACE / "protocol/calibration" / f"{date}.json")
            audit = floor_mod.floor_audit(estimate, checkerboard, slices)
            color_file = session_dir / "color" / f"{frame_id:06d}.jpg"
            records.append({
                "date": date, "session": session, "frame_id": frame_id,
                "roi_name": roi_name, "roi": list(roi),
                "residual_mm": estimate["residual_mm"],
                "checkerboard_offset_m": audit["checkerboard_plane_mean_offset_m"],
                "checkerboard_rms_mm": audit["checkerboard_plane_rms_mm"],
                "normal_angle_deg": audit["normal_angle_deg"],
                "normal": estimate["normal"].tolist(),
                "slices_floor": floor_mod.transform_points(
                    estimate["depth2floor"], slices),
                "slices": slices,
                "checkerboard": checkerboard,
                "color_file": color_file,
                "mask_file": mask_path_for_frame(session, color_file),
            })
    if not records:
        raise FileNotFoundError(f"no depth candidates for {date}/{session}")
    records.sort(key=lambda r: r["residual_mm"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", default="",
                        help="comma-separated dates; default: all four")
    args = parser.parse_args()
    dates = ([d for d in args.dates.split(",") if d]
             if args.dates else sorted(REPRESENTATIVE))

    all_rows = []
    for date in dates:
        if date not in REPRESENTATIVE:
            raise ValueError(f"no representative session for {date}")
        session = REPRESENTATIVE[date]
        records = audit_date(date, session)
        best = min(records, key=lambda r: r["residual_mm"])
        png = plot_floor_audit(date, records, best, {})
        print(f"{date}: {len(records)} candidates, best residual "
              f"{best['residual_mm']:.1f} mm, checkerboard offset "
              f"{best['checkerboard_offset_m']:.3f} m -> {png.name}")
        for record in records:
            record.pop("slices_floor", None)
            record.pop("slices", None)
            record.pop("checkerboard", None)
            record.pop("color_file", None)
            record.pop("mask_file", None)
        all_rows.extend(records)

    csv_path = OUT_DIR / "floor_candidates.csv"
    fieldnames = ["date", "session", "frame_id", "roi_name", "roi",
                  "residual_mm", "checkerboard_offset_m",
                  "checkerboard_rms_mm", "normal_angle_deg", "normal"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_rows:
            writer.writerow(record)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
