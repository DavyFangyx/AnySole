"""D_Test5: 四基线触觉适配模式验证（Agent_06 前置）。

把 AnySole 原始触觉（每脚 4 宽行 x 12 长列 = 48 格）逐帧重采样到四个 baseline
实际消费的触觉点上。**点与点的对应关系一律按绝对空间最近点**（欧氏距离），
不用分带/块膨胀等索引式映射：

    共同绝对空间 = SMPL 模板足底坐标系（FPP-Net essentials 的 smpl_template.obj）：
      - MMVP 31x11 每个掩码像素的位置 = insole2smpl 顶点映射（实测每像素 1 顶点）
      - 我方 48 格位置 = 厂商布局文件（calibration/foot_sensor_layout/*foot_dots.csv，
        dot_index = 通道号+1 已数值验证），norm 坐标按各脚足底包围盒对齐到模板足底系
      - 布局文件的 x 轴（内外侧方向）与模板 x 轴同向为默认假定，--mirror-x 可翻转

    四基线面板（2x2 排布，触觉渲染风格与 r_test1_visualize_anysole.py 一致，
    复用 render_common.render_foot_panel / pressure_to_heatmap）：
      MotionPRO       FRAPPE 实际消费的 96x96 图像（与我方 48 格同源，bilinear resize）
      Step2Motion     16 通道/脚，通道空间位置 = process_gait 冻结池化组中心
                      （48 格按最近组中心归入 16 通道，与冻结池化等价）
      pressure_tookit 31x11 原始压力（每掩码像素 = 最近我方格的值；load_contact 只判 !=0）
      VP-MoCap        31x11 sigmoidNorm(weight) 归一化压力（weight = 静立帧总压，D7 简化口径）

输出 results_display/data/d_test5_baseline_tactile/：
    <session>_adapted_tactile.gif   2x2 逐帧动画（同帧对齐）
    numbers.json                    关键数值（最近点距离统计等）
    summary.md                      适配口径说明

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test5_baseline_tactile.py
    python results_display/script/d_test5_baseline_tactile.py --stride 4 --max-frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loguru import logger as log  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from utils import render_common as rc  # noqa: E402

DEFAULT_SEQ_DIR = REPO_ROOT / "AnysoleWorkspace/derived/MotionPRO/sequences/cam3/20260808/S12/S12072"
DEFAULT_ESSENTIALS = REPO_ROOT / "Baselines/VP-MoCap/FPP-Net/essentials/insole2cont"
DEFAULT_OUT = REPO_ROOT / "results_display" / "data" / "d_test5_baseline_tactile"
LAYOUT_DIR = REPO_ROOT / "AnysoleWorkspace" / "calibration" / "foot_sensor_layout"

# 面板基线标注色（与各面板标题同色，仅作身份标识）
C_MOTIONPRO = (255, 170, 80)
C_S2M = (80, 200, 255)
C_TOOLKIT = (230, 120, 230)
C_FPP = (200, 120, 200)

GAP = 16
HEADER_H = 58
DOWNSCALE = 0.62


# ---------------------------------------------------------------- 几何

def sole_bbox(ess: dict, foot: str) -> tuple[float, float, float, float]:
    """模板足底包围盒 (xmin, xmax, zmin, zmax)。z：+z=脚尖，-z=脚跟（实测 corr≈−0.996）。"""
    ids = ess["footIds" + foot]
    v = ess["v_template"][ids]
    return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 2].min()), float(v[:, 2].max())


def load_dots(foot: str) -> np.ndarray:
    """厂商布局 48 点 norm 坐标（(48,2)，行序 = 通道序，已数值验证）。"""
    name = {"L": "leftfoot", "R": "rightfoot"}[foot]
    return np.genfromtxt(LAYOUT_DIR / f"{name}_dots.csv", delimiter=",",
                         skip_header=1, usecols=(1, 2)).astype(np.float64)


def our_cell_positions(ess: dict, foot: str, mirror_x: bool = False) -> np.ndarray:
    """我方 48 格在模板足底系的 (x,z) 位置（(48,2)，索引 = 通道号）。

    厂商布局 norm 坐标：y = 脚长方向（大端=脚尖），x = 脚宽方向。
    对齐规则：x/y 极值分别对齐该脚足底包围盒的 x/z 极值（归一化文件无绝对 mm
    尺度）。x 轴（内外侧）方向假定与模板 x 同向，--mirror-x 翻转。
    """
    dots = load_dots(foot)
    xmin, xmax, zmin, zmax = sole_bbox(ess, foot)
    xn = (dots[:, 0] - dots[:, 0].min()) / max(dots[:, 0].max() - dots[:, 0].min(), 1e-9)
    yn = (dots[:, 1] - dots[:, 1].min()) / max(dots[:, 1].max() - dots[:, 1].min(), 1e-9)
    if mirror_x:
        xn = 1.0 - xn
    x = xmin + xn * (xmax - xmin)
    z = zmax - yn * (zmax - zmin)                            # y 大端（脚尖）→ +z
    return np.stack([x, z], axis=1)


def mmvp_pixel_positions(ess: dict, foot: str) -> np.ndarray:
    """MMVP 31x11 每个像素的模板 (x,z) 位置，掩码外为 NaN。"""
    mapping = ess["insole2smpl" + foot]
    pos = np.full((31, 11, 2), np.nan)
    for v_str, rc_ in mapping.items():
        if not rc_[0].size:
            continue
        v = ess["v_template"][int(v_str)]
        for r, c in zip(rc_[0], rc_[1]):
            pos[r, c] = v[[0, 2]]
    return pos


def nearest_cell_map(ess: dict, foot: str, mirror_x: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """每个 MMVP 掩码像素 → 最近我方格的编号 + 距离统计（模板系，米）。"""
    from scipy.spatial import cKDTree
    ours = our_cell_positions(ess, foot, mirror_x)
    pos = mmvp_pixel_positions(ess, foot)
    mask = np.isfinite(pos[..., 0])
    tree = cKDTree(ours)
    dist, idx = tree.query(pos[mask])                        # 每个掩码像素 → 最近格
    cell_idx = np.full((31, 11), -1, dtype=int)
    cell_idx[mask] = idx
    stats = {
        "n_px": int(mask.sum()),
        "mean_mm": float(dist.mean() * 1000),
        "max_mm": float(dist.max() * 1000),
        "median_mm": float(np.median(dist) * 1000),
        "n_cells_used": int(np.unique(idx).size),
    }
    return cell_idx, mask, stats


# ---------------------------------------------------------------- 数据

def load_nine_piece(seq_dir: Path) -> dict:
    d = {}
    d["pressure"] = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake = seq_dir / "fake_mask.npy"
    d["fake"] = np.load(fake).astype(bool).reshape(-1) if fake.is_file() else \
        np.zeros(len(d["pressure"]), dtype=bool)
    return d


def crop_cells(pressure_t: np.ndarray) -> np.ndarray:
    """(T,160,120) -> (T,2,4,12)：render_common 冻结脚框读回 48 格。"""
    out = []
    for box in (rc.LEFT_FOOT_BOX, rc.RIGHT_FOOT_BOX):
        block = np.asarray(pressure_t[:, box[0], box[1]], dtype=np.float32)
        out.append(block.reshape(-1, 4, 20, 12, 4).mean(axis=(2, 4)))
    return np.stack(out, axis=1)


def load_essentials(base: Path) -> dict:
    d = {"maskL": np.loadtxt(base / "insoleMaskL.txt").astype(np.int32),
         "maskR": np.loadtxt(base / "insoleMaskR.txt").astype(np.int32),
         "insole2smplL": np.load(base / "insole2smplL.npy", allow_pickle=True).item(),
         "insole2smplR": np.load(base / "insole2smplR.npy", allow_pickle=True).item(),
         "footIdsL": np.loadtxt(base / "footL_ids.txt").astype(np.int32),
         "footIdsR": np.loadtxt(base / "footR_ids.txt").astype(np.int32)}
    import trimesh
    d["v_template"] = np.array(trimesh.load(base / "smpl_template.obj", process=False).vertices)
    return d


def pool_48_to_16(cells: np.ndarray) -> np.ndarray:
    """(4,12) -> 16 通道（process_gait 冻结口径）：heel(col6-11) 8 通道在前，toe 8 在后。"""
    grid = np.asarray(cells).reshape(4, 12)
    toe = grid[:, :6].reshape(4, 2, 3).mean(axis=-1).reshape(8)
    heel = grid[:, 6:].reshape(4, 2, 3).mean(axis=-1).reshape(8)
    return np.concatenate([heel, toe])


def pool_ids() -> np.ndarray:
    ids = np.zeros((4, 12), dtype=int)
    for r in range(4):
        for g in range(2):
            ch = r * 2 + g
            ids[r, 6 + 3 * g:6 + 3 * g + 3] = ch
            ids[r, 3 * g:3 * g + 3] = ch + 8
    return ids


# ---------------------------------------------------------------- 面板（r_test1 风格）

def panel_with_label(panel: np.ndarray, label: str, color: tuple) -> np.ndarray:
    """在 render_foot_panel 画布左上角叠加基线身份标签（同款字体）。"""
    img = Image.fromarray(panel)
    ImageDraw.Draw(img).text((12, 10), label, font=rc.INFO_FONT, fill=color)
    return np.asarray(img)


def compose_2x2(panels: list[tuple[np.ndarray, str, tuple]], header: str) -> Image.Image:
    W, H = rc.FOOT_W, rc.CANVAS_H
    canvas = Image.new("RGB", (2 * W + GAP, HEADER_H + 2 * H + GAP), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=(48, 48, 56))
    draw.text((canvas.width // 2, 14), header, font=rc.TITLE_FONT,
              fill=(230, 230, 235), anchor="ma")
    for k, (panel, label, color) in enumerate(panels):
        x = (k % 2) * (W + GAP)
        y = HEADER_H + (k // 2) * (H + GAP)
        canvas.paste(Image.fromarray(panel_with_label(panel, label, color)), (x, y))
    if DOWNSCALE < 1.0:
        canvas = canvas.resize((int(canvas.width * DOWNSCALE), int(canvas.height * DOWNSCALE)),
                               Image.BILINEAR)
    return canvas


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test5: 四基线触觉适配模式验证（GIF）")
    p.add_argument("--seq-dir", type=Path, default=DEFAULT_SEQ_DIR)
    p.add_argument("--essentials", type=Path, default=DEFAULT_ESSENTIALS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stride", type=int, default=2, help="GIF 抽帧步长（40Hz 原始帧率）")
    p.add_argument("--max-frames", type=int, default=0, help="GIF 帧数上限（0=不限）")
    p.add_argument("--mirror-x", action="store_true",
                   help="布局 x 轴（内外侧）相对模板 x 翻转（默认同向假定）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_nine_piece(args.seq_dir)
    ess = load_essentials(args.essentials)
    cells = crop_cells(data["pressure"])                            # (T,2,4,12)
    valid = np.where(~data["fake"])[0]
    sid = args.seq_dir.name
    log.info("{}: {} 帧（有效 {}），4x12 值域 {:.0f}-{:.0f}",
             sid, len(cells), len(valid), cells.min(), cells.max())

    # ---- 几何映射（一次性）：每掩码像素 → 最近我方格（厂商布局坐标，模板足底系）
    maps = {}
    dist_stats = {}
    for foot in ("L", "R"):
        cell_idx, mask, stats = nearest_cell_map(ess, foot, args.mirror_x)
        maps[foot] = (cell_idx, mask)
        dist_stats[foot] = stats
        log.info("{}: 掩码像素 {}，最近格距离 均值 {:.1f}mm / 中位 {:.1f}mm / 最大 {:.1f}mm，"
                 "用到我方 {} / 48 格", foot, stats["n_px"], stats["mean_mm"],
                 stats["median_mm"], stats["max_mm"], stats["n_cells_used"])
    n_mask = int(maps["L"][1].sum() + maps["R"][1].sum())
    mmvp_mask = np.concatenate([maps["L"][1], maps["R"][1]], axis=1)

    # ---- FPP weight（D7 简化口径：左右压力和最大帧的总压，0-255 尺度）
    standing = valid[np.argmax(cells[valid].sum(axis=(1, 2, 3)))]
    weight = float(cells[standing].sum())
    pw = weight / n_mask
    log.info("FPP weight（静立帧 {}，D7 简化）= {:.1f}，每像素当量 pw = {:.4f}",
             standing, weight, pw)

    # ---- MotionPRO：FRAPPE 实际消费（160x120 整图 bilinear → 96x96，image_pressure 口径）
    import torch
    t = torch.from_numpy(np.ascontiguousarray(data["pressure"])).float()
    r96 = torch.nn.functional.interpolate(t.unsqueeze(1), size=(96, 96),
                                          mode="bilinear", align_corners=False).squeeze(1).numpy()

    pids = pool_ids()
    frames = valid[:: args.stride]
    if args.max_frames:
        frames = frames[: args.max_frames]
    log.info("GIF 帧数 {}", len(frames))

    imgs = []
    n = len(cells)
    for fi in frames:
        c = cells[fi]
        # 各面板的触觉块：与 render_common 口径一致（4x12 → rot90 成 (12,4) 竖版）
        block_raw = [np.rot90(c[0], k=1), np.rot90(c[1], k=1)]       # MotionPRO
        up_l = pool_48_to_16(c[0])[pids]                             # (4,12) 组值回填
        up_r = pool_48_to_16(c[1])[pids]
        block_s2m = [np.rot90(up_l, k=1), np.rot90(up_r, k=1)]
        grid_raw = np.zeros((31, 22), dtype=np.float32)              # MMVP 原始值
        for k, foot in enumerate(("L", "R")):
            cell_idx, mask = maps[foot]
            g = c[k].ravel()[cell_idx]
            g[~mask] = 0
            grid_raw[:, k * 11:(k + 1) * 11] = g
        sig = 1.0 / (1.0 + np.exp(-(grid_raw - pw) / max(pw, 1e-6)))
        block_toolkit = [grid_raw[:, :11], grid_raw[:, 11:]]         # 31x11/脚
        block_fpp = [(sig[:, :11] * 255.0), (sig[:, 11:] * 255.0)]

        # 四面板（r_test1 触觉面板风格：render_foot_panel）
        p_motion = rc.render_foot_panel(block_raw[0], block_raw[1], sid, fi, n, 40.0, True)
        p_s2m = rc.render_foot_panel(block_s2m[0], block_s2m[1], sid, fi, n, 40.0, True)
        p_toolkit = rc.render_foot_panel(block_toolkit[0], block_toolkit[1], sid, fi, n, 40.0, True)
        p_fpp = rc.render_foot_panel(block_fpp[0], block_fpp[1], sid, fi, n, 40.0, True)

        header = (f"{sid}  四基线触觉适配（源=我方 4x12 每脚 48 格）  "
                  f"frame {fi}/{n-1}  t={fi/40:.2f}s")
        frame_img = compose_2x2([
            (p_motion, "MotionPRO · FRAPPE 96x96 输入", C_MOTIONPRO),
            (p_s2m, "Step2Motion · 16 通道/脚", C_S2M),
            (p_toolkit, "pressure_tookit · 31x11 原始压力", C_TOOLKIT),
            (p_fpp, f"VP-MoCap · sigmoidNorm (pw={pw:.1f})", C_FPP),
        ], header)
        imgs.append(frame_img)
    log.info("合成 GIF {} 帧，每帧 {}x{}", len(imgs), *imgs[0].size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gif = args.out_dir / f"{sid}_adapted_tactile.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=40 * args.stride,
                 loop=0, optimize=False)
    log.info("wrote {}", gif)

    numbers = {
        "session": sid, "n_frames_gif": len(imgs), "stride": args.stride,
        "mmvp_mask_px": n_mask,
        "mapping": "nearest-cell by euclidean distance in SMPL template sole frame",
        "our_grid": "vendor layout (calibration/foot_sensor_layout/*foot_dots.csv), "
                    "dot_index = channel+1 (verified); x/y extents aligned to per-foot sole bbox",
        "mirror_x": bool(args.mirror_x),
        "nearest_dist_mm": dist_stats,
        "fpp_weight": weight, "fpp_pw": pw, "fpp_weight_frame": int(standing),
        "motionpro_input": "bilinear 96x96 /255 (image_pressure.py 口径)",
        "s2m_channels": "16/foot, heel[0-7] then toe[8-15] (process_gait 冻结)",
    }
    (args.out_dir / "numbers.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2))
    summary = f"""# D_Test5 四基线触觉适配模式验证（GIF）

产物：`{gif.name}` —— 2x2 逐帧动画（四基线同帧对齐，触觉面板风格与
r_test1_visualize_anysole.py 一致，复用 render_common.render_foot_panel）：
MotionPRO 96x96 实际输入 | Step2Motion 16 通道 | pressure_tookit 31x11 |
VP-MoCap sigmoidNorm 31x11。

## 适配口径（点对点绝对空间最近）

- 共同坐标系 = SMPL 模板足底（FPP-Net essentials smpl_template.obj）。
- MMVP 31x11 像素位置 = insole2smpl 顶点映射（实测每像素 1 顶点）。
- 我方 48 格位置 = 厂商布局文件（calibration/foot_sensor_layout/*foot_dots.csv，
  dot_index=通道+1 已数值验证；norm 无绝对 mm，x/y 极值对齐各脚足底包围盒）。
- 每个 MMVP 掩码像素 ← 欧氏距离最近的我方格值（scipy cKDTree）。
- Step2Motion 16 通道 = process_gait 冻结池化组（48 格→最近组中心归入，与池化等价）。
- FPP weight（D7 简化）= 左右压力和最大帧总压 {weight:.1f}，pw=weight/{n_mask}={pw:.4f}。
- MotionPRO 面板 = 我方 48 格（其 96x96 输入即此图 bilinear resize，信息同源）。

## 已知假定

布局文件未标注内外侧方向（x 轴哪端为内侧），默认与模板 x 轴同向；
`--mirror-x` 翻转。有内外侧标定后以标定为准。
"""
    (args.out_dir / "summary.md").write_text(summary)
    log.info("wrote numbers.json / summary.md")


if __name__ == "__main__":
    main()
