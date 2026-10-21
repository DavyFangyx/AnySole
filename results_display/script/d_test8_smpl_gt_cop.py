"""D_Test8: SMPL GT 足底热图/CoP × 触觉热图/CoP（2×2 同帧动画）。

四面板每帧同轴（facts 40Hz 视觉帧轴，mocap 经 frames.npz mocap_time_s 对齐）：

    左上 SMPL GT 足底热图   右上 触觉数据热图
    左下 SMPL GT 的 CoP     右下 触觉的 CoP

- SMPL GT 足底热图 —— smplx 正演 GT 网格（smpl_vertices_from_archive_params），
  每脚取踝以下顶点投影到随足 2D 系（跟→趾 × 内外侧），4 行 × 12 列与触觉
  网格同形，热值 = 每格离地 <GT_CONTACT_DZ 的顶点数（clip 0-4）。GT 侧口径
  （GT足底×触觉冲突审计 2026-10-20 类型 A/B/C）：
  ① 地面 = archive y=0（g0；SMPL 足部地面审计 2026-10-19 裁定：判「陷入/
  贴地」用 y=0——旧 estimate_floor_y 关节 p5 比物理地面高（S5091=3.6cm）
  致恒接触、悬空期 GT CoP 被钉在趾端）；
  ② GT_CONTACT_DZ=3cm = 鞋底厚度稳定裕度（支撑期足底顶点实测 y≈1.4–2.0cm，
  旧 2cm 阈值恰卡足底高度 → GT 接触逐帧闪烁；3cm 让支撑期热图稳定，摆腿期
  踝抬高但趾下垂仍显示低热——GT 足底接近地面的真实反映，与触觉面板
  （足离地即卸载）的差异本身即信息）；
  ③ 随足轴在 ‖d_xz‖<0.05m（足俯仰>50°，提踵/上楼梯）时沿用上一有效轴，
  防踝→foot 水平投影退化/翻转导致的热图列序错乱（类型 C）；
- 触觉数据热图 —— 默认 shared facts pressure_48.npz 原始 48 格/脚（0-1023），
  rot90 竖直显示（col0=脚尖，heel 在上），与 D_Test4 同口径；--tactile-input
  可切交接包降噪源（denoised108=v1 / denoised108_v2 / denoised108_v3，与
  anysole/data/dataset.py 同语义；fake/valid 标志仍取 shared facts）；
- SMPL GT 的 CoP —— **同一个** anysole.data.pressure.cop_from_grid 作用于
  GT 足底热图网格（着地顶点计数质心），与触觉 CoP 同公式同单位（网格坐标
  0-1），可直接对比；
- 触觉的 CoP —— cop_from_grid 作用于触觉压力。两 CoP 面板各叠加近 40 渲染
  帧轨迹与准星；force（和值）=0 帧不画点。显示坐标 = cop_x×宽、
  cop_y 0=跟 1=趾（同 rot90）；
- 标题行附 action 编号（session_manifest.csv session_id→action 列，无记录
  不显示；action 11 = 上楼梯）。
- 已知局限（不建模）：上楼梯（action 11）session 台阶面高于 y=0，单平面 GT
  对台阶支撑期必然显示"悬空"——台阶支撑期 GT 热图可为空而触觉面板有载荷，
  正常（GT足底×触觉冲突审计 2026-10-20 任务结构口径）。

只读，不写任何模型输入。输出（results_display/DataTest/D8Test_smpl_gt_cop/）：
每 session 一个 `<sid>_smpl_gt_cop.{gif|mp4}`。环境：touch_gait。

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test8_smpl_gt_cop.py --session S5091
    python results_display/script/d_test8_smpl_gt_cop.py --split test
    python results_display/script/d_test8_smpl_gt_cop.py --session S5091 --stride 2 --max-frames 20
    python results_display/script/d_test8_smpl_gt_cop.py --session S5091 --tactile-input denoised108_v2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/d_test8_matplotlib"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
from loguru import logger as log  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from utils import cli_common, render_common as rc  # noqa: E402
from anysole.data.handoff_pressure import (  # noqa: E402
    HANDOFF_PACKAGES,
    handoff_root,
    resample_handoff,
    video_start_for,
)
from anysole.data.dataset import load_offset_correction  # noqa: E402
from anysole.data.pressure import cop_from_grid  # noqa: E402
from anysole.data.smpl_io import (  # noqa: E402
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.workspace_adapter import (  # noqa: E402
    load_shared_session,
    shared_smpl_path,
)
from anysole.types import JOINT_NAMES  # noqa: E402
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402

OUT_DIR = REPO_ROOT / "results_display" / "DataTest" / "D8Test_smpl_gt_cop"
SPLITS = cli_common.DEFAULT_SPLIT_CSV
FPS = cli_common.DEFAULT_FPS                     # facts 帧轴回退帧率

# SMPL-24 关节下标（JOINT_NAMES 协议顺序）：跟 = 踝，趾 = foot 关节（标准树）。
ANKLE_L, ANKLE_R = JOINT_NAMES.index("left_ankle"), JOINT_NAMES.index("right_ankle")
FOOT_L, FOOT_R = JOINT_NAMES.index("left_foot"), JOINT_NAMES.index("right_foot")

# 随足 2D 网格（S5091 实测标定）：跟→趾 s ∈ [-0.06, 0.20]m，内外侧 lat ±0.07m。
GT_S0, GT_S1 = -0.06, 0.20
GT_LAT = 0.07
GT_COLS, GT_ROWS = 12, 4
# 接触阈值 3cm：鞋底厚度稳定裕度（支撑期足底顶点实测 y≈1.4–2.0cm，2cm 恰卡
# 足底高度致 GT 接触逐帧闪烁——GT足底×触觉冲突审计 2026-10-20 类型 B）。
GT_CONTACT_DZ = 0.03               # 顶点离地 <3cm 计为着地
GT_VMAX = 4.0                      # 每格着地顶点数显示上限
GT_ANKLE_DZ = 0.005                # 只取踝关节以下的足部顶点
# 随足轴退化阈值：‖d_xz‖ < 0.05m ≈ 足俯仰 >50°（平坦足 ‖d_xz‖≈0.10–0.16m，
# 提踵/上楼梯时踝→foot 水平投影退化甚至翻转——GT足底×触觉冲突审计 2026-10-20
# 类型 C）。
AXIS_MIN_NORM = 0.05

# --tactile-input → 交接包编号（与 anysole/data/dataset.py 的切换同语义：
# denoised108=v1 含平滑 / denoised108_v2=v2 无平滑 / denoised108_v3=v3
# 另线落地）。包是否可用按 handoff_pressure.HANDOFF_PACKAGES 注册表判断。
TACTILE_INPUT_PACKAGE = {
    "denoised108": "v1",
    "denoised108_v2": "v2",
    "denoised108_v3": "v3",
}

SENSOR_CLIP = 1023.0               # 触觉面板 raw 值域（D_Test4 口径）
TRAIL_FRAMES = 40                  # CoP 轨迹窗（渲染帧数）

PANEL_W = 460
PANEL_H = 410
GAP = 16
HEADER_H = 94
FOOT_W = 96
FOOT_H = 252
CONTENT_Y = 112
C_L = (180, 210, 255)              # 左脚色
C_R = (255, 190, 160)              # 右脚色
C_GT = (255, 170, 80)              # SMPL GT 面板色
C_COP = (120, 220, 150)            # 触觉 CoP 面板色
BG = (12, 12, 16)
OUTLINE = (48, 48, 56)


def fitted_font(draw: ImageDraw.ImageDraw, text: str, max_width: int,
                size: int, min_size: int = 12):
    for candidate_size in range(size, min_size - 1, -1):
        font = rc.load_font(candidate_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return rc.load_font(min_size)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, *,
                  fill: tuple, size: int, max_width: int,
                  min_size: int = 12) -> None:
    font = fitted_font(draw, text, max_width, size, min_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((draw.im.size[0] - (bbox[2] - bbox[0])) / 2, y), text,
              font=font, fill=fill)


def base_panel(title: str, subtitle: str, color: tuple) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=OUTLINE)
    draw_centered(draw, title, 12, fill=color, size=30, max_width=PANEL_W - 24)
    draw_centered(draw, subtitle, 55, fill=(150, 150, 160), size=18,
                  max_width=PANEL_W - 24, min_size=13)
    return canvas, draw


def split_sessions(split: str) -> list[str]:
    cols = ["train", "val", "test"] if split == "all" else [split]
    sessions: list[str] = []
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for col in cols:
                if (row.get(col) or "").strip():
                    sessions.append(row[col].strip())
    return sorted(set(sessions))


MANIFEST_CSV = REPO_ROOT / "AnysoleWorkspace" / "protocol" / "manifests" \
    / "session_manifest.csv"
_ACTION_BY_SESSION: dict[str, str] | None = None


def load_action_map() -> dict[str, str]:
    """session_id → action（session_manifest.csv；action 11 = 上楼梯）。"""
    global _ACTION_BY_SESSION
    if _ACTION_BY_SESSION is None:
        _ACTION_BY_SESSION = {}
        try:
            with MANIFEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    sid = (row.get("session_id") or "").strip()
                    if sid:
                        _ACTION_BY_SESSION[sid] = (row.get("action") or "").strip()
        except FileNotFoundError:
            log.warning("session_manifest.csv 缺失（{}），标题不显示 action",
                        MANIFEST_CSV)
    return _ACTION_BY_SESSION


def insole_positions() -> tuple[int, int]:
    """双鞋垫粘贴原点（与 D_Test4 相同的 L/R 并排居中布局）。"""
    gap = 24
    pair_w = 2 * FOOT_W + gap
    x0 = (PANEL_W - pair_w) // 2
    return x0, x0 + FOOT_W + gap


def paste_insoles(canvas: Image.Image, draw: ImageDraw.ImageDraw,
                  left_img: np.ndarray, right_img: np.ndarray,
                  left_label: str = "L", right_label: str = "R") -> None:
    x_l, x_r = insole_positions()
    draw.text((x_l + FOOT_W // 2, 86), left_label, font=rc.LABEL_FONT,
              fill=C_L, anchor="mt")
    draw.text((x_r + FOOT_W // 2, 86), right_label, font=rc.LABEL_FONT,
              fill=C_R, anchor="mt")
    canvas.paste(Image.fromarray(left_img), (x_l, CONTENT_Y))
    canvas.paste(Image.fromarray(right_img), (x_r, CONTENT_Y))


def update_foot_axis(d: np.ndarray, last: np.ndarray | None) -> np.ndarray:
    """最近有效随足轴：‖d‖ 退化（<AXIS_MIN_NORM）且已有有效轴则沿用，否则用
    当前 d 更新。返回未归一化方向，foot_heat 内部归一。"""
    if last is not None and np.linalg.norm(d) < AXIS_MIN_NORM:
        return last
    return d


def foot_heat(verts: np.ndarray, ankle: np.ndarray, foot: np.ndarray,
              floor_y: float, side: str = "L", *,
              axis: np.ndarray | None = None,
              contact_dz: float | None = None) -> np.ndarray:
    """单帧单脚的 GT 足底热图（4 行内外侧 × 12 列趾→跟，着地顶点计数）。

    与触觉 48 格同口径：**col0 = 趾、col11 = 跟**（触觉协议 col0=脚尖，
    热图列方向必须一致才能同 rot90 显示）；**row0 = 外侧**。row0=外侧 的
    依据（2026-10-16 实测）：foot_sensor_layout 两脚图纸互为精确 x 镜像
    （x_L+x_R=1.000），足型轮廓趾区凸起（大脚趾=内侧）左脚在 x 大端、右脚
    在 x 小端 → 两脚 row0 均为外侧。随足系：纵轴 s = 踝→趾方向投影
    （跟负趾正），横轴 lat = 垂直方向——左脚 +lat=内侧、右脚 +lat=外侧
    （跨足时 perp 翻转符号），故右脚乘 -1 统一解剖侧。顶点掩码 = 踝关节
    以下；热值 = 离地 <contact_dz 的顶点数。

    axis：随足轴（踝→趾水平方向，未归一亦可）。None = 用当前帧 d（d_test9
    每帧独立调用时的原行为，无跨帧状态）；连续渲染应传 update_foot_axis
    维护的最近有效轴（‖d_xz‖<0.05m ≈ 足俯仰>50° 时踝→foot 水平投影退化/
    翻转，沿用上一有效轴防热图列序错乱——GT足底×触觉冲突审计 2026-10-20
    类型 C）。
    contact_dz：接触阈值（缺省 GT_CONTACT_DZ=3cm，鞋底厚度稳定裕度；
    d_test9 显式传 2cm 保持其 smpl_contact 原口径）。
    """
    if axis is None:
        d = foot[[0, 2]] - ankle[[0, 2]]
    else:
        d = np.asarray(axis, dtype=np.float64).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-9)
    if contact_dz is None:
        contact_dz = GT_CONTACT_DZ
    perp = np.array([-d[1], d[0]])
    rel = verts[:, [0, 2]] - ankle[[0, 2]][None]
    s = rel @ d
    lat = rel @ perp
    if side == "R":  # 右脚的 +lat 指向外侧，取反使 +lat 恒为内侧
        lat = -lat
    below = verts[:, 1] < float(ankle[1]) + GT_ANKLE_DZ
    keep = below & (s >= GT_S0 - 0.01) & (s <= GT_S1 + 0.01) & (np.abs(lat) <= GT_LAT + 0.01)
    contact = verts[:, 1] < floor_y + contact_dz
    # 列序与触觉一致：col0 = s 最大端（趾）
    cols = np.clip(((GT_S1 - s) / (GT_S1 - GT_S0) * GT_COLS).astype(int), 0, GT_COLS - 1)
    rows = np.clip(((lat + GT_LAT) / (2 * GT_LAT) * GT_ROWS).astype(int), 0, GT_ROWS - 1)
    heat = np.zeros((GT_ROWS, GT_COLS), dtype=np.float32)
    np.add.at(heat, (rows[keep & contact], cols[keep & contact]), 1.0)
    return heat


def heat_insoles(heat_l: np.ndarray, heat_r: np.ndarray,
                 vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """GT/触觉热图 → 双鞋垫像素图（rot90 竖直 + inferno + 网格线，同口径）。"""
    left_img = cv2.resize(rc.pressure_to_heatmap(np.rot90(heat_l, k=1), vmax=vmax),
                          (FOOT_W, FOOT_H), interpolation=cv2.INTER_NEAREST)
    right_img = cv2.resize(rc.pressure_to_heatmap(np.rot90(heat_r, k=1), vmax=vmax),
                           (FOOT_W, FOOT_H), interpolation=cv2.INTER_NEAREST)
    return left_img, right_img


def render_gt_heat_panel(heat_l: np.ndarray, heat_r: np.ndarray,
                         counts: tuple[int, int]) -> np.ndarray:
    """左上：SMPL GT 足底热图（与触觉面板同形同 rot90 显示）。"""
    canvas, draw = base_panel(
        "SMPL GT 足底热图",
        f"离地<{int(GT_CONTACT_DZ * 100)}cm 顶点数 · 4 内外(row0=外)×12 趾跟(col0=趾) · clip 0-{int(GT_VMAX)}",
        C_GT)
    left_img, right_img = heat_insoles(heat_l, heat_r, GT_VMAX)
    paste_insoles(canvas, draw, left_img, right_img)
    draw.text((PANEL_W // 2, PANEL_H - 26),
              f"着地顶点 L {counts[0]} / R {counts[1]}",
              font=rc.INFO_FONT, fill=(200, 200, 210), anchor="mb")
    return np.asarray(canvas)


def render_tactile_panel(left48: np.ndarray, right48: np.ndarray,
                         source_label: str) -> np.ndarray:
    """右上：触觉数据热图（source_label = 触觉源，D_Test4 同口径显示）。

    列序修正（2026-10-21 实锤）：实际数据 col0=跟、col11=趾（串口点序与
    layout dots 映射反了——离地过渡帧力在标称跟列、跟触地帧力在标称趾列），
    显示前翻转列使趾端与 GT 面板（趾在下方）对齐。
    """
    canvas, draw = base_panel(
        "触觉数据热图",
        f"{source_label} · 0-{int(SENSOR_CLIP)} · 4×12/脚 · col0=跟（已翻转为趾下跟上）",
        (230, 230, 235))
    left_img, right_img = heat_insoles(left48.reshape(4, 12)[:, ::-1],
                                       right48.reshape(4, 12)[:, ::-1],
                                       SENSOR_CLIP)
    paste_insoles(canvas, draw, left_img, right_img)
    return np.asarray(canvas)


def draw_insoles_grid(draw: ImageDraw.ImageDraw, x0: int) -> None:
    """CoP 面板深色鞋垫网格（4 列 × 12 行，与热图 rot90 布局一致）。"""
    for c in range(4 + 1):
        x = x0 + int(round(c * FOOT_W / 4))
        draw.line([x, CONTENT_Y, x, CONTENT_Y + FOOT_H], fill=(40, 44, 52), width=1)
    for r in range(12 + 1):
        y = CONTENT_Y + int(round(r * FOOT_H / 12))
        draw.line([x0, y, x0 + FOOT_W, y], fill=(40, 44, 52), width=1)


def render_cop_panel(trail_l: list, trail_r: list,
                     force_l: float, force_r: float,
                     title: str, subtitle: str, color: tuple) -> np.ndarray:
    """CoP 面板（GT 热图质心 / 原始压力同公式同显示）：轨迹 + 准星 + 和值。"""
    canvas, draw = base_panel(title, subtitle, color)
    x_l, x_r = insole_positions()
    draw_insoles_grid(draw, x_l)
    draw_insoles_grid(draw, x_r)

    for x0, trail, base_color in ((x_l, trail_l, C_L), (x_r, trail_r, C_R)):
        # cop_y 0=跟 1=趾；rot90 显示跟在顶部 → y = cop_y × FOOT_H
        pts = [(x0 + p[0] * FOOT_W, CONTENT_Y + p[1] * FOOT_H)
               for p in trail]
        for k in range(len(pts) - 1):
            frac = (k + 1) / max(len(pts) - 1, 1)
            line_color = tuple(int(c * (0.20 + 0.75 * frac)) for c in base_color)
            draw.line([pts[k], pts[k + 1]], fill=line_color, width=3)
        last = pts[-1] if pts else None
        if last is not None:
            # 准星
            dim = tuple(int(c * 0.45) for c in base_color)
            draw.line([x0, last[1], x0 + FOOT_W, last[1]], fill=dim, width=1)
            draw.line([last[0], CONTENT_Y, last[0], CONTENT_Y + FOOT_H],
                      fill=dim, width=1)
            # 当前点
            r = 8
            draw.ellipse([last[0] - r, last[1] - r, last[0] + r, last[1] + r],
                         fill=(255, 255, 255), outline=base_color, width=3)

    draw.text((PANEL_W // 2, PANEL_H - 26),
              f"和值 L {force_l:.0f} / R {force_r:.0f}",
              font=rc.INFO_FONT, fill=(200, 200, 210), anchor="mb")
    return np.asarray(canvas)


def render_unavailable(title: str) -> np.ndarray:
    canvas, draw = base_panel(title, "无 SMPL 数据（BVH-only）", C_GT)
    draw.text((PANEL_W // 2, PANEL_H // 2), "SMPL NPZ 缺失\n无法正演足底热图/CoP",
              font=rc.INFO_FONT, fill=(150, 150, 160), anchor="mm")
    return np.asarray(canvas)


def compose_2x2(gt_heat: np.ndarray, tactile: np.ndarray,
                gt_cop: np.ndarray, t_cop: np.ndarray,
                sid: str, fi: int, n_frames: int, valid: bool,
                fps: float = FPS, action: str = "") -> np.ndarray:
    canvas = Image.new("RGB",
                       (2 * PANEL_W + GAP, HEADER_H + 2 * PANEL_H + GAP), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=OUTLINE)
    title = f"{sid} · GT 足底热图/CoP × 触觉热图/CoP"
    if action:  # session_manifest.csv 有记录才显示（action 11 = 上楼梯）
        title += f" · action {action}"
    draw_centered(draw, title,
                  12, fill=(230, 230, 235), size=30,
                  max_width=canvas.width - 48)
    draw_centered(draw,
                  f"facts {fps:.0f}Hz 帧轴 · frame {fi}/{n_frames - 1} · "
                  f"t={fi / fps:.2f}s · {'valid' if valid else 'invalid'}",
                  56, fill=(150, 150, 160), size=18,
                  max_width=canvas.width - 48, min_size=13)
    for pos, panel in ((0, gt_heat), (1, tactile), (2, gt_cop), (3, t_cop)):
        col, row = pos % 2, pos // 2
        canvas.paste(Image.fromarray(panel),
                     (col * (PANEL_W + GAP), HEADER_H + row * (PANEL_H + GAP)))
    return np.asarray(canvas)


def render_one(sid: str, args: argparse.Namespace) -> Path | None:
    shared = load_shared_session(sid)
    n_frames = int(len(shared["frames"]["frame_id"]))
    left48 = np.asarray(shared["pressure"]["left48"], dtype=np.float32)[:n_frames]
    right48 = np.asarray(shared["pressure"]["right48"], dtype=np.float32)[:n_frames]
    valid_flags = (~np.asarray(shared["pressure"]["fake"], dtype=bool))[:n_frames]
    fps = float(shared["meta"].get("fps", FPS))

    # 触觉数据源：raw108 = shared facts 原始（默认）；其余 = 交接包降噪，与
    # anysole/data/dataset.py 的切换同语义（同帧网格最近行，值换处理值；
    # fake/valid 标志仍取 shared facts）。root 必须显式传（handoff_pressure
    # 的默认值在模块定义时绑定，改注册表不会生效）。
    tactile_source = "shared facts raw108"
    if args.tactile_input != "raw108":
        package = TACTILE_INPUT_PACKAGE[args.tactile_input]
        if package not in HANDOFF_PACKAGES:
            raise ValueError(
                f"{sid}: --tactile-input {args.tactile_input} → 交接包 {package}"
                f" 尚未注册（HANDOFF_PACKAGES 无该项，v3 落地后自动可用）")
        left48, right48 = resample_handoff(
            sid, str(shared["meta"]["date"]), str(shared["meta"]["subject"]),
            video_start_for(shared["dir"]), n_frames,
            package=package, root=handoff_root(package))
        tactile_source = f"handoff {args.tactile_input}"
        log.info("{}: 触觉源 = handoff 包 {}（n_frames={}）", sid, package,
                 n_frames)

    if args.frame:
        frame_ids = [int(x) for x in cli_common.split_csv_arg(args.frame)]
        bad = [f for f in frame_ids if not 0 <= f < n_frames]
        if bad:
            raise ValueError(f"{sid}: --frame 越界 {bad}（n_frames={n_frames}）")
    else:
        frame_ids = list(range(0, n_frames, max(args.stride, 1)))
        if args.max_frames:
            frame_ids = frame_ids[: args.max_frames]
    # 0.5s 时间块模式：连续全速块（覆盖 --stride/--max-frames），块间插
    # 分隔帧（--block-secs > 0 时生效）。
    block_ids: dict[int, int] = {}
    if getattr(args, "block_secs", 0.0) > 0 and not args.frame:
        n_block = max(int(round(args.block_secs * fps)), 1)
        ids: list[int] = []
        for b, start in enumerate(range(0, n_frames, n_block)):
            for fi in range(start, min(start + n_block, n_frames)):
                ids.append(fi)
                block_ids[fi] = b
        frame_ids = ids

    # SMPL 网格（只算渲染帧）：GT 足底热图。
    mesh = None
    try:
        smpl_path = shared_smpl_path(shared)
        mocap_t = np.asarray(shared["frames"]["mocap_time_s"], dtype=np.float64)
        mocap_t = mocap_t[:n_frames]
        if args.corrected_offset:
            shift = load_offset_correction().get(sid, 0.0)
            if shift:
                mocap_t = mocap_t - shift
        # 地面 = archive y=0（g0）。SMPL 足部地面审计 2026-10-19 裁定：判
        # 「陷入/贴地」用 y=0；旧 estimate_floor_y（关节 p5）比物理地面高
        # （S5091=3.6cm）→ GT 恒接触、悬空期 CoP 钉趾端（GT足底×触觉冲突
        # 审计 2026-10-20 类型 A）。网格只算渲染帧。
        smpl_full = load_smpl(smpl_path, query_t=mocap_t)
        joints_full = fk_pose6d_np(smpl_full["pose_6d"], smpl_full["trans_m"],
                                   smpl_full["offsets_m"], smpl_full["parents"])
        floor_y = 0.0
        verts = smpl_vertices_from_archive_params(
            smpl_full["poses_aa"][frame_ids],
            smpl_full["model_trans_m"][frame_ids],
            smpl_full["betas"])
        joints_yup = joints_full[frame_ids]
        mesh = {"joints": joints_yup, "verts": verts, "floor": floor_y}
        log.info("{}: SMPL 网格 {} 帧（floor y=0，archive g0 口径）", sid,
                 verts.shape[0])
    except (FileNotFoundError, ValueError) as exc:
        log.warning("{}: SMPL 网格不可用（{}），GT 面板回退占位", sid, exc)

    action = load_action_map().get(sid, "")
    trail_gt_l: list[tuple[float, float]] = []
    trail_gt_r: list[tuple[float, float]] = []
    trail_t_l: list[tuple[float, float]] = []
    trail_t_r: list[tuple[float, float]] = []
    axis_l: np.ndarray | None = None
    axis_r: np.ndarray | None = None
    frames: list[np.ndarray] = []
    for step, fi in enumerate(frame_ids):
        fi = int(fi)
        # 左上：SMPL GT 足底热图；左下：GT CoP = 同一 cop_from_grid 作用于热图网格
        if mesh is not None:
            v = mesh["verts"][step]
            j = mesh["joints"][step]
            # 随足轴退化保护（GT足底×触觉冲突审计 2026-10-20 类型 C）：足俯仰
            # 陡时（提踵/上楼梯）踝→foot 水平投影 ‖d_xz‖ 退化甚至翻转（平坦足
            # ≈0.10–0.16m）；‖d‖<AXIS_MIN_NORM（≈俯仰>50°）且有上一有效轴时
            # 沿用，否则用当前 d 更新——防热图列序错乱、GT CoP 与 T CoP 反相关。
            d_l = j[FOOT_L][[0, 2]] - j[ANKLE_L][[0, 2]]
            d_r = j[FOOT_R][[0, 2]] - j[ANKLE_R][[0, 2]]
            axis_l = update_foot_axis(d_l, axis_l)
            axis_r = update_foot_axis(d_r, axis_r)
            heat_l = foot_heat(v, j[ANKLE_L], j[FOOT_L], mesh["floor"], "L",
                               axis=axis_l)
            heat_r = foot_heat(v, j[ANKLE_R], j[FOOT_R], mesh["floor"], "R",
                               axis=axis_r)
            counts = (int(heat_l.sum()), int(heat_r.sum()))
            gt_panel = render_gt_heat_panel(heat_l, heat_r, counts)
            cop_gt_l = cop_from_grid(heat_l.reshape(-1))[0]
            cop_gt_r = cop_from_grid(heat_r.reshape(-1))[0]
            if heat_l.sum() > 0:
                trail_gt_l.append((float(cop_gt_l[0]), float(cop_gt_l[1])))
            if heat_r.sum() > 0:
                trail_gt_r.append((float(cop_gt_r[0]), float(cop_gt_r[1])))
            gt_cop_panel = render_cop_panel(
                trail_gt_l[-TRAIL_FRAMES:],
                trail_gt_r[-TRAIL_FRAMES:], float(heat_l.sum()),
                float(heat_r.sum()),
                "SMPL GT 的 CoP",
                f"cop_from_grid(GT 热图) · 轨迹窗 {TRAIL_FRAMES} 渲染帧 · 准星=最近有效帧",
                C_GT)
        else:
            gt_panel = render_unavailable("SMPL GT 足底热图")
            gt_cop_panel = render_unavailable("SMPL GT 的 CoP")
        # 右上：触觉热图；右下：触觉 CoP = cop_from_grid(触觉压力)
        tactile = render_tactile_panel(left48[fi], right48[fi], tactile_source)
        cop_t_l = cop_from_grid(left48[fi])[0]
        cop_t_r = cop_from_grid(right48[fi])[0]
        force_l = float(left48[fi].sum())  # cop_t_l/r 仅用于轨迹追加
        force_r = float(right48[fi].sum())
        if force_l > 0:
            trail_t_l.append((float(cop_t_l[0]), float(cop_t_l[1])))
        if force_r > 0:
            trail_t_r.append((float(cop_t_r[0]), float(cop_t_r[1])))
        # 触觉 CoP 面板 y 轴按真实列序（col0=跟）标注：与 GT 面板同轴对比
        # 时符号相反——已知口径问题（2026-10-21 登记），cop 公式暂不改
        #（模型输入一致性优先）。
        t_cop_panel = render_cop_panel(
            trail_t_l[-TRAIL_FRAMES:],
            trail_t_r[-TRAIL_FRAMES:], force_l, force_r,
            "触觉的 CoP",
            f"cop_from_grid({args.tactile_input}) · 轨迹窗 {TRAIL_FRAMES} 渲染帧 · 准星=最近有效帧",
            C_COP)
        frames.append(compose_2x2(gt_panel, tactile, gt_cop_panel, t_cop_panel,
                                  sid, fi, n_frames, bool(valid_flags[fi]),
                                  fps=fps, action=action))
        # 块间分隔帧（黑帧 + 块号）
        if block_ids:
            nxt = frame_ids[step + 1] if step + 1 < len(frame_ids) else None
            if nxt is not None and block_ids.get(nxt, -1) != block_ids.get(fi, -1):
                sep, _ = base_panel(
                    "block %d/%d" % (block_ids[nxt] + 1, max(block_ids.values()) + 1),
                    "%.1fs" % (nxt / fps), (150, 150, 160))
                frames.append(compose_2x2(
                    np.asarray(sep), np.asarray(sep), np.asarray(sep), np.asarray(sep),
                    sid, fi, n_frames, bool(valid_flags[fi]), fps=fps, action=action))
    if not frames:
        raise RuntimeError(f"{sid}: no frames rendered")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.frame:
        paths = []
        for fi, frame_img in zip(frame_ids, frames):
            png = args.out_dir / f"{sid}_smpl_gt_cop_f{fi}.png"
            Image.fromarray(frame_img).save(png)
            paths.append(png)
            log.info("Wrote {}", png)
        return paths[0]
    media = args.out_dir / f"{sid}_smpl_gt_cop.{args.gen}"
    frame_fps = cli_common.viz_fps(fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, media, frame_fps)
    else:
        cli_common.write_mp4(frames, media, frame_fps)
    log.info("Wrote {} ({} frames, {}x{})", media, len(frames),
             frames[0].shape[1], frames[0].shape[0])
    return media


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test8: GT 足底热图/CoP × 触觉热图/CoP（只读 GIF/MP4）")
    p.add_argument("--session", type=str, default="",
                   help="Session ids，逗号分隔（默认空 = splits.csv 全部 session）")
    p.add_argument("--split", type=str, default="all",
                   choices=["train", "val", "test", "all"],
                   help="批量范围（默认 all = train ∪ val ∪ test；--session 优先）")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--gen", choices=("gif", "mp4"), default="gif",
                   help="输出动画格式（默认 gif；可选 mp4）")
    p.add_argument("--stride", type=int, default=4,
                   help="抽帧步长（40Hz 原始帧率；默认 4 控体积）")
    p.add_argument("--max-frames", type=int, default=0, help="帧数上限（0=不限）")
    p.add_argument("--frame", type=str, default="",
                   help="指定帧号（逗号分隔）静态渲染 PNG（<sid>_smpl_gt_cop_f<N>.png），"
                        "优先于 --stride/--max-frames")
    p.add_argument("--tactile-input", type=str, default="raw108",
                   choices=("raw108", "denoised108", "denoised108_v2",
                            "denoised108_v3"),
                   help="触觉数据源（默认 raw108 = shared facts 原始 48 格/脚；"
                        "denoised108 = v1 交接包（含自适应平滑）、denoised108_v2 = "
                        "v2（无平滑）、denoised108_v3 = v3（另线落地后可用），"
                        "与 anysole/data/dataset.py 同语义，fake/valid 仍取 "
                        "shared facts）")
    p.add_argument("--corrected-offset", action="store_true",
                   help="GT 面板动捕采样按修正表平移（B 类 session，"
                        "z_note/data_offset_fixpipe/offset_correction.csv）")
    p.add_argument("--block-secs", type=float, default=0.0,
                   help="连续全速时间块秒数（>0 时覆盖 --stride/--max-frames，"
                        "块间插分隔帧；如 0.5）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions = (cli_common.split_csv_arg(args.session) if args.session
                else split_sessions(args.split))
    if not sessions:
        raise SystemExit("无 session：--session 为空且 splits.csv 无数据")
    log.info("共 {} 个 session（split={}），输出格式={}，输出目录={}",
             len(sessions),
             args.split if not args.session else "-", args.gen.upper(),
             args.out_dir)

    wrote = 0
    failures: list[tuple[str, str]] = []
    for i, sid in enumerate(sessions, 1):
        log.info("[{}/{}] Session {}", i, len(sessions), sid)
        try:
            if render_one(sid, args) is not None:
                wrote += 1
        except Exception as exc:  # noqa: BLE001 批量跑单 session 失败不中断
            log.error("[{}/{}] {} 失败: {}", i, len(sessions), sid, exc)
            failures.append((sid, str(exc)))
    log.info("done: {} 写入 / {} 失败", wrote, len(failures))
    if failures:
        for sid, err in failures:
            log.error("FAILED {}: {}", sid, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
