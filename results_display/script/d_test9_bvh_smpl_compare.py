"""D_Test9: BVH × SMPL 脚部对照可视化（2×4 同帧动画，关节定义与趾间隙审计）。

动机（2026-10-15 D_Test3 修复验证发现）：BVH 骨架（Skeleton1/2/3，28 关节）
与 SMPL-24 在同一 session 上存在两类差异——① 关节定义高度偏移（BVH 踝关节
中心离地 ~10cm vs SMPL 踝 ~5cm，常数 ~6cm）；② 根轨迹平移（BVH 保留世界
位移，SMPL npz 根居中）。本实验把两源并排 + 同框对照，供逐帧核对：

    上排（1×4）SMPL 左足侧视 | BVH 左足侧视 | SMPL 右足侧视 | BVH 右足侧视
    下排      BVH×SMPL 同框对照 | SMPL 足底接触网格 | 高度时间线（跨2列，建议项）

- 上排分足分源：SMPL 足侧视画足底顶点云（随足系 s=跟→趾 × h=离地高度，
  与 D_Test8 同矩形口径）+ 踝/趾标记；BVH 足侧视画踝(LeftFoot)/趾(ToeBase)
  两点 + 趾尖 EndSite 落垂线；四面板同轴同刻度，地面线 h=0，阈值线 2cm
  （smpl_contact）与 5cm（bvh_h）——趾间隙低/趾下垂摆动一眼可见；
- 下排对照图：BVH（蓝）× SMPL（橙）同框叠加，各自 hip 对齐——直接看关节
  定义偏移（踝差 ~6cm）与肢体段长度差异；
- 高度时间线（跨 2 列，建议项）：全 session 每帧高度信号——BVH 踝（蓝实）、
  BVH 趾（蓝虚）、SMPL 踝（橙实）、SMPL 足底最低顶点（橙点，渲染帧）——
  加 2/5cm 阈值线与当前帧游标，底部 4 条接触标签带（bvh_h / smpl_contact
  × L/R，红=接触 绿=离地），直接把"bvh_h≈全接触"与帧级证据对上。

数据源（shared facts 契约）：session.json + frames.npz + pressure_48.npz；
BVH 经 shared_bvh_path + load_motion（query_t=mocap_time_s）；SMPL 经
load_session_gt（显示约定）与 smplx 正演（smpl_vertices_from_archive_params）。
接触标签读 seq_dir 下 contact_bvh_h.npy / contact_smpl_contact.npy（缺失则
标签带留空并警告）。只读，不写任何模型输入。环境：touch_gait。

输出（results_display/DataTest/D9Test_bvh_smpl_compare/）：每 session 一个
`<sid>_bvh_smpl.{gif|mp4}`。

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test9_bvh_smpl_compare.py --session S5091
    python results_display/script/d_test9_bvh_smpl_compare.py --session S5091 --max-frames 20
    python results_display/script/d_test9_bvh_smpl_compare.py --split test
    # 静态帧查验（--frame 逗号分隔帧号，出 PNG，优先于 --stride/--max-frames）
    python results_display/script/d_test9_bvh_smpl_compare.py --session S5091 --frame 316,366
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TOOL_DIR = REPO_ROOT / "AnysoleWorkspace" / "tool"
for path in (REPO_ROOT, SCRIPT_DIR, TOOL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/d_test9_matplotlib"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from loguru import logger as log  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from utils import cli_common, render_common as rc  # noqa: E402
from utils.motion_io import load_session_gt  # noqa: E402
from anysole.data.smpl_io import (  # noqa: E402
    estimate_floor_y,
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.workspace_adapter import (  # noqa: E402
    load_shared_session,
    shared_bvh_path,
    shared_smpl_path,
)
from anysole.types import JOINT_NAMES  # noqa: E402
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402
from Baselines.utils.motion_io import load_motion  # noqa: E402
import contact_labels as contact_methods  # noqa: E402

# D_Test8 的足底热图件（同层复用：foot_heat / heat_insoles / 几何常量）
from d_test8_smpl_gt_cop import (  # noqa: E402
    BG,
    C_GT,
    C_L,
    C_R,
    GT_ANKLE_DZ,
    GT_LAT,
    GT_S0,
    GT_S1,
    GT_VMAX,
    OUTLINE,
    PANEL_W,
    fitted_font,
    foot_heat,
    heat_insoles,
    paste_insoles,
    split_sessions,
)

OUT_DIR = REPO_ROOT / "results_display" / "DataTest" / "D9Test_bvh_smpl_compare"
SPLITS = cli_common.DEFAULT_SPLIT_CSV
FPS = cli_common.DEFAULT_FPS

PANEL_H = 430
GAP = 16
HEADER_H = 94
TIMELINE_W = 2 * PANEL_W + GAP     # 建议项高度时间线跨 2 列
C_BVH = (80, 200, 255)             # BVH 蓝
C_GROUND = (90, 96, 110)
# smpl_contact 标签口径 2cm。D_Test8 的 GT_CONTACT_DZ 已放宽至 3cm（鞋底厚度
# 稳定裕度），本脚本阈值线与接触网格保持原 2cm 口径不变（行为不变）。
SMPL_CONTACT_DZ = 0.02
C_THR_SMPL = (120, 220, 150)       # 2cm 阈值线（smpl_contact）
C_THR_BVH = (230, 210, 90)         # 5cm 阈值线（bvh_h）
C_CONTACT = (200, 50, 50)          # 红=接触
C_AIR = (60, 180, 90)              # 绿=离地
C_NODATA = (90, 90, 100)

SIDE_S_MIN, SIDE_S_MAX = -0.12, 0.26   # 侧视 x：跟→趾 (m)
SIDE_H_MAX = 0.45                       # 侧视 y：离地高度 (m)（楼梯/跳跃不削顶）

ANKLE_L, ANKLE_R = JOINT_NAMES.index("left_ankle"), JOINT_NAMES.index("right_ankle")
FOOT_L, FOOT_R = JOINT_NAMES.index("left_foot"), JOINT_NAMES.index("right_foot")


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, *,
                  fill: tuple, size: int, max_width: int,
                  min_size: int = 12) -> None:
    font = fitted_font(draw, text, max_width, size, min_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((draw.im.size[0] - (bbox[2] - bbox[0])) / 2, y), text,
              font=font, fill=fill)


def base_panel(title: str, subtitle: str, color: tuple, w: int = PANEL_W) -> tuple:
    canvas = Image.new("RGB", (w, PANEL_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w - 1, PANEL_H - 1], outline=OUTLINE)
    draw_centered(draw, title, 10, fill=color, size=28, max_width=w - 24)
    draw_centered(draw, subtitle, 52, fill=(150, 150, 160), size=16,
                  max_width=w - 24, min_size=12)
    return canvas, draw


def parents_to_edges(parents) -> list[tuple[int, int]]:
    edges = []
    for child, parent in enumerate(np.asarray(parents).tolist()):
        if parent >= 0:
            edges.append((parent, child))
    return edges


def draw_skeleton(draw: ImageDraw.ImageDraw, joints: np.ndarray,
                  edges: list, color: tuple, w: int, h: int,
                  width: int = 4, joint_r: int = 4) -> None:
    uv = rc.project_joints(joints, w, h - 34, margin=58)
    ground_y = int(h * 0.90)
    draw.ellipse([w * 0.10, ground_y - 12, w * 0.90, ground_y + 12],
                 outline=(50, 54, 62), width=2)
    for i, j in edges:
        if i >= len(uv) or j >= len(uv):
            continue
        draw.line([(int(uv[i, 0]), int(uv[i, 1])),
                   (int(uv[j, 0]), int(uv[j, 1]))], fill=color, width=width)
    for idx, (u, v) in enumerate(uv):
        r = joint_r + (1 if idx == 0 else 0)
        fill = (255, 220, 90) if idx == 0 else color
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill)


def render_overlay_panel(joints_smpl: np.ndarray | None, edges_smpl: list | None,
                         joints_bvh: np.ndarray, edges_bvh: list,
                         frame_idx: int, n_frames: int) -> np.ndarray:
    """对照图：BVH（蓝）与 SMPL（橙）同框叠加，各自 hip 对齐；SMPL 缺失只画 BVH。"""
    canvas, draw = base_panel("BVH × SMPL 同框对照",
                              "各自 pelvis/hips 对齐 · 踝差 ~6cm 应为关节定义偏移",
                              (200, 200, 210))
    draw_skeleton(draw, joints_bvh, edges_bvh, C_BVH, PANEL_W, PANEL_H,
                  width=5, joint_r=5)
    if joints_smpl is not None and edges_smpl is not None:
        draw_skeleton(draw, joints_smpl, edges_smpl, C_GT, PANEL_W, PANEL_H,
                      width=3, joint_r=3)
    else:
        draw.text((PANEL_W // 2, PANEL_H - 40), "SMPL 缺失（BVH-only）",
                  font=rc.INFO_FONT, fill=(150, 150, 160), anchor="mm")
    draw.text((PANEL_W - 10, PANEL_H - 20),
              f"frame {frame_idx}/{n_frames - 1}",
              font=rc.INFO_FONT, fill=(180, 180, 190), anchor="rs")
    return np.asarray(canvas)


# ---- 脚侧视（SMPL 足底顶点云 / BVH 踝趾两点，同轴同刻度）----------------------


def sole_cloud(verts: np.ndarray, ankle: np.ndarray, foot: np.ndarray,
               floor: float) -> tuple[np.ndarray, np.ndarray]:
    """(s, h) 足底顶点云：随足系 s=跟→趾、h=离地高度（与 D8 矩形口径同）。"""
    d = foot[[0, 2]] - ankle[[0, 2]]
    d = d / (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0]])
    rel = verts[:, [0, 2]] - ankle[[0, 2]][None]
    s = rel @ d
    lat = rel @ perp
    sel = ((verts[:, 1] < float(ankle[1]) + GT_ANKLE_DZ)
           & (s >= GT_S0 - 0.01) & (s <= GT_S1 + 0.01)
           & (np.abs(lat) <= GT_LAT + 0.01))
    return s[sel], verts[sel, 1] - floor


def side_axes(w: int, h: int) -> dict:
    """侧视绘图区映射：x=s(m)、y=h(m，上大下小)。"""
    top = 78
    bottom = PANEL_H - 52
    return {
        "x0": 42, "x1": w - 16, "top": top, "bottom": bottom,
        "sx": lambda s: 42 + (s - SIDE_S_MIN) / (SIDE_S_MAX - SIDE_S_MIN) * (w - 16 - 42),
        "sy": lambda hv: bottom - hv / SIDE_H_MAX * (bottom - top),
    }


def draw_side_axes(draw: ImageDraw.ImageDraw, ax: dict) -> None:
    # 地面线
    y0 = ax["sy"](0.0)
    draw.line([ax["x0"], y0, ax["x1"], y0], fill=C_GROUND, width=2)
    # 阈值线 2cm（smpl_contact）与 bvh_h 迟滞带 [4cm, 6cm]
    for thr, color, tag in ((SMPL_CONTACT_DZ, C_THR_SMPL, "2cm smpl_contact"),
                            (0.04, C_THR_BVH, ""), (0.06, C_THR_BVH, "bvh_h 4-6cm")):
        y = ax["sy"](thr)
        draw.line([ax["x0"], y, ax["x1"], y], fill=color, width=1)
        draw.text((ax["x1"], y), tag, font=rc.INFO_FONT, fill=color, anchor="rs")
    for m in (0.0, 0.1, 0.2, 0.3, 0.4):
        draw.text((ax["x0"] - 4, ax["sy"](m)), f"{m:.0f}m",
                  font=rc.INFO_FONT, fill=(140, 144, 154), anchor="rm")
    draw.text((ax["x0"], ax["top"] - 16), "h 离地(m)", font=rc.INFO_FONT,
              fill=(140, 144, 154), anchor="lt")
    draw.text((ax["x1"], ax["bottom"] + 14), "s 跟→趾(m)", font=rc.INFO_FONT,
              fill=(140, 144, 154), anchor="rs")


def draw_foot_marks(draw: ImageDraw.ImageDraw, ax: dict, ankle_s: float,
                    ankle_h: float, toe_s: float, toe_h: float,
                    color: tuple, tag: str,
                    end_s: float | None = None,
                    end_h: float | None = None) -> None:
    """踝/趾根/趾尖 三点标记 + 链线 + 落垂线（SMPL 与 BVH 共用）。"""
    p0 = (ax["sx"](ankle_s), ax["sy"](ankle_h))
    p1 = (ax["sx"](toe_s), ax["sy"](toe_h))
    draw.line([p0, p1], fill=color, width=3)
    # 实心点：踝、趾根；空心圈：趾尖（EndSite / SMPL 网格趾尖顶点）
    for p, r in ((p0, 6), (p1, 5)):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                     fill=(255, 255, 255), outline=color, width=2)
    if end_s is not None and end_h is not None:
        p2 = (ax["sx"](end_s), ax["sy"](end_h))
        draw.line([p1, p2], fill=color, width=2)
        draw.ellipse([p2[0] - 4, p2[1] - 4, p2[0] + 4, p2[1] + 4],
                     outline=color, width=3)
    for p in ((p0,), (p1,)):
        draw.line([p[0][0], p[0][1], p[0][0], ax["sy"](0.0)], fill=color, width=1)
    if end_s is not None and end_h is not None:
        draw.line([p2[0], p2[1], p2[0], ax["sy"](0.0)], fill=color, width=1)
    draw.text((ax["sx"](ankle_s), ax["top"] - 4), tag, font=rc.INFO_FONT,
              fill=color, anchor="mt")


def render_smpl_foot_side(cloud: tuple, ankle: tuple, toe: tuple,
                          color: tuple, side: str,
                          counts: int) -> np.ndarray:
    """SMPL 单足侧视——足底顶点云 + 踝/趾根/趾尖三点标记（随足系）。

    趾尖 = 云中 s 最大（最靠趾端）的足底顶点，与 BVH EndSite 对等。
    """
    canvas, draw = base_panel(f"SMPL {'左' if side == 'L' else '右'}足侧视",
                              f"足底顶点云 · 着地(<2cm) {counts} 顶点",
                              C_GT)
    ax = side_axes(PANEL_W, PANEL_H)
    draw_side_axes(draw, ax)
    s_cloud, h_cloud = cloud
    for x, y in zip(ax["sx"](s_cloud), ax["sy"](h_cloud)):
        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=color)
    end_s = end_h = None
    if len(s_cloud):
        tip = int(np.argmax(s_cloud))
        end_s, end_h = float(s_cloud[tip]), float(h_cloud[tip])
    draw_foot_marks(draw, ax, 0.0, ankle[1], toe[0], toe[1], color, side,
                    end_s=end_s, end_h=end_h)
    return np.asarray(canvas)


def render_bvh_foot_side(f: dict, color: tuple, side: str) -> np.ndarray:
    """BVH 单足侧视——踝(LeftFoot)/趾(ToeBase)两点 + 趾尖 EndSite 落垂线。"""
    canvas, draw = base_panel(f"BVH {'左' if side == 'L' else '右'}足侧视",
                              "三点：踝=*Foot · 趾根=*ToeBase · 趾尖=EndSite",
                              C_BVH)
    ax = side_axes(PANEL_W, PANEL_H)
    draw_side_axes(draw, ax)
    if f is not None:
        draw_foot_marks(draw, ax, 0.0, f["ankle_h"], f["toe_s"], f["toe_h"],
                        color, side, end_s=f.get("end_s"), end_h=f.get("end_h"))
    return np.asarray(canvas)


def render_gt_heat_panel(heat_l: np.ndarray, heat_r: np.ndarray) -> np.ndarray:
    """下排：SMPL 足底接触网格（D_Test8 同口径：col0=趾、row0=外侧）。"""
    canvas, draw = base_panel("SMPL 足底接触网格",
                              f"离地<{int(SMPL_CONTACT_DZ * 100)}cm 顶点数 · 4 内外(row0=外)×12 趾跟(col0=趾) · clip 0-{int(GT_VMAX)}",
                              C_GT)
    left_img, right_img = heat_insoles(heat_l, heat_r, GT_VMAX)
    paste_insoles(canvas, draw, left_img, right_img)
    return np.asarray(canvas)


def render_unavailable(title: str) -> np.ndarray:
    canvas, draw = base_panel(title, "无 SMPL 数据（BVH-only）", C_GT)
    draw.text((PANEL_W // 2, PANEL_H // 2), "SMPL NPZ 缺失",
              font=rc.INFO_FONT, fill=(150, 150, 160), anchor="mm")
    return np.asarray(canvas)


# ---- 高度时间线（建议项，跨 2 列）---------------------------------------------


def render_timeline(series: list[dict], strips: list[dict], frame_idx: int,
                    n_frames: int) -> np.ndarray:
    """全 session 高度信号 + 2/5cm 阈值 + 游标 + 双方法接触标签带。"""
    canvas, draw = base_panel("高度时间线 × 接触标签（建议项）",
                              "摆动足 max(L,R) · 阈值 2cm / bvh_h 4-6cm 带 · 底=接触带",
                              (200, 200, 210), w=TIMELINE_W)
    top, bottom = 78, PANEL_H - 96
    y0, y1 = 22, bottom - 8
    strip_h = 16
    x0, x1 = 46, TIMELINE_W - 16

    def sx(f: int) -> float:
        return x0 + f / max(n_frames - 1, 1) * (x1 - x0)

    def sy(hv: float) -> float:
        return bottom - min(max(hv, 0.0), 0.45) / 0.45 * (bottom - top)

    # 阈值线（bvh_h 为 4-6cm 迟滞带）
    for thr, color, tag in ((SMPL_CONTACT_DZ, C_THR_SMPL, "2cm"),
                            (0.04, C_THR_BVH, ""), (0.06, C_THR_BVH, "4-6cm")):
        y = sy(thr)
        draw.line([x0, y, x1, y], fill=color, width=1)
        draw.text((x0 - 4, y), tag, font=rc.INFO_FONT, fill=color, anchor="rm")
    draw.text((x0, top - 14), "h (m)", font=rc.INFO_FONT,
              fill=(140, 144, 154), anchor="lt")

    # 信号线
    for spec in series:
        vals = spec["values"]
        if vals is None:
            continue
        if spec.get("points"):
            for f in range(0, len(vals)):
                if not np.isfinite(vals[f]):
                    continue
                x, y = sx(f), sy(vals[f])
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5],
                             fill=spec["color"])
            continue
        pts = []
        for f in range(n_frames):
            if not np.isfinite(vals[f]):
                pts.append(None)
            else:
                pts.append((sx(f), sy(vals[f])))
        segs = []
        cur = []
        for p in pts:
            if p is None:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            segs.append(cur)
        for seg in segs:
            for k in range(len(seg) - 1):
                if spec.get("dash"):
                    if k % 2 == 0:
                        draw.line([seg[k], seg[k + 1]], fill=spec["color"], width=1)
                else:
                    draw.line([seg[k], seg[k + 1]], fill=spec["color"], width=2)

    # 图例
    lx = x0
    for spec in series:
        draw.line([lx, y0 - 10, lx + 14, y0 - 10], fill=spec["color"], width=2)
        draw.text((lx + 17, y0 - 14), spec["name"], font=rc.INFO_FONT,
                  fill=(200, 200, 210), anchor="lt")
        lx += 90 + len(spec["name"]) * 9

    # 接触标签带
    for k, strip in enumerate(strips):
        band_y = y1 + k * (strip_h + 4)
        draw.text((x0 - 4, band_y + strip_h // 2), strip["label"],
                  font=rc.INFO_FONT, fill=(200, 200, 210), anchor="rm")
        vals = strip["values"]
        if vals is None:
            draw.rectangle([x0, band_y, x1, band_y + strip_h], fill=(40, 42, 50))
            draw.text((x0 + 6, band_y + 2), "无标签", font=rc.INFO_FONT,
                      fill=C_NODATA, anchor="lt")
            continue
        # 分块绘制（与 d_test3 时间线同口径）
        n_blocks = max(1, min(n_frames, 200))
        span = int(np.ceil(n_frames / n_blocks))
        for b in range(n_blocks):
            lo = b * span
            hi = min(n_frames, lo + span)
            seg = vals[lo:hi]
            color = C_CONTACT if seg.all() else (C_AIR if not seg.any() else (200, 180, 50))
            bx0 = x0 + int(round(b * (x1 - x0) / n_blocks))
            bx1 = x0 + int(round((b + 1) * (x1 - x0) / n_blocks))
            draw.rectangle([bx0, band_y, max(bx1 - 1, bx0), band_y + strip_h],
                           fill=color)

    # 游标
    x = sx(frame_idx)
    draw.line([x, top, x, y1 + 4 * (strip_h + 4) - 4], fill=(235, 235, 235), width=2)
    draw.text((x, top - 22), f"t={frame_idx / FPS:.2f}s", font=rc.INFO_FONT,
              fill=(235, 235, 235), anchor="mt" if x < x1 - 60 else "rt")
    return np.asarray(canvas)


# ---- 主流程 ------------------------------------------------------------------


def bvh_foot_feet(pts: np.ndarray, names: list, floor: float) -> dict:
    """BVH 双脚信号：踝/趾高度与趾的水平距离（display z-up 约定）。"""
    out = {}
    for side in ("l", "r"):
        full = "left" if side == "l" else "right"
        # 同 d_test3 foot_signal 的命名匹配：踝=*foot、趾=*toe（EndSite 排后不取）
        idx_ankle = [i for i, nm in enumerate(names)
                     if nm.lower().startswith(full + "foot")]
        idx_toe = [i for i, nm in enumerate(names)
                   if nm.lower().startswith(full + "toe")]
        if not idx_ankle or not idx_toe:
            log.warning("BVH {} foot joints not found: names={}", side, names)
            out[side] = None
            continue
        ankle = pts[:, idx_ankle[0], :]
        toe = pts[:, idx_toe[0], :]
        # 趾尖 = ToeBase 的 End Site（*ToeBase_EndSite*，无旋转通道，跟随趾根
        # 局部 +5cm 前向量）；足最低点 = 三点 min z（与 contact_labels.foot_signal
        # 同口径——bvh_h 判的就是它）
        end_idx = [i for i, nm in enumerate(names)
                   if nm.lower().startswith(full + "toe") and "endsite" in nm.lower()]
        chain = [idx_ankle[0]] + idx_toe
        low = pts[:, chain, 2].min(axis=1) - floor
        horiz = np.linalg.norm((toe - ankle)[:, :2], axis=1)
        end_s = end_h = None
        if end_idx:
            end = pts[:, end_idx[0], :]
            end_s = np.linalg.norm((end - ankle)[:, :2], axis=1)
            end_h = end[:, 2] - floor
        out[side] = {
            "ankle_h": ankle[:, 2] - floor,
            "toe_h": toe[:, 2] - floor,
            "toe_s": horiz,
            "low_h": low,
            "end_s": end_s,
            "end_h": end_h,
            "ankle_idx": idx_ankle[0],
            "toe_idx": idx_toe[0],
        }
    return out


def load_labels(seq_dir: Path, method: str) -> np.ndarray | None:
    path = seq_dir / f"contact_{method}.npy"
    if not path.is_file():
        log.warning("{}: 无 {} 标签，接触带留空", seq_dir.name, method)
        return None
    raw = np.load(path).astype(np.float32)
    return raw[:, [6, 7]] > 0.5 if raw.ndim == 2 and raw.shape[1] >= 8 else None


def render_one(sid: str, args: argparse.Namespace) -> Path | None:
    shared = load_shared_session(sid)
    seq_dir = Path(shared["dir"])
    n_frames = int(len(shared["frames"]["frame_id"]))
    fps = float(shared["meta"].get("fps", FPS))
    mocap_t = np.asarray(shared["frames"]["mocap_time_s"], dtype=np.float64)[:n_frames]

    # BVH（全身 z-up 显示约定）
    motion_bvh = load_motion(shared_bvh_path(shared), query_t=mocap_t)
    pts_bvh = np.asarray(motion_bvh["joints"], dtype=np.float32)
    if pts_bvh.shape[0] != n_frames:
        raise ValueError(f"{sid}: BVH joints {pts_bvh.shape} != ({n_frames}, J, 3)")
    floor_bvh = float(np.percentile(pts_bvh[:, :, 2], 5))
    edges_bvh = parents_to_edges(motion_bvh["parents"])
    feet = bvh_foot_feet(pts_bvh, motion_bvh["names"], floor_bvh)

    # SMPL（显示约定关节 + 渲染帧网格）。
    # load_session_gt 在 SMPL npz 缺失时静默回退 BVH（28 名）——SMPL-24 下标
    # 会错源（index 7/10 变成头部/前臂），故按 format 门控。
    gt = load_session_gt(seq_dir, n_frames, fps)
    gt_format = str(gt.get("format", "unknown"))
    if gt_format != "smpl":
        log.warning("{}: GT 源为 {}（非 smpl），SMPL 面板/时间线回退占位", sid, gt_format)
        joints_smpl = None
        edges_smpl = None
        floor_smpl = None
    else:
        joints_smpl = np.asarray(gt["joints"], dtype=np.float32)[:n_frames]
        edges_smpl = parents_to_edges(gt["parents"])
        floor_smpl = float(np.percentile(joints_smpl[:, :, 2], 5))

    if args.frame:
        frame_ids = [int(x) for x in cli_common.split_csv_arg(args.frame)]
        bad = [f for f in frame_ids if not 0 <= f < n_frames]
        if bad:
            raise ValueError(f"{sid}: --frame 越界 {bad}（n_frames={n_frames}）")
    else:
        frame_ids = list(range(0, n_frames, max(args.stride, 1)))
        if args.max_frames:
            frame_ids = frame_ids[: args.max_frames]

    mesh = None
    try:
        if gt_format != "smpl":
            raise FileNotFoundError(f"{sid}: GT 非 smpl，跳过 SMPL 网格")
        smpl_path = shared_smpl_path(shared)
        # 地面按全帧关节 p5 估计（与 smpl_contact 标签同口径）；网格只算渲染帧
        smpl_full = load_smpl(smpl_path, query_t=mocap_t)
        joints_full = fk_pose6d_np(smpl_full["pose_6d"], smpl_full["trans_m"],
                                   smpl_full["offsets_m"], smpl_full["parents"])
        floor_yup = float(estimate_floor_y(joints_full))
        verts = smpl_vertices_from_archive_params(
            smpl_full["poses_aa"][frame_ids],
            smpl_full["model_trans_m"][frame_ids],
            smpl_full["betas"])
        joints_yup = joints_full[frame_ids]
        smpl_h = contact_methods._smpl_foot_heights(verts, joints_yup, floor_yup)
        mesh = {"joints": joints_yup, "verts": verts,
                "floor": floor_yup, "smpl_h": smpl_h}
        log.info("{}: SMPL 网格 {} 帧（floor y={:.3f}m，全帧 p5）", sid,
                 verts.shape[0], floor_yup)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("{}: SMPL 网格不可用（{}），SMPL 面板回退占位", sid, exc)

    # 时间线信号（全 session 帧轴）：取 max(L, R) = 摆动足（植立足永远贴地，
    # min 会淹没摆腿间隙——趾间隙正是本实验要看的东西）
    series = [
        {"name": "BVH 踝 max(LR)", "color": C_BVH,
         "values": np.maximum(feet["l"]["ankle_h"], feet["r"]["ankle_h"])
         if feet["l"] and feet["r"] else None},
        {"name": "BVH 足最低 max(LR)", "color": (140, 160, 200), "dash": True,
         "values": np.maximum(feet["l"]["low_h"], feet["r"]["low_h"])
         if feet["l"] and feet["r"] else None},
        {"name": "SMPL 踝 max(LR)", "color": C_GT,
         "values": (np.maximum(joints_smpl[:, ANKLE_L, 2],
                               joints_smpl[:, ANKLE_R, 2]) - floor_smpl)
         if joints_smpl is not None else None},
    ]
    if mesh is not None:
        smpl_floor_disp = np.empty(len(frame_ids), dtype=np.float64)
        for k, fi in enumerate(frame_ids):
            smpl_floor_disp[k] = np.max(mesh["smpl_h"][k])
        full = np.full(n_frames, np.nan, dtype=np.float64)
        full[frame_ids] = smpl_floor_disp
        series.append({"name": "SMPL 足底 max(LR)", "color": (255, 140, 90),
                       "points": True, "values": full})
    else:
        series.append({"name": "SMPL 足底 max(LR)", "color": (255, 140, 90),
                       "points": True, "values": None})

    strips = [
        {"label": "bvh_h L", "values": None},
        {"label": "bvh_h R", "values": None},
        {"label": "smpl L", "values": None},
        {"label": "smpl R", "values": None},
    ]
    for method, a, b in (("bvh_h", 0, 1), ("smpl_contact", 2, 3)):
        lab = load_labels(seq_dir, method)
        if lab is not None and lab.shape[0] >= n_frames:
            strips[a]["values"] = lab[:n_frames, 0]
            strips[b]["values"] = lab[:n_frames, 1]

    frames: list[np.ndarray] = []
    for step, fi in enumerate(frame_ids):
        fi = int(fi)
        # 上排：SMPL 左足 | BVH 左足 | SMPL 右足 | BVH 右足（同轴同刻度侧视）
        # SMPL 侧视按 L/R 配色，BVH 侧视统一 C_BVH 蓝（与对照图/时间线一致）
        row1 = []
        if mesh is not None:
            v = mesh["verts"][step]
            j = mesh["joints"][step]
            for side, (aj, fj), color in (("L", (ANKLE_L, FOOT_L), C_L),
                                          ("R", (ANKLE_R, FOOT_R), C_R)):
                s_cloud, h_cloud = sole_cloud(v, j[aj], j[fj], mesh["floor"])
                ankle = (0.0, float(j[aj, 1] - mesh["floor"]))
                d = j[fj][[0, 2]] - j[aj][[0, 2]]
                toe = (float(np.linalg.norm(d)), float(j[fj, 1] - mesh["floor"]))
                counts = int((h_cloud < SMPL_CONTACT_DZ).sum())
                row1.append(render_smpl_foot_side((s_cloud, h_cloud), ankle,
                                                  toe, color, side, counts))
                key = "l" if side == "L" else "r"
                f = feet[key]
                feet_t = None if f is None else {
                    "ankle_h": float(f["ankle_h"][fi]),
                    "toe_h": float(f["toe_h"][fi]),
                    "toe_s": float(f["toe_s"][fi]),
                    "low_h": float(f["low_h"][fi]),
                    "end_s": None if f["end_s"] is None else float(f["end_s"][fi]),
                    "end_h": None if f["end_h"] is None else float(f["end_h"][fi]),
                }
                row1.append(render_bvh_foot_side(feet_t, C_BVH, side))
        else:
            for title in ("SMPL 左足侧视", "BVH 左足侧视",
                          "SMPL 右足侧视", "BVH 右足侧视"):
                row1.append(render_unavailable(title))
        # 下排：对照 | 接触网格 | 时间线（跨 2 列）
        p_overlay = render_overlay_panel(joints_smpl[fi], edges_smpl,
                                         pts_bvh[fi], edges_bvh, fi, n_frames)
        if mesh is not None:
            v = mesh["verts"][step]
            j = mesh["joints"][step]
            # 每帧独立调用：axis=None 用当前帧 d（无跨帧状态，原行为）；
            # contact_dz 显式 2cm 保持本脚本 smpl_contact 原口径（D8 已放宽 3cm）
            heat_l = foot_heat(v, j[ANKLE_L], j[FOOT_L], mesh["floor"], "L",
                               contact_dz=SMPL_CONTACT_DZ)
            heat_r = foot_heat(v, j[ANKLE_R], j[FOOT_R], mesh["floor"], "R",
                               contact_dz=SMPL_CONTACT_DZ)
            p_heat = render_gt_heat_panel(heat_l, heat_r)
        else:
            p_heat = render_unavailable("SMPL 足底接触网格")
        p_timeline = render_timeline(series, strips, fi, n_frames)

        canvas = Image.new("RGB", (4 * PANEL_W + 3 * GAP,
                                   HEADER_H + 2 * PANEL_H + GAP), BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=OUTLINE)
        draw_centered(draw, f"{sid} · BVH × SMPL 脚部对照", 12,
                      fill=(230, 230, 235), size=30,
                      max_width=canvas.width - 48)
        draw_centered(draw,
                      f"frame {fi}/{n_frames - 1} · t={fi / fps:.2f}s · "
                      f"BVH floor={floor_bvh:.3f}m · SMPL floor={floor_smpl:.3f}m",
                      56, fill=(150, 150, 160), size=18,
                      max_width=canvas.width - 48, min_size=13)
        for col, panel in enumerate(row1):
            canvas.paste(Image.fromarray(panel),
                         (col * (PANEL_W + GAP), HEADER_H))
        # 下排：对照 | 接触网格 | 时间线（跨 2 列）
        y2 = HEADER_H + PANEL_H + GAP
        canvas.paste(Image.fromarray(p_overlay), (0, y2))
        canvas.paste(Image.fromarray(p_heat), (PANEL_W + GAP, y2))
        canvas.paste(Image.fromarray(p_timeline), (2 * (PANEL_W + GAP), y2))
        frames.append(np.asarray(canvas))
    if not frames:
        raise RuntimeError(f"{sid}: no frames rendered")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.frame:
        paths = []
        for fi, frame_img in zip(frame_ids, frames):
            png = args.out_dir / f"{sid}_bvh_smpl_f{fi}.png"
            Image.fromarray(frame_img).save(png)
            paths.append(png)
            log.info("Wrote {}", png)
        return paths[0]
    media = args.out_dir / f"{sid}_bvh_smpl.{args.gen}"
    frame_fps = cli_common.viz_fps(fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, media, frame_fps)
    else:
        cli_common.write_mp4(frames, media, frame_fps)
    log.info("Wrote {} ({} frames, {}x{})", media, len(frames),
             frames[0].shape[1], frames[0].shape[0])
    return media


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test9: BVH × SMPL 脚部对照（只读 GIF/MP4）")
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
                   help="指定帧号（逗号分隔）静态渲染 PNG（<sid>_bvh_smpl_f<N>.png），"
                        "优先于 --stride/--max-frames")
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
