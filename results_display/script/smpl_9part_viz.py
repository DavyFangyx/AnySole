#!/usr/bin/env python3
"""V8 主模型的 9 部位 SMPL 可视化（2026-11-16）。

主模型（FT，enum fine-tune）的解码头把 SMPL-24 关节分成 9 个部位，每个
部位一个 token（``anysole.types.PART_NAMES`` / ``PART_JOINTS``，模型侧的
query 顺序，不是 SMPL 原始关节序）。本图用真实会话里的一帧 GT SMPL
pose（不是零姿势 canonical mesh，是动作序列中的一个片段），把每个顶点
硬分配给蒙皮权重最大的那个部位；另外用第 10 个 query（根速度 token）
积分出的世界系轨迹，在脚下铺一条该 token 专属颜色的地面路径——一个
身体、一个固定色，和 9 部位 token 的性质一样（每个 token 一个身份色，
不是按速度渐变的热力图）。

单场景一次渲染：身体 mesh 和路径 ribbon mesh 拼成一个 mesh，用同一个
相机正对着这一帧的身体渲染（camera 朝向 = 该帧身体朝向，和其余单体
渲染脚本同一套约定），路径与脚的世界坐标严格对齐。

口径
----
- 会话：S11051（subject S11，动作 05「绕圈跑」，第 1 次），真实 GT SMPL
  pose。
- 顶点→部位：对每个顶点，按 SMPL 蒙皮权重把 24 个关节的权重求和到 9
  个部位，取权重最大的部位（硬分配，姿态无关）。
- root（骨盆单关节）用中性灰，其余 8 个部位两两同色系、左右用明/暗两
  档区分，色相取自项目固定分类色板（蓝/橙/青绿/紫/红）。
- 地面路径：选中帧前 0.18 s 到后 0.08 s 的根轨迹（世界系），铺成贴地
  窄色带，高度钉在局部顶点的 1 百分位（略低于脚底）；单一固定颜色
  （根速度 token 的身份色，琥珀），不是速度渐变色。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from scipy.spatial.transform import Rotation as SciRotation  # noqa: E402

from anysole.data.smpl_io import load_smpl  # noqa: E402
from anysole.types import PART_JOINTS, PART_NAMES  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

import e5_v8_smpl_gain_heatmap as heat  # noqa: E402

OUT_DIR = REPO / "results_display" / "Etest_crossfit_physimu_vgate" / "1"
SID = "S11051"  # subject S11, action 05 = 绕圈跑 (circle run), trial 1
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"

# fixed hue per anatomical family (project categorical slots: blue/orange/
# aqua/violet/red); bilateral parts share a hue and split light/dark; root
# (a single joint, not a visible region of its own) is neutral grey instead
# of a manufactured 9th hue.
FAMILY_HUE = {
    "torso": "#2a78d6", "headneck": "#eb6834", "arm": "#1baf7a",
    "leg": "#4a3aa7", "foot": "#e34948",
}
ROOT_COLOR = "#9c9a92"
# 10th (root-velocity) token: its own fixed identity colour, same status as
# the 9 part colours — not a speed-gradient heatmap.
VEL_COLOR = "#8a5a12"
PATH_HALF_WIDTH_M = 0.045
PATH_ARROW_LEN_M = 0.16
PATH_ARROW_HALF_WIDTH_M = 0.07
PATH_PAST_S = 0.18
PATH_FUTURE_S = 0.08


def _shade(hex_color: str, factor: float) -> str:
    """factor>0 lightens toward white, factor<0 darkens toward black."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    target = 255.0 if factor > 0 else 0.0
    k = abs(factor)
    r = r + (target - r) * k
    g = g + (target - g) * k
    b = b + (target - b) * k
    return "#%02x%02x%02x" % (round(r), round(g), round(b))


def _hex_rgb01(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


PART_COLOR = {
    "root": ROOT_COLOR,
    "torso": FAMILY_HUE["torso"],
    "headneck": FAMILY_HUE["headneck"],
    "l_arm": _shade(FAMILY_HUE["arm"], 0.35),
    "r_arm": _shade(FAMILY_HUE["arm"], -0.22),
    "l_leg": _shade(FAMILY_HUE["leg"], 0.35),
    "r_leg": _shade(FAMILY_HUE["leg"], -0.22),
    "l_foot": _shade(FAMILY_HUE["foot"], 0.35),
    "r_foot": _shade(FAMILY_HUE["foot"], -0.22),
}
PART_LABEL = {
    "root": "root (pelvis)", "torso": "torso", "headneck": "head / neck",
    "l_arm": "left arm", "r_arm": "right arm",
    "l_leg": "left leg", "r_leg": "right leg",
    "l_foot": "left foot", "r_foot": "right foot",
}


def _vertex_part_rgb(weights: np.ndarray) -> np.ndarray:
    """(V,24) skinning weights -> (V,3) RGB via hard per-vertex part vote."""
    part_weight = np.stack(
        [weights[:, list(joints)].sum(axis=1) for joints in PART_JOINTS],
        axis=1)
    part_id = part_weight.argmax(axis=1)
    palette = np.array([_hex_rgb01(PART_COLOR[name]) for name in PART_NAMES])
    return palette[part_id]


def _path_ribbon(ground_xyz: np.ndarray, floor_y: float,
                 color_rgb: np.ndarray) -> tuple:
    """Thin flat ribbon mesh hugging the floor along a local track segment.

    ``ground_xyz`` is (n,3) native SMPL y-up world points (only x/z used,
    y is overridden with ``floor_y``); one solid colour throughout.
    """
    xz = ground_xyz[:, (0, 2)]
    tangent = np.gradient(xz, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left = xz + normal * PATH_HALF_WIDTH_M
    right = xz - normal * PATH_HALF_WIDTH_M
    n = len(xz)
    verts = np.empty((2 * n, 3), dtype=np.float64)
    verts[0::2, (0, 2)] = left
    verts[1::2, (0, 2)] = right
    verts[:, 1] = floor_y
    colors = np.tile(color_rgb, (2 * n, 1))
    faces = []
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.append((a, c, b))
        faces.append((b, c, d))
    arrow_base_left = xz[-1] + normal[-1] * PATH_ARROW_HALF_WIDTH_M
    arrow_base_right = xz[-1] - normal[-1] * PATH_ARROW_HALF_WIDTH_M
    arrow_tip = xz[-1] + tangent[-1] * PATH_ARROW_LEN_M
    arrow_verts = np.empty((3, 3), dtype=np.float64)
    arrow_verts[:, (0, 2)] = np.stack(
        [arrow_base_left, arrow_base_right, arrow_tip])
    arrow_verts[:, 1] = floor_y
    arrow_offset = len(verts)
    verts = np.concatenate([verts, arrow_verts], axis=0)
    colors = np.concatenate([colors, np.tile(color_rgb, (3, 1))], axis=0)
    faces.append((arrow_offset, arrow_offset + 2, arrow_offset + 1))
    return verts, np.asarray(faces, dtype=np.int64), colors


def main() -> int:
    session_dir = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts"
                          / "sessions" / "cam3").glob("*/*/%s" % SID))[0]
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    motion = load_smpl(resolve_uri(meta["source_files"]["smpl"], must_exist=True),
                       query_t=np.asarray(frames["mocap_time_s"]))
    poses = np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(motion["trans_m"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)[:10]
    lengths = {
        "mocap_time_s": len(t), "valid/fake": len(valid),
        "poses_aa": len(poses), "trans_m": len(trans),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError("frame-count mismatch: %s" % lengths)
    n = len(t)

    rest = heat._rest_joints({"betas": betas})
    _, faces, weights = heat._canonical_mesh()
    part_colors = _vertex_part_rgb(weights)

    valid_idx = np.flatnonzero(valid)
    center_frame = int(valid_idx[len(valid_idx) // 2])

    lo = max(0, center_frame - int(round(PATH_PAST_S * 40.0)))
    hi = min(n, center_frame + int(round(PATH_FUTURE_S * 40.0)) + 1)
    window = np.arange(lo, hi)
    window = window[valid[window]]

    body_v = heat.smpl_vertices(poses[center_frame:center_frame + 1],
                                trans[center_frame:center_frame + 1],
                                betas, rest)[0]
    floor_y = float(np.percentile(body_v[:, 1], 0.5)) - 0.01

    ribbon_v, ribbon_f, ribbon_c = _path_ribbon(
        trans[window], floor_y, _hex_rgb01(VEL_COLOR))
    combined_verts = np.concatenate([body_v, ribbon_v], axis=0)
    combined_faces = np.concatenate([faces, ribbon_f + body_v.shape[0]], axis=0)
    combined_colors = np.concatenate([part_colors, ribbon_c], axis=0)
    combined_display = heat.smpl_yup_to_display(combined_verts)

    rot = SciRotation.from_rotvec(poses[center_frame, 0]).as_matrix()
    fwd = rot @ np.array([0.0, 0.0, 1.0])
    disp = np.array([fwd[0], -fwd[2], fwd[1]])
    azim = float(np.degrees(np.arctan2(disp[1], disp[0])))
    elev = 16.0
    size = (1100, 1450)

    image = heat.render_mesh_frame(
        combined_display, combined_faces, vertex_colors=combined_colors,
        joints=heat.smpl_yup_to_display(body_v), ambient=0.9,
        light_intensity=0.75, bg_color=SURFACE, size=size,
        elev=elev, azim=azim)
    image = heat._crop_body(image, pad=40)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8.6, 9.6), facecolor=SURFACE)
    ax.imshow(image)
    ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, facecolor=PART_COLOR[name],
                         edgecolor="none") for name in PART_NAMES]
    handles.append(Line2D([0, 1], [0, 0], color=VEL_COLOR, linewidth=3.5,
                          marker=">", markevery=[1], markersize=8))
    labels = [PART_LABEL[name] for name in PART_NAMES] + [
        "root velocity (directed path)"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9.2,
              frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.1,
              handleheight=1.1, columnspacing=1.2)
    fig.suptitle("SMPL decoder: 9 part tokens + root-velocity token"
                " — %s, t=%.1fs (action 05, circle run)" % (SID, t[center_frame]),
                fontsize=12.5, color=INK, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "Fig_SMPL_9Part.png"
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
