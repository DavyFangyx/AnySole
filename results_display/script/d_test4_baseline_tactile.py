"""D_Test4: AnySole + 三基线触觉适配审计（只读，不生成任何产物）。

审计只读前端：不再调用任何生成器、不落盘中间转换产物，直接从现行数据新家读取
四个面板，先做源一致性审计，再渲染 1x4 横向逐帧动画。四个面板为：

    AnySole          shared/facts/sessions/cam3/<date>/<sub>/<sid>/pressure_48.npz
                     raw 4×12 / 脚（48 格/脚，0-1023 原始传感器单位）
    MotionPRO        model_inputs/MotionPRO/adapter_v1/cam3/<date>/<sub>/<sid>/pressure.npz
                     虚拟触觉毯（形状运行时读盘，并与 artifact.json raster.shape 对账）
    MMVP             shared/representations/tactile/mmvp_31x11/v1/<date>/<sub>/<sid>/
                     insole/%06d.npy（每帧 (2,31,11)）+ frame_id.npy
                     公共 31×11 / 脚 表示层（pressure_toolkit 与 FPP-Net 共用同一份）
    Step2Motion      无落盘 16ch 产物：按 AnysoleWorkspace/tool/adapters/Step2Motion/
                     build_gait.py 的 pool_48_to_16（冻结池化）运行时复算

每 session 审计（任一不过即抛错，不静默降级；旧生成器已删除，历史产物不再可信）：
    facts ↔ MMVP      artifact.json parameters.mapping ==
                      audited_4x12_to_31x11_nearest_cell；source_hashes.pressure_48
                      与 facts 实文件哈希一致；frame_id.npy 与 facts frame_id 逐值相等；
                      insole 帧数与 facts 帧数一致（parity 审计）
    facts ↔ MotionPRO artifact.json raster.shape 与 npz 实际形状一致；
                      source_hashes.pressure_48 一致；frame_id.npy 与 facts frame_id 相等

显示口径（与已删除的 generate_baseline_tactile.py 落盘契约兼容，纯只读复现）：
    - 4×12 / 脚 = row-major（col0 = 脚尖；assets/foot_sensor_layout/README.md 已数值
      验证），AnySole/Step2Motion 面板 rot90 竖直显示；
    - AnySole / MMVP / Step2Motion 三块共用同一值域裁剪 clip(v, 0, 1023) / 1023，等价于
      旧的冻结双足光栅 clip(raw, 0, 1023) / 1023 * 255 再 pressure_to_heatmap(vmax=255)
      （MotionPRO adapter 的 raster.value 约定），故显示与旧版逐像素一致；
    - Step2Motion 的 16 通道按 pool_ids 回填 4x12，仅作展示（每组 3 格同值）。

输出（results_display/DataTest/D4Test_baseline_tactile/）：每 session 仅一个
`<sid>_adapted_tactile.{gif|mp4}`（四面板 1x4 横向同帧对齐动画），不写其他文件。

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test4_baseline_tactile.py
        # 默认：splits.csv 全部 session（train ∪ val ∪ test），逐 session 审计+渲染
    python results_display/script/d_test4_baseline_tactile.py --split test
    python results_display/script/d_test4_baseline_tactile.py --session S5091
    python results_display/script/d_test4_baseline_tactile.py --session S5091 --stride 8 --max-frames 10
    python results_display/script/d_test4_baseline_tactile.py --session S5091 --gen mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep Matplotlib's cache warning out of the normal CLI output on this managed
# workspace, where the default user config directory is read-only.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/d_test4_matplotlib"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
from loguru import logger as log  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from AnysoleWorkspace.tool.artifacts import sha256_file  # noqa: E402
from utils import cli_common, render_common as rc  # noqa: E402

try:  # 复用 Step2Motion adapter 的冻结池化（build_gait.py:194 pool_48_to_16）
    from AnysoleWorkspace.tool.adapters.Step2Motion.build_gait import pool_48_to_16  # noqa: E402
except ImportError as exc:  # pragma: no cover - 仅供缺依赖环境兜底
    log.warning("build_gait.pool_48_to_16 导入失败（{}），改用同口径本地副本", exc)

    def pool_48_to_16(values48: np.ndarray) -> np.ndarray:
        """本地副本，与 build_gait.py:194-204 逐行同数学（来源行号见该模块）。"""
        grid = np.asarray(values48).reshape(-1, 4, 12)
        toe = grid[:, :, :6].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
        heel = grid[:, :, 6:].reshape(-1, 4, 2, 3).mean(axis=-1).reshape(-1, 8)
        return np.concatenate([heel, toe], axis=1).astype(np.float32)

# ---------------------------------------------------------------- 数据新家（只读）

WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
FACTS_ROOT = WORKSPACE / "shared" / "facts" / "sessions" / "cam3"
MMVP_ROOT = WORKSPACE / "shared" / "representations" / "tactile" / "mmvp_31x11" / "v1"
MMVP_MAPPING = "audited_4x12_to_31x11_nearest_cell"
MOTIONPRO_ROOT = WORKSPACE / "model_inputs" / "MotionPRO" / "adapter_v1" / "cam3"
SPLITS = WORKSPACE / "protocol" / "splits" / "default" / "splits.csv"
DEFAULT_OUT = REPO_ROOT / "results_display" / "DataTest" / "D4Test_baseline_tactile"

SENSOR_CLIP = 1023.0   # facts/MMVP 原始单位上限（冻结光栅口径 clip(raw,0,1023)/1023*255）
CARPET_VMAX = 255.0    # 虚拟毯值域 0-255（adapter artifact raster.value 约定）
DEFAULT_FPS = 40.0

# 面板基线标注色（工作名标题同色，仅作身份标识）
C_ANYSOLE = (120, 220, 150)
C_MOTIONPRO = (255, 170, 80)
C_TOOLKIT = (230, 120, 230)
C_S2M = (80, 200, 255)

PANEL_W = 360          # 单面板宽：1x4 横向对比
PANEL_H = 390          # 单面板高：标题区 + 标签区 + 252 高内容区
FOOT_W = 96            # 四个面板统一的单脚显示宽度
FOOT_H = 252           # 四个面板统一的单脚显示长度
CONTENT_Y = 112        # 内容区顶；与标题/左右脚标签完全分离
GAP = 20               # 面板间距
FOOT_GAP = 24          # 同一面板内左右脚间距
CARPET_MARGIN = 16     # 虚拟毯面板内容区留白
HEADER_H = 94          # 主标题区：标题和帧信息各占一行
DOWNSCALE = 0.75       # 输出缩放
FPS = DEFAULT_FPS      # 仅在 session 元数据缺 fps 时作回退

BG = (12, 12, 16)
OUTLINE = (48, 48, 56)


def fitted_font(draw: ImageDraw.ImageDraw, text: str, max_width: int,
                size: int, min_size: int = 12):
    """返回不超过 max_width 的字体，避免长标题侵入相邻区域。"""
    for candidate_size in range(size, min_size - 1, -1):
        font = rc.load_font(candidate_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return rc.load_font(min_size)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, *,
                  fill: tuple, size: int, max_width: int,
                  min_size: int = 12) -> None:
    """在指定的独立文字行内居中绘制，y 是文字顶部。"""
    font = fitted_font(draw, text, max_width, size, min_size)
    draw.text((draw.im.size[0] // 2, y), text, font=font, fill=fill, anchor="mt")


# ---------------------------------------------------------------- 审计读取层

def split_sessions(split: str) -> list[str]:
    """splits.csv 某一列（train/val/test）的 session 列表；all = 三列并集。"""
    cols = ["train", "val", "test"] if split == "all" else [split]
    sessions: list[str] = []
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for col in cols:
                if (row.get(col) or "").strip():
                    sessions.append(row[col].strip())
    return sorted(set(sessions))


def load_facts(session_id: str) -> dict:
    """读 shared facts（唯一公共触觉事实源）并校验帧轴。

    返回 dict：dir/date/subject/fps/n_frames/frame_id/left48/right48/valid/fake/sha256。
    """
    matches = sorted(FACTS_ROOT.glob(f"*/*/{session_id}/session.json"))
    if not matches:
        raise FileNotFoundError(f"{session_id}: shared facts 里没有该 session（{FACTS_ROOT}）")
    if len(matches) > 1:
        raise ValueError(f"{session_id}: facts 身份不唯一：{matches}")
    root = matches[0].parent
    meta = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if str(meta.get("session_id")) != session_id:
        raise ValueError(f"{root}: session.json 身份不符（{meta.get('session_id')!r}）")
    pressure = np.load(root / "pressure_48.npz", allow_pickle=True)
    frames = np.load(root / "frames.npz", allow_pickle=True)
    frame_id = np.asarray(pressure["frame_id"])
    if not np.array_equal(frame_id, np.asarray(frames["frame_id"])):
        raise ValueError(f"{root}: pressure_48.npz 与 frames.npz 的 frame_id 不一致")
    n = int(len(frame_id))
    for key in ("left48", "right48"):
        shape = np.asarray(pressure[key]).shape
        if shape != (n, 48):
            raise ValueError(f"{root}: {key} 形状 {shape} != ({n}, 48)")
    return {
        "session": session_id,
        "dir": root,
        "date": root.parts[-3],
        "subject": root.parts[-2],
        "fps": float(meta.get("fps", DEFAULT_FPS)),
        "n_frames": n,
        "frame_id": frame_id,
        "left48": np.asarray(pressure["left48"], dtype=np.float32),
        "right48": np.asarray(pressure["right48"], dtype=np.float32),
        "valid": np.asarray(pressure["valid"], dtype=np.uint8),
        "fake": np.asarray(pressure["fake"], dtype=np.uint8),
        "sha256": sha256_file(root / "pressure_48.npz"),
    }


def audit_motionpro(facts: dict) -> dict:
    """MotionPRO 虚拟毯：读盘 + 与 facts 对账（形状 / 源哈希 / frame_id）。"""
    root = MOTIONPRO_ROOT / facts["date"] / facts["subject"] / facts["session"]
    npz_path, art_path = root / "pressure.npz", root / "artifact.json"
    if not npz_path.is_file() or not art_path.is_file():
        raise FileNotFoundError(f"{facts['session']}: MotionPRO adapter_v1 产物缺失：{root}")
    artifact = json.loads(art_path.read_text(encoding="utf-8"))
    declared = tuple(artifact.get("parameters", {}).get("raster", {}).get("shape", ()))
    # 毯为 savez_compressed 产物：mmap 请求会按 numpy 既定行为退化为整段读入
    carpet = np.load(npz_path, mmap_mode="r")["pressure"]
    if tuple(carpet.shape[1:]) != declared:
        raise ValueError(f"{facts['session']}: MotionPRO 毯形状 {tuple(carpet.shape[1:])} "
                         f"!= artifact raster.shape {declared}")
    if carpet.shape[0] != facts["n_frames"]:
        raise ValueError(f"{facts['session']}: MotionPRO 帧数 {carpet.shape[0]} "
                         f"!= facts {facts['n_frames']}")
    source_hash = artifact.get("source_hashes", {}).get("pressure_48")
    if source_hash != facts["sha256"]:
        raise ValueError(f"{facts['session']}: MotionPRO 由另一版 pressure_48 生成"
                         f"（artifact hash {source_hash} != facts {facts['sha256']}），需重跑 adapter")
    frame_id_path = root / "frame_id.npy"
    if frame_id_path.is_file() and not np.array_equal(np.load(frame_id_path), facts["frame_id"]):
        raise ValueError(f"{facts['session']}: MotionPRO frame_id 与 facts 不一致")
    return {"carpet": carpet, "shape": tuple(carpet.shape[1:]), "root": root}


def audit_mmvp(facts: dict) -> dict:
    """MMVP 公共表示层：artifact 审计（mapping/哈希）+ frame_id parity；逐帧懒加载。"""
    root = MMVP_ROOT / facts["date"] / facts["subject"] / facts["session"]
    art_path = root / "artifact.json"
    if not art_path.is_file():
        raise FileNotFoundError(f"{facts['session']}: MMVP 表示层 artifact 缺失：{art_path}")
    artifact = json.loads(art_path.read_text(encoding="utf-8"))
    mapping = artifact.get("parameters", {}).get("mapping")
    if mapping != MMVP_MAPPING:
        raise ValueError(f"{facts['session']}: MMVP mapping {mapping!r} != {MMVP_MAPPING!r}"
                         "（表示层不是审计映射）")
    source_hash = artifact.get("source_hashes", {}).get("pressure_48")
    if source_hash != facts["sha256"]:
        raise ValueError(f"{facts['session']}: MMVP 表示层未跟上 facts"
                         f"（artifact hash {source_hash} != facts {facts['sha256']}），"
                         "先跑 build_shared.py --mmvp-only")
    frame_id_path = root / "frame_id.npy"
    if not frame_id_path.is_file():
        raise FileNotFoundError(f"{facts['session']}: MMVP frame_id.npy 缺失：{frame_id_path}")
    if not np.array_equal(np.load(frame_id_path), facts["frame_id"]):
        raise ValueError(f"{facts['session']}: MMVP frame_id 与 facts 不一致（parity 审计失败）")
    insole = root / "insole"
    n_files = sum(1 for path in insole.glob("*.npy"))
    if n_files != facts["n_frames"]:
        raise ValueError(f"{facts['session']}: MMVP insole 帧数 {n_files} "
                         f"!= facts {facts['n_frames']}")
    return {"dir": insole, "root": root, "artifact": artifact}


def mmvp_frame(mmvp: dict, fi: int) -> np.ndarray:
    """读 MMVP 表示层一帧 → (2,31,11)（[L,R]，raw 单位，与 facts 逐格相等）。"""
    payload = np.load(mmvp["dir"] / f"{fi:06d}.npy")
    if payload.shape != (2, 31, 11):
        raise ValueError(f"MMVP frame {fi}: 形状 {payload.shape} != (2,31,11)")
    return payload


# ---------------------------------------------------------------- 面板渲染

def base_panel(title: str, subtitle: str, color: tuple) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """统一面板底板：标题、副标题、标签和内容各占独立的垂直区域。"""
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=OUTLINE)
    draw_centered(draw, title, 12, fill=color, size=30, max_width=PANEL_W - 24)
    draw_centered(draw, subtitle, 55, fill=(150, 150, 160), size=18,
                  max_width=PANEL_W - 24, min_size=13)
    return canvas, draw


def render_foot_blocks(left_block: np.ndarray, right_block: np.ndarray, title: str,
                       subtitle: str, color: tuple, vmax: float = SENSOR_CLIP) -> np.ndarray:
    """双足热图面板：L/R 两鞋垫并排居中，各带标签（标题/副题统一走 base_panel）。

    vmax 为面板值域上限：AnySole / MMVP / Step2Motion 三块都是 raw 单位
    （0-1023），沿用冻结光栅口径，与旧版 0-255 画布显示逐像素等价。
    """
    canvas, draw = base_panel(title, subtitle, color)
    left_img = cv2.resize(rc.pressure_to_heatmap(left_block, vmax=vmax), (FOOT_W, FOOT_H),
                          interpolation=cv2.INTER_NEAREST)
    right_img = cv2.resize(rc.pressure_to_heatmap(right_block, vmax=vmax), (FOOT_W, FOOT_H),
                           interpolation=cv2.INTER_NEAREST)
    gap = FOOT_GAP
    pair_w = left_img.shape[1] + gap + right_img.shape[1]
    x0 = (PANEL_W - pair_w) // 2
    draw.text((x0 + left_img.shape[1] // 2, 86), "L", font=rc.LABEL_FONT,
              fill=(180, 210, 255), anchor="mt")
    draw.text((x0 + left_img.shape[1] + gap + right_img.shape[1] // 2, 86), "R",
              font=rc.LABEL_FONT, fill=(255, 190, 160), anchor="mt")
    canvas.paste(Image.fromarray(left_img), (x0, CONTENT_Y))
    canvas.paste(Image.fromarray(right_img), (x0 + left_img.shape[1] + gap, CONTENT_Y))
    return np.asarray(canvas)


def render_carpet_panel(carpet: np.ndarray, title: str, subtitle: str, color: tuple) -> np.ndarray:
    """MotionPRO 虚拟毯面板：整毯热图（4.0m 长轴水平，纯旋转保几何），非双脚块。"""
    canvas, draw = base_panel(title, subtitle, color)
    image = np.asarray(carpet, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"virtual carpet frame must be 2-D, got {image.shape}")
    cells = np.clip(image, 0.0, CARPET_VMAX) / CARPET_VMAX
    rgb = (cm.inferno(cells)[..., :3] * 255.0).astype(np.uint8)
    rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
    avail_w = PANEL_W - 2 * CARPET_MARGIN
    avail_h = PANEL_H - CONTENT_Y - CARPET_MARGIN
    scale = min(avail_w / rgb.shape[1], avail_h / rgb.shape[0])
    width = max(1, int(round(rgb.shape[1] * scale)))
    height = max(1, int(round(rgb.shape[0] * scale)))
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x0 = (PANEL_W - width) // 2
    y0 = CONTENT_Y + (avail_h - height) // 2
    canvas.paste(Image.fromarray(resized), (x0, y0))
    # 毯上无有效着墨时（out-of-bounds 帧）帮助区分“全黑”与“无数据”
    draw.text((PANEL_W // 2, PANEL_H - 26), f"painted {int(np.count_nonzero(image))} px",
              font=rc.INFO_FONT, fill=(150, 150, 160), anchor="mt")
    return np.asarray(canvas)


def compose_1x4(panels: list[np.ndarray], header: str, subheader: str) -> Image.Image:
    """四面板 1x4 横向排布，顶部双行标题与面板完全分离。"""
    if len(panels) != 4:
        raise ValueError(f"1x4 layout expects four panels, got {len(panels)}")
    canvas = Image.new("RGB", (4 * PANEL_W + 3 * GAP, HEADER_H + PANEL_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=OUTLINE)
    draw_centered(draw, header, 12, fill=(230, 230, 235), size=30,
                  max_width=canvas.width - 48)
    draw_centered(draw, subheader, 56, fill=(150, 150, 160), size=18,
                  max_width=canvas.width - 48, min_size=13)
    for column, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (column * (PANEL_W + GAP), HEADER_H))
    if DOWNSCALE < 1.0:
        canvas = canvas.resize((int(canvas.width * DOWNSCALE), int(canvas.height * DOWNSCALE)),
                               Image.BILINEAR)
    return canvas


def pool_ids() -> np.ndarray:
    """(4,12) 每格对应的 16 通道池化组编号（仅用于 Step2Motion 展示回填）。

    与 build_gait.pool_48_to_16 的分组逐格一致：heel 列 6-11 → 通道 0-7，
    toe 列 0-5 → 通道 8-15，每 3 列一组。
    """
    ids = np.zeros((4, 12), dtype=int)
    for r in range(4):
        for g in range(2):
            ch = r * 2 + g
            ids[r, 6 + 3 * g:6 + 3 * g + 3] = ch
            ids[r, 3 * g:3 * g + 3] = ch + 8
    return ids


# ---------------------------------------------------------------- CLI / main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test4: AnySole + 三基线触觉审计（只读 GIF/MP4）")
    p.add_argument("--session", type=str, default="",
                   help="单 session（默认空 = splits.csv 全部 session）")
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                   help="批量范围（默认 all = train ∪ val ∪ test；--session 优先）")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gen", choices=("gif", "mp4"), default="gif",
                   help="输出动画格式（默认 gif；可选 mp4）")
    p.add_argument("--stride", type=int, default=2, help="抽帧步长（40Hz 原始帧率）")
    p.add_argument("--max-frames", type=int, default=0, help="帧数上限（0=不限）")
    return p.parse_args()


def render_one(sid: str, args: argparse.Namespace) -> None:
    """单 session：审计四源 → 读盘渲染一个 GIF/MP4（只读，不写中间产物）。"""
    facts = load_facts(sid)
    motionpro = audit_motionpro(facts)
    mmvp = audit_mmvp(facts)
    n = facts["n_frames"]
    log.info("{}: {} 帧（valid {} / fake {}），四源审计通过（MMVP mapping={}，毯 {}）",
             sid, n, int(facts["valid"].sum()), int(facts["fake"].sum()),
             MMVP_MAPPING, "×".join(str(v) for v in motionpro["shape"]))

    valid = np.flatnonzero(~facts["fake"].astype(bool))
    if len(valid) == 0:  # 全 fake session：无有效帧可抽，GIF 回退全帧展示
        log.warning("{}: 无有效帧（all_fake），GIF 回退全帧", sid)
        valid = np.arange(n, dtype=int)
    frame_ids = valid[:: args.stride]
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    log.info("{} 帧数 {}", args.gen.upper(), len(frame_ids))

    fps = facts["fps"]
    pids = pool_ids()
    frames: list[np.ndarray] = []
    for fi in frame_ids:
        fi = int(fi)
        # AnySole 主方法：raw 4x12 原生格（row-major，col0=脚尖），竖直脚底显示。
        p_anysole = render_foot_blocks(
            np.rot90(facts["left48"][fi].reshape(4, 12), k=1),
            np.rot90(facts["right48"][fi].reshape(4, 12), k=1),
            "AnySole", "raw 4×12/脚 · 48 格", C_ANYSOLE)
        # MotionPRO：虚拟毯整幅（形状读盘，artifact 对账）
        p_motion = render_carpet_panel(
            motionpro["carpet"][fi], "MotionPRO",
            f"虚拟毯 {motionpro['shape'][0]}×{motionpro['shape'][1]} · 4.0×1.5m", C_MOTIONPRO)
        # MMVP：公共 31x11 表示层（唯一一份，两消费者共用）
        mmvp_pair = mmvp_frame(mmvp, fi)
        p_mmvp = render_foot_blocks(
            mmvp_pair[0], mmvp_pair[1],
            "MMVP", "公共 31×11 表示层 · nearest-cell", C_TOOLKIT)
        # Step2Motion：无落盘 16ch 产物，按 freezer 池化同口径复算后回填 4x12 展示。
        left16 = pool_48_to_16(facts["left48"][fi:fi + 1])[0]
        right16 = pool_48_to_16(facts["right48"][fi:fi + 1])[0]
        p_s2m = render_foot_blocks(
            np.rot90(left16[pids], k=1), np.rot90(right16[pids], k=1),
            "Step2Motion", "pool_48_to_16 复算 · 16 通道/脚", C_S2M)

        header = f"{sid} · 触觉四源审计（AnySole + 三基线）"
        subheader = (f"源 shared facts pressure_48 · frame {fi}/{n - 1} · "
                     f"t={fi / fps:.2f}s · {fps:.0f}Hz")
        frame_img = compose_1x4([p_anysole, p_motion, p_mmvp, p_s2m], header, subheader)
        frames.append(np.asarray(frame_img))
    log.info("合成 {} {} 帧，每帧 {}x{}", args.gen.upper(), len(frames),
             frames[0].shape[1], frames[0].shape[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    media = args.out_dir / f"{sid}_adapted_tactile.{args.gen}"
    frame_fps = cli_common.viz_fps(fps, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, media, frame_fps)
    else:
        cli_common.write_mp4(frames, media, frame_fps)
    log.info("Wrote {}", media)


def main() -> None:
    args = parse_args()
    sessions = [args.session] if args.session else split_sessions(args.split)
    if not sessions:
        raise SystemExit("无 session：--session 为空且 splits.csv 无数据")
    log.info("共 {} 个 session（split={}），输出格式={}，输出目录={}",
             len(sessions), args.split if not args.session else "-", args.gen.upper(), args.out_dir)

    failures: list[tuple[str, str]] = []
    for i, sid in enumerate(sessions, 1):
        log.info("[{}/{}] Session {}", i, len(sessions), sid)
        try:
            render_one(sid, args)
        except Exception as exc:  # noqa: BLE001 批量跑单 session 失败不中断
            log.error("[{}/{}] {} 失败: {}", i, len(sessions), sid, exc)
            failures.append((sid, str(exc)))
    log.info("done: {} 成功 / {} 失败", len(sessions) - len(failures), len(failures))
    if failures:
        for sid, err in failures:
            log.error("FAILED {}: {}", sid, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
