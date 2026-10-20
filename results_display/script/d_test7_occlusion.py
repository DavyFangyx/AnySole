"""D_Test7: cam3 bbox 底部遮挡管线可视化（四相机扩展 §10 遮挡组）。

对遮挡实验的像素级处理管线做逐位相同的可视化反映：

    raw JPEG ──occlude_lower_strip──► 遮挡帧 ──process_image──► 256×192 crop ──► HRNet
        │                                  │
        └── bbox → center/scale ───────────┴──► bbox_info 3-d（数值入副标题）

遮挡复算与裁剪调用与训练提取完全同一实现、同一参数、同一顺序
（``anysole.data.extract_hrnet`` 的 ``occlude_lower_strip`` +
``anysole.data.crop.process_image``，crop 尺寸 CROP_IMG_HEIGHT/WIDTH），
crop 面板直接显示 ``process_image`` 返回的 ``crop_img``（HRNet 的输入），
故所见即训练时进入编码器的像素。左面板原始帧叠加 bbox 绿框、crop 区域
黄框与切割线红线，核对遮挡锚定与档位语义（010=足部、020=足+踝、
030=足+踝+小腿、050=下肢全失）。

只读，不写任何模型输入；遮挡/crop 纯复算，不读 hrnet 缓存特征——缓存存在性
仅作"该 session 参与了 occ 实验"的闸门（无缓存即跳过）。环境：touch_gait。

输出（results_display/DataTest/D7Test_occlusion/）：每 session 每 occ 档一个
`<sid>_occ<FFF>.{gif|mp4}`（原始帧 | HRNet 输入 crop 1×2 横向同帧动画），
不写其他文件。

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test7_occlusion.py --occ 50 --session S5091
    python results_display/script/d_test7_occlusion.py --occ 50 --split test
    python results_display/script/d_test7_occlusion.py --occ 50 --split all
    python results_display/script/d_test7_occlusion.py --occ 50 --session S5091 --stride 4 --max-frames 8
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
    os.environ["MPLCONFIGDIR"] = "/tmp/d_test7_matplotlib"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from loguru import logger as log  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from utils import cli_common, render_common as rc  # noqa: E402
# 与训练提取共用同一实现与参数：遮挡（extract_hrnet.py:145 occlude_lower_strip，
# 逐帧 bbox 锚定 + 全宽置黑）→ 裁剪（crop.py:114 process_image，256×192）。
from anysole.data.crop import (  # noqa: E402
    CROP_IMG_HEIGHT,
    CROP_IMG_WIDTH,
    process_image,
)
from anysole.data.dataset import find_session_dir, hrnet_cache_path  # noqa: E402
from anysole.data.extract_hrnet import (  # noqa: E402
    CLIFF_FOCAL_LENGTH,
    occlude_lower_strip,
    occlusion_cache_dirname,
    read_rgb,
    session_frame_paths,
)
from anysole.types import HRNET_CACHE_ROOT, SEQ_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "results_display" / "DataTest" / "D7Test_occlusion"
SPLITS = cli_common.DEFAULT_SPLIT_CSV
BBOX_ROOT = HRNET_CACHE_ROOT.parent / "bbox_cam3"  # 遮挡组 bbox 复用 cam3（§10）
FPS = cli_common.DEFAULT_FPS                        # cam3 rgb 帧率（训练数据链口径）

PANEL_W = 640          # 单面板宽：原始帧 | HRNet 输入 crop 1x2 横向对比
HEADER_H = 94          # 主标题区：标题和帧信息各占一行
GAP = 16               # 面板间距
BG = (12, 12, 16)
OUTLINE = (48, 48, 56)
BBOX_COLOR = (80, 255, 120)    # bbox 绿框
CROP_COLOR = (255, 200, 60)    # crop 区域黄框（process_image 的 ul/br）
CUT_COLOR = (255, 90, 90)      # 切割线红线
LABEL_BG = (0, 0, 0, 150)      # 面板左上角标注底色（半透明）


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
    font = fitted_font(draw, text, max_width, size, min_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((draw.im.size[0] - (bbox[2] - bbox[0])) / 2, y), text,
              font=font, fill=fill)


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


def panel_label(img: Image.Image, text: str) -> None:
    """面板左上角半透明底标注（原始帧 / HRNet 输入 crop）。"""
    draw = ImageDraw.Draw(img, "RGBA")
    font = rc.load_font(20)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, bbox[2] + 16, bbox[3] + 10], fill=LABEL_BG)
    draw.text((8, 4), text, font=font, fill=(235, 235, 235))


def bbox_info_values(center, scale, img_width: int, img_height: int) -> str:
    """bbox_info 3-d 数值（与 extract_session 完全同算式）。"""
    center = np.asarray(center, dtype=np.float32)
    bbox_size = np.float32(scale) * 200.0
    v0 = (center[0] - img_width / 2.0) / CLIFF_FOCAL_LENGTH * 2.8
    v1 = (center[1] - img_height / 2.0) / CLIFF_FOCAL_LENGTH * 2.8
    v2 = (bbox_size - 0.24 * CLIFF_FOCAL_LENGTH) / (0.06 * CLIFF_FOCAL_LENGTH)
    return f"({v0:.2f}, {v1:.2f}, {v2:.2f})"


def compose_1x2(raw: np.ndarray, box: np.ndarray, ul, br, cut: int,
                crop_img: np.ndarray, frac: float, occ_pct: int, sid: str,
                fi: int, n_frames: int, bbox_info: str) -> np.ndarray:
    """原始帧 | HRNet 输入 crop 1×2 横向排布，顶部双行标题与面板完全分离。

    左面板叠加 bbox 绿框、crop 区域黄框（process_image 返回的 ul/br）与
    切割线红线（缩放后坐标）；右面板为 process_image 的 crop_img 直接放大
    （最近邻，像素级原样），即 HRNet 实际吃到的输入。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    h0, w0 = raw.shape[:2]
    scale = PANEL_W / w0
    panel_h = max(1, int(round(h0 * scale)))

    raw_img = Image.fromarray(raw).resize((PANEL_W, panel_h), Image.BILINEAR)
    panel_label(raw_img, "cam3 原始帧（遮挡 + 裁剪锚定）")
    draw = ImageDraw.Draw(raw_img)
    draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                   outline=BBOX_COLOR, width=3)
    draw.rectangle([ul[0] * scale, ul[1] * scale, br[0] * scale, br[1] * scale],
                   outline=CROP_COLOR, width=2)
    draw.line([0, cut * scale, PANEL_W - 1, cut * scale],
              fill=CUT_COLOR, width=3)

    crop_h, crop_w = crop_img.shape[:2]
    crop_scale = panel_h / crop_h
    crop_disp = Image.fromarray(crop_img.astype(np.uint8)).resize(
        (max(1, int(round(crop_w * crop_scale))), panel_h), Image.NEAREST)
    crop_panel = Image.new("RGB", (PANEL_W, panel_h), BG)
    crop_panel.paste(crop_disp, ((PANEL_W - crop_disp.width) // 2, 0))
    panel_label(crop_panel, f"HRNet 输入 crop {crop_w}×{crop_h}（遮挡后）")

    canvas = Image.new("RGB", (2 * PANEL_W + GAP, HEADER_H + panel_h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=OUTLINE)
    draw_centered(draw, f"{sid} · cam3 bbox 底部遮挡管线确认（occ {occ_pct}%）",
                  12, fill=(230, 230, 235), size=30,
                  max_width=canvas.width - 48)
    draw_centered(
        draw,
        f"bbox ({x1:.0f}, {y1:.0f})-({x2:.0f}, {y2:.0f}) · cut y={cut}"
        f"（bbox 下 f={frac:.2f}）· bbox_info {bbox_info} · "
        f"frame {fi}/{n_frames - 1} · t={fi / FPS:.2f}s",
        56, fill=(150, 150, 160), size=18, max_width=canvas.width - 48,
        min_size=13)
    canvas.paste(raw_img, (0, HEADER_H))
    canvas.paste(crop_panel, (PANEL_W + GAP, HEADER_H))
    return np.asarray(canvas)


def render_one(sid: str, args: argparse.Namespace) -> Path | None:
    """单 session：闸门 → 读帧 + bbox → 遮挡 → process_image 复算 → 渲染（只读）。

    遮挡与裁剪调用与 anysole/data/extract_hrnet.py 的 extract_session 完全
    同实现、同参数、同顺序；无对应 occ 缓存时跳过（该 session 未参与遮挡
    实验），返回 None。
    """
    frac = args.occ / 100.0
    cache_path = hrnet_cache_path(
        sid, HRNET_CACHE_ROOT.parent / occlusion_cache_dirname(3, frac))
    if not cache_path.is_file():
        log.warning("{}: 无 occ{:03d} 缓存，跳过", sid, args.occ)
        return None

    seq_dir = find_session_dir(SEQ_ROOT, sid)
    meta = json.loads((seq_dir / "session.json").read_text())
    n_frames = int(meta["frame_count"])
    image_paths = session_frame_paths(seq_dir, n_frames)
    bbox = np.load(str(BBOX_ROOT / f"{sid}.npy"))
    if bbox.shape[0] != n_frames:
        raise ValueError(
            "%s bbox.npy length %s != n_frames %d" % (sid, bbox.shape, n_frames))

    frame_ids = list(range(0, n_frames, max(args.stride, 1)))
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    log.info("{}: {} 帧（occ {}%，{} 帧渲染）", sid, n_frames, args.occ,
             len(frame_ids))

    frames: list[np.ndarray] = []
    for fi in frame_ids:
        fi = int(fi)
        raw = read_rgb(image_paths[fi])
        box = bbox[fi][1:5]
        # —— 与 extract_session 逐位相同的处理链 ——
        occ = occlude_lower_strip(raw, box, frac, "black")
        norm_img, center, scale, ul, br, crop_img = process_image(
            occ, box, crop_height=CROP_IMG_HEIGHT, crop_width=CROP_IMG_WIDTH)
        y1, y2 = float(box[1]), float(box[3])
        cut = int(round(y1 + (1.0 - frac) * (y2 - y1)))
        img_height, img_width = raw.shape[:2]
        info = bbox_info_values(center, scale, img_width, img_height)
        if cut <= 0 or cut >= img_height - 1:  # 与 occlude_lower_strip 同判据：实际未置黑
            info += "（cut 越界，未遮挡）"
        frames.append(compose_1x2(raw, box, ul, br, cut, crop_img, frac,
                                  args.occ, sid, fi, n_frames, info))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    media = args.out_dir / f"{sid}_occ{args.occ:03d}.{args.gen}"
    frame_fps = cli_common.viz_fps(FPS, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, media, frame_fps)
    else:
        cli_common.write_mp4(frames, media, frame_fps)
    log.info("Wrote {} ({} frames, {}x{})", media, len(frames),
             frames[0].shape[1], frames[0].shape[0])
    return media


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test7: cam3 bbox 底部遮挡管线可视化（只读 GIF/MP4）")
    p.add_argument("--occ", type=int, default=50,
                   help="遮挡档百分比（实验档位 10/20/30/50，默认 50）；"
                        "对应缓存根 cam3_occ_lower{FFF}")
    p.add_argument("--session", type=str, default="",
                   help="Session ids，逗号分隔（默认空 = splits.csv 全部 session）")
    p.add_argument("--split", type=str, default="all",
                   choices=["train", "val", "test", "all"],
                   help="批量范围（默认 all = train ∪ val ∪ test；--session 优先）")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--gen", choices=("gif", "mp4"), default="gif",
                   help="输出动画格式（默认 gif；可选 mp4）")
    p.add_argument("--stride", type=int, default=4,
                   help="抽帧步长（40Hz 原始帧率；照片帧 GIF 体积大，默认 4）")
    p.add_argument("--max-frames", type=int, default=0, help="帧数上限（0=不限）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions = (cli_common.split_csv_arg(args.session) if args.session
                else split_sessions(args.split))
    if not sessions:
        raise SystemExit("无 session：--session 为空且 splits.csv 无数据")
    log.info("共 {} 个 session（split={}），occ={}%，输出格式={}，输出目录={}",
             len(sessions),
             args.split if not args.session else "-", args.occ, args.gen.upper(),
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
    log.info("done: {} 写入 / {} 跳过 / {} 失败",
             wrote, len(sessions) - wrote - len(failures), len(failures))
    if failures:
        for sid, err in failures:
            log.error("FAILED {}: {}", sid, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
