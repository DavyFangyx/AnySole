"""D_Test5: AnySole + 三基线触觉适配可视化（Agent_06 前置）。

可视化前端，不做任何转换计算：先确保 AnysoleWorkspace/tool/generate_baseline_
tactile.py 的产物落盘（产物齐全则跳过生成，缺则调用生成脚本；--force 强制
重生成），再读统一压力源和落盘文件渲染 1x4 横向逐帧动画。四个面板为：

    AnySole         原始 pressure.npz → 原生 4x12 / 脚（主方法参考坐标）
    MotionPRO       AnysoleWorkspace/derived/MotionPRO/pressure_96/<sid>.npz
                    （FRAPPE 实际输入的 96x96；可视化按左右脚等尺度并旋转到参考坐标）
    MMVP            pressure_tookit 与 VP-MoCap 共用的 31x11 原始压力
                    （两棵目录逐帧校验一致，只显示一个逻辑工作）
    Step2Motion     AnysoleWorkspace/derived/Step2Motion/pressure_16ch/<sid>.npz
                    （16 通道/脚，回填为统一脚形仅作展示）

映射口径（生成器与 meta 为准）：点对点按 SMPL 模板足底系欧氏最近点，厂商布局
48 点对齐模板足底包围盒，--mirror-x 翻转内外侧（默认同向假定）。

输出（results_display/data/d_test5_baseline_tactile/）：每 session 仅一个
`<sid>_adapted_tactile.{gif|mp4}`（四面板 1x4 横向同帧对齐动画），不写其他文件。

用法（仓库根目录，touch_gait 环境）：
    python results_display/script/d_test5_baseline_tactile.py
        # 默认：splits.csv 全部 session（train ∪ val ∪ test），逐 session 生成+渲染
    python results_display/script/d_test5_baseline_tactile.py --split test
    python results_display/script/d_test5_baseline_tactile.py --session S12072
    python results_display/script/d_test5_baseline_tactile.py --session S12072 --force
    python results_display/script/d_test5_baseline_tactile.py --session S12072 --gen mp4
    python results_display/script/d_test5_baseline_tactile.py --stride 4 --max-frames 40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TOOL_DIR = REPO_ROOT / "AnysoleWorkspace" / "tool"
for path in (REPO_ROOT, SCRIPT_DIR, TOOL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep Matplotlib's cache warning out of the normal CLI output on this managed
# workspace, where the default user config directory is read-only.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/d_test5_matplotlib"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
from loguru import logger as log  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import generate_baseline_tactile as gen  # noqa: E402
from utils import cli_common, render_common as rc  # noqa: E402

GENERATOR = TOOL_DIR / "generate_baseline_tactile.py"
DEFAULT_OUT = REPO_ROOT / "results_display" / "data" / "d_test5_baseline_tactile"

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
HEADER_H = 94          # 主标题区：标题和帧信息各占一行
DOWNSCALE = 0.75       # 输出缩放
FPS = 40.0

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
                       subtitle: str, color: tuple) -> np.ndarray:
    """双足热图面板：L/R 两鞋垫并排居中，各带标签（标题/副题统一走 base_panel）。"""
    canvas, draw = base_panel(title, subtitle, color)
    left_img = cv2.resize(rc.pressure_to_heatmap(left_block), (FOOT_W, FOOT_H),
                          interpolation=cv2.INTER_NEAREST)
    right_img = cv2.resize(rc.pressure_to_heatmap(right_block), (FOOT_W, FOOT_H),
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


def motionpro_foot_crops(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从 MotionPRO 的 96x96 整图中裁出左右原始脚区。"""
    image = np.asarray(img, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"MotionPRO frame must be 2-D, got {image.shape}")
    height, width = image.shape
    # pressure.npz 的源图是 (160,120)，生成器将其整体双线性缩放到 96x96。
    source_h, source_w = 160.0, 120.0
    crops = []
    for foot_box in (rc.LEFT_FOOT_BOX, rc.RIGHT_FOOT_BOX):
        y_slice, x_slice = foot_box
        y0 = round(y_slice.start * height / source_h)
        y1 = round(y_slice.stop * height / source_h)
        x0 = round(x_slice.start * width / source_w)
        x1 = round(x_slice.stop * width / source_w)
        crops.append(image[y0:y1, x0:x1])
    return crops[0], crops[1]


def render_motionpro_foot_blocks(img: np.ndarray, title: str, subtitle: str,
                                 color: tuple) -> np.ndarray:
    """MotionPRO 96x96 输入按左右脚裁切、逆时针旋转 90° 后显示。"""
    canvas, draw = base_panel(title, subtitle, color)
    left_crop, right_crop = motionpro_foot_crops(img)
    foot_images = []
    for crop in (left_crop, right_crop):
        # Rasterizer 的压力图轴为 (4 rows, 12 cols)，与 AnySole 的竖直
        # (12 length, 4 width) 显示约定相差 90°；这里只改可视化，不改模型输入。
        canonical = np.ascontiguousarray(np.rot90(crop, k=1))
        resized = cv2.resize(np.clip(canonical, 0.0, 255.0), (FOOT_W, FOOT_H),
                             interpolation=cv2.INTER_LINEAR)
        cells = resized / 255.0
        rgb = (cm.inferno(cells)[..., :3] * 255.0).astype(np.uint8)
        foot_images.append(Image.fromarray(rgb))

    pair_w = 2 * FOOT_W + FOOT_GAP
    x0 = (PANEL_W - pair_w) // 2
    draw.text((x0 + FOOT_W // 2, 86), "L", font=rc.LABEL_FONT,
              fill=(180, 210, 255), anchor="mt")
    draw.text((x0 + FOOT_W + FOOT_GAP + FOOT_W // 2, 86), "R",
              font=rc.LABEL_FONT, fill=(255, 190, 160), anchor="mt")
    canvas.paste(foot_images[0], (x0, CONTENT_Y))
    canvas.paste(foot_images[1], (x0 + FOOT_W + FOOT_GAP, CONTENT_Y))
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


# ---------------------------------------------------------------- 生成衔接

def pool_ids() -> np.ndarray:
    """(4,12) 每格对应的 16 通道池化组编号（仅用于 Step2Motion 展示回填）。"""
    ids = np.zeros((4, 12), dtype=int)
    for r in range(4):
        for g in range(2):
            ch = r * 2 + g
            ids[r, 6 + 3 * g:6 + 3 * g + 3] = ch
            ids[r, 3 * g:3 * g + 3] = ch + 8
    return ids

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D_Test5: AnySole + 三基线触觉可视化（GIF/MP4）")
    p.add_argument("--session", type=str, default="",
                   help="单 session（默认空 = splits.csv 全部 session）")
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                   help="批量范围（默认 all = train ∪ val ∪ test；--session 优先）")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gen", choices=("gif", "mp4"), default="gif",
                   help="输出动画格式（默认 gif；可选 mp4）")
    p.add_argument("--stride", type=int, default=2, help="抽帧步长（40Hz 原始帧率）")
    p.add_argument("--max-frames", type=int, default=0, help="帧数上限（0=不限）")
    p.add_argument("--mirror-x", action="store_true",
                   help="透传生成器：布局 x 轴（内外侧）相对模板 x 翻转")
    p.add_argument("--force", action="store_true",
                   help="即使产物齐全也强制重新生成")
    p.add_argument("--skip-generate", action="store_true",
                   help="跳过生成步骤（产物必须已存在）")
    p.add_argument("--verbose", action="store_true", help="同时显示生成器内部的详细日志")
    return p.parse_args()


def ensure_generated(sid: str, args: argparse.Namespace) -> dict:
    """产物齐全则跳过；缺则调用生成脚本；--force 透传强制重生成。"""
    paths = gen.product_paths(sid)
    if args.skip_generate and not gen.products_complete(paths):
        raise FileNotFoundError(f"产物不全且 --skip-generate：{paths}")
    if not args.force and gen.products_complete(paths):
        log.info("{} 产物齐全，跳过生成", sid)
        return paths
    cmd = [sys.executable, str(GENERATOR), "--sessions", sid]
    if args.mirror_x:
        cmd.append("--mirror-x")
    if args.force:
        cmd.append("--force")
    log.info("调用生成器：{}", " ".join(cmd))
    run_kwargs = {"cwd": str(REPO_ROOT)}
    if not args.verbose:
        run_kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                           "text": True})
    proc = subprocess.run(cmd, **run_kwargs)
    if proc.returncode != 0:
        detail = (proc.stdout or "").strip() if not args.verbose else ""
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"生成器退出码 {proc.returncode}{suffix}")
    return paths


def load_insole_frame(insole_dir: Path, fi: int) -> np.ndarray:
    """读一帧 insole 文件 → (31,22) 双足网格。"""
    payload = np.load(insole_dir / f"{fi:03d}.npy", allow_pickle=True).item()
    return np.concatenate([payload["insole"][0], payload["insole"][1]], axis=1)


# ---------------------------------------------------------------- main

def render_one(sid: str, args: argparse.Namespace) -> None:
    """单 session：确保产物落盘 → 读盘渲染一个 GIF/MP4。"""
    paths = ensure_generated(sid, args)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    source = gen.load_nine_piece(gen.resolve_session(sid)["seq_dir"])
    anysole_cells = gen.crop_cells(source["pressure"])
    pressure96 = np.load(paths["motionpro_npz"])["pressure"]
    left16 = np.load(paths["s2m_npz"])["left16"]
    right16 = np.load(paths["s2m_npz"])["right16"]
    n = int(meta["n_frames"])
    log.info("{}: {} 帧（有效 {}），读取 AnySole / MotionPRO / MMVP / Step2Motion",
             sid, n, meta["n_valid"])

    valid = np.asarray(meta["valid_frame_indices"], dtype=int)
    if len(valid) == 0:  # 全 fake session：无有效帧可抽，GIF 回退全帧展示
        log.warning("{}: 无有效帧（all_fake），GIF 回退全帧", sid)
        valid = np.arange(n, dtype=int)
    frame_ids = valid[:: args.stride]
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    log.info("{} 帧数 {}", args.gen.upper(), len(frame_ids))

    pids = pool_ids()
    frames: list[np.ndarray] = []
    for fi in frame_ids:
        # AnySole 主方法：4x12 原生格，统一旋转到竖直脚底显示。
        p_anysole = render_foot_blocks(
            np.rot90(anysole_cells[fi, 0], k=1),
            np.rot90(anysole_cells[fi, 1], k=1),
            "AnySole", "原生 4×12 · 48 格/脚", C_ANYSOLE)
        # MotionPRO：生成器落盘的 96x96 实际输入
        p_motion = render_motionpro_foot_blocks(
            pressure96[fi], "MotionPRO", "FRAPPE 96×96 · L/R 等尺度", C_MOTIONPRO)
        # MMVP：pressure_tookit 与 VP-MoCap 两棵目录必须是同一份观测。
        grid = load_insole_frame(paths["toolkit_insole_dir"], fi)
        fpp_grid = load_insole_frame(paths["fpp_insole_dir"], fi)
        if not np.array_equal(grid, fpp_grid):
            raise ValueError(f"MMVP pressure_tookit/VP-MoCap mismatch at frame {fi}")
        p_mmvp = render_foot_blocks(
            grid[:, :11], grid[:, 11:],
            "MMVP", "pressure_tookit / VP-MoCap · shared 31×11", C_TOOLKIT)
        # Step2Motion：16 通道组值回填 4x12，再使用同一脚形显示尺寸。
        up_l, up_r = left16[fi][pids], right16[fi][pids]
        p_s2m = render_foot_blocks(
            np.rot90(up_l, k=1), np.rot90(up_r, k=1),
            "Step2Motion", "16 通道/脚 · 冻结池化", C_S2M)

        header = f"{sid} · AnySole + 三基线触觉适配"
        subheader = (f"源 AnySole 4×12 每脚 48 格 · frame {fi}/{n - 1} · "
                     f"t={fi / FPS:.2f}s · {FPS:.0f}Hz")
        frame_img = compose_1x4([p_anysole, p_motion, p_mmvp, p_s2m], header, subheader)
        frames.append(np.asarray(frame_img))
    log.info("合成 {} {} 帧，每帧 {}x{}", args.gen.upper(), len(frames),
             frames[0].shape[1], frames[0].shape[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    media = args.out_dir / f"{sid}_adapted_tactile.{args.gen}"
    frame_fps = cli_common.viz_fps(FPS, args.stride)
    if args.gen == "gif":
        cli_common.write_gif(frames, media, frame_fps)
    else:
        cli_common.write_mp4(frames, media, frame_fps)
    log.info("Wrote {}", media)


def main() -> None:
    args = parse_args()
    sessions = [args.session] if args.session else gen.split_sessions(args.split)
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
