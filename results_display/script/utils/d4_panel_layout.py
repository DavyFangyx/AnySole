"""D4 基线触觉面板布局口径（2026-10-06 用户裁定参照）。

自 d_test4_baseline_tactile.py 提升的公共件：r_test4_v2t.py 的双足块
布局常量与渲染函数，保留原口径逐位不变。
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils import render_common as rc

PANEL_W = 360          # 单面板宽：1x4 横向对比
PANEL_H = 390          # 单面板高：标题区 + 标签区 + 252 高内容区
FOOT_W = 96            # 四个面板统一的单脚显示宽度
FOOT_H = 252           # 四个面板统一的单脚显示长度
CONTENT_Y = 112        # 内容区顶；与标题/左右脚标签完全分离
GAP = 20               # 面板间距
FOOT_GAP = 24          # 同一面板内左右脚间距
HEADER_H = 94          # 主标题区：标题和帧信息各占一行
DOWNSCALE = 0.75       # 输出缩放
BG = (12, 12, 16)
OUTLINE = (48, 48, 56)
SENSOR_CLIP = 1023.0   # facts/MMVP 原始单位上限（冻结光栅口径 clip(raw,0,1023)/1023*255）


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
