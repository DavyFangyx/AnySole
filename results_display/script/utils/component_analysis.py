#!/usr/bin/env python3
"""C1/C2/C3/M1/M2 shared component analysis for the missing-rate task book.

The legacy probe and t-SNE views are intentionally not part of the main
experiment anymore.  This module implements the contrasts used by the
revised task book (the canonical analysis entry points import ``main``):

  --c1  V/T specialist division
  --c2  main VT versus the best V/T specialist (fusion completeness)
  --c3  main V-only versus main T-only (missing-condition division)
  --m1  main V versus V specialist and prior on T-domain rows
  --m2  main T versus T specialist and prior on V-domain rows

2026-10-07 起对比行集由 7 分量表改为区域分组（上半身/下半身/全身/足-地，
见 utils/missing_rate_analysis.py REGION_GROUPS）。表现结构按 2026-10-07
裁定：C1/C3 = 1×2 面板（左 V 右 T）、C2 = 最佳专才 vs 主线 1×2 + 三色合图、
M1/M2 = 三色合图 + 1×3 面板（M1 行 = T 域 9、M2 行 = V 域 10）。

画布由 ``build_figure`` 全手工排版（``fig.add_axes`` 绝对英寸几何，不用
``tight_layout`` / ``bbox_inches="tight"``），满足用户裁定：

  * 面板自足：每个面板自带行标签列与面板标题，任一面板可单独裁剪阅读；
  * 行标签按模态域着色：V 域 #2a78d6 蓝、T 域 #eb6834 橙（所有图）；
  * 经典森林图细线端点构图：细横线 + 白边小圆点 + 端点原生单位数值；
  * 混合单位按行归一（x = value / 行最大值），同图各面板共享该行刻度，
    参考线落在 x = 1.0（= 该行最大值），因此线长可跨面板直接比较；
  * 区域分组以组名行 + 跨图浅分隔线表示；轴外不画任何元素。

``--all`` runs all shared component analyses.  ``--bar`` is retained as a
compatibility alias for ``--c2``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, Union

SCRIPT_DIR = Path(__file__).resolve().parents[1]  # results_display/script
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from utils.missing_rate_analysis import (
    T_DOMAIN_ROWS,
    V_DOMAIN_ROWS,
    error_value,
    load_fseries,
    load_grid_prior,
    row_specs,
    row_specs_in_domain,
    write_rows,
)
from utils.mpl_fonts import setup_cjk_fonts

setup_cjk_fonts()
# 2026-10-07 用户裁定：图内文字全部加粗加大。图内文本全为英文，改用
# DejaVu Sans（matplotlib 自带真正的 Bold 字面；CJK 字体族无粗体，会被
# 静默降级为正体）。rcParams 只影响本模块产出的图。
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.weight"] = "bold"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
V_COLOR = "#2a78d6"
T_COLOR = "#eb6834"
MAIN_COLOR = "#1baf7a"
SPEC_COLOR = "#8c8b84"
PRIOR_COLOR = "#b9b7ae"

# ---------------------------------------------------------------- 版面几何
# 全部单位英寸；画布尺寸 = 各预留区之和，因此内容天然占满整幅、四周无空带。
# 2026-10-07 用户裁定：删去图题与说明性副标题，省下的版面全部给绘图区
# （行距 0.42→0.50，端点/线宽/字号同步放大），图内只留识别信息。
DPI = 200
ROW_H = 0.50           # 单序列面板每行（含组名行）高度
STACK_ROW_H = 0.50     # 多序列合图每行基础高度
STACK_STEP = 0.32      # 多序列合图内序列垂直间距（行高单位）
LABEL_W = 2.95         # 行标签列宽（含右侧方向箭头列；12.5pt 粗体最长标签 2.28in）
ARROW_COL_W = 0.30     # 行标签右侧方向箭头列宽（2026-10-07 用户裁定）
GUTTER_W = 3.25        # 行标签列 + 与面板的间隙
PANEL_W = 5.10         # 单序列面板宽
PANEL_W_STACK = 6.60   # 多序列合图面板宽
PANEL_GAP = 0.90       # 面板组之间的间距
LEFT_MARGIN = 0.30     # 左侧留白（脚注/行标签列共用此左边距）
RIGHT_MARGIN = 0.28    # 右侧留白（"row max" 刻度文字基本在面板内，仅留呼吸）
BOTTOM_MARGIN = 0.46   # 轴下方留白（x 刻度文字）
FOOTNOTE_H = 0.36      # 每行单位脚注占用的底部高度
TOP_PAD = 0.44         # 面板标题条（标题在轴上方，无图题）
LEGEND_PAD = 0.48      # 合图图例条（仅 stacked；在面板标题条之上）

X_MAX = 1.045          # x 上限：1.0 = 行最大值，留出端点圆的半宽

# 2026-10-07 用户裁定：所有字号加粗加大（DejaVu Sans Bold）。
ROW_LABEL_FS = 12.5
GROUP_LABEL_FS = 11.5
VALUE_FS = 10.0
PANEL_TITLE_FS = 15.5
NOTE_FS = 10.5
LEGEND_FS = 13.0
TICK_FS = 10.0

# 森林图端构图（细横线 + 白边端点圆 + 端点数值）。
LINE_W = 2.6           # 端点连线线宽
DOT_S = 56.0           # 端点圆面积（pt²）
DOT_EDGE_W = 1.1       # 白边宽
VALUE_DX = 8.5         # 数值标签相对端点的水平偏移（pt）
VALUE_DY = 2.5         # 数值标签相对端点的垂直偏移（pt）

# 唯一保留的脚注：单位与方向，单行短句（无说明性段落）。
UNIT_NOTE = "Units: mm · m/s² · % · ° · F1 — lower is better except Contact (higher better)."

ColorSpec = Union[dict, Callable[[dict, str], str]]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1-C3/M1-M2 missing-rate component analysis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--c1", action="store_true", help="C1: V/T specialist division")
    mode.add_argument("--all", action="store_true", help="run all shared component analyses")
    mode.add_argument("--c2", action="store_true", help="C2: main VT versus best specialist")
    mode.add_argument("--c3", action="store_true", help="C3: main V-only versus main T-only")
    mode.add_argument("--m1", action="store_true", help="M1: main V on T-domain rows")
    mode.add_argument("--m2", action="store_true", help="M2: main T on V-domain rows")
    mode.add_argument("--bar", action="store_true", help="deprecated alias for --c2")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--fs-main", type=Path,
                        default=REPO / "results" / "experiments" / "singlemodal_eval" / "V3_3B" / "metrics" / "val.json")
    parser.add_argument("--fs-vspecialist", type=Path, default=None,
                        help="V 专才明细文件 (metrics/<split>.json)；C2/M1 必需")
    parser.add_argument("--fs-tspecialist", type=Path, default=None,
                        help="T 专才明细文件 (metrics/<split>.json)；C2/M2 必需")
    parser.add_argument("--prior-grid", type=Path,
                        default=REPO / "results" / "experiments" / "rho_grid_eval" / "V3_3B" / "grid_metrics.json",
                        help="主线 rhoV0_rhoT0 先验单元，M1/M2 必需")
    parser.add_argument("--prior-cell", default="rhoV0_rhoT0")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test1_ComplementTest" / "C2_fusion")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="误差差值分类容差；混合单位时默认 0，只按方向分类")
    return parser.parse_args(argv)


def metric_block(data: dict[str, Any], config: str) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    if config not in metrics:
        raise ValueError("明细文件缺配置 %s；已有：%s" % (config, ", ".join(metrics)))
    return metrics[config]


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    main = load_fseries(args.fs_main)
    out = {"main": main}
    if args.fs_vspecialist:
        out["v_specialist"] = load_fseries(args.fs_vspecialist)
    if args.fs_tspecialist:
        out["t_specialist"] = load_fseries(args.fs_tspecialist)
    return out


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CSV 行（去掉渲染用的 values/aliases/higher 键）。"""
    return [{k: v for k, v in row.items() if k not in ("values", "aliases", "higher")} for row in rows]


def domain_label_colors(rows: list[dict[str, Any]]) -> dict[str, str]:
    """V 域行标签蓝色、T 域行标签橙色。"""
    return {row["row_id"]: V_COLOR if row["row_id"] in V_DOMAIN_ROWS else T_COLOR
            for row in rows}


def _fmt(row: dict[str, Any], value: float) -> str:
    if row["row_id"] == "contact":
        return "%.3f" % value
    if row["row_id"] == "accel":
        return "%.2f" % value
    if row["row_id"] == "root_traj":
        return "%.1f%%" % value
    if row["row_id"] == "root_orient":
        return "%.1f°" % value
    return "%.1f" % value


# ------------------------------------------------------------------ 渲染器
def _levels(rows: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """[("group", 组名) | ("row", 行)]，同一区域组只出现一次组名行。"""
    levels: list[tuple[str, Any]] = []
    last_group: str | None = None
    for row in rows:
        if row["region_label"] != last_group:
            levels.append(("group", row["region_label"]))
            last_group = row["region_label"]
        levels.append(("row", row))
    return levels


def _row_refs(rows: list[dict[str, Any]]) -> dict[str, float]:
    """每行的归一化基准 = 该行所有序列值的最大值（<=0 时退化为 1.0）。"""
    refs: dict[str, float] = {}
    for row in rows:
        values = [float(v) for v in row["values"].values() if v is not None and np.isfinite(float(v))]
        top = max(values) if values else 0.0
        refs[row["row_id"]] = top if top > 0 else 1.0
    return refs


def _series_color(colors: ColorSpec, row: dict[str, Any], name: str) -> str:
    if callable(colors):
        return colors(row, name)
    return colors.get(name, INK2)


def _draw_value(ax, row: dict[str, Any], value: float, y: float, ref: float,
                color: str, label_fs: float = VALUE_FS, fig=None) -> None:
    """一行：细横线 0→x、白边端点圆、端点原生单位数值（经典森林图）。

    数值标签用点偏移（不随面板宽度变化），故面板加宽后间距仍稳定。
    """
    x = value / ref
    ax.hlines(y, 0.0, x, color=color, linewidth=LINE_W, zorder=3)
    ax.scatter([x], [y], s=DOT_S, color=color, edgecolors="white",
               linewidths=DOT_EDGE_W, zorder=4)
    text = _fmt(row, value)
    right = x > 0.78
    transform = offset_copy(ax.transData, fig=fig,
                            x=-VALUE_DX if right else VALUE_DX, y=VALUE_DY,
                            units="points")
    ax.text(x, y, text, transform=transform,
            ha="right" if right else "left", va="bottom",
            fontsize=label_fs, color=INK, zorder=6)


def _as_lines(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value]


def _panel_title(ax, fig, text: str, marker: str | None) -> None:
    """面板标题：白边小圆点（序列色）+ 正文（INK），始终可读且给出颜色对应。"""
    if not text:
        return
    offset = 0.0
    if marker:
        ax.plot([0.0], [1.0],
                transform=offset_copy(ax.transAxes, fig=fig, x=5.0, y=15.5, units="points"),
                marker="o", markersize=9.5, markeredgecolor="white", markeredgewidth=1.2,
                color=marker, linestyle="none", clip_on=False, zorder=6)
        offset = 15.5
    ax.text(0.0, 1.0, text,
            transform=offset_copy(ax.transAxes, fig=fig, x=offset, y=9.0, units="points"),
            ha="left", va="bottom", fontsize=PANEL_TITLE_FS, color=INK, zorder=7)


def build_figure(rows: list[dict[str, Any]], series: Sequence[str], colors: ColorSpec,
                 title: str, *, subtitle: str | Sequence[str] | None = None,
                 # subtitle 仅为兼容旧调用保留；按用户裁定不再落图。
                 footnote: str | Sequence[str] | None = None,
                 layout: str = "columns",
                 panel_titles: Sequence[str] | None = None,
                 panel_title_markers: Sequence[str | None] | None = None,
                 label_colors: dict[str, str] | None = None,
                 dpi: int = DPI,
                 row_height: float | None = None,
                 panel_width: float | None = None):
    """构建森林图并返回 Figure（不落盘，便于校验脚本复检）。

    layout="columns"：每序列一个面板（1×N），面板各自带行标签列；
    layout="stacked"：单一面板内多序列按行内垂直错开，图例置于面板标题条之上。
    每行按该行最大值归一，各面板共享同一刻度，参考线在 x = 1.0。

    2026-10-07 用户裁定：图内不画图题与说明性副标题（``title`` 仅保留为
    调用方语义标识，不落图）；识别信息 = 面板标题 + 图例 + 行/组标签 +
    端点数值，底部只留一行单位/方向脚注。省下的版面全部给绘图区。
    """
    if layout not in ("columns", "stacked"):
        raise ValueError("layout 只能是 columns / stacked：%r" % layout)
    if not rows:
        raise ValueError("rows 为空")
    if not series:
        raise ValueError("series 为空")

    series = [str(name) for name in series]
    known: set[str] = set()
    for row in rows:
        known.update(row["values"].keys())
    unknown = [name for name in series if name not in known]
    if unknown:
        # 防回归：序列名必须在每行的 values 里（曾出现 "Best specialist" 全 "—"）
        raise ValueError("series 名不在行 values 中：%s（可用：%s）" % (unknown, sorted(known)))

    stacked = layout == "stacked"
    n_series = len(series)
    n_panels = 1 if stacked else n_series
    levels = _levels(rows)
    n_levels = len(levels)
    label_colors = label_colors or {}

    if stacked:
        row_h = row_height if row_height is not None else STACK_ROW_H + STACK_STEP * (n_series - 1)
        panel_w = panel_width if panel_width is not None else PANEL_W_STACK
    else:
        row_h = row_height if row_height is not None else ROW_H
        panel_w = panel_width if panel_width is not None else PANEL_W

    if panel_titles:
        panel_titles = [str(item) for item in panel_titles]
    else:
        # 列布局：面板标题即序列名（充当图例）；合图：图例在标题区，面板不重复标题。
        panel_titles = [""] * n_panels if stacked else list(series)
    if len(panel_titles) != n_panels:
        raise ValueError("panel_titles 数量 %d 与面板数 %d 不符" % (len(panel_titles), n_panels))
    if panel_title_markers is not None:
        panel_title_markers = list(panel_title_markers)
    elif stacked:
        panel_title_markers = [None] * n_panels
    else:
        panel_title_markers = [_series_color(colors, rows[0], name) for name in series]
    if len(panel_title_markers) != n_panels:
        raise ValueError("panel_title_markers 数量与面板数不符")
    footnote_lines = _as_lines(footnote)

    # 顶部只有面板标题条（合图再加一条图例条）；图题/副标题已按用户裁定删除。
    legend_y = 0.24 if stacked else None
    top = TOP_PAD + (LEGEND_PAD if stacked else 0.0)
    bottom = BOTTOM_MARGIN + FOOTNOTE_H * len(footnote_lines)

    group_w = GUTTER_W + panel_w
    width = LEFT_MARGIN + n_panels * group_w + (n_panels - 1) * PANEL_GAP + RIGHT_MARGIN
    height = top + n_levels * row_h + bottom

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)

    ax_b = bottom / height
    ax_h = n_levels * row_h / height
    refs = _row_refs(rows)
    ylim = (-0.5, n_levels - 0.5)
    y_of = {index: n_levels - 1 - index for index in range(n_levels)}
    separators = [index for index, (kind, _) in enumerate(levels) if kind == "group" and index > 0]

    # ---- 图例条（仅合图）：置于面板标题条之上，紧贴上缘
    if stacked:
        handles = [
            plt.Line2D([], [], color=_series_color(colors, rows[0], name), marker="o",
                       markersize=9.5, markeredgecolor="white", markeredgewidth=1.2,
                       linewidth=LINE_W)
            for name in series
        ]
        fig.legend(handles, series, loc="upper left",
                   bbox_to_anchor=(LEFT_MARGIN / width, 1.0 - legend_y / height),
                   frameon=False, prop=FontProperties(family="DejaVu Sans", weight="bold",
                                                      size=LEGEND_FS),
                   ncol=n_series, handlelength=1.6, handletextpad=0.6, columnspacing=2.0,
                   borderpad=0.0, borderaxespad=0.0)

    # ---- 底部单位/方向脚注（单行短句，无说明性段落）
    for row_index, line in enumerate(footnote_lines):
        fig.text(LEFT_MARGIN / width, (0.26 + 0.30 * (len(footnote_lines) - 1 - row_index)) / height, line,
                 ha="left", va="bottom", fontsize=NOTE_FS, color=MUTED)

    # ---- 面板：每面板 = 自带的行标签列 + 绘图区（面板自足）
    for panel in range(n_panels):
        gx = LEFT_MARGIN + panel * (group_w + PANEL_GAP)

        ax_lab = fig.add_axes([gx / width, ax_b, LABEL_W / width, ax_h])
        ax_lab.set_xlim(0.0, 1.0)
        ax_lab.set_ylim(*ylim)
        ax_lab.axis("off")
        label_x = (LABEL_W - ARROW_COL_W) / LABEL_W      # 指标名右缘（箭头列左界）
        arrow_x = (LABEL_W - ARROW_COL_W + 0.06) / LABEL_W  # 箭头列左缘（统一对齐）
        for index, (kind, item) in enumerate(levels):
            y = y_of[index]
            if kind == "group":
                ax_lab.text(0.995, y, item, ha="right", va="center",
                            fontsize=GROUP_LABEL_FS, color=INK2)
            else:
                ax_lab.text(label_x, y, item["label"], ha="right", va="center",
                            fontsize=ROW_LABEL_FS,
                            color=label_colors.get(item["row_id"], INK))
                # 方向箭头：higher=True → ↑（越大越好），False → ↓（越小越好）
                ax_lab.text(arrow_x, y, "↑" if item["higher"] else "↓",
                            ha="left", va="center", fontsize=ROW_LABEL_FS, color=INK2)
        ax = fig.add_axes([(gx + GUTTER_W) / width, ax_b, panel_w / width, ax_h])
        ax.set_facecolor(SURFACE)
        ax.set_ylim(*ylim)
        ax.set_xlim(0.0, X_MAX)
        ax.set_yticks([])
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "row max"], fontsize=TICK_FS, color=INK2)
        ax.tick_params(axis="x", length=0, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(1.2)
        ax.axvline(1.0, color=GRID, linewidth=1.2, zorder=1)
        # 区域分隔线：跨"标签列 + 间隙 + 面板"画整条浅规则线（图坐标），
        # 组边界因此一眼可读且中途不断开。
        for index in separators:
            frac = (n_levels - index) / float(n_levels)
            y_fig = ax_b + frac * ax_h
            fig.add_artist(Line2D([gx / width, (gx + group_w) / width], [y_fig, y_fig],
                                  transform=fig.transFigure, color=GRID,
                                  linewidth=1.1, zorder=1))

        for index, (kind, item) in enumerate(levels):
            if kind == "group":
                continue
            row = item
            ref = refs[row["row_id"]]
            if stacked:
                for slot, name in enumerate(series):
                    value = row["values"].get(name)
                    if value is None:
                        continue
                    y = y_of[index] + STACK_STEP * (slot - (n_series - 1) / 2.0)
                    _draw_value(ax, row, value, y, ref, _series_color(colors, row, name), fig=fig)
            else:
                name = series[panel]
                value = row["values"].get(name)
                if value is None:
                    ax.text(0.03, y_of[index], "—", ha="left", va="center",
                            fontsize=VALUE_FS, color=MUTED)
                    continue
                _draw_value(ax, row, value, y_of[index], ref, _series_color(colors, row, name), fig=fig)

        _panel_title(ax, fig, panel_titles[panel], panel_title_markers[panel])

    return fig


def render_figure(path: Path, rows: list[dict[str, Any]], series: Sequence[str],
                  colors: ColorSpec, title: str, **kwargs) -> None:
    """建图 + 落盘（无 bbox_inches，版面即画布，四周无空带）。"""
    fig = build_figure(rows, series, colors, title, **kwargs)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- 分析
def run_c1(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C1 需要 --fs-vspecialist 与 --fs-tspecialist")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for spec in row_specs():
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        delta = t_err - v_err
        winner = "V specialist" if delta > args.tol else "T specialist" if delta < -args.tol else "tie"
        rows.append({
            **spec,
            "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "V_specialist": round(v_raw, 6), "T_specialist": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6), "winner": winner,
            "values": {"V specialist": v_raw, "T specialist": t_raw},
        })
    out = args.out
    write_rows(out / "specialist_table.csv", _csv_rows(rows))
    render_figure(
        out / "specialist_panels.png", rows,
        ["V specialist", "T specialist"],
        {"V specialist": V_COLOR, "T specialist": T_COLOR},
        "C1 · modality-specialist division across body-region rows",
        footnote=[UNIT_NOTE],
        panel_titles=["V specialist", "T specialist"],
        panel_title_markers=[V_COLOR, T_COLOR],
        label_colors=domain_label_colors(rows),
    )
    summary = {"V_wins": [r["row_id"] for r in rows if r["winner"] == "V specialist"],
               "T_wins": [r["row_id"] for r in rows if r["winner"] == "T specialist"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c2(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    if "v_specialist" not in inputs or "t_specialist" not in inputs:
        raise SystemExit("C2 需要 --fs-vspecialist 与 --fs-tspecialist")
    main = metric_block(inputs["main"], "VT2M")
    v = metric_block(inputs["v_specialist"], "V2M")
    t = metric_block(inputs["t_specialist"], "T2M")
    rows = []
    for spec in row_specs():
        main_key, main_raw, main_err = error_value(main, spec["aliases"], spec["higher"])
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        best_err = min(v_err, t_err)
        best_raw = v_raw if v_err <= t_err else t_raw
        best_src = "V specialist" if v_err <= t_err else "T specialist"
        delta = main_err - best_err
        category = "synergy" if delta < -args.tol else "interference" if delta > args.tol else "keep or select"
        rows.append({
            **spec,
            "metric": main_key,
            "main_VT": round(main_raw, 6), "V_specialist": round(v_raw, 6),
            "T_specialist": round(t_raw, 6), "best_specialist": round(best_raw, 6),
            "best_source": best_src,
            "delta_error_VT_minus_best": round(delta, 6), "category": category,
            "values": {"V specialist": v_raw, "T specialist": t_raw, "Main VT": main_raw,
                       "Best specialist": best_raw},
            "source_color": V_COLOR if v_err <= t_err else T_COLOR,
        })
    out = args.out
    write_rows(out / "fusion_table.csv", _csv_rows(rows))
    # 最佳专才：逐行取 V/T 中更好者的原始值，端点颜色 = 胜出来源（蓝 V / 橙 T）
    render_figure(
        out / "fusion_best_panels.png", rows,
        ["Best specialist", "Main VT"],
        lambda row, name: MAIN_COLOR if name == "Main VT" else row["source_color"],
        "C2 · best single-modality specialist vs fused main VT",
        footnote=[UNIT_NOTE],
        panel_titles=["Best specialist", "Main VT"],
        panel_title_markers=[None, MAIN_COLOR],
        label_colors=domain_label_colors(rows),
    )
    render_figure(
        out / "fusion_series.png", rows,
        ["V specialist", "T specialist", "Main VT"],
        {"V specialist": V_COLOR, "T specialist": T_COLOR, "Main VT": MAIN_COLOR},
        "C2 · specialists and fused main VT on shareable scales",
        footnote=[UNIT_NOTE],
        layout="stacked",
        label_colors=domain_label_colors(rows),
        panel_titles=["Main VT vs specialists"],
    )
    summary = {"synergy": sum(r["category"] == "synergy" for r in rows),
               "selection": sum(r["category"] == "keep or select" for r in rows),
               "interference": sum(r["category"] == "interference" for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_c3(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    v = metric_block(inputs["main"], "V2M")
    t = metric_block(inputs["main"], "T2M")
    rows = []
    for spec in row_specs():
        v_key, v_raw, v_err = error_value(v, spec["aliases"], spec["higher"])
        t_key, t_raw, t_err = error_value(t, spec["aliases"], spec["higher"])
        delta = t_err - v_err
        winner = "V-only" if delta > args.tol else "T-only" if delta < -args.tol else "tie"
        expected = "V-only" if spec["row_id"] in V_DOMAIN_ROWS else "T-only"
        rows.append({
            **spec,
            "metric": v_key if v_key == t_key else "%s / %s" % (v_key, t_key),
            "main_V": round(v_raw, 6), "main_T": round(t_raw, 6),
            "delta_error_T_minus_V": round(delta, 6), "winner": winner,
            "expected_winner": expected, "matches_expected": winner == expected,
            "values": {"Main V-only": v_raw, "Main T-only": t_raw},
        })
    out = args.out
    write_rows(out / "main_branch_table.csv", _csv_rows(rows))
    render_figure(
        out / "main_branch_panels.png", rows,
        ["Main V-only", "Main T-only"],
        {"Main V-only": V_COLOR, "Main T-only": T_COLOR},
        "C3 · main model on V-only vs T-only input",
        footnote=[UNIT_NOTE],
        panel_titles=["Main V-only", "Main T-only"],
        panel_title_markers=[V_COLOR, T_COLOR],
        label_colors=domain_label_colors(rows),
    )
    summary = {"matches_expected": sum(bool(r["matches_expected"]) for r in rows), "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def branch_table(args: argparse.Namespace, inputs: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch == "m1":
        specialist_name, specialist_cfg, main_cfg = "v_specialist", "V2M", "V2M"
        domain_rows = T_DOMAIN_ROWS
        main_label, spec_label = "Main V", "V specialist"
        main_color = V_COLOR
        series_title = "M1 · main V vs V specialist vs prior on T-domain rows"
        series_panel_title = "Main V vs specialist and prior (T-domain rows)"
        panel_title = "M1 · main V vs V specialist vs prior, one series per panel"
    else:
        specialist_name, specialist_cfg, main_cfg = "t_specialist", "T2M", "T2M"
        domain_rows = V_DOMAIN_ROWS
        main_label, spec_label = "Main T", "T specialist"
        main_color = T_COLOR
        series_title = "M2 · main T vs T specialist vs prior on V-domain rows"
        series_panel_title = "Main T vs specialist and prior (V-domain rows)"
        panel_title = "M2 · main T vs T specialist vs prior, one series per panel"
    if specialist_name not in inputs:
        raise SystemExit("%s 需要对应专才明细文件" % branch.upper())
    main = metric_block(inputs["main"], main_cfg)
    specialist = metric_block(inputs[specialist_name], specialist_cfg)
    prior = load_grid_prior(args.prior_grid, args.prior_cell)
    rows = []
    for spec in row_specs_in_domain(domain_rows):
        m_key, m_raw, m_err = error_value(main, spec["aliases"], spec["higher"])
        s_key, s_raw, s_err = error_value(specialist, spec["aliases"], spec["higher"])
        p_key, p_raw, p_err = error_value(prior, spec["aliases"], spec["higher"])
        d_spec = m_err - s_err
        d_prior = m_err - p_err
        rows.append({
            **spec,
            "metric": m_key,
            "prior": round(p_raw, 6), "specialist": round(s_raw, 6), "main": round(m_raw, 6),
            "delta_error_main_minus_specialist": round(d_spec, 6),
            "delta_error_main_minus_prior": round(d_prior, 6),
            "beats_specialist": d_spec < -args.tol,
            "beats_prior": d_prior < -args.tol,
            "holds": d_spec < -args.tol and d_prior < -args.tol,
            "values": {main_label: m_raw, spec_label: s_raw, "Prior": p_raw},
        })
    out = args.out
    write_rows(out / "branch_table.csv", _csv_rows(rows))
    series = [main_label, spec_label, "Prior"]
    colors = {main_label: main_color, spec_label: SPEC_COLOR, "Prior": PRIOR_COLOR}
    # M1 行全为 T 域（橙标签）、M2 行全为 V 域（蓝标签）
    label_colors = domain_label_colors(rows)
    render_figure(
        out / "branch_series.png", rows, series, colors, series_title,
        footnote=[UNIT_NOTE],
        layout="stacked", label_colors=label_colors,
        panel_titles=[series_panel_title],
    )
    render_figure(
        out / "branch_panels.png", rows, series, colors, panel_title,
        footnote=[UNIT_NOTE],
        layout="columns", label_colors=label_colors,
        panel_titles=series, panel_title_markers=[main_color, SPEC_COLOR, PRIOR_COLOR],
    )
    summary = {"holds": [r["row_id"] for r in rows if r["holds"]],
               "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args)
    if args.bar:
        args.c2 = True
    modes = {"c1": args.c1, "c2": args.c2, "c3": args.c3, "m1": args.m1, "m2": args.m2}
    if args.all:
        modes = {key: True for key in modes}
    elif not any(modes.values()):
        modes = {key: False for key in modes}
        modes["c2"] = True
    results = {}
    for mode_key in ("c1", "c2", "c3", "m1", "m2"):
        if not modes[mode_key]:
            continue
        # --all 一次跑多模式时按模式分目录，避免同名文件互相覆盖
        mode_out = args.out / mode_key if args.all else args.out
        mode_out.mkdir(parents=True, exist_ok=True)
        mode_args = argparse.Namespace(**{**vars(args), "out": mode_out})
        if mode_key == "c1":
            results[mode_key] = run_c1(mode_args, inputs)
        elif mode_key == "c2":
            results[mode_key] = run_c2(mode_args, inputs)
        elif mode_key == "c3":
            results[mode_key] = run_c3(mode_args, inputs)
        elif mode_key == "m1":
            results[mode_key] = branch_table(mode_args, inputs, "m1")
        elif mode_key == "m2":
            results[mode_key] = branch_table(mode_args, inputs, "m2")
    (args.out / "analysis_summary.json").write_text(
        json.dumps({key: value["summary"] for key, value in results.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("完成：%s（%s）" % (args.out, ", ".join(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
