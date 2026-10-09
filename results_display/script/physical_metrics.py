#!/usr/bin/env python3
"""T 判定后续 · 实验 2：非 PA 物理指标面板（per-session bootstrap CI）。

对每个目标（模型 × 条件）读取 pooled 明细（metrics/<split>.json，display
名）与其同目录的会话明细（session_detail_<split>.json，snake_case 键），
产出：

  physical_table.csv    — 行 = 指标，列 = 各目标均值 + 95% 会话簇 bootstrap CI
  physical_ci.json      — 完整 CI 数值
  physical_panels.png   — 森林面板（误差棒 = CI；混合单位按行最大值归一，
                          与 Test1/Test2 同一裁定）
  analysis_summary.json — V vs VT 配对差 CI 与判定（误差类指标低好，
                          contact_f1 高好）

用法：--target NAME=GROUP:PATH（可重复；GROUP ∈ {VT2M,V2M,T2M}）
     --pair VT2M:V2M    同一明细文件内两组的会话行按 session_id 配对
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anysole.utils.metrics import cluster_bootstrap_ci, paired_cluster_bootstrap
from utils.mpl_fonts import setup_cjk_fonts

setup_cjk_fonts()
# 与 component_analysis.py 同一裁定：图内文字全英文、DejaVu Sans 粗体。
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.weight"] = "bold"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
V_COLOR = "#2a78d6"
V_LIGHT = "#6ea8e8"
T_COLOR = "#eb6834"
T_LIGHT = "#f2a37d"
MAIN_COLOR = "#1baf7a"
RESID_COLOR = "#0f7a52"

DPI = 200

# (display key in the pooled detail, snake_case key in the session dump, unit label)
METRIC_PAIRS = [
    ("PA-MPJPE", "pa_mpjpe_mm", "mm"),
    ("MPJPE", "mpjpe_mm", "mm"),
    ("GlobalMPJPE", "global_mpjpe_mm", "mm"),
    ("W-MPJPE100", "w_mpjpe100_mm", "mm"),
    ("WA-MPJPE100", "wa_mpjpe100_mm", "mm"),
    ("traj_ATE", "root_ate_mm", "mm"),
    ("root_rte_percent", "root_rte_percent", "%"),
    ("RootOrientation", "root_orientation_deg", "deg"),
    ("RootOrientationDrift", "root_orientation_drift_deg", "deg"),
    ("AccelError", "accel_error_m_s2", "m/s^2"),
    ("Jitter_pred", "jitter_pred_m_s3", "m/s^3"),
    ("Jitter_gt", "jitter_gt_m_s3", "m/s^3"),
    ("FootSliding", "foot_sliding_mm", "mm"),
    ("foot_sliding_vertex_mm", "foot_sliding_vertex_mm", "mm"),
    ("contact_f1", "contact_f1", "F1"),
    ("seam_jump_mm", "seam_jump_mm", "mm"),
    ("GroundPen", "ground_pen_mm", "mm"),
    ("GroundPenFrac", "ground_pen_frac", "frac"),
]
HIGHER_IS_BETTER = {"contact_f1"}


def _fmt(value: float, unit: str) -> str:
    if unit == "frac":
        return "%.3f" % value
    if unit == "F1":
        return "%.3f" % value
    return "%.1f" % value


class Target:
    def __init__(self, name: str, group: str, detail_path: Path):
        self.name = name
        self.group = group
        self.detail_path = detail_path
        if not detail_path.is_file():
            raise FileNotFoundError("detail json missing: %s" % detail_path)
        self.detail = json.loads(detail_path.read_text(encoding="utf-8"))
        groups = self.detail.get("metrics", {})
        if group not in groups:
            raise ValueError("%s: no %s group (have %s)"
                             % (detail_path, group, ", ".join(groups)))
        self.pooled = groups[group]
        self.sessions: list[dict[str, Any]] = []
        session_path = detail_path.with_name("session_detail_%s.json" % detail_path.stem)
        if session_path.is_file():
            doc = json.loads(session_path.read_text(encoding="utf-8"))
            cond = doc.get("conditions", {}).get(group, {})
            self.sessions = cond.get("sessions", [])
        if not self.sessions:
            print("[warn] %s: no session rows at %s — CI unavailable" % (name, session_path))

    def session_values(self, key: str) -> np.ndarray:
        return np.asarray([row["metrics"].get(key, np.nan) for row in self.sessions],
                          dtype=np.float64)

    def pooled_value(self, key: str) -> float:
        return float(self.pooled.get(key, np.nan))


def parse_targets(raw: list[str]) -> list[Target]:
    targets: list[Target] = []
    for item in raw:
        if "=" not in item or ":" not in item:
            raise ValueError("--target expects NAME=GROUP:PATH, got %r" % item)
        name, rest = item.split("=", 1)
        group, path = rest.split(":", 1)
        targets.append(Target(name, group, Path(path)))
    return targets


def paired_rows(a: Target, b: Target) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Session rows paired by session_id for every metric."""
    ids = {row["session_id"]: row for row in a.sessions}
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    for row in b.sessions:
        sid = row["session_id"]
        if sid not in ids:
            continue
        for _, key, _unit in METRIC_PAIRS:
            va = ids[sid]["metrics"].get(key)
            vb = row["metrics"].get(key)
            if va is None or vb is None:
                continue
            pairs.setdefault(key, ([], []))
            pairs[key][0].append(float(va))
            pairs[key][1].append(float(vb))
    return {key: (np.asarray(x), np.asarray(y)) for key, (x, y) in pairs.items()}


def _draw_panel(rows: list[dict[str, Any]], series: list[str], colors: dict[str, str],
                path: Path) -> None:
    """经典森林图 + CI 误差棒：混合单位按行最大值归一（Test1/Test2 裁定）。"""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13.0, 0.42 * n + 1.2), dpi=DPI)
    refs = {row["row_id"]: row["ref"] for row in rows}
    y_positions = list(reversed(range(n)))
    for y, row in zip(y_positions, rows):
        ref = refs[row["row_id"]]
        for name in series:
            value = row["values"].get(name)
            if value is None or not np.isfinite(value):
                continue
            ci_lo = row["ci_lo"].get(name)
            ci_hi = row["ci_hi"].get(name)
            x = value / ref
            offset = 0.016 * series.index(name)
            color = colors[name]
            ax.hlines(y + offset, 0.0, x, color=color, linewidth=2.0, zorder=3)
            if ci_lo is not None and ci_hi is not None and np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ax.hlines(y + offset, ci_lo / ref, ci_hi / ref, color=color,
                          linewidth=5.0, alpha=0.25, zorder=2)
            ax.scatter([x], [y + offset], s=26, color=color, edgecolors="white",
                       linewidths=1.4, zorder=4)
            ax.text(x, y + offset, " " + _fmt(value, row["unit"]),
                    va="bottom", ha="left", fontsize=10, color=INK, zorder=6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=11, color=INK)
    ax.set_xlim(0.0, 1.18)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=1.0, zorder=0)
    handles = [plt.Line2D([0], [0], color=colors[name], linewidth=2.0, label=name)
               for name in series]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True,
                        help="NAME=GROUP:PATH (repeatable)")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test4_PhysicalTest" / "V3_3B")
    args = parser.parse_args(argv)
    targets = parse_targets(args.target)
    series = [t.name for t in targets]
    colors = {
        "Main VT": MAIN_COLOR, "Main V": V_COLOR, "Main T": T_COLOR,
        "V specialist": V_LIGHT, "T specialist": T_LIGHT,
        "Residual VT": RESID_COLOR,
    }
    for name in series:
        colors.setdefault(name, MUTED)

    # 每行（指标）：各目标均值 + CI；行最大值归一。
    panel_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for display, key, unit in METRIC_PAIRS:
        values: dict[str, float] = {}
        ci_lo: dict[str, float] = {}
        ci_hi: dict[str, float] = {}
        table_row: dict[str, Any] = {"metric": display, "unit": unit}
        for target in targets:
            sv = target.session_values(key)
            mean, lo, hi = cluster_bootstrap_ci(sv)
            table_row[target.name] = mean
            table_row["%s_ci_lo" % target.name] = lo
            table_row["%s_ci_hi" % target.name] = hi
            values[target.name] = mean
            ci_lo[target.name] = lo
            ci_hi[target.name] = hi
        finite = [v for v in values.values() if v is not None and np.isfinite(v)]
        ref = max(finite) if finite else 1.0
        panel_rows.append({"row_id": display, "label": "%s (%s)" % (display, unit),
                           "unit": unit, "values": values,
                           "ci_lo": ci_lo, "ci_hi": ci_hi, "ref": ref if ref > 0 else 1.0})
        table_rows.append(table_row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "physical_table.csv").write_text(_csv(table_rows), encoding="utf-8")
    _draw_panel(panel_rows, series, colors, args.out / "physical_panels.png")

    ci_payload = {
        "split": args.split,
        "targets": {t.name: {"group": t.group, "path": str(t.detail_path),
                             "n_sessions": len(t.sessions)} for t in targets},
        "metrics": {row["metric"]: {k: v for k, v in row.items() if k != "metric"}
                    for row in table_rows},
    }
    write_json(args.out / "physical_ci.json", ci_payload)

    # V vs VT 配对差 CI（同文件两组，按 session_id 配对）
    verdicts: dict[str, Any] = {}
    for name_vt, name_v in (("Main VT", "Main V"), ("Residual VT", "V specialist")):
        t_vt = next((t for t in targets if t.name == name_vt), None)
        t_v = next((t for t in targets if t.name == name_v), None)
        if t_vt is None or t_v is None:
            continue
        pairs = paired_rows(t_v, t_vt)
        for display, key, _unit in METRIC_PAIRS:
            if key not in pairs:
                continue
            a, b = pairs[key]  # a = V, b = VT
            if a.size < 5:
                continue
            delta, lo, hi = paired_cluster_bootstrap(b, a)
            sign = 1.0 if display in HIGHER_IS_BETTER else -1.0
            # 误差类指标：VT 差 = delta>0；contact_f1 反之。
            worse = (delta * sign) > 0.0 and lo * sign > 0.0
            better = (delta * sign) < 0.0 and hi * sign < 0.0
            verdicts.setdefault("%s_vs_%s" % (name_vt, name_v), {})[display] = {
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
                "verdict": "VT worse" if worse else ("VT better" if better else "within CI"),
            }
    write_json(args.out / "analysis_summary.json", {"split": args.split, "paired": verdicts})
    print("wrote %s (%d metrics x %d targets)" % (args.out, len(table_rows), len(targets)))
    return 0


def _csv(rows: list[dict[str, Any]]) -> str:
    import io
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("%.4f" % v) if isinstance(v, float) else v
                         for k, v in row.items()})
    return handle.getvalue()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
                    encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
