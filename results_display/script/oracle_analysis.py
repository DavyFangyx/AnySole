#!/usr/bin/env python3
"""T 判定后续 · 实验 3：oracle 互补上限 + 难例分层（主线同网络 V vs VT）。

读取 `python -m anysole.eval --dump-detail` 产出的逐窗口明细
（window_detail_<split>.json，snake_case 键），按 (session_id, frame_start)
配对 V2M / VT2M 两条件，逐区域计算：

  V_mean / VT_mean   — 两条件逐窗口均值（pooled 口径）
  oracle_mean        — mean_w min(err_V, err_VT)：完美开关下的互补上限
  oracle_gain        — V_mean − oracle_mean（可实现的增益上限）
  vt_win_frac        — VT 胜出的窗口占比（平局计半）

另有 per-session oracle 变体（先按会话聚合再取 min），以及按 overall V
误差十等分的 decile 曲线（Overall / Ankle-foot / Upper 三面板）。

输出：oracle_table.csv / oracle_deciles.png / oracle_summary.json /
analysis_summary.json（headline：oracle gain 与 VT 胜率）。
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

from anysole.types import JOINT_HIERARCHY
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
T_COLOR = "#eb6834"
MAIN_COLOR = "#1baf7a"

DPI = 200


def _regions() -> list[tuple[str, str, str]]:
    """(row_id, window key, label) — overall + the JOINT_HIERARCHY levels."""
    rows = [("overall", "pa_mpjpe_mm", "Overall")]
    for name in JOINT_HIERARCHY:
        if name == "overall":
            continue
        rows.append((name, "PA-MPJPE_%s" % name, name))
    return rows


def _load_condition(path: Path, condition: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("window detail missing: %s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    conditions = data.get("conditions", {})
    if condition not in conditions:
        raise ValueError("%s: no %s condition (have %s)"
                         % (path, condition, ", ".join(conditions)))
    return conditions[condition]


def _paired_windows(v_cond: dict[str, Any], vt_cond: dict[str, Any]) -> dict[str, Any]:
    """Pair windows by (session_id, frame_start); returns {key: (v, vt)} per
    metric key plus the joined id arrays."""
    def index(cond: dict[str, Any]) -> dict[tuple[str, int], int]:
        return {
            (str(s), int(f)): i
            for i, (s, f) in enumerate(zip(cond["session_ids"], cond["frame_starts"]))
        }
    v_idx, vt_idx = index(v_cond), index(vt_cond)
    keys = [k for k in v_idx if k in vt_idx]
    if not keys:
        raise ValueError("no matching (session_id, frame_start) windows between V2M and VT2M")
    return {"keys": keys, "v": v_cond, "vt": vt_cond,
            "v_idx": v_idx, "vt_idx": vt_idx}


def _win_series(paired: dict[str, Any], metric: str) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(paired["v"]["metrics"][metric], dtype=np.float64)
    vt = np.asarray(paired["vt"]["metrics"][metric], dtype=np.float64)
    v_sel = [paired["v_idx"][k] for k in paired["keys"]]
    vt_sel = [paired["vt_idx"][k] for k in paired["keys"]]
    return v[v_sel], vt[vt_sel]


def _region_stats(v: np.ndarray, vt: np.ndarray) -> dict[str, float]:
    oracle = np.minimum(v, vt)
    wins = (vt < v).astype(np.float64) + 0.5 * (vt == v).astype(np.float64)
    return {
        "v_mean": float(v.mean()),
        "vt_mean": float(vt.mean()),
        "oracle_mean": float(oracle.mean()),
        "oracle_gain": float(v.mean() - oracle.mean()),
        "vt_win_frac": float(wins.mean()),
    }


def _session_stats(paired: dict[str, Any], metric: str) -> dict[str, float]:
    """Session-level oracle: mean per session first, then min/win over sessions."""
    v, vt = _win_series(paired, metric)
    session_ids = np.asarray([paired["keys"][i][0] for i in range(len(paired["keys"]))])
    sess_v: dict[str, list[float]] = {}
    sess_vt: dict[str, list[float]] = {}
    for s, a, b in zip(session_ids, v, vt):
        sess_v.setdefault(str(s), []).append(float(a))
        sess_vt.setdefault(str(s), []).append(float(b))
    v_mean = np.asarray([float(np.mean(sess_v[s])) for s in sess_v])
    vt_mean = np.asarray([float(np.mean(sess_vt[s])) for s in sess_v])
    oracle = np.minimum(v_mean, vt_mean)
    wins = (vt_mean < v_mean).astype(np.float64) + 0.5 * (vt_mean == v_mean).astype(np.float64)
    return {
        "session_v_mean": float(v_mean.mean()),
        "session_vt_mean": float(vt_mean.mean()),
        "session_oracle_mean": float(oracle.mean()),
        "session_oracle_gain": float(v_mean.mean() - oracle.mean()),
        "session_vt_win_frac": float(wins.mean()),
        "n_sessions": int(v_mean.size),
    }


def _deciles(paired: dict[str, Any]) -> list[dict[str, Any]]:
    """Ten equal bins ordered by overall V error; per bin the V/VT/oracle means
    for overall, anklefoot and upper."""
    v_overall, _ = _win_series(paired, "pa_mpjpe_mm")
    order = np.argsort(v_overall, kind="stable")
    bins = np.array_split(order, 10)
    tracks = [
        ("Overall", "pa_mpjpe_mm"),
        ("Ankle-foot", "PA-MPJPE_anklefoot"),
        ("Upper", "PA-MPJPE_upper"),
    ]
    rows: list[dict[str, Any]] = []
    for bi, idx in enumerate(bins, start=1):
        row: dict[str, Any] = {"decile": bi, "n_windows": int(idx.size)}
        for label, key in tracks:
            v, vt = _win_series(paired, key)
            v_b, vt_b = v[idx], vt[idx]
            row["%s_v" % label] = float(v_b.mean())
            row["%s_vt" % label] = float(vt_b.mean())
            row["%s_oracle" % label] = float(np.minimum(v_b, vt_b).mean())
        rows.append(row)
    return rows


def _draw_deciles(rows: list[dict[str, Any]], path: Path) -> None:
    deciles = [row["decile"] for row in rows]
    panels = [("Overall", V_COLOR, T_COLOR, MAIN_COLOR),
              ("Ankle-foot", V_COLOR, T_COLOR, MAIN_COLOR),
              ("Upper", V_COLOR, T_COLOR, MAIN_COLOR)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=DPI)
    for ax, (label, c_v, c_vt, c_or) in zip(axes, panels):
        v = [row["%s_v" % label] for row in rows]
        vt = [row["%s_vt" % label] for row in rows]
        oracle = [row["%s_oracle" % label] for row in rows]
        for y, c, name in ((v, c_v, "V-only"), (vt, c_vt, "V+T"), (oracle, c_or, "Oracle")):
            ax.plot(deciles, y, marker="o", markersize=5, linewidth=2.0,
                    color=c, label=name, zorder=3)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.set_xticks(deciles)
        ax.set_xlabel("V-error decile (1 = easiest windows)", color=INK)
        ax.set_ylabel("PA-MPJPE (mm)", color=INK)
        ax.tick_params(colors=INK2)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(label, color=INK, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2m-detail", type=Path, required=True,
                        help="mainline V3_3B window_detail_<split>.json")
    parser.add_argument("--vt2m-detail", type=Path, required=True,
                        help="mainline V3_3B window_detail_<split>.json")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "Test5_OracleTest" / "V3_3B")
    args = parser.parse_args(argv)

    v_cond = _load_condition(args.v2m_detail, "V2M")
    vt_cond = _load_condition(args.vt2m_detail, "VT2M")
    paired = _paired_windows(v_cond, vt_cond)
    n_windows = len(paired["keys"])

    table_rows: list[dict[str, Any]] = []
    for row_id, key, label in _regions():
        v, vt = _win_series(paired, key)
        row = {"row_id": row_id, "label": label, "metric": key}
        row.update(_region_stats(v, vt))
        row.update(_session_stats(paired, key))
        table_rows.append(row)

    decile_rows = _deciles(paired)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oracle_table.csv").write_text(_csv(table_rows), encoding="utf-8")
    (args.out / "oracle_deciles.csv").write_text(_csv(decile_rows), encoding="utf-8")
    _draw_deciles(decile_rows, args.out / "oracle_deciles.png")

    overall = table_rows[0]
    summary = {
        "split": args.split,
        "n_windows": n_windows,
        "n_sessions": overall["n_sessions"],
        "overall": {k: overall[k] for k in
                    ("v_mean", "vt_mean", "oracle_mean", "oracle_gain", "vt_win_frac")},
        "headline": ("oracle gain %.2fmm (perfect-switch ceiling); VT beats V on "
                     "%.1f%% of windows" % (overall["oracle_gain"],
                                           100.0 * overall["vt_win_frac"])),
    }
    write_json(args.out / "oracle_summary.json", summary)
    verdict = {"oracle_gain_positive": overall["oracle_gain"] > 0.0,
               "vt_win_frac_above_half": overall["vt_win_frac"] > 0.5}
    write_json(args.out / "analysis_summary.json", {**verdict, "split": args.split})
    print("wrote %s (n_windows=%d, overall oracle gain %.2fmm, VT win %.1f%%)"
          % (args.out, n_windows, overall["oracle_gain"], 100.0 * overall["vt_win_frac"]))
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
