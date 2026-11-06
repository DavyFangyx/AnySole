#!/usr/bin/env python3
"""E6 门控 × 遮挡（2026-11-05）：门随遮挡是否打开（行为侧）。

对每个遮挡档 occ000–occ100，用同一分解口径（E6 门控两张主表）比较
Cv7（无门）与 Cv5（双门）：

    Δ_s = VT_s − V_s（正 = 触觉引入退化）
    ΔE⁺ = Σ w_s max(Δ_s, 0) / Σ w_s   （有害分量）
    ΔE⁻ = Σ w_s min(Δ_s, 0) / Σ w_s   （有益分量）

门控效应（每档）：
    抑制率   = 1 − ΔE⁺_Cv5 / ΔE⁺_Cv7
    保留率   = ΔE⁻_Cv5 / ΔE⁻_Cv7
    ΔE 净值  = ΔE_Cv5 − ΔE_Cv7（附加参照，不是纯门效应）

数据源：results/cells/<cell>/metrics/session_detail_val.json（act8020 无 test）。
输出：results_display/Etest_act8020/E6_ContactPhaseAnalysis/gate_occlusion/
  gate_occlusion_metrics.csv   长表：档 × 模型 × 指标 × {net,plus,minus}
  gate_occlusion_summary.csv   宽表：档 × 关键指标 × 两模型分量 + 抑制/保留率
  gate_occlusion_pa_mpjpe.csv  论文用精简表（PA-MPJPE 一列到底）

用法（base env 即可）：python results_display/script/e6_gate_occlusion.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "cells"
OUT = (REPO / "results_display" / "Etest_act8020"
       / "E6_ContactPhaseAnalysis" / "gate_occlusion")

LEVELS = ("000", "010", "020", "030", "040", "050", "060", "070", "080",
          "090", "100")
CELLS_BY_MODEL = {
    "V7_Cv7": "V7_Cv7_act8020_s1",
    "V7_Cv5": "V7_Cv5_act8020_s1",
}
METRICS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "W-MPJPE100": "w_mpjpe100_mm",
    "MPJRE": "mpjae_deg",
    "traj_ATE@0.5s": "root_ate_seg20_mm",
    "RootOrientationDrift": "root_orientation_drift_deg",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}
KEY_METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "traj_ATE@0.5s",
               "FootSliding", "GroundPen", "Jitter_pred")
MIN_EFFECT = 0.05


def cell_id(base: str, level: str) -> str:
    return base if level == "000" else "%s_evalocc%s" % (base, level)


def decompose(cell: str, split: str = "val") -> dict:
    path = CELLS / cell / "metrics" / ("session_detail_%s.json" % split)
    detail = json.loads(path.read_text(encoding="utf-8"))["conditions"]
    vt = {e["session_id"]: e["metrics"] for e in detail["VT2M"]}
    v = {e["session_id"]: e["metrics"] for e in detail["V2M"]}
    n_frames = {e["session_id"]: e["n_frames"] for e in detail["V2M"]}
    sids = sorted(set(v) & set(vt))
    out = {}
    for name, key in METRICS.items():
        delta = np.array([vt[s][key] - v[s][key] for s in sids], dtype=float)
        w = np.array([float(n_frames[s]) for s in sids])
        total = w.sum()
        out[name] = {
            "net": float((w * delta).sum() / total),
            "plus": float((w * np.maximum(delta, 0.0)).sum() / total),
            "minus": float((w * np.minimum(delta, 0.0)).sum() / total),
        }
    return out


def pct(num: float, den: float, signed: bool = False) -> str:
    if abs(den) < MIN_EFFECT:
        return "—"
    if signed:
        return "%+.1f%%" % (100.0 * num / den)
    return "%.1f%%" % (100.0 * num / den)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}
HET_LEVELS = ("000", "090", "100")


def export_heterogeneity(out_dir: Path) -> None:
    """门值异质性（2026-11-05）：按动作/会话检查“门不开”是否被平均掩盖。

    同时给出会话级 gate 变化与触觉收益缺口的相关系数（Cv5 净 ΔE − Cv7 净 ΔE；
    正 = Cv5 拿到更少收益，负相关 = 根门开得越多缺口越小——弱证据）。
    """
    gate_json = out_dir / "gate_values_occlusion.json"
    if not gate_json.is_file():
        print("[skip] heterogeneity: 缺 %s" % gate_json)
        return
    levels = json.loads(gate_json.read_text(encoding="utf-8"))["levels"]
    actions = {}
    manifest = (REPO / "AnysoleWorkspace/protocol/manifests/"
                "session_manifest.csv")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            actions[row["session_id"]] = ACTION_GROUPS.get(row["action"],
                                                           row["action"])
    sids = sorted(levels["000"])
    groups: dict[str, list[str]] = {}
    for sid in sids:
        groups.setdefault(actions.get(sid, "?"), []).append(sid)

    def mean_gate(level: str, key: str, subset: list[str]) -> float:
        return float(np.mean([levels[level][s][key] for s in subset]))

    rows = []
    for group, subset in sorted(groups.items()):
        rows.append([
            group, len(subset),
            "%.3f" % mean_gate("000", "vel", subset),
            "%.3f" % mean_gate("090", "vel", subset),
            "%.3f" % mean_gate("100", "vel", subset),
            "%.3f" % mean_gate("000", "other", subset),
            "%.3f" % mean_gate("090", "other", subset),
            "%.3f" % mean_gate("100", "other", subset),
            "%+.3f" % (mean_gate("090", "vel", subset)
                       - mean_gate("000", "vel", subset)),
        ])
    write_csv(out_dir / "gate_occlusion_heterogeneity.csv",
              ["action", "n", "root_000", "root_090", "root_100",
               "pose_000", "pose_090", "pose_100", "d_root_090_000"], rows)

    def session_delta(level: str, base: str) -> dict:
        cell = cell_id(base, level)
        detail = json.loads(
            (CELLS / cell / "metrics" / "session_detail_val.json")
            .read_text(encoding="utf-8"))["conditions"]
        vt = {e["session_id"]: e["metrics"]["pa_mpjpe_mm"]
              for e in detail["VT2M"]}
        v = {e["session_id"]: e["metrics"]["pa_mpjpe_mm"]
             for e in detail["V2M"]}
        return {sid: vt[sid] - v[sid] for sid in vt}

    stats = {}
    for level in ("090", "100"):
        c5 = session_delta(level, CELLS_BY_MODEL["V7_Cv5"])
        c7 = session_delta(level, CELLS_BY_MODEL["V7_Cv7"])
        common = sorted(set(c5) & set(c7))
        gap = np.array([c5[s] - c7[s] for s in common])
        d_root = np.array([levels[level][s]["vel"] - levels["000"][s]["vel"]
                           for s in common])
        d_pose = np.array([levels[level][s]["other"] - levels["000"][s]["other"]
                           for s in common])
        stats["occ" + level] = {
            "n": len(common),
            "mean_gap_mm": float(gap.mean()),
            "corr_root_gate_change_gap": float(np.corrcoef(d_root, gap)[0, 1]),
            "corr_pose_gate_change_gap": float(np.corrcoef(d_pose, gap)[0, 1]),
            "n_root_gate_gt_0.5": int(sum(
                levels[level][s]["vel"] > 0.5 for s in common)),
            "n_pose_gate_gt_0.5": int(sum(
                levels[level][s]["other"] > 0.5 for s in common)),
        }
    (out_dir / "gate_occlusion_heterogeneity.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("[json] %s" % (out_dir / "gate_occlusion_heterogeneity.json"))


def export_figure(out_dir: Path) -> None:
    """三面板：行为侧 ΔE 净值 / 抑制率 + 模型侧门值（x = 遮挡 %）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _read_csv(out_dir / "gate_occlusion_summary.csv")
    gate_path = out_dir / "gate_values_occlusion_summary.csv"
    gates = _read_csv(gate_path) if gate_path.is_file() else []
    xs = [int(r["occ"]) for r in summary]
    color_cv5, color_cv7, color_alt = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

    ax = axes[0]
    ax.plot(xs, [float(r["PA-MPJPE_Cv5_net"]) for r in summary], color=color_cv5,
            lw=2.2, marker="o", ms=5, label="Cv5 (double gate)")
    ax.plot(xs, [float(r["PA-MPJPE_Cv7_net"]) for r in summary], color=color_cv7,
            lw=2.2, ls="--", marker="s", ms=5, label="Cv7 (no gate)")
    ax.axhline(0.0, color="#c8c8c8", lw=0.9, ls=":", zorder=0)
    ax.set_xlabel("Occlusion (%)", fontsize=10)
    ax.set_ylabel("PA-MPJPE ΔE = VT − V (mm)", fontsize=10)
    ax.set_title("Tactile effect (lower = tactile helps)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="best")

    ax = axes[1]
    supp = [float(r["PA-MPJPE_suppression"].rstrip("%"))
            if r["PA-MPJPE_suppression"] != "—" else float("nan")
            for r in summary]
    ax.plot(xs, supp, color=color_cv5, lw=2.2, marker="o", ms=5)
    ax.set_xlabel("Occlusion (%)", fontsize=10)
    ax.set_ylabel("Harm suppression vs no-gate (%)", fontsize=10)
    ax.set_title("Cv5 gate harm suppression (PA-MPJPE)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_ylim(0, 105)
    ax.set_xlim(-2, 102)
    ax.text(0.98, 0.04, "undefined where no harmful component (occ≥50)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#52514e")

    ax = axes[2]
    if gates:
        by_occ = {}
        for row in gates:
            by_occ.setdefault(row["occ"], {})[row["group"]] = float(row["mean"])
        lx = [int(o) for o in sorted(by_occ)]
        ax.plot(lx, [by_occ[o]["vel"] for o in sorted(by_occ)], color=color_cv5,
                lw=2.2, marker="o", ms=5, label="root gate")
        ax.plot(lx, [by_occ[o]["other"] for o in sorted(by_occ)], color=color_alt,
                lw=2.2, marker="s", ms=5, label="pose gate")
        ax.axhline(0.5, color="#c8c8c8", lw=0.9, ls=":", zorder=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean gate value g (sigmoid)", fontsize=10)
        ax.set_title("Cv5 tactile gate vs occlusion", fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "gate values not available", ha="center",
                va="center", transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
    ax.set_xlabel("Occlusion (%)", fontsize=10)

    fig.tight_layout()
    out = out_dir / "Full_Fig_E6_GateOcclusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[png] %s" % out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir

    data = {}
    for level in LEVELS:
        for model, base in CELLS_BY_MODEL.items():
            cid = cell_id(base, level)
            path = CELLS / cid / "metrics" / ("session_detail_%s.json" % args.split)
            if not path.is_file():
                print("[skip] %s: 缺 %s" % (cid, path))
                continue
            data.setdefault(level, {})[model] = decompose(cid, args.split)
        if level in data and len(data[level]) < 2:
            print("[warn] occ%s 只有 %d 个模型" % (level, len(data[level])))

    # 长表
    rows_long = []
    for level in LEVELS:
        for model in CELLS_BY_MODEL:
            comp = data.get(level, {}).get(model)
            if comp is None:
                continue
            for metric in METRICS:
                c = comp[metric]
                rows_long.append([level, model, metric,
                                  "%.4f" % c["net"], "%.4f" % c["plus"],
                                  "%.4f" % c["minus"]])
    write_csv(out_dir / "gate_occlusion_metrics.csv",
              ["occ", "model", "metric", "net", "plus", "minus"], rows_long)

    # 宽表：关键指标 × 两模型分量 + 抑制/保留率
    header = ["occ"]
    for metric in KEY_METRICS:
        header += ["%s_Cv7_plus" % metric, "%s_Cv5_plus" % metric,
                   "%s_suppression" % metric,
                   "%s_Cv7_minus" % metric, "%s_Cv5_minus" % metric,
                   "%s_preservation" % metric,
                   "%s_Cv5_net" % metric, "%s_Cv7_net" % metric]
    rows = []
    for level in LEVELS:
        d = data.get(level, {})
        if not all(m in d for m in CELLS_BY_MODEL):
            continue
        row = [level]
        for metric in KEY_METRICS:
            n = d["V7_Cv7"][metric]
            g = d["V7_Cv5"][metric]
            row += ["%.3f" % n["plus"], "%.3f" % g["plus"],
                    pct(n["plus"] - g["plus"], n["plus"]),
                    "%+.3f" % n["minus"], "%+.3f" % g["minus"],
                    pct(g["minus"], n["minus"], signed=True),
                    "%+.3f" % g["net"], "%+.3f" % n["net"]]
        rows.append(row)
    write_csv(out_dir / "gate_occlusion_summary.csv", header, rows)

    # 精简：PA-MPJPE
    header2 = ["occ", "Cv7_plus", "Cv5_plus", "suppression",
               "Cv7_minus", "Cv5_minus", "preservation",
               "Cv7_net", "Cv5_net"]
    rows2 = []
    for level in LEVELS:
        d = data.get(level, {})
        if not all(m in d for m in CELLS_BY_MODEL):
            continue
        n = d["V7_Cv7"]["PA-MPJPE"]
        g = d["V7_Cv5"]["PA-MPJPE"]
        rows2.append([level, "%.3f" % n["plus"], "%.3f" % g["plus"],
                      pct(n["plus"] - g["plus"], n["plus"]),
                      "%+.3f" % n["minus"], "%+.3f" % g["minus"],
                      pct(g["minus"], n["minus"], signed=True),
                      "%+.3f" % n["net"], "%+.3f" % g["net"]])
    write_csv(out_dir / "gate_occlusion_pa_mpjpe.csv", header2, rows2)
    export_heterogeneity(out_dir)
    if not args.no_figure:
        export_figure(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
