#!/usr/bin/env python3
"""E6 正式配对统计（V7_C门控修改方案 §9，2026-10-31）。

判读单位 = 36 条会话（先逐侧、再同会话左右侧等权合并为一个会话结果）；
不把双脚、帧、窗口当作额外独立会话。

逐帧配对（同会话、同侧、同 frame_id；有效掩码取候选与参照的交集）：
    Δ_i = VT_i − V_i（mm，骨盆对齐踝足 MPJPE）
    退化量 H = mean(max(Δ, 0))、收益量 A = mean(max(−Δ, 0))、
    绝对误差 E = mean(VT_i)
候选 vs 参照（B = V7_Cv5_s1，ckpt_last）：
    dE_s = E_M,s − E_R,s（负有利）、dH_s（负有利）、dA_s（正有利）
每指标：胜/平/负（平局容忍带 |d| < 0.05 mm，test 前预注册）、配对差值
中位数与四分位范围、会话等权均值差、联合通过（E 改善且 H 不恶化、
A 不下降）会话数、会话配对 bootstrap 95% CI（mean 与 median，10000 次）。
相位级：同（会话,侧,相位）配对差值 + 胜/平/负；相位缺席的会话留空不补零，
报告实际参与会话数。

用法（touch_gait env，仓库根）：
  python results_display/script/e6_paired_stats.py
产物（results_display/Etest/E6_ContactPhaseAnalysis/）：
  e6_paired_stats.json / E6_配对统计.md / e6_paired_sessions.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_note/HMR2.0/hmr2_gait"))
from anysole_tta.e6_phase_analysis import build_session_sets, load_session  # noqa: E402
from anysole_tta.phase_common import PHASE_NAMES, PHASE_NAMES_ZH  # noqa: E402

OUT = REPO / "results_display/Etest/E6_ContactPhaseAnalysis"
TIE_TOL = 0.05          # mm，平局容忍带（预注册）
N_BOOT = 10000          # 会话配对 bootstrap 次数
REF = "V7_Cv5_s1"       # B 基线（双门 + 枚举，ckpt_last）
CANDS = ("V7_Cv5T_s1", "V7_Cv5LR_s1", "V7_Cv5TLR_s1")   # 本轮候选
AUX = ("V7_Cv4_s1", "V7_Cv7_s1")                        # 单门/无门辅助参照


def per_session_metrics(data: dict, mask: np.ndarray) -> dict:
    """一条会话一模型：逐侧 H/A/E → 左右等权合并（会话单位）。"""
    side_vals = {}
    for foot in ("left", "right"):
        v = np.asarray(data["V2M_" + foot], dtype=np.float64)
        vt = np.asarray(data["VT2M_" + foot], dtype=np.float64)
        m = mask & np.isfinite(v) & np.isfinite(vt)
        if not m.any():
            side_vals[foot] = None
            continue
        d = vt[m] - v[m]
        side_vals[foot] = {
            "H": float(np.maximum(d, 0.0).mean()),
            "A": float(np.maximum(-d, 0.0).mean()),
            "E": float(vt[m].mean()),
            "V": float(v[m].mean()),
        }
    ok = [s for s in side_vals.values() if s is not None]
    if not ok:
        return {"H": None, "A": None, "E": None, "V": None,
                "n_frames": int(mask.sum())}
    merged = {k: float(np.mean([s[k] for s in ok]))
              for k in ("H", "A", "E", "V")}
    merged["n_frames"] = int(mask.sum())
    return merged


def per_phase_metrics(data: dict, mask: np.ndarray) -> dict:
    """{(side, phase): {H,A,E}}——只保留该相位存在的组合。"""
    out = {}
    for side_i, foot in enumerate(("left", "right")):
        v = np.asarray(data["V2M_" + foot], dtype=np.float64)
        vt = np.asarray(data["VT2M_" + foot], dtype=np.float64)
        for ph in range(4):
            m = mask & (np.asarray(data["phase"])[:, side_i] == ph) \
                & np.isfinite(v) & np.isfinite(vt)
            if not m.any():
                continue
            d = vt[m] - v[m]
            out[(side_i, ph)] = {
                "H": float(np.maximum(d, 0.0).mean()),
                "A": float(np.maximum(-d, 0.0).mean()),
                "E": float(vt[m].mean()),
            }
    return out


def wdl(diffs: np.ndarray, higher_better: bool = False) -> tuple[int, int, int]:
    """胜/平/负。dE/dH 负有利（higher_better=False），dA 正有利。"""
    wins = (diffs > TIE_TOL) if higher_better else (diffs < -TIE_TOL)
    ties = np.abs(diffs) <= TIE_TOL
    losses = (diffs < -TIE_TOL) if higher_better else (diffs > TIE_TOL)
    return int(wins.sum()), int(ties.sum()), int(losses.sum())


def boot_ci(diffs: np.ndarray) -> dict:
    """会话配对 bootstrap 95% CI（mean / median 百分位区间）。"""
    rng = np.random.default_rng(7)
    n = len(diffs)
    means, medians = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        s = diffs[idx]
        means.append(s.mean())
        medians.append(np.median(s))
    return {
        "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "median_ci": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
    }


def paired_summary(diffs: dict[str, np.ndarray]) -> dict:
    """每指标独立统计；dE/dH/dA 带胜/平/负，dV 只报分布（V-only 基线漂移
    诊断，无有利方向）。bootstrap CI 逐指标分别计算（2026-10-31 修正：
    此前只算 dE 一份并复用）。"""
    out = {}
    for metric, d in diffs.items():
        w, t, l = wdl(d, higher_better=(metric == "dA"))
        out[metric] = {
            "wins": w, "ties": t, "losses": l,
            "median": float(np.median(d)),
            "iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))],
            "mean": float(d.mean()),
            "boot": boot_ci(d),
        }
    joint = np.asarray(diffs["dE"]) < -TIE_TOL
    joint &= np.asarray(diffs["dH"]) <= TIE_TOL
    joint &= np.asarray(diffs["dA"]) >= -TIE_TOL
    out["joint_pass_sessions"] = int(joint.sum())
    out["n_sessions"] = int(len(diffs["dE"]))
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=REF, help="参照模型（默认 V7_Cv5_s1 = B）")
    parser.add_argument("--cands", default=",".join(CANDS + AUX),
                        help="候选模型（逗号分隔；默认 T/LR/TLR + Cv4/Cv7）")
    parser.add_argument("--out-suffix", default="",
                        help="产物文件名后缀（不同参照并行出结果用）")
    args = parser.parse_args()
    ref = args.ref
    cands = tuple(c.strip() for c in args.cands.split(",") if c.strip())
    session_sets = build_session_sets()
    sids = sorted(sids_set(session_sets))
    models = [ref, *cands]
    cache = {}

    def data_of(model: str, sid: str) -> dict:
        key = (model, sid)
        if key not in cache:
            cache[key] = load_session(model, sid)
        return cache[key]

    ref_data = {sid: data_of(ref, sid) for sid in sids}
    results, session_rows = {}, []
    for cand in cands:
        dE, dH, dA, dV, per_sid = [], [], [], [], []
        phase_diffs = {ph: {"dE": [], "dH": [], "dA": [], "sessions": set()}
                       for ph in range(4)}
        for sid in sids:
            r = ref_data[sid]
            c = data_of(cand, sid)
            mask = np.asarray(r["valid"]) & np.asarray(c["valid"])
            rm = per_session_metrics(r, mask)
            cm = per_session_metrics(c, mask)
            if None in (rm["E"], cm["E"]):
                continue
            dE.append(cm["E"] - rm["E"])
            dH.append(cm["H"] - rm["H"])
            dA.append(cm["A"] - rm["A"])
            dV.append(cm["V"] - rm["V"])
            per_sid.append({
                "session": sid, "candidate": cand,
                "dE": round(cm["E"] - rm["E"], 3),
                "dH": round(cm["H"] - rm["H"], 3),
                "dA": round(cm["A"] - rm["A"], 3),
                "dV": round(cm["V"] - rm["V"], 3),
                "E_cand": round(cm["E"], 2), "E_ref": round(rm["E"], 2),
            })
            rp = per_phase_metrics(r, mask)
            cp = per_phase_metrics(c, mask)
            for key in set(rp) & set(cp):
                side, ph = key
                phase_diffs[ph]["dE"].append(cp[key]["E"] - rp[key]["E"])
                phase_diffs[ph]["dH"].append(cp[key]["H"] - rp[key]["H"])
                phase_diffs[ph]["dA"].append(cp[key]["A"] - rp[key]["A"])
                phase_diffs[ph]["sessions"].add(sid)
        diffs = {"dE": np.asarray(dE), "dH": np.asarray(dH),
                 "dA": np.asarray(dA), "dV": np.asarray(dV)}
        results[cand] = paired_summary(diffs)
        for ph, v in phase_diffs.items():
            if not v["dE"]:
                continue
            ps = paired_summary({"dE": np.asarray(v["dE"]),
                                 "dH": np.asarray(v["dH"]),
                                 "dA": np.asarray(v["dA"])})
            ps["n_sessions"] = len(v["sessions"])
            ps["n_pairs"] = len(v["dE"])
            results[cand].setdefault("phase", {})[str(ph)] = ps
        session_rows.extend(per_sid)

    (OUT / ("e6_paired_stats%s.json" % args.out_suffix)).write_text(
        json.dumps({"tie_tol_mm": TIE_TOL, "ref": ref,
                    "models": results, "sessions": session_rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    with open(OUT / ("e6_paired_sessions%s.csv" % args.out_suffix),
              "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session", "candidate", "dE", "dH",
                                          "dA", "dV", "E_cand", "E_ref"])
        w.writeheader()
        w.writerows(session_rows)

    # ---- markdown ----
    lines = ["# E6 正式配对统计（方案 §9 口径）", "",
             "参照 = %s（ckpt_last）；配对单位 = 36 条会话（逐侧计算后左右" % ref,
             "等权合并）；平局容忍带 |d| < %.2f mm；Δ = VT − V（mm）。" % TIE_TOL,
             "dE/dH 负有利、dA 正有利；联合通过 = E 改善且 H 不恶化、A 不下降；",
             "dV = V-only 基线漂移诊断（无有利方向，正值 = 候选 V 基线更差）。", ""]
    for cand in cands:
        r = results[cand]
        lines.append("## %s vs %s" % (cand, ref))
        lines.append("")
        lines.append("| 指标 | 胜/平/负 | 中位数 [IQR] | 均值差 | 联合通过 | 均值 CI95 (bootstrap) |")
        lines.append("|---|---|---|---|---|---|")
        for metric, label in (("dE", "E 差值（VT 误差）"),
                              ("dH", "H 差值（退化量）"),
                              ("dA", "A 差值（收益量）"),
                              ("dV", "dV（V-only 漂移诊断）")):
            m = r[metric]
            ci = m["boot"]["mean_ci"]
            if metric == "dV":
                lines.append("| %s | — | %+.2f [%+.2f, %+.2f] | %+.2f | — | [%+.2f, %+.2f] |"
                             % (label, m["median"], m["iqr"][0], m["iqr"][1],
                                m["mean"], ci[0], ci[1]))
            else:
                lines.append("| %s | %d/%d/%d | %+.2f [%+.2f, %+.2f] | %+.2f | — | [%+.2f, %+.2f] |"
                             % (label, m["wins"], m["ties"], m["losses"],
                                m["median"], m["iqr"][0], m["iqr"][1], m["mean"],
                                ci[0], ci[1]))
        lines.append("| 联合通过（E↑且H不恶化且A不降） | — | — | — | %d/%d | — |"
                     % (r["joint_pass_sessions"], r["n_sessions"]))
        lines.append("")
        lines.append("### %s 相位级（同会话同侧同相位配对；缺席相位不补零）" % cand)
        lines.append("")
        lines.append("| 相位 | 参与会话 | 指标 | 胜/平/负 | 中位数 [IQR] | 均值差 |")
        lines.append("|---|---|---|---|---|---|")
        for ph in sorted(r["phase"], key=int):
            p = r["phase"][ph]
            first = True
            for metric, label in (("dE", "E"), ("dH", "H"), ("dA", "A")):
                m = p[metric]
                lines.append("| %s | %d | %s | %d/%d/%d | %+.2f [%+.2f, %+.2f] | %+.2f |"
                             % (PHASE_NAMES_ZH[int(ph)] if first else "",
                                p["n_sessions"], label, m["wins"], m["ties"],
                                m["losses"], m["median"], m["iqr"][0],
                                m["iqr"][1], m["mean"]))
                first = False
        lines.append("")
    lines.append("附注：walk/run 各仅 1 条会话，相位级结果仅作描述；36 会话区间"
                 "不是跨被试泛化显著性的证明（同被试会话仍有相关性）。")
    (OUT / ("E6_配对统计%s.md" % args.out_suffix)).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s / E6_配对统计%s.md" % (OUT, args.out_suffix))
    for cand in cands:
        r = results[cand]
        print("  %s: E 胜/平/负=%d/%d/%d 中位%+.2f | H %d/%d/%d 中位%+.2f | "
              "A %d/%d/%d 中位%+.2f | dV 中位%+.2f | 联合通过 %d/%d"
              % (cand, r["dE"]["wins"], r["dE"]["ties"], r["dE"]["losses"],
                 r["dE"]["median"], r["dH"]["wins"], r["dH"]["ties"],
                 r["dH"]["losses"], r["dH"]["median"], r["dA"]["wins"],
                 r["dA"]["ties"], r["dA"]["losses"], r["dA"]["median"],
                 r["dV"]["median"], r["joint_pass_sessions"],
                 r["n_sessions"]))
    return 0


def sids_set(session_sets: dict[str, list[str]]) -> set[str]:
    return set(sum(session_sets.values(), []))


if __name__ == "__main__":
    raise SystemExit(main())
