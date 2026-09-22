#!/usr/bin/env python3
"""R_Test6 互补分析（缺失率任务书实验 2）：融合是互补还是拼贴。

子命令（按任务书阶段逐步落地）：
  --bar   2a 互补 bar：每部位 V2M / T2M / VT2M 三根柱（PA-MPJPE）+ Δ 判据图 + 辅助指标。
          纯读已落盘的 fseries（metrics/<split>_fseries.json），零重训、不加载模型。
  --probe 2b 探针表：Ft/Fv/融合 F 的 ridge 读出（依赖 script/utils/ridge_probe.py 扩展，未落地）。
  --tsne  2c t-SNE：三配置 F 的降维散点（已实现：读 R_Test5 的表征 dump，自带 torch t-SNE）。

2a 判据（任务书）：下肢/接触 VT2M 应优于 V2M（T 的贡献），全局轨迹/yaw VT2M 应优于
T2M（V 的贡献），两个方向同时成立才叫互补；只取两者更好的那个 = 拼贴。
注意：V4B val 上总体 V2M(35.4) < VT2M(40.4)，所以必须按部位看。
CSV/图内 Δ 统一"负 = 融合更好"口径（higher-is-better 指标已取反）。

Usage (touch_gait env, run from repo root):
  python results_display/script/r_test6_complement.py --bar
  python results_display/script/r_test6_complement.py --bar --split val \
      --model-dir results/AnySole/V4B_joint_and
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- 配色（dataviz 参考调色板，已验证：categorical 前 3 槽 all-pairs 通过）---
C_V2M = "#2a78d6"   # categorical slot 1 (blue)
C_T2M = "#eb6834"   # categorical slot 2 (orange)
C_VT2M = "#1baf7a"  # categorical slot 3 (aqua; 对比度 WARN → 必须有直标/表视图)
C_GAIN = "#2a78d6"  # diverging arm: 融合优于单模态（Δ<0，越小越好指标）
C_LOSS = "#e34948"  # diverging arm: 融合更差（Δ>0）
C_FLAT = "#c3c2b7"  # diverging neutral（|Δ|<=tol）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CONFIG_ORDER = ("V2M", "T2M", "VT2M")
CONFIG_COLORS = {"V2M": C_V2M, "T2M": C_T2M, "VT2M": C_VT2M}

# 2c t-SNE：三个角配置（ρ 网格角格 = 与三配置评测同机制）；动作着色用前 8 槽，其余折 Other
TSNE_CORNERS = {"VT2M": (100, 100), "V2M": (100, 0), "T2M": (0, 100)}
TSNE_ACTION_COLORS = {
    "0": "#2a78d6", "1": "#eb6834", "2": "#1baf7a", "3": "#eda100",
    "4": "#e87ba4", "5": "#008300", "6": "#4a3aa7", "7": "#e34948",
}

# 9 部位（fseries 键）+ 聚合 + 总体，按解剖顺序排列
ANATOMICAL = ["root", "torso", "headneck",
              "l_arm", "r_arm", "l_leg", "r_leg", "l_foot", "r_foot"]
AGGREGATES = ["upper", "lower", "anklefoot", "hands"]
PARTS = ANATOMICAL + AGGREGATES


def pa_key(g: str) -> str:
    """PA-MPJPE 的 fseries 键：解剖部位带 _part_ 前缀，聚合不带，overall 无后缀。"""
    if g == "overall":
        return "PA-MPJPE"
    if g in ANATOMICAL:
        return "PA-MPJPE_part_%s" % g
    return "PA-MPJPE_%s" % g
PART_LABELS = {
    "root": "root", "torso": "torso", "headneck": "headneck",
    "l_arm": "l-arm", "r_arm": "r-arm", "l_leg": "l-leg", "r_leg": "r-leg",
    "l_foot": "l-foot", "r_foot": "r-foot",
    "upper": "upper", "lower": "lower", "anklefoot": "ankle-foot", "hands": "hands",
}

# 判据分组：T 方向（VT 应优于 V2M）/ V 方向（VT 应优于 T2M）/ 上半身（观察组）
T_DIRECTION = ["l_leg", "r_leg", "l_foot", "r_foot", "lower", "anklefoot"]
V_DIRECTION = ["root", "torso"]
UPPER = ["headneck", "l_arm", "r_arm", "upper", "hands"]

# 辅助指标：数值越小越好（lower_is_better=False 的指标 Δ 方向取反）
AUX_METRICS = [
    ("RTE_norm", "RTE (norm)", True),
    ("yaw_abs_deg", "yaw abs (deg)", True),
    ("yaw_drift_deg", "yaw drift (deg)", True),
    ("contact_acc", "contact acc", False),
    ("contact_f1", "contact f1", False),
    ("air_recall", "air recall", False),
    ("foot_slide_mm", "foot slide (mm)", True),
    ("seam_jump_mm", "seam jump (mm)", True),
    ("jitter_mm", "jitter (mm)", True),
    ("accel_err_ms2", "accel err (m/s2)", True),
    ("accel_dist_err_upper_ms2", "accel upper (m/s2)", True),
    ("joint_limit_viol_knee", "knee viol", True),
    ("joint_limit_viol_elbow", "elbow viol", True),
]
# 接触类指标归入 T 方向判据；轨迹/朝向类归入 V 方向
T_DIRECTION_AUX = ["contact_acc", "contact_f1", "air_recall", "foot_slide_mm"]
V_DIRECTION_AUX = ["RTE_norm", "yaw_abs_deg", "yaw_drift_deg", "seam_jump_mm"]


# 各指标"持平"容差（单位随指标）：未列出的指标沿用 --tol（默认 2.0，PA 口径 mm）
TOL_OVERRIDES = {
    "RTE_norm": 1.0,                  # mm
    "yaw_abs_deg": 2.0,               # deg
    "yaw_drift_deg": 2.0,             # deg
    "contact_acc": 0.02,              # 0-1
    "contact_f1": 0.02,               # 0-1
    "air_recall": 0.02,               # 0-1
    "accel_err_ms2": 0.5,             # m/s^2
    "accel_dist_err_upper_ms2": 0.5,  # m/s^2
    "joint_limit_viol_knee": 0.005,   # 比例
    "joint_limit_viol_elbow": 0.005,  # 比例
}


def tol_for(key: str, base: float) -> float:
    return TOL_OVERRIDES.get(key, base)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test6 互补分析（任务书实验 2）")
    parser.add_argument("--bar", action="store_true", help="2a 互补 bar（已实现）")
    parser.add_argument("--probe", action="store_true", help="2b 探针表（未落地）")
    parser.add_argument("--tsne", action="store_true", help="2c t-SNE（已实现：三配置 F 降维散点）")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--model-dir", type=Path,
                        default=REPO / "results" / "AnySole" / "V4B_joint_and",
                        help="模型目录（含 metrics/<split>_fseries.json）")
    parser.add_argument("--repr-dir", type=Path, default=None,
                        help="ρ 网格表征 dump 目录（默认 result/r_test5_rho_grid/repr/）")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "result" / "r_test6_complement",
                        help="产物目录")
    parser.add_argument("--sessions", type=int, default=None,
                        help="t-SNE 用到的 session 数（默认全部）")
    parser.add_argument("--max-points", type=int, default=800,
                        help="t-SNE 每配置采样帧数上限")
    parser.add_argument("--tol", type=float, default=2.0,
                        help="|Δ|<=tol 视为持平（PA-MPJPE 口径 mm；各指标另有按量纲的默认容差）")
    return parser.parse_args(argv)


def load_fseries(model_dir: Path, split: str) -> dict:
    path = model_dir / "metrics" / ("%s_fseries.json" % split)
    if not path.is_file():
        raise SystemExit("fseries 缺失：%s（先跑 python -m anysole.eval 生成）" % path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics")
    missing = [c for c in CONFIG_ORDER if c not in metrics]
    if missing:
        raise SystemExit("fseries 缺配置 %s：%s" % (missing, path))
    return data


def delta(metrics: dict, part_key: str, lower_is_better: bool = True) -> tuple:
    """(ΔVT−V2M, ΔVT−T2M)；lower_is_better=False 时取反（增益方向统一为正）。"""
    sign = 1.0 if lower_is_better else -1.0
    d_v = sign * (metrics["VT2M"][part_key] - metrics["V2M"][part_key])
    d_t = sign * (metrics["VT2M"][part_key] - metrics["T2M"][part_key])
    return d_v, d_t


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_chart(metrics: dict, out_dir: Path, tol: float) -> None:
    """每部位三根柱（PA-MPJPE）+ 组顶 Δ 注释。"""
    groups = PARTS + ["overall"]
    labels = [PART_LABELS.get(g, "overall") for g in groups]
    keys = [pa_key(g) for g in groups]
    values = {c: [metrics[c][k] for k in keys] for c in CONFIG_ORDER}

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    x = np.arange(len(groups))
    width = 0.24
    for i, cfg in enumerate(CONFIG_ORDER):
        ax.bar(x + (i - 1) * width, values[cfg], width,
               color=CONFIG_COLORS[cfg], label=cfg, zorder=3)
    # 直标：仅 VT2M（融合参照柱）；Δ 注释写组顶（文本一律用墨色，不用系列色）
    for g, (v2m, t2m, vt2m) in enumerate(zip(values["V2M"], values["T2M"], values["VT2M"])):
        ax.text(x[g], vt2m + max(values["T2M"]) * 0.015, "%.0f" % vt2m,
                ha="center", va="bottom", fontsize=7, color=INK2)
        dv = vt2m - v2m
        dt = vt2m - t2m
        ax.text(x[g], max(v2m, t2m, vt2m) + max(values["T2M"]) * 0.075,
                "dV%+.1f  dT%+.1f" % (dv, dt),
                ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PA-MPJPE (mm)", color=INK2, fontsize=9)
    ax.set_title("Per-part PA-MPJPE by conditioning  (%s)" % metrics["VT2M"].get("split", ""),
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ymax = max(values["T2M"]) * 1.22
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_dir / "complement_bar.png", facecolor=SURFACE)
    plt.close(fig)


def delta_chart(metrics: dict, out_dir: Path, tol: float) -> None:
    """发散条形：每部位两行（T 向 ΔVT−V2M / V 向 ΔVT−T2M）。

    蓝 = 融合优于单模态（gain），红 = 融合更差（loss），灰 = 持平。
    """
    groups = PARTS + ["overall"]
    rows = []
    for g in groups:
        dv, dt = delta(metrics, pa_key(g))
        rows.append(("%s  T-way" % PART_LABELS.get(g, "overall"), dv, tol))
        rows.append(("%s  V-way" % PART_LABELS.get(g, "overall"), dt, tol))

    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    y = np.arange(len(rows))[::-1]
    for yi, (label, dv, row_tol) in zip(y, rows):
        color = C_GAIN if dv < -row_tol else (C_LOSS if dv > row_tol else C_FLAT)
        ax.barh(yi, dv, height=0.62, color=color, zorder=3)
        ax.text(dv + (0.8 if dv >= 0 else -0.8), yi, "%+.1f" % dv,
                ha="left" if dv >= 0 else "right", va="center",
                fontsize=7, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("Δ PA-MPJPE (mm), negative = fusion better", color=INK2, fontsize=9)
    ax.set_title("Fusion vs single-modality by part  (blue=gain, red=loss, gray=flat)",
                 color=INK, fontsize=11, loc="left", pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GAIN, label="fusion better"),
                       Patch(color=C_LOSS, label="fusion worse"),
                       Patch(color=C_FLAT, label="flat (per-metric tol)")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "complement_delta.png", facecolor=SURFACE)
    plt.close(fig)


def aux_chart(metrics: dict, out_dir: Path) -> None:
    """辅助指标小倍图：每个指标一个面板 × 三配置柱。"""
    n = len(AUX_METRICS)
    ncols, nrows = 7, int(np.ceil(n / 7))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.6 * nrows), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for i, (key, label, lower_better) in enumerate(AUX_METRICS):
        ax = axes[i]
        style_ax(ax)
        vals = [metrics[c][key] for c in CONFIG_ORDER]
        ax.bar(CONFIG_ORDER, vals, 0.55,
               color=[CONFIG_COLORS[c] for c in CONFIG_ORDER], zorder=3)
        ax.set_title(label, fontsize=8, color=INK2, loc="left", pad=4)
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)
    fig.suptitle("Aux metrics by conditioning", color=INK, fontsize=11, x=0.01, ha="left")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CONFIG_COLORS[c]) for c in CONFIG_ORDER],
               labels=CONFIG_ORDER,
               frameon=False, fontsize=9, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_dir / "complement_aux.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)


def verdict(metrics: dict, tol: float) -> dict:
    """互补判据：T 向（VT vs V2M）与 V 向（VT vs T2M）各自的 gain/loss/flat 计数。"""
    def tally(part_keys, aux_keys, which):
        """which='T' 用 Δ(VT−V2M)，which='V' 用 Δ(VT−T2M)。"""
        lower_map = {k: lb for k, _label, lb in AUX_METRICS}
        gains = losses = flats = 0
        for g in part_keys:
            dv = delta(metrics, pa_key(g))[0 if which == "T" else 1]
            if dv < -tol:
                gains += 1
            elif dv > tol:
                losses += 1
            else:
                flats += 1
        for k in aux_keys:
            row_tol = tol_for(k, tol)
            dv = delta(metrics, k, lower_map[k])[0 if which == "T" else 1]
            if dv < -row_tol:
                gains += 1
            elif dv > row_tol:
                losses += 1
            else:
                flats += 1
        return gains, losses, flats

    t_g, t_l, t_f = tally(T_DIRECTION, T_DIRECTION_AUX, "T")
    v_g, v_l, v_f = tally(V_DIRECTION, V_DIRECTION_AUX, "V")
    u_g, u_l, u_f = tally(UPPER, [], "T")
    overall_dv, overall_dt = delta(metrics, "PA-MPJPE")

    t_ok = t_g > t_l and t_g >= t_f
    v_ok = v_g > v_l and v_g >= v_f
    if t_ok and v_ok:
        kind = "互补（双向增益）"
    elif t_ok or v_ok:
        kind = "拼贴（仅单向增益：%s）" % ("T 向" if t_ok else "V 向")
    else:
        kind = "融合无增益（两向都未占优）"
    return {
        "T_direction": {"gain": t_g, "loss": t_l, "flat": t_f,
                        "parts": T_DIRECTION, "aux": T_DIRECTION_AUX},
        "V_direction": {"gain": v_g, "loss": v_l, "flat": v_f,
                        "parts": V_DIRECTION, "aux": V_DIRECTION_AUX},
        "upper_body": {"gain": u_g, "loss": u_l, "flat": u_f, "parts": UPPER},
        "overall_delta": {"dV_VT_minus_V2M": overall_dv, "dT_VT_minus_T2M": overall_dt},
        "tol": tol,
        "verdict": kind,
    }


def run_bar(args: argparse.Namespace) -> int:
    data = load_fseries(args.model_dir, args.split)
    metrics = data["metrics"]
    metrics["VT2M"]["split"] = args.split  # 仅供图题，不改数据
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_chart(metrics, out_dir, args.tol)
    delta_chart(metrics, out_dir, args.tol)
    aux_chart(metrics, out_dir)

    # 汇总表（表视图 = 可及性兜底，也供贴文/表格复用）
    groups = PARTS + ["overall"]
    keys = [pa_key(g) for g in groups]
    rows = []
    for g, k in zip(groups, keys):
        v2m, t2m, vt2m = (metrics[c][k] for c in CONFIG_ORDER)
        dv, dt = delta(metrics, k)
        rows.append({"group": g, "V2M": round(v2m, 2), "T2M": round(t2m, 2),
                     "VT2M": round(vt2m, 2), "dVT_minus_V2M": round(dv, 2),
                     "dVT_minus_T2M": round(dt, 2)})
    for key, label, lower_better in AUX_METRICS:
        v2m, t2m, vt2m = (metrics[c][key] for c in CONFIG_ORDER)
        dv, dt = delta(metrics, key, lower_better)
        rows.append({"group": key, "V2M": round(v2m, 4), "T2M": round(t2m, 4),
                     "VT2M": round(vt2m, 4), "dVT_minus_V2M": round(dv, 4),
                     "dVT_minus_T2M": round(dt, 4)})
    summary = {"checkpoint": data.get("checkpoint"), "split": args.split,
               "verdict": verdict(metrics, args.tol), "rows": rows}
    (out_dir / "complement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import csv
    with open(out_dir / "complement_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    v = summary["verdict"]
    print("T 向（VT vs V2M）: gain %d / loss %d / flat %d" %
          (v["T_direction"]["gain"], v["T_direction"]["loss"], v["T_direction"]["flat"]))
    print("V 向（VT vs T2M）: gain %d / loss %d / flat %d" %
          (v["V_direction"]["gain"], v["V_direction"]["loss"], v["V_direction"]["flat"]))
    print("上半身            : gain %d / loss %d / flat %d" %
          (v["upper_body"]["gain"], v["upper_body"]["loss"], v["upper_body"]["flat"]))
    print("总体 Δ(VT−V2M)=%+.1f  Δ(VT−T2M)=%+.1f mm" %
          (v["overall_delta"]["dV_VT_minus_V2M"], v["overall_delta"]["dT_VT_minus_T2M"]))
    print("判定：%s" % v["verdict"])
    print("产物：%s" % out_dir)
    return 0


def run_tsne(args: argparse.Namespace) -> int:
    """2c t-SNE：三个角配置（VT2M/V2M/T2M）的 F 降维散点，按配置/动作着色。

    期望（任务书）：按动作聚成团、而不是按模态分成两堆 = 模态不变的共享状态。
    """
    if not args.tsne:
        return 0
    repr_dir = args.repr_dir or (
        REPO / "results_display" / "result" / "r_test5_rho_grid" / "repr")
    if not repr_dir.is_dir():
        raise SystemExit("repr dump 缺失：%s（先跑 python -m anysole.rho_grid）" % repr_dir)
    npz_files = sorted(repr_dir.glob("*_repr.npz"))
    if not npz_files:
        raise SystemExit("repr 目录为空：%s" % repr_dir)
    sessions = sorted({p.name.split("_")[0] for p in npz_files})
    if args.sessions is not None:
        sessions = sessions[: args.sessions]

    frames, configs, actions = [], [], []
    rng = np.random.RandomState(0)
    for cfg, (rV, rT) in TSNE_CORNERS.items():
        f_list, a_list = [], []
        for sid in sessions:
            path = repr_dir / ("%s_rhoV%d_rhoT%d_repr.npz" % (sid, rV, rT))
            if not path.is_file():
                continue
            dump = np.load(path)
            if "F" not in dump or "v_tok" not in dump:
                raise SystemExit("%s 缺 F/v_tok（旧版 dump？）" % path)
            tok = dump["v_tok"]
            n_frames = tok.shape[0]
            f = dump["F"]
            per_frame = f.shape[0] // n_frames
            f = f.reshape(n_frames, per_frame * f.shape[-1])
            f_list.append(f)
            a_list.append(np.full(n_frames, sid[2:4], dtype=object))
        if not f_list:
            raise SystemExit("配置 %s 无 session 产物" % cfg)
        F = np.concatenate(f_list)
        A = np.concatenate(a_list)
        if len(F) > args.max_points:
            idx = rng.choice(len(F), args.max_points, replace=False)
            F, A = F[idx], A[idx]
        frames.append(torch.from_numpy(F).float())
        configs.append(np.full(len(F), cfg, dtype=object))
        actions.append(A)
    X = torch.cat(frames, dim=0)
    cfg_labels = np.concatenate(configs)
    act_labels = np.concatenate(actions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("t-SNE: %d 帧 × %d 维（device=%s）" % (X.shape[0], X.shape[1], device))
    Y = _tsne(X, device=device)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "tsne_coords.csv",
               np.column_stack([Y, cfg_labels, act_labels]),
               fmt="%s", delimiter=",", header="x,y,config,action", comments="")
    _tsne_scatter(Y, cfg_labels, "config", out_dir / "tsne_config.png")
    _tsne_scatter(Y, act_labels, "action", out_dir / "tsne_action.png")
    print("产物：%s（tsne_coords.csv / tsne_config.png / tsne_action.png）" % out_dir)
    return 0


def _tsne_scatter(Y, labels, kind, path):
    uniq = sorted(set(labels), key=str)
    colors = TSNE_ACTION_COLORS if kind == "action" else CONFIG_COLORS
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.set_axisbelow(True)
    for label in uniq:
        m = labels == label
        color = colors.get(str(label), "#c3c2b7")
        ax.scatter(Y[m, 0], Y[m, 1], s=9, alpha=0.55, color=color,
                   label=str(label), linewidths=0)
    ax.set_xlabel("t-SNE 1", color=INK2, fontsize=9)
    ax.set_ylabel("t-SNE 2", color=INK2, fontsize=9)
    ax.set_title("F per %s (three conditioning corners)" % kind,
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8, markerscale=1.4, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pca(X, n_components):
    X = X - X.mean(dim=0, keepdim=True)
    _, _, V = torch.linalg.svd(X, full_matrices=False)
    return X @ V.T[:, :n_components]


def _tsne(X, perplexity=30.0, n_iter=300, seed=0, device="cpu"):
    """精确 t-SNE（小规模：PCA 预降维 + 对称 P + 早夸大 + 动量；无 sklearn 依赖）。"""
    torch.manual_seed(seed)
    X = _pca(X, n_components=min(30, X.shape[1], X.shape[0] - 1)).to(device)
    n = X.shape[0]
    D2 = torch.cdist(X, X) ** 2
    d2max = D2[D2 > 0].max() + 1e-9
    P = torch.zeros(n, n, device=device)
    target = float(np.log(perplexity))
    for i in range(n):
        d2 = D2[i]
        lo, hi = 0.0, 1e6
        for _ in range(60):
            beta = (lo + hi) / 2.0
            w = torch.exp(-d2 * beta)
            w[i] = 0.0
            s = w.sum()
            if s == 0:
                lo = beta
                continue
            H = torch.log(s) + beta * (d2 * w).sum() / s
            if H < target:
                lo = beta
            else:
                hi = beta
            if abs(H - target) < 1e-4:
                break
        # 防全下溢：exp(-d2*beta) 至少保留 e^-700 的动态范围
        beta = min(beta, 700.0 / d2max)
        w = torch.exp(-d2 * beta)
        w[i] = 0.0
        P[i] = w / w.sum()
    P = (P + P.T) / (2 * n)
    P = P * 12.0  # early exaggeration
    Y = torch.randn(n, 2, device=device) * 0.0001
    Y_prev = Y.clone()
    eta = 100.0
    for it in range(n_iter):
        if it == 100:
            P = P / 12.0
        diff = Y[:, None, :] - Y[None, :, :]
        dist2 = (diff ** 2).sum(dim=-1) + 1e-12
        W = 1.0 / (1.0 + dist2)
        W.fill_diagonal_(0.0)
        Q = W / W.sum()
        # 经典梯度：∂C/∂y = 4 Σ_j (p − q)(y_i − y_j)·w_ij（q·Z = w）
        grad = 4.0 * (((P - Q) * W).unsqueeze(-1) * diff).sum(dim=1)
        Y_new = Y + 0.8 * (Y - Y_prev) - eta * grad
        Y_prev = Y
        Y = Y_new
        if torch.isnan(Y).any():
            raise RuntimeError("t-SNE diverged at iter %d（数据或学习率问题）" % it)
    return Y.detach().cpu().numpy()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.bar:
        return run_bar(args)
    if args.tsne:
        return run_tsne(args)
    if args.probe:
        print("2b 探针表未落地：需先扩展 script/utils/ridge_probe.py 支持三配置 F 与 contact/yaw 目标（任务书 §二 实验 2b）。")
        return 1
    parse_args(["--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
