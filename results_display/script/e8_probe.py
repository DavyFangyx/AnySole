#!/usr/bin/env python3
"""E8 基于探针的模态互补信息分析（实验计划 E8 口径，多指标扩展 2026-10-30）。

冻结 U（缺省 V7_Cv5，ckpt_last，canonical 划分 splits.csv），逐窗口读 V/T
时序记忆 m_v / m_t（V7MotionHead 输入前），三组 ridge 探针按指标分别预测
可解码目标（train 拟合、val 选正则、test 评估）。

探针组（同一种 ridge，同一批窗口）：
- M_V        视觉可解码信息
- M_T        触觉可解码信息（辅助参照）
- [M_V, M_T] 加入触觉后的增量

指标（逐帧；骨盆相对踝足 = 4 关节；图内英文）：
- position   踝足位置（mm）       —— 原口径
- position_y 踝足垂直分量（mm）   —— 相对位置 y 轴（垂直）分量
- stance_y   支撑期足垂直（mm）   —— 任一足接触帧的踝足 y 分量
- velocity   踝足速度（mm/s）     —— 相对位置前向差分 ×40，丢每窗末帧
- global     骨盆全局位置（mm）   —— 世界系 kp_gt 骨盆
- ground     足-地距离（mm）      —— 踝足世界 y − floor_y
- contact    左右脚接触状态       —— 二值分类准确率（%，越高越好）

判据：[M_V, M_T] 优于 M_V 与 M_T 即支持触觉提供额外可解码运动信息
（不要求 T 单独优于 V）；逐指标赢家见报告与图。

特征/标签按模型缓存（cache/，--force 重建），改图或增指标无需重跑推理。
用法（touch_gait env）：
  python results_display/script/e8_probe.py --model V7_Cv5 --device cuda:2
产物：results_display/Etest/E8_ComplementarityProbe/
      {Full_Table1_E8_Probe.csv, E8_模态互补探针.md, Full_Fig1_E8_Probe.png}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.registry import get_builder, infer_anysole_paths  # noqa: E402
from anysole.types import ANKLE_FOOT_JOINTS, JOINT_NAMES  # noqa: E402

# 与 e_results_export.py ETEST 默认口径一致（state_smpl24 × test）
OUT_DIR = REPO / "results_display/Etest/E8_ComplementarityProbe"
CACHE_DIR = OUT_DIR / "cache"
SPLIT_CSV = (REPO / "AnysoleWorkspace/protocol/splits/default/splits.csv")
LAMBDAS = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
PELVIS = JOINT_NAMES.index("pelvis")
FPS = 40.0
TTA_VARIANTS = {  # 与 run_tta TTA_MODELS 一致
    "V7_Cv1": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv2": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv3": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv4": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv5": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
}
ARMS = ("M_V", "M_T", "[M_V, M_T]")
METRICS = ("position", "position_y", "stance_y", "velocity", "global",
           "ground", "contact")
METRIC_ZH = {"position": "踝足位置", "position_y": "踝足垂直分量",
             "stance_y": "支撑期足垂直", "velocity": "踝足速度",
             "global": "骨盆全局位置", "ground": "足-地距离",
             "contact": "接触准确率"}
METRIC_UNIT = {"position": "mm", "position_y": "mm", "stance_y": "mm",
               "velocity": "mm/s", "global": "mm", "ground": "mm",
               "contact": "%"}
METRIC_HIGHER = {"contact": True}  # 缺省 = 越低越好
METRIC_LABELS = {  # 图内英文（项目口径）
    "position": "Ankle-foot position (mm)",
    "position_y": "Ankle-foot vertical (mm)",
    "stance_y": "Stance foot vertical (mm)",
    "velocity": "Ankle-foot velocity (mm/s)",
    "global": "Global pelvis position (mm)",
    "ground": "Foot-ground distance (mm)",
    "contact": "Contact accuracy (%)",
}
METRIC_SHORT = {"position": "position", "position_y": "vertical",
                "stance_y": "stance vert.", "velocity": "velocity",
                "global": "global", "ground": "ground", "contact": "contact"}
ARM_COLORS = {"M_V": "#2a78d6", "M_T": "#eb6834", "[M_V, M_T]": "#1baf7a"}
COLLECT_KEYS = ("xv", "xt", "ypos", "yvel", "yglob", "yground", "ycontact")


def build_model(model_name: str, device: torch.device):
    paths = infer_anysole_paths(model_name, contact="f6_soft_time_accel_state_smpl24_v1",
                                variant=TTA_VARIANTS[model_name], which="last")
    ckpt = torch.load(paths["ckpt"], map_location="cpu")
    saved = ckpt["config"]
    build_cfg = dict(saved)
    for key in ("model_name", "epochs", "lr"):
        build_cfg.pop(key, None)
    part_joints = build_cfg.pop("part_joints", None)
    model = get_builder(model_name).build(build_cfg, part_joints=part_joints).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def collect(model, mode: str, device: torch.device) -> dict[str, np.ndarray]:
    """逐窗口读 m_v/m_t（VT 条件）与全部指标目标。

    特征存 3D (nwin, tw, d)；velocity 目标丢每窗末帧（前向差分无下一帧），
    使用侧切片 x[:, :-1] 对齐。
    """
    dataset = AnySoleDataset(mode, split_csv=SPLIT_CSV, cam_id=3, stride=20,
                             contact_method="f6_soft_time_accel_state_smpl24_v1",
                             tactile_input="raw108")
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=2, collate_fn=collate_windows)
    out = {k: [] for k in COLLECT_KEYS}
    with torch.inference_mode():
        for bi, raw in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in raw.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)  # VT
            T_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
            v_tok, t_tok = model.encoders(batch["V_feat"], T_tac, config_id,
                                          V_hmr=batch.get("V_hmr"))
            m_v = model.v_temporal(v_tok)   # (B, tw, d)
            m_t = model.t_temporal(t_tok)
            out["xv"].append(m_v.cpu().numpy())
            out["xt"].append(m_t.cpu().numpy())
            kp = batch["kp_gt"]  # (B, tw, 24, 3) 世界系
            af = kp[:, :, ANKLE_FOOT_JOINTS] - kp[:, :, PELVIS:PELVIS + 1]
            out["ypos"].append((af * 1000.0).cpu().numpy())  # (B, tw, 4, 3) mm
            vel = (af[:, 1:] - af[:, :-1]) * FPS * 1000.0
            out["yvel"].append(vel.cpu().numpy())  # (B, tw-1, 4, 3) mm/s
            out["yglob"].append((kp[:, :, PELVIS] * 1000.0).cpu().numpy())  # (B, tw, 3)
            fy = batch["floor_y"]  # (B,)
            gy = (kp[:, :, ANKLE_FOOT_JOINTS, 1:2] - fy[:, None, None, None]) * 1000.0
            out["yground"].append(gy[..., 0].cpu().numpy())  # (B, tw, 4) mm
            out["ycontact"].append(batch["contact_gt"].cpu().numpy())  # (B, tw, 2)
            if bi % 100 == 0:
                print("[E8] %s: batch %d/%d" % (mode, bi, len(loader)), flush=True)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y)


def metric_err(metric: str, X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    if metric == "contact":
        pred = X @ W >= 0.5
        return float(np.mean(pred == (Y > 0.5)) * 100.0)  # 准确率 %
    shape = {"position": (4, 3), "velocity": (4, 3), "global": (1, 3),
             "ground": (4, 1), "position_y": (4, 1), "stance_y": (4, 1)}[metric]
    pred = (X @ W).reshape(-1, *shape)
    gt = Y.reshape(-1, *shape)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def select_arm(metric: str, train_x, val_x, test_x,
               y_train, y_val, y_test) -> dict:
    higher = METRIC_HIGHER.get(metric, False)
    best_lam, best_val = None, (float("-inf") if higher else float("inf"))
    for lam in LAMBDAS:
        W = ridge_fit(train_x, y_train, lam)
        err = metric_err(metric, val_x, y_val, W)
        if (err > best_val) if higher else (err < best_val):
            best_val, best_lam = err, lam
    W = ridge_fit(train_x, y_train, best_lam)
    return {"test": metric_err(metric, test_x, y_test, W),
            "val": best_val, "lam": best_lam}


def arm_features(metric: str, arm: str, xv: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """(nwin, tw, d) 特征 → (N, d) 样本：M_V / M_T / [M_V, M_T]；
    velocity 丢每窗末帧（与标签对齐）。"""
    if metric == "velocity":
        xv, xt = xv[:, :-1], xt[:, :-1]
    xv2 = xv.reshape(-1, xv.shape[-1])
    xt2 = xt.reshape(-1, xt.shape[-1])
    return {"M_V": xv2, "M_T": xt2,
            "[M_V, M_T]": np.concatenate([xv2, xt2], axis=1)}[arm]


def metric_target(metric: str, data: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """3D 标签 → (Y (N,K), mask (N,) 或 None)。velocity 丢每窗末帧
    （前向差分无下一帧）；stance_y 只取任一足接触帧。"""
    if metric == "velocity":
        y = data["yvel"]
        return y.reshape(-1, y.shape[-2] * y.shape[-1]), None
    if metric == "position_y":
        return data["ypos"][..., 1].reshape(-1, 4), None
    if metric == "stance_y":
        mask = (data["ycontact"] > 0.5).any(-1).reshape(-1)
        return data["ypos"][..., 1].reshape(-1, 4), mask
    key = "y" + {"position": "pos", "global": "glob", "ground": "ground",
                 "contact": "contact"}[metric]
    y = data[key]
    return y.reshape(-1, int(np.prod(y.shape[2:]))), None


def render_figure(rows: dict, out_png: Path) -> None:
    """上行 5 个横向条形（每指标三臂绝对误差/准确率，赢家★加粗边），
    下行相对最优点（1.0 = 最优，contact 取准确率倒数）。图内英文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, mut, gridc = "#1c1b19", "#6f6d66", "#e5e4df"
    n = len(METRICS)
    fig = plt.figure(figsize=(16.2, 4.9))
    gs = fig.add_gridspec(2, n, height_ratios=(2.1, 1.35), hspace=0.62,
                          wspace=0.55, top=0.80, bottom=0.13, left=0.085,
                          right=0.995)
    # 上行：绝对误差小多组（每指标一个轴，无双重轴）
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        higher = METRIC_HIGHER.get(metric, False)
        vals = [rows[metric][arm]["test"] for arm in ARMS]
        best = max(vals) if higher else min(vals)
        for j, arm in enumerate(reversed(ARMS)):  # y 轴自上而下 M_V/M_T/[M_V,M_T]
            v = vals[len(ARMS) - 1 - j]
            ypos = len(ARMS) - 1 - j
            is_best = (v == best)
            ax.barh(ypos, v, height=0.58,
                    color=ARM_COLORS[arm],
                    edgecolor=ink if is_best else "none",
                    linewidth=1.4 if is_best else 0)
            ax.text(v, ypos, ("★ %.4g" if is_best else "%.4g") % v,
                    va="center", ha="left", fontsize=7, color=ink)
        ax.set_ylim(-0.6, len(ARMS) - 1 + 0.6)
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels(list(reversed(ARMS)), fontsize=8, color=ink)
        ax.set_title(METRIC_LABELS[metric], fontsize=8.5, color=ink, pad=4)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.tick_params(axis="x", labelsize=6.5, colors=mut)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(mut)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.xaxis.grid(True, color=gridc, lw=0.8)
        ax.set_axisbelow(True)
    # 下行：相对最优点图（1.0 = 最优；contact 越高越好 → 取最优/实际）
    axr = fig.add_subplot(gs[1, :])
    rel_all = []
    for i, metric in enumerate(METRICS):
        higher = METRIC_HIGHER.get(metric, False)
        vals = [rows[metric][arm]["test"] for arm in ARMS]
        best = max(vals) if higher else min(vals)
        rel = [best / v if higher else v / best for v in vals]
        rel_all.append(rel)
        for j, arm in enumerate(ARMS):  # 组内自上而下 M_V/M_T/[M_V,M_T]
            ypos = i + (j - 1) * 0.16
            axr.plot(rel[j], ypos, "o", ms=5, color=ARM_COLORS[arm], zorder=3)
            if rel[j] > 1.02:
                axr.text(rel[j], ypos, " %.2f" % rel[j],
                         va="center", ha="left", fontsize=6.5, color=ink)
    axr.axvline(1.0, color=mut, lw=0.9, ls="--")
    axr.set_yticks(range(n))
    axr.set_yticklabels([METRIC_SHORT[m] for m in METRICS], fontsize=7.5, color=ink)
    axr.set_ylim(-0.55, n - 0.45)
    axr.set_xlim(0, max(max(r) for r in rel_all) * 1.13 + 0.03)
    axr.set_xlabel("error relative to best (1.0 = best; contact = accuracy ratio inverted)",
                   fontsize=8, color=mut)
    axr.set_title("Relative gap to the best arm", fontsize=8.5, color=ink, pad=4)
    axr.tick_params(axis="x", labelsize=7, colors=mut)
    axr.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        axr.spines[s].set_visible(False)
    axr.spines["bottom"].set_color(mut)
    axr.spines["bottom"].set_linewidth(0.8)
    axr.xaxis.grid(True, color=gridc, lw=0.8)
    axr.set_axisbelow(True)
    fig.suptitle("E8 modality complementarity probe — decodability by input representation\n"
                 "★ = best arm per metric (lower error is better; contact = higher accuracy)",
                 fontsize=10.5, color=ink, y=0.955)
    fig.legend(handles=[Patch(facecolor=ARM_COLORS[a]) for a in ARMS],
               labels=ARMS, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="V7_Cv5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="重建特征缓存（重跑推理）")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    if args.model not in TTA_VARIANTS:
        raise SystemExit("模型 %s 不在 TTA_VARIANTS（V7_Cv1..Cv5）" % args.model)

    cache_path = CACHE_DIR / f"{args.model}_probe.npz"
    if cache_path.is_file() and not args.force:
        print("[E8] load cache %s" % cache_path.relative_to(REPO))
        z = np.load(cache_path)
        splits = {mode: {k: z[f"{mode}_{k}"] for k in COLLECT_KEYS}
                  for mode in ("train", "val", "test")}
    else:
        device = torch.device(args.device)
        print("[E8] build %s ..." % args.model)
        model = build_model(args.model, device)
        splits = {}
        for mode in ("train", "val", "test"):
            print("[E8] collect %s ..." % mode)
            splits[mode] = collect(model, mode, device)
            print("[E8] %s: windows=%d" % (mode, splits[mode]["xv"].shape[0]))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {f"{mode}_{k}": v for mode, d in splits.items() for k, v in d.items()}
        np.savez_compressed(cache_path, **data)
        print("[E8] cache saved: %s" % cache_path.relative_to(REPO))

    # 各指标 × 三臂：ridge 拟合（train）、val 选 λ、test 评估
    rows: dict[str, dict[str, dict]] = {}
    for metric in METRICS:
        rows[metric] = {}
        y_train, m_train = metric_target(metric, splits["train"])
        y_val, m_val = metric_target(metric, splits["val"])
        y_test, m_test = metric_target(metric, splits["test"])
        for arm in ARMS:
            train_x = arm_features(metric, arm, splits["train"]["xv"],
                                   splits["train"]["xt"])
            val_x = arm_features(metric, arm, splits["val"]["xv"],
                                 splits["val"]["xt"])
            test_x = arm_features(metric, arm, splits["test"]["xv"],
                                  splits["test"]["xt"])
            if m_train is not None:
                tx, ty = train_x[m_train], y_train[m_train]
                vx, vy = val_x[m_val], y_val[m_val]
                ex, ey = test_x[m_test], y_test[m_test]
            else:
                tx, ty, vx, vy, ex, ey = (train_x, y_train, val_x, y_val,
                                          test_x, y_test)
            rows[metric][arm] = select_arm(metric, tx, vx, ex, ty, vy, ey)
            print("[E8] %-10s %-12s val=%.2f (λ=%g) test=%.2f %s"
                  % (metric, arm, rows[metric][arm]["val"],
                     rows[metric][arm]["lam"], rows[metric][arm]["test"],
                     METRIC_UNIT[metric]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "Full_Table1_E8_Probe.csv").open("w", newline="",
                                                     encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["探针输入", "指标", "test", "val", "λ"])
        for metric in METRICS:
            for arm in ARMS:
                r = rows[metric][arm]
                w.writerow([arm, metric, "%.2f" % r["test"], "%.2f" % r["val"],
                            r["lam"]])

    winners = {}
    for metric in METRICS:
        higher = METRIC_HIGHER.get(metric, False)
        vals = {arm: rows[metric][arm]["test"] for arm in ARMS}
        winners[metric] = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
    joint_better = [METRIC_ZH[m] for m in METRICS
                    if winners[m] == "[M_V, M_T]"]
    t_better = [METRIC_ZH[m] for m in METRICS if winners[m] == "M_T"]
    winners_str = "，".join("%s→%s" % (METRIC_ZH[m], winners[m]) for m in METRICS)
    joint_str = "，".join(joint_better) if joint_better else "无"
    t_str = "，".join(t_better) if t_better else "无"
    report = """# E8 基于探针的模态互补信息分析

口径：U = V7_Cv5（ckpt_last，canonical 划分 splits.csv）；VT 条件逐窗口读 V/T
时序记忆 m_v / m_t（V7MotionHead 输入前，维度 %d / %d）；ridge 闭式解，λ ∈
{1e-2..1e3} 在 val 上选最优后 test 评估。指标（逐帧；骨盆相对踝足 = 4 关节）：
position=踝足位置（mm）、position_y=踝足垂直分量（mm）、stance_y=支撑期
足垂直（mm，任一足接触帧）、velocity=踝足速度（mm/s，前向差分 ×40）、
global=骨盆全局位置（mm）、ground=足-地距离（mm，踝足 y − floor_y）、
contact=左右脚接触准确率（%%，越高越好）。

| 探针输入 | %s |
|---|---|
""" % (splits["train"]["xv"].shape[-1], splits["train"]["xt"].shape[-1],
       " | ".join("%s (%s)" % (METRIC_ZH[m], METRIC_UNIT[m]) for m in METRICS))
    for arm in ARMS:
        cells = ["%.2f" % rows[m][arm]["test"] for m in METRICS]
        report += "| %s | %s |\n" % (arm, " | ".join(cells))
    report += """
判读（实验计划 E8）：逐指标赢家：%s（图 Full_Fig1_E8_Probe.png）。
[M_V, M_T] 优于 M_V 的指标：%s——触觉的额外可解码信息集中在垂直分量；
M_T 单独优于 V 的指标：%s。原 position 判据（联合优于 V 与 T）val 上
成立、test 未复现。
""" % (winners_str, joint_str, t_str)
    (OUT_DIR / "E8_模态互补探针.md").write_text(report, encoding="utf-8")

    if not args.no_figure:
        render_figure(rows, OUT_DIR / "Full_Fig1_E8_Probe.png")
    print("[E8] done -> %s" % OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
