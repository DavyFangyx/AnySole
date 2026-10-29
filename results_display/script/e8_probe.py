#!/usr/bin/env python3
"""E8 基于探针的模态互补信息分析（实验计划 E8 口径）。

冻结 U（缺省 V7_Cv5，ckpt_last，wo05 划分），逐窗口读 V/T 时序记忆
m_v / m_t（V7MotionHead 输入前），三组 ridge 探针预测骨盆相对踝足位置
（4 关节 × 3 维，mm），训练集拟合、验证集选正则、测试集评估。

探针组（同一种 ridge，同一批窗口）：
- M_V        视觉可解码信息
- M_T        触觉可解码信息（辅助参照）
- [M_V, M_T] 加入触觉后的增量
判据：[M_V, M_T] 优于 M_V 与 M_T 即支持触觉提供额外可解码运动信息
（不要求 T 单独优于 V）。

用法（touch_gait env）：
  python results_display/script/e8_probe.py --model V7_Cv5 --device cuda:6
产物：results_display/Etest/E8_ComplementarityProbe/
      {Full_Table1_E8_Probe.csv, E8_模态互补探针.md}
"""
from __future__ import annotations

import argparse
import csv
import json
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

OUT_DIR = REPO / "results_display/Etest/E8_ComplementarityProbe"
SPLIT_CSV = (REPO / "AnysoleWorkspace/protocol/splits/default/splits_wo05.csv")
LAMBDAS = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
PELVIS = JOINT_NAMES.index("pelvis")
TTA_VARIANTS = {  # 与 run_tta TTA_MODELS 一致
    "V7_Cv1": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv2": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv3": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv4": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
    "V7_Cv5": "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1",
}


def build_model(model_name: str, device: torch.device):
    paths = infer_anysole_paths(model_name, contact="f6_soft",
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


def collect(model, mode: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """逐窗口读 m_v/m_t（VT 条件）与骨盆相对踝足位置（mm）。"""
    dataset = AnySoleDataset(mode, split_csv=SPLIT_CSV, cam_id=3, stride=20,
                             contact_method="f6_soft", tactile_input="raw108")
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=2, collate_fn=collate_windows)
    x_v, x_t, y = [], [], []
    with torch.inference_mode():
        for raw in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in raw.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)  # VT
            T_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
            v_tok, t_tok = model.encoders(batch["V_feat"], T_tac, config_id,
                                          V_hmr=batch.get("V_hmr"))
            m_v = model.v_temporal(v_tok)   # (B, tw, d)
            m_t = model.t_temporal(t_tok)
            x_v.append(m_v.reshape(-1, m_v.shape[-1]).cpu().numpy())
            x_t.append(m_t.reshape(-1, m_t.shape[-1]).cpu().numpy())
            kp = batch["kp_gt"]  # (B, tw, 24, 3)
            af = kp[:, :, ANKLE_FOOT_JOINTS] - kp[:, :, PELVIS:PELVIS + 1]
            y.append((af * 1000.0).reshape(-1, 12).cpu().numpy())  # mm
    return (np.concatenate(x_v), np.concatenate(x_t), np.concatenate(y))


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y)


def ridge_err(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    pred = (X @ W).reshape(-1, 4, 3)
    gt = Y.reshape(-1, 4, 3)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="V7_Cv5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.model not in TTA_VARIANTS:
        raise SystemExit("模型 %s 不在 TTA_VARIANTS（V7_Cv1..Cv5）" % args.model)
    device = torch.device(args.device)
    print("[E8] build %s ..." % args.model)
    model = build_model(args.model, device)

    splits = {}
    for mode in ("train", "val", "test"):
        print("[E8] collect %s ..." % mode)
        x_v, x_t, y = collect(model, mode, device)
        splits[mode] = (x_v, x_t, y)
        print("[E8] %s: frames=%d" % (mode, y.shape[0]))

    # 各臂的 val/test 特征（train 拟合用 train 特征）
    arms = {}
    for name, key in (("M_V", 0), ("M_T", 1), ("[M_V, M_T]", 2)):
        train_x = (splits["train"][0], splits["train"][1],
                   np.concatenate([splits["train"][0], splits["train"][1]], axis=1))[key]
        val_x = (splits["val"][0], splits["val"][1],
                 np.concatenate([splits["val"][0], splits["val"][1]], axis=1))[key]
        test_x = (splits["test"][0], splits["test"][1],
                  np.concatenate([splits["test"][0], splits["test"][1]], axis=1))[key]
        arms[name] = (train_x, val_x, test_x)

    rows = []
    for name, (train_x, val_x, test_x) in arms.items():
        best_lam, best_val = None, float("inf")
        for lam in LAMBDAS:
            W = ridge_fit(train_x, splits["train"][2], lam)
            err = ridge_err(val_x, splits["val"][2], W)
            if err < best_val:
                best_val, best_lam = err, lam
        test_err = ridge_err(test_x, splits["test"][2],
                             ridge_fit(train_x, splits["train"][2], best_lam))
        rows.append([name, "%.2f" % test_err, "%.2f" % best_val, best_lam])
        print("[E8] %-12s val=%.2f (λ=%g) test=%.2f mm" % (name, best_val, best_lam, test_err))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "Full_Table1_E8_Probe.csv").open("w", newline="",
                                                     encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["探针输入", "踝足位置误差(mm, test)", "val 误差(mm)", "λ"])
        w.writerows(rows)
    report = """# E8 基于探针的模态互补信息分析

口径：U = V7_Cv5（ckpt_last，wo05 划分）；VT 条件逐窗口读 V/T 时序记忆
m_v / m_t（V7MotionHead 输入前，维度 %d / %d）；目标 = 骨盆相对踝足位置
（4 关节 × 3 维，mm）；ridge 闭式解，λ ∈ {1e-2..1e3} 在 val 上选最优后
test 评估。

| 探针输入 | test 误差 (mm) |
|---|---|
""" % (splits["train"][0].shape[1], splits["train"][1].shape[1])
    for name, test_err, val_err, lam in rows:
        report += "| %s | %s |\n" % (name, test_err)
    report += """
判读（实验计划 E8）：[M_V, M_T] 优于 M_V 与 M_T → 触觉提供额外可解码
运动信息；不要求 M_T 单独优于 M_V。
"""
    (OUT_DIR / "E8_模态互补探针.md").write_text(report, encoding="utf-8")
    print("[E8] done -> %s" % OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
