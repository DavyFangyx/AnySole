#!/usr/bin/env python3
"""R_Test7 dropout 消融（missing-rate 任务书实验 3）。

C1 对比表（--c1）：训练时**不做**模态 dropout 的模型（--config-probs 1,0,0 训出，
暂名 nodrop）与主线的 fseries 逐指标对比。判据：测试时强行只喂 V / 只喂 T，
nodrop 应明显崩掉（说明不加 dropout，T 就搭 V 的便车）——即 nodrop 的
V2M/T2M 行应显著差于主线，VT2M 行应接近。
C2 单流探针（--c2）：从融合**前**的纯流 token（ρ 网格 repr dump 的 v_tok/t_tok）
ridge 解码 pose / contact / yaw——每条流自己扛得住自己该扛的信息吗。
（口径注意：Ft/Fv 是融合后的表征；单流是编码器原始输出。ridge 在落盘 split
上拟合并评测；train-fit 版本需先跑 --split train 的 ρ 网格。）

Usage (touch_gait env, 仓库根执行):
  python results_display/script/r_test7_dropout_ablation.py --c1 \
      --fs-nodrop results/AnySole/<nodrop>/metrics/val_fseries.json
  python results_display/script/r_test7_dropout_ablation.py --c2 --split val --sessions 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from anysole.data.dataset import AnySoleDataset, load_split_ids
from anysole.train import load_config
from anysole.utils.eval_protocol import (
    ANKLE_FOOT_JOINTS,
    HAND_JOINTS,
    LOWER_JOINTS,
    PART_JOINTS,
    PART_NAMES,
    UPPER_JOINTS,
    _pa_align,
)
from anysole.utils.geometry import fk_pose6d
from utils.repr_readout import align_stream, gt_targets, repr_frames
from utils.ridge_probe import ridge_apply, ridge_fit

KEY_METRICS = ["PA-MPJPE", "MPJPE", "RTE_norm", "yaw_abs_deg", "contact_f1",
               "foot_slide_mm", "joint_limit_viol_elbow", "joint_limit_viol_knee"]
STREAM_CORNERS = {"v_tok": (100, 0), "t_tok": (0, 100)}  # V2M / T2M 角


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test7 dropout 消融（任务书实验 3）")
    parser.add_argument("--c1", action="store_true", help="C1 对比表（主线 vs nodrop fseries）")
    parser.add_argument("--c2", action="store_true", help="C2 单流探针（repr ridge 解码）")
    parser.add_argument("--fs-main", type=Path,
                        default=REPO / "results" / "AnySole" / "V4B_joint_and" / "metrics" / "val_fseries.json")
    parser.add_argument("--fs-nodrop", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", type=Path, default=REPO / "anysole" / "configs" / "v1.yaml")
    parser.add_argument("--repr-dir", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--ridge-lam", type=float, default=1.0)
    parser.add_argument("--fit-frac", type=float, default=0.7,
                        help="前 fit-frac 比例的 session 拟合、其余评测（避免 in-sample 过拟合假象）")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "result" / "r_test7_dropout_ablation",
                        help="产物目录")
    return parser.parse_args(argv)


def run_c1(args: argparse.Namespace) -> int:
    if args.fs_nodrop is None:
        raise SystemExit("C1 需要 --fs-nodrop（nodrop 模型的 fseries 路径）")
    for p in (args.fs_main, args.fs_nodrop):
        if not p.is_file():
            raise SystemExit("fseries 缺失：%s" % p)
    main = json.loads(args.fs_main.read_text(encoding="utf-8"))["metrics"]
    nodrop = json.loads(args.fs_nodrop.read_text(encoding="utf-8"))["metrics"]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in ("VT2M", "V2M", "T2M"):
        for metric in KEY_METRICS:
            if metric not in main[cfg] or metric not in nodrop[cfg]:
                continue
            m, n = main[cfg][metric], nodrop[cfg][metric]
            rows.append({"config": cfg, "metric": metric,
                         "main": round(m, 4), "nodrop": round(n, 4),
                         "delta": round(n - m, 4)})
    with open(out_dir / "c1_dropout_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("C1 对比表（nodrop − main；V2M/T2M 行 delta 应明显 >0 = 没 dropout 就崩）：")
    for r in rows:
        print("  %-5s %-24s main=%9.4f nodrop=%9.4f delta=%+9.4f"
              % (r["config"], r["metric"], r["main"], r["nodrop"], r["delta"]))
    print("产物：%s/c1_dropout_table.csv" % out_dir)
    return 0


def _circ_diff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def run_c2(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    repr_dir = args.repr_dir or (
        REPO / "results_display" / "result" / "r_test5_rho_grid" / "repr")
    session_ids = load_split_ids(Path(config["split_csv"]),
                                 {"val": "val", "test": "test"}[args.split])
    if args.sessions is not None:
        session_ids = session_ids[: args.sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        session_ids=session_ids,
        contact_method=str(config.get("contact_method", "joint_and")),
    )
    targets = gt_targets(dataset, set(session_ids))
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fit_sessions = max(1, int(round(len(session_ids) * args.fit_frac)))
    fit_sids = session_ids[:n_fit_sessions]
    eval_sids = session_ids[n_fit_sessions:]
    for stream, (rV, rT) in STREAM_CORNERS.items():
        frames = repr_frames(repr_dir, rV, rT, session_ids)
        X_all, Y_pose, Y_contact, Y_yaw, _metas = align_stream(frames, targets, stream)
        X_fit = {sid: frames[sid] for sid in fit_sids}
        X_eval = {sid: frames[sid] for sid in eval_sids}
        Xf, Ypf, Ycf, Yyf, _ = align_stream(X_fit, targets, stream)
        Xe, Ype, Yce, Yye, _ = align_stream(X_eval, targets, stream)
        Xf = torch.from_numpy(Xf).float().to(device)
        Xe = torch.from_numpy(Xe).float().to(device)
        print("%s: fit %d 帧 × %d 维 / eval %d 帧" % (stream, Xf.shape[0], Xf.shape[1], Xe.shape[0]))
        for name, Yf, Ye, post in (
            ("pose", torch.from_numpy(Ypf).float().to(device),
             torch.from_numpy(Ype).float().to(device), "pose"),
            ("contact", torch.from_numpy(Ycf).float().to(device),
             torch.from_numpy(Yce).float().to(device), "contact"),
            ("yaw", torch.from_numpy(Yyf).float().to(device),
             torch.from_numpy(Yye).float().to(device), "yaw"),
        ):
            W = ridge_fit(Xf, Yf, args.ridge_lam, device)
            pred = ridge_apply(W, Xe, device)
            if post == "pose":
                err = _eval_pose(pred, targets, eval_sids)
                for agg, v in err.items():
                    rows.append({"stream": stream, "target": "pose_PA_" + agg,
                                 "value": round(v, 2)})
            elif post == "contact":
                acc = ((pred > 0.5) == (Ye > 0.5)).float().mean().item()
                rows.append({"stream": stream, "target": "contact_acc",
                             "value": round(acc, 4)})
            else:
                yaw_p = torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()
                yaw_g = torch.atan2(Ye[:, 0], Ye[:, 1]).cpu().numpy()
                mae = float(np.abs(_circ_diff(yaw_p, yaw_g)).mean() * 180.0 / np.pi)
                rows.append({"stream": stream, "target": "yaw_MAE_deg",
                             "value": round(mae, 2)})
    with open(out_dir / "c2_singlestream_readout.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("C2 单流读出（期望：v_tok 强于全局/yaw，t_tok 强于接触/下肢）：")
    for r in rows:
        print("  %s %-14s %s" % (r["stream"], r["target"],
                                 {k: v for k, v in r.items() if k not in ("stream", "target")}))
    print("产物：%s/c2_singlestream_readout.csv" % out_dir)
    return 0


def _eval_pose(pred: torch.Tensor, targets: dict, session_ids) -> dict:
    """解码姿态按帧 Procrustes + 部位聚合（mm）。"""
    total = 0
    err = torch.zeros(24, device=pred.device)
    pa = torch.zeros(24, device=pred.device)
    n = 0
    idx = 0
    for sid in session_ids:
        t = targets.get(sid)
        if t is None:
            continue
        T = len(t["pose_gt"])
        pose = pred[idx: idx + T].reshape(1, T, -1)
        trans = torch.from_numpy(t["gt_trans"]).float().to(pred.device)[None]
        kp = fk_pose6d(pose, trans,
                       torch.from_numpy(t["offsets"]).float().to(pred.device)[None],
                       torch.from_numpy(t["parents"]).long().to(pred.device)[None])[0]
        kp_gt = torch.from_numpy(t["kp_gt"]).float().to(pred.device)
        e = torch.linalg.vector_norm(kp - kp_gt, dim=-1)      # (T,24)
        aligned = _pa_align(kp, kp_gt)
        p = torch.linalg.vector_norm(aligned - kp_gt, dim=-1)
        err += e.sum(dim=0)
        pa += p.sum(dim=0)
        n += T
        idx += T
    err /= n
    pa /= n
    out = {"overall": float(pa.mean().item()) * 1000.0}
    for agg, joints in (("upper", UPPER_JOINTS), ("lower", LOWER_JOINTS),
                        ("anklefoot", ANKLE_FOOT_JOINTS), ("hands", HAND_JOINTS)):
        flat = np.asarray(joints).ravel().tolist()
        out[agg] = float(pa[flat].mean().item()) * 1000.0
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.c1:
        return run_c1(args)
    if args.c2:
        return run_c2(args)
    parse_args(["--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
