#!/usr/bin/env python3
"""R_Test8 T2M 上半身（missing-rate 任务书实验 4 / D1）。

T 没有上肢信号，但 T2M 能出"合理上半身"——合理性从哪来？判据：
用 ridge 直接从 t_tok（ρ 网格 T2M 角 repr dump）回归上肢姿态 = **无先验基线**
（纯线性读出，只有 T 信息）；模型 T2M 输出的上半身误差与之的差距 = 先验贡献。

口径注意：ridge 在落盘 split 上按 fit/eval 切分（默认 7:3 session），fseries 的
T2M 上半身行是 split 全域模型输出——完整可比口径需全会话（不 --sessions）。

Usage (touch_gait env, 仓库根执行):
  python results_display/script/r_test8_t2m_upper.py --split val --sessions 6
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

import torch

from anysole.data.dataset import AnySoleDataset, load_split_ids
from anysole.train import load_config
from utils.repr_readout import align_stream, gt_targets, repr_frames
from utils.ridge_probe import ridge_apply, ridge_fit
from r_test7_dropout_ablation import _eval_pose

UPPER_KEYS = ("upper", "hands", "headneck", "l_arm", "r_arm")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test8 T2M 上半身（任务书实验 4/D1）")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", type=Path, default=REPO / "anysole" / "configs" / "v1.yaml")
    parser.add_argument("--fs", type=Path,
                        default=REPO / "results" / "AnySole" / "V4B_joint_and" / "metrics" / "val_fseries.json")
    parser.add_argument("--repr-dir", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--ridge-lam", type=float, default=1.0)
    parser.add_argument("--fit-frac", type=float, default=0.7)
    parser.add_argument("--out", type=Path,
                        default=REPO / "results_display" / "result" / "r_test8_t2m_upper",
                        help="产物目录")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    repr_dir = args.repr_dir or (
        REPO / "results_display" / "result" / "r_test5_rho_grid" / "repr")
    if not args.fs.is_file():
        raise SystemExit("fseries 缺失：%s" % args.fs)
    fseries = json.loads(args.fs.read_text(encoding="utf-8"))
    split = fseries.get("split", args.split)
    args.split = split  # 与 fseries 口径一致
    t2m = fseries["metrics"]["T2M"]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fit = max(1, int(round(len(session_ids) * args.fit_frac)))
    fit_sids, eval_sids = session_ids[:n_fit], session_ids[n_fit:]

    frames = repr_frames(repr_dir, 0, 100, session_ids)  # T2M 角
    Xf, Ypf, _, _, _ = align_stream({s: frames[s] for s in fit_sids}, targets, "t_tok")
    Xe, _, _, _, _ = align_stream({s: frames[s] for s in eval_sids}, targets, "t_tok")
    Xf = torch.from_numpy(Xf).float().to(device)
    Xe = torch.from_numpy(Xe).float().to(device)
    print("t_tok: fit %d 帧 / eval %d 帧" % (Xf.shape[0], Xe.shape[0]))
    W = ridge_fit(Xf, torch.from_numpy(Ypf).float().to(device), args.ridge_lam, device)
    pred = ridge_apply(W, Xe, device)
    ridge_upper = _eval_pose(pred, targets, eval_sids)

    rows = []
    for key, label in (("PA-MPJPE_upper", "upper"), ("PA-MPJPE_hands", "hands")):
        model = t2m.get(key)
        if model is None:
            continue
        ridge = ridge_upper.get(label)
        rows.append({"metric": label, "model_T2M": round(model, 2),
                     "ridge_t_tok": round(ridge, 2),
                     "prior_gap": round(model - ridge, 2)})
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "t2m_upper_prior_gap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("T2M 上半身：模型输出 vs t_tok 无先验 ridge 基线（gap = 先验贡献）：")
    for r in rows:
        print("  %-6s model=%.1f ridge=%.1f gap=%.1f mm" %
              (r["metric"], r["model_T2M"], r["ridge_t_tok"], r["prior_gap"]))
    print("产物：%s/t2m_upper_prior_gap.csv" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
