#!/usr/bin/env python3
"""E6 门控 × 遮挡（2026-11-05）：门值随遮挡是否打开（模型侧）。

固定 Cv5 act8020 的 ckpt_last（双门），在 val 会话上逐个遮挡档
（occ000–occ100）前向 VT，直接读取 route_gate g ∈ (0,1)
（g 越大 = 越信任触觉支路；Cv7 无门，等价 g ≡ 1）。

落盘 results_display/Etest_act8020/E6_ContactPhaseAnalysis/gate_occlusion/：
  gate_values_occlusion.npz          逐档逐窗门值（四组 + query 轴）
  gate_values_occlusion_summary.csv  逐档组均值 / 中位 / >0.5 占比
  gate_values_occlusion.json         汇总 + 每会话均值

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/e6_gate_values_occlusion.py --device cuda:0
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
sys.path.insert(0, str(REPO / "z_note/HMR2.0/hmr2_gait"))

from anysole.data.dataset import (AnySoleDataset, collate_windows,  # noqa: E402
                                  load_split_ids)
from anysole.registry import get_builder  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch  # noqa: E402
from anysole.types import CONFIG_VT  # noqa: E402

OUT = (REPO / "results_display" / "Etest_act8020"
       / "E6_ContactPhaseAnalysis" / "gate_occlusion")
GROUPS = ("vel", "other", "left", "right")
GROUP_QUERIES = ((0,), tuple(range(1, 6)), (6, 8), (7, 9))
LEVELS = [("000", "anysole/configs/v1_act8020.yaml")] + [
    ("%03d" % n, "anysole/configs/v1_act8020_occ%03d.yaml" % n)
    for n in range(10, 101, 10)]


def build_model(cell: str, device: torch.device):
    ckpt_txt = REPO / "results" / "cells" / cell / "checkpoint_used.txt"
    ckpt_path = Path(ckpt_txt.read_text(encoding="utf-8").strip())
    if "_fS" in cell:                 # 四折 LSO：<fam>_fS..._s1
        name = cell.split("_fS")[0]
    else:
        name = cell.split("_act8020")[0]
    entry = get_builder(name)
    cfg = {"d_model": 256, "tw": 20, "dropout": 0.1, "pose_layers": 6,
           **entry.STRUCTURE, **entry.CONFIG_EXTRA}
    net = entry.build(cfg).to(device).eval()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict"))
    if state is None:
        raise RuntimeError("checkpoint without model state: %s" % ckpt_path)
    net.load_state_dict(state)
    t_phys_v2 = bool(entry.STRUCTURE.get("t_phys_v2", False))
    return net, ckpt_path, t_phys_v2


def build_loader(config_path: str, split: str, batch_size: int,
                 t_phys_v2: bool = False) -> DataLoader:
    config = load_config(REPO / config_path)
    tw = int(config.get("tw", 20))
    split_csv = Path(config["split_csv"])
    session_ids = load_split_ids(split_csv, split)
    dataset = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=split_csv,
        cache_root=Path(config["cache_root"]), window_length=tw,
        session_ids=session_ids, contact_method=str(config["contact_method"]),
        tactile_input="raw108", no_imu=False, v_input="hrnet",
        f2_repr=False, smpl_roots=config.get("smpl_roots"), cam_id=int(config.get("cam_id", 3)),
        t_phys_v2=t_phys_v2,
        stride=None, obs_context=tw, smpl_support_policy="exclude")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, collate_fn=collate_windows)


def run_level(net, loader, device, level):
    gates, sids = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            cid = torch.full((batch["V_feat"].shape[0],), CONFIG_VT,
                             dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
            out = net(v_feat, t_raw, t_phys, cid, batch.get("subject_ids"),
                      T_s2m=t_s2m)
            g = out.get("route_gate")
            if g is None:
                raise RuntimeError("模型没有 route_gate（%s 不是双门结构）" % level)
            gates.append(g[..., 0].cpu().numpy())  # (B,tw,10)
            sids.extend(batch["session_id"])
    gates = np.concatenate(gates)  # (N,tw,10)
    return gates, sids


def summarize(level: str, gates: np.ndarray, sids: list[str]) -> list[dict]:
    rows = []
    for k, name in enumerate(GROUPS):
        vals = gates[..., list(GROUP_QUERIES[k])].reshape(-1)
        rows.append({
            "occ": level, "group": name,
            "mean": float(vals.mean()), "std": float(vals.std()),
            "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "open_frac": float((vals > 0.5).mean()),
        })
    per_session = {}
    for sid in sorted(set(sids)):
        idx = [i for i, s in enumerate(sids) if s == sid]
        per_session[sid] = {
            name: float(gates[idx][..., list(GROUP_QUERIES[k])].mean())
            for k, name in enumerate(GROUPS)}
    return rows, per_session


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["occ", "group", "mean", "std", "median", "p05", "p95", "open_frac"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="V7_Cv5_act8020_s1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", default="val")
    ap.add_argument("--fold", default="",
                    help="四折 LSO fold（如 fS5S6S11）；设置后 yaml 用 "
                         "v1_fold_<fold>.yaml / v1_occ<level>_fold_<fold>.yaml")
    ap.add_argument("--s13out", action="store_true",
                    help="fS6S7S13 折的 woS13 口径：yaml 用 S6S7 split")
    ap.add_argument("--levels", default="",
                    help="逗号分隔遮挡档（如 000,090；缺省 = 全部 000–100）")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    net, ckpt_path, t_phys_v2 = build_model(args.cell, device)
    print("[ckpt] %s  t_phys_v2=%s" % (ckpt_path, t_phys_v2))

    all_rows, arrays, per_session_all = [], {}, {}
    if args.fold:
        fold = args.fold
        fold_yaml = fold[1:] if fold.startswith("f") else fold
        if args.s13out:
            fold_yaml = "S6S7"
        levels = [("000", "anysole/configs/v1_fold_%s.yaml" % fold_yaml)] + [
            ("%03d" % n,
             "anysole/configs/v1_occ%03d_fold_%s.yaml" % (n, fold_yaml))
            for n in (20, 40, 60, 80, 100)]
    else:
        levels = LEVELS
    if args.levels:
        wanted = {item.strip() for item in args.levels.split(",") if item.strip()}
        levels = [item for item in levels if item[0] in wanted]
    for level, config_path in levels:
        loader = build_loader(config_path, args.split, args.batch_size,
                              t_phys_v2=t_phys_v2)
        gates, sids = run_level(net, loader, device, level)
        arrays[level] = gates
        rows, per_session = summarize(level, gates, sids)
        all_rows.extend(rows)
        per_session_all[level] = per_session
        print("[occ%s] windows=%d mean=%s" % (
            level, gates.shape[0],
            " ".join("%s=%.3f" % (r["group"], r["mean"])
                     for r in rows)))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "gate_values_occlusion_summary.csv", all_rows)
    np.savez_compressed(out_dir / "gate_values_occlusion.npz", **arrays)
    (out_dir / "gate_values_occlusion.json").write_text(json.dumps({
        "cell": args.cell, "checkpoint": str(ckpt_path), "split": args.split,
        "groups": list(GROUPS), "levels": per_session_all,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[npz] %s" % (out_dir / "gate_values_occlusion.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
