#!/usr/bin/env python3
"""修改 10 改动 A（2026-11-05）：事后平移锚头拟合（零重训）。

对任一已完成的 V7 Cv 族 ckpt，用 train split 跑一遍前向、取
``out["m_v"].mean(1)``（联合训练时锚头的输入；stopgrad 契约下主干从未被
锚头影响，因此事后拟合与联合训练在信息流上等价），拟合同构的
``Linear(d,d) + GELU + Linear(d,3)``，存为 ckpt 同目录的
``anchor_head.pt``。评估时 ``anysole.eval`` 自动发现并挂载，四个自锚列
（GlobalMPJPE_self / W-MPJPE100_self / RootAbs_self / AnchorErr）即可得。

用法：
  python -m anysole.fit_anchor_head --ckpt <.../ckpt_last.pt> \
      [--device cuda] [--epochs 60] [--batch-size 256] [--lr 1e-3] \
      [--out <ckpt_dir>/anchor_head.pt] [--split train]

口径：train split 拟合、val/test 评估、主干冻结（docstring 已声明）；
provenance 在 eval 侧写 ``anchor_source: "post_hoc"``，与联合训练版
（``"joint"``）区分。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.eval import _load_model
from anysole.train import (condition_inputs, load_config, model_observation_kwargs,
                           move_batch, resolve_device)
from anysole.types import CONFIG_V


def _dataset_from_config(config: Dict[str, object], split: str, tw: int) -> AnySoleDataset:
    split_csv = Path(str(config["split_csv"]))
    session_ids = load_split_ids(split_csv, split)
    return AnySoleDataset(
        mode="train",
        seq_root=Path(str(config["seq_root"])),
        split_csv=split_csv,
        cache_root=Path(str(config["cache_root"])),
        window_length=tw,
        session_ids=session_ids,
        contact_method=str(config["contact_method"]),
        stride=config.get("stride"),
        tactile_input=str(config.get("tactile_input", "raw108")),
        tactile_weight_norm=bool(config.get("tactile_weight_norm", False)),
        no_imu=bool(config.get("no_imu", False)),
        v_input=str(config.get("v_input", "hrnet")),
        f2_repr=bool(config.get("f2_repr", False)),
        smpl_roots=config.get("smpl_roots"),
        cam_id=int(config.get("cam_id", 3)),
        v7_ctx=bool(config.get("v7_prev_ctx", False)),
        v7_ctx3=bool(config.get("v7_ctx3", False) or config.get("v7_ctx4", False)
                     or config.get("v7_ctx5", False)),
        v7_ctx5=bool(config.get("v7_ctx5", False)),
        v7_ar=bool(config.get("v7_ar", False)),
        v7_ar2=bool(config.get("v7_ar2", False)),
        obs_context=config.get("obs_context"),
        long_obs=bool(config.get("v7_vcur", False)),
        load_v5_1_gt=bool(config.get("v5_1", False)),
        t_phys_v2=bool(config.get("t_phys_v2", False)),
        d8_heads=bool(config.get("d8_heads", False)),
        smpl_support_policy=str(config.get("smpl_support_policy", "exclude")),
        v7_s_label=config.get("v7_s_label"),
    )


@torch.no_grad()
def collect_features(model, dataset: AnySoleDataset, device: torch.device,
                     batch_size: int, num_workers: int):
    """One pass over the train split: (m_v window mean, GT anchor)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_windows,
                        pin_memory=device.type == "cuda")
    feats: List[torch.Tensor] = []
    anchors: List[torch.Tensor] = []
    for batch in loader:
        batch = move_batch(batch, device)
        bsz = batch["pose_gt"].shape[0]
        config_id = torch.full((bsz,), CONFIG_V, device=device, dtype=torch.long)
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
        out = model(v_feat, t_raw, t_phys, config_id, batch.get("session_id"),
                    T_s2m=t_s2m, **model_observation_kwargs(batch))
        feats.append(out["m_v"].mean(dim=1).detach().float().cpu())
        anchors.append(batch["trans_anchor"].detach().float().cpu())
    return torch.cat(feats, dim=0), torch.cat(anchors, dim=0)


def fit_head(features: torch.Tensor, anchors: torch.Tensor, device: torch.device,
             epochs: int, lr: float, batch_size: int, seed: int = 20261105) -> dict:
    """Fit the isomorphic anchor head on frozen features (smooth-L1, normalized)."""
    torch.manual_seed(seed)
    d = int(features.shape[1])
    mean = anchors.mean(dim=0)
    scale = anchors.std(dim=0, unbiased=False).clamp(min=1e-2)
    target = (anchors - mean) / scale

    n = features.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, int(0.9 * n))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    x_train = features[train_idx].to(device)
    y_train = target[train_idx].to(device)
    x_val = features[val_idx].to(device)
    y_val = target[val_idx].to(device)

    head = torch.nn.Sequential(
        torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 3)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    for epoch in range(1, int(epochs) + 1):
        head.train()
        order = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            diff = head(x_train[idx]) - y_train[idx]
            loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val = float(F.smooth_l1_loss(
                head(x_val), y_val, beta=1.0)) if x_val.numel() else 0.0
        if val < best["val"]:
            best = {"val": val, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        pred = head(features.to(device)) * scale.to(device) + mean.to(device)
    anchor_err = float((pred - anchors.to(device)).norm(dim=-1).mean()) * 1000.0
    return {"state_dict": best["state"], "anchor_mean": mean, "anchor_scale": scale,
            "val_loss": best["val"], "best_epoch": best["epoch"],
            "train_anchor_err_mm": anchor_err, "n_windows": n}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % ckpt_path)
    saved = dict(checkpoint.get("config", {}))
    if str(saved.get("modal", "")) != "anysolev2":
        raise ValueError("post-hoc anchor fitting requires an anysolev2 (V7) checkpoint")
    config_path = Path(str(saved.get("config") or "anysole/configs/v1.yaml"))
    config = load_config(config_path)
    config.update({k: v for k, v in saved.items() if v is not None})

    device = resolve_device(args.device)
    model = _load_model(checkpoint, config, device)
    if getattr(model, "f2_repr", False) or getattr(model, "ar2", False) or getattr(model, "ar", False):
        raise ValueError("post-hoc anchor fitting is defined for the window-independent "
                         "(non-AR / non-f2) Cv line; free-integration models use "
                         "n_gt_anchors_used = 0 and do not need it")
    tw = int(saved.get("tw", config["tw"]))
    dataset = _dataset_from_config(config, args.split, tw)
    print("fit_anchor_head: %s | %d windows | tw=%d" %
          (ckpt_path.parent.name, len(dataset), tw))

    features, anchors = collect_features(
        model, dataset, device, args.batch_size, args.num_workers)
    print("features: %s | anchors mean %s std %s" %
          (tuple(features.shape), anchors.mean(0).tolist(), anchors.std(0).tolist()))

    result = fit_head(features, anchors, device, args.epochs, args.lr, args.batch_size)
    out_path = Path(args.out) if args.out else ckpt_path.parent / "anchor_head.pt"
    payload = {
        "state_dict": result["state_dict"],
        "anchor_mean": result["anchor_mean"],
        "anchor_scale": result["anchor_scale"],
        "source": "post_hoc",
        "meta": {
            "ckpt": str(ckpt_path),
            "split": args.split,
            "n_windows": result["n_windows"],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "best_epoch": result["best_epoch"],
            "val_smooth_l1": result["val_loss"],
            "train_anchor_err_mm": result["train_anchor_err_mm"],
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }
    torch.save(payload, out_path)
    print("wrote %s | train AnchorErr %.2f mm | best epoch %d (val %.6f)" %
          (out_path, result["train_anchor_err_mm"], result["best_epoch"],
           result["val_loss"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
