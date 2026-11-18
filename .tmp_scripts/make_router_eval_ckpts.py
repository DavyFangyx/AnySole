#!/usr/bin/env python3
"""Create eval-ready router checkpoints with the router entry config."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/diagnostics/20261112_v8_gate_frozen"
O = REPO / "results/diagnostics/20261114_router_ablation_v2/eval_ckpts"
O.mkdir(parents=True, exist_ok=True)
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")

def patch(src: Path, dst: Path, model_name: str):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update({
        "model_name": model_name,
        "v7_reliability_route": True,
        "v7_task_routes": True,
        "v7_route_router": True,
        "v7_root_sd": True,
        "v7_root_sd_balance": True,
        "v7_trans_anchor_head": True,
        "tactile_input": "raw96_phys_imu",
        "lambda_tanchor": 0.0,
    })
    torch.save({"config": cfg, "model": ck["model"], "meta": ck.get("meta", {})}, dst)
    print("patched", src.name, "->", dst.name, "model_name", model_name)

# Full router v2rerun (supplementary)
for fold in FOLDS:
    src = D / ("router_ft_enum_cr_%s_l1.0_v2rerun.pt" % fold)
    if src.is_file():
        patch(src, O / ("V8_IMU_Router_Full_%s_eval.pt" % fold),
              "V8_Cv7_PhysIMU_Router")
# Router-only (main-text) checkpoints, if present
for fold in FOLDS:
    src = REPO / ("results/diagnostics/20261114_router_ablation_v2/"
                  "V8_IMU_Router_%s.pt" % fold)
    if src.is_file():
        patch(src, O / ("V8_IMU_Router_%s_eval.pt" % fold),
              "V8_Cv7_PhysIMU_Router_Frozen")
