#!/usr/bin/env python3
"""构建 V8_Cv5VUG 四折 ckpt（CPU）：Cv5 冻结 + unc_expert（IG 监督）+ (a,b)。

对每折：
1. 载入 V8_Cv5_<fold>_s1 的 ckpt → VUG（strict=False，缺 unc_expert）。
2. 用拟合集逐窗数据（ig_frontier_fit_<fold>.npz 的 v_tok/contact）训练
   unc_expert：窗均值 token → IG 标签（冻结接触头 A + 接触 GT）。
3. 拟合 b：a=32 固定，硬约束拟合集 clean 档 g 的 p05 ≥ 0.99（margin），
   可行域内取最小 b（= 约束边界 + 数值余量同前验）。
4. unc_a/unc_b 写入 buffer，ckpt 存盘（config 含 model_name=V8_Cv5VUG_<fold>，
   供 anysole.eval 结构校验）。

用法（touch_gait env，CPU 即可）：
  python results_display/script/build_vug_ckpts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e6_gate_values_occlusion import build_model  # noqa: E402
from ig_contact_pilot import ContactHead  # noqa: E402
from anysole.models.v8_cv5vug import build as build_vug  # noqa: E402

OUT = REPO / "results/diagnostics/20261107_stage2_vig"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
A_FIX = 32.0
CLEAN_P05 = 0.99


def ig_labels(v_tok, contact, head):
    x = torch.from_numpy(v_tok).float()
    c = torch.from_numpy(contact).float()
    with torch.no_grad():
        e_v = torch.nn.functional.binary_cross_entropy_with_logits(
            head(x), c, reduction="none").mean(dim=(1, 2))
        e_0 = torch.nn.functional.binary_cross_entropy_with_logits(
            head(torch.zeros_like(x)), c, reduction="none").mean(dim=(1, 2))
    return (1 - e_v / e_0.clamp(min=1e-6)).numpy()


def main():
    for fi, fold in enumerate(FOLDS):
        base_net, ckpt, tp = build_model("V8_Cv5_%s_s1" % fold, torch.device("cpu"))
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        src_cfg = payload.get("config", {}) or {}
        cfg = {"d_model": 256, "tw": 20, "dropout": 0.1, "pose_layers": 6,
               "v7_task_routes": True, "v7_root_sd": True,
               "v7_v_gate": True, "v7_unc_gate": True, "v7_gate_2sigma": False,
               "v7_trans_anchor_head": True}
        net = build_vug(cfg)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print("[%s] missing=%s unexpected=%s" % (fold, missing, unexpected))
        net.eval()

        # 专家 = 验证版协议（train_unc_experts.py，seed 0 折序 RNG）落盘权重
        exp_pt = torch.load(OUT / ("unc_expert_%s.pt" % fold), map_location="cpu")
        net.unc_expert.load_state_dict(
            {k[len("net."):]: v for k, v in exp_pt["expert"].items()})
        b_fit = float(exp_pt["b"])
        net.unc_a.fill_(A_FIX)
        net.unc_b.fill_(b_fit)
        print("[%s] expert loaded (validated); b=%.2f (a=%.0f)"
              % (fold, b_fit, A_FIX))

        vug_cfg = dict(src_cfg)
        vug_cfg.update({"model_name": "V8_Cv5VUG_%s" % fold,
                        "v7_v_gate": True, "v7_unc_gate": True,
                        "v7_gate_2sigma": False, "lambda_tanchor": 1.0})
        out = OUT / ("V8_Cv5VUG_%s.pt" % fold)
        torch.save({"model": net.state_dict(), "config": vug_cfg, "fold": fold,
                    "unc_a": A_FIX, "unc_b": b_fit}, out)
        print("[out] %s" % out)


if __name__ == "__main__":
    main()
