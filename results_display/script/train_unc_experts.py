#!/usr/bin/env python3
"""逐字复刻 20261107 离线验证版的不确定性专家训练（seed 0 单次、折序 RNG
流），并把专家权重 + 拟合 b 落盘。烘焙脚本只做装载，不再训练。

与验证版 ig_expert_frontier（3/4 折通过线）的唯一差异 = 多了存盘。
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

from ig_contact_pilot import ContactHead  # noqa: E402

OUT = REPO / "results/diagnostics/20261107_stage2_vig"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
A_FIX = 32.0
CLEAN_P05 = 0.99


class UncExpert(torch.nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    torch.manual_seed(0)   # 与验证版一致：单次 seed、折序 RNG 流
    for fold in FOLDS:
        fit = np.load(OUT / ("ig_frontier_fit_%s.npz" % fold))
        head = ContactHead(256)                      # 消费 RNG（与验证版同序）
        head.load_state_dict(torch.load(OUT / ("contact_head_%s_A.pt" % fold),
                                        map_location="cpu"))
        head.eval()
        exp = UncExpert()                            # 消费 RNG（与验证版同序）
        vtr = fit["v_tok"].mean(axis=1)
        ytr = ig_labels(fit["v_tok"], fit["contact"], head)
        opt = torch.optim.Adam(exp.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.from_numpy(vtr).float()
        yt = torch.from_numpy(ytr).float()
        for ep in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(exp(xt), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            u_fit = exp(xt).numpy()
        lv_f = fit["level"]
        u_clean = u_fit[lv_f == 0]
        b_fit = None
        for b in np.arange(-4.0, 8.01, 0.25):
            p05 = float(np.percentile(1.0 / (1.0 + np.exp(-(A_FIX * u_clean + b))), 5))
            if p05 >= CLEAN_P05:
                b_fit = float(b)
                break
        if b_fit is None:
            raise RuntimeError("%s: no b satisfies p05 >= %.2f" % (fold, CLEAN_P05))
        torch.save({"expert": exp.state_dict(), "a": A_FIX, "b": b_fit},
                   OUT / ("unc_expert_%s.pt" % fold))
        print("[%s] saved expert; b=%.2f; fit clean g p05=%.3f"
              % (fold, b_fit,
                 float(np.percentile(1.0 / (1.0 + np.exp(-(A_FIX * u_clean + b_fit))), 5))))


if __name__ == "__main__":
    main()
