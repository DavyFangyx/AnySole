"""Auxiliary reconstruction heads on fused memory slices."""

from __future__ import annotations

import torch.nn as nn

from anysole.types import D_MODEL, T_RAW_DIM, TW, V_FEAT_DIM


class AuxHeads(nn.Module):
    def __init__(self, dim=D_MODEL, tw=TW):
        super().__init__()
        self.tw = tw
        self.t_rec = nn.Linear(dim, T_RAW_DIM)
        self.v_rec = nn.Linear(dim, V_FEAT_DIM)

    def forward(self, F):
        pressure_hat = self.t_rec(F[:, self.tw : 2 * self.tw])
        vfeat_hat = self.v_rec(F[:, : self.tw])
        return pressure_hat, vfeat_hat
