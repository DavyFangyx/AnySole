"""AnySole V1: encode V/T, fuse, then pose diffusion + traj/aux heads."""

from __future__ import annotations

import torch.nn as nn

from anysole.models.aux_heads import AuxHeads
from anysole.models.embeddings import SharedEmbeddings
from anysole.models.encoders import ModalEncoders
from anysole.models.fusion import FusionTransformer
from anysole.models.pose_head import PoseHead
from anysole.models.traj_head import TrajHead
from anysole.types import D_MODEL, TW


class AnySoleModel(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1):
        super().__init__()
        self.d = d
        self.tw = tw
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(self.embeddings, dim=d, tw=tw)
        self.fusion = FusionTransformer(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.pose_head = PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout
        )
        self.traj_head = TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.aux_heads = AuxHeads(dim=d, tw=tw)

    def forward(self, V_feat, T_raw, T_phys, x_tau, tau, config_id):
        v_tok, traw_tok, tphys_tok = self.encoders(V_feat, T_raw, T_phys, config_id)
        fused = self.fusion(v_tok, traw_tok, tphys_tok)
        x0_hat = self.pose_head(x_tau, tau, fused)
        v_hat, trans_hat = self.traj_head(fused)
        pressure_hat, vfeat_hat = self.aux_heads(fused)
        return {
            "x0_hat": x0_hat,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "pressure_hat": pressure_hat,
            "vfeat_hat": vfeat_hat,
            "F": fused,
        }
