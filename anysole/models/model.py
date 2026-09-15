"""AnySole V1: encode V/T, fuse, then pose diffusion + traj/aux heads."""

from __future__ import annotations

import torch.nn as nn
import torch

from anysole.models.aux_heads import AuxHeads
from anysole.models.embeddings import SharedEmbeddings
from anysole.models.encoders import ModalEncoders
from anysole.models.fusion import FusionTransformer
from anysole.models.pose_head import PoseHead
from anysole.models.traj_head import TrajHead
from anysole.types import D_MODEL, TW
from anysole.ablations.insole_drift import InsoleDriftCompensator
from anysole.ablations.insole_drift.physical_tokens import physical_tokens


MODEL_ANYSOLEV1 = "anysolev1"
MODEL_ANYSOLEV1_INSOLE_DRIFT = "anysolev1_insole_drift"
MODEL_NAMES = (MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT)


class AnySoleModel(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, modal=MODEL_ANYSOLEV1, use_insole_drift=False, templates=None, subject_to_index=None, drift_kwargs=None, pose_layers=6):
        super().__init__()
        self.d = d
        self.tw = tw
        if modal not in MODEL_NAMES:
            raise ValueError("Unknown modal %r; expected one of %s" % (modal, MODEL_NAMES))
        self.modal = modal
        self.use_insole_drift = bool(use_insole_drift or modal == MODEL_ANYSOLEV1_INSOLE_DRIFT)
        if self.use_insole_drift:
            if templates is None:
                raise ValueError("templates are required when use_insole_drift=True")
            self.drift_compensator = InsoleDriftCompensator(templates, subject_to_index or {}, **(drift_kwargs or {}))
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(self.embeddings, dim=d, tw=tw)
        self.fusion = FusionTransformer(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.pose_head = PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout, n_layers=pose_layers
        )
        self.traj_head = TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.aux_heads = AuxHeads(dim=d, tw=tw)

    def forward(self, V_feat, T_raw, T_phys, x_tau, tau, config_id, subject_ids=None):
        if self.use_insole_drift:
            # V-only rows contain zero tactile input, so compensation is harmless.
            T_raw, _, _ = self.drift_compensator(T_raw, subject_ids)
            T_phys = physical_tokens(T_raw)
        v_tok, t_tok = self.encoders(V_feat, T_raw, T_phys, config_id)
        fused = self.fusion(v_tok, t_tok)
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
