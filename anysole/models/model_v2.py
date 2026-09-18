"""AnySole V2: the F-series model (fix_plan_v2.md).

F0b: structure identical to AnySole V1 (encoders / fusion / traj / aux heads
are the same modules as model.py), with one difference — the pose head runs in
``head_mode="regress"``: a direct F -> pose regression, no diffusion chain.
The forward signature drops the diffusion pair (x_tau, tau); L_pose applies
directly to the regression output (losses.py unchanged).

``AnySoleModel`` (anysolev1) and ``anysole/diffusion.py`` stay untouched so old
checkpoints keep loading strictly.  F1+ steps (HMR features, F2
representation, F5 part decoder + gating, ...) replace the V1 building blocks
in this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.aux_heads import AuxHeads
from anysole.models.embeddings import SharedEmbeddings
from anysole.models.encoders import ModalEncoders
from anysole.models.fusion import FusionTransformer
from anysole.models.pose_head import PoseHead
from anysole.models.traj_head import TrajHead
from anysole.types import D_MODEL, TW


MODEL_ANYSOLEV2 = "anysolev2"


class AnySoleModelV2(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1,
                 pose_layers=6, tactile_input="raw108",
                 tactile_direct=False, no_imu=False):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = MODEL_ANYSOLEV2
        self.tactile_input = str(tactile_input)
        self.tactile_direct = bool(tactile_direct)
        self.no_imu = bool(no_imu)
        if self.tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50', got %r" % self.tactile_input)
        if self.tactile_direct and self.tactile_input != "s2m50":
            raise ValueError("tactile_direct requires tactile_input='s2m50'")
        if self.no_imu and self.tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50' (raw108 has no IMU channels)")
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw,
            tactile_input=self.tactile_input, tactile_direct=self.tactile_direct,
            no_imu=self.no_imu,
        )
        self.fusion = FusionTransformer(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.pose_head = PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, repr="6d", head_mode="regress",
        )
        self.traj_head = TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.aux_heads = AuxHeads(dim=d, tw=tw)

    def forward(self, V_feat, T_raw, T_phys, config_id, subject_ids=None, T_s2m=None):
        """F0b: same inputs as V1 minus the diffusion pair (x_tau, tau)."""
        if self.tactile_input == "s2m50":
            if T_s2m is None:
                raise ValueError("T_s2m is required when tactile_input='s2m50'")
            T_tac = T_s2m
        else:
            T_tac = torch.cat([T_raw, T_phys], dim=-1)
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id)
        fused = self.fusion(v_tok, t_tok)
        memory = fused
        if self.tactile_direct:
            # E6.6b-style direct tactile memory next to the fused F (the null
            # token dropout already zeroes t_tok for V-only configs).
            memory = torch.cat([fused, t_tok], dim=1)
        # Regression head: the fused memory is the only pose source.  Keyword
        # call — the pose head rejects positional diffusion-style arguments.
        x0_hat = self.pose_head(F=memory)
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
