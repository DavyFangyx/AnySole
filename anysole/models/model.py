"""AnySole V1: encode V/T, fuse, then pose diffusion + traj/aux heads.

E6.6: ``tactile_input="s2m50"`` feeds the Step2Motion-口径 50-dim tactile
channel instead of cat([T_raw, T_phys]); ``tactile_direct=True`` additionally
gives the pose head a direct cross-attention path to the tactile tokens
(cat([F, t_tok])), bypassing the V-dominated fusion dilution.

--no-imu (E6.8): with tactile_input="s2m50" the tactile input is the 38-dim
T_s2m (IMU channels deleted at data build time) and the encoder has no IMU
encoding groups.
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
from anysole.ablations.insole_drift import InsoleDriftCompensator
from anysole.ablations.insole_drift.physical_tokens import physical_tokens


MODEL_ANYSOLEV1 = "anysolev1"
MODEL_ANYSOLEV1_INSOLE_DRIFT = "anysolev1_insole_drift"
# E6.1: position-space variant (same encoders/fusion/decoder, pose head
# diffuses root-local positions instead of 6D rotations).
MODEL_ANYSOLEV1_POS = "anysolev1_pos"
MODEL_NAMES = (MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS)


class AnySoleModel(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, modal=MODEL_ANYSOLEV1,
                 use_insole_drift=False, templates=None, subject_to_index=None,
                 drift_kwargs=None, pose_layers=6, tactile_input="raw108",
                 tactile_direct=False, no_imu=False):
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
        self.tactile_input = str(tactile_input)
        self.tactile_direct = bool(tactile_direct)
        self.no_imu = bool(no_imu)
        if self.tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50', got %r" % self.tactile_input)
        if self.tactile_direct and self.tactile_input != "s2m50":
            raise ValueError("tactile_direct requires tactile_input='s2m50'")
        if self.no_imu and self.tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50' (raw108 has no IMU channels)")
        if self.use_insole_drift and self.tactile_input != "raw108":
            raise ValueError("insole drift compensation is raw108-only (it rewrites T_raw)")
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw,
            tactile_input=self.tactile_input, tactile_direct=self.tactile_direct,
            no_imu=self.no_imu,
        )
        self.fusion = FusionTransformer(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.pose_head = PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout, n_layers=pose_layers,
            repr="pos" if modal == MODEL_ANYSOLEV1_POS else "6d",
        )
        self.traj_head = TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.aux_heads = AuxHeads(dim=d, tw=tw)

    def forward(self, V_feat, T_raw, T_phys, x_tau, tau, config_id, subject_ids=None, T_s2m=None):
        if self.use_insole_drift:
            # V-only rows contain zero tactile input, so compensation is harmless.
            T_raw, _, _ = self.drift_compensator(T_raw, subject_ids)
            T_phys = physical_tokens(T_raw)
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
            # E6.6b: direct per-frame tactile memory next to the fused F. The
            # null-token dropout (encoders) already zeroes t_tok for V-only
            # configs, so the direct path is masked consistently.
            memory = torch.cat([fused, t_tok], dim=1)
        x0_hat = self.pose_head(x_tau, tau, memory)
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
