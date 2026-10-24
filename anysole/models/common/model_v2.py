"""AnySole V2: the F-series model (fix_plan_v2.md).

F0b: structure identical to AnySole V1 (encoders / fusion / traj / aux heads
are the same modules as model.py), with one difference — the pose head runs in
``head_mode="regress"``: a direct F -> pose regression, no diffusion chain.
The forward signature drops the diffusion pair (x_tau, tau); L_pose applies
directly to the regression output (losses.py unchanged).

``AnySoleModel`` (anysolev1) and ``anysole/diffusion.py`` stay untouched so old
checkpoints keep loading strictly.  F1+ steps (HMR features, F2
representation, F4a foot-conv encoder, ...) replace the V1 building blocks
in this file.  (v2 §F5 part decoder is archived — see fix_plan_v3.md.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.common.aux_heads import AuxHeads
from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.fusion import FusionTransformer
from anysole.models.common.pose_head import PoseHead
from anysole.models.common.position_head import PositionHead
from anysole.models.common.t_residual import TResidualBranch
from anysole.models.common.traj_head import TrajHead
from anysole.types import CONFIG_VT, D_MODEL, FPS, POSE_DIM, TW


MODEL_ANYSOLEV2 = "anysolev2"


class AnySoleModelV2(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1,
                 pose_layers=6, tactile_input="raw108",
                 tactile_direct=False, no_imu=False, v_input="hrnet",
                 t_encoder="linear", f2_repr=False, pose_parts=3,
                 soft_parts=False, gate="none", part_joints=None,
                 residual_t=False, position_head=False):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = MODEL_ANYSOLEV2
        self.tactile_input = str(tactile_input)
        self.tactile_direct = bool(tactile_direct)
        self.no_imu = bool(no_imu)
        self.v_input = str(v_input)
        self.t_encoder = str(t_encoder)
        self.f2_repr = bool(f2_repr)
        self.pose_parts = int(pose_parts)
        self.soft_parts = bool(soft_parts)
        self.gate = str(gate)
        self.residual_t = bool(residual_t)
        self.position_head = bool(position_head)
        if self.position_head and self.residual_t:
            raise ValueError("position_head and residual_t are mutually exclusive (V3_Dv1 vs V3_Cv1)")
        if self.position_head and self.f2_repr:
            raise ValueError("position_head requires f2_repr=False (V3_Dv1)")
        if self.v_input not in ("hrnet", "hmr_gvhmr"):
            raise ValueError("v_input must be 'hrnet' or 'hmr_gvhmr', got %r" % self.v_input)
        if self.tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50', got %r" % self.tactile_input)
        if self.t_encoder not in ("linear", "foot_conv"):
            raise ValueError("t_encoder must be 'linear' or 'foot_conv', got %r" % self.t_encoder)
        if self.t_encoder == "foot_conv" and self.tactile_input != "raw108":
            raise ValueError("t_encoder='foot_conv' requires tactile_input='raw108' (F4a)")
        if self.tactile_direct and self.tactile_input != "s2m50":
            raise ValueError("tactile_direct requires tactile_input='s2m50'")
        if self.no_imu and self.tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50' (raw108 has no IMU channels)")
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw,
            tactile_input=self.tactile_input, tactile_direct=self.tactile_direct,
            no_imu=self.no_imu, v_input=self.v_input, t_encoder=self.t_encoder,
        )
        self.fusion = FusionTransformer(dim=d, tw=tw, nhead=nhead, dropout=dropout)
        self.pose_head = PoseHead(
            self.embeddings, dim=d, tw=tw, nhead=nhead, dropout=dropout,
            n_layers=pose_layers, repr="6d", head_mode="regress",
            n_parts=self.pose_parts, soft_parts=self.soft_parts, gate=self.gate,
            part_joints=part_joints,
        )
        self.traj_head = TrajHead(dim=d, tw=tw, nhead=nhead, dropout=dropout,
                                  out_dim=4 if self.f2_repr else 3)
        self.aux_heads = AuxHeads(dim=d, tw=tw)
        # V3_Cv1: frozen-V corrector reading the tactile stream only.
        self.t_residual = TResidualBranch(d, POSE_DIM) if self.residual_t else None
        # V3_Dv1: joint position-space readout of the pre-fusion tokens.
        self.position_head = PositionHead(d, tw, nhead=nhead, dropout=dropout) \
            if self.position_head else None

    def forward(self, V_feat, T_raw, T_phys, config_id, subject_ids=None, T_s2m=None,
                V_hmr=None, mask_v=None, mask_t=None, v_tok_override=None):
        """F0b: same inputs as V1 minus the diffusion pair (x_tau, tau).

        The visual stream follows ``v_input``: hrnet = V_feat (2051-d);
        hmr_gvhmr (F1) = V_hmr (1156-d GVHMR channel).  The tactile stream
        follows ``tactile_input``: raw108 = cat([T_raw, T_phys]) (encoded by
        LinearTemporalEncoder or FootConvEncoder per t_encoder); s2m50 =
        T_s2m.  ``mask_v/mask_t`` (bool, B×tw) are the ρ-grid per-frame
        null-token masks (missing-rate 实验 1); None = old path.

        ``v_tok_override``（TTA 实验注入点，2026-10-21）：给定 (B,tw,dim) 张量时
        跳过视觉编码器的输出，直接用其作为 v_tok（推理时优化的 per-view 变量）；
        默认 None = 原行为不变。
        """
        if self.tactile_input == "s2m50":
            if T_s2m is None:
                raise ValueError("T_s2m is required when tactile_input='s2m50'")
            T_tac = T_s2m
        else:
            T_tac = torch.cat([T_raw, T_phys], dim=-1)
        if self.v_input == "hmr_gvhmr" and V_hmr is None:
            raise ValueError("V_hmr is required when v_input='hmr_gvhmr'")
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        if v_tok_override is not None:
            v_tok = v_tok_override  # TTA 实验注入点
        if self.t_residual is not None or self.position_head is not None:
            # V3_Cv1/V3_Dv1: the frozen base never sees the real tactile stream
            # — it runs in its V-only regime under every condition (the fused
            # F is bit-identical to its V-only F), and the real t_tok reaches
            # the model only through the residual/position branch.  This is
            # what makes the init-time guarantee exact: out == base V-only
            # output, and the branch can only add what the tactile stream
            # carries.
            fused = self.fusion(v_tok, self.embeddings.null_t.expand_as(t_tok))
        else:
            fused = self.fusion(v_tok, t_tok)
        memory = fused
        if self.tactile_direct:
            # E6.6b-style direct tactile memory next to the fused F (the null
            # token dropout already zeroes t_tok for V-only configs).
            memory = torch.cat([fused, t_tok], dim=1)
        # Regression head: the fused memory is the only pose source.  Keyword
        # call — the pose head rejects positional diffusion-style arguments.
        x0_hat = self.pose_head(F=memory)
        residual = None
        if self.t_residual is not None:
            # V3_Cv1: the real tactile stream feeds only this branch; the
            # mask zeroes the correction under V-only/T-only (outputs stay
            # bit-identical to the frozen base's V-only regime), and at init
            # the zeroed final layer makes the VT output identical to the
            # frozen base's V-only output (the exact VT == V guarantee).
            residual = self.t_residual(t_tok)
            t_present = (
                config_id.to(device=x0_hat.device, dtype=torch.long).reshape(-1) == CONFIG_VT
            ).to(x0_hat.dtype).view(-1, 1, 1)
            x0_hat = x0_hat + t_present * residual
        v_hat, trans_hat = self.traj_head(fused)
        pos_corr = None
        if self.position_head is not None:
            # V3_Dv1: the corrected root velocity (zero-init dv keeps trans_hat
            # bit-identical to the frozen base at init) and the pelvis-relative
            # six-joint position correction (meters).  The head is active under
            # every condition — the encoders already null-replaced the missing
            # stream — so the V-only/T-only outputs get the same readout.
            pos_corr, v_corr = self.position_head(v_tok, t_tok)
            v_hat = v_hat + v_corr
            trans_hat = torch.cumsum(v_hat[..., :3], dim=1) / float(FPS)
        if hasattr(self, "pressure_readout"):
            # V3_Bv2/Bv3: query pressure readout replaces the linear aux
            # t_rec (same λ_trec supervision position).  B windows are fully
            # valid at fixed 40 Hz, so the head derives both frame_valid and
            # timestamps; callers with per-frame validity pass them through.
            pressure_hat = self.pressure_readout(fused)["pressure_hat"]
            vfeat_hat = self.aux_heads.v_rec(fused[:, :self.tw])
        else:
            pressure_hat, vfeat_hat = self.aux_heads(fused)
        out = {
            "x0_hat": x0_hat,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "pressure_hat": pressure_hat,
            "vfeat_hat": vfeat_hat,
            "F": fused,
            # ρ 网格 repr dump（missing-rate 实验 2c/5 消费）：编码后 token 流。
            "v_tok": v_tok,
            "t_tok": t_tok,
            # V3-3/V4A: the learned (N_PARTS, N_JOINTS) assignment logits for
            # L_assign in losses.py, plus the temperature-scaled softmax A
            # actually used by the forward (None for every other config).
            "part_logits": self.pose_head.part_logits if self.soft_parts else None,
            "part_assignment": self.pose_head.part_assignment() if self.soft_parts else None,
            # V3_Cv1: detached tactile correction (zero under V-only/T-only;
            # smoke tests and probes read this).
            "x0_residual": residual.detach() if residual is not None else None,
            # V3_Dv1: pelvis-relative six-joint position correction (meters);
            # eval_protocol applies it to the FK keypoints (soft skeleton),
            # losses.py supervises it (L_pos).
            "pos_corr": pos_corr,
        }
        if self.gate == "sigma":
            # Beta-NLL sigma supervision inputs (losses.py): the last layer's
            # raw sigma logits (grad-carrying), detached gate weights
            # (attribution), detached per-slot hypotheses + assignment and the
            # pose stats (the normalized-space error target).
            out.update({
                "gate_sigma": self.pose_head.last_gate_sigma(),
                "gate_g": self.pose_head.last_gate_g(),
                "part_hypotheses": self.pose_head.last_part_hypotheses(),
                "pose_mean": self.pose_head.pose_mean,
                "pose_std": self.pose_head.pose_std,
                "sigma_frozen": bool(self.pose_head.decoder.layers[-1].sigma_frozen),
            })
        return out
