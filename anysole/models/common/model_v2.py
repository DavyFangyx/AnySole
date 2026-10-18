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
from anysole.models.common.embeddings import SharedEmbeddings, relative_time_pe
from anysole.models.common.encoders import ModalEncoders
from anysole.models.common.fusion import FusionTransformer
from anysole.models.common.pose_head import PoseHead
from anysole.models.common.t_residual import TResidualBranch
from anysole.models.common.traj_head import TrajHead
from anysole.types import CONFIG_VT, D_MODEL, POSE_DIM, TW, V_FEAT_DIM


MODEL_ANYSOLEV2 = "anysolev2"


class AnySoleModelV2(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1,
                 pose_layers=6, tactile_input="raw108",
                 tactile_direct=False, no_imu=False, v_input="hrnet",
                 t_encoder="linear", f2_repr=False, pose_parts=3,
                 soft_parts=False, gate="none", part_joints=None,
                 residual_t=False, visual_only=False, pressure_head="legacy",
                 pressure_time_pe="before_fusion", time_pe_scale=40.0):
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
        self.visual_only = bool(visual_only)
        self.pressure_head = str(pressure_head)
        self.pressure_time_pe_enabled = self.pressure_head == "time_pe"
        self.pressure_time_pe = str(pressure_time_pe)
        self.time_pe_scale = float(time_pe_scale)
        if self.pressure_head not in ("legacy", "time_pe"):
            raise ValueError("pressure_head must be 'legacy' or 'time_pe'")
        if self.pressure_time_pe != "before_fusion":
            raise ValueError("pressure_time_pe must be 'before_fusion'")
        if self.time_pe_scale != 40.0:
            raise ValueError("visual time_pe_scale is frozen at 40.0 nominal frames/s")
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
        # V3_3C: frozen-V corrector reading the tactile stream only.
        self.t_residual = TResidualBranch(d, POSE_DIM) if self.residual_t else None

    def forward(self, V_feat, T_raw, T_phys, config_id, subject_ids=None, T_s2m=None,
                V_hmr=None, mask_v=None, mask_t=None):
        """F0b: same inputs as V1 minus the diffusion pair (x_tau, tau).

        The visual stream follows ``v_input``: hrnet = V_feat (2051-d);
        hmr_gvhmr (F1) = V_hmr (1156-d GVHMR channel).  The tactile stream
        follows ``tactile_input``: raw108 = cat([T_raw, T_phys]) (encoded by
        LinearTemporalEncoder or FootConvEncoder per t_encoder); s2m50 =
        T_s2m.  ``mask_v/mask_t`` (bool, B×tw) are the ρ-grid per-frame
        null-token masks (missing-rate 实验 1); None = old path.
        """
        if self.visual_only:
            raise ValueError("visual students accept only forward_visual(V_feat, frame_valid, timestamps)")
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
        if self.t_residual is not None:
            # V3_3C: the frozen base never sees the real tactile stream — it
            # runs in its V-only regime under every condition (the fused F is
            # bit-identical to its V-only F), and the real t_tok reaches the
            # model only through the residual branch.  This is what makes the
            # init-time guarantee exact: out == base V-only output, and the
            # branch can only add what the tactile stream carries.
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
            # V3_3C: the real tactile stream feeds only this branch; the
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
            # V3_3C: detached tactile correction (zero under V-only/T-only;
            # smoke tests and probes read this).
            "x0_residual": residual.detach() if residual is not None else None,
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

    def forward_visual(self, V_feat, frame_valid, timestamps):
        """Strict single-view RGB inference, with supervision kept outside.

        N is the number of nonempty views after the caller flattens groups.
        The only cross-view interaction belongs in the output loss, never in
        this function. Invalid RGB is removed before any learned operation.
        """
        if self.v_input != "hrnet" or self.f2_repr or self.tactile_direct or self.residual_t:
            raise ValueError("strict visual forward requires the non-F2 HRNet regression model")
        if (V_feat.ndim != 3 or V_feat.shape[1:] != (self.tw, V_FEAT_DIM)
                or not V_feat.is_floating_point()):
            raise ValueError("V_feat must be floating [N,%d,%d]" % (self.tw, V_FEAT_DIM))
        if V_feat.shape[0] == 0:
            raise ValueError("filter empty/all-invalid views before forward_visual")
        if frame_valid.shape != V_feat.shape[:2] or frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be bool [N,T], True means valid")
        if timestamps.shape != frame_valid.shape or not timestamps.is_floating_point():
            raise ValueError("timestamps must be floating [N,T] target times")
        if frame_valid.device != V_feat.device or timestamps.device != V_feat.device:
            raise ValueError("visual features, validity and timestamps must share a device")
        if not bool(frame_valid.any(dim=1).all()):
            raise ValueError("filter all-invalid views before forward_visual")
        if not bool(torch.isfinite(V_feat[frame_valid]).all()):
            raise ValueError("valid visual features contain NaN or Inf")
        if (not bool(torch.isfinite(timestamps).all())
                or not bool((timestamps[:, 1:] > timestamps[:, :-1]).all())):
            raise ValueError("target timestamps must be finite and strictly increasing")

        time_pe = None
        if self.pressure_head == "time_pe":
            time_pe = relative_time_pe(timestamps, self.d, self.time_pe_scale, V_feat.dtype)
        v_tok, t_tok = self.encoders.forward_visual(V_feat, frame_valid, time_pe)
        memory_padding_mask = torch.cat([~frame_valid, ~frame_valid], dim=1)
        # None preserves the old full-valid attention kernel and exact legacy
        # V-only prediction, including the constant-pressure control.
        padding = memory_padding_mask if bool(memory_padding_mask.any()) else None
        fused = self.fusion(v_tok, t_tok, memory_padding_mask=padding)
        fused = torch.where(memory_padding_mask.unsqueeze(-1), torch.zeros_like(fused), fused)
        x0_hat = self.pose_head(F=fused, frame_valid=frame_valid,
                                 memory_padding_mask=padding)
        v_hat, trans_hat = self.traj_head(fused, frame_valid=frame_valid, timestamps=timestamps)
        pressure_hat, vfeat_hat = self.aux_heads(fused)
        out = {
            "x0_hat": x0_hat,
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "pressure_hat": pressure_hat,
            "vfeat_hat": vfeat_hat,
            "F": fused,
            "v_tok": v_tok,
            "t_tok": t_tok,
            "frame_valid": frame_valid,
            "memory_padding_mask": memory_padding_mask,
            "part_logits": self.pose_head.part_logits if self.soft_parts else None,
            "part_assignment": self.pose_head.part_assignment() if self.soft_parts else None,
            "x0_residual": None,
        }
        if self.gate == "sigma":
            out.update({
                "gate_sigma": self.pose_head.last_gate_sigma(),
                "gate_g": self.pose_head.last_gate_g(),
                "part_hypotheses": self.pose_head.last_part_hypotheses(),
                "pose_mean": self.pose_head.pose_mean,
                "pose_std": self.pose_head.pose_std,
                "sigma_frozen": bool(self.pose_head.decoder.layers[-1].sigma_frozen),
            })
        return out
