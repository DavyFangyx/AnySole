"""V4_Av3 (2026-10-25): V4 correction branch mounted on the frozen
V3_Bv1 backbone — the B-base carrier line (用户裁定「B 的 V 部分作为
通路、T 解毒」的折中方案).

Design rationale (z_note/multimodal_fusion/V4_方案 §6 V4_Av3 条目):
- B's V stream was trained WITH T in the shared fusion transformer — its
  V2M readout carries the fusion tax (+1.1 PA vs the E-class V).  Frozen
  as the scaffold, it gives the T-side correction a well-defined job on
  EVERY axis: on healthy windows the correction learns to undo the tax
  (L_corr target = gt − v residual includes it), on V-fail windows it
  learns the ground-related help.  Unlike the frozen-E carrier (T had
  nothing to do on cam3) and unlike scratch (no scaffold, absorption).
- Two-stage training: stage 1 = the V3_Bv1 production run (ckpt_best,
  frozen); stage 2 = this run trains ONLY the correction branch
  (PartXAttn + ContactGate + PositionCorrHeads, zero-init) + the contact
  head (f6_soft, from zero).  The backbone never moves.

State-dict contract: the backbone is constructed EXACTLY as V3_Bv1's
entry (AnySoleModelV2 with the B STRUCTURE) so the B ckpt's 378 keys
load key-identical; the correction branch + contact head are new keys.
The correction classes are imported from model_v4_1 (same mechanism as
V4_Av1/V4_Av2 — reuse over copy: no state-dict independence issue since
V4_Av3 never loads a V4 ckpt; deviation from the copy convention noted).

Three conditions = three readouts:
- V2M = B V-only forward (frozen; eval zeroes T);
- VT2M = B VT forward + pos_corr (soft skeleton at metric level);
- T2M = B T-only forward + pos_corr_prior (second forward, V zeroed,
  no_grad — the frozen backbone's own T2M row).
"""

from __future__ import annotations

import torch

from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.models.common.model_v4_1 import (
    ContactGate,
    PartXAttn,
    PositionCorrHeads,
    REACH_PARTS,
    SG7Smooth,
    V41ContactHead,
)

__all__ = ["AnySoleModelV4B"]


class AnySoleModelV4B(AnySoleModelV2):
    """B backbone (state-dict identical to V3_Bv1) + V4 correction branch."""

    def __init__(self, d, tw, nhead=8, dropout=0.1, pose_layers=6, beta=1.0):
        # B backbone verbatim: the V3_Bv1 entry's STRUCTURE, part_joints=None.
        super().__init__(
            d=d, tw=tw, nhead=nhead, dropout=dropout, pose_layers=pose_layers,
            tactile_input="raw108", tactile_direct=False, no_imu=False,
            v_input="hrnet", t_encoder="foot_conv", f2_repr=False,
            pose_parts=9, soft_parts=True, gate="sigma", part_joints=None,
        )
        self.beta = float(beta)
        # Correction branch + contact head (from zero; the only trainable
        # components of stage 2).
        self.contact_head = V41ContactHead(dim=d)
        self.corr_attn = PartXAttn(dim=d, tw=tw, nhead=nhead, dropout=dropout,
                                   beta=beta)
        self.contact_gate = ContactGate(n_reach=len(REACH_PARTS))
        self.corr_heads = PositionCorrHeads(dim=d)
        self.sg7 = SG7Smooth()

    _CORRECTION_PREFIXES = (
        "contact_head.", "corr_attn.", "contact_gate.", "corr_heads.", "sg7.")

    def carrier_param_ids(self):
        """骨架（B backbone）参数 id 集——V4_Av4 双阶段精调的学习率分组
        （carrier_lr_mult=0.1）。修正分支 + 接触头不在骨架内（1.0）。"""
        return {id(p) for name, p in self.named_parameters()
                if not name.startswith(self._CORRECTION_PREFIXES)}

    def _correction(self, z, t_tok, contact_logits):
        """Correction branch on one source's part tokens -> Δx (B, tw, 18).

        z: (B, tw, 9, dim) B pose-head part tokens (slots 5-8 = l/r leg,
        l/r foot — the learned A converged to the natural body-part
        partition, verified against the B production ckpt); t_tok:
        (B, tw, dim); contact_logits: (B, tw, 2).  Structural deletion:
        only the REACH_PARTS slots are sliced out (V4_Av1 同款)."""
        z_r = z[:, :, REACH_PARTS, :]                 # (B, tw, 4, dim)
        attn = self.corr_attn(z_r, t_tok)             # XAttn joint readout
        a_p = self.contact_gate(contact_logits)       # (B, tw, 4), zero-init
        dz = torch.tanh(a_p).unsqueeze(-1) * attn     # contact-gated
        zc = z_r + dz                                 # z'_p = z_p + Δz_p
        return self.sg7(self.corr_heads(zc))          # (B, tw, 18), SG-7

    def forward(self, V_feat, T_raw, T_phys, config_id=None, subject_ids=None,
                T_s2m=None, V_hmr=None, mask_v=None, mask_t=None,
                v_tok_override=None):
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        # VT pass: B pipeline verbatim + return_z（config_id 零化与 B 同口径
        # —— eval 三条件的 V2M/VT2M 由评估侧输入零化区分，训练 = VT 窗）。
        v_tok, t_tok = self.encoders(V_feat, T_tac, config_id, V_hmr=V_hmr,
                                     mask_v=mask_v, mask_t=mask_t)
        if v_tok_override is not None:
            v_tok = v_tok_override  # TTA 注入点（与 V4_Av1_TTA 同款）
        fused = self.fusion(v_tok, t_tok)
        x0_hat, z_v = self.pose_head(F=fused, return_z=True)
        v_hat, trans_hat = self.traj_head(fused)
        contact_logits = self.contact_head(t_tok)
        pos_corr = self._correction(z_v, t_tok, contact_logits)   # VT2M row
        if hasattr(self, "pressure_readout"):
            pressure_hat = self.pressure_readout(fused)["pressure_hat"]
            vfeat_hat = self.aux_heads.v_rec(fused[:, : self.tw])
        else:
            pressure_hat, vfeat_hat = self.aux_heads(fused)
        # T pass（V 零化、冻结骨架 no_grad）：T2M 行 = B 自身的 T-only 读出
        # + pos_corr_prior。z_t 只作修正分支的输入值（XAttn q），不建骨架
        # 梯度图。
        with torch.no_grad():
            v_tok_t, t_tok_t = self.encoders(
                torch.zeros_like(V_feat), T_tac,
                torch.zeros(V_feat.shape[0], dtype=torch.long,
                            device=V_feat.device),
                V_hmr=(torch.zeros_like(V_hmr) if V_hmr is not None else None))
            fused_t = self.fusion(v_tok_t, t_tok_t)
            x0_prior, z_t = self.pose_head(F=fused_t, return_z=True)
            v_prior, trans_prior = self.traj_head(fused_t)
        pos_corr_prior = self._correction(z_t, t_tok_t, contact_logits)
        return {
            "x0_hat": x0_hat,            # B VT 读出（骨架冻结）
            "v_hat": v_hat,
            "trans_hat": trans_hat,
            "x0_prior": x0_prior,        # B T-only 读出（冻结）
            "v_prior": v_prior,
            "trans_prior": trans_prior,
            "contact_logits": contact_logits,
            "pos_corr": pos_corr,        # VT2M 行修正（软骨架，metrics）
            "pos_corr_prior": pos_corr_prior,   # T2M 行修正
            "F": fused,                  # train.py finite check
            "v_tok": v_tok,
            "t_tok": t_tok,
            "pressure_hat": pressure_hat,
            "vfeat_hat": vfeat_hat,
            "part_logits": self.pose_head.part_logits if self.soft_parts else None,
            "part_assignment": self.pose_head.part_assignment() if self.soft_parts else None,
        }
