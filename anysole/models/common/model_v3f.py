"""V3_3F (ruled 2026-10-17): the prior + observation-guidance sampling model.

A fresh, independent architecture — NOT the anysolev2 fusion/gating lineage,
NOT the V3_3D frozen-base lineage, NOT the V3_3E constraint-fusion lineage
(no code is shared with V3_3E; its numpy constraint layer stays its own).

Design (z_note/multimodal_fusion/V3_3F方案.md):
- x = [pose 144 | trans 3 | contact 2] per frame, per-component normalized,
  diffused with unit noise (linear T=1000 schedule, x0 prediction, DDIM).
- Denoiser: DiT-style transformer over frame tokens with a timestep
  embedding and cross-attention to v_tok (the frozen vonly specialist's
  visual tokens — zeroed at 10% of training steps, the unconditional pass;
  no learned null tokens).
- T grounding layer: contact head (f6_soft, pos-weighted BCE — the ONLY
  supervised T output; root velocity / foot height are NOT regressed — the
  probe showed both unreadable from t_tok alone, T信息量判定实验 §3.1) plus
  differentiable contact/ground energies whose root/foot corrections follow
  from the contact constraint mathematically, not from supervision.
- Sampling: DDIM with optional per-step guidance (E_contact + E_ground in
  v1; E_support is refinement-only — its sliding evidence is mixed, 方案
  §2.3).  The three eval conditions are one sampler with switches:
  VT2M = V cond + guidance + refinement; V2M = V cond only; T2M =
  unconditional (V_feat zeroed by the caller) + guidance + refinement.
- Refinement (post-sampling, zero-parameter, differentiable): root
  translation + leg-chain axis-angle deltas (LEG_JOINTS only — the upper
  body is structurally unreachable), a few Adam steps minimizing
  E_ground + E_support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.common.embeddings import SharedEmbeddings, TimestepEmbedding
from anysole.models.common.encoders import ModalEncoders
from anysole.types import (
    D_MODEL,
    FPS,
    FOOT_JOINTS,
    LEG_JOINTS,
    N_JOINTS,
    POSE_DIM,
    STATE_DIM,
    TW,
)
from anysole.utils.diffusion import _timestep_schedule
from anysole.utils.geometry import fk_pose6d, rot6d_to_rotmat, rotmat_to_6d

N_LEG = len(LEG_JOINTS)


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis-angle (..., 3) -> rotation matrix (..., 3, 3)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = w / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    K2 = K @ K
    return eye + torch.sin(theta).unsqueeze(-1) * K \
        + (1.0 - torch.cos(theta)).unsqueeze(-1) * K2


def _split_state(x: torch.Tensor):
    """(B, tw, STATE_DIM) -> (pose, trans, contact)."""
    return x[..., :POSE_DIM], x[..., POSE_DIM:POSE_DIM + 3], x[..., POSE_DIM + 3:]


class V3FDenoiser(nn.Module):
    """x0-prediction denoiser over the full normalized state (B, tw, STATE_DIM)."""

    def __init__(self, dim=D_MODEL, tw=TW, nhead=8, dropout=0.1, n_layers=4,
                 dim_feedforward=1024):
        super().__init__()
        self.state_proj = nn.Linear(STATE_DIM, dim)
        self.t_emb = TimestepEmbedding(dim)
        self.layers = nn.ModuleList([
            _V3FDiTBlock(dim, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, STATE_DIM)
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_std", torch.ones(STATE_DIM))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.state_mean = mean.to(self.state_mean.dtype)
        self.state_std = std.to(self.state_std.dtype)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.state_mean) / self.state_std.clamp(min=1e-2)

    def unnormalize(self, x_n: torch.Tensor) -> torch.Tensor:
        return x_n * self.state_std.clamp(min=1e-2) + self.state_mean

    def forward(self, x_tau, tau, v_tok):
        """x_tau: (B, tw, STATE_DIM) normalized; returns x0_hat (same space)."""
        h = self.state_proj(x_tau)
        h = h + self.t_emb(tau).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, v_tok)
        return self.out_proj(self.norm(h))


class _V3FDiTBlock(nn.Module):
    """Pre-LN self-attention over time, cross-attention to the V memory, FFN."""

    def __init__(self, dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim), nn.Dropout(dropout),
        )

    def forward(self, h, v_tok):
        m1 = self.norm1(h)
        h = h + self.self_attn(m1, m1, m1, need_weights=False)[0]
        m2 = self.norm2(h)
        h = h + self.cross_attn(m2, v_tok, v_tok, need_weights=False)[0]
        h = h + self.ffn(self.norm3(h))
        return h


class V3FContactHead(nn.Module):
    """Per-foot contact logits from the tactile tokens (f6_soft supervision)."""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, t_tok):
        return self.mlp(t_tok)  # (B, tw, 2)


class AnySoleModelV3F(nn.Module):
    def __init__(self, d=D_MODEL, tw=TW, nhead=8, dropout=0.1, denoise_layers=4):
        super().__init__()
        self.d = d
        self.tw = tw
        self.modal = "anysolev2"  # registered-entry machinery in train/eval
        self.v3f = True
        self.v_input = "hrnet"
        self.tactile_input = "raw108"
        self.t_encoder = "foot_conv"
        self.f2_repr = False
        self.embeddings = SharedEmbeddings(dim=d, tw=tw)
        self.encoders = ModalEncoders(
            self.embeddings, dim=d, tw=tw, tactile_input="raw108",
            tactile_direct=False, no_imu=False, v_input="hrnet",
            t_encoder="foot_conv",
        )
        self.denoiser = V3FDenoiser(dim=d, tw=tw, nhead=nhead, dropout=dropout,
                                    n_layers=denoise_layers)
        self.contact_head = V3FContactHead(dim=d)

    def encode(self, V_feat, T_tac):
        """Always-VT encoding: no config conditioning, no null replacement.

        The three eval conditions are sampler switches, not input masks —
        only V_feat itself is zeroed by the caller for the T2M condition
        (the information-free unconditional pass).
        """
        cid = torch.zeros(V_feat.shape[0], dtype=torch.long, device=V_feat.device)
        return self.encoders(V_feat, T_tac, cid)

    def forward(self, V_feat, T_raw, T_phys, x_tau, tau, config_id=None,
                subject_ids=None, T_s2m=None, V_hmr=None, mask_v=None, mask_t=None):
        """Diffusion-era call contract; config_id is accepted and ignored."""
        T_tac = torch.cat([T_raw, T_phys], dim=-1)
        v_tok, t_tok = self.encode(V_feat, T_tac)
        x0_hat = self.denoiser(x_tau, tau, v_tok)
        contact_logits = self.contact_head(t_tok)
        # Native-space split of x0_hat (L_kp world-frame FK supervision,
        # losses.py — the pose-space bottleneck experiment, 2026-10-17).
        pose, trans, contact = _split_state(self.denoiser.unnormalize(x0_hat))
        return {
            "x0_hat": x0_hat,          # normalized full state
            "x0_pose": pose,
            "x0_trans": trans,
            "contact_logits": contact_logits,
            "v_tok": v_tok,
            "t_tok": t_tok,
            "F": x0_hat,               # train.py finite check
        }


def energy_contact(x_contact: torch.Tensor, contact_p: torch.Tensor) -> torch.Tensor:
    """Soft BCE between the sampled contact and the T-head readout (detached)."""
    c = x_contact.clamp(1e-7, 1.0 - 1e-7)
    p = contact_p.detach().clamp(1e-7, 1.0 - 1e-7)
    return -(p * c.log() + (1.0 - p) * (1.0 - c).log()).mean()


def energy_ground(pose, trans_world, contact_p, floor_y, offsets, parents):
    """Contact feet pinned to the floor; swing feet pushed above it (Y-up)."""
    kp = fk_pose6d(pose, trans_world, offsets, parents)
    rel = kp[..., FOOT_JOINTS, :][..., 1] - floor_y.view(-1, 1, 1)  # (B, tw, 2)
    p = contact_p.detach().clamp(0.0, 1.0)
    return (p * rel.square()).mean() + ((1.0 - p) * rel.clamp(max=0.0).square()).mean()


def energy_support(kp: torch.Tensor, contact_p: torch.Tensor) -> torch.Tensor:
    """Contact feet must not slide: world velocity -> 0 (frame-t mask,
    the V3_3E convention)."""
    foot = kp[..., FOOT_JOINTS, :]                                  # (B, tw, 2, 3)
    v = (foot[:, 1:] - foot[:, :-1]) * float(FPS)                   # (B, tw-1, 2, 3)
    p = contact_p.detach()[:, :-1].clamp(0.0, 1.0)                  # (B, tw-1, 2)
    return (p.unsqueeze(-1) * v.square()).mean()


def _apply_guidance(x0_n, denoiser, contact_p, anchor, floor_y, offsets, parents,
                    w_contact=1.0, w_ground=0.5):
    """Push x0_hat down the E_contact + E_ground gradient (the v1 guidance
    set; E_support is refinement-only — 方案 §2.3)."""
    # inference_mode(False) is the canonical exit: enable_grad alone cannot
    # escape inference mode (inference tensors refuse requires_grad).
    with torch.inference_mode(False):
        # clone() first: x0_n is an inference tensor (created by the caller
        # under inference_mode) and refuses requires_grad even after detach.
        x = x0_n.detach().clone().requires_grad_(True)
        pose, trans, contact = _split_state(denoiser.unnormalize(x))
        energy = w_contact * energy_contact(contact, contact_p) \
            + w_ground * energy_ground(pose, trans + anchor, contact_p,
                                       floor_y, offsets, parents)
        grad = torch.autograd.grad(energy, x)[0]
    return x0_n - grad


def refine_v3f(pose, trans_world, contact_p, floor_y, offsets, parents,
               steps=8, lr=0.02, w_support=0.5):
    """Zero-parameter differentiable contact refinement: root translation +
    leg-chain rotation deltas only.  Upper-body joints are not optimization
    variables — T-driven corrections cannot touch them structurally."""
    with torch.inference_mode(False):
        device = pose.device
        B, tw, _ = pose.shape
        base_rot = rot6d_to_rotmat(pose.reshape(B, tw, N_JOINTS, 6)).detach()
        leg_rot = base_rot[:, :, LEG_JOINTS]                       # (B, tw, 8, 3, 3)
        d_trans = torch.zeros(B, tw, 3, device=device, requires_grad=True)
        d_rot = torch.zeros(B, tw, N_LEG, 3, device=device, requires_grad=True)
        opt = torch.optim.Adam([d_trans, d_rot], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            rot = base_rot.clone()
            rot[:, :, LEG_JOINTS] = leg_rot @ _so3_exp(d_rot)
            pose_new = rotmat_to_6d(rot).reshape(B, tw, POSE_DIM)
            trans_new = trans_world + d_trans
            kp = fk_pose6d(pose_new, trans_new, offsets, parents)
            loss = energy_ground(pose_new, trans_new, contact_p, floor_y,
                                 offsets, parents) \
                + w_support * energy_support(kp, contact_p)
            loss.backward()
            opt.step()
        with torch.no_grad():
            rot = base_rot.clone()
            rot[:, :, LEG_JOINTS] = leg_rot @ _so3_exp(d_rot)
            pose_new = rotmat_to_6d(rot).reshape(B, tw, POSE_DIM)
            trans_new = trans_world + d_trans
        return pose_new, trans_new


def sample_v3f(model, V_feat, T_raw, T_phys, *, anchor, floor_y, offsets, parents,
               use_t=True, refine=True, diffusion=None, steps=50, eta=0.0,
               w_contact=1.0, w_ground=0.5, device=None):
    """One full-state DDIM sample.

    Returns the unnormalized (pose, trans, contact) plus the contact head's
    readout.  ``use_t`` toggles the T guidance; ``refine`` toggles the
    post-sampling contact refinement; ``steps``/``eta`` are the DDIM knobs.
    The caller zeroes V_feat for the T2M (unconditional) condition.
    """
    if diffusion is None:
        from anysole.utils.diffusion import GaussianDiffusion
        diffusion = GaussianDiffusion()
    B = V_feat.shape[0]
    tw = model.tw
    device = V_feat.device
    T_tac = torch.cat([T_raw, T_phys], dim=-1)
    v_tok, t_tok = model.encode(V_feat, T_tac)
    contact_logits = model.contact_head(t_tok)
    contact_p = torch.sigmoid(contact_logits).detach()
    den = model.denoiser
    x = torch.randn(B, tw, STATE_DIM, device=device)
    timesteps = _timestep_schedule(diffusion.n_train_steps, int(steps))
    for i, t in enumerate(timesteps):
        tau = torch.full((B,), int(t), device=device, dtype=torch.long)
        x0_n = den(x, tau, v_tok)
        if use_t:
            x0_n = _apply_guidance(x0_n, den, contact_p, anchor, floor_y,
                                   offsets, parents, w_contact, w_ground)
        if int(t) == 0:
            x = x0_n
            break
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        tau_prev = torch.full((B,), int(t_prev), device=device, dtype=torch.long)
        x = diffusion.ddim_step(x, tau, tau_prev, x0_n, eta=eta)
    pose, trans, contact = _split_state(den.unnormalize(x))
    contact = contact.clamp(0.0, 1.0)
    if refine:
        pose, trans_w = refine_v3f(pose, trans + anchor, contact_p, floor_y,
                                   offsets, parents)
        trans = trans_w - anchor
    return {
        "x0_hat": pose,
        "trans_hat": trans,
        "contact": contact,
        "contact_logits": contact_logits,
        "v_tok": v_tok,
        "t_tok": t_tok,
    }
