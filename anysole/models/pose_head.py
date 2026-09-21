"""Pose diffusion head: predict clean pose x0 from x_tau, tau, and fused memory F.

MDM/RoHM-style decoder (structural rewrite of the V1 head). The probes showed the
old head converged to a condition-only solution g(F) that ignored x_tau, because
the timestep signal (additive, after self-attention, ~18% amplitude) was too weak
to learn the noise-level gating. The rewrite fixes that:

- ``[t_emb; x_tokens]`` prepend: the diffusion-step embedding is the first token
  and is visible to every self-attention layer, so each layer can gate how much
  of x_tau it uses by noise level (MDM/RoHM do exactly this).
- Cross-attention to F in every layer (MDM decoder block), instead of one
  post-self-attention cross-attn.
- 6-8 decoder layers instead of 2+1+1.
- Pre-LayerNorm on every residual stream (self-attn / cross-attn / FFN).
- GELU everywhere (the old self-attn encoder used the PyTorch default ReLU).
- Per-dim mean/std normalization of the pose representation (MDM/RoHM
  normalize the motion representation before diffusing); train.py fits the
  stats on the training set and they are saved with the checkpoint.

E6.1: ``repr="pos"`` switches the diffused target from 6D rotations to
root-local 3D positions (23 non-root joints) without touching the decoder
structure - only the per-joint I/O width changes (6 -> 3). Default "6d" keeps
the pre-E6 behavior exactly.

F0b: ``head_mode="regress"`` turns the head into a direct F -> pose regression
(fix_plan_v2.md): the x_tau input projections are replaced by a learned query
``(1, tw, n_tokens, dim)`` and ``forward`` takes only the fused memory F —
no x_tau/tau, no timestep token.  ``head_mode="diffusion"`` (default) keeps
the pre-F0b layout byte-for-byte so old checkpoints load strictly.

V3-2: ``n_parts=9`` (regress only) swaps the SMPL-24 joint queries (3 groups) for
9 part queries with one unembed head per part (fix_plan_v3.md §V3-2).
Everything else — decoder layers, pose stats, time PE, normalization — is
shared; ``n_parts=3`` (default) keeps the pre-V3-2 path byte-for-byte.

V3-3 (fix_plan_v3.md §V3-3/§8): two orthogonal mechanisms on top of n_parts=9.
- ``soft_parts=True`` replaces the hard part heads with a learned soft
  assignment matrix A (9 slots x 24 joints): every slot emits a full-pose
  hypothesis and A blends them per joint.  A is initialized to reproduce the
  frozen PART_JOINTS grouping, then adapts during training (see losses.py
  L_assign: per-joint entropy + slot-load balance).  This is mechanism C's
  structural half: the grouping itself is prior knowledge.
- ``gate="sigma"`` replaces the plain decoder layers with GatedDecoderLayer
  (mechanism B): self-attn -> dual cross-attn to the V/T views of F ->
  per-slot sigma routing over [V, T, prior] with a learnable per-part prior
  bias tau_p.  The prior branch (g_∅) is the residual z — the self-attention
  output with no modality injected (mechanism C's host, plan §V3-3).
  tau_p is initialized FROM the assignment (A when soft, PART_JOINTS
  otherwise): T-informative parts (root/legs/feet) get low tau, V-informative
  parts (torso/headneck/arms) get high tau.  Submodule names mirror
  nn.TransformerDecoderLayer so self-attn / V-branch cross-attn / norms / FFN
  warm-start by name from a standard-decoder checkpoint.
The two switches are independent: soft_parts alone = learned grouping without
gating; gate alone = gating on the hard PART_JOINTS grouping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anysole.models.embeddings import SharedEmbeddings
from anysole.types import (
    BODY_JOINTS,
    D_MODEL,
    LEFT_LEG_JOINTS,
    N_JOINTS,
    N_PARTS,
    PART_JOINTS,
    POSE_DIM,
    POSE_POS_DIM,
    RIGHT_LEG_JOINTS,
    TW,
)


def _part_place_idx() -> torch.Tensor:
    """(POSE_DIM,) long tensor mapping joint-space position -> flat index in
    the part-order concatenation (V3-2: 9 unembed heads emit in PART_JOINTS
    order, which is NOT joint order — r_leg/r_foot are swapped)."""
    flat = []  # (joint-space pos, part-flat index)
    for joints in PART_JOINTS:
        for j in joints:
            for d in range(6):
                flat.append((6 * j + d, len(flat)))
    flat.sort()
    return torch.tensor([idx for _, idx in flat], dtype=torch.long)


# V3-3: PART_JOINTS entries split by which modality informs them, driving the
# tau_p prior-bias init (plan §V3-3): T-informative parts (root/legs/feet)
# start with LOW tau (modality streams preferred), V-informative parts
# (torso/headneck/arms) with HIGH tau (prior preferred when only T is present).
_T_INFORMATIVE_PARTS = (0, 5, 6, 7, 8)
_V_INFORMATIVE_PARTS = (1, 2, 3, 4)
_T_INFORMATIVE_JOINTS = tuple(j for p in _T_INFORMATIVE_PARTS for j in PART_JOINTS[p])
_V_INFORMATIVE_JOINTS = tuple(j for p in _V_INFORMATIVE_PARTS for j in PART_JOINTS[p])
_TAU_BASE = 0.0
_TAU_SCALE = 2.0


def _part_logits_init() -> torch.Tensor:
    """(N_PARTS, N_JOINTS) logits whose softmax reproduces PART_JOINTS: +3 on
    the member joints, -3 elsewhere, so training starts from the manual
    grouping instead of a random scramble (V3-3 soft assignment init)."""
    logits = torch.full((N_PARTS, N_JOINTS), -3.0)
    for p, joints in enumerate(PART_JOINTS):
        logits[p, joints] = 3.0
    return logits


def _tau_from_assignment(assign: torch.Tensor) -> torch.Tensor:
    """(N_PARTS,) tau_p prior bias derived from a (N_PARTS, N_JOINTS)
    assignment: low for T-informative slots, high for V-informative slots."""
    t_mass = assign[:, _T_INFORMATIVE_JOINTS].sum(dim=1)
    v_mass = assign[:, _V_INFORMATIVE_JOINTS].sum(dim=1)
    return _TAU_BASE + _TAU_SCALE * (v_mass - t_mass)


class GatedDecoder(nn.Module):
    """Container holding the GatedDecoderLayer stack under a ``layers``
    ModuleList so state-dict keys read ``decoder.layers.<i>.<submodule>`` —
    exactly the nn.TransformerDecoder layout, which is what makes the
    by-name warm-start from a standard-decoder checkpoint work."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z, F, masks):
        for layer in self.layers:
            z = layer(z, F, masks)
        return z


class GatedDecoderLayer(nn.Module):
    """V3-3 decoder block (fix_plan_v3.md §V3-3): pre-LN self-attn -> dual
    cross-attn to the V/T views of F -> per-slot sigma routing (three-way gate
    [V, T, prior], tau_p per-part prior bias) -> pre-LN FFN.

    Submodule names mirror nn.TransformerDecoderLayer (self_attn,
    multihead_attn, norm1/2/3, linear1/2, dropout*, activation) so the shared
    parts warm-start by name from a standard-decoder checkpoint; the V-branch
    cross-attn is named ``multihead_attn`` on purpose — it inherits the
    trained full-F cross-attn weights.  The T-branch cross-attn
    (``multihead_attn_t``), the sigma MLP and tau_p are new and stay fresh.
    """

    def __init__(self, dim, nhead=8, dim_feedforward=1024, dropout=0.1,
                 tw=TW, n_parts=N_PARTS):
        super().__init__()
        self.dim = dim
        self.tw = tw
        self.n_parts = n_parts
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        # V view: cross-attn over F's V-source tokens (masked T).
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        # T view: cross-attn over F's T-source tokens (masked V).
        self.multihead_attn_t = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        # Per-slot per-stream trust logits sigma_V/sigma_T.  Input = [z, e_V,
        # e_T] — the three gate candidates — so sigma can see what each stream
        # would inject (2026-09-21 V3-3B audit: feeding only z made layer-1
        # sigma structurally input-blind — layer-1 z is the self-attention of
        # the fixed query — and sigma never learned).  Near-zero init so the
        # gate starts from the tau_p prior bias alone.
        self.sigma = nn.Linear(3 * dim, 2)
        nn.init.normal_(self.sigma.weight, std=0.01)
        nn.init.zeros_(self.sigma.bias)
        self.sigma_frozen = False  # train.py sets for the first sigma_freeze_frac of steps
        # Learnable per-part prior bias; initialized from the part assignment
        # by PoseHead (set_tau_p) after construction.
        self.tau_p = nn.Parameter(torch.zeros(n_parts))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_t = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.last_g = None      # (B, tw, n_parts, 3) detach, for gate_summary()
        self.last_sigma = None  # (B, tw, n_parts, 2) raw, for the beta-NLL loss

    def set_tau_p(self, tau: torch.Tensor) -> None:
        with torch.no_grad():
            self.tau_p.copy_(torch.as_tensor(tau, dtype=self.tau_p.dtype, device=self.tau_p.device))

    def forward(self, z, F, masks):
        # NOTE: the memory parameter is named F like the fusion output, which
        # shadows the module-level `torch.nn.functional as F` — functional
        # calls in here must go through nn.functional explicitly.
        ignore_t_mask, ignore_v_mask = masks  # (B, S) bool, True = ignored token
        # Self-attention (identical to the standard block).
        z = z + self.dropout1(self.self_attn(self.norm1(z), self.norm1(z), self.norm1(z), need_weights=False)[0])
        n = self.norm2(z)
        # Dual cross-attn: e_V reads only F's V-source tokens, e_T only T's.
        e_v = self.dropout2(
            self.multihead_attn(n, F, F, key_padding_mask=ignore_t_mask, need_weights=False)[0]
        )
        e_t = self.dropout2_t(
            self.multihead_attn_t(n, F, F, key_padding_mask=ignore_v_mask, need_weights=False)[0]
        )
        # Per-slot sigma routing: g = softmax([sigma_V, sigma_T, tau_p]); the
        # prior branch (g_∅) is the residual z with no modality injected.
        # sigma sees [z, e_V, e_T] so it can compare what each stream would
        # inject (fix for the V3-3B "static gate" audit, 2026-09-21).
        # tau_p is clamped to [-4, 4]: the V3_3B audit showed the unclamped
        # bias drifting to +9.8/-3.8, which saturates the softmax
        # (g -> [0,0,1] constant) and kills the sigma gradient (g(1-g) ~ 0) —
        # the gate then routes nothing.  With the clamp a saturated prior
        # still leaves the streams able to compete (e.g. sigma ~ 1.8 reaches
        # g_V = 0.1 at tau = +4), and init tau = +-2 stays inside the range so
        # it keeps a nonzero gradient from the first step.
        batch, seq = z.shape[:2]
        if self.sigma_frozen:
            # First sigma_freeze_frac of training (plan §3): sigma fixed at 0
            # (variance 1), the gate runs on the tau_p prior bias alone.
            sigma = torch.zeros(batch, self.tw, self.n_parts, 2, device=z.device)
        else:
            sigma = self.sigma(torch.cat([z, e_v, e_t], dim=-1)).view(batch, self.tw, self.n_parts, 2)
        tau = self.tau_p.clamp(-4.0, 4.0).view(1, 1, self.n_parts, 1).expand(batch, self.tw, self.n_parts, 1)
        g = nn.functional.softmax(torch.cat([sigma, tau], dim=-1), dim=-1)  # (B, tw, n_parts, 3)
        g_flat = g.reshape(batch, seq, 3)
        z = z + g_flat[..., 0:1] * e_v + g_flat[..., 1:2] * e_t
        # FFN (identical to the standard block).
        z = z + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(self.norm3(z))))))
        self.last_g = g.detach()
        self.last_sigma = sigma
        return z


class PoseHead(nn.Module):
    def __init__(self, embeddings, dim=D_MODEL, tw=TW, nhead=8, dim_feedforward=1024, dropout=0.1, n_layers=6, repr="6d", head_mode="diffusion", n_parts=3, soft_parts=False, gate="none"):
        super().__init__()
        if head_mode not in ("diffusion", "regress"):
            raise ValueError("Unknown head_mode %r; expected 'diffusion' or 'regress'" % head_mode)
        if n_parts not in (3, 9):
            raise ValueError("n_parts must be 3 or 9, got %r" % n_parts)
        if n_parts != 3 and (head_mode != "regress" or repr != "6d"):
            raise ValueError("n_parts=9 requires head_mode='regress' and repr='6d' (V3-2)")
        if gate not in ("none", "sigma"):
            raise ValueError("gate must be 'none' or 'sigma', got %r" % gate)
        if soft_parts and n_parts != 9:
            raise ValueError("soft_parts requires n_parts=9 (V3-3)")
        if gate == "sigma" and n_parts != 9:
            raise ValueError("gate='sigma' requires n_parts=9 (V3-3)")
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.n_layers = n_layers
        self.repr = repr
        self.head_mode = head_mode
        self.n_parts = n_parts
        self.soft_parts = bool(soft_parts)
        self.gate = gate
        self._last_hypo = None  # (N_PARTS, B, tw, N_JOINTS, 6) detach, gate mode
        if repr == "pos":
            # All 23 non-root SMPL joints as root-local positions (root excluded).
            self.n_body = len(BODY_JOINTS) - 1
            self.n_left = len(LEFT_LEG_JOINTS)
            self.n_right = len(RIGHT_LEG_JOINTS)
            self.per_joint = 3
            self.pose_dim = POSE_POS_DIM
            n_tokens = self.n_body + self.n_left + self.n_right
            group_ids = torch.zeros(n_tokens, dtype=torch.long)
            group_ids[self.n_body : self.n_body + self.n_left] = 1
            group_ids[self.n_body + self.n_left :] = 2
        elif repr == "6d":
            self.n_body = len(BODY_JOINTS)
            self.n_left = len(LEFT_LEG_JOINTS)
            self.n_right = len(RIGHT_LEG_JOINTS)
            self.per_joint = 6
            self.pose_dim = POSE_DIM
            if n_parts == 9:
                # V3-2: one query token per part (9) instead of one per joint
                # (24 in 3 groups); the part embedding IS the part identity.
                group_ids = torch.arange(N_PARTS, dtype=torch.long)
            else:
                # Tokens are concatenated [body, left, right] in _embed and
                # _forward_regress; tag by TOKEN position, not joint index —
                # SMPL-24 interleaves leg joints with the body, so joint-index
                # tagging gives 16/24 tokens the wrong group embedding.
                group_ids = torch.zeros(N_JOINTS, dtype=torch.long)
                group_ids[self.n_body : self.n_body + self.n_left] = 1
                group_ids[self.n_body + self.n_left :] = 2
        else:
            raise ValueError("Unknown pose repr %r; expected '6d' or 'pos'" % repr)

        self.group_emb = nn.Embedding(self.n_parts, dim)
        nn.init.normal_(self.group_emb.weight, std=0.02)
        self.register_buffer("group_ids", group_ids, persistent=True)
        n_tokens = self.n_body + self.n_left + self.n_right
        if n_parts == 9:
            n_tokens = N_PARTS
        self.n_tokens = n_tokens

        if head_mode == "regress":
            # F0b: replace the x_tau input projections with a fixed learned
            # query — one free token position per (group joint, frame), no
            # diffused input and no timestep token.
            self.query = nn.Parameter(torch.zeros(1, tw, n_tokens, dim))
            nn.init.normal_(self.query, std=0.02)
        else:
            self.proj_body = nn.Linear(self.per_joint, dim)
            self.proj_left = nn.Linear(self.per_joint, dim)
            self.proj_right = nn.Linear(self.per_joint, dim)
            # Learned "position" for the prepended diffusion-step token.
            self.timestep_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.timestep_token, std=0.02)

        # MDM decoder block per layer: pre-LN self-attn -> pre-LN cross-attn(F)
        # -> pre-LN FFN, GELU. Full attention on [t_emb; x_tokens].
        # gate="sigma" (V3-3) replaces the standard block with the gated one;
        # its submodule names mirror TransformerDecoderLayer for warm-start.
        if gate == "sigma":
            self.decoder = GatedDecoder([
                GatedDecoderLayer(
                    dim, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, tw=tw, n_parts=n_parts,
                )
                for _ in range(n_layers)
            ])
        else:
            layer = nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        if n_parts == 9 and soft_parts:
            # V3-3 soft assignment: one head per slot, each emitting a full
            # 144-d pose hypothesis; the learned A matrix (part_logits) blends
            # the hypotheses per joint.  A is initialized to reproduce the
            # frozen PART_JOINTS grouping (joint order output, no scatter).
            self.out_parts = nn.ModuleList(
                [nn.Linear(dim, POSE_DIM) for _ in range(N_PARTS)]
            )
            self.part_logits = nn.Parameter(_part_logits_init())
            self.out_body = self.out_left = self.out_right = None
        elif n_parts == 9:
            # V3-2: one unembed head per part; outputs concatenated in PART
            # order then scattered back to joint order via part_place_idx.
            self.out_parts = nn.ModuleList(
                [nn.Linear(dim, len(joints) * 6) for joints in PART_JOINTS]
            )
            self.register_buffer("part_place_idx", _part_place_idx(), persistent=True)
            self.out_body = self.out_left = self.out_right = None
        else:
            self.out_body = nn.Linear(dim, self.per_joint)
            self.out_left = nn.Linear(dim, self.per_joint)
            self.out_right = nn.Linear(dim, self.per_joint)

        if gate == "sigma":
            # V3-3: derive the per-part prior bias from the part assignment
            # (the initialized softmax A when soft_parts, the frozen
            # PART_JOINTS otherwise): T-informative parts (root/legs/feet)
            # start with LOW tau, V-informative (torso/headneck/arms) HIGH.
            if soft_parts:
                assign = F.softmax(self.part_logits.detach(), dim=0)
            else:
                assign = torch.zeros(N_PARTS, N_JOINTS)
                for p, joints in enumerate(PART_JOINTS):
                    assign[p, joints] = 1.0
            tau = _tau_from_assignment(assign)
            for block in self.decoder.layers:
                block.set_tau_p(tau)

        # Per-dim mean/std of the pose representation over the training set.
        # Defaults are identity so a fresh/unfitted head still runs; train.py
        # calls set_stats() before the first step and the fitted values live in
        # the checkpoint state_dict.
        self.register_buffer("pose_mean", torch.zeros(self.pose_dim), persistent=True)
        self.register_buffer("pose_std", torch.ones(self.pose_dim), persistent=True)

    def set_stats(self, mean, std):
        """Fix the (pose_dim,) normalization stats used in forward."""
        mean = torch.as_tensor(mean, dtype=self.pose_mean.dtype, device=self.pose_mean.device)
        std = torch.as_tensor(std, dtype=self.pose_std.dtype, device=self.pose_std.device)
        if mean.shape != (self.pose_dim,) or std.shape != (self.pose_dim,):
            raise ValueError("pose stats must have shape (%d,)" % self.pose_dim)
        self.pose_mean.copy_(mean)
        # Floor keeps near-constant dims from being amplified by 1e5x.
        self.pose_std.copy_(std.clamp(min=1e-2))

    def _embed(self, x_tau):
        batch, tw, _ = x_tau.shape
        def gather(joints):
            idx = torch.tensor(
                [j * self.per_joint + d for j in joints for d in range(self.per_joint)],
                device=x_tau.device, dtype=torch.long,
            )
            return x_tau.index_select(-1, idx).reshape(batch, tw, len(joints), self.per_joint)
        if self.repr == "pos":
            body_joints = tuple(j - 1 for j in BODY_JOINTS if j != 0)
        else:
            body_joints = tuple(BODY_JOINTS)
        body = gather(body_joints)
        left_joints = tuple(j if self.repr == "6d" else j - 1 for j in LEFT_LEG_JOINTS)
        right_joints = tuple(j if self.repr == "6d" else j - 1 for j in RIGHT_LEG_JOINTS)
        left = gather(left_joints)
        right = gather(right_joints)
        tokens = torch.cat(
            [self.proj_body(body), self.proj_left(left), self.proj_right(right)],
            dim=2,
        )
        tokens = tokens + self.embeddings.time_pe.to(dtype=tokens.dtype).view(1, tw, 1, self.dim)
        tokens = tokens + self.group_emb(self.group_ids).to(dtype=tokens.dtype).view(
            1, 1, self.group_ids.shape[0], self.dim
        )
        return tokens.reshape(batch, tw * self.group_ids.shape[0], self.dim)

    def _unembed(self, h):
        batch = h.shape[0]
        if self.n_parts == 9 and self.soft_parts:
            # V3-3: 9 slots -> 9 full-pose hypotheses -> per-joint blend by
            # the learned soft assignment A (joint order out, no scatter).
            tokens = h.view(batch, self.tw, self.n_tokens, self.dim)
            parts = torch.stack(
                [head(tokens[:, :, p]) for p, head in enumerate(self.out_parts)],
                dim=0,
            )  # (N_PARTS, B, tw, POSE_DIM)
            parts = parts.reshape(N_PARTS, batch, self.tw, N_JOINTS, 6)
            assign = F.softmax(self.part_logits, dim=0)  # (N_PARTS, N_JOINTS)
            out = torch.einsum("pj,pbtjc->btjc", assign, parts)
            return out.reshape(batch, self.tw, POSE_DIM)
        if self.n_parts == 9:
            # V3-2: 9 part queries -> 9 part heads -> scatter to joint order.
            tokens = h.view(batch, self.tw, self.n_tokens, self.dim)
            parts = torch.cat(
                [head(tokens[:, :, p]) for p, head in enumerate(self.out_parts)],
                dim=-1,
            )  # (B, tw, POSE_DIM) in PART_JOINTS order
            return parts.gather(-1, self.part_place_idx.expand(batch, self.tw, -1))
        n_tokens = self.n_body + self.n_left + self.n_right
        tokens = h.view(batch, self.tw, n_tokens, self.dim)
        body = self.out_body(tokens[:, :, : self.n_body, :])
        left = self.out_left(tokens[:, :, self.n_body : self.n_body + self.n_left, :])
        right = self.out_right(tokens[:, :, self.n_body + self.n_left :, :])
        part_values = torch.cat([body, left, right], dim=2)
        if self.repr == "6d":
            joint_order = tuple(BODY_JOINTS) + tuple(LEFT_LEG_JOINTS) + tuple(RIGHT_LEG_JOINTS)
        else:
            joint_order = tuple(j - 1 for j in BODY_JOINTS if j != 0) + tuple(j - 1 for j in LEFT_LEG_JOINTS) + tuple(j - 1 for j in RIGHT_LEG_JOINTS)
        flat = part_values.reshape(batch, self.tw, -1)
        out = flat.new_zeros(batch, self.tw, self.pose_dim)
        for k, j in enumerate(joint_order):
            out[..., j * self.per_joint:(j + 1) * self.per_joint] = flat[..., k * self.per_joint:(k + 1) * self.per_joint]
        return out

    def forward(self, x_tau=None, tau=None, F=None):
        """Dispatch by head_mode.

        Diffusion keeps the ``(x_tau, tau, F)`` positional signature so every
        pre-F0b caller/checkpoint works unchanged.  Regress takes only the
        fused memory — call it with ``F=...`` (positional args are rejected so
        a diffusion-style call cannot silently feed x_tau in as the memory).
        """
        if self.head_mode == "regress":
            if x_tau is not None or tau is not None:
                raise ValueError(
                    "regress head takes the fused memory F only; call it as "
                    "pose_head(F=memory), not with diffusion-style (x_tau, tau, F)"
                )
            if F is None:
                raise ValueError("regress head requires F")
            return self._forward_regress(F)
        if x_tau is None or tau is None or F is None:
            raise ValueError("diffusion head requires (x_tau, tau, F)")
        return self._forward_diffusion(x_tau, tau, F)

    def _forward_diffusion(self, x_tau, tau, F):
        """Pre-F0b denoising forward, unchanged."""
        x_tau = (x_tau - self.pose_mean) / self.pose_std
        h = self._embed(x_tau)
        # Prepended timestep token: every decoder layer's self-attention can see
        # the noise level and gate x_tau accordingly.  The timestep embedding is
        # LayerNorm'd (embeddings.py) to sqrt(dim)=16; *0.05 keeps |t_tok| ~0.8,
        # the same scale as the pose tokens, so the t-token neither dominates
        # self-attention nor gets nulled out (tau-blind head, E1/E2 diagnosis).
        t_tok = self.embeddings.timestep(tau).unsqueeze(1) * 0.05 + self.timestep_token
        h = torch.cat([t_tok, h], dim=1)
        h = self.decoder(h, F)
        h = h[:, 1:]  # drop the timestep token before unembedding
        x0 = self._unembed(h)
        return x0 * self.pose_std + self.pose_mean

    def _forward_regress(self, F):
        """F0b: token = query + time PE + group embedding, straight through the
        6-layer decoder (self-attention, cross-attention to F, FFN), then
        unembed and denormalize with the same fitted stats as the diffusion
        head.

        gate="sigma" (V3-3): F must be the fused memory (B, 2*tw, dim) with
        the first tw tokens V-sourced and the last tw T-sourced (fusion's cat
        order).  Each layer cross-attends twice — once per modality view —
        and routes them per slot via the sigma gate (tau_p = prior bias).
        """
        batch = F.shape[0]
        tokens = self.query.expand(batch, -1, -1, -1)  # (B, tw, n_tokens, dim)
        tokens = tokens + self.embeddings.time_pe.to(dtype=tokens.dtype).view(1, self.tw, 1, self.dim)
        tokens = tokens + self.group_emb(self.group_ids).to(dtype=tokens.dtype).view(
            1, 1, self.n_tokens, self.dim
        )
        h = tokens.reshape(batch, self.tw * self.n_tokens, self.dim)
        if self.gate == "sigma":
            if F.shape[1] != 2 * self.tw:
                raise ValueError(
                    "gate='sigma' expects the fused memory F with shape "
                    "(B, 2*tw, d): first tw tokens V-sourced, last tw T-sourced "
                    "(fusion.forward cat order); got %s" % (F.shape,)
                )
            ignore_t_mask = torch.zeros(batch, 2 * self.tw, dtype=torch.bool, device=F.device)
            ignore_t_mask[:, self.tw:] = True  # e_V: ignore the T-source tokens
            ignore_v_mask = ~ignore_t_mask      # e_T: ignore the V-source tokens
            h = self.decoder(h, F, (ignore_t_mask, ignore_v_mask))
            # Per-slot hypotheses for the beta-NLL sigma supervision; detached
            # (the supervision stops the gradient at the error target).
            self._last_hypo = self._part_hypotheses(h).detach()
        else:
            h = self.decoder(h, F)
        x0 = self._unembed(h)
        return x0 * self.pose_std + self.pose_mean

    def _part_hypotheses(self, h):
        """(N_PARTS, B, tw, N_JOINTS, 6) per-slot pose hypotheses in joint
        order, raw (normalized space, before denormalization).  Used by the
        beta-NLL sigma supervision (losses.py) to compute each slot's error.
        Soft mode = each slot's full-pose head output; hard mode = each part
        head placed at its PART_JOINTS block (zeros elsewhere)."""
        batch = h.shape[0]
        tokens = h.view(batch, self.tw, self.n_tokens, self.dim)
        if self.soft_parts:
            parts = torch.stack(
                [head(tokens[:, :, p]) for p, head in enumerate(self.out_parts)],
                dim=0,
            )  # (N_PARTS, B, tw, POSE_DIM)
            return parts.reshape(N_PARTS, batch, self.tw, N_JOINTS, 6)
        out = h.new_zeros(N_PARTS, batch, self.tw, N_JOINTS, 6)
        for p, (joints, head) in enumerate(zip(PART_JOINTS, self.out_parts)):
            out[p, :, :, joints, :] = head(tokens[:, :, p]).view(batch, self.tw, len(joints), 6)
        return out

    def part_assignment(self) -> torch.Tensor:
        """(N_PARTS, N_JOINTS) assignment matrix: the learned softmax A in
        soft mode, the frozen one-hot PART_JOINTS in hard mode."""
        if self.soft_parts:
            return F.softmax(self.part_logits.detach(), dim=0)
        assign = torch.zeros(N_PARTS, N_JOINTS, device=self.group_emb.weight.device)
        for p, joints in enumerate(PART_JOINTS):
            assign[p, joints] = 1.0
        return assign

    def set_sigma_frozen(self, flag: bool) -> None:
        """Freeze sigma at 0 for the first sigma_freeze_frac of training
        (plan §3: gate runs on the tau_p prior bias alone, then sigma comes
        online).  Only meaningful for gate='sigma'."""
        if self.gate != "sigma":
            return
        for block in self.decoder.layers:
            block.sigma_frozen = bool(flag)

    def last_gate_sigma(self) -> torch.Tensor:
        """(B, tw, N_PARTS, 2) raw sigma logits of the last decoder layer
        (grad-carrying; the beta-NLL loss supervises these)."""
        if self.gate == "none":
            raise ValueError("last_gate_sigma() is only defined for gate='sigma'")
        return self.decoder.layers[-1].last_sigma

    def last_gate_g(self) -> torch.Tensor:
        """(B, tw, N_PARTS, 3) detached gate weights [g_V, g_T, g_∅] of the
        last decoder layer — the attribution weights in the beta-NLL loss."""
        if self.gate == "none":
            raise ValueError("last_gate_g() is only defined for gate='sigma'")
        return self.decoder.layers[-1].last_g

    def last_part_hypotheses(self) -> torch.Tensor:
        """(N_PARTS, B, tw, N_JOINTS, 6) detached per-slot hypotheses from the
        last forward (gate mode only; the beta-NLL error target)."""
        if self.gate == "none":
            raise ValueError("last_part_hypotheses() is only defined for gate='sigma'")
        return self._last_hypo

    def assignment(self) -> torch.Tensor:
        """Detached (N_PARTS, N_JOINTS) soft assignment matrix A (soft_parts
        only) — the learned grouping, printable/auditable every checkpoint."""
        if not self.soft_parts:
            raise ValueError("assignment() is only defined for soft_parts=True")
        return F.softmax(self.part_logits.detach(), dim=0)

    def gate_summary(self) -> torch.Tensor:
        """(N_PARTS, 3) mean gate weights [g_V, g_T, g_∅] over batch and frames
        from the last decoder layer (gate='sigma' only; call after a forward).
        Feeds the plan's F4 acceptance (foot g_T - arm g_T > 0.2 etc.)."""
        if self.gate == "none":
            raise ValueError("gate_summary() is only defined for gate='sigma'")
        last_g = self.decoder.layers[-1].last_g
        if last_g is None:
            raise ValueError("gate_summary() requires a forward pass first")
        return last_g.mean(dim=(0, 1))
