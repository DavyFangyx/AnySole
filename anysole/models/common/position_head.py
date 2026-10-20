"""Position head (V3_3D): the joint position-space readout of [v_tok; t_tok].

Design (T信息量判定实验.md §3.2 残差范式 ridge 界 → 联合读出裁定):

- Reads the PRE-FUSION encoder tokens [v_tok; t_tok] concatenated per frame
  (B, tw, 512).  The fusion transformer destroys T's position-level
  increments (encoder-level vt −10.3mm → fusion-level F_VT +10.0mm on the
  six-joint relative-position target), so the head bypasses it entirely.
- A small 2-layer temporal transformer lets V and T tokens interact in the
  SAME readout — the co-adaptation term that carries ~90% of T's input-level
  increment (a frozen-V additive residual is capped at ~−1mm; only the joint
  readout reaches the −10.6/−19.9mm ridge bounds).
- Two zero-initialized output heads:
    dpos (B, tw, 18): pelvis-relative position correction of the six
        lower-limb joints (l/r knee, ankle, foot; KNEE_ANKLE_FOOT_JOINTS
        order), meters, world axes — added to the frozen base's FK keypoints
        (soft skeleton; metrics on the corrected positions, the base pose
        stays the rendering/mesh channel).
    dv   (B, tw, 3):  root velocity correction (m/s) added to the frozen
        traj head's v_hat before the cumsum — T's realized-value channel
        (实验 2: global MPJPE / root ATE / ground penetration).
- Zero final layers ⇒ at init the corrected outputs are bit-identical to the
  frozen base (the V3_3C structural guarantee, now in position space).
- Active under every condition (VT / V-only / T-only): the encoders' null
  replacement handles the missing stream, so the head is trained with the
  same config sampling as the main model (0.5 / 0.25 / 0.25).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.types import KNEE_ANKLE_FOOT_JOINTS

N_SIX = len(KNEE_ANKLE_FOOT_JOINTS)  # 6
POS_CORR_DIM = N_SIX * 3              # 18
VEL_CORR_DIM = 3


class PositionHead(nn.Module):
    def __init__(self, dim: int = 256, tw: int = 20, nhead: int = 8,
                 n_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.tw = int(tw)
        layer = nn.TransformerEncoderLayer(
            d_model=2 * dim,           # [v_tok; t_tok] per frame
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        kwargs = {"num_layers": n_layers}
        try:
            self.encoder = nn.TransformerEncoder(layer, enable_nested_tensor=False, **kwargs)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, **kwargs)
        self.pos_proj = nn.Linear(2 * dim, POS_CORR_DIM)
        self.vel_proj = nn.Linear(2 * dim, VEL_CORR_DIM)
        # Structural guarantee: zero correction at init (the corrected
        # outputs are bit-identical to the frozen base).
        for proj in (self.pos_proj, self.vel_proj):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, v_tok: torch.Tensor, t_tok: torch.Tensor):
        """(B, tw, 256) x2 -> (dpos (B, tw, 18) meters, dv (B, tw, 3) m/s)."""
        x = torch.cat([v_tok, t_tok], dim=-1)
        h = self.encoder(x)
        return self.pos_proj(h), self.vel_proj(h)
