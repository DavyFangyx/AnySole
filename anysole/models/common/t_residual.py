"""T-residual branch (V3_Cv1): a frozen-V corrector reading only the tactile
token stream.

The branch consumes the frozen ``t_tok`` (B, tw, 256) and outputs a per-frame
correction in native 6D pose space (B, tw, 144).  The final layer is
zero-initialized so the correction is exactly zero at init — combined with
model_v2's residual regime (the frozen base runs in its V-only mode under
every condition; the real tactile stream feeds only this branch), the V3_Cv1
output starts bit-identical to the frozen V3_Bv1_vonly base's V-only output
(VT == V at step 0); the trained model can then only improve on the base
where the tactile stream carries signal.

Capacity is deliberately small (~100k params): the ridge information-ceiling
probe measured at most 1~2.6mm of tactile-only marginal pose information at
the feet/ankle, so a low-capacity branch matches the expected effect size and
cannot grow a large correction out of overfitting.  Temporal context is
already baked into each t_tok frame by the frozen FootConvEncoder's internal
4-layer temporal transformer, so a per-frame shared MLP suffices.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.types import D_MODEL, POSE_DIM


class TResidualBranch(nn.Module):
    def __init__(self, dim: int = D_MODEL, pose_dim: int = POSE_DIM, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pose_dim),
        )
        # Structural guarantee: zero correction at init (VT == V bitwise).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t_tok: torch.Tensor) -> torch.Tensor:
        """(B, tw, dim) -> (B, tw, pose_dim) per-frame correction."""
        return self.mlp(self.norm(t_tok))
