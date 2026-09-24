"""E6.6b: Step2Motion-style per-group tactile encoder.

T_s2m layout (25/foot): pressure16 (heel[8] toes[8]) + acc3 + gyro3 + force1 + cop2.
Eight groups (left/right x heel pressure / toes pressure / IMU / others) each get
their own MLP, mirroring Step2Motion's ControlTransformer condition embeddings
(model.py:93-148); the per-frame streams are summed into one token stream
(B, TW, d) so the fusion transformer keeps its input shape unchanged. The final
LayerNorm plays the role of Step2Motion's per-dataset z-score normalizer, which
anysole does not have.

--no-imu (E6.8): input is the 38-dim T_s2m (pressure16 + force1 + cop2 per foot);
the two IMU groups are structurally deleted (no ``*imu*`` parameters exist), so
six groups remain: left/right x heel pressure / toes pressure / others.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anysole.models.common.embeddings import SharedEmbeddings
from anysole.types import D_MODEL, T_S2M_DIM, T_S2M_NOIMU_DIM, TW


class TactileEncoder(nn.Module):
    # T_s2m group slices, Step2Motion ControlTransformer order:
    # l_heel, l_toes, l_imu, l_others, r_heel, r_toes, r_imu, r_others.
    GROUPS = (
        (0, 8), (8, 16), (16, 22), (22, 25),
        (25, 33), (33, 41), (41, 47), (47, 50),
    )
    # --no-imu: 38-dim layout (pressure16 + force1 + cop2 per foot); the IMU
    # groups are deleted, not masked — no imu encoding receives any input.
    GROUPS_NOIMU = (
        (0, 8), (8, 16), (16, 19),
        (19, 27), (27, 35), (35, 38),
    )

    def __init__(self, embeddings=None, dim=D_MODEL, tw=TW, no_imu=False):
        super().__init__()
        if embeddings is None:
            embeddings = SharedEmbeddings(dim=dim, tw=tw)
        self.embeddings = embeddings
        self.dim = dim
        self.tw = tw
        self.no_imu = bool(no_imu)
        groups = self.GROUPS_NOIMU if self.no_imu else self.GROUPS
        self.group_mlps = nn.ModuleList(
            [self._mlp(hi - lo, dim) for lo, hi in groups]
        )
        # Sum of the streams is unnormalized scale-wise; LayerNorm matches the
        # LinearTemporalEncoder norm that the raw108 path applies to its input.
        self.merge = nn.LayerNorm(dim)

    @staticmethod
    def _mlp(in_dim: int, dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = T_S2M_NOIMU_DIM if self.no_imu else T_S2M_DIM
        if x.shape[-1] != expected:
            raise ValueError(
                "TactileEncoder expects %s (%d-dim), got %d"
                % ("no-imu T_s2m" if self.no_imu else "T_s2m", expected, x.shape[-1])
            )
        groups = self.GROUPS_NOIMU if self.no_imu else self.GROUPS
        h = sum(
            mlp(x[..., lo:hi]) for mlp, (lo, hi) in zip(self.group_mlps, groups)
        )
        h = self.merge(h)
        h = h + self.embeddings.time_pe.to(dtype=h.dtype)
        modality = self.embeddings.modality(
            torch.tensor(1, device=x.device, dtype=torch.long)
        )
        return h + modality.to(dtype=h.dtype)
