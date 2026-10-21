"""V4_1_v2 entry: V4_1A structure on the v2 无平滑交接包
(tactile_input=denoised108_v2; data variant of V4_1, same network).

Data variant of v4_1 (2026-10-21): identical architecture, identical
carrier contract (frozen E_offfix V stream) — the only difference is the
T stream reading the v2 handoff pressure package instead of raw108.  The
build / carrier-loading logic is inherited from v4_1 unchanged.
"""

from anysole.models.v4_1 import (  # noqa: F401
    BETA_SWEEP,
    CONFIG_EXTRA,
    build,
    STRUCTURE as _BASE_STRUCTURE,
)

STRUCTURE = {**_BASE_STRUCTURE, "tactile_input": "denoised108_v2"}
