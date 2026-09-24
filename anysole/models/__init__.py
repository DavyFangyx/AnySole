"""AnySole model catalog.

Top level of this package = one entry script per registered model
(``f0b.py`` … ``v4b.py``): each entry hard-codes that model's structure
(encoders / representation / part mechanism) and exposes ``STRUCTURE`` +
``build(config, …)``.  Every training run of any (model, hyperparams)
combination starts from random init — there is no warm-start lineage.

Shared building blocks (encoders, fusion, pose/traj/aux heads, V1/V2
assembly classes) live in ``anysole.models.common``.

The legacy V1 names are re-exported here so older probes keep importing
from ``anysole.models`` unchanged.
"""

from anysole.models.common.model import (
    AnySoleModel,
    MODEL_NAMES,
    MODEL_ANYSOLEV1,
    MODEL_ANYSOLEV1_INSOLE_DRIFT,
    MODEL_ANYSOLEV1_POS,
)
from anysole.models.common.model_v2 import AnySoleModelV2, MODEL_ANYSOLEV2

__all__ = [
    "AnySoleModel",
    "AnySoleModelV2",
    "MODEL_NAMES",
    "MODEL_ANYSOLEV1",
    "MODEL_ANYSOLEV1_INSOLE_DRIFT",
    "MODEL_ANYSOLEV1_POS",
    "MODEL_ANYSOLEV2",
]
