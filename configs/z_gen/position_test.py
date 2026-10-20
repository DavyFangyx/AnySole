#!/usr/bin/env python3
"""Generate the train+eval task for the V3_3D frozen-V + position-head model.

One conf (TASK=model_run): the runner trains V3_3D for 200 epochs from the
V3_3C-derived base (--init-from in the registry), then runs the formal
val/test evaluation.
"""

from __future__ import annotations

import argparse

from common import emit_experiment
from model_registry import MODELS

EXPERIMENT_ID = "position_test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs (default: the experiment's MODEL_IDS)")
    args = parser.parse_args()
    selected = [item.strip() for item in args.models.split(",") if item.strip()] or None
    if selected:
        unknown = [model_id for model_id in selected if model_id not in MODELS]
        if unknown:
            raise SystemExit("unknown model IDs: %s (registered: %s)"
                             % (",".join(unknown), ",".join(MODELS)))
    count = emit_experiment(EXPERIMENT_ID, selected)
    print(f"generated {count} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
