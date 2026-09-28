#!/usr/bin/env python3
"""Removed producer entry point.

D_Test4 is read-only.  Public tactile facts and the single MMVP representation
are produced by ``build_shared.py``; model adapters must write their own
``model_inputs/<model>`` tree and may not recreate this historical generator.
"""

raise SystemExit(
    "generate_baseline_tactile.py was removed; run build_shared.py to build "
    "shared/facts and shared/representations, then let D_Test4 audit only."
)
