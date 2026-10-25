#!/usr/bin/env python3
"""Generate the wo05 survey round queue tasks (2026-10-24, user ruling).

Four training runs — V4_Av1 / V5_Av4 x (always-dual 基名, dropout _drop);
the vonly/tonly experts are skipped this round.  Every run trains with
survey_save=True (anysole/train.py): dual-condition checkpoint series
(val dPA(VT-V) <= 0.5 and val T2M PA <= 90), logged to survey_log.jsonl.

The survey series evaluation phase is emitted separately by
survey_eval_emit.py after the training confs reach done/.
"""

from __future__ import annotations

import argparse

from common import emit_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="",
                        help="comma-separated model IDs to subset the group")
    args = parser.parse_args()
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    n = emit_experiment("wo05_survey", selected or None)
    print(f"generated {n} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
