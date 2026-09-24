#!/usr/bin/env python3
"""Generate singlemodal evaluation tasks into configs/queue/."""

from __future__ import annotations

import argparse

from common import emit_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="comma-separated main model IDs")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    count = emit_experiment("singlemodal_eval", models or None)
    print(f"generated {count} queue task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
