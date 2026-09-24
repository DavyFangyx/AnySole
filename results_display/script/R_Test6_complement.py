#!/usr/bin/env python3
"""Canonical R_Test6 / A1 display entry point."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs.tools.common import model_run_name, specialist_id
from utils.component_analysis import main as component_main
from utils.display_common import common_parser, load_context, model_ids_for, model_output


def main(argv=None) -> int:
    args = common_parser("fusion completeness", "singlemodal_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "complement")
    for model_id in model_ids_for(spec, args.model):
        main_root = model_output(registry, args.experiment, model_id)
        v_root = model_output(registry, args.experiment, specialist_id(model_id, "vonly"))
        t_root = model_output(registry, args.experiment, specialist_id(model_id, "tonly"))
        out = out_root / model_run_name(registry, model_id)
        return component_main([
            "--a1", "--split", args.split,
            "--fs-main", str(main_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-vspecialist", str(v_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-tspecialist", str(t_root / "metrics" / (args.split + "_fseries.json")),
            "--out", str(out),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
