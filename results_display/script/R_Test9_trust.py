#!/usr/bin/env python3
"""Canonical R_Test9 / B2 display entry point."""
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
    args = common_parser("T-branch global information", "singlemodal_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "trust")
    for model_id in model_ids_for(spec, args.model):
        main_root = model_output(registry, "singlemodal_eval", model_id)
        grid_root = model_output(registry, "rho_grid_eval", model_id)
        t_root = model_output(registry, "singlemodal_eval", specialist_id(model_id, "tonly"))
        out = out_root / model_run_name(registry, model_id)
        return component_main([
            "--b2", "--split", args.split,
            "--fs-main", str(main_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-tspecialist", str(t_root / "metrics" / (args.split + "_fseries.json")),
            "--prior-grid", str(grid_root / ("grid_metrics_%s.json" % args.split if args.split != "val" else "grid_metrics.json")),
            "--out", str(out),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
