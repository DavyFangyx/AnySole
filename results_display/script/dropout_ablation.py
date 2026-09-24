"""A2: missing-condition/dropout analysis."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs.tools.common import model_run_name
from utils.component_analysis import main as component_main
from utils.display_common import common_parser, load_context, model_ids_for, model_output


def main(argv=None) -> int:
    args = common_parser("missing-condition branch division", "singlemodal_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "ATest/A2Test_dropout_ablation")
    for model_id in model_ids_for(spec, args.model):
        root = model_output(registry, args.experiment, model_id)
        out = out_root / model_run_name(registry, model_id)
        return component_main([
            "--a2", "--split", args.split,
            "--fs-main", str(root / "metrics" / (args.split + "_fseries.json")),
            "--out", str(out),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
