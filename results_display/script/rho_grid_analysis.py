"""B3: rho-grid analysis."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs.tools.common import model_run_name
from utils.display_common import common_parser, load_context, model_ids_for, model_output
from utils.rho_grid_display import main as rho_grid_main


def main(argv=None) -> int:
    args = common_parser("rho-grid missingness", "rho_grid_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "BTest/B3Test_rho_grid")
    for model_id in model_ids_for(spec, args.model):
        root = model_output(registry, args.experiment, model_id)
        metrics = root / ("grid_metrics_%s.json" % args.split if args.split != "val" else "grid_metrics.json")
        out = out_root / model_run_name(registry, model_id)
        return rho_grid_main(["--metrics", str(metrics), "--out-dir", str(out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
