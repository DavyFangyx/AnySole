"""B1: V-to-T upper-branch analysis."""
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
    args = common_parser("V-branch tactile information", "singlemodal_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "BTest/B1Test_v2t_upper")
    for model_id in model_ids_for(spec, args.model):
        main_root = model_output(registry, "singlemodal_eval", model_id)
        grid_root = model_output(registry, "rho_grid_eval", model_id)
        v_root = model_output(registry, "singlemodal_eval", specialist_id(model_id, "vonly"))
        out = out_root / model_run_name(registry, model_id)
        return component_main([
            "--b1", "--split", args.split,
            "--fs-main", str(main_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-vspecialist", str(v_root / "metrics" / (args.split + "_fseries.json")),
            "--prior-grid", str(grid_root / ("grid_metrics_%s.json" % args.split if args.split != "val" else "grid_metrics.json")),
            "--out", str(out),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
