"""A0: singlemodal specialist division analysis."""
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
    args = common_parser("singlemodal specialist division", "singlemodal_eval").parse_args(argv)
    registry, spec, out_root = load_context(args.experiment, "ATest/A0Test_specialist")
    for main_id in model_ids_for(spec, args.model):
        v_id = specialist_id(main_id, "vonly")
        t_id = specialist_id(main_id, "tonly")
        main_root = model_output(registry, args.experiment, main_id)
        v_root = model_output(registry, args.experiment, v_id)
        t_root = model_output(registry, args.experiment, t_id)
        out = out_root / model_run_name(registry, main_id)
        return component_main([
            "--a0", "--split", args.split,
            "--fs-main", str(main_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-vspecialist", str(v_root / "metrics" / (args.split + "_fseries.json")),
            "--fs-tspecialist", str(t_root / "metrics" / (args.split + "_fseries.json")),
            "--out", str(out),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
