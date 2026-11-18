#!/usr/bin/env python3
"""Populate the s10 counterpart of Etest_crossfit_physimu_vgate.

The s10 tree uses exactly the same experiment definitions as the main tree;
only inference changes to tw=20 + --overlap-stride 10 (triangular blend).
Raw eval roots:
  results/diagnostics/20261114_imu_eval_v2_s10       (E4)
  results/diagnostics/20261114_imu_eval_v2_all_s10   (E2/E5)
Output root:
  results_display/Etest_crossfit_physimu_vgate_s10

This wrapper reuses the main tree's export scripts by patching their module
globals, so the original scripts stay unchanged. Run after the s10 eval queue.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "results_display" / "script"
EVAL = REPO / "results/diagnostics/20261114_imu_eval_v2_s10"
EVAL_ALL = REPO / "results/diagnostics/20261114_imu_eval_v2_all_s10"
OUT = REPO / "results_display/Etest_crossfit_physimu_vgate_s10"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(name: str, patches: dict) -> None:
    path = SCRIPT / (name + ".py")
    if not path.is_file():
        print("[skip] missing", path)
        return
    mod = load(name)
    for key, val in patches.items():
        setattr(mod, key, val)
    if not hasattr(mod, "main"):
        print("[skip] no main", name)
        return
    print("=== %s" % name, flush=True)
    mod.main()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # E2 main table (uses the V8_IMU_Residual clean s10 eval).
    run("e2_v2_table", {
        "SRC": EVAL_ALL / "V8_IMU_Residual" / "clean",
        "OUT": OUT / "E2_MainPerformance",
    })
    # E4 tables/figures.
    run("e4_v2_tables", {
        "EVAL": EVAL,
        "OUT": OUT / "E4_ComponentAblation",
    })
    run("e4_v2_figures", {
        "EV": EVAL,
        "EV_ALL": EVAL_ALL,
        "OUT": OUT / "E4_ComponentAblation",
    })
    # E5 tables/figures/summary.
    run("e5_v2_export", {
        "EVAL": EVAL_ALL / "V8_IMU_Residual",
        "OUT": OUT,
    })
    run("e5_v2_figures", {"OUT": OUT})
    mod = load("e5_v2_summary")
    mod.OUT = OUT / "E5_VisualDegradation_Summary"
    # The summary derives OCC/SCN/GAP from OUT at import time; recompute them.
    mod.OCC = OUT / "E5_OcclusionRobustness" / "Main_Table2_E5_Occlusion_MainVsBaselines.csv"
    mod.SCN = OUT / "E5_SceneOcclusion" / "Main_Table2_E5_SceneOcclusion_MainVsBaselines.csv"
    mod.GAP = OUT / "E5_TimeGap" / "Main_Table2_E5_TimeGap_MainVsBaselines.csv"
    print("=== e5_v2_summary", flush=True)
    mod.main()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
