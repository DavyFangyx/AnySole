#!/usr/bin/env python3
"""Wait for the v4 E5 feature trees, then inference -> eval -> MotionPRO table."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
LOG = ROOT / "results/diagnostics/baselines_fourfold_20261109/motionpro_e5_v4_watch.log"
INPUTS = ROOT / "AnysoleWorkspace/model_inputs/MotionPRO"
OCC = ("010", "020", "030", "040", "050", "060", "070", "080", "090")
GAP = ("010", "020", "040", "060")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def n_feat(tree: str) -> int:
    return len(list((INPUTS / tree / "cam3").glob("*/*/*/feature_hrnet.pth")))


def ready() -> bool:
    for lv in OCC:
        if n_feat(f"adapter_v4_anysolebbox_occ{lv}") < 97:
            return False
    for lv in GAP:
        if n_feat(f"adapter_v4_anysolebbox_gap_r{lv}") < 97:
            return False
    if n_feat("adapter_v4_anysolebbox_scene") < 25:
        return False
    return True


def run(args: list[str]) -> int:
    log(f"run: {' '.join(args)}")
    return subprocess.call([PY, *args], cwd=str(ROOT))


def main() -> int:
    while not ready():
        log(f"waiting: occ={[n_feat('adapter_v4_anysolebbox_occ'+lv) for lv in OCC]} "
            f"gap={[n_feat('adapter_v4_anysolebbox_gap_r'+lv) for lv in GAP]} "
            f"scene={n_feat('adapter_v4_anysolebbox_scene')}")
        time.sleep(300)
    log("features ready; inference")
    rc = run([".tmp_scripts/motionpro_e5_inference.py"])
    if rc != 0:
        log(f"inference rc={rc}; abort")
        return rc
    log("inference done; eval")
    rc = run([".tmp_scripts/motionpro_e5_eval.py"])
    if rc != 0:
        log(f"eval rc={rc}; abort")
        return rc
    log("eval done; motionpro tables")
    rc = run(["results_display/script/e5_motionpro_export.py"])
    log(f"tables rc={rc}; done")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
