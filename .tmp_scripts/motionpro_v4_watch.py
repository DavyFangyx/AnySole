#!/usr/bin/env python3
"""Wait for the v4 AnySole-bbox features, then retrain + re-evaluate E2."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
LOG = ROOT / "results/diagnostics/baselines_fourfold_20261109/motionpro_v4_watch.log"
TREE = ROOT / "AnysoleWorkspace/model_inputs/MotionPRO/adapter_v4_anysolebbox/cam3"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    while len(list(TREE.glob("*/*/*/feature_hrnet.pth"))) < 140:
        log(f"features: {len(list(TREE.glob('*/*/*/feature_hrnet.pth')))}/140")
        time.sleep(300)
    log("features ready; training")
    rc = subprocess.call([PY, ".tmp_scripts/motionpro_v4_train.py"], cwd=str(ROOT))
    log(f"training rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
