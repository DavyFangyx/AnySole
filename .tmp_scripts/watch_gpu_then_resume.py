#!/usr/bin/env python3
"""Wait until fangyuxuan has no GPU processes, then resume the four-fold pipeline.

The lab gpu-watch SIGTERMs *all* fangyuxuan GPU processes when the user uses
five distinct GPUs.  The traj_bridge experiment currently occupies four GPUs
(2,3,5,6); starting any of our jobs would make five and kill both experiments.
This watcher waits for the bridge jobs (and any other fangyuxuan GPU job) to
release the GPUs, then starts the baseline pipeline once.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
LOG = ROOT / "results/diagnostics/baselines_fourfold_20261109/resume_watcher.log"
PIPELINE = ROOT / ".tmp_scripts/baseline_fourfold_pipeline.py"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def fangyuxuan_gpus() -> set[int]:
    import pwd
    idx = {}
    for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True).splitlines():
        i, u = [x.strip() for x in line.split(",")]
        idx[u] = int(i)
    gpus = set()
    for line in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader"], text=True).splitlines():
        if not line.strip():
            continue
        u, pid = [x.strip() for x in line.split(",")]
        try:
            user = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except Exception:
            continue
        if user == "fangyuxuan":
            gpus.add(idx.get(u, -1))
    return gpus


def pipeline_running() -> bool:
    out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    return any("baseline_fourfold_pipeline.py" in line and "grep" not in line
               for line in out.splitlines())


def main() -> int:
    log("watcher start: waiting for fangyuxuan GPU set to empty")
    while True:
        gpus = fangyuxuan_gpus()
        if gpus:
            log(f"fangyuxuan uses GPUs {sorted(gpus)}; waiting")
        elif pipeline_running():
            log("GPUs free but pipeline already running; exiting")
            return 0
        else:
            log("fangyuxuan has no GPU processes; starting pipeline")
            handle = (ROOT / "results/diagnostics/baselines_fourfold_20261109"
                      / "nohup.log").open("a", encoding="utf-8")
            subprocess.Popen([PY, str(PIPELINE)], cwd=str(ROOT),
                             stdout=handle, stderr=subprocess.STDOUT,
                             start_new_session=True)
            log("pipeline launched")
            return 0
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
