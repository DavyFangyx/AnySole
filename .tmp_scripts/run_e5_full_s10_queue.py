#!/usr/bin/env python3
"""E5 publication s10 queue.

- Cv4/Cv2/Cv6 x 15 scenarios x 4 folds = 180 main-table evals (V2M/VT2M).
- Cv5/Cv7 x 5 occlusion levels (020/040/060/080/100) x 4 folds = 40
  SMPL-heatmap evals with motion export (V2M/VT2M).
Resumable: main jobs skip if test.json exists; heatmap jobs skip if the
overlap_s10 prediction dir already has npz files.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MAIN_SUFFIXES = (
    ["_evalocc%03d" % i for i in range(10, 100, 10)]
    + ["_evalgap%03d" % i for i in (10, 20, 40, 60)]
    + ["_evalsceneclean", "_evalscene"]
)
HEAT_LEVELS = ("020", "040", "060", "080", "100")
GPUS = tuple(["cuda:2"] * 3 + ["cuda:3"] * 3 + ["cuda:6"] * 3 + ["cuda:7"] * 3)


def run_one(job, gpus: queue.Queue) -> None:
    model, fold, suffix, motion = job
    cell = "%s_%s_s1%s" % (model, fold, suffix)
    cell_dir = REPO / "results" / "cells" / cell
    outdir = cell_dir / "metrics" / "overlap_s10"
    outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = cell_dir / "predictions" / "overlap_s10" / "eval_motion"
    marker = pred_dir / ".complete"
    if not motion and (outdir / "test.json").is_file():
        print("[skip] %s" % cell, flush=True)
        return
    if motion and marker.is_file():
        print("[skip-motion] %s" % cell, flush=True)
        return
    cfg = json.loads((cell_dir / "effective_config.json").read_text())["config"]
    ckpt = (cell_dir / "checkpoint_used.txt").read_text().strip()
    gpu = gpus.get()
    log_name = "run_heatmap.log" if motion else "run.log"
    try:
        print("[start] %s %s %s" % (cell, gpu, time.strftime("%F %T")), flush=True)
        cmd = [PY, "-m", "anysole.eval", "--config", cfg, "--ckpt", ckpt,
               "--split", "test", "--overlap-stride", "10",
               "--config-id", "V2M,VT2M", "--no-robustness",
               "--metrics-out", str(outdir / "test_brief.json"),
               "--protocol-out", str(outdir / "test.json"),
               "--device", gpu]
        if motion:
            cmd += ["--write-motion", str(pred_dir)]
        else:
            cmd += ["--no-write-motion"]
        with (outdir / log_name).open("w") as log:
            proc = subprocess.run(cmd, cwd=str(REPO), stdout=log,
                                  stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not (outdir / "test.json").is_file():
            print("[fail] %s rc=%d" % (cell, proc.returncode), flush=True)
            (outdir / "test.json").unlink(missing_ok=True)
        else:
            if motion:
                pred_dir.mkdir(parents=True, exist_ok=True)
                marker.write_text("ok")
            print("[done] %s %s" % (cell, time.strftime("%F %T")), flush=True)
    finally:
        gpus.put(gpu)


def main() -> int:
    jobs = []
    for model in ("V8_Cv4", "V8_Cv2", "V8_Cv6"):
        for fold in FOLDS:
            for suffix in MAIN_SUFFIXES:
                jobs.append((model, fold, suffix, False))
    for model in ("V8_Cv5", "V8_Cv7"):
        for fold in FOLDS:
            for level in HEAT_LEVELS:
                jobs.append((model, fold, "_evalocc" + level, True))
    print("[queue] %d jobs" % len(jobs), flush=True)
    gpus: queue.Queue = queue.Queue()
    for g in GPUS:
        gpus.put(g)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        list(pool.map(lambda job: run_one(job, gpus), jobs))
    print("[queue done] %s" % time.strftime("%F %T"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
