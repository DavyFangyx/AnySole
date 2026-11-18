#!/usr/bin/env python3
"""Rerun fS6S7 E5 scenario s10 cells with the corrected V8_Cv5 checkpoint."""
from __future__ import annotations

import json
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
CKPT = (REPO / "results/AnySole/V8_Cv5_fS6S7_f6_soft_time_accel_state_smpl24_v1/"
        "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
        "checkpoints/ckpt_last.pt")
FOLD = "fS6S7"
SUFFIXES = (
    ["_evalocc%03d" % i for i in range(10, 100, 10)]
    + ["_evalocc100"]
    + ["_evalgap010", "_evalgap020"]
)
GPUS = ("cuda:2", "cuda:3", "cuda:6", "cuda:7")


def run_one(suffix, gpus: queue.Queue) -> None:
    cell = "V8_Cv5_%s_s1%s" % (FOLD, suffix)
    cell_dir = REPO / "results" / "cells" / cell
    outdir = cell_dir / "metrics" / "overlap_s10"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((cell_dir / "effective_config.json").read_text())["config"]
    gpu = gpus.get()
    try:
        print("[start] %s %s %s" % (cell, gpu, time.strftime("%F %T")), flush=True)
        with (outdir / "run.log").open("w") as log:
            proc = subprocess.run(
                [PY, "-m", "anysole.eval", "--config", cfg, "--ckpt", str(CKPT),
                 "--split", "test", "--overlap-stride", "10",
                 "--config-id", "V2M,VT2M", "--no-robustness",
                 "--write-motion",
                 str(cell_dir / "predictions" / "overlap_s10" / "eval_motion"),
                 "--metrics-out", str(outdir / "test_brief.json"),
                 "--protocol-out", str(outdir / "test.json"),
                 "--device", gpu],
                cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not (outdir / "test.json").is_file():
            print("[fail] %s rc=%d" % (cell, proc.returncode), flush=True)
            (outdir / "test.json").unlink(missing_ok=True)
        else:
            print("[done] %s %s" % (cell, time.strftime("%F %T")), flush=True)
    finally:
        gpus.put(gpu)


def main() -> int:
    gpus: queue.Queue = queue.Queue()
    for g in GPUS:
        gpus.put(g)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        list(pool.map(lambda s: run_one(s, gpus), SUFFIXES))
    print("[fS6S7 scenarios done] %s" % time.strftime("%F %T"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
