#!/usr/bin/env python3
"""E5 occ100 for Cv4/Cv2/Cv6 (full_tables 6-level x 5-model)."""
from __future__ import annotations

import json
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
GPUS = tuple(["cuda:2"] * 3 + ["cuda:3"] * 3 + ["cuda:6"] * 3 + ["cuda:7"] * 3)


def run_one(job, gpus: queue.Queue) -> None:
    model, fold = job
    cell = "%s_%s_s1_evalocc100" % (model, fold)
    cell_dir = REPO / "results" / "cells" / cell
    outdir = cell_dir / "metrics" / "overlap_s10"
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "test.json").is_file():
        print("[skip] %s" % cell, flush=True)
        return
    cfg = json.loads((cell_dir / "effective_config.json").read_text())["config"]
    ckpt = (cell_dir / "checkpoint_used.txt").read_text().strip()
    gpu = gpus.get()
    try:
        print("[start] %s %s %s" % (cell, gpu, time.strftime("%F %T")), flush=True)
        with (outdir / "run.log").open("w") as log:
            proc = subprocess.run(
                [PY, "-m", "anysole.eval", "--config", cfg, "--ckpt", ckpt,
                 "--split", "test", "--overlap-stride", "10",
                 "--config-id", "V2M,VT2M", "--no-write-motion",
                 "--no-robustness",
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
    jobs = [(model, fold) for model in ("V8_Cv4", "V8_Cv2", "V8_Cv6")
            for fold in FOLDS]
    gpus: queue.Queue = queue.Queue()
    for g in GPUS:
        gpus.put(g)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        list(pool.map(lambda job: run_one(job, gpus), jobs))
    print("[occ100 done] %s" % time.strftime("%F %T"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
