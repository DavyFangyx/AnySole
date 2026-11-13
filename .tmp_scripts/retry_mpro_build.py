#!/usr/bin/env python3
"""Retry a MotionPRO E5 feature build on the allowed GPU with most free memory."""
from __future__ import annotations

import os
import subprocess
import sys
import time

ALLOWED = (2, 3, 6, 7)
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"


def free_mem() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True)
    free = {}
    for line in out.splitlines():
        i, m = [x.strip() for x in line.split(",")[:2]]
        free[int(i)] = int(m)
    return free


def main() -> int:
    args = sys.argv[1:]
    attempts = 0
    while attempts < 30:
        free = free_mem()
        choices = [g for g in ALLOWED if free.get(g, 0) >= 4000]
        if not choices:
            time.sleep(120)
            continue
        gpu = max(choices, key=lambda g: free[g])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        code = subprocess.call([PY, "-m",
                                "AnysoleWorkspace.tool.adapters.MotionPRO.build_e5_variants",
                                *args], env=env)
        if code == 0:
            print(f"OK on gpu {gpu}")
            return 0
        attempts += 1
        print(f"attempt {attempts} failed rc={code}; retrying", flush=True)
        time.sleep(60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
