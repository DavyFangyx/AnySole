#!/usr/bin/env python3
"""Wait for the eight four-fold baseline comparison JSONs, then refresh E2."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = ("MotionPRO", "VP-MoCap")
LOG = ROOT / "results/diagnostics/baselines_fourfold_20261109/e2_export_watcher.log"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def main() -> int:
    log("watcher start")
    while True:
        missing = []
        for model in MODELS:
            for fold in FOLDS:
                path = ROOT / f"results/baselines/{model}/metrics/{fold}_comparison.json"
                if not path.is_file():
                    missing.append(f"{model}/{fold}")
        if not missing:
            log("all eight comparison JSONs present; running E2 export")
            cmd = [PY, "results_display/script/e2_crossfit_export.py",
                   "--with-baselines", "--force"]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            log(f"e2 export rc={proc.returncode}")
            log(proc.stdout[-4000:])
            log(proc.stderr[-4000:])
            if proc.returncode == 0:
                log("done")
                return 0
        else:
            log(f"waiting: missing {len(missing)} ({', '.join(missing[:6])}...)")
        time.sleep(600)


if __name__ == "__main__":
    raise SystemExit(main())
