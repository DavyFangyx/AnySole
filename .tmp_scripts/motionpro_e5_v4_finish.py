#!/usr/bin/env python3
"""Finish the v4 E5 chain after inference exits.

The running inference was started before the stale-clean-copy fix, so its
in-memory ``terminal`` list still holds the 8 copy-verify failures and it will
return rc=1.  The copies themselves are already refreshed and verified; this
script waits for inference to exit, re-verifies every (variant, fold), then
runs eval -> MotionPRO tables -> Main_Table2 -> degradation summary.
"""
from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
LOG = ROOT / "results/diagnostics/baselines_fourfold_20261109/motionpro_e5_v4_finish.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def inference_alive() -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "motionpro_e5_inference.py" in cmd and "finish" not in cmd:
            return True
    return False


def run(args: list[str]) -> int:
    log(f"run: {' '.join(args)}")
    return subprocess.call([PY, *args], cwd=str(ROOT))


def load_inference():
    spec = importlib.util.spec_from_file_location(
        "mpe5", ROOT / ".tmp_scripts/motionpro_e5_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    while inference_alive():
        time.sleep(30)
    log("inference exited; verifying variants")
    m = load_inference()
    bad = []
    for variant, _tree, kind in m.VARIANTS:
        for fold in m.FOLDS:
            ok, reason = m.verify(variant, fold)
            if not ok:
                bad.append(f"{variant}__{fold}: {reason}")
                log(f"[verify FAIL] {variant} {fold}: {reason}")
    if bad:
        log(f"verification incomplete ({len(bad)}); abort")
        return 1
    log("all variants verified; eval")
    rc = run([".tmp_scripts/motionpro_e5_eval.py"])
    if rc != 0:
        log(f"eval rc={rc}; abort")
        return rc
    log("eval done; MotionPRO tables")
    rc = run(["results_display/script/e5_motionpro_export.py"])
    if rc != 0:
        log(f"motionpro export rc={rc}; abort")
        return rc
    log("main tables")
    rc = run(["results_display/script/v8_e5_main_tables.py"])
    if rc != 0:
        log(f"main tables rc={rc}; abort")
        return rc
    log("degradation summary")
    rc = run(["results_display/script/e5_visual_degradation_summary.py"])
    log(f"summary rc={rc}; done")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
