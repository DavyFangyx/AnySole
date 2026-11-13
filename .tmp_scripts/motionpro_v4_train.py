#!/usr/bin/env python3
"""Retrain MotionPRO four folds on AnySole-bbox (v4) features.

Each fold trains with the LR-floor early stop (config: lr < 5e-8), and the
training entry auto-runs test_frappe on the fold test split at the end,
writing predictions to
results/baselines/MotionPRO/predictions/carpet_v4_anysolebbox_<fold>/eval_motion.
"""
from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
LOGROOT = ROOT / "results/diagnostics/baselines_fourfold_20261109/motionpro_v4"
GPUS = (2, 3, 6, 7)
MAX_RUNS = 6

FOLDS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12",
    "fS5S11S12": "splits_test_S5S11S12_val_S13",
    "fS6S7": "splits_test_S6S7_val_S14",
    "fS5S6S8": "splits_test_S5S6S8_val_S7",
}


def fold_test_sessions(fold: str) -> list[str]:
    path = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{FOLDS[fold]}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return [(r.get("test") or "").strip() for r in rows
            if (r.get("test") or "").strip() and r.get("test").strip() != "S12102"]


def complete(fold: str) -> bool:
    root = ROOT / f"results/baselines/MotionPRO/predictions/carpet_v4_anysolebbox_{fold}/eval_motion"
    return all((root / f"{s}.npz").is_file() for s in fold_test_sessions(fold))


def make_cmd(fold: str) -> list[str]:
    return [PY, "-m", "app.train_frappe",
            "task=imagepressure2smpl_carpet_v4_anysolebbox_cam3",
            f"task.split_csv=protocol://splits/default/{FOLDS[fold]}.csv",
            "task.seq_root=model-input://MotionPRO/adapter_v4_anysolebbox/cam3",
            f"task.name=imagepressure2smpl_v4_{fold}",
            f"task.unified_prediction_dir=results://baselines/MotionPRO/predictions/carpet_v4_anysolebbox_{fold}/eval_motion"]


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
    pending = list(FOLDS)[::-1]
    running: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    terminal: list[str] = []
    while pending or running:
        for fold in list(running):
            rec = running[fold]
            code = rec["proc"].poll()
            if code is None:
                continue
            del running[fold]
            if code == 0 and complete(fold):
                print(f"[ok] {fold}", flush=True)
                continue
            attempts[fold] = attempts.get(fold, 0) + 1
            print(f"[fail] {fold} rc={code} complete={complete(fold)} "
                  f"attempts={attempts[fold]}", flush=True)
            if attempts[fold] < MAX_RUNS:
                pending.append(fold)
            else:
                terminal.append(fold)
        free = free_mem()
        busy = {rec["gpu"] for rec in running.values()}
        choices = [g for g in GPUS if g not in busy and free.get(g, 0) >= 16000]
        for gpu in choices:
            if len(running) >= 4 or not pending:
                break
            fold = pending.pop()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WANDB_MODE"] = "disabled"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = LOGROOT / f"{fold}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(f"\n=== {fold} gpu={gpu} ===\n")
            handle.flush()
            proc = subprocess.Popen(make_cmd(fold), cwd=str(ROOT / "Baselines/MotionPRO"),
                                    env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[fold] = {"proc": proc, "gpu": gpu}
            print(f"[start] {fold} gpu={gpu}", flush=True)
        time.sleep(60)
    if terminal:
        print(f"TERMINAL FAILURES: {terminal}")
        return 1
    print("v4 training done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
