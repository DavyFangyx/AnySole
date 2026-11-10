#!/usr/bin/env python3
"""Four-fold LSO pipeline for the MotionPRO and MMVP (FPP-Net + PoseTransOpt)
baselines on the 2026-10-24 raw (BVH24/SMPL-new + tactile v3.1).

Stages per fold (fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8):

    motionpro_train  -> (test + unified export inside the training entry)
    fpp_train        -> fpp_infer -> pto_build -> pto_run -> pto_export
    eval (needs motionpro predictions and pto_export)

The runner is resumable (artifact checks + state file), waits for a free GPU
in the user-approved set {2,3,5,6}, keeps at most 4 concurrent jobs, and
auto-requeues failed jobs (GPU OOM is the common transient).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/fangyuxuan/projects/gait")
TOUCH = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
MMVP = "/data/fangyuxuan/miniconda3/envs/mmvp/bin/python"
# The lab gpu-watch enforces at most 4 distinct GPUs per Unix user and
# SIGTERMs *all* of that user's GPU processes on violation.  Keep a fixed
# 4-GPU set, one job per GPU, and never touch GPU0/1/2/4 (other users /
# other fangyuxuan jobs would push the user over the limit).
ALLOWED_GPUS = (3, 5, 6, 7)
GPU_SLOTS = {3: 1, 5: 1, 6: 1, 7: 1}
MAX_CONCURRENT = 4
MAX_MOTIONPRO_RUNNING = 0  # all four MP folds are complete under the LR-floor protocol
CPU_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
STATE = ROOT / ".tmp_scripts" / "baseline_fourfold_state.json"
LOG_ROOT = ROOT / "results" / "diagnostics" / "baselines_fourfold_20261109"

FOLDS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12",
    "fS5S11S12": "splits_test_S5S11S12_val_S13",
    "fS6S7": "splits_test_S6S7_val_S14",
    "fS5S6S8": "splits_test_S5S6S8_val_S7",
}
# All-fake / no-valid-window sessions: present in the split CSV but every
# evaluator and every model dataset drops them.  They must not gate artifact
# completion or be passed to PoseTransOpt.
EMPTY_SESSIONS = {"S12102"}
GPU_REQ = {"motionpro_train": 16000, "fpp_train": 7000, "fpp_infer": 4500,
           "pto_run": 6500}
MARGIN = {"motionpro_train": 500, "fpp_train": 2500, "fpp_infer": 2000,
          "pto_run": 2500}
MAX_ATTEMPTS = 30
RETRY_SECONDS = 600


class AdoptedProc:
    """Popen-like view of a process this runner did not launch."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        return None if Path(f"/proc/{self.pid}").exists() else 0


def process_table() -> list[tuple[int, int, str]]:
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def adopt_running(jobs: dict[str, dict], state: dict) -> dict[str, dict]:
    """Re-attach to jobs a previous runner launched but did not reap."""
    table = process_table()
    adopted = {}
    for job_id, record in state.items():
        if record.get("status") != "running" or job_id not in jobs:
            continue
        job = jobs[job_id]
        if job["kind"] == "motionpro_train":
            needle = f"task.name=imagepressure2smpl_v3_{job['fold']}"
        elif job["kind"] == "fpp_train":
            needle = f"--config configs/temporalKPSMPLCont_series5_mlp_{job['fold']}.yaml"
        elif job["kind"] == "fpp_infer":
            needle = f"--config configs/temporalKPSMPLCont_series5_mlp_{job['fold']}.yaml"
        elif job["kind"] == "pto_run":
            needle = f"pose_optimization_{job['fold']}"
        else:
            continue
        main = [pid for pid, ppid, args in table
                if needle in args and ppid == 1 and "app.train" in args]
        if job["kind"] == "pto_run":
            main = [pid for pid, ppid, args in table
                    if needle in args and ppid == 1]
        if main:
            adopted[job_id] = {"proc": AdoptedProc(main[0]),
                               "gpu": record.get("gpu"), "handle": None,
                               "job": job}
    return adopted


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def fold_sessions(fold: str, column: str) -> list[str]:
    path = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{FOLDS[fold]}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return [r[column].strip() for r in rows if (r.get(column) or "").strip()]


def evaluable_sessions(fold: str) -> list[str]:
    return [sid for sid in fold_sessions(fold, "test") if sid not in EMPTY_SESSIONS]


def state_load() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def state_save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def gpu_free() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    free = {}
    for line in out.splitlines():
        idx, mem = [part.strip() for part in line.split(",")[:2]]
        free[int(idx)] = int(mem)
    return free


def artifact_done(job: dict) -> bool:
    kind = job["kind"]
    fold = job["fold"]
    if kind == "motionpro_train":
        root = ROOT / f"results/baselines/MotionPRO/predictions/carpet_v3_bvh24_{fold}/eval_motion"
        return all((root / f"{sid}.npz").is_file() for sid in evaluable_sessions(fold))
    if kind == "fpp_train":
        latest = ROOT / f"results/baselines/FPP-Net/checkpoints/tempKPSMPL_series5_mlp_{fold}/latest"
        log_path = LOG_ROOT / f"fpp_train_{fold}.log"
        completed = False
        if log_path.is_file():
            completed = "Training completed" in log_path.read_text(
                encoding="utf-8", errors="ignore")[-50000:]
        return latest.is_file() and completed
    if kind == "fpp_infer":
        root = ROOT / f"AnysoleWorkspace/work/VP-MoCap/v1/fpp_predictions_{fold}"
        return all((root / d / s / sid / "pred_contact_smpl").is_dir()
                   for sid, d, s in _session_parts(fold))
    if kind == "pto_build":
        root = ROOT / f"AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1_{fold}"
        return all((root / d / s / sid / "join_manifest.json").is_file()
                   for sid, d, s in _session_parts(fold))
    if kind == "pto_run":
        root = ROOT / f"AnysoleWorkspace/work/VP-MoCap/v1/pose_optimization_{fold}"
        return all((root / d / s / sid / "opt_result.pth").is_file()
                   for sid, d, s in _session_parts(fold))
    if kind == "pto_export":
        root = ROOT / f"results/baselines/VP-MoCap/predictions/vp_mocap_{fold}/eval_motion"
        return all((root / f"{sid}.npz").is_file() for sid in evaluable_sessions(fold))
    if kind == "eval":
        return ((ROOT / f"results/baselines/MotionPRO/metrics/{fold}_comparison.json").is_file()
                and (ROOT / f"results/baselines/VP-MoCap/metrics/{fold}_comparison.json").is_file())
    return False


_MANIFEST = None


def _session_parts(fold: str):
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {}
        for line in (ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                _MANIFEST[row["session_id"]] = row
    for sid in evaluable_sessions(fold):
        row = _MANIFEST[sid]
        video = row["video_path"].replace("raw://", str(ROOT / "AnysoleWorkspace/hf-raw") + "/")
        parts = Path(video).parts
        yield sid, parts[-3], parts[-2]


def build_jobs() -> list[dict]:
    jobs = []
    for fold in FOLDS:
        env = dict(os.environ)
        jobs.append({
            "id": f"motionpro_train_{fold}", "kind": "motionpro_train", "fold": fold,
            "deps": [], "gpu": True,
            "cmd": [TOUCH, "-m", "app.train_frappe",
                    "task=imagepressure2smpl_carpet_v3_bvh24_cam3",
                    f"task.split_csv=protocol://splits/default/{FOLDS[fold]}.csv",
                    f"task.name=imagepressure2smpl_v3_{fold}",
                    f"task.unified_prediction_dir=results://baselines/MotionPRO/predictions/carpet_v3_bvh24_{fold}/eval_motion"],
            "cwd": ROOT / "Baselines/MotionPRO",
            "env_extra": {"WANDB_MODE": "disabled",
                          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                          **CPU_ENV},
        })
        jobs.append({
            "id": f"fpp_train_{fold}", "kind": "fpp_train", "fold": fold,
            "deps": [], "gpu": True,
            "cmd": [TOUCH, "-m", "app.train_temporal",
                    "--config", f"configs/temporalKPSMPLCont_series5_mlp_{fold}.yaml",
                    "--batch_size", "32", "--num_threads", "4", "--gpus", "0",
                    "--early-stop-lr-floor", "5e-8"],
            "cwd": ROOT / "Baselines/VP-MoCap/FPP-Net",
            "env_extra": {"NVIDIA_TF32_OVERRIDE": "0", **CPU_ENV},
        })
        jobs.append({
            "id": f"fpp_infer_{fold}", "kind": "fpp_infer", "fold": fold,
            "deps": [f"fpp_train_{fold}"], "gpu": True,
            "cmd": [TOUCH, "-m", "app.infer_smplcont",
                    "--config", f"configs/temporalKPSMPLCont_series5_mlp_{fold}.yaml",
                    "--phase", "test", "--batch_size", "32", "--num_threads", "4",
                    "--no_visualization", "--gpus", "0"],
            "cwd": ROOT / "Baselines/VP-MoCap/FPP-Net",
            "env_extra": {"NVIDIA_TF32_OVERRIDE": "0", **CPU_ENV},
        })
        sessions = ",".join(evaluable_sessions(fold))
        jobs.append({
            "id": f"pto_build_{fold}", "kind": "pto_build", "fold": fold,
            "deps": [f"fpp_infer_{fold}"], "gpu": False,
            "cmd": [TOUCH, "AnysoleWorkspace/tool/adapters/mmvp_series/posetransopt/build_inputs.py",
                    "--sessions", sessions,
                    "--fpp-root", f"AnysoleWorkspace/work/VP-MoCap/v1/fpp_predictions_{fold}",
                    "--input-root", f"AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1_{fold}",
                    "--force"],
            "cwd": ROOT, "env_extra": {},
        })
        jobs.append({
            "id": f"pto_run_{fold}", "kind": "pto_run", "fold": fold,
            "deps": [f"pto_build_{fold}"], "gpu": True,
            "cmd": [MMVP, "run_full_mmvp.py", "--sessions", sessions,
                    "--input-root", str(ROOT / f"AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1_{fold}"),
                    "--output-root", str(ROOT / f"AnysoleWorkspace/work/VP-MoCap/v1/pose_optimization_{fold}"),
                    "--gpu", "0", "--no-visualization"],
            "cwd": ROOT / "Baselines/VP-MoCap/PoseTransOpt", "env_extra": {},
        })
        jobs.append({
            "id": f"pto_export_{fold}", "kind": "pto_export", "fold": fold,
            "deps": [f"pto_run_{fold}"], "gpu": False,
            "cmd": [TOUCH, "AnysoleWorkspace/tool/export_baseline_motion.py",
                    "--model", "vp_mocap", "--session", sessions,
                    "--native-root", f"AnysoleWorkspace/work/VP-MoCap/v1/pose_optimization_{fold}",
                    "--variant", f"vp_mocap_{fold}", "--force"],
            "cwd": ROOT, "env_extra": {},
        })
        jobs.append({
            "id": f"eval_{fold}", "kind": "eval", "fold": fold,
            "deps": [f"motionpro_train_{fold}", f"pto_export_{fold}"], "gpu": False,
            "cmd": [TOUCH, "results_display/script/r_test2_compare.py",
                    "--split", fold,
                    "--split-csv", f"AnysoleWorkspace/protocol/splits/crossfit_woS13/{fold}.csv",
                    "--models-config", f"AnysoleWorkspace/protocol/splits/crossfit_woS13/motionpro_mmvp_registry_{fold}.yaml",
                    "--write-model-metrics", "--force",
                    "--out-dir", f"results_display/Etest_crossfit_v8/detail/fourfold_baseline_eval/{fold}"],
            "cwd": ROOT, "env_extra": {},
        })
    return jobs


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state = state_load()
    jobs = {job["id"]: job for job in build_jobs()}
    adopted = adopt_running(jobs, state)
    # only un-adopted "running" records are stale (their process is gone)
    for job_id, record in list(state.items()):
        if record.get("status") == "running" and job_id not in adopted:
            state[job_id] = {"status": "pending", "at": time.time()}
    for job_id in adopted:
        log(f"ADOPT {job_id} pid={adopted[job_id]['proc'].pid} gpu={adopted[job_id]['gpu']}")
    running: dict[str, dict] = dict(adopted)
    log(f"pipeline start: {len(jobs)} jobs, allowed GPUs {ALLOWED_GPUS}, max {MAX_CONCURRENT}")
    state_save(state)
    while True:
        # ---- reap finished processes
        for job_id, record in list(running.items()):
            code = record["proc"].poll()
            if code is None:
                continue
            running.pop(job_id)
            if record.get("handle") is not None:
                record["handle"].close()
            job = jobs[job_id]
            if code == 0 and not artifact_done(job):
                # adopted/exited process without its completion artifacts
                code = 1
            if code == 0:
                job["done"] = True
                state[job_id] = {"status": "done", "at": time.time()}
                log(f"OK {job_id}")
            else:
                attempts = state.get(job_id, {}).get("attempts", 0) + 1
                state[job_id] = {"status": "failed", "attempts": attempts,
                                 "code": code, "at": time.time()}
                log(f"FAIL {job_id} code={code} attempts={attempts}")
            state_save(state)

        done = {jid for jid, job in jobs.items()
                if state.get(jid, {}).get("status") == "done" or job.get("done")}
        # promote artifact-complete jobs (resume without a state file)
        for jid, job in jobs.items():
            if jid not in done and job["kind"] != "eval" and artifact_done(job):
                job["done"] = True
                done.add(jid)
                state[jid] = {"status": "done", "at": time.time(), "artifact": True}
        if len(done) == len(jobs):
            log("all jobs done")
            break

        # ---- schedule
        free = gpu_free()
        slots = MAX_CONCURRENT - len(running)
        gpu_load: dict[int, int] = {gpu: 0 for gpu in ALLOWED_GPUS}
        for record in running.values():
            if record["gpu"] is not None:
                gpu_load[record["gpu"]] += 1
        if slots > 0:
            candidates = []
            for jid, job in jobs.items():
                if jid in done or jid in running:
                    continue
                if state.get(jid, {}).get("status") == "failed" and \
                        state[jid].get("attempts", 0) > MAX_ATTEMPTS:
                    continue
                if any(dep not in done for dep in job["deps"]):
                    continue
                # retry backoff
                rec = state.get(jid, {})
                if rec.get("status") == "failed" and time.time() - rec.get("at", 0) < RETRY_SECONDS:
                    continue
                candidates.append(job)
            # long/GPU-heavy first, but let light jobs fill gaps
            candidates.sort(key=lambda j: (not j["gpu"], j["kind"] != "motionpro_train"))
            mp_active = sum(1 for r in running.values()
                            if r["job"]["kind"] == "motionpro_train")
            for job in candidates:
                if slots <= 0:
                    break
                if job["kind"] == "motionpro_train" and mp_active >= MAX_MOTIONPRO_RUNNING:
                    continue
                if job["gpu"]:
                    need = GPU_REQ[job["kind"]] + MARGIN[job["kind"]]
                    choices = [g for g in ALLOWED_GPUS
                               if gpu_load.get(g, 0) < GPU_SLOTS.get(g, 1)
                               and free.get(g, 0) >= need]
                    if not choices:
                        # allow a lighter job to take a GPU that MotionPRO cannot
                        continue
                    gpu = max(choices, key=lambda g: free[g])
                    env = os.environ.copy()
                    env.update(job["env_extra"])
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    job["gpu_id"] = gpu
                    free[gpu] -= need
                    gpu_load[gpu] += 1
                else:
                    gpu = None
                    env = os.environ.copy()
                    env.update(job["env_extra"])
                log_path = LOG_ROOT / f"{job['id']}.log"
                log(f"START {job['id']} gpu={gpu} log={log_path.name}")
                handle = log_path.open("a", encoding="utf-8")
                handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START {job['id']} =====\n")
                handle.flush()
                proc = subprocess.Popen(job["cmd"], cwd=str(job["cwd"]), env=env,
                                        stdout=handle, stderr=subprocess.STDOUT)
                running[job["id"]] = {"proc": proc, "gpu": gpu, "handle": handle, "job": job}
                state[job["id"]] = {"status": "running", "gpu": gpu, "at": time.time()}
                if job["kind"] == "motionpro_train":
                    mp_active += 1
                slots -= 1
            state_save(state)

        time.sleep(60)

    log("pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
