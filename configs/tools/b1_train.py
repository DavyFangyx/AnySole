#!/usr/bin/env python3
"""B1 config/27-node DAG, real pilots and full-state benchmark verification.

GPU admission remains root-owned. Formal workers require the shared controller
lease; pilot and benchmark commands are intended for explicitly allocated GPUs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.tools.a2_controller import atomic_json, digest, file_hash, inside, read_json, b1_plan_config_hash

B1_RUNS = ("B0", "Teacher", "B0-Continue", "A-Pressure", "B-Pose", "B-Pressure-HF",
           "B-Pressure-ALL", "B-Load-HF", "B-Load-ALL")
DEFAULT_RESULTS = WORKSPACE / "3_Result/AnySole/B1"
DEFAULT_CONTROL = DEFAULT_RESULTS / "controller"


def run_config(base, run="B0", seed=1, *, stage="pilot"):
    if run not in B1_RUNS:
        raise ValueError("unknown B1 run")
    value = copy.deepcopy(base)
    value.pop("a2_run_id", None)
    value.pop("structure_hash", None)
    kd = run.startswith("B-")
    value.update(b1_version="anysole.b1.v1", a2_version="anysole.a3.v1", b1_run_id=run,
        b1_role="teacher" if run == "Teacher" else "baseline" if run == "B0" else "student",
        seed=seed, a2_head_seed=seed, stage=stage, budget_mode="b1",
        protocol="anysole.b.cam3_holdout.v1", experiment="press_pair" if run == "A-Pressure" else "base_pool",
        pressure_head="query_factorized", query_memory="fusion_tactile_slice", query_layers=1, query_heads=8,
        query_hidden=128, pressure_loss="mse", lambda_contact=.01,
        train_cameras=[1, 2, 4], validation_cameras=[1, 2, 4], teacher_cameras=[1, 2, 4],
        teacher_reference_camera=1, teacher_view_dropout={"three_view_probability": .7, "min_views": 2},
        student_output_frame="camera_relative_origin", teacher_output_frame="camera1_relative_origin",
        coordinate_frame="camera_relative_origin", clock_anchor_camera=1,
        supervised_pressure_route="all", teacher_supervised_pressure_route="all", pressure_pair_route="HF",
        pressure_kd_route="all" if run.endswith("-ALL") else "HF", kd_quality_mode="valid_only",
        kd_pose_local23=.2 if kd else 0., kd_pressure=.02 if run.startswith("B-Pressure-") else 0.,
        kd_total=.02 if run.startswith("B-Load-") else 0., kd_allocation=.01 if run.startswith("B-Load-") else 0.,
        kd_contact=.005 if run.startswith("B-Load-") else 0., kd_ramp_frac=.1,
        kd_root=0., kd_velocity=0., kd_body_latent=0., kd_load_latent=0.,
        lambda_mv=.2 if run == "A-Pressure" else 0., mv_pose_weight=0., mv_pressure_weight=1.,
        effective_pose_pair_peak=0., effective_pressure_pair_peak=.2 if run == "A-Pressure" else 0.,
        effective_batch_rgb_windows=252, microbatch_rgb_windows=int(base.get("microbatch_rgb_windows") or 6),
        teacher_effective_batch_groups=84, target_updates=20720 if run in ("B0", "Teacher") else 10360,
        selection_motion_weight=.8, selection_pressure_weight=.2, num_workers=0, amp=True,
        save_initial_checkpoint=True, gradient_probe_updates=[], snapshot_updates=[],
        validation_every_updates=200, checkpoint_every_updates=200,
        wandb_mode="offline", wandb_online_preferred=False,
        wandb_entity="foundary-f-beijing-institute-of-technology", wandb_project="Anysole-B1",
        wandb_group="B1", wandb_job_type=run, lr_schedule="constant",
        results_root=str(DEFAULT_RESULTS), control_dir=str(base.get("control_dir") or DEFAULT_CONTROL),
        wandb_shared_dir=str(DEFAULT_RESULTS / "wandb-shared"),
        max_retries=2, disk_min_free_bytes=10*1024**3, heartbeat_sec=10,
        resource_version="anysole.b1.resource.v1")
    value["target_rgb_windows"] = value["target_updates"] * 252
    for key in ("reference_epochs", "reference_camera", "n_cam3_train_windows", "reference_batch_rgb_windows",
                "reference_steps_per_epoch", "pilot_schedule_update_offset", "timing_warmup_updates"):
        value.pop(key, None)
    if stage == "pilot":
        value.update(target_updates=25, target_rgb_windows=25*252, validation_every_updates=25,
                     checkpoint_every_updates=25, log_every_updates=1, timing_warmup_updates=5,
                     plan_frozen=False, resource_frozen=False)
        for key in ("selection_m_ref", "selection_p_ref"):
            if value.get(key) is None:
                value[key] = 1.
    else:
        value.pop("measurement_replica_id", None)
    return value


def base_config(science=None):
    import yaml
    base = yaml.safe_load((REPO / "anysole/configs/view_consistency.yaml").read_text())
    base = {key: value.replace("/private/Tactile", str(WORKSPACE)) if isinstance(value, str) else value
            for key, value in base.items()}
    entry = yaml.safe_load((REPO / "configs/b1.yaml").read_text())
    for key in ("dataset_root", "manifest_path", "split_csv", "feature_cache_root", "stats_path", "smpl_model_path"):
        entry[key] = str(inside(WORKSPACE / entry[key]))
    base.update(entry)
    base["control_dir"] = str(DEFAULT_CONTROL)
    if science is None:
        statistics = read_json(base["stats_path"])["pressure_train_statistics"]
        science = {**statistics["science_config"], "pressure_highload_threshold": statistics["highload_threshold_q90"]}
    base.update(science)
    base.update(split_csv=str(WORKSPACE / "tmp/splits.csv"),
                stats_path=str(REPO / "AnysoleWorkspace/model_inputs/stats/b1_stats.json"))
    for key in ("manifest_hash", "split_hash", "stats_hash", "stats_file_hash", "cache_index_hash", "frontend_hash",
                "frontend_runtime_hash", "source_hash", "structure_hash", "plan_hash", "selection_coverage_hash",
                "selection_m_ref", "selection_p_ref", "smpl_runtime_model_hash"):
        base[key] = None
    value = run_config(base)
    value.update(selection_m_ref=1., selection_p_ref=1.)
    return value


def fingerprint_config(config):
    from anysole import train
    from anysole.data.dataset import GroupedViewDataset
    from anysole.registry import get_builder
    result = dict(config)
    result.pop("structure_hash", None)
    result = get_builder("visual").effective_config(result)
    values, _, _, _ = train.visual_fingerprints(result)
    result.update(values)
    dataset = GroupedViewDataset(manifest=Path(result["manifest_path"]), dataset_root=Path(result["dataset_root"]),
        feature_cache_root=Path(result["feature_cache_root"]), smpl_model_path=Path(result["smpl_model_path"]),
        window_length=20, stride=result["train_stride"], cameras=(1, 2, 4), split="train", cover_tail=False)
    result.update(n_train_group_windows=len(dataset), n_train_rgb_windows=sum(len(w[3]) for w in dataset.windows),
                  train_sample_keys_hash=train.canonical_hash([dataset.sample_key(i) for i in range(len(dataset))]))
    return result


def bind_parent(config, run_dir, kind, path=None):
    """Bind future DAG edges to terminal parent's actual best checkpoint."""
    prefix = "baseline" if kind == "B0" else "teacher"
    source = "init_from_run" if kind == "B0" else "teacher_from_run"
    if path is None:
        parent = inside(config[source])
        record = read_json(parent / "run.json")
        if config["stage"] == "formal" and not (record.get("formal_complete") and record.get("state") == "completed"):
            raise ValueError("B1 parent is not a completed formal run")
        path = parent / "best.pt"
        if file_hash(path) != record.get("best_checkpoint_hash"):
            raise ValueError("B1 parent best checkpoint not atomically published")
    path = inside(path)
    config[prefix+"_checkpoint_path"] = str(path)
    config[prefix+"_checkpoint_hash"] = file_hash(path)


def verify_run(root, *, formal=True):
    record = read_json(root / "run.json")
    expected_state = "completed" if formal else "pilot_completed"
    if record.get("state") != expected_state or bool(record.get("formal_complete")) != formal:
        raise ValueError("B1 incomplete terminal state")
    import torch
    from anysole import train
    from anysole.utils.b1_training import validate_exposure
    checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["semantic_hash"] != record["semantic_hash"] or train.visual_run_identity(config) != record["semantic_hash"]:
        raise ValueError("B1 checkpoint/run identity mismatch")
    if checkpoint["training"]["successful_updates"] != config["target_updates"]:
        raise ValueError("B1 successful optimizer budget missing")
    if file_hash(root / "last.pt") != record.get("last_checkpoint_hash"):
        raise ValueError("B1 last checkpoint hash mismatch")
    if formal and (not (root / "best.pt").is_file() or file_hash(root / "best.pt") != record.get("best_checkpoint_hash")):
        raise ValueError("B1 qualified selected checkpoint missing")
    validate_exposure(checkpoint["training"], config, terminal=True)
    return True


def inner_training_interval(report_path):
    """Read immutable worker logs: step5->24 excludes terminal validation."""
    path = inside(Path(report_path).parent / "metrics.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_step = {row["progress/update"]: row for row in rows}
    begin, end = by_step[5]["unix_time"], by_step[24]["unix_time"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (begin, end)) or end <= begin:
        raise ValueError("B1 missing real inner training interval before terminal validation")
    return {"metrics_path": str(path), "metrics_sha256": file_hash(path),
            "training_started_unix": begin, "training_finished_unix": end}


def benchmark_evidence(root, config, began):
    import torch
    import subprocess
    from configs.tools.a2_train import reload_smoke
    from anysole import train
    record = read_json(root / "run.json")
    verify_run(root, formal=False)
    reloaded = reload_smoke(root, leased=False)
    rows = [json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()]
    by_step = {row["progress/update"]: row for row in rows}
    warmup, steps = int(config.get("timing_warmup_updates", 5)), config["target_updates"]
    if steps <= warmup or not set(range(1, steps+1)) <= set(by_step):
        raise ValueError("B1 benchmark needs all real step logs and timed updates after warmup")
    elapsed = by_step[steps]["unix_time"] - by_step[warmup]["unix_time"]
    timed_rgb = by_step[steps]["progress/rgb_windows"] - by_step[warmup]["progress/rgb_windows"]
    last = record["last_validation"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or int(visible) not in range(7):
        raise ValueError("B1 resource benchmark requires exactly one root-allocated physical GPU index")
    gpu_rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    hardware = {int(row.split(",")[0]): row.split(",") for row in gpu_rows}[int(visible)]
    checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    state = checkpoint["training"]
    ledger = [json.loads(line) for line in (root / "update_ledger.jsonl").read_text().splitlines()]
    timed_updates = [row for row in ledger if row.get("attempt_id") == record["attempt_id"] and
                     warmup < row["successful_update"] <= steps]
    if len(timed_updates) != steps-warmup:
        raise ValueError("B1 benchmark timed successful update ledger is incomplete")
    update_seconds = [row["seconds"] for row in timed_updates]
    import statistics
    training_seconds = sum(update_seconds)
    evidence = {**reloaded, "schema": "anysole.b1.benchmark.v1", "b1_run_id": config["b1_run_id"],
        "seed": config["seed"], "source_hash": config["source_hash"], "config_sha256": train.visual_run_identity(config),
        "warmup_updates": warmup, "timed_updates": steps-warmup, "successful_updates": steps,
        "timed_seconds": elapsed, "wall_seconds": time.time()-began,
        "timing_scope": "end-to-end step5 log through step25 log, including terminal full validation; not pure training throughput",
        "training_update_seconds_sum": training_seconds,
        "training_update_seconds_mean": statistics.mean(update_seconds),
        "training_update_seconds_median": statistics.median(update_seconds),
        "training_update_seconds_max": max(update_seconds),
        "training_update_rgb_windows_per_second": timed_rgb/training_seconds,
        "unattributed_data_validation_logging_overhead_seconds": max(0., elapsed-training_seconds),
        "phase_timing_limit": "update ledger times forward/backward/optimizer; data preparation and full validation are not individually timed",
        "checkpoint_seconds": record.get("checkpoint_seconds"),
        "timed_rgb_windows": timed_rgb,
        "timed_nominal_rgb_windows": (steps-warmup)*252,
        "timed_physical_group_windows": by_step[steps]["progress/physical_group_windows"]-by_step[warmup]["progress/physical_group_windows"],
        "timed_rgb_windows_per_second": timed_rgb/elapsed,
        "actual_rgb_windows": record["rgb_windows"], "effective_batch_rgb_windows": 252,
        "full_validation": last.get("val/motion_frames", 0) > 0 and last.get("val/pressure_cells", 0) > 0,
        "validation_coverage_hash": last.get("val/coverage_hash"), "amp_overflow_attempts": state["amp_overflow_attempts"],
        "attempt_steps": state["attempt_steps"],
        "timed_started_unix": by_step[warmup]["unix_time"], "timed_finished_unix": by_step[steps]["unix_time"],
        "physical_gpu_index": int(visible), "physical_gpu_uuid": hardware[1].strip(),
        "physical_gpu_total_bytes": int(hardware[2].strip())*1024**2,
        "cuda_visible_devices": visible, "microbatch_rgb_windows": config["microbatch_rgb_windows"],
        "cpu_threads": torch.get_num_threads(),
        "pressure_kd_route": config["pressure_kd_route"],
        "load_class": "teacher" if config["b1_role"] == "teacher" else "kd" if config["b1_run_id"].startswith("B-") else "baseline",
        "workload_contract": {key: config.get(key) for key in ("b1_run_id", "b1_role", "microbatch_rgb_windows",
            "effective_batch_rgb_windows", "amp", "pressure_kd_route", "kd_pose_local23", "kd_pressure", "kd_total", "kd_allocation", "kd_contact",
            "source_hash", "manifest_hash", "stats_hash", "cache_index_hash", "baseline_checkpoint_hash", "teacher_checkpoint_hash", "cpu_threads")},
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_rss_bytes": train.visual_health(torch.device("cuda:0"), root)["health/peak_rss_bytes"],
        "started_unix": began, "finished_unix": time.time(), "gpu_memory_fraction": config.get("gpu_memory_fraction")}
    atomic_json(root / "b1_benchmark.json", evidence)
    return evidence


def verify_resource(resource):
    """Bind measured per-class/full-loss benchmarks to a safe slot envelope."""
    if resource.get("b1_version") != "anysole.b1.v1" or resource.get("frozen") is not True:
        raise ValueError("B1 requires its own measured frozen resource envelope")
    common_source, contracts = None, {}
    for name, profile in resource["profiles"].items():
        if profile["kind"] == "cpu":
            continue
        refs = resource.get("evidence", {}).get(name, [])
        if not refs:
            raise ValueError(f"B1 missing real benchmark evidence for {name}")
        ns = []
        for ref in refs:
            path = inside(ref["path"])
            if file_hash(path) != ref["sha256"]:
                raise ValueError("B1 resource evidence changed")
            report = read_json(path)
            if report.get("schema") != "anysole.b1.cohort.v1" or report.get("load_class") != name or report.get("success") is not True:
                raise ValueError("B1 resource needs a successful measured cohort")
            if report.get("concurrent_overlap_seconds", 0) <= 0:
                raise ValueError("B1 resource workers lack overlapping measured update windows")
            ns.append(report["n"])
            contracts.setdefault(name, []).append(report.get("workload_contract"))
            workers = report.get("workers", [])
            if len(workers) != report["n"]:
                raise ValueError("B1 cohort worker count mismatch")
            worker_values = []
            for worker in workers:
                report_path = inside(worker["path"])
                if file_hash(report_path) != worker["sha256"]:
                    raise ValueError("B1 worker benchmark changed")
                value = read_json(report_path)
                worker_values.append(value)
                inner = inner_training_interval(report_path)
                if any(worker.get(key) != expected for key, expected in inner.items()):
                    raise ValueError("B1 cohort inner training interval/log hash mismatch")
                import yaml
                actual_config = yaml.safe_load((report_path.parent / "config.yaml").read_text())
                required = {"source_hash", "b1_run_id", "b1_role", "microbatch_rgb_windows", "effective_batch_rgb_windows",
                            "amp", "pressure_kd_route", "kd_pose_local23", "kd_pressure", "kd_total", "kd_allocation", "kd_contact",
                            "manifest_hash", "stats_hash", "cache_index_hash", "baseline_checkpoint_hash", "teacher_checkpoint_hash", "cpu_threads"}
                contract = value.get("workload_contract", {})
                if not required <= set(contract):
                    raise ValueError("B1 benchmark workload contract is incomplete")
                if any(actual_config.get(key) != expected for key, expected in contract.items()):
                    raise ValueError("B1 benchmark workload differs from saved trainer config")
                actual_run = actual_config["b1_run_id"]
                expected_coefficients = {
                    "kd_pose_local23": .2 if actual_run.startswith("B-") else 0.,
                    "kd_pressure": .02 if actual_run.startswith("B-Pressure-") else 0.,
                    "kd_total": .02 if actual_run.startswith("B-Load-") else 0.,
                    "kd_allocation": .01 if actual_run.startswith("B-Load-") else 0.,
                    "kd_contact": .005 if actual_run.startswith("B-Load-") else 0.}
                if actual_run not in B1_RUNS or any(actual_config.get(key) != expected for key, expected in expected_coefficients.items()):
                    raise ValueError("B1 measured workload must execute its full declared losses")
                if actual_run.startswith("B-") and (not actual_config.get("baseline_checkpoint_hash") or not actual_config.get("teacher_checkpoint_hash")):
                    raise ValueError("B1 KD measurement lacks actual teacher/baseline lineage")
                actual_class = "teacher" if actual_run == "Teacher" else "kd" if actual_run.startswith("B-") else "baseline"
                if (value.get("load_class") != name or actual_class != name or value.get("b1_run_id") != actual_run or
                        value.get("source_hash") != actual_config["source_hash"] or
                        value.get("cuda_visible_devices") != str(value.get("physical_gpu_index"))):
                    raise ValueError("B1 benchmark cannot be relabeled as a different workload")
                if (value.get("physical_gpu_uuid") != report.get("physical_gpu_uuid") or
                        value.get("physical_gpu_index") != report.get("physical_gpu_index")):
                    raise ValueError("B1 cohort workers did not use the same physical GPU")
                if value.get("workload_contract") != report.get("workload_contract"):
                    raise ValueError("B1 cohort workload/loss/source contract differs from its workers")
                source = actual_config["source_hash"]
                if not source or (common_source is not None and source != common_source):
                    raise ValueError("B1 resource evidence mixes scientific source revisions")
                common_source = source
                if (not value.get("full_validation") or not value.get("state_complete") or
                        not value.get("checkpoint_reloaded") or value.get("successful_updates") != 25 or
                        value.get("warmup_updates") != 5 or value.get("timed_updates") != 20 or
                        value.get("peak_gpu_reserved_bytes", math.inf) > profile["peak_gpu_bytes"] or
                        value.get("peak_rss_bytes", math.inf) > profile["peak_rss_bytes"]):
                    raise ValueError("B1 profile exceeds actual measured worker evidence")
            timed_begin = min(w["timed_started_unix"] for w in worker_values)
            timed_end = max(w["timed_finished_unix"] for w in worker_values)
            overlap = min(w["timed_finished_unix"] for w in worker_values)-max(w["timed_started_unix"] for w in worker_values)
            if overlap <= 0 or not math.isclose(overlap, report["concurrent_overlap_seconds"], rel_tol=1e-12):
                raise ValueError("B1 cohort measured update intervals did not actually overlap")
            inner_values = [inner_training_interval(w["path"]) for w in workers]
            inner_overlap = min(w["training_finished_unix"] for w in inner_values)-max(w["training_started_unix"] for w in inner_values)
            if inner_overlap <= 0 or not math.isclose(inner_overlap, report.get("training_overlap_seconds", -1), rel_tol=1e-12):
                raise ValueError("B1 cohort actual pre-validation training intervals did not overlap")
            throughput = sum(w["timed_rgb_windows"] for w in worker_values)/(timed_end-timed_begin)
            if not math.isclose(throughput, report.get("aggregate_rgb_windows_per_second", -1), rel_tol=1e-12):
                raise ValueError("B1 cohort throughput must use the common measured interval")
            if report["n"] == profile["cap"] and report["workload_contract"].get("microbatch_rgb_windows") != profile.get("microbatch_rgb_windows"):
                raise ValueError("B1 selected capacity/microbatch combination was not measured")
            if report["n"] == profile["cap"] and report["workload_contract"].get("cpu_threads") != profile.get("cpu_threads"):
                raise ValueError("B1 selected CPU threads differ from actual measured configuration")
            reserve = max(resource["gpu_reserve_bytes"], report["physical_gpu_total_bytes"]*resource["gpu_reserve_fraction"])
            if profile["cap"]*profile["peak_gpu_bytes"] > report["physical_gpu_total_bytes"]-reserve:
                raise ValueError("B1 selected concurrency exceeds conservative aggregate GPU reserve")
        if 1 not in ns or profile["cap"] not in ns:
            raise ValueError("B1 capacity requires actual n1 and selected-n measurements")
        if profile.get("source_hash") != common_source:
            raise ValueError("B1 frozen profile source is not its measured source")
    host_total = resource.get("host_total_bytes")
    if not isinstance(host_total, int) or host_total <= 0:
        raise ValueError("B1 requires measured host total memory")
    worst = max((p["peak_rss_bytes"]*p["cap"] for p in resource["profiles"].values() if p["kind"] != "cpu"), default=0)
    if 7*worst > host_total*(1-resource["host_reserve_fraction"]):
        raise ValueError("B1 seven-card aggregate host peak exceeds reserve")
    quota = resource.get("cpu_quota_cores")
    budget = resource.get("cpu_threads_budget")
    if not isinstance(quota, (int, float)) or not isinstance(budget, int) or not 0 < budget <= quota:
        raise ValueError("B1 CPU thread budget must bind the actual cgroup quota")
    reference = resource.get("seven_card_evidence", {})
    if not reference or file_hash(inside(reference["path"])) != reference.get("sha256"):
        raise ValueError("B1 needs a content-bound actual seven-card confirmation")
    seven = read_json(inside(reference["path"]))
    if seven.get("schema") != "anysole.b1.seven-card.v1" or seven.get("success") is not True:
        raise ValueError("B1 invalid seven-card receipt")
    representative = resource["profiles"].get(seven.get("load_class"), {})
    if seven.get("workload_contract") not in contracts.get(seven.get("load_class"), []):
        raise ValueError("B1 seven-card workload does not match a measured representative class contract")
    n = seven.get("n_per_gpu")
    counts, times, uuids = {}, [], {}
    for ref in seven.get("workers", []):
        path = inside(ref["path"])
        if file_hash(path) != ref["sha256"]:
            raise ValueError("B1 seven-card worker receipt changed")
        value = read_json(path)
        inner = inner_training_interval(path)
        if any(ref.get(key) != expected for key, expected in inner.items()):
            raise ValueError("B1 seven-card inner training interval/log hash mismatch")
        if (not value.get("full_validation") or not value.get("state_complete") or
                value.get("source_hash") != common_source or value.get("load_class") != seven["load_class"] or
                value.get("workload_contract") != seven.get("workload_contract") or
                value.get("peak_gpu_reserved_bytes", math.inf) > representative.get("peak_gpu_bytes", 0) or
                value.get("peak_rss_bytes", math.inf) > representative.get("peak_rss_bytes", 0) or
                value.get("successful_updates") != 25 or value.get("warmup_updates") != 5 or value.get("timed_updates") != 20):
            raise ValueError("B1 seven-card worker class/source/full-measurement mismatch")
        gpu = value.get("physical_gpu_index")
        if gpu not in range(7) or value.get("cuda_visible_devices") != str(gpu):
            raise ValueError("B1 seven-card worker physical GPU binding mismatch")
        counts[gpu] = counts.get(gpu, 0)+1
        uuids.setdefault(gpu, set()).add(value.get("physical_gpu_uuid"))
        times.append((inner["training_started_unix"], inner["training_finished_unix"]))
    if type(n) is not int or n < 1 or counts != {gpu: n for gpu in range(7)}:
        raise ValueError("B1 seven-card confirmation requires actual n workers on every GPU0..6")
    if any(len(v) != 1 or None in v for v in uuids.values()) or len(set.union(*uuids.values())) != 7:
        raise ValueError("B1 seven-card UUIDs do not represent seven distinct physical GPUs")
    overlap = min(end for _, end in times)-max(begin for begin, _ in times)
    if overlap <= 0 or not math.isclose(overlap, seven.get("training_overlap_seconds", -1), rel_tol=1e-12):
        raise ValueError("B1 seven-card actual pre-validation training intervals did not overlap")
    for profile in resource["profiles"].values():
        if profile["kind"] == "cpu":
            continue
        if (profile["cap"] > n or profile["peak_gpu_bytes"] > representative.get("peak_gpu_bytes", 0) or
                profile["peak_rss_bytes"] > representative.get("peak_rss_bytes", 0) or
                profile["cpu_threads"] > representative.get("cpu_threads", 0)):
            raise ValueError("B1 seven-card inheritance needs a conservative representative memory/CPU/cap envelope")
    return True


def selection_reference(base, root):
    import yaml
    from anysole import train
    root = inside(root)
    run_path, config_path, benchmark_path = (root/name for name in ("run.json", "config.yaml", "b1_benchmark.json"))
    run, benchmark = read_json(run_path), read_json(benchmark_path)
    config = yaml.safe_load(config_path.read_text())
    if (config.get("b1_run_id") != "B0" or config.get("seed") != 1 or config.get("stage") != "pilot" or
            config.get("target_updates") != 25 or config.get("lambda_mv") != 0 or config.get("kd_pose_local23") != 0 or
            config.get("source_hash") != base.get("source_hash") or run.get("state") != "pilot_completed" or
            run.get("successful_updates") != 25 or benchmark.get("successful_updates") != 25 or
            not all(benchmark.get(key) for key in ("checkpoint_reloaded", "state_complete", "full_validation")) or
            train.visual_run_identity(config) != run.get("semantic_hash") or benchmark.get("config_sha256") != run.get("semantic_hash")):
        raise ValueError("B1 selection references require the exact new-source clean seed1 B0 pilot at25 successful updates")
    values = train.visual_selection_references(run["last_validation"])
    values["selection_reference"] = {"run_json": str(run_path), "run_json_sha256": file_hash(run_path),
        "config_path": str(config_path), "config_sha256": file_hash(config_path),
        "benchmark_path": str(benchmark_path), "benchmark_sha256": file_hash(benchmark_path),
        "successful_update": 25, "seed": 1, "source_hash": config["source_hash"], "split": "val"}
    return {**base, **values}


def make_plan(base, directory, results, *, resource=None, reference_run=None):
    import yaml
    from anysole import train
    from anysole.registry import get_builder
    if reference_run is not None:
        base = selection_reference(base, reference_run)
    directory, results = inside(directory), inside(results)
    directory.mkdir(parents=True, exist_ok=True)
    sources = {str(p): file_hash(p) for tree in (REPO / "anysole", REPO / "configs/tools") for p in sorted(tree.rglob("*.py"))}
    jobs = []
    for seed in (1, 2, 3):
        for run in B1_RUNS:
            config = run_config(base, run, seed, stage="formal")
            load_class = "teacher" if run == "Teacher" else "kd" if run.startswith("B-") else "baseline"
            if resource is not None:
                profile = resource["profiles"][load_class]
                config.update(microbatch_rgb_windows=profile["microbatch_rgb_windows"],
                              cpu_threads=profile["cpu_threads"], runs_per_gpu=profile["cap"],
                              gpu_memory_fraction=profile.get("gpu_memory_fraction", min(.95, .85/profile["cap"])))
            run_id = f"{run}-s{seed}"
            out = results / run_id
            if run != "B0":
                config["init_from_run"] = str(results / f"B0-s{seed}")
            if run.startswith("B-"):
                config["teacher_from_run"] = str(results / f"Teacher-s{seed}")
            config.update(plan_frozen=True, resource_frozen=True, results_root=str(results))
            config = get_builder("visual").effective_config(config)
            path = directory / f"{run_id}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=True))
            argv = [sys.executable, "-m", "configs.tools.b1_train", "train", "--config", str(path), "--run-dir", str(out)]
            deps = [] if run == "B0" else [f"B0-s{seed}"]
            if run.startswith("B-"):
                deps.append(f"Teacher-s{seed}")
            job = {"id": run_id, "kind": "train", "b1_run_id": run, "seed": seed,
                "argv": argv, "resume_argv": argv+["--resume", str(out / "last.pt")],
                "verify_argv": [sys.executable, "-m", "configs.tools.b1_train", "verify", "--run-dir", str(out)],
                "cwd": str(REPO), "output_dir": str(out), "config_path": str(path), "config_sha256": file_hash(path),
                "source_files": sources, "inputs": {}, "dependencies": deps, "load_class": load_class,
                "target_updates": config["target_updates"], "target_rgb_windows": config["target_rgb_windows"],
                "allowed_gpus": list(range(7)), "max_recoveries": 1, "auto_recover_oom": True,
                "priority": 10 if run == "Teacher" else 0, "estimated_seconds": config["target_updates"],
                "heartbeat_timeout_sec": 300, "startup_timeout_sec": 1800}
            ref = config.get("selection_reference")
            if ref:
                job["inputs"].update({ref["run_json"]: ref["run_json_sha256"],
                                      ref["config_path"]: ref["config_sha256"],
                                      ref["benchmark_path"]: ref["benchmark_sha256"]})
            if run.startswith("B-"):
                job["quality_gate"] = {"baseline_run": str(results / f"B0-s{seed}"),
                    "teacher_run": str(results / f"Teacher-s{seed}"), "require_pressure": run != "B-Pose",
                    "pose_min_gain": .005, "pressure_min_gain": .005, "tail_max_relative_degradation": .02,
                    "contact_max_absolute_degradation": .02,
                    "interpretation": "predeclared pragmatic margin; not statistical proof"}
            jobs.append(job)
    plan = {"schema": "anysole.a2.plan.v1", "a2_version": "anysole.a3.v1", "b1_version": "anysole.b1.v1",
            "purpose": "formal", "frozen": True, "jobs": jobs,
            "capacity_max_n": max(p["cap"] for p in resource["profiles"].values() if p["kind"] != "cpu") if resource is not None else (base.get("runs_per_gpu") or 1),
            "config_plan_hash_scheme": "SHA256 complete plan excluding config_plan_hash and each job.config_sha256"}
    if resource is not None:
        plan["resource_sha256"] = digest(resource)
    plan["config_plan_hash"] = b1_plan_config_hash(plan)
    for job in jobs:
        path = Path(job["config_path"])
        value = yaml.safe_load(path.read_text())
        value["plan_hash"] = plan["config_plan_hash"]
        path.write_text(yaml.safe_dump(value, sort_keys=True))
        job["config_sha256"] = file_hash(path)
    atomic_json(directory / "plan.json", plan)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "plan", "train", "pilot", "benchmark", "verify", "verify-pilot", "cohort", "seven", "verify-resource"))
    p.add_argument("--config", type=Path)
    p.add_argument("--science-config", type=Path)
    p.add_argument("--resource", type=Path)
    p.add_argument("--reference-run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--run-id", choices=B1_RUNS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after-updates", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--baseline-checkpoint", type=Path)
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--microbatch-rgb", type=int)
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--load-class", choices=("baseline", "teacher", "kd"))
    p.add_argument("--reports", nargs="*", type=Path)
    a = p.parse_args(argv)
    import yaml
    if a.action == "prepare":
        config = fingerprint_config(base_config(read_json(a.science_config) if a.science_config else None))
        inside(a.output).write_text(yaml.safe_dump(config, sort_keys=True))
        print(json.dumps({"config": str(a.output), "source_hash": config["source_hash"]}))
        return 0
    if a.action == "plan":
        plan = make_plan(yaml.safe_load(a.config.read_text()), a.output, a.results_root,
                         resource=read_json(a.resource) if a.resource else None, reference_run=a.reference_run)
        print(json.dumps({"plan": str(a.output / "plan.json"), "jobs": len(plan["jobs"]), "sha256": digest(plan)}))
        return 0
    if a.action == "verify-resource":
        print(json.dumps({"verified": verify_resource(read_json(a.config))}))
        return 0
    if a.action in ("cohort", "seven"):
        reports = [read_json(path) for path in a.reports]
        if not reports or any(r.get("schema") != "anysole.b1.benchmark.v1" for r in reports):
            raise ValueError("cohort requires real B1 worker benchmark reports")
        if any(r.get("load_class") != a.load_class for r in reports):
            raise ValueError("B1 cohort worker workload does not match the declared class")
        if a.action == "cohort" and (len({r["physical_gpu_uuid"] for r in reports}) != 1 or len({r["physical_gpu_index"] for r in reports}) != 1):
            raise ValueError("B1 n measurement requires workers on the same physical GPU")
        if len({digest(r["workload_contract"]) for r in reports}) != 1:
            raise ValueError("B1 cohort source/data/loss/microbatch contracts differ")
        begin, end = min(r["timed_started_unix"] for r in reports), max(r["timed_finished_unix"] for r in reports)
        success = all(r.get("full_validation") and r.get("state_complete") and r.get("checkpoint_reloaded") for r in reports)
        value = {"schema": "anysole.b1.cohort.v1", "n": len(reports), "load_class": a.load_class,
            "success": success, "started_unix": begin, "finished_unix": end,
            "concurrent_overlap_seconds": min(r["timed_finished_unix"] for r in reports)-max(r["timed_started_unix"] for r in reports),
            "aggregate_rgb_windows_per_second": sum(r["timed_rgb_windows"] for r in reports)/(end-begin),
            "physical_gpu_index": reports[0]["physical_gpu_index"], "physical_gpu_uuid": reports[0]["physical_gpu_uuid"],
            "physical_gpu_total_bytes": reports[0]["physical_gpu_total_bytes"],
            "workload_contract": reports[0]["workload_contract"],
            "timing_scope": "end-to-end20 successful updates including terminal validation; actual training overlap uses step5->24",
            "workers": [{"path": str(inside(path)), "sha256": file_hash(path), **inner_training_interval(path)} for path in a.reports]}
        if value["concurrent_overlap_seconds"] <= 0:
            raise ValueError("workers did not actually overlap")
        value["training_overlap_seconds"] = min(w["training_finished_unix"] for w in value["workers"])-max(w["training_started_unix"] for w in value["workers"])
        if value["training_overlap_seconds"] <= 0:
            raise ValueError("actual step5->24 training intervals did not overlap")
        if a.action == "seven":
            counts = {gpu: sum(r["physical_gpu_index"] == gpu for r in reports) for gpu in range(7)}
            if not all(counts.values()) or len(set(counts.values())) != 1 or len({r["physical_gpu_uuid"] for r in reports}) != 7:
                raise ValueError("seven confirmation needs equal actualn on GPUs0..6 and seven distinct physical UUIDs")
            value.update(schema="anysole.b1.seven-card.v1", n_per_gpu=counts[0], per_gpu_workers=counts)
        atomic_json(a.output, value)
        print(json.dumps(value))
        return 0
    root = inside(a.run_dir)
    if a.action in ("verify", "verify-pilot"):
        print(json.dumps({"verified": verify_run(root, formal=a.action == "verify")}))
        return 0
    config = yaml.safe_load(a.config.read_text())
    if a.action in ("pilot", "benchmark"):
        config = run_config(config, a.run_id or config["b1_run_id"], a.seed)
        if a.steps is not None:
            config.update(target_updates=a.steps, target_rgb_windows=a.steps*252,
                          validation_every_updates=a.steps, checkpoint_every_updates=a.steps)
    if a.memory_fraction is not None:
        config["gpu_memory_fraction"] = a.memory_fraction
    if a.microbatch_rgb is not None:
        if config["stage"] == "formal":
            raise ValueError("formal microbatch must come from its frozen config")
        config["microbatch_rgb_windows"] = a.microbatch_rgb
    if a.resume is not None:
        # Reuse the exact resolved parent hashes and successful stage horizon.
        saved_config = yaml.safe_load((root / "config.yaml").read_text())
        for key in ("baseline_checkpoint_path", "baseline_checkpoint_hash", "teacher_checkpoint_path", "teacher_checkpoint_hash"):
            if key in saved_config:
                config[key] = saved_config[key]
    else:
        if config["b1_run_id"] != "B0":
            bind_parent(config, root, "B0", a.baseline_checkpoint)
        if config["b1_run_id"].startswith("B-"):
            bind_parent(config, root, "Teacher", a.teacher_checkpoint)
    if config["stage"] == "formal":
        if "measurement_replica_id" in config:
            raise ValueError("B1 formal runs prohibit pilot measurement_replica_id")
        from configs.tools.a2_controller import require_worker_lease
        lease = require_worker_lease(kind="train")
        if str(root) != str(inside(lease["job"]["output_dir"])):
            raise ValueError("B1 output directory differs from leased job")
        if config["plan_hash"] != lease.get("config_plan_hash"):
            raise ValueError("B1 config plan binding differs from exact granted plan")
    from anysole import train
    import torch
    selected_threads = a.cpu_threads if a.cpu_threads is not None else config.get("cpu_threads")
    if selected_threads is not None:
        if type(selected_threads) is not int or selected_threads < 1:
            raise ValueError("positive CPU thread count required")
        torch.set_num_threads(selected_threads)
    config["cpu_threads"] = torch.get_num_threads()
    if config["stage"] == "pilot":
        config["measurement_replica_id"] = str(root.resolve())
    began = time.time()
    result = train.train_visual_run(config, root, resume=a.resume, stop_after_updates=a.stop_after_updates)
    if a.action == "benchmark":
        benchmark_evidence(root, yaml.safe_load((root / "config.yaml").read_text()), began)
    print(json.dumps({"run_dir": str(root), "state": result["state"], "successful_updates": result["successful_updates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
