"""Synthetic CPU checks of B1 scientific admission, never GPU receipts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tools.a2_controller import b1_quality_gate, b1_plan_config_hash, ready_jobs, file_hash
from configs.tools.b1_train import verify_resource, inner_training_interval


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def quality_case(root, *, pose_gain=.01, pressure_gain=.01):
    def metrics(factor):
        result = {"val/gate_coverage_hash": "samephysicalframes"}
        for name in ("pose_local23", "pressure_rmse", "total_rmse", "active_rmse", "highload_rmse", "peak_mae"):
            result["val/gate/"+name] = factor
            result["val/gate/"+name+"_count"] = 100
        result["val/gate/contact_f1"] = .8
        return result
    baseline = metrics(1.)
    teacher = metrics(1-pressure_gain)
    teacher["val/gate/pose_local23"] = 1-pose_gain
    for name, values in (("baseline", baseline), ("teacher", teacher)):
        directory = root/name
        directory.mkdir(parents=True)
        (directory/"best.pt").write_bytes(b"synthetic checkpoint unit fixture")
        write_json(directory/"run.json", {"state": "completed", "formal_complete": True,
            "best_checkpoint_hash": file_hash(directory/"best.pt"), "best_validation": values})
    return dict(baseline_run=str(root/"baseline"), teacher_run=str(root/"teacher"), require_pressure=True,
                pose_min_gain=.005, pressure_min_gain=.005, tail_max_relative_degradation=.02,
                contact_max_absolute_degradation=.02)


def test_teacher_gate_requires_pose_and_pressure_gain_and_exact_support(tmp_path):
    contract = quality_case(tmp_path)
    assert b1_quality_gate(contract)["passed"]
    path = tmp_path/"teacher/run.json"
    record = json.loads(path.read_text())
    record["best_validation"]["val/gate/highload_rmse"] = 1.03
    write_json(path, record)
    assert not b1_quality_gate(contract)["passed"]
    assert b1_quality_gate({**contract, "require_pressure": False})["passed"]
    record["best_validation"]["val/gate_coverage_hash"] = "differentframes"
    write_json(path, record)
    with pytest.raises(ValueError, match="exact same"):
        b1_quality_gate(contract)


def test_teacher_gate_preserves_no_advantage_and_checks_actual_checkpoint(tmp_path):
    contract = quality_case(tmp_path, pose_gain=0.)
    result = b1_quality_gate(contract)
    assert not result["passed"] and not result["pose_eligible"]
    assert result["reason"] == "teacher_pose_advantage_absent"
    (tmp_path/"teacher/best.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="publication"):
        b1_quality_gate(contract)


def test_pose_gate_does_not_require_nonempty_pressure_tail_support(tmp_path):
    contract = quality_case(tmp_path)
    path = tmp_path/"teacher/run.json"
    record = json.loads(path.read_text())
    record["best_validation"]["val/gate/highload_rmse_count"] = 0
    record["best_validation"]["val/gate/highload_rmse"] = None
    write_json(path, record)
    assert b1_quality_gate({**contract, "require_pressure": False})["passed"]
    result = b1_quality_gate(contract)
    assert not result["passed"] and result["pose_eligible"] and not result["pressure_eligible"]
    assert "highload" in result["pressure_evidence_error"]


def test_ready_dag_cannot_dispatch_unpassed_or_failed_scientific_gate():
    plan = {"jobs": [{"id": "B0", "kind": "train"}, {"id": "Teacher", "kind": "train", "dependencies": ["B0"]},
                     {"id": "B-Pose", "kind": "train", "dependencies": ["B0", "Teacher"], "quality_gate": {"required": True}}]}
    state = {"jobs": {"B0": {"state": "completed"}, "Teacher": {"state": "completed"}, "B-Pose": {"state": "pending"}}}
    assert not ready_jobs(plan, state)
    state["jobs"]["B-Pose"]["quality_gate"] = {"passed": False}
    assert not ready_jobs(plan, state)
    state["jobs"]["B-Pose"]["quality_gate"] = {"passed": True}
    assert [j["id"] for j in ready_jobs(plan, state)] == ["B-Pose"]


def test_plan_binding_does_not_reference_embedded_config_serialization():
    plan = {"schema": "test", "jobs": [{"id": "B0-s1", "config_sha256": "first", "target_updates": 20720}]}
    before = b1_plan_config_hash(plan)
    plan["config_plan_hash"] = before
    plan["jobs"][0]["config_sha256"] = "after embedding plan hash in yaml"
    assert b1_plan_config_hash(plan) == before
    plan["jobs"][0]["target_updates"] = 1
    assert b1_plan_config_hash(plan) != before


def synthetic_resource(root):
    """Shape-only receipts for validator tests; not real run evidence."""
    profiles, evidence = {}, {}
    for name, run in (("baseline", "B0"), ("teacher", "Teacher"), ("kd", "B-Load-HF")):
        directory = root/name
        directory.mkdir(parents=True)
        workload = {"source_hash": "a"*64, "b1_run_id": run, "microbatch_rgb_windows": 84,
            "b1_role": "teacher" if run == "Teacher" else "baseline" if run == "B0" else "student",
            "effective_batch_rgb_windows": 252, "amp": True, "pressure_kd_route": "HF", "cpu_threads": 2,
            "kd_pose_local23": .2 if run.startswith("B-") else 0., "kd_pressure": 0.,
            "kd_total": .02 if run.startswith("B-Load-") else 0.,
            "kd_allocation": .01 if run.startswith("B-Load-") else 0.,
            "kd_contact": .005 if run.startswith("B-Load-") else 0.,
            "manifest_hash": "a"*64, "stats_hash": "a"*64, "cache_index_hash": "a"*64,
            "baseline_checkpoint_hash": None if run == "B0" else "a"*64,
            "teacher_checkpoint_hash": "a"*64 if run.startswith("B-") else None}
        (directory/"config.yaml").write_text(yaml.safe_dump(workload))
        worker = dict(schema="anysole.b1.benchmark.v1", b1_run_id=run, load_class=name,
            source_hash="a"*64, workload_contract=workload, full_validation=True, state_complete=True,
            checkpoint_reloaded=True, successful_updates=25, warmup_updates=5, timed_updates=20,
            timed_started_unix=10., timed_finished_unix=20., timed_rgb_windows=5040,
            physical_gpu_index=0, physical_gpu_uuid="synthetic-GPU0", cuda_visible_devices="0", peak_gpu_reserved_bytes=1024**3,
            peak_rss_bytes=1024**3)
        worker_path = directory/"worker.json"
        write_json(worker_path, worker)
        (directory/"metrics.jsonl").write_text("\n".join(json.dumps({"progress/update": step, "unix_time": 7.5+.5*step}) for step in range(1, 26))+"\n")
        cohort = dict(schema="anysole.b1.cohort.v1", n=1, load_class=name, success=True,
            concurrent_overlap_seconds=10., training_overlap_seconds=9.5, aggregate_rgb_windows_per_second=504.,
            physical_gpu_index=0, physical_gpu_uuid="synthetic-GPU0", physical_gpu_total_bytes=48*1024**3,
            workload_contract=workload, workers=[{"path": str(worker_path), "sha256": file_hash(worker_path), **inner_training_interval(worker_path)}])
        cohort_path = directory/"cohort.json"
        write_json(cohort_path, cohort)
        profiles[name] = dict(kind="train", cap=1, microbatch_rgb_windows=84, source_hash="a"*64,
                              peak_gpu_bytes=2*1024**3, peak_rss_bytes=2*1024**3, cpu_threads=2)
        evidence[name] = [{"path": str(cohort_path), "sha256": file_hash(cohort_path)}]
    base = json.loads((root/"baseline/worker.json").read_text())
    workers = []
    for gpu in range(7):
        value = {**base, "physical_gpu_index": gpu, "cuda_visible_devices": str(gpu), "physical_gpu_uuid": f"synthetic-GPU{gpu}"}
        path = root/"seven"/f"gpu{gpu}"/"worker.json"
        write_json(path, value)
        (path.parent/"metrics.jsonl").write_text((root/"baseline/metrics.jsonl").read_text())
        workers.append({"path": str(path), "sha256": file_hash(path), **inner_training_interval(path)})
    seven_path = root/"seven.json"
    write_json(seven_path, {"schema": "anysole.b1.seven-card.v1", "load_class": "baseline", "success": True,
        "n_per_gpu": 1, "training_overlap_seconds": 9.5, "workers": workers, "workload_contract": base["workload_contract"]})
    return dict(b1_version="anysole.b1.v1", frozen=True, profiles=profiles, evidence=evidence,
                seven_card_evidence={"path": str(seven_path), "sha256": file_hash(seven_path)},
                gpu_reserve_bytes=2*1024**3, gpu_reserve_fraction=.1,
                host_reserve_fraction=.2, host_total_bytes=256*1024**3,
                cpu_quota_cores=32, cpu_threads_budget=32)


def test_resource_refuses_baseline_receipt_relabelled_kd_and_stale_source(tmp_path):
    resource = synthetic_resource(tmp_path)
    assert verify_resource(resource)
    resource["evidence"]["kd"] = resource["evidence"]["baseline"]
    with pytest.raises(ValueError, match="cohort"):
        verify_resource(resource)
    resource = synthetic_resource(tmp_path/"second")
    resource["profiles"]["kd"]["source_hash"] = "b"*64
    with pytest.raises(ValueError, match="source"):
        verify_resource(resource)


def test_resource_refuses_unmeasured_capacity_and_unsafe_aggregate_memory(tmp_path):
    resource = synthetic_resource(tmp_path)
    resource["profiles"]["kd"]["cap"] = 2
    with pytest.raises(ValueError, match="selected-n"):
        verify_resource(resource)
    resource = synthetic_resource(tmp_path/"second")
    resource["profiles"]["kd"]["peak_gpu_bytes"] = 48*1024**3
    with pytest.raises(ValueError, match="aggregate GPU reserve"):
        verify_resource(resource)


def test_host_snapshot_uses_container_cpu_and_memory_limits(tmp_path, monkeypatch):
    import psutil
    from types import SimpleNamespace
    from configs.tools import a2_controller as ctl
    original = Path.read_text
    values = {"/sys/fs/cgroup/cpu.max": "3200000 100000",
              "/sys/fs/cgroup/memory.max": str(256*1024**3),
              "/sys/fs/cgroup/memory.current": str(96*1024**3)}
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: values[str(path)] if str(path) in values else original(path, *a, **kw))
    monkeypatch.setattr(ctl.os, "sched_getaffinity", lambda _: set(range(128)))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=1024*1024**3, available=800*1024**3))
    row = ctl.host_snapshot(tmp_path)
    assert row["cpu_count"] == row["cpu_quota_cores"] == 32
    assert row["cpu_visible_count"] == 128
    assert row["total"] == 256*1024**3
    assert row["available"] == 160*1024**3
