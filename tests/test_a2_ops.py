"""A2 process/controller fixtures: CPU execution, simulated physical GPUs only."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tools import a2_controller as ctl
from configs.tools import a2_resource as resource


def envelope():
    return {"schema": ctl.RESOURCE_SCHEMA, "frozen": False, "gpus": list(range(7)),
            "profiles": {"train": {"kind": "train", "cap": 3, "peak_gpu_bytes": 1024**3, "peak_rss_bytes": 1024**2, "cpu_threads": 1},
                         "eval": {"kind": "eval", "cap": 2, "peak_gpu_bytes": 1024**3, "peak_rss_bytes": 1024**2, "cpu_threads": 1},
                         "cpu": {"kind": "cpu", "cap": 2, "peak_gpu_bytes": 0, "peak_rss_bytes": 1024**2, "cpu_threads": 1}},
            "gpu_reserve_fraction": .1, "gpu_reserve_bytes": 2 * 1024**3,
            "host_reserve_fraction": .2, "cpu_threads_budget": 64, "disk_min_free_bytes": 1,
            "mixed_combinations": [{"train": 1, "eval": 1}], "compatible_load_classes": [["eval", "train"]]}


def job(tmp_path, name="one", kind="cpu"):
    return {"id": name, "kind": kind, "argv": [sys.executable, "-c", "pass"],
            "verify_argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path),
            "output_dir": str(tmp_path / name), "load_class": kind,
            "source_files": {str(Path(__file__).resolve()): ctl.file_hash(__file__)},
            "inputs": {}, "dependencies": [], "target_updates": 5180,
            "target_rgb_windows": 1326080, "heartbeat_sec": .05}


def plan(jobs, purpose="fixture"):
    return {"schema": ctl.SCHEMA, "a2_version": "anysole.a2.v1", "frozen": True,
            "purpose": purpose, "jobs": jobs}


def grant(p, r):
    return {"schema": "anysole.a2.controller-grant.v1", "authorized_by": "root-a2",
            "plan_sha256": ctl.digest(p), "resource_sha256": ctl.digest(r), "purpose": p["purpose"],
            "expires_unix": time.time() + 3600, "allow_cuda": True}


def machine():
    return {"total": 1024**4, "available": 1024**4, "swap_used": 0,
            "disk_free": 1024**4, "cpu_count": 64, "load1": 0}


def devices():
    return {g: {"total": 48 * 1024**3, "free": 48 * 1024**3,
                "used": 0, "utilization": 0, "pids": []} for g in range(7)}


def test_no_legacy_or_unauthorized_admission(tmp_path):
    p, r = plan([job(tmp_path, kind="train")], "measurement"), envelope()
    with pytest.raises(ValueError, match="root-a2"):
        ctl.validate_bundle(p, r)
    with pytest.raises(ValueError, match="fixtures"):
        ctl.validate_bundle(p, r, fixture=True)
    bad = grant(p, r)
    bad["plan_sha256"] = "old15jobreceipt"
    with pytest.raises(ValueError, match="exact plan"):
        ctl.validate_bundle(p, r, bad)
    r["schema"] = "anysole.resource-candidate.v1"
    with pytest.raises(ValueError, match="A2 resources"):
        ctl.validate_bundle(p, r, bad)


def test_source_drift_and_dependency_cycle(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("first")
    j = job(tmp_path)
    j["source_files"] = {str(source): ctl.file_hash(source)}
    source.write_text("second")
    with pytest.raises(ValueError, match="drift"):
        ctl.validate_job(j)
    a, b = job(tmp_path, "a"), job(tmp_path, "b")
    a["dependencies"], b["dependencies"] = ["b"], ["a"]
    with pytest.raises(ValueError, match="cycle"):
        ctl.validate_bundle(plan([a, b]), envelope(), fixture=True)


def test_resources_mixed_external_and_boundaries(tmp_path):
    j, r, host, gpu = job(tmp_path, kind="train"), envelope(), machine(), devices()
    assert ctl.admission(j, 0, [], r, host, gpu) is None
    gpu[0]["pids"] = [123456]
    assert ctl.admission(j, 0, [], r, host, gpu) == "external_gpu_process"
    assert ctl.admission(j, 0, [], r, host, gpu, owned_pids={123456}) is None
    gpu[0]["pids"] = []
    active = [{"kind": "eval", "load_class": "eval", "gpu": 0}]
    assert ctl.admission(j, 0, active, r, host, gpu) is None
    active.append({"kind": "train", "load_class": "train", "gpu": 0})
    assert ctl.admission(j, 0, active, r, host, gpu) == "unmeasured_mixed_combination"
    host["available"] = .2 * host["total"]
    assert ctl.admission(j, 0, [], r, host, gpu) == "host_memory_reserve"
    host = machine()
    host["swap_growth_bytes"] = 128 * 1024**2
    assert ctl.admission(j, 0, [], r, host, gpu) == "swap_growth"
    host = machine()
    gpu[0]["free"] = 2 * 1024**3
    assert ctl.admission(j, 0, [], r, host, gpu) == "gpu_memory_reserve"


def test_seven_card_backlog_fairness_priority_and_refill(tmp_path, monkeypatch):
    jobs = [job(tmp_path, f"j{i}", "train") for i in range(24)]
    jobs[14]["a2_run_id"] = "B-CAM3"
    p, r = plan(jobs, "measurement"), envelope()
    c = ctl.Controller(p, r, tmp_path / "state", grant=grant(p, r))
    launches = []
    def fake_launch(j, gpu):
        record = c.state["jobs"][j["id"]]
        record.update(state="running", attempts=[{"gpu": gpu, "pid": -1, "starttime": "fixture", "directory": str(tmp_path), "started_unix": time.time()}])
        launches.append((j["id"], gpu))
    monkeypatch.setattr(c, "launch", fake_launch)
    monkeypatch.setattr(c, "reconcile", lambda: None)
    c.tick(host=machine(), devices=devices())
    assert launches[0][0] == "j14"
    assert len(launches) == 21  # exact cap3, seven cards; three genuinely queued.
    assert [sum(gpu == g for _, gpu in launches) for g in range(7)] == [3] * 7
    for name, gpu in launches[:3]:
        c.state["jobs"][name]["state"] = "completed"
    before = len(launches)
    c.tick(host=machine(), devices=devices())
    assert len(launches) == before + 3  # every freed slot refilled on next tick.
    assert all(sum(a["gpu"] == gpu for a in c.active()) == 3 for gpu in range(7))
    # A 15-run panel balances to3+2+2+2+2+2+2, rather than concentrating3s.
    jobs15 = jobs[:15]
    c2 = ctl.Controller(plan(jobs15, "measurement"), r, tmp_path / "other", grant=grant(plan(jobs15, "measurement"), r))
    placed = []
    def place(j, gpu):
        c2.state["jobs"][j["id"]].update(state="running", attempts=[{"gpu": gpu, "pid": -1, "starttime": "fixture", "directory": str(tmp_path), "started_unix": time.time()}])
        placed.append(gpu)
    monkeypatch.setattr(c2, "launch", place)
    monkeypatch.setattr(c2, "reconcile", lambda: None)
    c2.tick(host=machine(), devices=devices())
    assert sorted(placed.count(g) for g in range(7)) == [2] * 6 + [3]
    assert len(ctl.read_json(c2.root / "status.json")["cards"]) == 7


def test_heartbeat_gaps_wrong_pid_and_staleness(tmp_path):
    attempt = {"directory": str(tmp_path), "pid": 123, "starttime": "abc", "started_unix": 0}
    ctl.atomic_json(tmp_path / "heartbeat.json", {"pid": 123, "starttime": "abc", "unix_time": 100})
    assert ctl.heartbeat_status(attempt, 100, timeout=10) is None
    (tmp_path / "heartbeat.json").unlink()
    assert ctl.heartbeat_status(attempt, 105, timeout=10) is None
    ctl.atomic_json(tmp_path / "heartbeat.json", {"pid": 124, "starttime": "abc", "unix_time": 112})
    assert ctl.heartbeat_status(attempt, 112, timeout=10) == "heartbeat_stale"


def test_immutable_checkpoint_drift(tmp_path):
    source, dest = tmp_path / "best.pt", tmp_path / "eval" / "frozen.pt"
    source.write_bytes(b"complete first checkpoint")
    expected = ctl.file_hash(source)
    ctl.immutable_checkpoint(source, expected, dest)
    source.write_bytes(b"new checkpoint published atomically")
    assert dest.read_bytes() == b"complete first checkpoint"
    with pytest.raises(ValueError, match="published complete"):
        ctl.immutable_checkpoint(source, expected, dest)


def test_bounded_oom_only_recovery(tmp_path):
    j = job(tmp_path, kind="train")
    j.update(auto_recover_oom=True, resume_argv=j["argv"])
    Path(j["output_dir"]).mkdir()
    (Path(j["output_dir"]) / "last.pt").write_bytes(b"fixture checkpoint")
    p, r = plan([j], "measurement"), envelope()
    c = ctl.Controller(p, r, tmp_path / "state", grant=grant(p, r))
    record = c.state["jobs"][j["id"]]
    attempt = {"number": 0, "gpu": 0}
    record["attempts"] = [attempt]
    c.fail(j, record, attempt, "cuda_oom")
    assert record["state"] == "pending" and c.state["cap_overrides"]["0"] == 2
    again = {"number": 1, "gpu": 0}
    record["attempts"].append(again)
    c.fail(j, record, again, "cuda_oom")
    assert record["state"] == "failed"  # repeated same failure never loops.
    c.fail(j, record, again, "scientific_or_implementation_failure")
    assert record["state"] == "failed"


def test_real_cpu_worker_resume_controller_and_terminal_verifier(tmp_path):
    script = tmp_path / "fixture.py"
    script.write_text("import os,json,sys,time\nfrom pathlib import Path\n"
                      "assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
                      "root=Path(sys.argv[1]);root.mkdir(exist_ok=True)\n"
                      "time.sleep(.15)\n(root/'terminal.json').write_text(json.dumps({'complete':True}))\n")
    jobs = [job(tmp_path, f"j{i}") for i in range(4)]
    for j in jobs:
        j["argv"] = [sys.executable, str(script), j["output_dir"]]
        j["source_files"][str(script)] = ctl.file_hash(script)
        j["verify_argv"] = [sys.executable, "-c", "import json,sys;from pathlib import Path;assert json.loads(Path(sys.argv[1]).read_text())['complete']", str(Path(j["output_dir"]) / "terminal.json")]
    jobs[-1]["dependencies"] = ["j0"]
    p, r = plan(jobs), envelope()
    c = ctl.Controller(p, r, tmp_path / "state", fixture=True)
    c.tick()
    assert len(c.active()) == 2
    # Simulate main/controller restart while wrappers continue independently.
    c2 = ctl.Controller(p, r, tmp_path / "state", fixture=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        # Reap children through original owner, but resumed controller reads results.
        for child in c.processes.values():
            child.poll()
        c2.tick()
        status = ctl.read_json(c2.root / "status.json")
        if status["terminal"]:
            break
        time.sleep(.05)
    assert status["success"]
    assert all(len(record["attempts"]) == 1 for record in status["jobs"].values())
    assert all(record["attempts"][0]["result"]["verify_exit_code"] == 0 for record in status["jobs"].values())
    assert not any(record["attempts"][0]["gpu"] is not None for record in status["jobs"].values())


def test_exit_zero_never_substitutes_completion(tmp_path):
    j = job(tmp_path)
    j["verify_argv"] = [sys.executable, "-c", "raise SystemExit(7)"]
    c = ctl.Controller(plan([j]), envelope(), tmp_path / "state", fixture=True)
    for _ in range(100):
        c.tick()
        status = ctl.read_json(c.root / "status.json")
        if status["terminal"]:
            break
        time.sleep(.03)
    assert not status["success"] and status["counts"]["failed"] == 1
    assert status["jobs"][j["id"]]["attempts"][0]["reason"] == "terminal_verification_failed"


def test_cuda_lease_absence_and_candidate_start():
    with pytest.raises(ValueError, match="lease"):
        ctl.require_worker_lease(kind="eval")
    assert resource.suggest_next([]) == {"action": "measure", "n": 1, "reason": "safe first representative cohort"}
    with pytest.raises(ValueError, match="fresh A2"):
        resource.validate_cohort({"schema": "anysole.resource-candidate.v1"})


def test_finite_selection_memory_gain_and_n4_gate(monkeypatch):
    # Synthetic receipt arithmetic only; real receipt validation tested separately.
    monkeypatch.setattr(resource, "validate_cohort", lambda row: row)
    def row(n, speed, peak=2 * 1024**3, utilization=50):
        return {"stage": "single", "n": n, "successful_rgb_windows_per_second": speed,
                "peak_gpu_bytes": peak, "gpu_total_bytes": 48 * 1024**3,
                "host_total_bytes": 1024**4, "minimum_host_available_bytes": 900 * 1024**3,
                "worker_peak_rss_bytes": n * 1024**3, "mean_gpu_utilization": utilization}
    assert resource.suggest_next([row(1, 10)])["n"] == 2
    assert resource.suggest_next([row(1, 10), row(2, 10.4)])["selected_n"] == 1
    assert resource.suggest_next([row(1, 10), row(2, 20), row(3, 30, utilization=95)])["action"] == "confirm_seven"
    assert resource.suggest_next([row(1, 10), row(2, 20), row(3, 30)])["n"] == 4
    assert resource.suggest_next([row(1, 10, peak=40 * 1024**3)])["selected_n"] == 1
    with pytest.raises(ValueError, match="repeat"):
        resource.suggest_next([row(1, 10), row(1, 11)])


def test_launch_reservation_recovery_and_dependency_failure(tmp_path, monkeypatch):
    j = job(tmp_path)
    c = ctl.Controller(plan([j]), envelope(), tmp_path / "state", fixture=True)
    directory = tmp_path / "attempt"
    directory.mkdir()
    ctl.atomic_json(directory / "spec.json", {"fixture": True})
    owner = {"pid": os.getpid(), "pgid": os.getpgrp(), "starttime": ctl.starttime(os.getpid()),
             "spec_sha256": ctl.file_hash(directory / "spec.json")}
    ctl.atomic_json(directory / "owner.json", owner)
    c.state["jobs"][j["id"]].update(state="launching", attempts=[{"number": 0, "gpu": None,
        "directory": str(directory), "started_unix": time.time()}])
    c.reconcile()
    assert c.state["jobs"][j["id"]]["state"] == "running"
    assert c.state["jobs"][j["id"]]["attempts"][0]["pid"] == os.getpid()
    # A failed parent turns a dependent job blocked, never successful/ownerless.
    a, b = job(tmp_path, "a"), job(tmp_path, "b")
    b["dependencies"] = ["a"]
    c = ctl.Controller(plan([a, b]), envelope(), tmp_path / "dep", fixture=True)
    c.state["jobs"]["a"].update(state="failed", reason="fixture")
    c.save()
    assert c.state["jobs"]["b"]["state"] == "blocked"
    assert ctl.read_json(c.root / "status.json")["terminal"]


def test_eval_checkpoint_must_be_immutable(tmp_path):
    j = job(tmp_path, kind="eval")
    cp = tmp_path / "best.pt"
    cp.write_bytes(b"checkpoint fixture")
    j["checkpoint"] = {"path": str(cp), "sha256": ctl.file_hash(cp)}
    with pytest.raises(ValueError, match="read-only"):
        ctl.validate_job(j)
    cp.chmod(0o444)
    ctl.validate_job(j)


def test_single_card_capacity_measurement_pinning(tmp_path, monkeypatch):
    jobs = [job(tmp_path, f"j{i}", "train") for i in range(3)]
    for j in jobs:
        j["allowed_gpus"] = [4]
    p, r = plan(jobs, "measurement"), envelope()
    c = ctl.Controller(p, r, tmp_path / "state", grant=grant(p, r))
    placements = []
    def launch(j, gpu):
        c.state["jobs"][j["id"]].update(state="running", attempts=[{"gpu": gpu, "pid": -1,
            "starttime": "fixture", "directory": str(tmp_path), "started_unix": time.time()}])
        placements.append(gpu)
    monkeypatch.setattr(c, "launch", launch)
    monkeypatch.setattr(c, "reconcile", lambda: None)
    c.tick(host=machine(), devices=devices())
    assert placements == [4, 4, 4]


def test_inherited_lease_checks_identity_and_checkpoint(tmp_path, monkeypatch):
    cp = tmp_path / "immutable.pt"
    cp.write_bytes(b"fixture")
    spec = {"schema": "anysole.a2.lease.v1", "gpu": 3,
            "job": {"kind": "eval", "checkpoint": {"path": str(cp), "sha256": ctl.file_hash(cp)}}}
    path = tmp_path / "spec.json"
    ctl.atomic_json(path, spec)
    for key, value in {"A2_LEASE_SPEC": str(path), "A2_LEASE_SHA256": ctl.file_hash(path),
                       "A2_LEASE_PID": str(os.getpid()), "A2_LEASE_STARTTIME": ctl.starttime(os.getpid()),
                       "CUDA_VISIBLE_DEVICES": "3"}.items():
        monkeypatch.setenv(key, value)
    assert ctl.require_worker_lease(kind="eval", checkpoint=cp)["gpu"] == 3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="physical GPU"):
        ctl.require_worker_lease(kind="eval", checkpoint=cp)


def test_checkpoint_templates_deduplicate_and_training_terminal(tmp_path):
    parent = job(tmp_path, "trainer", "train")
    output = Path(parent["output_dir"])
    output.mkdir()
    for selection in ("best", "last"):
        (output / f"{selection}.pt").write_bytes(b"same complete checkpoint fixture")
    checkpoint_hash = ctl.file_hash(output / "last.pt")
    ctl.atomic_json(output / "run.json", {"last_checkpoint_hash": checkpoint_hash,
        "best_checkpoint_hash": checkpoint_hash, "successful_updates": 5180, "selection": {"best_step": 5180}})
    evaluation = job(tmp_path, "unused-template-id", "eval")
    evaluation["output_dir"] = str(tmp_path / "eval" / "{eval_id}")
    evaluation["argv"] = [sys.executable, "-c", "pass", "{checkpoint}", "{checkpoint_sha256}"]
    contract = {"axis": "native_rgb", "frames": 36516, "protocol": "fixture"}
    parent["evaluation_templates"] = [{"selection": selection, "source_path": str(output / f"{selection}.pt"),
        "published_hash_field": selection + "_checkpoint_hash", "inference_contract": contract,
        "inference_contract_sha256": ctl.digest(contract), "job": evaluation, "dispatch": True}
        for selection in ("best", "last")]
    p, r = plan([parent], "measurement"), envelope()
    c = ctl.Controller(p, r, tmp_path / "state", grant=grant(p, r))
    c.state["jobs"][parent["id"]]["state"] = "completed"
    c.publish_checkpoints()
    c.save()
    assert len(c.state["dynamic_jobs"]) == 1
    publication = next(iter(c.state["publications"].values()))
    assert len(publication["aliases"]) == 2
    assert publication["checkpoint_sha256"] == checkpoint_hash
    assert Path(publication["checkpoint_path"]).stat().st_mode & 0o222 == 0
    assert "{checkpoint}" not in publication["job"]["argv"]
    status = ctl.read_json(c.root / "status.json")
    assert status["train_terminal"] and status["train_success"]
    assert not status["terminal"]  # pending evaluation never extends training goal.
    # A restarted controller sees exactly the same immutable evaluation queue.
    resumed = ctl.Controller(p, r, tmp_path / "state", grant=grant(p, r))
    resumed.publish_checkpoints()
    assert len(resumed.all_jobs()) == 2
    resumed.state["jobs"][publication["job"]["id"]]["state"] = "completed"
    resumed.save()
    assert ctl.read_json(resumed.root / "status.json")["success"]


def test_eval_receipt_uses_numeric_eval_coverage_not_training_budget(tmp_path):
    config = tmp_path / "config.json"
    ctl.atomic_json(config, {"fixture": "evaluation configuration"})
    attempt = tmp_path / "result.json"
    ctl.atomic_json(attempt, {"schema": "anysole.a2.attempt.v1", "success": True, "exit_code": 0, "verify_exit_code": 0})
    evaluation = tmp_path / "evaluation.json"
    numeric = {"complete_representative_recording": True, "finite": True,
               "expected_frame_count": 1999, "actual_frame_count": 1999, "finite_frame_count": 1999,
               "new_student_inference": True}
    ctl.atomic_json(evaluation, numeric)
    report = {"schema": resource.COHORT_SCHEMA, "purpose": "measurement", "stage": "eval", "n": 1,
              "wall_seconds": 1, "peak_gpu_bytes": 1024**3, "gpu_total_bytes": 48 * 1024**3,
              "host_total_bytes": 1024**4, "minimum_host_available_bytes": 900 * 1024**3,
              "disk_min_free_bytes": 100 * 1024**3, "expected_worker_count": 1,
              "source_files": {str(Path(__file__).resolve()): ctl.file_hash(__file__)},
              "contracts": {"native": ctl.digest({"fixture": True})}, "evaluation_frames_per_second": 1999,
              "workers": [{"kind": "eval", "gpu": 0, "attempt_result": resource.reference(attempt),
                           "config": resource.reference(config), "config_sha256": ctl.file_hash(config),
                           "evaluation": resource.reference(evaluation)}]}
    assert resource.validate_cohort(report)  # no invented training updates/RGB throughput.
    report["n"] = 3
    with pytest.raises(ValueError, match="actual worker count"):
        resource.validate_cohort(report)
    report["n"] = 1
    numeric["finite_frame_count"] = 1998
    ctl.atomic_json(evaluation, numeric)
    report["workers"][0]["evaluation"] = resource.reference(evaluation)
    with pytest.raises(ValueError, match="numerical"):
        resource.validate_cohort(report)


def _frozen_gate_fixture(monkeypatch):
    """Synthetic IO isolation, not GPU measurement or an admission receipt."""
    monkeypatch.setattr(resource, "verify_reference", lambda row: row)
    monkeypatch.setattr(resource, "validate_cohort", lambda row: row)
    def profile(name, cap=3, kind="train"):
        return {"kind": kind, "cap": cap, "cpu_threads": 1, "contract_sha256": name + "-current"}
    def report(name, n, stage="single", kind="train", memory=2, speed=10):
        cards = range(7) if stage == "seven" else [0]
        return {"stage": stage, "n": n, "contracts": {name: name + "-current"},
                "workers": [{"gpu": gpu, "load_class": name, "kind": kind} for gpu in cards for _ in range(n)],
                "successful_rgb_windows_per_second": speed * n, "evaluation_frames_per_second": speed * n,
                "peak_gpu_bytes": memory * n * 1024**3, "gpu_total_bytes": 48 * 1024**3,
                "host_total_bytes": 1024**4, "minimum_host_available_bytes": 900 * 1024**3,
                "worker_peak_rss_bytes": memory * n * 1024**2, "mean_gpu_utilization": 95,
                "source_files": {"fixture-common-source": "same-current-source"}}
    frozen = {"schema": resource.RESOURCE_SCHEMA, "frozen": True,
              "profiles": {"route": profile("route")},
              "evidence": {"route": [report("route", n) for n in (1, 2, 3)],
                           "seven": [report("route", 3, "seven")]}, "mixed_combinations": []}
    return frozen, profile, report


def test_seven_receipt_rejects_wrong_cap_and_unrelated_current_contract(monkeypatch):
    frozen, _, report = _frozen_gate_fixture(monkeypatch)
    assert resource.verify_frozen_resource(frozen)
    wrong_n = copy.deepcopy(frozen)
    wrong_n["evidence"]["seven"] = [report("route", 1, "seven")]
    with pytest.raises(ValueError, match="current selected capacity"):
        resource.verify_frozen_resource(wrong_n)
    unrelated = copy.deepcopy(frozen)
    unrelated["evidence"]["seven"] = [report("old-unrelated", 3, "seven")]
    with pytest.raises(ValueError, match="current selected profiles"):
        resource.verify_frozen_resource(unrelated)
    stale_contract = copy.deepcopy(frozen)
    stale_contract["evidence"]["seven"][0]["contracts"]["route"] = "old-source-contract"
    with pytest.raises(ValueError, match="current selected profiles"):
        resource.verify_frozen_resource(stale_contract)


def test_seven_inheritance_is_explicit_measured_and_conservative(monkeypatch):
    frozen, profile, report = _frozen_gate_fixture(monkeypatch)
    frozen["profiles"]["old_all"] = profile("old_all")
    frozen["evidence"]["old_all"] = [report("old_all", n, memory=1, speed=12) for n in (1, 2, 3)]
    with pytest.raises(ValueError, match="explicit seven-card inheritance"):
        resource.verify_frozen_resource(frozen)
    proof = {"schema": "anysole.a2.seven-inheritance.v1", "representative_profile": "route",
             "profile_contract_sha256": "old_all-current", "representative_contract_sha256": "route-current",
             "reason": "same measured host source; lower selected-cap memory/CPU and faster measured throughput",
             "seven_receipt": frozen["evidence"]["seven"][0],
             "candidate_receipt": frozen["evidence"]["old_all"][-1],
             "representative_receipt": frozen["evidence"]["route"][-1],
             "throughput_argument": {"reason": "different class work; shared host bounds are measured separately",
                "candidate_unit": "rgb_windows/s", "representative_unit": "rgb_windows/s",
                "candidate_selected_rate": 36, "candidate_n1_rate": 12, "representative_selected_rate": 30}}
    frozen["evidence"]["seven_inheritance"] = {"old_all": proof}
    assert resource.verify_frozen_resource(frozen)
    too_heavy = copy.deepcopy(frozen)
    too_heavy["evidence"]["old_all"][-1]["peak_gpu_bytes"] = 30 * 1024**3
    # Explicitly rebind to that heavier selected-cap receipt, rather than merely
    # testing a stale-proof mismatch.
    too_heavy["evidence"]["seven_inheritance"]["old_all"]["candidate_receipt"] = too_heavy["evidence"]["old_all"][-1]
    with pytest.raises(ValueError, match="exceeds measured GPU envelope"):
        resource.verify_frozen_resource(too_heavy)
    slower = copy.deepcopy(frozen)
    slower["evidence"]["old_all"][-1]["successful_rgb_windows_per_second"] = 29
    slower["evidence"]["seven_inheritance"]["old_all"]["candidate_receipt"] = slower["evidence"]["old_all"][-1]
    slower["evidence"]["seven_inheritance"]["old_all"]["throughput_argument"]["candidate_selected_rate"] = 29
    assert resource.verify_frozen_resource(slower)  # family speed differs; own n1 and envelope still pass.
    fabricated_rate = copy.deepcopy(frozen)
    fabricated_rate["evidence"]["seven_inheritance"]["old_all"]["throughput_argument"]["candidate_selected_rate"] = 999
    with pytest.raises(ValueError, match="differs from actual measurements"):
        resource.verify_frozen_resource(fabricated_rate)


def test_mixed_receipt_binds_every_participating_current_contract(monkeypatch):
    frozen, profile, report = _frozen_gate_fixture(monkeypatch)
    frozen["profiles"] = {"route": profile("route", 1), "native": profile("native", 1, "eval")}
    seven = report("route", 1, "seven")
    eval_seven = report("native", 1, "seven", "eval")
    seven["workers"] += eval_seven["workers"]
    seven["contracts"].update(eval_seven["contracts"])
    seven["n"] = 2
    mixed = report("route", 1, "mixed")
    mixed["workers"] += report("native", 1, "mixed", "eval")["workers"]
    mixed["contracts"]["native"] = "native-current"
    mixed["combination"] = {"train": 1, "eval": 1}
    frozen["evidence"] = {"route": [report("route", 1)], "native": [report("native", 1, "eval", "eval")],
                          "seven": [seven], "mixed": [mixed]}
    frozen["mixed_combinations"] = [{"train": 1, "eval": 1}]
    assert resource.verify_frozen_resource(frozen)
    bad = copy.deepcopy(frozen)
    bad["evidence"]["mixed"][0]["contracts"]["native"] = "native-previous-inference-contract"
    with pytest.raises(ValueError, match="current selected profiles"):
        resource.verify_frozen_resource(bad)


def test_atomic_json_detects_missing_or_corrupt_post_rename_publication(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    original = ctl.os.replace
    def lose_after_rename(source, destination):
        original(source, destination)
        Path(destination).unlink()
    monkeypatch.setattr(ctl.os, "replace", lose_after_rename)
    with pytest.raises(OSError):
        ctl.atomic_json(target, {"complete": True})
    def corrupt_after_rename(source, destination):
        original(source, destination)
        Path(destination).write_text('{"complete":false}\n')
    monkeypatch.setattr(ctl.os, "replace", corrupt_after_rename)
    with pytest.raises(OSError, match="differs after atomic rename"):
        ctl.atomic_json(target, {"complete": True})


def test_controller_refuses_phantom_queue_reset(tmp_path, monkeypatch):
    p, r = plan([job(tmp_path)]), envelope()
    root = tmp_path / "state"
    c = ctl.Controller(p, r, root, fixture=True)
    c.save()
    original = ctl.read_json
    def unavailable(path):
        if Path(path) == root / "a2_queue.json":
            raise FileNotFoundError("synthetic EPC_NFS stat/read inconsistency")
        return original(path)
    monkeypatch.setattr(ctl, "read_json", unavailable)
    with pytest.raises(OSError, match="refusing to reset"):
        ctl.Controller(p, r, root, fixture=True)
