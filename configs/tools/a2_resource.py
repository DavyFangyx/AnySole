#!/usr/bin/env python3
"""Finite A2 capacity protocol and evidence-bound resource selection.

One cohort per n by default: 5 warmup + 20 successful timed updates, real
validation/save/reload. A next n is proposed only from measured memory headroom.
This module observes devices and summarizes receipts; it never launches CUDA.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.tools.a2_controller import (RESOURCE_SCHEMA, atomic_json, digest, file_hash,
    gpu_snapshot, host_snapshot, inside, positive, read_json)

COHORT_SCHEMA = "anysole.a2.cohort.v1"


def reference(path):
    path = inside(path)
    return {"path": str(path), "sha256": file_hash(path)}


def summarize(state_dir, stage, n):
    """Derive a cohort receipt from actual controller/worker/trainer outputs."""
    root = inside(state_dir)
    queue, status = read_json(root / "a2_queue.json"), read_json(root / "status.json")
    if not status.get("terminal") or not status.get("success"):
        raise ValueError("only a terminal successful measurement cohort can be summarized")
    workers, results, smokes, sources, contracts, evaluations = [], [], [], {}, {}, []
    for record in queue["jobs"].values():
        if len(record["attempts"]) != 1:
            raise ValueError("measurement fault/repeat requires an explicit amendment")
        attempt = record["attempts"][0]
        directory = Path(attempt["directory"])
        spec, result = read_json(directory / "spec.json"), read_json(directory / "result.json")
        job = spec["job"]
        contract = job.get("resource_contract_sha256")
        if not contract or not job.get("config_sha256"):
            raise ValueError("actual job configuration/contract hashes required")
        name = job["load_class"]
        if name in contracts and contracts[name] != contract:
            raise ValueError("one load class has conflicting frozen contracts")
        contracts[name] = contract
        for path, sha in job["source_files"].items():
            if path in sources and sources[path] != sha:
                raise ValueError("source conflicts within cohort")
            sources[path] = sha
        row = {"kind": job["kind"], "load_class": name, "gpu": spec["gpu"],
               "config_sha256": job["config_sha256"], "config": reference(job["config_path"]),
               "attempt_result": reference(directory / "result.json")}
        if job["kind"] == "train":
            output = Path(job["output_dir"])
            smoke = read_json(output / "a2_smoke.json")
            row.update(run=reference(output / "run.json"), smoke=reference(output / "a2_smoke.json"),
                       checkpoint=reference(output / "last.pt"))
            smokes.append(smoke)
        elif job["kind"] == "eval":
            row["evaluation"] = reference(job["measurement_evidence"])
            evaluations.append(read_json(job["measurement_evidence"]))
        else:
            raise ValueError("GPU measurement receipts cannot include CPU fixture jobs")
        workers.append(row)
        results.append(result)
    samples = [json.loads(line) for line in (root / "resource_samples.jsonl").read_text().splitlines()]
    if not samples or not any(sample["devices"] for sample in samples):
        raise ValueError("actual GPU resource samples missing")
    devices = [device for sample in samples for device in sample["devices"].values()]
    host = [sample["host"] for sample in samples]
    peaks = [device["used"] for device in devices]
    for gpu in range(7):
        peaks.append(sum(smoke.get("peak_gpu_reserved_bytes", 0) for worker, smoke in
            zip([w for w in workers if w["kind"] == "train"], smokes) if worker["gpu"] == gpu))
    report = {"schema": COHORT_SCHEMA, "purpose": "measurement", "stage": stage, "n": n,
              "a2_version": queue.get("a2_version", "anysole.a2.v1"), "capacity_max_n": queue.get("capacity_max_n", 4),
              "plan_sha256": queue["plan_sha256"], "resource_sha256": queue["resource_sha256"],
              "source_files": sources, "contracts": contracts, "workers": workers,
              "expected_worker_count": len(workers), "controller_state": reference(root / "a2_queue.json"),
              "resource_samples": reference(root / "resource_samples.jsonl"),
              "wall_seconds": max(r["finished_unix"] for r in results) - min(r["started_unix"] for r in results),
              "peak_gpu_bytes": max(peaks), "gpu_total_bytes": min(d["total"] for d in devices),
              "mean_gpu_utilization": statistics.mean(d["utilization"] for d in devices if d["used"] > 0),
              "host_total_bytes": min(h["total"] for h in host),
              "minimum_host_available_bytes": min(h["available"] for h in host),
              "worker_peak_rss_bytes": sum(r["peaks"]["rss_bytes"] for r in results),
              "swap_growth_bytes": max(h["swap_used"] for h in host) - min(h["swap_used"] for h in host),
              "io_blocked": any(h.get("iowait_percent", 0) >= 30 for h in host),
              "disk_min_free_bytes": min(h["disk_free"] for h in host)}
    if smokes:
        report.update(warmup_updates=5, timed_updates=20,
            successful_rgb_windows_per_second=sum(s["timed_rgb_windows"] for s in smokes) /
                (max(s["timed_finished_unix"] for s in smokes) - min(s["timed_started_unix"] for s in smokes)))
    if evaluations:
        report["evaluation_frames_per_second"] = sum(e["native_rgb_frames"] for e in evaluations) / report["wall_seconds"]
    if stage == "mixed":
        per_card = [{kind: sum(w["kind"] == kind and w["gpu"] == gpu for w in workers) for kind in ("train", "eval")}
                    for gpu in set(w["gpu"] for w in workers)]
        if any(row != per_card[0] for row in per_card):
            raise ValueError("mixed receipt requires one explicit representative combination")
        report["combination"] = per_card[0]
    validate_cohort(report)
    return report


def verify_reference(reference):
    path = inside(reference["path"])
    if file_hash(path) != reference.get("sha256"):
        raise ValueError(f"resource evidence drift: {path}")
    return read_json(path)


def validate_cohort(report):
    if report.get("schema") != COHORT_SCHEMA or report.get("purpose") != "measurement":
        raise ValueError("fresh A2 measurement cohort required")
    if report.get("stage") not in ("single", "seven", "mixed", "eval"):
        raise ValueError("unknown cohort stage")
    positive(report.get("n"), "cohort n")
    a3 = report.get("a2_version") == "anysole.a3.v1"
    maximum = report.get("capacity_max_n", 4) if a3 else 4
    if type(maximum) is not int or maximum < 1 or report["n"] > maximum:
        raise ValueError("cohort exceeds frozen finite protocol maximum n")
    for key in ("wall_seconds", "peak_gpu_bytes", "gpu_total_bytes", "host_total_bytes", "minimum_host_available_bytes", "disk_min_free_bytes"):
        positive(report.get(key), key)
    if report["peak_gpu_bytes"] > report["gpu_total_bytes"] - max(2 * 1024**3, .1 * report["gpu_total_bytes"]):
        raise ValueError("cohort exceeded GPU reserve")
    if report["minimum_host_available_bytes"] < .2 * report["host_total_bytes"]:
        raise ValueError("cohort exceeded host reserve")
    if report.get("swap_growth_bytes", 0) > 64 * 1024**2 or report.get("io_blocked"):
        raise ValueError("unstable swap or IO")
    if report.get("disk_min_free_bytes", 0) < report.get("disk_required_free_bytes", 10 * 1024**3):
        raise ValueError("disk below low watermark")
    workers = report.get("workers", [])
    if not workers or len(workers) != report.get("expected_worker_count"):
        raise ValueError("incomplete worker coverage")
    cards = {w["gpu"] for w in workers}
    if not cards <= set(range(7)):
        raise ValueError("measurement worker physical GPU invalid")
    if report["stage"] in ("single", "eval") and (len(cards) != 1 or len(workers) != report["n"]):
        raise ValueError("single-card measurement n differs from actual worker count")
    if report["stage"] == "seven" and (cards != set(range(7)) or any(sum(w["gpu"] == gpu for w in workers) != report["n"] for gpu in range(7))):
        raise ValueError("seven-card confirmation actual worker counts differ from n")
    if any(w["kind"] == "train" for w in workers) and (report.get("warmup_updates") != 5 or report.get("timed_updates") != 20):
        raise ValueError("training protocol is exactly 5 warmup + 20 timed successful updates")
    if not report.get("source_files") or any(file_hash(inside(path)) != sha for path, sha in report["source_files"].items()):
        raise ValueError("cohort implementation source evidence missing or drifted")
    if not report.get("contracts"):
        raise ValueError("cohort must bind actual load-class configuration contracts")
    for worker in workers:
        if worker.get("config", {}).get("sha256") != worker.get("config_sha256"):
            raise ValueError("worker frozen configuration identity missing")
        if file_hash(inside(worker["config"]["path"])) != worker["config_sha256"]:
            raise ValueError("worker frozen configuration drift")
        result = verify_reference(worker["attempt_result"])
        if result.get("schema") != "anysole.a2.attempt.v1" or not result.get("success") or result.get("exit_code") != 0 or result.get("verify_exit_code") != 0:
            raise ValueError("cohort worker failed")
        if worker["kind"] == "train":
            import yaml
            config = yaml.safe_load(Path(worker["config"]["path"]).read_text())
            if a3 and config.get("a2_version") != "anysole.a3.v1":
                raise ValueError("A3 capacity cannot use an A2 training configuration")
            batch_rgb = config.get("effective_batch_rgb_windows", 256) if a3 else 256
            if type(batch_rgb) is not int or batch_rgb <= 0:
                raise ValueError("invalid frozen training RGB batch")
            run = verify_reference(worker["run"])
            smoke = verify_reference(worker["smoke"])
            if run.get("successful_updates") != 25 or run.get("rgb_windows") != 25 * batch_rgb:
                raise ValueError("cohort successful training budget incomplete")
            if run.get("last_validation", {}).get("val/pressure_cells", 0) <= 0 or run.get("last_validation", {}).get("val/motion_frames", 0) <= 0:
                raise ValueError("complete validation missing")
            if not all(smoke.get(k) is True for k in ("checkpoint_reloaded", "state_complete", "full_validation", "maximum_effective_batch")):
                raise ValueError("save/reload/validation/maximum batch evidence missing")
            if smoke.get("validated_motion_frames", 0) <= 0 or smoke.get("validated_pressure_cells", 0) <= 0 or smoke.get("minimum_actual_batch_rgb") != batch_rgb or smoke.get("maximum_actual_batch_rgb") != batch_rgb:
                raise ValueError("numerical coverage/batch evidence missing")
            if file_hash(inside(worker["checkpoint"]["path"])) != worker["checkpoint"].get("sha256") or smoke.get("checkpoint_sha256") != worker["checkpoint"]["sha256"]:
                raise ValueError("reload checkpoint identity mismatch")
            if smoke.get("warmup_updates") != 5 or smoke.get("timed_updates") != 20:
                raise ValueError("timed update coverage mismatch")
            positive(smoke.get("timed_seconds"), "timed_seconds")
        elif worker["kind"] == "eval":
            evidence = verify_reference(worker["evaluation"])
            if not evidence.get("complete_representative_recording") or not evidence.get("finite"):
                raise ValueError("complete long recording evaluation missing")
            if evidence.get("expected_frame_count", 0) <= 0 or evidence.get("actual_frame_count", evidence.get("frame_count")) != evidence["expected_frame_count"] or evidence.get("finite_frame_count") != evidence["expected_frame_count"]:
                raise ValueError("numerical long-recording evaluation coverage missing")
            if evidence.get("new_student_inference") is not True:
                raise ValueError("resource measurement cannot reuse old student predictions")
        else:
            raise ValueError("unknown cohort worker kind")
    if report["stage"] == "seven" and set(w["gpu"] for w in workers) != set(range(7)):
        raise ValueError("seven-card confirmation must cover all physical cards")
    if report["stage"] == "mixed" and {w["kind"] for w in workers} != {"train", "eval"}:
        raise ValueError("mixed receipt must contain actual training and evaluation")
    if report["stage"] == "mixed":
        combination = report.get("combination")
        if not combination or any(sum(w["gpu"] == gpu and w["kind"] == kind for w in workers) != combination[kind] for gpu in cards for kind in ("train", "eval")):
            raise ValueError("mixed actual worker counts differ from frozen combination")
    if any(w["kind"] == "train" for w in workers):
        positive(report.get("successful_rgb_windows_per_second", 0), "whole cohort RGB throughput")
    if any(w["kind"] == "eval" for w in workers):
        positive(report.get("evaluation_frames_per_second", 0), "actual evaluation throughput")
    return report


def suggest_next(reports):
    """Finite, memory-justified protocol; no accidental repetitions/grid sweep."""
    if not reports:
        return {"action": "measure", "n": 1, "reason": "safe first representative cohort"}
    for report in reports:
        validate_cohort(report)
    singles = sorted([r for r in reports if r["stage"] in ("single", "eval")], key=lambda r: r["n"])
    if not singles:
        raise ValueError("capacity selection requires actual single-card train/eval measurements")
    if len({r["n"] for r in singles}) != len(singles):
        raise ValueError("repeat requires a separately recorded noise/failure amendment")
    if [r["n"] for r in singles] != list(range(1, len(singles) + 1)):
        raise ValueError("measure n in order from one")
    last = singles[-1]
    is_eval = last["stage"] == "eval"
    rate = "evaluation_frames_per_second" if is_eval else "successful_rgb_windows_per_second"
    confirm = "confirm_mixed" if is_eval else "confirm_seven"
    selected = last
    a3 = last.get("a2_version") == "anysole.a3.v1"
    if a3:
        if any(r.get("a2_version") != "anysole.a3.v1" or r.get("capacity_max_n") != last.get("capacity_max_n") for r in singles):
            raise ValueError("capacity protocol changed between cohorts")
        # A flat n2 result cannot establish that n3..n6 are slower. Inspect all
        # safely fitting task-bounded occupancies; favor higher n within 1% noise.
        best_rate = max(r[rate] for r in singles)
        selected = max((r for r in singles if r[rate] >= .99 * best_rate), key=lambda r: r["n"])
    elif len(singles) > 1:
        gain = last[rate] / singles[-2][rate] - 1
        if gain < .05:
            selected = singles[-2]
            return {"action": confirm, "selected_n": selected["n"], "reason": "throughput gain below 5 percent", "gain": gain}
    n = last["n"] + 1
    per_worker = last["peak_gpu_bytes"] / last["n"]
    reserve = max(2 * 1024**3, .1 * last["gpu_total_bytes"])
    host_per_worker = last.get("worker_peak_rss_bytes", last["host_total_bytes"] - last["minimum_host_available_bytes"]) / last["n"]
    memory_ok = per_worker * n <= last["gpu_total_bytes"] - reserve and host_per_worker * n < .8 * last["host_total_bytes"]
    if a3:
        available_without_cohort = last["minimum_host_available_bytes"] + last.get("worker_peak_rss_bytes", 0)
        memory_ok = memory_ok and host_per_worker * n * 7 < available_without_cohort - .2 * last["host_total_bytes"]
        if n <= last["capacity_max_n"] and memory_ok:
            return {"action": "measure", "n": n, "reason": "A3 task-bounded measured memory permits next occupancy"}
        return {"action": confirm, "selected_n": selected["n"], "reason": "A3 measured throughput maximum within 1 percent; finite task/memory boundary"}
    # n4 only if n3 actually remained underutilized, rather than merely fitting RAM.
    underutilized = n < 4 or last.get("mean_gpu_utilization", 100) < 85
    if n <= 4 and memory_ok and underutilized:
        return {"action": "measure", "n": n, "reason": "measured peak predicts reserve and prior throughput gain passes"}
    return {"action": confirm, "selected_n": selected["n"], "reason": "finite memory/utilization boundary"}


def verify_frozen_resource(resource):
    if resource.get("schema") != RESOURCE_SCHEMA or resource.get("frozen") is not True:
        raise ValueError("frozen A2 resource required")
    evidence = resource.get("evidence", {})
    profiles = {name: p for name, p in resource["profiles"].items() if p["kind"] != "cpu"}
    def current_contracts(report):
        contracts = report.get("contracts", {})
        if not contracts or any(name not in profiles or sha != profiles[name].get("contract_sha256") for name, sha in contracts.items()):
            raise ValueError("cohort contracts do not match current selected profiles")
        represented = {worker.get("load_class") for worker in report.get("workers", [])}
        if represented != set(contracts):
            raise ValueError("cohort worker classes differ from its current contracts")
        if any(worker["kind"] != profiles[worker["load_class"]]["kind"] for worker in report["workers"]):
            raise ValueError("cohort worker kind differs from selected profile")
    seven_pairs = [(ref, validate_cohort(verify_reference(ref))) for ref in evidence.get("seven", [])]
    bindings = {name: [] for name in profiles}
    for ref, report in seven_pairs:
        if report.get("stage") != "seven":
            raise ValueError("seven confirmation must be an actual seven-card cohort")
        current_contracts(report)
        for name, profile in profiles.items():
            if name in report["contracts"] and all(sum(w["gpu"] == gpu and w["load_class"] == name for w in report["workers"]) == profile["cap"] for gpu in range(7)):
                bindings[name].append((ref, report))
    if not any(bindings.values()):
        raise ValueError("seven confirmation is not bound to any current selected capacity/contract")
    own_reports = {}
    for name, profile in resource["profiles"].items():
        if profile["kind"] == "cpu":
            continue
        references = evidence.get(name, [])
        if not references:
            raise ValueError(f"fresh load-class evidence missing: {name}")
        reports = [validate_cohort(verify_reference(ref)) for ref in references]
        for report in reports:
            current_contracts(report)
        own_reports[name] = list(zip(references, reports))
        if any(name not in r.get("contracts", {}) or r["contracts"][name] != profile.get("contract_sha256") for r in reports):
            raise ValueError("resource load class mismatch")
        if profile["kind"] == "train":
            singles = [r for r in reports if r["stage"] == "single"]
            if not singles or min(r["n"] for r in singles) != 1:
                raise ValueError("no safe n1 parent")
            if profile["cap"] not in {r["n"] for r in singles}:
                raise ValueError("selected cap was never measured")
            selection = suggest_next(singles)
            if singles[0].get("a2_version") == "anysole.a3.v1" and (selection.get("action") != "confirm_seven" or profile["cap"] != selection.get("selected_n")):
                raise ValueError("A3 cap must equal the completed measured throughput selection")
            if "selected_n" in selection and profile["cap"] > selection["selected_n"]:
                raise ValueError("selected cap exceeds finite throughput/memory decision")
        elif not any(r["stage"] == "eval" and r["n"] == profile["cap"] for r in reports):
            raise ValueError("evaluation cap lacks actual long-recording measurement")
    # One representative seven-card run can cover another measured class only
    # through an explicit source/contract-bound conservative envelope proof.
    for name, profile in profiles.items():
        if bindings[name]:
            continue
        proof = evidence.get("seven_inheritance", {}).get(name, {})
        representative = proof.get("representative_profile")
        if proof.get("schema") != "anysole.a2.seven-inheritance.v1" or not proof.get("reason") or representative not in profiles:
            raise ValueError(f"profile {name} lacks explicit seven-card inheritance")
        parent = profiles[representative]
        if proof.get("profile_contract_sha256") != profile.get("contract_sha256") or proof.get("representative_contract_sha256") != parent.get("contract_sha256"):
            raise ValueError("seven inheritance contract mismatch")
        seven_ref = proof.get("seven_receipt")
        if not any(ref == seven_ref for ref, _ in bindings[representative]):
            raise ValueError("seven inheritance representative is not directly confirmed at selected cap")
        def selected_report(class_name, reference):
            stage = "single" if profiles[class_name]["kind"] == "train" else "eval"
            found = [report for ref, report in own_reports[class_name] if ref == reference and report["stage"] == stage and report["n"] == profiles[class_name]["cap"]]
            if len(found) != 1:
                raise ValueError("seven inheritance must bind actual selected-cap class measurement")
            return found[0]
        candidate = selected_report(name, proof.get("candidate_receipt"))
        reference = selected_report(representative, proof.get("representative_receipt"))
        if candidate["peak_gpu_bytes"] > reference["peak_gpu_bytes"]:
            raise ValueError("seven inheritance candidate exceeds measured GPU envelope")
        if candidate.get("worker_peak_rss_bytes", float("inf")) > reference.get("worker_peak_rss_bytes", 0) or profile["cpu_threads"] * profile["cap"] > parent["cpu_threads"] * parent["cap"]:
            raise ValueError("seven inheritance candidate exceeds measured host envelope")
        if any(candidate["source_files"][path] != sha for path, sha in reference["source_files"].items() if path in candidate["source_files"]):
            raise ValueError("seven inheritance source implementations conflict")
        # Family workloads need not have equal speed, and cross-kind rates have
        # different units. Bind the actual numbers and explicit host-envelope
        # reasoning; require the selected class cap to beat its own safe n1.
        rate = "successful_rgb_windows_per_second" if profile["kind"] == "train" else "evaluation_frames_per_second"
        parent_rate = "successful_rgb_windows_per_second" if parent["kind"] == "train" else "evaluation_frames_per_second"
        first = [r for _, r in own_reports[name] if r["n"] == 1 and r["stage"] in ("single", "eval")]
        argument = proof.get("throughput_argument", {})
        expected = {"candidate_selected_rate": candidate[rate], "candidate_n1_rate": first[0][rate] if first else None,
                    "representative_selected_rate": reference[parent_rate]}
        if not first or candidate[rate] < first[0][rate] or not argument.get("reason"):
            raise ValueError("seven inheritance lacks measured own-class throughput argument")
        if argument.get("candidate_unit") != ("rgb_windows/s" if profile["kind"] == "train" else "native_frames/s") or argument.get("representative_unit") != ("rgb_windows/s" if parent["kind"] == "train" else "native_frames/s"):
            raise ValueError("seven inheritance throughput units do not match measured classes")
        if any(isinstance(argument.get(key), bool) or not isinstance(argument.get(key), (int, float)) or not math.isclose(argument[key], value, rel_tol=1e-12, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError("seven inheritance throughput argument differs from actual measurements")
    if resource.get("mixed_combinations"):
        references = evidence.get("mixed", [])
        reports = [validate_cohort(verify_reference(ref)) for ref in references]
        for report in reports:
            current_contracts(report)
            if report.get("stage") != "mixed" or any(sum(w["gpu"] == gpu and w["load_class"] == name for w in report["workers"]) > profile["cap"] for gpu in {w["gpu"] for w in report["workers"]} for name, profile in profiles.items()):
                raise ValueError("mixed receipt exceeds current selected profile capacities")
        for combination in resource["mixed_combinations"]:
            if not any(r["stage"] == "mixed" and r.get("combination") == combination for r in reports):
                raise ValueError("unmeasured mixed combination")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "suggest", "summarize", "verify-cohort", "verify-frozen"))
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--stage", choices=("single", "seven", "mixed", "eval"))
    p.add_argument("--n", type=int)
    args = p.parse_args(argv)
    if args.action == "inspect":
        result = {"schema": "anysole.a2.resource-observation.v1", "unix_time": time.time(),
                  "devices": gpu_snapshot(), "host": host_snapshot(WORKSPACE)}
    elif args.action == "suggest":
        result = suggest_next([read_json(path) for path in args.paths])
    elif args.action == "summarize":
        result = summarize(args.state_dir, args.stage, args.n)
    elif args.action == "verify-cohort":
        result = {"valid": True, "cohort": validate_cohort(read_json(args.paths[0]))}
    else:
        result = {"valid": verify_frozen_resource(read_json(args.paths[0]))}
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


WORKSPACE = Path("/private/Tactile")
if __name__ == "__main__":
    raise SystemExit(main())
