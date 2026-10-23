#!/usr/bin/env python3
"""Single-writer A2 queue, process leases and finite resource admission.

The plan is immutable. A root-issued grant binds its exact hash and resource
hash; candidate resources can run measurement jobs only. No legacy resource
receipt is accepted. The worker wrapper is deliberately independent of torch.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

WORKSPACE = Path("/private/Tactile")
SCHEMA = "anysole.a2.plan.v1"
RESOURCE_SCHEMA = "anysole.a2.resource.v1"
QUEUE_SCHEMA = "anysole.a2.queue.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tmp.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if tmp.read_text() != payload:
        raise OSError("temporary JSON differs before atomic rename")
    os.replace(tmp, path)
    sync_directory(path.parent)
    if path.read_text() != payload:
        raise OSError("published JSON differs after atomic rename")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"path outside workspace: {path}")
    return path


def starttime(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, identity):
    if identity is None or starttime(pid) != identity:
        return False
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


@contextlib.contextmanager
def exclusive(path):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another controller/writer owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < (0 if zero else 1):
        raise ValueError(f"invalid {name}")


def validate_job(job):
    if not isinstance(job.get("id"), str) or not job["id"] or "/" in job["id"]:
        raise ValueError("job requires a path-safe unique id")
    if job.get("kind") not in ("train", "eval", "cpu"):
        raise ValueError("job kind must be train, eval, or cpu")
    for key in ("argv", "verify_argv"):
        if not isinstance(job.get(key), list) or not job[key] or not all(isinstance(v, str) and v for v in job[key]):
            raise ValueError(f"{key} must be a nonempty argv list")
    inside(job["cwd"])
    inside(job["output_dir"])
    if not isinstance(job.get("source_files"), dict) or not job["source_files"]:
        raise ValueError("every job requires frozen source_files")
    if not isinstance(job.get("inputs", {}), dict):
        raise ValueError("inputs must map absolute paths to SHA256")
    for key in ("source_files", "inputs"):
        for path, expected in job.get(key, {}).items():
            if file_hash(inside(path)) != expected:
                raise ValueError(f"frozen {key} drift: {path}")
    if job.get("config_path") and file_hash(inside(job["config_path"])) != job.get("config_sha256"):
        raise ValueError("frozen config drift")
    if job["kind"] == "train":
        positive(job["target_updates"], "target_updates")
        positive(job["target_rgb_windows"], "target_rgb_windows")
    if job["kind"] == "eval":
        cp = job.get("checkpoint", {})
        path = inside(cp["path"])
        if file_hash(path) != cp.get("sha256"):
            raise ValueError("evaluation checkpoint must have a frozen hash")
        if path.stat().st_mode & 0o222:
            raise ValueError("evaluation requires a read-only immutable checkpoint copy")
    if job.get("max_recoveries", 2) not in (0, 1, 2):
        raise ValueError("at most two recoveries are permitted")
    allowed = job.get("allowed_gpus", list(range(7)))
    if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed) or any(type(gpu) is not int or gpu not in range(7) for gpu in allowed):
        raise ValueError("allowed_gpus must be unique physical indices in0..6")


def validate_bundle(plan, resource, grant=None, *, fixture=False):
    if plan.get("schema") != SCHEMA or plan.get("a2_version") not in ("anysole.a2.v1", "anysole.a3.v1"):
        raise ValueError("explicit A2/A3 plan schema required")
    if not plan.get("frozen") or resource.get("schema") != RESOURCE_SCHEMA:
        raise ValueError("frozen plan and A2 resources required")
    if plan.get("purpose") not in ("formal", "measurement", "diagnostic", "fixture"):
        raise ValueError("unknown purpose")
    if plan.get("purpose") == "formal" and not resource.get("frozen"):
        raise ValueError("formal jobs require measured frozen resources")
    if resource.get("gpus") != list(range(7)):
        raise ValueError("all physical GPUs 0..6 must be explicitly represented")
    if not isinstance(resource.get("profiles"), dict) or not resource["profiles"]:
        raise ValueError("resource profiles missing")
    for name, profile in resource["profiles"].items():
        if profile.get("kind") not in ("train", "eval", "cpu"):
            raise ValueError(f"invalid profile {name}")
        for key in ("cap", "peak_rss_bytes", "cpu_threads"):
            positive(profile.get(key), f"profile {name} {key}")
            if not isinstance(profile[key], int):
                raise ValueError("resource caps/bytes/threads must be integers")
        positive(profile.get("peak_gpu_bytes", 0), "peak_gpu_bytes", zero=True)
    for key in ("host_reserve_fraction", "gpu_reserve_fraction"):
        value = resource.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= value < 1:
            raise ValueError(f"invalid reserve {key}")
    if resource["host_reserve_fraction"] < .2:
        raise ValueError("host reserve must be at least 20 percent")
    if resource.get("gpu_reserve_bytes", 0) < 2 * 1024**3:
        raise ValueError("GPU reserve must be at least 2 GiB")
    jobs = plan.get("jobs", [])
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("nonempty unique jobs required")
    outputs = [str(inside(j["output_dir"])) for j in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("one output writer per job required")
    ids = {j["id"] for j in jobs}
    for job in jobs:
        validate_job(job)
        if job.get("load_class") not in resource["profiles"] or resource["profiles"][job["load_class"]]["kind"] != job["kind"]:
            raise ValueError("job load class/kind mismatch")
        if set(job.get("dependencies", [])) - ids or job["id"] in job.get("dependencies", []):
            raise ValueError("unknown/self dependency")
        if plan["purpose"] == "formal" and job["kind"] == "train" and job.get("measurement"):
            raise ValueError("measurement may not count as formal training")
        for template in job.get("evaluation_templates", []):
            if job["kind"] != "train" or template.get("selection") not in ("best", "last", "snapshot"):
                raise ValueError("checkpoint evaluation template selection/parent invalid")
            if template.get("inference_contract_sha256") != digest(template.get("inference_contract")):
                raise ValueError("full inference contract identity missing")
            source = inside(template["source_path"])
            if not source.is_relative_to(inside(job["output_dir"])):
                raise ValueError("checkpoint template source must belong to its trainer")
            if template["job"].get("kind") != "eval":
                raise ValueError("checkpoint template must produce an evaluation job")
            if not any(token in template["job"].get("output_dir", "") for token in ("{checkpoint_sha256}", "{eval_id}")):
                raise ValueError("evaluation template output must be content addressed")
            if template.get("dispatch") and template["job"].get("load_class") not in resource["profiles"]:
                raise ValueError("automatic eval dispatch requires measured evaluation profile")
    visiting, visited = set(), set()
    by_id = {j["id"]: j for j in jobs}
    def visit(key):
        if key in visiting:
            raise ValueError("dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for child in by_id[key].get("dependencies", []):
            visit(child)
        visiting.remove(key)
        visited.add(key)
    for key in ids:
        visit(key)
    if fixture:
        if plan["purpose"] != "fixture" or any(j["kind"] != "cpu" for j in jobs):
            raise ValueError("fixtures may run CPU jobs only")
    else:
        owner = "root-a3" if plan["a2_version"] == "anysole.a3.v1" else "root-a2"
        if not grant or grant.get("schema") != "anysole.a2.controller-grant.v1" or grant.get("authorized_by") != owner:
            raise ValueError(f"specific {owner} controller grant required")
        if grant.get("plan_sha256") != digest(plan) or grant.get("resource_sha256") != digest(resource):
            raise ValueError("grant does not bind this exact plan/resource")
        if grant.get("purpose") != plan["purpose"] or grant.get("revoked"):
            raise ValueError("grant purpose/revocation mismatch")
        if grant.get("expires_unix", 0) <= time.time():
            raise ValueError("expired controller grant")
        if any(j["kind"] != "cpu" for j in jobs) and grant.get("allow_cuda") is not True:
            raise ValueError("CUDA not granted")
        if resource.get("frozen"):
            from configs.tools.a2_resource import verify_frozen_resource
            verify_frozen_resource(resource)


def host_snapshot(root):
    import psutil
    mem, swap, disk = psutil.virtual_memory(), psutil.swap_memory(), shutil.disk_usage(root)
    return {"total": mem.total, "available": mem.available, "swap_used": swap.used,
            "disk_free": disk.free, "cpu_count": os.cpu_count() or 1,
            "load1": os.getloadavg()[0], "iowait_percent": getattr(psutil.cpu_times_percent(), "iowait", 0)}


def gpu_snapshot():
    """nvidia-smi observes physical indices; it does not create a CUDA context."""
    fields = "index,memory.total,memory.free,memory.used,utilization.gpu"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    rows = {}
    for line in result.stdout.splitlines():
        gpu, total, free, used, util = [int(v.strip()) for v in line.split(",")]
        rows[gpu] = {"total": total * 1024**2, "free": free * 1024**2,
                     "used": used * 1024**2, "utilization": util, "pids": []}
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=15)
    uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=15)
    indices = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in uuids.stdout.splitlines()}
    for line in procs.stdout.splitlines():
        uuid, pid = [v.strip() for v in line.split(",")]
        if uuid in indices:
            rows[indices[uuid]]["pids"].append(int(pid))
    return rows


def ready_jobs(plan, state):
    rows = []
    for position, job in enumerate(plan["jobs"]):
        record = state["jobs"][job["id"]]
        if record["state"] == "pending" and all(state["jobs"][d]["state"] == "completed" for d in job.get("dependencies", [])):
            # The longest mandatory run starts first; bounded aging preserves fairness.
            rows.append((0 if job.get("a2_run_id") == "B-CAM3" else 1,
                         0 if job["kind"] == "train" else 1,
                         -int(job.get("priority", 0)), -int(job.get("estimated_seconds", 0)), position, job))
    return [r[-1] for r in sorted(rows, key=lambda row: row[:-1])]


def admission(job, gpu, active, resource, host, devices, *, owned_pids=(), cap_overrides=None):
    profile = resource["profiles"][job["load_class"]]
    host_reserve = host["total"] * resource["host_reserve_fraction"]
    if host["available"] - profile["peak_rss_bytes"] < host_reserve:
        return "host_memory_reserve"
    if host["disk_free"] < resource.get("disk_min_free_bytes", 10 * 1024**3):
        return "disk_low_watermark"
    if host.get("swap_growth_bytes", 0) > resource.get("max_swap_growth_bytes", 64 * 1024**2):
        return "swap_growth"
    if host.get("iowait_percent", 0) >= resource.get("max_iowait_percent", 30):
        return "io_blocked"
    cpu = sum(resource["profiles"][r["load_class"]]["cpu_threads"] for r in active)
    if cpu + profile["cpu_threads"] > resource.get("cpu_threads_budget", host["cpu_count"]):
        return "cpu_budget"
    if job["kind"] == "cpu":
        return None if sum(r["kind"] == "cpu" for r in active) < profile["cap"] else "cpu_slot_cap"
    device = devices.get(gpu)
    if not device:
        return "device_unavailable"
    if set(device.get("pids", [])) - set(owned_pids):
        return "external_gpu_process"
    reserve = max(resource["gpu_reserve_bytes"], device["total"] * resource["gpu_reserve_fraction"])
    if device["free"] - profile["peak_gpu_bytes"] < reserve:
        return "gpu_memory_reserve"
    same = [r for r in active if r.get("gpu") == gpu]
    cap = min(profile["cap"], (cap_overrides or {}).get(str(gpu), profile["cap"]))
    if sum(r["kind"] == job["kind"] for r in same) >= cap:
        return "class_slot_cap"
    counts = {"train": sum(r["kind"] == "train" for r in same) + (job["kind"] == "train"),
              "eval": sum(r["kind"] == "eval" for r in same) + (job["kind"] == "eval")}
    if all(counts.values()) and counts not in resource.get("mixed_combinations", []):
        return "unmeasured_mixed_combination"
    # Different measured load classes can share a pure card only when explicitly approved.
    classes = {r["load_class"] for r in same} | {job["load_class"]}
    if len(classes) > 1 and sorted(classes) not in resource.get("compatible_load_classes", []):
        return "unmeasured_class_combination"
    return None


def heartbeat_status(attempt, now, *, startup_timeout=900, timeout=300):
    """Cache a valid pulse over atomic-replace/read gaps, bound to process identity."""
    try:
        pulse = read_json(Path(attempt["directory"]) / "heartbeat.json")
    except (OSError, ValueError):
        pulse = None
    if pulse and pulse.get("pid") == attempt["pid"] and pulse.get("starttime") == attempt["starttime"]:
        stamp = pulse.get("unix_time")
        if not isinstance(stamp, bool) and isinstance(stamp, (float, int)) and math.isfinite(stamp) and stamp <= now:
            attempt["last_valid_heartbeat"] = max(stamp, attempt.get("last_valid_heartbeat", 0))
            attempt["pulse"] = pulse
    stamp = attempt.get("last_valid_heartbeat")
    reference = stamp if stamp is not None else attempt["started_unix"]
    return "heartbeat_stale" if now - reference > (timeout if stamp is not None else startup_timeout) else None


def immutable_checkpoint(source, expected_hash, destination):
    """Copy one complete atomic trainer checkpoint into a content-addressed name."""
    source, destination = inside(source), inside(destination)
    if file_hash(source) != expected_hash:
        raise ValueError("checkpoint is not the published complete checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != expected_hash:
            raise ValueError("immutable checkpoint collision")
        return destination
    tmp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, tmp)
    if file_hash(tmp) != expected_hash or file_hash(source) != expected_hash:
        tmp.unlink()
        raise ValueError("checkpoint changed during publication; retry next tick")
    os.chmod(tmp, 0o444)
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, destination)
    sync_directory(destination.parent)
    if file_hash(destination) != expected_hash:
        raise OSError("published immutable checkpoint differs after atomic rename")
    return destination


def require_worker_lease(*, kind, checkpoint=None):
    """CUDA entrypoints call before importing/initializing a CUDA backend."""
    path = os.environ.get("A2_LEASE_SPEC")
    if not path or file_hash(inside(path)) != os.environ.get("A2_LEASE_SHA256"):
        raise ValueError("missing or changed A2 controller lease")
    spec = read_json(path)
    pid = int(os.environ.get("A2_LEASE_PID", "-1"))
    if spec.get("schema") != "anysole.a2.lease.v1" or not alive(pid, os.environ.get("A2_LEASE_STARTTIME")):
        raise ValueError("A2 controller lease owner is not alive")
    if spec["job"]["kind"] != kind or spec.get("gpu") not in range(7) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["gpu"]):
        raise ValueError("A2 lease job kind/physical GPU mismatch")
    if checkpoint is not None:
        expected = spec["job"].get("checkpoint", {})
        if str(Path(checkpoint).resolve()) != str(Path(expected.get("path", "")).resolve()) or file_hash(checkpoint) != expected.get("sha256"):
            raise ValueError("A2 lease checkpoint mismatch")
    return spec


def initial_state(plan, resource):
    return {"schema": QUEUE_SCHEMA, "plan_sha256": digest(plan), "resource_sha256": digest(resource),
            "a2_version": plan["a2_version"], "capacity_max_n": plan.get("capacity_max_n", 4),
            "created_unix": time.time(), "jobs": {j["id"]: {"state": "pending", "attempts": []} for j in plan["jobs"]},
            "cap_overrides": {}, "events": [], "dynamic_jobs": [], "publications": {}}


def _owned_processes(active):
    import psutil
    pids = set()
    for attempt in active:
        if alive(attempt["pid"], attempt["starttime"]):
            pids.add(attempt["pid"])
            try:
                pids.update(p.pid for p in psutil.Process(attempt["pid"]).children(recursive=True))
            except psutil.Error:
                pass
    return pids


class Controller:
    def __init__(self, plan, resource, state_dir, *, grant=None, fixture=False):
        validate_bundle(plan, resource, grant, fixture=fixture)
        self.plan, self.resource = plan, resource
        self.grant = grant
        self.root, self.fixture = inside(state_dir), fixture
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "a2_queue.json"
        try:
            self.state = read_json(path)
        except FileNotFoundError:
            if any(entry.name == path.name for entry in self.root.iterdir()):
                raise OSError("queue entry visible but unreadable; refusing to reset persistent state")
            self.state = initial_state(plan, resource)
        if self.state["plan_sha256"] != digest(plan) or self.state["resource_sha256"] != digest(resource):
            raise ValueError("queue belongs to a different immutable plan/resource")
        self.processes, self.last_swap = {}, None

    def all_jobs(self):
        return self.plan["jobs"] + self.state.get("dynamic_jobs", [])

    def effective_plan(self):
        return {**self.plan, "jobs": self.all_jobs()}

    def publish_checkpoints(self):
        for parent in self.plan["jobs"]:
            if parent["kind"] != "train" or self.state["jobs"][parent["id"]]["state"] not in ("running", "completed"):
                continue
            try:
                run = read_json(Path(parent["output_dir"]) / "run.json")
            except (OSError, ValueError):
                continue
            for template in parent.get("evaluation_templates", []):
                # Publish terminal best/last once; fixed-step snapshots may overlap
                # training. Repeatedly chasing a changing best path is forbidden.
                if template["selection"] in ("best", "last") and self.state["jobs"][parent["id"]]["state"] != "completed":
                    continue
                source = inside(template["source_path"])
                if not source.is_file():
                    continue
                field = template.get("published_hash_field")
                expected = run.get(field) if field else file_hash(source)
                if not expected:
                    continue
                contract = template["inference_contract_sha256"]
                key = digest({"checkpoint_sha256": expected, "inference_contract_sha256": contract})
                alias = {"parent_id": parent["id"], "selection": template["selection"], "source_path": str(source),
                         "successful_updates": (run.get("selection", {}).get("best_step") if template["selection"] == "best" else
                                                template.get("step", run.get("successful_updates")))}
                publications = self.state.setdefault("publications", {})
                if key in publications:
                    if alias not in publications[key]["aliases"]:
                        publications[key]["aliases"].append(alias)
                        atomic_json(self.root / "published_jobs" / (key + ".json"), publications[key])
                    continue
                destination = inside(parent.get("checkpoint_archive_dir", self.root / "checkpoints")) / (expected + ".pt")
                try:
                    immutable_checkpoint(source, expected, destination)
                except (OSError, ValueError) as exc:
                    self.state["events"].append({"event": "checkpoint_publication_deferred", "parent_id": parent["id"], "unix_time": time.time(), "reason": str(exc)})
                    continue
                def substitute(value):
                    if isinstance(value, str):
                        return value.replace("{checkpoint}", str(destination)).replace("{checkpoint_sha256}", expected).replace("{eval_id}", key)
                    if isinstance(value, list):
                        return [substitute(v) for v in value]
                    if isinstance(value, dict):
                        return {k: substitute(v) for k, v in value.items()}
                    return value
                job = substitute(template["job"])
                job.update(id="eval-" + key, checkpoint={"path": str(destination), "sha256": expected}, dependencies=[])
                validate_job(job)
                if str(inside(job["output_dir"])) in {str(inside(existing["output_dir"])) for existing in self.all_jobs()}:
                    raise ValueError("dynamic evaluation output writer collision")
                publication = {"schema": "anysole.a2.checkpoint-publication.v1", "checkpoint_sha256": expected,
                    "checkpoint_path": str(destination), "inference_contract_sha256": contract,
                    "template_sha256": digest(template), "aliases": [alias], "job": job,
                    "dispatch": bool(template.get("dispatch")), "published_unix": time.time()}
                atomic_json(self.root / "published_jobs" / (key + ".json"), publication)
                publications[key] = publication
                if template.get("dispatch"):
                    self.state.setdefault("dynamic_jobs", []).append(job)
                    self.state["jobs"][job["id"]] = {"state": "pending", "attempts": []}
                self.save()

    def active(self):
        rows = []
        for job in self.all_jobs():
            record = self.state["jobs"][job["id"]]
            if record["state"] == "running":
                rows.append({**record["attempts"][-1], "kind": job["kind"], "load_class": job["load_class"], "job_id": job["id"]})
        return rows

    def launch(self, job, gpu):
        validate_job(job)  # Fail closed if any source/config/input drifted while queued.
        record = self.state["jobs"][job["id"]]
        number = len(record["attempts"])
        attempt_dir = self.root / "attempts" / job["id"] / str(number)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        spec = {"schema": "anysole.a2.lease.v1", "job": job, "gpu": gpu, "attempt": number, "directory": str(attempt_dir),
                "plan_sha256": digest(self.plan), "resource_sha256": digest(self.resource)}
        if number:
            if not job.get("resume_argv"):
                raise ValueError("recovery requires explicit unchanged-budget resume_argv")
            spec["job"] = {**job, "argv": job["resume_argv"]}
            spec["recovery_note"] = record["recovery_note"]
        atomic_json(attempt_dir / "spec.json", spec)
        attempt = {"number": number, "gpu": gpu, "directory": str(attempt_dir),
                   "started_unix": time.time(), "state": "launching"}
        record["attempts"].append(attempt)
        record["state"] = "launching"
        self.save()  # Reservation survives a controller crash during Popen/bind.
        env = dict(os.environ, TMPDIR=str(attempt_dir / "tmp"), PYTHONDONTWRITEBYTECODE="1")
        Path(env["TMPDIR"]).mkdir()
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        env["A2_RECOVERY_NOTE"] = spec.get("recovery_note", "")
        with (attempt_dir / "wrapper.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(attempt_dir / "spec.json")],
                                       cwd=job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        attempt.update(pid=process.pid, pgid=process.pid, starttime=starttime(process.pid), state="running")
        if attempt["starttime"] is None:
            raise RuntimeError("worker identity unavailable")
        record["state"] = "running"
        self.processes[process.pid] = process
        self.state["events"].append({"event": "launch", "job_id": job["id"], **attempt})
        self.save()

    def reconcile(self):
        now = time.time()
        for job in self.all_jobs():
            record = self.state["jobs"][job["id"]]
            if record["state"] == "launching":
                attempt = record["attempts"][-1]
                owner = Path(attempt["directory"]) / "owner.json"
                if owner.exists():
                    identity = read_json(owner)
                    if identity.get("spec_sha256") != file_hash(Path(attempt["directory"]) / "spec.json"):
                        raise ValueError("launch reservation owner mismatch")
                    attempt.update(pid=identity["pid"], pgid=identity["pgid"], starttime=identity["starttime"], state="running")
                    record["state"] = "running"
                elif now - attempt["started_unix"] > job.get("startup_timeout_sec", 900):
                    attempt["state"] = "failed"
                    self.fail(job, record, attempt, "launch_reservation_without_owner")
            if record["state"] != "running":
                continue
            attempt = record["attempts"][-1]
            result_path = Path(attempt["directory"]) / "result.json"
            if result_path.exists():
                result = read_json(result_path)
                if result.get("spec_sha256") != file_hash(Path(attempt["directory"]) / "spec.json") or result.get("pid") != attempt["pid"] or result.get("starttime") != attempt["starttime"]:
                    raise ValueError("worker result identity mismatch")
                process = self.processes.get(attempt["pid"])
                if process:
                    code = process.poll()
                    if code is None:
                        continue
                    result["wrapper_exit_code"] = code
                    if code != 0:
                        result.update(success=False, reason=result.get("reason") or "wrapper_exit_nonzero")
                elif alive(attempt["pid"], attempt["starttime"]):
                    continue
                attempt.update(result=result, state="completed" if result.get("success") else "failed", finished_unix=now)
                if result.get("success"):
                    record["state"] = "completed"
                else:
                    self.fail(job, record, attempt, result.get("reason", "worker_failed"))
                continue
            if not alive(attempt["pid"], attempt["starttime"]):
                attempt.update(state="failed", finished_unix=now)
                self.fail(job, record, attempt, "worker_disappeared_without_result")
                continue
            reason = heartbeat_status(attempt, now, startup_timeout=job.get("startup_timeout_sec", 900), timeout=job.get("heartbeat_timeout_sec", 300))
            pulse = attempt.get("pulse", {})
            progress = pulse.get("progress", {})
            update = progress.get("successful_updates", progress.get("update"))
            if update is not None and update != attempt.get("last_update"):
                attempt.update(last_update=update, last_progress_unix=now)
            if job.get("progress_timeout_sec") and now - attempt.get("last_progress_unix", attempt["started_unix"]) > job["progress_timeout_sec"]:
                reason = "progress_stalled"
            if reason and "stop_requested_unix" not in attempt:
                # Signal only the exact owned process group, never an external PID.
                os.killpg(attempt["pgid"], signal.SIGTERM)
                attempt.update(stop_requested_unix=now, watchdog_reason=reason)
            elif attempt.get("stop_requested_unix") and now - attempt["stop_requested_unix"] > 60:
                os.killpg(attempt["pgid"], signal.SIGKILL)

    def fail(self, job, record, attempt, reason):
        attempt["reason"] = reason
        record.update(state="failed", reason=reason)
        prior = [a.get("reason") for a in record["attempts"][:-1]]
        # Only a first diagnosed OOM with a complete last checkpoint is automatic.
        checkpoint = Path(job["output_dir"]) / "last.pt"
        if reason == "cuda_oom" and job.get("auto_recover_oom") and reason not in prior and len(record["attempts"]) <= job.get("max_recoveries", 2) and job.get("resume_argv") and checkpoint.is_file():
            gpu = str(attempt["gpu"])
            self.state["cap_overrides"][gpu] = max(1, int(self.resource["profiles"][job["load_class"]]["cap"]) - 1)
            record.update(state="pending", recovery_note=f"attempt {attempt['number']} CUDA OOM; lower concurrency; strict complete last checkpoint resume")
        self.state["events"].append({"event": "failure", "job_id": job["id"], "reason": reason, "unix_time": time.time(), "next_state": record["state"]})

    def tick(self, *, host=None, devices=None):
        self.reconcile()
        self.publish_checkpoints()
        host = host or host_snapshot(self.root)
        host["swap_growth_bytes"] = max(0, host["swap_used"] - self.last_swap) if self.last_swap is not None else 0
        self.last_swap = host["swap_used"]
        if devices is None:
            try:
                devices = {} if self.fixture else gpu_snapshot()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                devices = {}
                self.state["events"].append({"event": "resource_observation_failed", "unix_time": time.time(), "reason": str(exc)})
        self.state["host"], self.state["devices"] = host, devices
        with (self.root / "resource_samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"unix_time": time.time(), "host": host, "devices": devices}, sort_keys=True) + "\n")
        refusals = {}
        for job in ready_jobs(self.effective_plan(), self.state):
            if self.grant and self.grant.get("expires_unix", 0) <= time.time():
                refusals[job["id"]] = {str(g): "controller_grant_expired" for g in range(7)}
                continue
            active = self.active()
            owned = _owned_processes(active)
            candidates = [None] if job["kind"] == "cpu" else sorted(job.get("allowed_gpus", list(range(7))), key=lambda gpu: (sum(r.get("gpu") == gpu for r in active), gpu))
            for gpu in candidates:
                reason = admission(job, gpu, active, self.resource, host, devices, owned_pids=owned, cap_overrides=self.state["cap_overrides"])
                if reason is None:
                    try:
                        self.launch(job, gpu)
                    except (OSError, ValueError, RuntimeError) as exc:
                        record = self.state["jobs"][job["id"]]
                        record.update(state="failed", reason=f"launch_preflight: {exc}")
                        self.state["events"].append({"event": "launch_preflight_failed", "job_id": job["id"], "unix_time": time.time(), "reason": str(exc)})
                        break
                    # Reserve host and device peaks between launches in this tick.
                    profile = self.resource["profiles"][job["load_class"]]
                    host["available"] -= profile["peak_rss_bytes"]
                    if gpu is not None:
                        devices[gpu]["free"] -= profile["peak_gpu_bytes"]
                    break
                refusals.setdefault(job["id"], {})[str(gpu)] = reason
        self.state["refusals"] = refusals
        self.save()
        return self.state

    def save(self):
        self.state["updated_unix"] = time.time()
        atomic_json(self.root / "a2_queue.json", self.state)
        active = self.active()
        ready = ready_jobs(self.effective_plan(), self.state)
        cards = []
        for gpu in range(7):
            rows = [r for r in active if r.get("gpu") == gpu]
            reasons = {v.get(str(gpu)) for v in self.state.get("refusals", {}).values()} - {None}
            caps = {key: p["cap"] for key, p in self.resource["profiles"].items() if p["kind"] != "cpu"}
            eligible = sum(job["kind"] != "cpu" and gpu in job.get("allowed_gpus", list(range(7))) for job in ready)
            idle = len(rows) < max(caps.values(), default=0)
            idle_reason = None
            if not rows or idle:
                idle_reason = sorted(reasons) or (["no_ready_gpu_job"] if not eligible else ["next_tick_refill"])
            cards.append({"gpu": gpu, "running": len(rows), "jobs": [r["job_id"] for r in rows],
                          "caps": caps, "pending_eligible": eligible,
                          "running_by_kind": {kind: sum(row["kind"] == kind for row in rows) for kind in ("train", "eval")},
                          "resource": self.state.get("devices", {}).get(gpu, self.state.get("devices", {}).get(str(gpu))),
                          "idle_reason": idle_reason,
                          "ready_backlog": len(ready)})
        terminal = all(r["state"] in ("completed", "failed", "blocked") for r in self.state["jobs"].values())
        # Dependency failure is explicit; an empty runnable queue cannot mean success.
        for job in self.all_jobs():
            record = self.state["jobs"][job["id"]]
            if record["state"] == "pending" and any(self.state["jobs"][d]["state"] in ("failed", "blocked") for d in job.get("dependencies", [])):
                record.update(state="blocked", reason="dependency_failed")
        terminal = all(r["state"] in ("completed", "failed", "blocked") for r in self.state["jobs"].values())
        training = [self.state["jobs"][j["id"]] for j in self.plan["jobs"] if j["kind"] == "train"]
        train_terminal = bool(training) and all(r["state"] in ("completed", "failed", "blocked") for r in training)
        intervals, process_seconds = {gpu: [] for gpu in range(7)}, 0
        for record in self.state["jobs"].values():
            for attempt in record["attempts"]:
                if attempt.get("gpu") is not None:
                    finish = attempt.get("finished_unix", self.state["updated_unix"])
                    intervals[attempt["gpu"]].append((attempt["started_unix"], finish))
                    process_seconds += max(0, finish - attempt["started_unix"])
        card_seconds = 0
        for spans in intervals.values():
            end = 0
            for begin, finish in sorted(spans):
                card_seconds += max(0, finish - max(begin, end))
                end = max(end, finish)
        atomic_json(self.root / "published_evaluations.json", self.state.get("publications", {}))
        atomic_json(self.root / "status.json", {"schema": "anysole.a2.status.v1", "updated_unix": self.state["updated_unix"],
                    "plan_sha256": self.state["plan_sha256"], "cards": cards,
                    "counts": {s: sum(r["state"] == s for r in self.state["jobs"].values()) for s in ("pending", "launching", "running", "completed", "failed", "blocked")},
                    "jobs": self.state["jobs"], "host": self.state.get("host"), "terminal": terminal,
                    "train_terminal": train_terminal, "train_success": train_terminal and all(r["state"] == "completed" for r in training),
                    "published_evaluation_count": len(self.state.get("publications", {})),
                    "cost": {"physical_card_busy_hours": card_seconds / 3600,
                             "allocated_process_gpu_hours": process_seconds / 3600},
                    "success": terminal and all(r["state"] == "completed" for r in self.state["jobs"].values())})
        atomic_json(self.root / "a2_queue.json", self.state)


def worker(spec_path):
    import psutil
    spec_path = inside(spec_path)
    spec = read_json(spec_path)
    job, directory = spec["job"], inside(spec["directory"])
    atomic_json(directory / "owner.json", {"pid": os.getpid(), "pgid": os.getpgrp(),
                "starttime": starttime(os.getpid()), "spec_sha256": file_hash(spec_path)})
    validate_job(job)
    output = inside(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    identity, pid = starttime(os.getpid()), os.getpid()
    os.environ.update(A2_LEASE_SPEC=str(spec_path), A2_LEASE_SHA256=file_hash(spec_path),
                      A2_LEASE_PID=str(pid), A2_LEASE_STARTTIME=str(identity))
    stopped, child, peaks = threading.Event(), None, {"rss_bytes": 0, "read_bytes": 0, "write_bytes": 0}
    started = time.time()
    def pulse():
        while not stopped.is_set():
            progress = {}
            try:
                record = read_json(output / "run.json")
                progress = {k: record.get(k) for k in ("successful_updates", "rgb_windows", "attempt_steps", "state")}
            except (OSError, ValueError):
                pass
            processes = [psutil.Process(pid)]
            try:
                processes += processes[0].children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes)
                ios = [p.io_counters() for p in processes]
                peaks["rss_bytes"] = max(peaks["rss_bytes"], rss)
                peaks["read_bytes"] = max(peaks["read_bytes"], sum(i.read_bytes for i in ios))
                peaks["write_bytes"] = max(peaks["write_bytes"], sum(i.write_bytes for i in ios))
            except psutil.Error:
                pass
            atomic_json(directory / "heartbeat.json", {"pid": pid, "starttime": identity, "pgid": os.getpgrp(),
                        "child_pid": child.pid if child else None, "unix_time": time.time(), "progress": progress, "peaks": peaks})
            stopped.wait(float(job.get("heartbeat_sec", 2)))
    monitor = threading.Thread(target=pulse, daemon=True)
    success, code, verify_code, reason = False, None, None, "worker_failed"
    def stop_handler(*_):
        if child is not None and child.poll() is None:
            child.terminate()
    old = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGTERM, signal.SIGINT)}
    with exclusive(output / "a2_writer.lock"):
        monitor.start()
        try:
            with (directory / "command.log").open("a") as log:
                child = subprocess.Popen(job["argv"], cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                if code == 0:
                    verified = subprocess.run(job["verify_argv"], cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT)
                    verify_code = verified.returncode
                    success = verify_code == 0
                text = (directory / "command.log").read_text(errors="replace")[-128000:].lower()
                if success:
                    reason = None
                elif "out of memory" in text and ("cuda" in text or "cublas" in text):
                    reason = "cuda_oom"
                elif any(t in text for t in ("non-finite", "nonfinite", "nan loss", "shape mismatch", "support mismatch")):
                    reason = "scientific_or_implementation_failure"
                elif code == 0:
                    reason = "terminal_verification_failed"
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            stopped.set()
            monitor.join(timeout=5)
            for sig, handler in old.items():
                signal.signal(sig, handler)
            atomic_json(directory / "result.json", {"schema": "anysole.a2.attempt.v1", "pid": pid, "starttime": identity,
                        "spec_sha256": file_hash(spec_path), "success": success, "exit_code": code,
                        "verify_exit_code": verify_code, "reason": reason, "started_unix": started,
                        "finished_unix": time.time(), "wall_seconds": time.time() - started, "peaks": peaks,
                        "gpu": spec["gpu"], "attempt": spec["attempt"], "job_id": job["id"]})
    return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "status", "worker", "snapshot"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--tick", type=float, default=10)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if args.action == "worker":
        return worker(args.spec)
    if args.action == "snapshot":
        print(immutable_checkpoint(args.source, args.sha256, args.destination))
        return 0
    if args.action == "status":
        print(json.dumps(read_json(args.state_dir / "status.json"), indent=2))
        return 0
    plan, resource = read_json(args.plan), read_json(args.resource)
    grant = read_json(args.grant) if args.grant else None
    validate_bundle(plan, resource, grant, fixture=args.fixture)
    if args.action == "validate":
        print(json.dumps({"valid": True, "jobs": len(plan["jobs"]), "plan_sha256": digest(plan), "resource_sha256": digest(resource)}))
        return 0
    positive(args.tick, "tick")
    with exclusive(args.state_dir / "controller.lock"):
        controller = Controller(plan, resource, args.state_dir, grant=grant, fixture=args.fixture)
        identity = {"pid": os.getpid(), "starttime": starttime(os.getpid()), "pgid": os.getpgrp(),
                    "plan_sha256": digest(plan), "resource_sha256": digest(resource), "started_unix": time.time()}
        atomic_json(args.state_dir / "controller_identity.json", identity)
        code, error = 1, None
        try:
            while True:
                controller.tick()
                status = read_json(args.state_dir / "status.json")
                if args.once or status["terminal"]:
                    code = 0 if args.once or status["success"] else 1
                    return code
                time.sleep(args.tick)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            atomic_json(args.state_dir / "controller_exit.json", {**identity, "finished_unix": time.time(),
                        "exit_code": code, "error": error, "once": args.once})


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
