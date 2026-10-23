#!/usr/bin/env python3
"""Controller-only A2 trainer adapter, terminal verifier and real reload smoke."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.tools.a2_controller import atomic_json, file_hash, inside, read_json, require_worker_lease


def reload_smoke(root):
    """Restore every trainer state through its production loader, on leased CUDA."""
    require_worker_lease(kind="train")
    import torch
    from anysole import train
    from anysole.data.dataset import GroupedViewDataset
    checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    keys_required = {"model", "optimizer", "scheduler", "scaler", "sampler", "selector", "training", "rng", "stats", "schedule"}
    if not keys_required <= checkpoint.keys():
        raise ValueError("incomplete checkpoint recovery state")
    device = torch.device("cuda:0")
    cameras = tuple(config.get("train_cameras", (3,) if config["experiment"] in ("base_legacy", "base_time") else (1, 2, 3, 4)))
    dataset = GroupedViewDataset(manifest=Path(config["manifest_path"]), dataset_root=Path(config["dataset_root"]),
        feature_cache_root=Path(config["feature_cache_root"]), smpl_model_path=Path(config["smpl_model_path"]),
        window_length=int(config["window_length"]), stride=int(config["train_stride"]), cameras=cameras,
        split="train", cover_tail=False)
    train.restrict_pilot_windows(dataset, config)
    sampler = train.GroupBudgetSampler([len(w[3]) for w in dataset.windows],
        [dataset.sample_key(i) for i in range(len(dataset))], int(config["seed"]))
    model = train.get_builder("visual").build(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    target = int(config["target_updates"])
    warmup = max(1, int(float(config.get("lr_warmup_frac", .05)) * target))
    def factor(step):
        value = min((step + 1) / warmup, 1.)
        if config.get("lr_schedule", "constant") == "cosine":
            value *= .5 * (1 + math.cos(math.pi * min(step, target) / target))
        return value
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp")))
    selector = train.JointSelector(config.get("selection_m_ref"), config.get("selection_p_ref"), config.get("selection_coverage_hash"),
        motion_weight=config.get("selection_motion_weight", .5), pressure_weight=config.get("selection_pressure_weight", .5))
    loaded = train.load_visual_checkpoint(root / "last.pt", config, model, optimizer, scheduler, scaler,
                                         sampler, selector, device)
    def equal(actual, expected):
        if isinstance(actual, torch.Tensor):
            return isinstance(expected, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
        if isinstance(actual, dict):
            return actual.keys() == expected.keys() and all(equal(actual[k], expected[k]) for k in actual)
        if isinstance(actual, (list, tuple)):
            return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
        return actual == expected
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), checkpoint["model"][name]):
            raise ValueError("reloaded model tensor mismatch")
    if sampler.state_dict() != checkpoint["sampler"] or scheduler.state_dict() != checkpoint["scheduler"] or scaler.state_dict() != checkpoint["scaler"]:
        raise ValueError("reloaded sampler/scheduler/scaler mismatch")
    if not equal(optimizer.state_dict(), checkpoint["optimizer"]):
        raise ValueError("reloaded optimizer moment/parameter-group mismatch")
    if not torch.equal(torch.get_rng_state(), checkpoint["rng"]["torch"]):
        raise ValueError("reloaded CPU RNG mismatch")
    for actual, expected in zip(torch.cuda.get_rng_state_all(), checkpoint["rng"]["cuda"] or []):
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError("reloaded CUDA RNG mismatch")
    return {"checkpoint_reloaded": True, "state_complete": True,
            "checkpoint_sha256": file_hash(root / "last.pt"),
            "sampler_sha256": train.canonical_hash(loaded["sampler"]),
            "successful_updates": loaded["training"]["successful_updates"]}


def verify(root, *, smoke=False, identity=None):
    from configs.tools.runner import verified_visual_completion
    record = read_json(root / "run.json")
    if smoke:
        evidence = read_json(root / "a2_smoke.json")
        if not evidence.get("checkpoint_reloaded") or evidence["checkpoint_sha256"] != file_hash(root / "last.pt"):
            raise ValueError("missing real smoke reload evidence")
        if record.get("successful_updates") != 25 or record.get("rgb_windows") != 25 * evidence.get("effective_batch_rgb_windows", 256):
            raise ValueError("smoke successful budget mismatch")
        if not all(evidence.get(k) for k in ("state_complete", "full_validation", "maximum_effective_batch")):
            raise ValueError("incomplete real smoke")
    elif not verified_visual_completion(root, identity or record["semantic_hash"]):
        raise ValueError("formal terminal verification failed")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "smoke", "verify", "verify-smoke"))
    p.add_argument("--config", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--identity")
    args = p.parse_args(argv)
    root = inside(args.run_dir)
    if args.action.startswith("verify"):
        print(json.dumps({"verified": verify(root, smoke=args.action == "verify-smoke", identity=args.identity)}))
        return 0
    lease = require_worker_lease(kind="train")
    if str(root) != str(Path(lease["job"]["output_dir"]).resolve()):
        raise ValueError("trainer output is not leased")
    import torch
    import yaml
    from anysole import train
    config = yaml.safe_load(args.config.read_text())
    if config.get("a2_version") not in ("anysole.a2.v1", "anysole.a3.v1"):
        raise ValueError("trainer adapter requires explicit A2/A3 schema")
    recovery_note = os.environ.get("A2_RECOVERY_NOTE")
    if recovery_note:
        config["recovery_note"] = recovery_note
    smoke = args.action == "smoke"
    batch_rgb = config.get("effective_batch_rgb_windows")
    if smoke and (config.get("stage") != "pilot" or config.get("validation_every_updates") != 25 or config.get("timing_warmup_updates") != 5
                  or type(batch_rgb) is not int or batch_rgb <= 0
                  or (config["a2_version"] == "anysole.a2.v1" and batch_rgb != 256)):
        raise ValueError("resource smoke requires pilot, validation25, warmup5 and a positive frozen RGB batch (A2=256)")
    began = time.time()
    record = train.train_visual_run(config, root, resume=args.resume, stop_after_updates=25 if smoke else None)
    if smoke:
        gc.collect()
        torch.cuda.empty_cache()
        evidence = reload_smoke(root)
        ledger = [json.loads(line) for line in (root / "update_ledger.jsonl").read_text().splitlines()]
        ledger = [row for row in ledger if row.get("attempt_id") == record.get("attempt_id")]
        if [row["successful_update"] for row in ledger] != list(range(1, 26)):
            raise ValueError("actual successful update ledger coverage mismatch")
        actual_batches = [sum(len(cameras) for cameras in row["cameras"]) for row in ledger]
        if any(value != batch_rgb for value in actual_batches):
            raise ValueError("actual measured effective batch differs from frozen RGB budget")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        by_step = {int(row["progress/update"]): row for row in rows}
        if not set(range(1, 26)) <= by_step.keys():
            raise ValueError("resource smoke requires log_every_updates=1")
        timed = by_step[25]["unix_time"] - by_step[5]["unix_time"]
        last = record.get("last_validation") or {}
        evidence.update(schema="anysole.a2.smoke.v1", warmup_updates=5, timed_updates=20,
            a2_version=config["a2_version"], effective_batch_rgb_windows=batch_rgb,
            timed_seconds=timed, timed_rgb_windows=20 * batch_rgb, timed_rgb_windows_per_second=20 * batch_rgb / timed,
            timed_started_unix=by_step[5]["unix_time"], timed_finished_unix=by_step[25]["unix_time"],
            wall_seconds=time.time() - began,
            full_validation=last.get("val/motion_frames", 0) > 0 and last.get("val/pressure_cells", 0) > 0,
            maximum_effective_batch=min(actual_batches) == max(actual_batches) == batch_rgb,
            minimum_actual_batch_rgb=min(actual_batches), maximum_actual_batch_rgb=max(actual_batches),
            validated_motion_frames=last.get("val/motion_frames", 0),
            validated_pressure_cells=last.get("val/pressure_cells", 0),
            validation_coverage_hash=last.get("val/coverage_hash"),
            measured_lambda_mv_min=min(row["lambda_mv"] for row in ledger),
            measured_lambda_mv_max=max(row["lambda_mv"] for row in ledger),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(root / "a2_smoke.json", evidence)
    print(json.dumps({"run_dir": str(root), "state": record["state"], "successful_updates": record["successful_updates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
