#!/usr/bin/env python3
"""Execute one queued data-producing AnySole experiment task.

The runner intentionally keeps model training and evaluation in one experiment
transaction, following SurvPGC's ``run.sh -> main.py`` behaviour.  A model
entry is materialized below ``results/<experiment_id>/<model_run_name>/``;
analysis results are written by independent scripts below
``results_display/script/``.

Analysis and visualization are separate scripts under results_display/script.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ``python configs/tools/runner.py`` puts only ``configs/tools/`` on sys.path;
# add the repository root so the package-style imports below work in that mode.
_REPO_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_REPO_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_FOR_IMPORT))

from configs.tools.common import (
    REPO,
    CONFIG_DIR,
    TRAIN_ARG_FLAG_MAP,
    TRAIN_STORE_TRUE,
    checkpoint_path,
    config_path,
    effective_value,
    effective_variant,
    experiment_root,
    experiment_spec,
    git_sha,
    load_conf,
    load_registry,
    model_root,
    model_run_name,
    resolve_path,
    resolved_model_spec,
    sha256_file,
    source_checkpoint,
    specialist_paths,
    split_specialist,
    write_json,
)
from anysole.registry import infer_anysole_paths


PYTHON = sys.executable


def _command_log(command: list[str], path: Path, env: dict[str, str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(subprocess.list2cmdline([item]) for item in command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write("[exit=%d]\n" % completed.returncode)
    return int(completed.returncode)


def _merge_model_spec(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    merged = dict(registry.get("defaults", {}))
    merged.update(resolved_model_spec(registry, model_id))
    return merged


# train-arg keys (conf MODEL_<ID>_* lines) -> anysole.train CLI flags.
_TRAIN_FLAG_MAP = TRAIN_ARG_FLAG_MAP
_TRAIN_STORE_TRUE = TRAIN_STORE_TRUE


def _append_train_args(command: list[str], spec: dict[str, Any]) -> None:
    for key, flag in _TRAIN_FLAG_MAP.items():
        value = (spec.get("train_args") or {}).get(key)
        if value is None:
            continue
        if key in _TRAIN_STORE_TRUE:
            if bool(value):
                command.append(flag)
        elif isinstance(value, (list, tuple)):
            command += [flag, *(str(item) for item in value)]
        else:
            command += [flag, str(value)]


def _write_initial_manifest(registry: dict[str, Any], experiment_id: str, model_id: str, root: Path) -> Path:
    spec = _merge_model_spec(registry, model_id)
    # V3_3C fine-tune: the sanctioned provenance is the BASE ckpt passed to
    # --init-from (deliberate exception to the 2026-09-24 no-warm-start
    # independence rule; see z_note/multimodal_fusion/后续实验设计.md).
    init_from = (spec.get("train_args") or {}).get("init_from")
    if init_from:
        source = resolve_path(str(init_from))
        if source is None or not source.is_file():
            raise FileNotFoundError(
                "init_from base checkpoint missing for %s: %s" % (model_id, init_from))
    else:
        source = source_checkpoint(registry, model_id)
    manifest = {
        "version": 1,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_run_name": model_run_name(registry, model_id),
        "git_sha": git_sha(),
        "status": "running",
        "model_spec": spec,
        "checkpoint_source": str(source or ""),
        "checkpoint_source_sha256": sha256_file(source) if source else None,
        "fine_tune_base": bool(init_from),
    }
    path = root / "run_manifest.json"
    write_json(path, manifest)
    return path


def _update_manifest(path: Path, **updates: Any) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload.update(updates)
    write_json(path, payload)


def _train_command(registry: dict[str, Any], model_id: str) -> list[str]:
    spec = _merge_model_spec(registry, model_id)
    config = config_path(registry, model_id)
    modal = str(spec.get("modal", "anysolev2"))
    contact = str(spec.get("contact_method", "joint_and"))
    epochs = int(spec.get("epochs", 800))
    batch_size = int(spec.get("batch_size", 256))
    seed = int(spec.get("train_seed", spec.get("seed", 1)))
    device = str(spec.get("device", registry.get("defaults", {}).get("device", "cuda")))
    command = [
        PYTHON,
        "-m",
        "anysole.train",
        "--config",
        str(config),
        "--contact-method",
        contact,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    if spec.get("specialist_kind"):
        # Derived single-stream specialist: NOT a registered model.  It uses
        # the BASE model's entry structure (--model-name) with an explicit
        # --out-dir to the specialist's own dir (<base>_<kind>_<contact>;
        # --config-probs is not a stacked-variant field, so a plain
        # --model-name would clobber the base's own ckpt dir).  No
        # warm-start: every specialist trains from random init.
        paths = specialist_paths(registry, model_id)
        if paths is None:
            raise ValueError("specialist resolution failed for %s" % model_id)
        base_id, _kind = split_specialist(model_id)
        base_spec = _merge_model_spec(registry, base_id)
        base_name = str(base_spec.get("name") or base_id)
        command += ["--model-name", base_name]
        command += ["--out-dir", str(paths["model_dir"] / "checkpoints")]
    else:
        # Registered model: --model-name infers the out-dir (zero handwritten
        # paths); the structure comes from the model's entry script and every
        # run trains from random init.
        command += ["--model-name", str(spec["name"])]
    _append_train_args(command, spec)
    # V4B guard: the learned partition is structural data.  It is derived
    # from RAW training-data kinematics (no trained model — 2026-09-24);
    # fail loudly with the re-derivation recipe instead of a bare
    # FileNotFoundError.
    part_json = (spec.get("train_args") or {}).get("part_json")
    if part_json:
        path = resolve_path(str(part_json))
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "V4B partition file missing: %s\n"
                "Re-derive it (data-only, no trained model needed): run "
                "z_note/probes/probe_part_cluster_b_data.py, then point "
                "V4B's part_json at one of "
                "results/AnySole/partitions/partitions_kinematic_b_K*.json."
                % part_json
            )
    return command


def _materialize_checkpoint(registry: dict[str, Any], experiment_id: str, model_id: str, root: Path, log: Path, force: bool) -> Path:
    spec = _merge_model_spec(registry, model_id)
    train = bool(spec.get("train", False))
    if train:
        contact = str(spec.get("contact_method", "joint_and"))
        if spec.get("specialist_kind"):
            paths = specialist_paths(registry, model_id)
            if paths is None:
                raise ValueError("specialist resolution failed for %s" % model_id)
            candidates = [paths["ckpt_best"], paths["ckpt_last"]]
        else:
            variant = effective_variant(spec)
            candidates = [
                infer_anysole_paths(str(spec["name"]), variant=variant, contact=contact, which=which)["ckpt"]
                for which in ("best", "last")
            ]
        if not force:
            existing = next((candidate for candidate in candidates if candidate.is_file()), None)
            if existing is not None:
                return existing
        command = _train_command(registry, model_id)
        code = _command_log(command, log)
        if code != 0:
            raise RuntimeError("training failed for %s (exit %d); see %s" % (model_id, code, log))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("training completed without checkpoint for %s (expected %s)" % (model_id, candidates[0]))

    ckpt = checkpoint_path(registry, experiment_id, model_id)
    if not ckpt.is_file():
        raise FileNotFoundError("checkpoint missing for registered model %s: %s" % (model_id, ckpt))
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "source_checkpoint.txt").write_text(str(ckpt) + "\n", encoding="utf-8")
    return ckpt


def _eval_command(registry: dict[str, Any], model_id: str, root: Path, ckpt: Path, split: str) -> list[str]:
    spec = _merge_model_spec(registry, model_id)
    config = config_path(registry, model_id)
    modal = str(spec.get("modal", "anysolev2"))
    contact = str(spec.get("contact_method", "joint_and"))
    device = str(spec.get("device", registry.get("defaults", {}).get("device", "cuda")))
    metrics_dir = root / "metrics"
    command = [
        PYTHON,
        "-m",
        "anysole.eval",
        "--config",
        str(config),
        "--ckpt",
        str(ckpt),
        "--split",
        split,
        "--device",
        device,
        "--contact-method",
        contact,
        "--metrics-out",
        str(metrics_dir / (split + "_brief.json")),
        "--protocol-out",
        str(metrics_dir / (split + ".json")),
    ]
    if split == "val":
        command.append("--no-write-motion")
    else:
        command += ["--write-motion", str(root / "predictions" / "eval_motion")]
    return command


def _formal_eval(registry: dict[str, Any], model_id: str, root: Path, ckpt: Path, log: Path) -> None:
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        code = _command_log(_eval_command(registry, model_id, root, ckpt, split), log)
        if code != 0:
            raise RuntimeError("formal %s evaluation failed for %s; see %s" % (split, model_id, log))


def _run_model(registry: dict[str, Any], experiment_id: str, model_id: str, force: bool = False) -> Path:
    root = model_root(registry, experiment_id, model_id)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = config_path(registry, model_id)
    if snapshot.is_file():
        shutil.copyfile(snapshot, root / "config.snapshot.yaml")
    (root / "git.sha").write_text(git_sha() + "\n", encoding="utf-8")
    (root / "started_at").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    write_json(root / "effective_config.json", _merge_model_spec(registry, model_id))
    manifest = _write_initial_manifest(registry, experiment_id, model_id, root)
    log = root / "run.log"
    try:
        ckpt = _materialize_checkpoint(registry, experiment_id, model_id, root, log, force)
        (root / "checkpoint_used.txt").write_text(str(ckpt) + "\n", encoding="utf-8")
        _formal_eval(registry, model_id, root, ckpt, log)
        _update_manifest(
            manifest,
            status="complete",
            checkpoint=str(ckpt),
            checkpoint_sha256=sha256_file(ckpt),
            metrics={
                "val": str(root / "metrics" / "val.json"),
                "test": str(root / "metrics" / "test.json"),
            },
        )
        (root / "finished_at").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
        (root / ".done").write_text("complete\n", encoding="utf-8")
        return root
    except Exception as exc:
        (root / "finished_at").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
        _update_manifest(manifest, status="failed", error=str(exc))
        raise


def _rho_grid_commands(registry: dict[str, Any], experiment_id: str, model_id: str, root: Path) -> list[tuple[str, list[str]]]:
    spec = experiment_spec(registry, experiment_id)
    ckpt = checkpoint_path(registry, experiment_id, model_id)
    config = config_path(registry, model_id)
    device = str(effective_value(registry, model_id, "device", "cuda"))

    def seed_list(key: str, default: list[int]) -> list[int]:
        value = spec.get(key, default)
        if value is None or value == "":
            return default
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        return [int(value)]

    rhos = ",".join(str(value) for value in seed_list("rhos", [0, 20, 40, 60, 80, 100]))
    train_rhos = ",".join(str(value) for value in seed_list("train_repr_rhos", [0, 100]))
    val_seeds = ",".join(str(value) for value in seed_list("val_seeds", [0, 1, 2]))
    test_seeds = ",".join(str(value) for value in seed_list("test_seeds", [0]))
    commands: list[tuple[str, list[str]]] = []
    for split, seeds, grid in (
        ("train", "0", train_rhos),
        ("val", val_seeds, rhos),
        ("test", test_seeds, rhos),
    ):
        # Keep one shared repr/npz directory per model.  Train/val/test session
        # IDs are disjoint in the fixed AnySole split, while the grid metrics
        # remain split-specific JSON files in this same model directory.
        commands.append((split, [
            PYTHON,
            "-m",
            "anysole.rho_grid",
            "--ckpt",
            str(ckpt),
            "--config",
            str(config),
            "--split",
            split,
            "--seeds",
            seeds,
            "--rhos",
            grid,
            "--out-dir",
            str(root),
            "--device",
            device,
        ]))
    return commands


def _rho_grid(registry: dict[str, Any], experiment_id: str, model_id: str, root: Path, log: Path) -> None:
    for split, command in _rho_grid_commands(registry, experiment_id, model_id, root):
        code = _command_log(command, log)
        if code != 0:
            raise RuntimeError("rho grid failed for %s/%s; see %s" % (model_id, split, log))


def _rho_grid_cell(
    registry: dict[str, Any], experiment_id: str, model_id: str,
    split: str, seed: int, rho_v: int, rho_t: int, root: Path, log: Path,
) -> None:
    """Execute exactly one model + split + seed + rhoV/rhoT configuration."""
    ckpt = checkpoint_path(registry, experiment_id, model_id)
    config = config_path(registry, model_id)
    device = str(effective_value(registry, model_id, "device", "cuda"))
    command = [
        PYTHON, "-m", "anysole.rho_grid",
        "--ckpt", str(ckpt), "--config", str(config),
        "--split", split, "--seeds", str(seed),
        "--rho-v", str(rho_v), "--rho-t", str(rho_t),
        "--out-dir", str(root), "--device", device,
    ]
    code = _command_log(command, log)
    if code != 0:
        raise RuntimeError(
            "rho grid cell failed for %s/%s/%s/%s/%s (exit %d)"
            % (model_id, split, seed, rho_v, rho_t, code)
        )


def _validate_model_capabilities(registry: dict[str, Any], experiment_id: str) -> None:
    """Generation-time gating re-checked at run time (defense against
    hand-edited confs): every main model must provide the capabilities the
    experiment declares."""
    spec = experiment_spec(registry, experiment_id)
    requires = [str(value) for value in spec.get("requires", []) or []]
    if not requires:
        return
    for model_id in [str(value) for value in spec.get("models", []) or []]:
        provides = set(str(value) for value in resolved_model_spec(registry, model_id).get("provides", []) or [])
        missing = [cap for cap in requires if cap not in provides]
        if missing:
            raise ValueError(
                "experiment %s requires capabilities %s but model %s provides %s"
                % (experiment_id, ",".join(missing), model_id, ",".join(sorted(provides)))
            )


def _print_dry_run(registry: dict[str, Any], experiment_id: str, model_id: str) -> None:
    """Print the exact commands a model_run task would execute, without running them."""
    spec = _merge_model_spec(registry, model_id)
    root = model_root(registry, experiment_id, model_id)
    print("# task model_run  experiment=%s  model=%s  run_name=%s" % (experiment_id, model_id, model_run_name(registry, model_id)))
    if spec.get("train"):
        print("$ " + subprocess.list2cmdline(_train_command(registry, model_id)))
    else:
        try:
            ckpt = checkpoint_path(registry, experiment_id, model_id)
        except FileNotFoundError:
            ckpt = "<missing>"
        print("# train skipped: external checkpoint %s" % ckpt)
    for split in ("val", "test"):
        print("$ " + subprocess.list2cmdline(_eval_command(registry, model_id, root, Path("<ckpt>"), split)))
    if str(experiment_spec(registry, experiment_id).get("kind")) == "rho_grid":
        try:
            rho_commands = _rho_grid_commands(registry, experiment_id, model_id, root)
        except FileNotFoundError as exc:
            print("# rho grid skipped: %s" % exc)
        else:
            for _split, command in rho_commands:
                print("$ " + subprocess.list2cmdline(command))


def _run_task(
    registry: dict[str, Any],
    experiment_id: str,
    values: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute one generated task conf (SurvPGC one-conf-one-run granularity)."""
    # Generated confs explicitly request a fresh run. CLI --force remains
    # available for manually supplied confs and takes precedence.
    force = force or str(values.get("FORCE", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    task = str(values.get("TASK", "")).strip()
    if task == "model_run":
        model_id = str(values.get("MODEL_ID", "")).strip()
        if not model_id:
            raise ValueError("TASK=model_run without MODEL_ID")
        if dry_run:
            _print_dry_run(registry, experiment_id, model_id)
            return
        root = _run_model(registry, experiment_id, model_id, force=force)
        if str(experiment_spec(registry, experiment_id).get("kind")) == "rho_grid":
            _rho_grid(registry, experiment_id, model_id, root, root / "run.log")
    elif task == "rho_grid":
        model_id = str(values.get("MODEL_ID", "")).strip()
        split = str(values.get("GRID_SPLIT", "")).strip()
        seed = int(values.get("GRID_SEED", 0))
        rho_v = int(values.get("RHO_V", 0))
        rho_t = int(values.get("RHO_T", 0))
        if not model_id or split not in {"train", "val", "test"}:
            raise ValueError("rho_grid task requires MODEL_ID and GRID_SPLIT")
        root = model_root(registry, experiment_id, model_id)
        root.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(
                "# task rho_grid experiment=%s model=%s split=%s seed=%d rhoV=%d rhoT=%d"
                % (experiment_id, model_id, split, seed, rho_v, rho_t)
            )
            return
        _rho_grid_cell(
            registry, experiment_id, model_id, split, seed, rho_v, rho_t,
            root, root / "run.log",
        )
    elif task == "display":
        raise ValueError(
            "display tasks are not queue tasks anymore; run the matching "
            "script under results_display/script directly"
        )
    else:
        raise ValueError("Unknown TASK=%r in task conf" % task)


def run_experiment(
    registry: dict[str, Any],
    experiment_id: str,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Foreground compatibility runner for already queued data tasks.

    Normal execution is scheduler.sh.  This command deliberately does not
    discover or execute display analyses.
    """
    _validate_model_capabilities(registry, experiment_id)
    task_files = sorted((CONFIG_DIR / "queue").glob(f"*__{experiment_id}__*.conf"))
    if not task_files:
        raise FileNotFoundError(
            f"No queued tasks for {experiment_id}; run configs/z_gen/{experiment_id}.py first"
        )
    for conf_path in task_files:
        values = load_conf(conf_path)
        _run_task(registry, experiment_id, values, force=force, dry_run=dry_run)


def list_registry(registry: dict[str, Any]) -> None:
    print("Registered models (from the generator's MODELS table):")
    for model_id in registry.get("models", {}):
        print("  %-16s %s" % (model_id, model_run_name(registry, model_id)))
    from configs.z_gen.common import MODELS

    print("Master model table (configs/z_gen/common.py):")
    for model_id, spec in MODELS.items():
        provides = ",".join(spec.get("provides", ()))
        mode = "train" if spec.get("train") else "eval"
        print("  %-16s %-8s %s [%s]" % (model_id, spec["label"], mode, provides))
    print("Registered experiments:")
    for experiment_id, spec in registry.get("experiments", {}).items():
        print("  %-32s kind=%s" % (experiment_id, spec.get("kind", "?")))
    queue = CONFIG_DIR / "queue"
    print("Queued data tasks:")
    for experiment_id in registry.get("experiments", {}):
        names = sorted(path.name for path in queue.glob(f"*__{experiment_id}__*.conf"))
        print("  %-24s %s" % (experiment_id, ", ".join(names) if names else "<none>"))


def visual_slot_assignments(gpus, runs_per_gpu, max_jobs=None):
    """Stable slot-major order shared by production and resource benchmarks."""
    gpus = list(gpus)
    if (not gpus or len(set(gpus)) != len(gpus) or
            any(isinstance(g, bool) or not isinstance(g, int) or not 0 <= g <= 6 for g in gpus)):
        raise ValueError("GPU resources must be unique physical indices in 0..6")
    if isinstance(runs_per_gpu, bool) or not isinstance(runs_per_gpu, int) or runs_per_gpu < 1:
        raise ValueError("positive integer GPU concurrency required")
    if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 0):
        raise ValueError("max_jobs must be a nonnegative integer")
    slots = [(gpu, slot) for slot in range(runs_per_gpu) for gpu in gpus]
    return slots if max_jobs is None else slots[:max_jobs]


class SlotLedger:
    """One training/evaluation GPU ledger; separate stable lock inode."""
    def __init__(self, path):
        self.path = Path(path)

    @contextlib.contextmanager
    def transaction(self):
        from anysole.train import atomic_json
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.is_file() else {"schema": "anysole.slots.v1", "slots": {}}
            try:
                yield data
                atomic_json(self.path, data)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def claim(self, slots, identity, *, job_type="train", job_class=None,
              max_job_class_per_gpu=None, resource_version=None):
        """Atomically apply the total approved slot set and canonical class cap.

        Versioned formal/eval owners must pass the same resources.version. None
        only supports old unversioned controls, never mixed versioned owners.
        """
        from anysole.train import process_starttime
        aliases = {"train": "train", "pilot": "train", "capacity520": "train",
                   "n_calibration_train": "train", "eval": "eval", "infer": "eval",
                   "n_calibration_eval": "eval"}
        if job_type not in aliases or (job_class is not None and job_class != aliases[job_type]):
            raise ValueError("unknown or conflicting GPU job type/class")
        job_class = aliases[job_type]
        if resource_version is not None and (not isinstance(resource_version, str) or not resource_version.strip()):
            raise ValueError("resource_version must be a nonempty string")
        slots = list(slots)
        if not slots or any(not isinstance(item, (tuple, list)) or len(item) != 2 or
                            any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in item) or
                            item[0] > 6 for item in slots):
            raise ValueError("approved GPU slots must be physical indices and nonnegative slot indices")
        slots = [tuple(item) for item in slots]
        if len(set(slots)) != len(slots):
            raise ValueError("approved GPU slots must be unique")
        approved = {gpu: {slot for g, slot in slots if g == gpu} for gpu, _ in slots}
        if any(values != set(range(len(values))) for values in approved.values()):
            raise ValueError("pass the complete contiguous approved slot set per GPU")
        if max_job_class_per_gpu is not None and (isinstance(max_job_class_per_gpu, bool) or
                not isinstance(max_job_class_per_gpu, int) or max_job_class_per_gpu < 1 or
                any(max_job_class_per_gpu > len(values) for values in approved.values())):
            raise ValueError("job class cap must be a positive integer within total slots")
        if job_class == "eval" and max_job_class_per_gpu is None:
            raise ValueError("evaluation requires an explicit per-GPU class cap")
        with self.transaction() as data:
            live_entries = []
            for key, value in data["slots"].items():
                live = value.get("state") == "active" and value.get("starttime") is not None and process_starttime(value.get("pid", -1)) == value.get("starttime")
                if value.get("state") == "active" and not live:
                    value.update(state="stale", stale_at=time.time())
                if not live:
                    continue
                if value.get("semantic_hash") == identity:
                    raise RuntimeError("this run already owns an active GPU slot")
                gpu, slot = value.get("gpu"), value.get("slot")
                if isinstance(gpu, bool) or not isinstance(gpu, int) or isinstance(slot, bool) or not isinstance(slot, int) or key != f"{gpu}:{slot}":
                    raise RuntimeError("malformed live GPU lease; owner verification required")
                if gpu not in approved:
                    continue
                actual_class = aliases.get(value.get("job_type"))
                if actual_class is None or value.get("job_class", actual_class) != actual_class:
                    raise RuntimeError("unknown or conflicting live GPU job class")
                if (slot not in approved[gpu] or value.get("resource_version") != resource_version or
                        value.get("approved_slots") != sorted(approved[gpu]) or
                        value.get("total_slots_per_gpu") != len(approved[gpu])):
                    raise RuntimeError("incompatible live GPU resource version or total slot set")
                if (actual_class == job_class and max_job_class_per_gpu is not None and
                        value.get("job_class_limit") is not None and value["job_class_limit"] != max_job_class_per_gpu):
                    raise RuntimeError("incompatible live GPU job class cap")
                live_entries.append((gpu, actual_class))
            for gpu, slot in slots:
                if max_job_class_per_gpu is not None and sum(g == gpu and c == job_class for g, c in live_entries) >= max_job_class_per_gpu:
                    continue
                key = f"{gpu}:{slot}"
                if data["slots"].get(key, {}).get("state") == "active":
                    continue
                data["slots"][key] = {"state": "active", "gpu": gpu, "slot": slot,
                                     "pid": os.getpid(), "pgid": os.getpgrp(),
                                     "starttime": process_starttime(os.getpid()),
                                     "semantic_hash": identity, "job_type": job_type, "job_class": job_class,
                                     "job_class_limit": max_job_class_per_gpu, "resource_version": resource_version,
                                     "approved_slots": sorted(approved[gpu]), "total_slots_per_gpu": len(approved[gpu]),
                                     "owner": "view-runtime", "claimed_at": time.time()}
                return key
        return None

    def bind(self, key, identity, pid):
        from anysole.train import process_starttime
        with self.transaction() as data:
            entry = data["slots"][key]
            if entry["semantic_hash"] != identity or entry["state"] != "active":
                raise RuntimeError("slot owner changed")
            entry.update(pid=pid, starttime=process_starttime(pid), heartbeat=time.time())

    def release(self, key, identity, exit_code):
        with self.transaction() as data:
            entry = data["slots"][key]
            if entry["semantic_hash"] != identity:
                raise RuntimeError("cannot release another run's slot")
            entry.update(state="released", exit_code=int(exit_code), released_at=time.time())


def plan_hash(plan):
    from anysole.train import canonical_hash
    return canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})


def _visual_resource_reference(reference, label, *, yaml_file=False):
    """Read only an explicitly hashed absolute reference inside the workspace."""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} requires exactly path and sha256")
    raw, digest = reference["path"], reference["sha256"]
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(raw).resolve()
    if not path.is_relative_to(REPO.parent) or not path.is_file():
        raise ValueError(f"{label} file missing or outside workspace")
    if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"{label} content hash mismatch")
    try:
        if yaml_file:
            import yaml
            value = yaml.safe_load(path.read_text())
        else:
            value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"{label} malformed document") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return path, value


def _validate_visual_short_single_parent(protocol, fingerprints, config, selected_n):
    """Inherit actual old single evidence without relabeling its provenance."""
    parent = protocol.get("single_parent", {})
    _, selection = _visual_resource_reference(parent.get("selection"), "parent single selection")
    _, acceptance = _visual_resource_reference(parent.get("acceptance"), "parent single acceptance")
    _, bridge = _visual_resource_reference(parent.get("source_bridge"), "resource source bridge")
    _, amendment = _visual_resource_reference(protocol.get("user_amendment"), "user shortened-test amendment")
    if (amendment.get("schema") != "anysole.user-test-scope-amendment.v1" or
            amendment.get("approved_by") != "root" or amendment.get("authority") != "latest_explicit_user_instruction"):
        raise ValueError("short resource protocol requires explicit user amendment")
    if (selection.get("schema") != "anysole.inherited-single-selection.v1" or
            type(selection.get("new_single_measurements")) is not int or selection["new_single_measurements"] != 0 or
            type(selection.get("selected_n")) is not int or selection["selected_n"] != selected_n or
            selection.get("seven_card_throughput_comparison") is not None or
            acceptance.get("gate") != "pass" or acceptance.get("scope") != "ACTUAL_FULL_SINGLE_PARENT_ONLY" or
            acceptance.get("selected_candidate_for_short_confirmation") != selected_n):
        raise ValueError("inherited single selection/independent acceptance mismatch")
    _, old_protocol = _visual_resource_reference(selection.get("protocol"), "parent single protocol")
    _, old_authorization = _visual_resource_reference(selection.get("authorization"), "parent single authorization")
    old_config_path, old_config = _visual_resource_reference(selection.get("final_config"), "parent single config", yaml_file=True)
    old_fingerprints = old_authorization.get("input_fingerprints", {})
    if (old_protocol.get("schema") != "anysole.n-benchmark.v2" or old_protocol.get("plan_hash") != plan_hash(old_protocol) or
            old_authorization.get("scope") != "TODO090_REAL_N_ONLY" or old_authorization.get("approved_by") != "root" or
            old_authorization.get("measurement_authorized") is not True or old_authorization.get("plan_hash") != old_protocol["plan_hash"] or
            old_authorization.get("final_config_path") != str(old_config_path) or
            old_authorization.get("final_config_sha256") != selection["final_config"]["sha256"] or
            set(old_fingerprints) != set(fingerprints) or
            any(old_fingerprints[k] != v for k, v in fingerprints.items() if k != "source_hash") or
            {k:v for k,v in old_config.items() if k != "source_hash"} != {k:v for k,v in config.items() if k != "source_hash"}):
        raise ValueError("parent scientific config/input/protocol differs")
    if old_protocol.get("n_candidates") != [1,2,3] or any(type(old_protocol.get(k)) is not int or old_protocol[k] != v
            for k,v in (("single_warmup",50),("single_timed",200),("single_repeats",2))):
        raise ValueError("original single protocol workload mismatch")
    if (bridge.get("schema") != "anysole.resource-admission-source-bridge.v1" or bridge.get("gate") != "pass" or
            bridge.get("old_source_hash") != old_fingerprints.get("source_hash") or
            bridge.get("new_source_hash") != fingerprints["source_hash"] or
            bridge.get("learning_implementation_unchanged") is not True or
            bridge.get("changed_production_paths") != ["configs/tools/runner.py"] or
            bridge.get("config_changed_fields") != ["source_hash"] or
            acceptance.get("old_source_hash") != old_fingerprints["source_hash"]):
        raise ValueError("parent source requires the exact resource-admission-only bridge")
    for key in ("source_hash", *[k for k in fingerprints if k != "source_hash"]):
        if old_config.get(key) != old_fingerprints.get(key):
            raise ValueError("parent config fingerprint mismatch")
    references = selection.get("measurement_records")
    if not isinstance(references, list) or len(references) != 6:
        raise ValueError("exact six original single summaries required")
    accepted_references = acceptance.get("six_original_summary_references")
    if (not isinstance(accepted_references,list) or len(accepted_references) != 6 or
            any(not isinstance(r,dict) for r in accepted_references) or
            [{k:r.get(k) for k in ("path","sha256")} for r in accepted_references] != references or
            acceptance.get("old_config") != selection["final_config"] or
            acceptance.get("old_permit") != selection["authorization"]):
        raise ValueError("independent parent acceptance does not bind these original records")
    seen = set()
    for reference in references:
        _, row = _visual_resource_reference(reference, "original single summary")
        n, repeat = row.get("n"), row.get("repeat")
        if (type(n) is not int or n not in (1,2,3) or type(repeat) is not int or repeat not in (1,2) or
                (n,repeat) in seen or row.get("schema") != old_protocol["schema"] or row.get("phase") != "single" or
                row.get("all_pass") is not True or row.get("failure_rate") != 0 or
                row.get("memory_envelope", {}).get("passed") is not True or
                row.get("benchmark_plan_hash") != old_protocol["plan_hash"] or
                row.get("authorization_sha256") != selection["authorization"]["sha256"] or
                row.get("final_config_sha256") != selection["final_config"]["sha256"] or
                row.get("input_fingerprints") != old_fingerprints):
            raise ValueError("original single record identity/provenance/completeness mismatch")
        seen.add((n,repeat))
        workers = row.get("workers")
        if not isinstance(workers,list) or len(workers) != n:
            raise ValueError("original single resident count mismatch")
        for worker in workers:
            if (worker.get("passed") is not True or worker.get("mode") != "train" or
                    type(worker.get("physical_gpu")) is not int or worker["physical_gpu"] != old_protocol.get("single_gpu") or
                    type(worker.get("n")) is not int or worker["n"] != n or
                    any(type(worker.get(k)) is not int or worker[k] != v for k,v in
                        (("warmup_updates",50),("timed_updates",200),("total_successful_updates",250))) or worker.get("complete_val_save") is not True or
                    worker.get("input_fingerprints") != old_fingerprints or
                    worker.get("final_config_sha256") != selection["final_config"]["sha256"]):
                raise ValueError("original single successful workload incomplete")
    rows = selection.get("rows", [])
    if (not isinstance(rows,list) or len(rows) != 3 or
            {r.get("n") for r in rows} != {1,2,3} or
            any(type(r.get("n")) is not int or isinstance(r.get("forecast_seconds"),bool) or
                not isinstance(r.get("forecast_seconds"),(int,float)) or
                not math.isfinite(r["forecast_seconds"]) or r["forecast_seconds"] <= 0 or
                r.get("seven_card_rgb_per_second") is not None for r in rows)):
        raise ValueError("parent forecast rows invalid; unmeasured seven throughput must remain null")
    eligible = [r for r in rows if r.get("eligible") is True]
    if not eligible:
        raise ValueError("no stable complete parent single candidate")
    best = min(r["forecast_seconds"] for r in eligible)
    chosen = min(r["n"] for r in eligible if (r["forecast_seconds"]-best)/best < .02)
    if chosen != selected_n:
        raise ValueError("single forecast/2percent lower-n selection mismatch")


def validate_visual_resource_plan(plan):
    """Verify content and measurement provenance before any GPU/lease action.

    Hashes establish integrity, not an independent performance audit/signature.
    Actual measurement selection and formal freezing still require root review.
    This production helper never imports benchmark scripts or local .ai state.
    """
    if not isinstance(plan, dict) or plan.get("schema") != "anysole.visual-plan.v1" or plan.get("plan_hash") != plan_hash(plan):
        raise ValueError("resource plan schema/hash mismatch")
    resources = plan.get("resources", {})
    if (plan.get("frozen") is not True or plan.get("candidate_only") is True or
            not isinstance(resources, dict) or resources.get("frozen") is not True):
        raise ValueError("candidate/unmeasured resource plan cannot launch GPU work")
    n, cap, gpus = (resources.get(k) for k in ("runs_per_gpu", "eval_runs_per_gpu", "gpus"))
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("runs_per_gpu must be a measured positive integer")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= n:
        raise ValueError("evaluation concurrency must be an explicit positive integer within total slots")
    if not isinstance(gpus, list):
        raise ValueError("resource gpus must be an explicit list")
    visual_slot_assignments(gpus, n)  # Shared physical-index/uniqueness validation.
    version = resources.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("frozen resources require a nonempty version")
    _, report = _visual_resource_reference(resources.get("measurement"), "measurement")
    if (report.get("schema") != "anysole.resource-candidate.v1" or report.get("frozen") is not False or
            report.get("mixed_evaluation_passed") is not True or type(report.get("formal_task_count")) is not int or
            report.get("formal_task_count") != 15):
        raise ValueError("measurement report schema/completeness mismatch")
    for key, value in (("gpus", gpus), ("runs_per_gpu_candidate", n), ("eval_runs_per_gpu_candidate", cap)):
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError("measurement resource envelope mismatch: " + key)
    config_path, config = _visual_resource_reference(report.get("final_config"), "measurement final config", yaml_file=True)
    protocol_path, protocol = _visual_resource_reference(report.get("benchmark_plan"), "benchmark plan")
    _, authorization = _visual_resource_reference(report.get("authorization"), "measurement authorization")
    shortened = protocol.get("schema") == "anysole.n-benchmark.user-shortened.v1"
    if (protocol.get("schema") not in ("anysole.n-benchmark.v2", "anysole.n-benchmark.user-shortened.v1") or protocol.get("plan_hash") != plan_hash(protocol) or
            protocol.get("plan_hash") != report.get("benchmark_plan_hash")):
        raise ValueError("benchmark protocol schema/hash mismatch")
    candidates = protocol.get("n_candidates")
    if (not isinstance(candidates, list) or not candidates or
            any(type(value) is not int or value < 1 for value in candidates) or len(set(candidates)) != len(candidates)):
        raise ValueError("benchmark protocol candidate occupancies are invalid")
    workloads = (("single_warmup",0),("single_timed",0),("single_repeats",0),
                 ("seven_warmup",5),("seven_timed",20),("seven_repeats",1),
                 ("mixed_warmup",5),("mixed_timed",20),("mixed_repeats",1)) if shortened else (
                 ("single_warmup",50),("single_timed",200),("single_repeats",2),
                 ("seven_warmup",50),("seven_timed",50),("seven_repeats",2))
    for key, expected in workloads:
        if type(protocol.get(key)) is not int or protocol[key] != expected:
            raise ValueError("measurement protocol workload mismatch: " + key)
    if type(protocol.get("mixed_eval_per_gpu")) is not int or protocol["mixed_eval_per_gpu"] != cap:
        raise ValueError("measurement protocol evaluation cap mismatch")
    if shortened and (candidates != [n] or gpus != list(range(7)) or cap != 1 or
            type(protocol.get("seven_max_jobs")) is not int or protocol["seven_max_jobs"] != 15 or
            protocol.get("higher_n_allowed") is not False or protocol.get("eval_bootstrap_samples") != 1000):
        raise ValueError("short confirmation requires one candidate, seven GPUs,15 jobs,eval1/bootstrap1000,no higher n")
    if (authorization.get("scope") != "TODO090_REAL_N_ONLY" or authorization.get("approved_by") != "root" or
            authorization.get("measurement_authorized") is not True or authorization.get("plan_hash") != protocol["plan_hash"]):
        raise ValueError("measurement authorization does not bind this protocol")
    if shortened and any(authorization.get(key) != value for key,value in (
            ("user_amendment_path",protocol["user_amendment"]["path"]),
            ("user_amendment_sha256",protocol["user_amendment"]["sha256"]),
            ("parent_single_acceptance_sha256",protocol["single_parent"]["acceptance"]["sha256"]),
            ("source_bridge_sha256",protocol["single_parent"]["source_bridge"]["sha256"]))):
        raise ValueError("short authorization does not bind amendment/parent/source bridge")
    if (authorization.get("final_config_path") != str(config_path) or
            authorization.get("final_config_sha256") != report["final_config"]["sha256"] or
            report.get("selected_head_config_sha256") != report["final_config"]["sha256"]):
        raise ValueError("measurement final config binding mismatch")
    fingerprints = report.get("input_fingerprints")
    required = {"source_hash", "manifest_hash", "split_hash", "stats_hash", "stats_file_hash",
                "frontend_hash", "frontend_runtime_hash", "cache_index_hash", "smpl_runtime_model_hash"}
    base = plan.get("base_config", {})
    if (not isinstance(fingerprints, dict) or not required.issubset(fingerprints) or
            any(not isinstance(value, str) or not value for value in fingerprints.values()) or
            authorization.get("input_fingerprints") != fingerprints or
            any(config.get(key) != value or base.get(key) != value for key, value in fingerprints.items())):
        raise ValueError("measurement input fingerprints disagree")
    from anysole.train import visual_source_hash
    if fingerprints["source_hash"] != visual_source_hash():
        raise ValueError("measurement source differs from current production source")
    if shortened:
        _validate_visual_short_single_parent(protocol, fingerprints, config, n)
        if (isinstance(config.get("lambda_mv"),bool) or not isinstance(config.get("lambda_mv"),(int,float)) or
                not math.isfinite(config["lambda_mv"]) or config["lambda_mv"] <= 0 or
                any(type(config.get(k)) is not type(v) or config[k] != v for k,v in
                    (("mv_pose_weight",1.),("mv_pressure_weight",1.),("pair_shuffle",False),("amp",True),
                     ("num_workers",0),("effective_batch_rgb_windows",256),("microbatch_rgb_windows",8),
                     ("deterministic_algorithms",True),("save_initial_checkpoint",False)))):
            raise ValueError("short confirmation must preserve the actual heavy deterministic training contract")
    for key in ("pressure_head", "structure_hash", "visual_structure_version", "pressure_time_pe_enabled",
                "pressure_time_pe", "time_pe_scale", "d_model", "nhead", "tw", "pose_layers", "modal", "pose_repr"):
        if key in config or key in base:
            if key not in config or key not in base or type(config[key]) is not type(base[key]) or config[key] != base[key]:
                raise ValueError("measurement head/structure differs from formal base: " + key)
    if not isinstance(config.get("structure_hash"), str) or not config["structure_hash"] or not config.get("pressure_head"):
        raise ValueError("measurement selected structure is missing")
    for name, field, expected in (("TODO080", "todo080_gate", ("passed", True)),
                                  ("independent", "independent_receipt", ("gate", "pass")),
                                  ("eval cap", "eval_cap_receipt", ("scope", "PER_GPU_EVAL_CAP")),
                                  ("queue policy", "queue_policy_receipt", ("scope", "SLOT_MAJOR_15_TASK_QUEUE"))):
        reference = {"path": authorization.get(field + "_path"), "sha256": authorization.get(field + "_sha256")}
        _, receipt = _visual_resource_reference(reference, name + " receipt")
        if receipt.get(expected[0]) != expected[1] or (expected[1] is True and receipt.get(expected[0]) is not True):
            raise ValueError(name + " measurement prerequisite not passed")
        if field in ("eval_cap_receipt", "queue_policy_receipt") and (receipt.get("gate") != "pass" or receipt.get("production_source_sha256") != fingerprints["source_hash"]):
            raise ValueError(name + " receipt source/pass mismatch")
        if shortened and field == "independent_receipt" and (receipt.get("scope") != "USER_SHORTENED_N_PREPARATION" or
                receipt.get("source_hash") != fingerprints["source_hash"] or receipt.get("plan_hash") != protocol["plan_hash"]):
            raise ValueError("short independent preparation does not bind current source/protocol")
    for name in ("protocol.py", "benchmark.py", "worker.py"):
        path = (protocol_path.parent / name).resolve()
        if not path.is_relative_to(REPO.parent) or not path.is_file() or authorization.get("script_sha256", {}).get(name) != sha256_file(path):
            raise ValueError("measurement benchmark script content mismatch: " + name)
    for key in ("formal_update_horizon", "formal_validation_every", "formal_checkpoint_every", "approved_formal_run_count"):
        value = authorization.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("measurement authorization missing exact positive budget: " + key)
    if authorization["approved_formal_run_count"] != 15:
        raise ValueError("measurement authorization formal task count mismatch")
    records = report.get("measurement_records")
    if not isinstance(records, list) or not records:
        raise ValueError("measurement summary references missing")
    seen = set()
    for reference in records:
        _, summary = _visual_resource_reference(reference, "measurement summary")
        phase, repeat, occupancy = (summary.get(k) for k in ("phase", "repeat", "n"))
        if (summary.get("schema") != protocol["schema"] or phase not in (("seven","mixed") if shortened else ("single", "seven", "mixed")) or
                type(repeat) is not int or repeat not in ((1,) if shortened else (1, 2)) or type(occupancy) is not int or
                occupancy not in protocol.get("n_candidates", []) or summary.get("all_pass") is not True or
                summary.get("failure_rate") != 0 or summary.get("memory_envelope", {}).get("passed") is not True):
            raise ValueError("measurement summary incomplete or wrong phase/repeat/n")
        if (summary.get("benchmark_plan_hash") != protocol["plan_hash"] or
                summary.get("authorization_sha256") != report["authorization"]["sha256"] or
                summary.get("final_config_sha256") != report["final_config"]["sha256"] or
                summary.get("input_fingerprints") != fingerprints):
            raise ValueError("measurement summary protocol/authorization/config chain mismatch")
        identity = (phase, occupancy, repeat)
        if identity in seen:
            raise ValueError("duplicate measurement summary phase/repeat/n")
        seen.add(identity)
        if phase == "mixed" and occupancy != n:
            raise ValueError("mixed summary uses a different selected n")
        warm = protocol.get("single_warmup" if phase == "single" else "seven_warmup")
        timed = protocol.get("single_timed" if phase == "single" else "seven_timed")
        expected_gpus = [protocol.get("single_gpu")] if phase == "single" else protocol.get("seven_gpus")
        if phase != "single" and expected_gpus != gpus:
            raise ValueError("measurement protocol GPUs differ from frozen envelope")
        workers = summary.get("workers")
        if not isinstance(workers, list) or len(workers) != min(len(expected_gpus) * occupancy, 15):
            raise ValueError("measurement workload resident count incomplete")
        counts = {gpu: {"train": 0, "eval": 0} for gpu in expected_gpus}
        for worker in workers:
            gpu, mode = worker.get("physical_gpu"), worker.get("mode")
            if (worker.get("passed") is not True or type(worker.get("n")) is not int or worker["n"] != occupancy or
                    type(gpu) is not int or gpu not in counts or mode not in ("train", "eval") or
                    worker.get("input_fingerprints") != fingerprints or
                    worker.get("final_config_sha256") != report["final_config"]["sha256"]):
                raise ValueError("measurement worker provenance/completeness mismatch")
            counts[gpu][mode] += 1
            if mode == "train" and (worker.get("warmup_updates") != warm or worker.get("timed_updates") != timed or
                    worker.get("total_successful_updates") != warm + timed or worker.get("complete_val_save") is not True):
                raise ValueError("measurement train workload or full validation/save incomplete")
            if mode == "eval" and (phase != "mixed" or worker.get("optimizer_created") is not False or
                    not worker.get("full_val_metrics") or not worker.get("prediction_exports")):
                raise ValueError("measurement full evaluation incomplete")
            if shortened and mode == "train" and (worker.get("timed_sigma_unfrozen") is not True or
                    worker.get("timed_lambda_mv") != config.get("lambda_mv") or
                    any(type(worker.get("timed_pair_count_min", {}).get(key)) is not int or
                        worker["timed_pair_count_min"][key] <= 0 for key in ("local23_pair","pressure_pair"))):
                raise ValueError("short timed training must exercise live sigma and both peak-weight pair losses")
            if shortened and mode == "eval" and (worker.get("bootstrap_samples") != 1000 or worker.get("surfaces") is not True):
                raise ValueError("short evaluation still requires full surface/export and disclosed bootstrap1000")
        if any(not 1 <= sum(value.values()) <= occupancy or value["eval"] != (cap if phase == "mixed" else 0) for value in counts.values()):
            raise ValueError("measurement per-card train/eval envelope incomplete")
    if shortened:
        if seen != {("seven",n,1),("mixed",n,1)}:
            raise ValueError("short confirmation requires exactly one complete seven and mixed record")
    else:
        if any((phase, n, repeat) not in seen for phase in ("single", "seven", "mixed") for repeat in (1, 2)):
            raise ValueError("selected n requires both complete single/seven/mixed repeats")
        if any((phase, occupancy, 3-repeat) not in seen for phase, occupancy, repeat in seen):
            raise ValueError("compared candidate requires both complete repeats")
    return dict(resources)


def resolve_visual_plan_run(plan, run):
    from anysole.registry import get_builder
    config = {**plan["base_config"], **run["overrides"]}
    # These are derived architecture fields; each variant must regenerate.
    config.pop("structure_hash", None)
    config.pop("pressure_time_pe_enabled", None)
    config = get_builder("visual").effective_config(config)
    config.update(plan_hash=plan.get("plan_hash"), resource_frozen=plan["resources"].get("frozen", False),
                  runs_per_gpu=plan["resources"].get("runs_per_gpu"),
                  resource_version=plan["resources"].get("version"))
    return config


def create_visual_candidate_plan(config_path, output):
    import yaml
    from anysole.train import atomic_json, visual_run_identity
    base = yaml.safe_load(Path(config_path).read_text())
    base.update(stage="formal", plan_frozen=False)
    plan = {"schema": "anysole.visual-plan.v1", "frozen": False, "candidate_only": True,
            "base_config": base, "resources": {"gpus": list(range(7)), "runs_per_gpu": None,
                                                "frozen": False, "version": None, "eval_runs_per_gpu": None, "measurement": None},
            "packing_rule": "subset_sum_pending.v1", "runs": []}
    for experiment in ("base_legacy", "base_time", "base_pool", "view_consist",
                       "pose_pair", "press_pair", "pair_shuffle"):
        for seed in ((1, 2, 3) if experiment.startswith("base_") or experiment == "view_consist" else (1,)):
            overrides = {"experiment": experiment, "seed": seed,
                         "pressure_head": "legacy" if experiment == "base_legacy" else base["pressure_head"],
                         "lambda_mv": 0. if experiment.startswith("base_") else base.get("lambda_mv"),
                         "mv_pose_weight": 0. if experiment == "press_pair" else 1.,
                         "mv_pressure_weight": 0. if experiment == "pose_pair" else 1.,
                         "pair_shuffle": experiment == "pair_shuffle"}
            run = {"overrides": overrides}
            config = resolve_visual_plan_run(plan, run)
            identity = visual_run_identity(config)
            run.update(experiment=experiment, seed=seed, semantic_hash=identity,
                       run_dir=str(Path(base["results_root"]) / f"{experiment}_s{seed}_{identity[:10]}"))
            plan["runs"].append(run)
    plan["plan_hash"] = plan_hash(plan)
    output = Path(output)
    if output.exists():
        raise FileExistsError("candidate plan already exists")
    atomic_json(output, plan)
    return plan


def verified_visual_completion(run_dir, identity):
    """A filename or empty queue cannot establish a completed formal run."""
    from anysole.train import VIEW_CHECKPOINT_SCHEMA, visual_run_identity
    import torch
    root = Path(run_dir)
    try:
        record = json.loads((root / "run.json").read_text())
        if record["state"] != "completed" or not record["formal_complete"] or record["exit_code"] != 0 or record["semantic_hash"] != identity:
            return False
        if record["successful_updates"] != record["target_updates"] or record["rgb_windows"] != record["target_rgb_windows"]:
            return False
        if sha256_file(root / "last.pt") != record["last_checkpoint_hash"] or sha256_file(root / "best.pt") != record["best_checkpoint_hash"]:
            return False
        checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["schema_version"] != VIEW_CHECKPOINT_SCHEMA or checkpoint["semantic_hash"] != identity or visual_run_identity(checkpoint["config"]) != identity:
            return False
        if checkpoint["training"]["successful_updates"] != record["target_updates"] or checkpoint["training"]["rgb_windows"] != record["target_rgb_windows"]:
            return False
        if (best["schema_version"] != VIEW_CHECKPOINT_SCHEMA or best["semantic_hash"] != identity or
                visual_run_identity(best["config"]) != identity or
                best["config"]["structure_hash"] != checkpoint["config"]["structure_hash"]):
            return False
        if not checkpoint["selector"]["best_step"] or checkpoint["selector"]["best_step"] != best["training"]["successful_updates"]:
            return False
        # A real model reconstruction catches corrupt/missing state tensors.
        from anysole.registry import get_builder
        model = get_builder("visual").build(best["config"])
        model.load_state_dict(best["model"], strict=True)
        metrics = record["last_validation"]
        if metrics["val/coverage_hash"] != checkpoint["selector"]["coverage_hash"] or metrics["val/coverage_hash"] != best["selector"]["coverage_hash"]:
            return False
        selected = [row for row in checkpoint["selector"]["candidates"] if row["step"] == checkpoint["selector"]["best_step"]]
        return bool(len(selected) == 1 and math.isfinite(selected[0]["score"]) and
                    all(math.isfinite(float(metrics[key])) for key in ("val/mpjpe", "val/pressure_rmse")) and
                    metrics["val/motion_frames"] > 0 and metrics["val/pressure_cells"] > 0)
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, EOFError):
        return False


def check_visual_gpu_owners(gpus, ledger):
    """Refuse unknown compute owners; never kill or appropriate a process."""
    import pynvml
    from anysole.train import process_starttime
    with ledger.transaction() as data:
        allowed = {int(gpu): {entry["pid"] for entry in data["slots"].values()
                   if entry.get("gpu") == gpu and entry.get("state") == "active" and
                   process_starttime(entry.get("pid", -1)) == entry.get("starttime") and
                   entry.get("starttime") is not None} for gpu in gpus}
    pynvml.nvmlInit()
    try:
        for gpu in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            unknown = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid not in allowed[int(gpu)]]
            if unknown:
                raise RuntimeError(f"GPU {gpu} has compute PID(s) outside this task slot ledger: {unknown}")
    finally:
        pynvml.nvmlShutdown()


def visual_worker_heartbeat(root, pid, starttime, started_at, config, *, now=None, heartbeat_state=None):
    """Retain this exact child's last valid pulse across live-file read gaps."""
    state = {} if heartbeat_state is None else heartbeat_state
    identity = (str(Path(root).resolve()), pid, starttime, started_at)
    if state.get("worker_identity") != identity:
        state.clear()
        state.update(worker_identity=identity, last_valid_unix_time=None)
    try:
        pulse = json.loads((Path(root)/"heartbeat.json").read_text())
    except (OSError, ValueError):
        pulse = None
        state["read_failures"] = state.get("read_failures", 0) + 1
    # Take the default clock after read so a concurrent valid pulse is not
    # mistaken for a future timestamp merely because the read crossed a tick.
    now = time.time() if now is None else now
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        return "heartbeat_stale"
    own = (isinstance(pulse, dict) and type(pulse.get("pid")) is int and pulse["pid"] == pid
           and isinstance(pulse.get("starttime"), str) and pulse["starttime"] == starttime)
    timestamp = pulse.get("unix_time") if own else None
    valid = (not isinstance(timestamp, bool) and isinstance(timestamp, (int, float))
             and math.isfinite(timestamp) and timestamp <= now)
    if valid:
        previous = state.get("last_valid_unix_time")
        previous_valid = (not isinstance(previous, bool) and isinstance(previous, (int, float))
                          and math.isfinite(previous) and previous <= now)
        state["last_valid_unix_time"] = max(previous, float(timestamp)) if previous_valid else float(timestamp)
    elif pulse is not None:
        state["invalid_pulses"] = state.get("invalid_pulses", 0) + 1
    last = state.get("last_valid_unix_time")
    seen = (not isinstance(last, bool) and isinstance(last, (int, float))
            and math.isfinite(last) and last <= now)
    reference = last if seen else started_at
    timeout = float(config.get("heartbeat_timeout_sec", 300) if seen else config.get("worker_startup_timeout_sec", 900))
    if (isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference)
            or not math.isfinite(timeout) or timeout <= 0):
        return "heartbeat_stale"
    age = now - reference
    return "heartbeat_stale" if age > timeout else None


def run_visual_plan(path, *, dry_run=False):
    import yaml
    from anysole.train import atomic_json, visual_run_identity
    from anysole.types import validate_view_config
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan.get("schema") != "anysole.visual-plan.v1" or plan.get("plan_hash") != plan_hash(plan):
        raise ValueError("plan schema/hash mismatch")
    resources = validate_visual_resource_plan(plan) if not dry_run else None
    resolved = []
    for run in plan["runs"]:
        config = resolve_visual_plan_run(plan, run)
        identity = visual_run_identity(config)
        if identity != run["semantic_hash"]:
            raise ValueError("plan run semantic identity mismatch")
        resolved.append((run, config, identity))
    if len({identity for _, _, identity in resolved}) != len(resolved):
        raise ValueError("duplicate run identities in plan")
    if dry_run:
        print(json.dumps({"candidate_only": plan.get("candidate_only"), "frozen": plan["frozen"],
                          "run_count": len(resolved), "runs": [
                              {"identity": identity, "directory": run["run_dir"], "experiment": config["experiment"],
                               "seed": config["seed"]} for run, config, identity in resolved]}, indent=2))
        return 0
    if not plan.get("frozen") or plan.get("candidate_only") or not plan["resources"].get("frozen"):
        raise ValueError("candidate/unmeasured resource plan cannot launch training")
    n = plan["resources"].get("runs_per_gpu")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("runs_per_gpu must come from the real frozen benchmark")
    gpus = plan["resources"].get("gpus", [])
    if not gpus or len(set(gpus)) != len(gpus) or any(isinstance(g, bool) or not isinstance(g, int) or not 0 <= g <= 6 for g in gpus):
        raise ValueError("frozen GPU resources must be unique physical indices in 0..6")
    for _, config, _ in resolved:
        validate_view_config(config, formal=True)
    eval_cap = plan["resources"].get("eval_runs_per_gpu")
    version = plan["resources"].get("version")
    if isinstance(eval_cap, bool) or not isinstance(eval_cap, int) or not 1 <= eval_cap <= n:
        raise ValueError("frozen resources require an explicit evaluation concurrency within total slots")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("frozen resources require a nonempty version")
    slots = visual_slot_assignments(gpus, n)
    ledger = SlotLedger(Path(plan["base_config"]["control_dir"]) / "gpu_slots.json")
    check_visual_gpu_owners(plan["resources"]["gpus"], ledger)
    pending, active, failed = list(resolved), [], []
    status_path = Path(plan["base_config"]["control_dir"]) / "plan_status.json"
    while pending or active:
        for run, config, identity in pending[:]:
            root = Path(run["run_dir"])
            if verified_visual_completion(root, identity):
                pending.remove((run, config, identity))
                continue
            key = ledger.claim(slots, identity, job_class="train", resource_version=version)
            if key is None:
                break
            root.mkdir(parents=True, exist_ok=True)
            command = [PYTHON, "-m", "anysole.train", "--config", str(root / "config.yaml"),
                       "--run-dir", str(root)]
            record = json.loads((root / "run.json").read_text()) if (root / "run.json").is_file() else {}
            if record:
                if not (root / "last.pt").is_file():
                    ledger.release(key, identity, 1)
                    raise ValueError("existing incomplete run has no resumable checkpoint")
                if record.get("state") == "failed" and not config.get("recovery_note"):
                    ledger.release(key, identity, 1)
                    raise ValueError("failed run requires a recorded fix/recovery_note before retry")
                command += ["--resume", str(root / "last.pt")]
            config["run_dir"] = str(root)
            (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
            env = dict(os.environ)
            gpu = key.rsplit(":", 1)[0]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["TMPDIR"] = str(Path(config["control_dir"]) / "tmp")
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            log = (root / "worker.log").open("a")
            try:
                process = subprocess.Popen(command, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except Exception:
                log.close()
                ledger.release(key, identity, 1)
                raise
            ledger.bind(key, identity, process.pid)
            from anysole.train import process_starttime
            process._view_starttime = process_starttime(process.pid)
            process._view_started_at = time.time()
            process._view_config = config
            process._view_watchdog = None
            process._view_heartbeat_state = {}
            active.append((process, log, key, identity, root))
            pending.remove((run, config, identity))
        for entry in active[:]:
            process, log, key, identity, root = entry
            code = process.poll()
            if code is None:
                reason = visual_worker_heartbeat(root, process.pid, process._view_starttime,
                                                 process._view_started_at, process._view_config,
                                                 heartbeat_state=process._view_heartbeat_state)
                from anysole.train import process_starttime
                owned = process._view_starttime is not None and process_starttime(process.pid) == process._view_starttime
                if reason and process._view_watchdog is None and owned:
                    process._view_watchdog = time.time()
                    atomic_json(root/"watchdog_failure.json", {"reason": reason, "pid": process.pid,
                        "starttime": process._view_starttime, "semantic_hash": identity, "unix_time": time.time()})
                    process.terminate()  # Our exact child only; allow atomic stop boundary first.
                elif process._view_watchdog is not None and time.time()-process._view_watchdog > 60 and owned:
                    process.kill()  # Unresponsive owned worker; last.pt remains its recovery boundary.
                continue
            log.close()
            ledger.release(key, identity, code)
            if code != 0 or process._view_watchdog is not None or not verified_visual_completion(root, identity):
                failed.append({"run_dir": str(root), "semantic_hash": identity, "exit_code": code,
                               "watchdog_failed": process._view_watchdog is not None})
            active.remove(entry)
        atomic_json(status_path, {"plan_hash": plan["plan_hash"], "pending": len(pending),
                                  "active": [{"pid": p.pid, "slot": key, "identity": identity} for p, _, key, identity, _ in active],
                                  "failed": failed, "updated_at": time.time()})
        if pending or active:
            time.sleep(2)
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queued data-producing experiments")
    parser.add_argument("action", nargs="?", choices=("list", "run", "task"))
    parser.add_argument("--plan", type=Path, default=None, help="Execute only a frozen independent visual plan.")
    parser.add_argument("--make-plan", type=Path, default=None, help="Write an unlaunchable 15-run candidate plan.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("task_conf", nargs="?", default=None, help="task conf path (action=task)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without executing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.make_plan:
        if args.config is None or args.plan or args.action:
            raise ValueError("--make-plan requires --config and no legacy action")
        plan = create_visual_candidate_plan(args.config, args.make_plan)
        print(json.dumps({"path": str(args.make_plan), "runs": len(plan["runs"]), "frozen": False}))
        return 0
    if args.plan:
        if args.action:
            raise ValueError("--plan cannot consume a legacy queue action")
        return run_visual_plan(args.plan, dry_run=args.dry_run)
    if args.action is None:
        raise ValueError("specify --plan, --make-plan or a legacy action")
    if args.action == "task":
        if not args.task_conf:
            raise SystemExit("action=task requires a task conf path")
        conf_path = Path(args.task_conf)
        if not conf_path.is_file():
            raise SystemExit("task conf not found: %s" % conf_path)
        registry = load_registry()
        values = load_conf(conf_path)
        experiment_id = str(values.get("EXPERIMENT_ID", "")).strip()
        if not experiment_id:
            raise SystemExit("task conf missing EXPERIMENT_ID: %s" % conf_path)
        _run_task(registry, experiment_id, values, force=args.force, dry_run=args.dry_run)
        return 0
    registry = load_registry()
    if args.action == "list":
        list_registry(registry)
        return 0
    if not args.experiment:
        raise SystemExit("--experiment is required for %s" % args.action)
    run_experiment(registry, args.experiment, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
