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
import json
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
from anysole.types import variant_from_config


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


def _effective_variant(spec: dict[str, Any]) -> str:
    """Variant dir a train run will write to — the exact rule train.py uses
    (types.variant_from_config: fixed field order, every present field
    written, no default omission)."""
    args = dict(spec.get("train_args") or {})
    cfg = {
        "tw": int(args.get("tw", 20)),
        "stride": args.get("stride"),
        "lr": float(args.get("lr", 1e-4)),
        "lambda_pose": float(args.get("lambda_pose", 3.0)),
        "lambda_traj": float(args.get("lambda_traj", 1.0)),
        "lambda_kp": float(args.get("lambda_kp", 1.0)),
        "epochs": int(spec.get("epochs", 800)),
        "batch_size": int(spec.get("batch_size", 256)),
        "seed": int(spec.get("train_seed", spec.get("seed", 1))),
    }
    for key in ("lambda_assign", "lambda_assign_ent", "lambda_assign_conc",
                "lambda_assign_dead", "assign_dead_beta", "assign_temp_init",
                "assign_temp_final", "assign_anneal_frac", "assign_lock_frac",
                "assign_lr_mult", "lambda_sigma", "sigma_freeze_frac",
                "lr_warmup_frac", "grad_clip", "loss_cap"):
        if key in args:
            cfg[key] = float(args[key])
    # V4B: the partition K is part of the run's record (train derives it from
    # --part-json); read the file so the predicted dir matches train's.
    part_json = args.get("part_json")
    if part_json:
        path = resolve_path(str(part_json))
        if path is not None and path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            groups = doc["partition"] if isinstance(doc, dict) and "partition" in doc else doc
            if isinstance(groups, dict):
                groups = list(groups.values())
            cfg["part_joints"] = [list(group) for group in groups]
    return variant_from_config(cfg)


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
    manifest = {
        "version": 1,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_run_name": model_run_name(registry, model_id),
        "git_sha": git_sha(),
        "status": "running",
        "model_spec": spec,
        "checkpoint_source": str(source_checkpoint(registry, model_id) or ""),
        "checkpoint_source_sha256": sha256_file(source_checkpoint(registry, model_id))
        if source_checkpoint(registry, model_id)
        else None,
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
            variant = _effective_variant(spec)
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
        str(metrics_dir / (split + ".json")),
        "--protocol-out",
        str(metrics_dir / (split + "_fseries.json")),
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
                "val": str(root / "metrics" / "val_fseries.json"),
                "test": str(root / "metrics" / "test_fseries.json"),
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
            "--reuse",
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
        "--out-dir", str(root), "--device", device, "--reuse",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queued data-producing experiments")
    parser.add_argument("action", choices=("list", "run", "task"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("task_conf", nargs="?", default=None, help="task conf path (action=task)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without executing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
