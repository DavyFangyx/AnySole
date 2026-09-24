"""Shared registry and queue writer for the two data-producing experiments."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.z_gen.model_registry import COMMON, MODELS


CONFIG_DIR = Path(__file__).resolve().parents[1]
QUEUE_DIR = CONFIG_DIR / "queue"
STATES = ("queue", "running", "done", "failed")

# These are the only experiments that produce model/evaluation data.  Analysis
# and visualization live under results_display and are intentionally absent.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "singlemodal_eval": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B"],
        "SPECIALISTS": ["vonly", "tonly"],
        "REQUIRES": [],
    },
    "rho_grid_eval": {
        "KIND": "rho_grid",
        "MODEL_IDS": ["V3_3B"],
        "REQUIRES": [],
        "RHOS": "0,20,40,60,80,100",
        "TRAIN_REPR_RHOS": "0,100",
        "VAL_SEEDS": "0,1,2",
        "TEST_SEEDS": "0",
    },
}


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def build_registry() -> dict[str, Any]:
    defaults = {
        "config": COMMON.get("CONFIG", "anysole/configs/v1.yaml"),
        "results_root": COMMON.get("RESULTS_ROOT", "results"),
        "display_root": COMMON.get("DISPLAY_ROOT", "results_display"),
        "device": COMMON.get("DEVICE", "cuda"),
        "modal": COMMON.get("MODAL", "anysolev2"),
        "contact_method": COMMON.get("CONTACT_METHOD", "joint_and"),
        "train_seed": COMMON.get("TRAIN_SEED", 1),
        "epochs": COMMON.get("EPOCHS", 800),
        "batch_size": COMMON.get("BATCH_SIZE", 256),
        "protocol_seed": COMMON.get("PROTOCOL_SEED", 0),
    }
    registry: dict[str, Any] = {
        "version": 2,
        "defaults": defaults,
        "models": {
            model_id: {**spec, "train_args": dict(spec.get("train_args") or {})}
            for model_id, spec in MODELS.items()
        },
        "experiments": {},
    }
    for experiment_id, raw in EXPERIMENTS.items():
        requires = [str(item) for item in raw.get("REQUIRES", [])]
        models = []
        for model_id in raw["MODEL_IDS"]:
            missing = [cap for cap in requires if cap not in MODELS[model_id].get("provides", ())]
            if not missing:
                models.append(model_id)
        spec = {
            "kind": str(raw["KIND"]),
            "models": models,
            "depends_on": [],
            "display": False,
            "requires": requires,
            "specialists": [str(item) for item in raw.get("SPECIALISTS", [])],
        }
        for source, target in (
            ("RHOS", "rhos"),
            ("TRAIN_REPR_RHOS", "train_repr_rhos"),
            ("VAL_SEEDS", "val_seeds"),
            ("TEST_SEEDS", "test_seeds"),
        ):
            if source in raw:
                spec[target] = _parse_value(str(raw[source]))
        registry["experiments"][experiment_id] = spec
    return registry


def _next_sequence() -> int:
    pattern = re.compile(r"^(\d+)(?:__|_)")
    values: list[int] = []
    for state in STATES:
        directory = CONFIG_DIR / state
        if not directory.is_dir():
            continue
        for path in directory.glob("*.conf"):
            match = pattern.match(path.name)
            if match:
                values.append(int(match.group(1)))
    return max(values, default=0) + 1


def _task_specs(spec: dict[str, Any], models: list[str]) -> list[tuple[str, str | None]]:
    kind = str(spec["KIND"])
    tasks: list[tuple[str, str | None]] = []
    if kind == "singlemodal_compare":
        for model_id in models:
            tasks.append(("model_run", model_id))
            for specialist in spec.get("SPECIALISTS", []):
                tasks.append(("model_run", f"{model_id}_{specialist}"))
    elif kind == "rho_grid":
        train_rhos = [int(value) for value in str(spec["TRAIN_REPR_RHOS"]).split(",")]
        rhos = [int(value) for value in str(spec["RHOS"]).split(",")]
        val_seeds = [int(value) for value in str(spec["VAL_SEEDS"]).split(",")]
        test_seeds = [int(value) for value in str(spec["TEST_SEEDS"]).split(",")]
        for model_id in models:
            for split, split_rhos, seeds in (
                ("train", train_rhos, [0]),
                ("val", rhos, val_seeds),
                ("test", rhos, test_seeds),
            ):
                for seed in seeds:
                    for rho_v in split_rhos:
                        for rho_t in split_rhos:
                            tasks.append((
                                "rho_grid",
                                f"{model_id}|{split}|{seed}|{rho_v}|{rho_t}",
                            ))
    else:
        raise ValueError(f"unsupported data-producing experiment kind: {kind}")
    return tasks


def emit_experiment(experiment_id: str, selected_models: list[str] | None = None) -> int:
    try:
        raw = EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise SystemExit(f"unknown data-producing experiment: {experiment_id}") from exc

    selected = set(selected_models or [])
    models = [str(item) for item in raw["MODEL_IDS"] if not selected or item in selected]
    if not models:
        raise SystemExit(f"no models selected for {experiment_id}")

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    sequence = _next_sequence()
    count = 0
    for task, model_id in _task_specs(raw, models):
        task_model = model_id or "display"
        if task == "rho_grid":
            base_model, split, seed, rho_v, rho_t = task_model.split("|")
            suffix = f"{base_model}__{split}__s{seed}__rhoV{rho_v}__rhoT{rho_t}"
        else:
            suffix = task_model
        path = QUEUE_DIR / f"{sequence:05d}__{experiment_id}__{suffix}.conf"
        lines = [
            f"# Generated by configs/z_gen/{experiment_id}.py",
            f"EXPERIMENT_ID={experiment_id}",
            f"TASK={task}",
            f"MODEL_ID={task_model.split('|')[0] if task == 'rho_grid' else task_model}" if model_id else "",
            f"GRID_SPLIT={task_model.split('|')[1]}" if task == "rho_grid" else "",
            f"GRID_SEED={task_model.split('|')[2]}" if task == "rho_grid" else "",
            f"RHO_V={task_model.split('|')[3]}" if task == "rho_grid" else "",
            f"RHO_T={task_model.split('|')[4]}" if task == "rho_grid" else "",
            "FORCE=false",
        ]
        path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")
        print(f"[queue] {path}")
        sequence += 1
        count += 1
    return count
