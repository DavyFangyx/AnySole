#!/usr/bin/env python3
"""Generate PGC-style per-model task confs for the AnySole R_Test pipeline.

SurvPGC ``configs/z_exp_gen`` counterpart: the generation scripts live here,
the scheduler internals in ``configs/tools/``.  There are no experiment-level
conf files anymore — the registry is built straight from the MODELS /
EXPERIMENTS tables below (``configs/tools/common.py: build_registry``) and the
queue items are one conf per model task (SurvPGC granularity: one conf = one
train run).

Model table: one entry per registered AnySole model (``anysole/registry.py``).
Every ckpt address is inferred from ``name`` (+ variant) by the registry —
conf files never carry handwritten CHECKPOINT / INIT_FROM paths.  ``train``
marks models the scheduler may (re)train; the warm-start lineage comes from
the registry.  ``train_args`` are the CLI knobs of the command manual, passed
verbatim by ``configs/tools/runner.py``.

Single-stream specialists (vonly / tonly / nodrop) are NOT model entries:
they are the main model trained with ``--config-probs`` changed.  The knob is
dataset-side modality sampling (``anysole/train.py`` ``sample_config_ids``),
so every model supports it; the scheduler derives the specialist
``<base>_<kind>`` on demand.  Experiments declare the specialists they need
via ``SPECIALISTS`` — MODEL_IDS always lists main models only.

Capability gating: experiments declare ``REQUIRES`` capabilities and models
declare ``provides``.  A model missing a required capability is filtered out
at generation time with a warning (e.g. only V3_3B / V3_4c provide
``gate_sigma``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tools.common import CONFIG_DIR, TRAIN_ARG_FLAG_MAP, load_conf

DEFAULT_OUT = CONFIG_DIR / "generated"
DEFAULTS_CONF = CONFIG_DIR / "defaults.conf"

# V4B loads the learned partition discovered from the F4a representations
# (manual §V4B; the file lives under F4a's metrics, not under V4B itself).
_V4B_PART_JSON = "results/AnySole/F4a_joint_and/metrics/partitions_cluster_b_K9.json"

# ---------------------------------------------------------------------------
# Model table.  ``which`` = best|last (manual default: last; V4B keeps best,
# the ckpt the first comparison layer was evaluated against).  ``epochs`` =
# the manual's measured budget, used only when the scheduler (re)trains.
# ---------------------------------------------------------------------------
MODELS = {
    # -- 历史链（非 f2 口径，已训完） --
    "F0b": dict(label="F0b", name="F0b", which="last", train=False, epochs=400,
                provides=("base",),
                train_args={"grad_clip": 5.0}),
    "F4a": dict(label="F4a", name="F4a", which="last", train=False, epochs=400,
                provides=("base", "foot_conv"),
                train_args={"t_encoder": "foot_conv", "grad_clip": 5.0}),
    "F2": dict(label="F2", name="F2", which="last", train=False, epochs=400,
               provides=("base", "f2_repr"),
               train_args={"f2_repr": True, "grad_clip": 5.0}),
    "F2p4": dict(label="F2+4", name="F2p4", which="last", train=False, epochs=400,
                 provides=("base", "foot_conv", "f2_repr"),
                 train_args={"f2_repr": True, "t_encoder": "foot_conv", "grad_clip": 5.0}),
    "V3_2": dict(label="V3-2", name="V3_2", which="last", train=False, epochs=740,
                 provides=("base", "foot_conv", "pose_parts9"),
                 train_args={"t_encoder": "foot_conv", "pose_parts": 9,
                             "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_3A": dict(label="V3-3A", name="V3_3A", which="last", train=False, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts"),
                  train_args={"t_encoder": "foot_conv", "pose_parts": 9, "soft_parts": True,
                              "lambda_assign": 0.05, "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_3B": dict(label="V3-3B", name="V3_3B", which="last", train=False, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"t_encoder": "foot_conv", "pose_parts": 9, "soft_parts": True,
                              "gate": "sigma", "lambda_assign": 0.05,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # -- 现行链（f2 标准表征）。V3-4a 尚无 ckpt：train=True，由 registry
    #    血缘自动 warm-start F2p4（manual §V3-4 第一步）。 --
    "V3_4a": dict(label="V3-4a", name="V3_4a", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9"),
                  train_args={"t_encoder": "foot_conv", "f2_repr": True, "pose_parts": 9,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4b": dict(label="V3-4b", name="V3_4b", which="last", train=False, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts"),
                  train_args={"t_encoder": "foot_conv", "f2_repr": True, "pose_parts": 9,
                              "soft_parts": True, "lambda_assign": 0.05,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4c": dict(label="V3-4c", name="V3_4c", which="last", train=False, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"t_encoder": "foot_conv", "f2_repr": True, "pose_parts": 9,
                              "soft_parts": True, "gate": "sigma",
                              "lambda_assign": 0.05, "lambda_sigma": 0.01, "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # -- 证明组（V4 线） --
    "V4A": dict(label="V4A", name="V4A", which="last", train=False, epochs=740,
                provides=("base", "foot_conv", "pose_parts24", "soft_parts", "assign_cluster"),
                train_args={"t_encoder": "foot_conv", "pose_parts": 24, "soft_parts": True,
                            "assign_cluster": True,
                            "lambda_assign": 1.0, "lambda_assign_ent": 0.05,
                            "lambda_assign_conc": 0.01, "lambda_assign_dead": 0.02,
                            "assign_dead_beta": 2.0, "assign_temp_init": 1.0,
                            "assign_temp_final": 0.2, "assign_anneal_frac": 0.7,
                            "assign_lock_frac": 0.7, "assign_lr_mult": 5.0,
                            "lr_warmup_frac": 0.05, "grad_clip": 5.0, "loss_cap": 10.0}),
    "V4B": dict(label="V4B", name="V4B", which="best", train=False, epochs=740,
                provides=("base", "foot_conv", "pose_parts9", "learned_partition"),
                train_args={"t_encoder": "foot_conv", "pose_parts": 9,
                            "part_json": _V4B_PART_JSON,
                            "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
}


EXPERIMENTS = {
    "R_Test5_rho_grid": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B"],
        "SPECIALISTS": ["vonly", "tonly"],
        "DEPENDS_ON": "",
        "REQUIRES": [],
    },
    "R_Test6_complement": {
        "KIND": "complement",
        "MODEL_IDS": ["V4B"],
        "DEPENDS_ON": "R_Test5_rho_grid",
        "REQUIRES": [],
    },
    "R_Test7_dropout_ablation": {
        "KIND": "a2_branch",
        "MODEL_IDS": ["V4B"],
        "DEPENDS_ON": "R_Test5_rho_grid",
        "REQUIRES": [],
    },
    "R_Test8_t2m_upper": {
        "KIND": "b1_v_tactile",
        "MODEL_IDS": ["V4B"],
        "DEPENDS_ON": "R_Test5_rho_grid,R_Test10_singlemodal_compare",
        "REQUIRES": [],
    },
    "R_Test9_trust": {
        "KIND": "b2_t_global",
        "MODEL_IDS": ["V4B"],
        "DEPENDS_ON": "R_Test5_rho_grid,R_Test10_singlemodal_compare",
        "REQUIRES": [],
    },
    "R_Test10_singlemodal_compare": {
        "KIND": "rho_grid",
        "MODEL_IDS": ["V3_3B"],
        "DEPENDS_ON": "R_Test5_rho_grid",
        "REQUIRES": [],
        "RHOS": "0,20,40,60,80,100",
        "TRAIN_REPR_RHOS": "0,100",
        "VAL_SEEDS": "0,1,2",
        "TEST_SEEDS": "0",
    },
}


# Shared defaults.  configs/defaults.conf is the single editable source
# (SurvPGC defaults.conf role); the fallback below only fills keys the file
# does not define.
_COMMON_FALLBACK = {
    "CONFIG": "anysole/configs/v1.yaml",
    "RESULTS_ROOT": "results",
    "DISPLAY_ROOT": "results_display",
    "MODAL": "anysolev2",
    "CONTACT_METHOD": "joint_and",
    "DEVICE": "cuda",
    "TRAIN_SEED": "1",
    "EPOCHS": "800",
    "BATCH_SIZE": "256",
    "PROTOCOL_SEED": "0",
    "DISPLAY": "true",
}
COMMON = {
    **_COMMON_FALLBACK,
    **{key: value for key, value in load_conf(DEFAULTS_CONF).items() if key in _COMMON_FALLBACK},
}


def _validate_train_args() -> None:
    unknown = sorted(
        {key for spec in MODELS.values() for key in (spec.get("train_args") or {})}
        - set(TRAIN_ARG_FLAG_MAP)
    )
    if unknown:
        raise SystemExit("Models use unregistered train args: %s" % ",".join(unknown))


_validate_train_args()


# Experiment kinds whose whole work is analysis over dependency outputs —
# they only get a display task, no model runs.
ANALYSIS_KINDS = {"complement", "a2_branch", "b1_v_tactile", "b2_t_global"}


def _topo_order(experiment_ids: list[str]) -> list[str]:
    """Experiments in execution order (DEPENDS_ON first)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in seen:
            return
        seen.add(experiment_id)
        raw = EXPERIMENTS.get(experiment_id, {}).get("DEPENDS_ON", "") or ""
        for dependency in [item.strip() for item in raw.split(",") if item.strip()]:
            visit(dependency)
        order.append(experiment_id)

    for experiment_id in experiment_ids:
        visit(experiment_id)
    return order


def _task_specs(merged: dict) -> list[tuple[str | None, str]]:
    """(model_id, task) pairs for one experiment — one queue item per model.

    ``model_run`` covers train (if needed) + formal val/test eval; rho_grid
    experiments resolve the grid from the experiment spec at run time, so the
    task conf stays thin.  Analysis-only kinds get a single display task.
    """
    kind = str(merged["KIND"])
    models = [str(value) for value in merged.get("MODEL_IDS", [])]
    raw_specialists = merged.get("SPECIALISTS", "") or ""
    if isinstance(raw_specialists, (list, tuple)):
        specialists = [str(item).strip() for item in raw_specialists]
    else:
        specialists = [item.strip() for item in str(raw_specialists).split(",") if item.strip()]
    tasks: list[tuple[str | None, str]] = []
    if kind == "singlemodal_compare":
        for model_id in models:
            tasks.append((model_id, "model_run"))
            for specialist in specialists:
                tasks.append(("%s_%s" % (model_id, specialist), "model_run"))
    elif kind == "rho_grid":
        tasks.extend((model_id, "model_run") for model_id in models)
    elif kind not in ANALYSIS_KINDS:
        # dropout_ablation and any future kind: one model_run per model.
        tasks.extend((model_id, "model_run") for model_id in models)
    if bool(merged.get("DISPLAY", True)):
        tasks.append((None, "display"))
    return tasks


def _write_task_conf(path: Path, experiment_id: str, task: str, model_id: str | None) -> None:
    lines = [
        "# Generated by configs/z_exp_gen/generate_r_test_configs.py",
        "EXPERIMENT_ID=%s" % experiment_id,
        "TASK=%s" % task,
    ]
    if model_id is not None:
        lines.append("MODEL_ID=%s" % model_id)
    lines.append("FORCE=false")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_tasks(per_experiment_models: dict[str, list[str]], out_root: Path, queue: bool) -> int:
    """Per-model task confs, SurvPGC one-conf-one-run granularity.

    Staged under ``out_root/<EXPERIMENT_ID>/`` by default, or written straight
    into the shared queue with ``--queue``.  The zero-padded sequence keeps
    the scheduler's lexicographic pick order equal to execution order.
    """
    count = 0
    sequence = 0
    for experiment_id in _topo_order(list(per_experiment_models)):
        spec = dict(EXPERIMENTS[experiment_id])
        spec["MODEL_IDS"] = per_experiment_models[experiment_id]
        merged = dict(COMMON)
        merged.update(spec)
        for model_id, task in _task_specs(merged):
            sequence += 1
            name = "%03d__%s__%s.conf" % (sequence, experiment_id, model_id if model_id is not None else "display")
            target = (
                CONFIG_DIR / "queue" / name
                if queue
                else out_root / experiment_id / name
            )
            _write_task_conf(target, experiment_id, task, model_id)
            print("[task] %s" % target)
            count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate AnySole R_Test task .conf files")
    parser.add_argument("--experiments", default="all", help="all or comma-separated experiment IDs")
    parser.add_argument("--models", default="", help="optional comma-separated model IDs to restrict generation")
    parser.add_argument("--queue", action="store_true", help="write task confs directly into configs/queue/ instead of staging under out-dir")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    experiment_ids = list(EXPERIMENTS) if args.experiments == "all" else [item.strip() for item in args.experiments.split(",") if item.strip()]
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    per_experiment_models: dict[str, list[str]] = {}
    for experiment_id in experiment_ids:
        if experiment_id not in EXPERIMENTS:
            raise SystemExit("Unknown experiment: %s" % experiment_id)
        spec = EXPERIMENTS[experiment_id]
        requires = [str(item) for item in spec.get("REQUIRES", [])]
        model_ids = []
        for model_id in spec["MODEL_IDS"]:
            if selected and model_id not in selected:
                continue
            provides = MODELS[model_id].get("provides", ())
            missing = [cap for cap in requires if cap not in provides]
            if missing:
                print("[drop] %s for %s: %s missing capabilities %s" % (
                    model_id, experiment_id, MODELS[model_id]["label"], ",".join(missing)))
                continue
            model_ids.append(model_id)
        if not model_ids:
            raise SystemExit("No models remain for %s (REQUIRES=%s)" % (experiment_id, ",".join(requires)))
        per_experiment_models[experiment_id] = model_ids
    target = CONFIG_DIR / "queue" if args.queue else args.out_dir / "tasks"
    task_count = _emit_tasks(per_experiment_models, target, args.queue)
    print("Generated %d task conf(s) under %s" % (task_count, target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
