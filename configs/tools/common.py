"""Helpers for the conf-driven AnySole data-producing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from anysole.registry import infer_anysole_paths


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parents[1]
# Kept as a compatibility constant for older callers.  New generators write
# directly to queue/ and do not create a generated task staging tree.
GENERATED_DIR = CONFIG_DIR / "generated"

# Training knobs (snake_case model-spec keys -> anysole.train CLI flags).
# The generator validates MODELS train_args against this map; runner.py
# appends them to the train command.  The flag vocabulary mirrors the
# command manual.
TRAIN_ARG_FLAG_MAP = {
    "t_encoder": "--t-encoder",
    "tactile_input": "--tactile-input",
    "v_input": "--v-input",
    "f2_repr": "--f2-repr",
    "pose_parts": "--pose-parts",
    "part_json": "--part-json",
    "soft_parts": "--soft-parts",
    "gate": "--gate",
    "assign_cluster": "--assign-cluster",
    "lambda_assign": "--lambda-assign",
    "lambda_assign_ent": "--lambda-assign-ent",
    "lambda_assign_conc": "--lambda-assign-conc",
    "lambda_assign_dead": "--lambda-assign-dead",
    "assign_dead_beta": "--assign-dead-beta",
    "assign_temp_init": "--assign-temp-init",
    "assign_temp_final": "--assign-temp-final",
    "assign_anneal_frac": "--assign-anneal-frac",
    "assign_lock_frac": "--assign-lock-frac",
    "assign_lr_mult": "--assign-lr-mult",
    "lambda_sigma": "--lambda-sigma",
    "sigma_freeze_frac": "--sigma-freeze-frac",
    "lr_warmup_frac": "--lr-warmup-frac",
    "grad_clip": "--grad-clip",
    "loss_cap": "--loss-cap",
    "tw": "--tw",
    "stride": "--stride",
    "dropout": "--dropout",
    "lr": "--lr",
    "config_probs": "--config-probs",
    "lambda_pose": "--lambda-pose",
    "lambda_kp": "--lambda-kp",
    "lambda_traj": "--lambda-traj",
    "lambda_trec": "--lambda-trec",
    "lambda_vrec": "--lambda-vrec",
    "lambda_con": "--lambda-con",
    "tactile_direct": "--tactile-direct",
    "no_imu": "--no-imu",
    "tau_max": "--tau-max",
    "tau_fixed": "--tau-fixed",
}
TRAIN_STORE_TRUE = {"f2_repr", "soft_parts", "assign_cluster", "tactile_direct", "no_imu"}


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    try:
        parts = shlex.split(value)
    except ValueError:
        parts = [value]
    if len(parts) == 1:
        value = parts[0]
    else:
        value = " ".join(parts)
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if "," in value:
        return [_parse_value(item) for item in value.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_conf(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid conf line %s:%d: %s" % (path, line_no, raw_line))
        key, raw_value = line.split("=", 1)
        # Tolerate SurvPGC-style trailing comments after values.
        raw_value = re.sub(r"\s+#.*$", "", raw_value)
        values[key.strip()] = _parse_value(raw_value)
    return values


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def build_registry() -> dict[str, Any]:
    """Build the registry for the two data-producing experiments.

    Analysis experiments are intentionally not registered here.  Their
    scripts under ``results_display/script`` consume these outputs directly.
    """
    from configs.gen.common import COMMON, EXPERIMENTS, MODELS

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
        "version": 1,
        "defaults": defaults,
        "models": {},
        "experiments": {},
    }
    for model_id, spec in MODELS.items():
        registry["models"][model_id] = {**spec, "train_args": dict(spec.get("train_args") or {})}
    for experiment_id, spec in EXPERIMENTS.items():
        requires = [str(item) for item in spec.get("REQUIRES", [])]
        model_ids = []
        for model_id in spec["MODEL_IDS"]:
            provides = MODELS[model_id].get("provides", ())
            missing = [cap for cap in requires if cap not in provides]
            if missing:
                print("[drop] %s for %s: %s missing capabilities %s" % (
                    model_id, experiment_id, MODELS[model_id]["label"], ",".join(missing)))
                continue
            model_ids.append(model_id)
        experiment: dict[str, Any] = {
            "kind": str(spec.get("KIND", "")),
            "models": model_ids,
            "depends_on": [str(item) for item in _as_list(_parse_value(str(spec.get("DEPENDS_ON", ""))))],
            # Display is a separate post-processing layer.  It never becomes
            # a queue task, even if defaults.conf contains DISPLAY=true.
            "display": False,
            "requires": requires,
            "specialists": [str(item) for item in spec.get("SPECIALISTS", [])],
        }
        for source, target in (
            ("RHOS", "rhos"),
            ("TRAIN_REPR_RHOS", "train_repr_rhos"),
            ("VAL_SEEDS", "val_seeds"),
            ("TEST_SEEDS", "test_seeds"),
            ("MAIN_MODEL", "main_model"),
            ("COMPARE_MODELS", "compare_models"),
            ("VONLY_MODEL", "vonly_model"),
            ("TONLY_MODEL", "tonly_model"),
        ):
            if source in spec:
                experiment[target] = _parse_value(str(spec[source]))
        registry["experiments"][experiment_id] = experiment
    return registry


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Backwards-compatible name: the registry comes straight from the
    generator tables (``path`` is ignored)."""
    return build_registry()


def resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def experiment_root(registry: dict[str, Any], experiment_id: str) -> Path:
    results_root = resolve_path(registry.get("defaults", {}).get("results_root", "results"))
    assert results_root is not None
    return results_root / experiment_id


def display_root(registry: dict[str, Any], experiment_id: str) -> Path:
    display = resolve_path(registry.get("defaults", {}).get("display_root", "results_display"))
    assert display is not None
    return display / experiment_id


def model_spec(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    try:
        spec = registry["models"][model_id]
    except KeyError as exc:
        raise KeyError("Unknown experiment model %r" % model_id) from exc
    return dict(spec)


def experiment_spec(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    try:
        spec = registry["experiments"][experiment_id]
    except KeyError as exc:
        raise KeyError("Unknown experiment %r" % experiment_id) from exc
    return dict(spec)


# ---------------------------------------------------------------------------
# Single-stream specialists.  A specialist is NOT an independent model: it is
# the main model trained with the modality-presence interface (--config-probs)
# changed.  The knob is dataset-side (anysole/train.py: sample_config_ids), so
# every registered model supports it; the scheduler derives the specialist
# dir <base>_<kind>_<contact> instead of registering it.
# ---------------------------------------------------------------------------

SPECIALIST_KINDS = {
    "vonly": (0.0, 1.0, 0.0),
    "tonly": (0.0, 0.0, 1.0),
    "nodrop": (1.0, 0.0, 0.0),
}


def specialist_id(base_id: str, kind: str) -> str:
    if kind not in SPECIALIST_KINDS:
        raise ValueError("Unknown specialist kind %r (allowed: %s)" % (kind, sorted(SPECIALIST_KINDS)))
    return "%s_%s" % (base_id, kind)


def split_specialist(model_id: str) -> tuple[str, str] | None:
    for kind in SPECIALIST_KINDS:
        suffix = "_" + kind
        if model_id.endswith(suffix) and len(model_id) > len(suffix):
            return model_id[: -len(suffix)], kind
    return None


def resolved_model_spec(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Model spec for a main model or a derived single-stream specialist.

    Precedence: conf MODEL_* lines > the generator's MODELS table (the master
    model table — models that no experiment conf references are still
    addressable through it).  Specialists are resolved from their base's
    spec + the config_probs override.
    """
    if model_id in registry.get("models", {}):
        return dict(model_spec(registry, model_id))
    split = split_specialist(model_id)
    if split is not None:
        base_id, kind = split
        spec = _lookup_model_spec(registry, base_id)
        train_args = dict(spec.get("train_args") or {})
        train_args["config_probs"] = SPECIALIST_KINDS[kind]
        spec["train_args"] = train_args
        spec["specialist_of"] = base_id
        spec["specialist_kind"] = kind
        # A specialist is declared by an experiment in order to be TRAINED
        # (main + --config-probs change) — never inherit the base's train
        # flag.  Its experiment materialization dir is also distinct from
        # the base's, so eval metrics of main and specialist never collide.
        spec["train"] = True
        spec["run_name"] = "%s_%s" % (str(spec.get("run_name") or base_id), kind)
        return dict(spec)
    return dict(_lookup_model_spec(registry, model_id))


def _lookup_model_spec(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    if model_id in registry.get("models", {}):
        return dict(model_spec(registry, model_id))
    from configs.gen.common import MODELS as MASTER_MODELS

    if model_id in MASTER_MODELS:
        return dict(MASTER_MODELS[model_id])
    raise KeyError("Unknown experiment model %r" % model_id)


def specialist_paths(registry: dict[str, Any], model_id: str) -> dict[str, Path] | None:
    """Addresses of a derived specialist (not registered, so built directly).

    Warm-start follows the base model's own lineage parent — the same ckpt the
    base itself started from; --config-probs is dataset-side and architecture
    is given by the base's train args.
    """
    from anysole.registry import get_model_entry
    from anysole.types import anysole_model_dir

    split = split_specialist(model_id)
    if split is None:
        return None
    base_id, _kind = split
    spec = _lookup_model_spec(registry, base_id)
    contact = str(spec.get("contact_method") or registry.get("defaults", {}).get("contact_method", "joint_and"))
    model_dir = anysole_model_dir(model_id, contact)
    entry = get_model_entry(str(spec.get("name") or base_id))
    init_from = None
    if entry.parent is not None:
        init_from = anysole_model_dir(entry.parent, contact) / "checkpoints" / "ckpt_last.pt"
    return {
        "model_dir": model_dir,
        "ckpt_best": model_dir / "checkpoints" / "ckpt_best.pt",
        "ckpt_last": model_dir / "checkpoints" / "ckpt_last.pt",
        "init_from": init_from,
    }


def registry_ckpt(registry: dict[str, Any], model_id: str, which: str | None = None) -> Path | None:
    """ckpt address inferred from the anysole registry (zero handwritten paths)."""
    spec = resolved_model_spec(registry, model_id)
    name = spec.get("name")
    if not name:
        return None
    contact = str(spec.get("contact_method") or registry.get("defaults", {}).get("contact_method", "joint_and"))
    variant = spec.get("variant")
    which = str(which or spec.get("which", "last"))
    return infer_anysole_paths(name, variant=variant, contact=contact, which=which)["ckpt"]


def model_run_name(registry: dict[str, Any], model_id: str) -> str:
    spec = resolved_model_spec(registry, model_id)
    return str(spec.get("run_name") or model_id)


def model_root(registry: dict[str, Any], experiment_id: str, model_id: str) -> Path:
    return experiment_root(registry, experiment_id) / model_run_name(registry, model_id)


def checkpoint_path(registry: dict[str, Any], experiment_id: str, model_id: str) -> Path:
    spec = resolved_model_spec(registry, model_id)
    if spec.get("name"):
        if split_specialist(model_id):
            paths = specialist_paths(registry, model_id)
            assert paths is not None
            for candidate in (paths["ckpt_best"], paths["ckpt_last"]):
                if candidate.is_file():
                    return candidate
        else:
            for which in (str(spec.get("which", "last")), "last", "best"):
                candidate = registry_ckpt(registry, model_id, which=which)
                if candidate is not None and candidate.is_file():
                    return candidate
    root = model_root(registry, experiment_id, model_id)
    for candidate in (root / "checkpoints" / "ckpt_best.pt", root / "checkpoints" / "ckpt_last.pt"):
        if candidate.is_file():
            return candidate
    external = resolve_path(spec.get("checkpoint"))
    if external is None:
        raise FileNotFoundError("No checkpoint for %s: expected registry ckpt under %s or a conf CHECKPOINT" % (model_id, root))
    return external


def source_checkpoint(registry: dict[str, Any], model_id: str) -> Path | None:
    if split_specialist(model_id):
        paths = specialist_paths(registry, model_id)
        return paths["init_from"] if paths else None
    inferred = registry_ckpt(registry, model_id)
    if inferred is not None:
        return inferred
    return resolve_path(resolved_model_spec(registry, model_id).get("checkpoint"))


def config_path(registry: dict[str, Any], model_id: str | None = None) -> Path:
    defaults = registry.get("defaults", {})
    value = defaults.get("config", "anysole/configs/v1.yaml")
    if model_id is not None:
        value = resolved_model_spec(registry, model_id).get("config", value)
    path = resolve_path(value)
    assert path is not None
    return path


def effective_value(registry: dict[str, Any], model_id: str, key: str, default: Any = None) -> Any:
    spec = model_spec(registry, model_id)
    return spec.get(key, registry.get("defaults", {}).get(key, default))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
