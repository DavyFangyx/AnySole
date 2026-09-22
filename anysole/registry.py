"""Model registry + centralized address inference for the whole AnySole pipeline.

Pattern adapted from SurvPGC's dataset_deployment/registry.py
(``DATASET_CONFIGS`` + ``infer_standard_paths``): every real path derives
from a model identifier plus semantic knobs — the CLI never writes raw
paths.  train / eval / infer / ridge / tune all resolve their addresses
through :func:`infer_anysole_paths`.

The registry is the single source of truth for the warm-start lineage.
``parent`` is what the AUTO-INFERRED --init-from loads when the CLI omits
--init-from.  **A wrong ``parent`` does not error** — the run silently
warm-starts from the wrong lineage (contamination; cf. the V3-3B epoch-1
warm-start incident) — so new models must be registered with their true
parent before their first run.  An unregistered name errors loudly with the
valid list instead.

Variant rule (fixed, needs no registration): a variant dir (``tw40``,
``t0003_tw40_lr3e4``, ...) always warm-starts from ITS BASE model's
ckpt_last; only the base models' lineage lives in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anysole.types import anysole_model_dir


@dataclass(frozen=True)
class ModelEntry:
    name: str
    parent: Optional[str]  # None = from-scratch
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Historical chain (non-f2, trained, frozen).
    "F0b": ModelEntry("F0b", None, "regress from-scratch"),
    "F4a": ModelEntry("F4a", "F0b", "foot_conv"),
    "F2": ModelEntry("F2", "F0b", "f2 repr"),
    "F2p4": ModelEntry("F2p4", "F2", "f2 + foot_conv"),
    "V3_2": ModelEntry("V3_2", "F4a", "9 parts"),
    "V3_3A": ModelEntry("V3_3A", "V3_2", "soft A"),
    "V3_3B": ModelEntry("V3_3B", "V3_3A", "sigma gate"),
    # Current chain (f2 standard representation).
    "V3_4a": ModelEntry("V3_4a", "F2p4", "9 parts on f2"),
    "V3_4b": ModelEntry("V3_4b", "V3_4a", "soft A on f2"),
    "V3_4c": ModelEntry("V3_4c", "V3_4b", "sigma gate on f2 (beta-NLL fill)"),
    # Proof groups (user's V4 line, manual §V4A/V4B: both warm-start from F4a).
    "V4A": ModelEntry("V4A", "F4a", "24-slot gradient clustering"),
    "V4B": ModelEntry("V4B", "F4a", "learned-grouping hard retrain, control = V3_2"),
}


def list_registered_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_model_entry(model_name: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown model %r. Expected one of: %s" % (model_name, sorted(MODEL_REGISTRY))
        ) from exc


def infer_anysole_paths(
    model_name: str,
    *,
    variant: Optional[str] = None,
    contact: str = "joint_and",
    which: str = "last",
) -> dict:
    """Central address inference (the only place real paths are built).

    ``model_name``: registry identifier (e.g. V3_4a).  ``variant``: stacked
    hyperparameter fields under the model dir (tw40 / t0003_tw40_lr3e4),
    built automatically by train from its effective config and passed
    explicitly by eval/infer/ridge.  ``contact``: contact-label scheme in the
    dir name.  ``which``: "last" | "best" — which checkpoint to address.
    Default "last": ckpt_last is written at the end of every run, while
    ckpt_best is the val-metric-best INTERMEDIATE and can be several epochs
    older — use last for the current model, switch to best via --which best
    when producing a final report.
    """
    if which not in ("last", "best"):
        raise ValueError("which must be 'last' or 'best', got %r" % which)
    entry = get_model_entry(model_name)
    model_dir = anysole_model_dir(model_name, contact, variant)
    ckpt_name = "ckpt_last.pt" if which == "last" else "ckpt_best.pt"
    # init_from: variant dirs always warm-start from their BASE model;
    # base models follow the registered lineage (None = from-scratch).
    if variant is not None:
        base_dir = anysole_model_dir(model_name, contact)
        init_from = base_dir / "checkpoints" / "ckpt_last.pt"
    elif entry.parent is not None:
        parent_dir = anysole_model_dir(entry.parent, contact)
        init_from = parent_dir / "checkpoints" / "ckpt_last.pt"
    else:
        init_from = None
    return {
        "model_dir": model_dir,
        "ckpt": model_dir / "checkpoints" / ckpt_name,
        "init_from": init_from,
        "predictions_dir": model_dir / "predictions" / "eval_motion",
        "metrics_dir": model_dir / "metrics",
    }
