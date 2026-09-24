"""Helpers shared by the canonical R_Test display entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ``configs`` lives at the repository root; the R_TestN entry scripts put
# both this directory and the repo root on sys.path before importing here.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from configs.tools.common import (
    REPO,
    display_root,
    experiment_spec,
    load_registry,
    model_root,
)


def common_parser(
    description: str,
    default_experiment: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--experiment",
        default=default_experiment,
        required=default_experiment is None,
        help="data-producing experiment ID (defaults to the script's declared input)",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Restrict display to one or more registered model IDs; repeat the option.",
    )
    return parser


def load_context(
    experiment_id: str,
    display_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registry = load_registry()
    spec = experiment_spec(registry, experiment_id)
    out = display_root(registry, display_id or experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    return registry, spec, out


def load_display_context(
    source_experiment: str,
    display_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load one data experiment and allocate an independent display output."""
    return load_context(source_experiment, display_id)


def model_ids_for(spec: dict[str, Any], selected: list[str] | None = None) -> list[str]:
    if "models" in spec:
        model_ids = [str(value) for value in spec.get("models", [])]
        return [value for value in model_ids if selected is None or value in selected]
    if spec.get("kind") == "dropout_ablation":
        model_ids = [str(spec["main_model"]), *(str(value) for value in spec.get("compare_models", []))]
        return [value for value in model_ids if selected is None or value in selected]
    return []


def model_output(registry: dict[str, Any], experiment_id: str, model_id: str) -> Path:
    path = model_root(registry, experiment_id, model_id)
    if not path.is_dir():
        raise FileNotFoundError(
            "Experiment model output missing: %s (run the registered experiment first)" % path
        )
    return path


def dependency_model_output(
    registry: dict[str, Any],
    experiment_id: str,
    model_id: str,
    dependency_index: int = 0,
) -> Path:
    spec = experiment_spec(registry, experiment_id)
    dependencies = [str(value) for value in spec.get("depends_on", []) or []]
    if len(dependencies) <= dependency_index:
        raise ValueError("%s has no dependency at index %d" % (experiment_id, dependency_index))
    return model_output(registry, dependencies[dependency_index], model_id)
