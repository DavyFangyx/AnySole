"""Pinned workspace paths for the GVHMR 2024 adapter."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("VB_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
CODE_ROOT = WORKSPACE_ROOT / "2_Code"
UPSTREAM_ROOT = WORKSPACE_ROOT / "third_party" / "GVHMR"
ASSET_ROOT = WORKSPACE_ROOT / "models" / "gvhmr"
DEFAULT_CHECKPOINT = ASSET_ROOT / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
DEFAULT_BODY_MODELS = UPSTREAM_ROOT / "inputs" / "checkpoints" / "body_models"
DEFAULT_SMPLX_MODEL = WORKSPACE_ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_SMPL_MODEL = WORKSPACE_ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
DEFAULT_SMPLX_TO_SMPL = UPSTREAM_ROOT / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt"
DEFAULT_HMR2_CHECKPOINT = ASSET_ROOT / "checkpoints" / "hmr2" / "epoch=10-step=25000.ckpt"
DEFAULT_VITPOSE_CHECKPOINT = ASSET_ROOT / "checkpoints" / "vitpose" / "vitpose-h-multi-coco.pth"
DEFAULT_PYTHON = Path(os.environ.get("VB_GVHMR_PYTHON", "python"))
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "1_Dataset"
DEFAULT_RESULT_ROOT = WORKSPACE_ROOT / ".ai" / "tmp" / "gvhmr_baseline"


def required_assets(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    smplx_model: str | Path = DEFAULT_SMPLX_MODEL,
    smpl_model: str | Path = DEFAULT_SMPL_MODEL,
    mapper: str | Path = DEFAULT_SMPLX_TO_SMPL,
) -> dict[str, Path]:
    """Return resolved required assets, rejecting missing and LFS pointer files."""

    assets = {
        "checkpoint": Path(checkpoint).expanduser().resolve(),
        # Preserve the path's directory structure: the 2024 `smplx.create`
        # factory appends `/smplx` to the body-model root, and custom licensed
        # files are exposed via a run-local symlink at `root/smplx/...`.
        "smplx_model": Path(smplx_model).expanduser().absolute(),
        "smpl_model": Path(smpl_model).expanduser().resolve(),
        "smplx_to_smpl_mapper": Path(mapper).expanduser().resolve(),
    }
    missing = [f"{name}: {path}" for name, path in assets.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing GVHMR adapter assets: " + "; ".join(missing))
    pointers = []
    unreadable = []
    minimum_sizes = {
        "checkpoint": 10 * 1024 * 1024,
        "smplx_model": 1_000_000,
        "smpl_model": 1_000_000,
        "smplx_to_smpl_mapper": 50_000,
    }
    for name, path in assets.items():
        if path.stat().st_size < minimum_sizes[name]:
            unreadable.append(f"{name}: only {path.stat().st_size} bytes at {path}")
        if path.stat().st_size <= 256:
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeDecodeError, IndexError):
                continue
            if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                pointers.append(f"{name}: {path}")
        try:
            with path.open("rb") as handle:
                prefix = handle.read(512).lstrip().lower()
        except OSError as exc:
            unreadable.append(f"{name}: cannot read {path}: {exc}")
            continue
        if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
            unreadable.append(f"{name}: HTML/XML download response at {path}")
    if pointers:
        raise FileNotFoundError("GVHMR assets are Git LFS pointers, not model files: " + "; ".join(pointers))
    if unreadable:
        raise ValueError("GVHMR assets are too small or are not binary model files: " + "; ".join(unreadable))
    return assets


def required_preprocess_assets(
    *,
    hmr2_checkpoint: str | Path = DEFAULT_HMR2_CHECKPOINT,
    vitpose_checkpoint: str | Path = DEFAULT_VITPOSE_CHECKPOINT,
) -> dict[str, Path]:
    """Validate the official feature-extractor weights used by the demo."""

    assets = {
        "hmr2_checkpoint": Path(hmr2_checkpoint).expanduser().resolve(),
        "vitpose_checkpoint": Path(vitpose_checkpoint).expanduser().resolve(),
    }
    missing = [f"{name}: {path}" for name, path in assets.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing GVHMR preprocessing assets: " + "; ".join(missing))
    too_small = [f"{name}: {path.stat().st_size} bytes at {path}" for name, path in assets.items() if path.stat().st_size < 1_000_000]
    if too_small:
        raise ValueError("GVHMR preprocessing assets are missing/truncated: " + "; ".join(too_small))
    for name, path in assets.items():
        with path.open("rb") as handle:
            prefix = handle.read(512).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
            raise ValueError(f"{name} is an HTML/XML download response, not model weights: {path}")
    return assets
