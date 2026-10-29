"""Deterministic manifest/annotation preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from wham_baseline.constraints import build_annotation_index
from wham_baseline.dataset import build_inventory, write_manifest_jsonl

from .config import DATASET_ROOT, REQUIRED_ASSETS, RESULT_ROOT, UPSTREAM_ROOT
from .io import atomic_json, checked_result_root, now, sha256


def _set(values: Iterable[str] | None) -> set[str] | None:
    return None if values is None else {str(value) for value in values}


def filter_manifest(
    manifest: Mapping[str, Any],
    *,
    sessions: Iterable[str] | None = None,
    cameras: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_sessions = _set(sessions)
    selected_cameras = _set(cameras)
    units = [
        dict(unit)
        for unit in manifest.get("units", [])
        if (selected_sessions is None or str(unit["session"]) in selected_sessions)
        and (selected_cameras is None or str(unit["camera"]) in selected_cameras)
    ]
    if not units:
        raise ValueError("selection produced no units")
    for index, unit in enumerate(units):
        unit["unit_index"] = index
    result = dict(manifest)
    result["units"] = units
    result["complete"] = bool(
        manifest.get("complete") and selected_sessions is None and selected_cameras is None
    )
    selection = dict(manifest.get("selection") or {})
    selection.update(
        sessions=None if selected_sessions is None else sorted(selected_sessions),
        cameras=None if selected_cameras is None else sorted(selected_cameras),
    )
    result["selection"] = selection
    result["summary"] = {
        "unit_count": len(units),
        "frame_count": sum(int(unit["frame_count"]) for unit in units),
        "available_frame_count": sum(int(unit["available_frame_count"]) for unit in units),
        "source_image_count": sum(int(unit["source_image_count"]) for unit in units),
        "issue_count": len(result.get("issues", [])),
    }
    return result


def asset_inventory() -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name, raw_path in REQUIRED_ASSETS.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required local asset is missing: {name}: {path}")
        inventory[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return inventory


def code_hash() -> str:
    digest = hashlib.sha256()
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files += [
        UPSTREAM_ROOT / "sam_3d_body" / "models" / "backbones" / "dinov3.py",
        UPSTREAM_ROOT / "tools" / "build_fov_estimator.py",
        UPSTREAM_ROOT / "mocap" / "core" / "multiview_mhr2smpl.py",
    ]
    for path in files:
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(
    *,
    dataset_root: str | Path = DATASET_ROOT,
    result_root: str | Path = RESULT_ROOT,
    dates: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    cameras: Iterable[str] | None = None,
    max_frames_per_unit: int | None = None,
) -> dict[str, Any]:
    root = checked_result_root(result_root)
    run_root = root / ".run"
    run_root.mkdir(parents=True, exist_ok=True)
    base = build_inventory(
        dataset_root,
        dates=dates,
        methods=methods,
        max_frames_per_unit=max_frames_per_unit,
        strict=True,
    )
    manifest = filter_manifest(base, sessions=sessions, cameras=cameras)
    manifest_path = write_manifest_jsonl(run_root / "manifest.jsonl", manifest)
    selected_dates = sorted({str(unit["date"]) for unit in manifest["units"]})
    roi = build_annotation_index(dataset_root, run_root / "roi.sqlite", dates=selected_dates)
    assets = asset_inventory()
    config = {
        "schema_id": "tactile.sam3db-baseline.config",
        "schema_version": 1,
        "created_at": now(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "annotation_index": str((run_root / "roi.sqlite").resolve()),
        "annotation_index_sha256": roi["output_sha256"],
        "selection": manifest["selection"],
        "summary": manifest["summary"],
        "full_corpus": bool(
            manifest["complete"]
            and all(
                manifest["selection"].get(name) is None
                for name in ("dates", "methods", "sessions", "cameras")
            )
        ),
        "policy": {
            "local_only": True,
            "target_bbox": "date annotation supplied directly; no detector fallback",
            "inference_type": "body",
            "fov": "local MoGe2 small, level 0, fast head, no TensorRT",
            "compile": False,
            "fps": 30.0,
            "public_layout": ["result.pkl", "overlay.mp4", "state.json"],
        },
        "code_hash": code_hash(),
        "assets": assets,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    atomic_json(run_root / "config.json", config)
    ready = {
        "status": "ready",
        "at": now(),
        "config_hash": config_hash,
        "manifest_sha256": config["manifest_sha256"],
        "units": manifest["summary"]["unit_count"],
        "frames": manifest["summary"]["frame_count"],
        "full_manifest": bool(config["full_corpus"]),
    }
    atomic_json(run_root / "ready.json", ready)
    return {"manifest": manifest, "config": config, "ready": ready}
