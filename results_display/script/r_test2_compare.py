"""Protocol-aware comparison evaluator (Test2) for AnySole and BVH baselines.

AnySole predicts native SMPL-24 while Step2Motion predicts Skeleton3/BVH-23.
Those arrays must never be truncated or compared by numeric joint index.  This
evaluator identifies each prediction protocol, selects an explicit 19-joint
semantic intersection, and evaluates against GT expressed in the *same native
protocol* as that prediction.  SMPL is converted to the display/common z-up
coordinate convention by ``motion_io``; BVH remains z-up.

Writes metrics to ``results_display/r_test2_compare``:
``comparison_per_session.csv`` (full detail), ``comparison_summary.csv``
(model x metrics only) and ``comparison_summary.png`` (table render of the
summary).  With ``--by-mode`` the summary is additionally split into one
table per generation mode (``by_mode/<mode>/comparison_summary.{csv,png}``)
plus a stacked ``by_mode/mode_overview.png``; rows from different modes never
share a block (a model without a resolvable mode is excluded from the mode
blocks and reported in ``evaluation.log``, never guessed).

Visualization is Test1's job and lives in
``results_display/Test1_visualization``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from utils import cli_common
from utils.compare_core import (
    COMMON_JOINTS,  # noqa: F401  (re-exported for historical importers)
    METRICS,
    MODES,
    MODE_METRICS,
    aggregate,
    array_from_file,
    find_prediction,
    load_mode_registry,
    load_split_ids,
    load_contact_gt,
    load_v2t_archive,
    metrics,
    protocol_gt,
    protocol_gt_surface,
    protocol_gt_rotations,
    read_manifest,
    resolve_mode_for,
    resolve_repo_path,
    select_common_joints,
    select_common_rotations,
    contact_from_joints,
    v2t_metrics,
    sha256,
    valid_mask,
)

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anysole.types import JOINT_NAMES  # noqa: E402  (historical import surface)
from utils.motion_io import LEGACY_BVH_NAMES, load_motion, load_rotations, load_surface  # noqa: E402


def expand_requested_models(results_root: Path, modals: list[str], contact_methods: list[str], config_ids: list[str]) -> list[dict[str, str]]:
    """Expand AnySole modal/contact-method/config combinations and retain baseline models."""
    models: list[dict[str, str]] = []
    anysole_root = results_root / "AnySole"
    for modal in modals:
        for contact_method in contact_methods:
            model_dir = cli_common.anysole_model_dir(modal, contact_method)
            root = anysole_root / model_dir
            if not root.is_dir():
                continue
            for config_id in config_ids:
                if config_id == "V2T":
                    continue
                models.append({
                    "name": f"AnySole/{model_dir}/{config_id}",
                    "variant": config_id,
                    "prediction_root": str(root / "predictions" / "eval_motion"),
                    "pattern": f"{{session_id}}_{config_id}.npz",
                    "checkpoint": str(root / "checkpoints" / "ckpt_last.pt"),
                    "modal": modal,
                    "contact_method": contact_method,
                    "config_id": config_id,
                    "mode": config_id,
                })
            if "V2T" in config_ids:
                models.append({
                    "name": f"AnySole/{model_dir}/V2T",
                    "variant": "V2T",
                    "prediction_root": str(root / "predictions" / "eval_motion"),
                    "pattern": "{session_id}_V2T.npz",
                    "checkpoint": str(root / "checkpoints" / "ckpt_last.pt"),
                    "modal": modal,
                    "contact_method": contact_method,
                    "config_id": "V2T",
                    "mode": "V2T",
                })
    # Baselines are independent models and remain in the comparison alongside AnySole.
    for model in discover_models(results_root):
        if model["name"].startswith("AnySole/"):
            continue
        models.append(model)
    return models


def discover_models(results_root: Path) -> list[dict[str, str]]:
    """Discover self-contained model result directories under results/."""
    found = []
    for model_root in sorted(p for p in results_root.rglob("*") if p.is_dir()):
        if model_root.name in {"checkpoints", "predictions", "metrics", "logs", "tensorboard"}:
            continue
        has_artifact = any((model_root / name).is_dir() for name in ("checkpoints", "predictions", "metrics"))
        if not has_artifact:
            continue
        rel = model_root.relative_to(results_root)
        name = "/".join(rel.parts)
        # Archived BVH-era results live under *_backup dirs; leave them out of
        # the automatic sweep (list them explicitly via --models-config).
        if any("backup" in part.lower() for part in rel.parts):
            continue
        prediction_root = model_root / "predictions"
        # MotionPRO/Step2Motion may keep predictions at the model family root.
        if not prediction_root.is_dir():
            prediction_root = model_root
        pattern = None
        if "AnySole" in rel.parts:
            prediction_root = model_root / "predictions" / "eval_motion"
            pattern = "{session_id}*.npz"
        checkpoint_dir = model_root / "checkpoints"
        found.append({
            "name": name,
            "variant": rel.parts[-1],
            "prediction_root": str(prediction_root),
            "pattern": pattern or "",
            "checkpoint": str(checkpoint_dir) if checkpoint_dir.is_dir() else "",
        })
    # Keep only the deepest model directories, avoiding AnySole parent duplicates.
    def is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    return [m for m in found if not any(is_relative_to(Path(m["prediction_root"]), Path(other["prediction_root"])) and m is not other for other in found)]


def registry_baselines(results_root: Path, registry: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Registry baselines as comparison rows (independent of what is on disk).

    Models without any exported prediction yet still get a row with
    status=missing inside their mode block, so the table tells the reader
    explicitly what was compared and what is not available.
    """
    rows = []
    for mode in MODES:
        for entry in registry.get(mode, []):
            name = str(entry.get("name") or "")
            if not name:
                continue
            pred_root = str(entry.get("prediction_root") or "")
            resolved = cli_common.resolve_path(pred_root, results_root) if pred_root else results_root / name / "predictions"
            rows.append({
                "name": name,
                "variant": name,
                "prediction_root": str(resolved),
                "pattern": str(entry.get("pattern") or ""),
                "checkpoint": "",
                "family": str(entry.get("family") or "baseline"),
                "mode": mode,
            })
    return rows


def _model_result_root(prediction_root: Path) -> Path:
    """Return the owning ``results/<model>`` directory for metric provenance."""
    prediction_root = Path(prediction_root)
    if prediction_root.name == "eval_motion" and prediction_root.parent.name == "predictions":
        return prediction_root.parent.parent
    if prediction_root.name == "predictions":
        return prediction_root.parent
    return prediction_root


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_model_metrics(
    model: dict[str, str],
    prediction_root: Path,
    split: str,
    split_csv: Path,
    manifest_path: Path,
    mode: str,
    family: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    surface_metrics: bool = False,
) -> Path:
    """Persist the same comparison metrics beside each model's results.

    Native model logs remain untouched.  ``<split>_comparison.json`` is the
    cross-model contract: common19 semantic joints, native-protocol GT,
    canonical metric names/units, and the exact split/manifest hashes used.
    """
    result_root = _model_result_root(prediction_root)
    output = result_root / "metrics" / f"{split}_comparison.json"
    metric_rows = []
    for row in rows:
        metric_rows.append({
            key: _json_safe(row.get(key))
            for key in ("session_id", "status", "reason", "n_valid_frames", *METRICS)
        })
    payload = {
        "schema_version": "mmvp_common_metrics_v1",
        "model": model.get("name", ""),
        "family": family,
        "mode": mode,
        "split": split,
        "split_csv": str(split_csv),
        "split_sha256": sha256(split_csv),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "joint_set": "common19",
        "protocol_rule": "native_protocol_gt_with_explicit_semantic_joint_mapping",
        "surface_metrics": bool(surface_metrics),
        "evaluation_fps": float(summary.get("eval_fps", 40.0)),
        "units": {
            "*_mm": "millimetres",
            "*_deg": "degrees",
            "root_rte_percent": "percent",
            "accel_error_m_s2": "m/s^2",
            "jitter_pred_m_s3": "m/s^3",
            "jitter_gt_m_s3": "m/s^3",
            "pressure_cop_error_*": "normalized sensor-grid units",
        },
        "summary": {
            key: _json_safe(summary.get(key))
            for key in ("n_valid_frames", *METRICS)
        },
        "n_sessions": int(sum(row.get("status") == "ok" for row in rows)),
        "sessions": metric_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def render_summary_png(csv_path: Path, png_path: Path, title: str) -> None:
    """Render the slim summary CSV as a plain table image (identity via row
    labels, neutral ink; no categorical color coding)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        print(f"[skip] no summary rows for {png_path}")
        return

    def cell(value: Any, field: str) -> str:
        if value == "" or value is None:
            return "-"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return "-"
        return f"{number:.3f}" if field in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3") else f"{number:.2f}"

    def short_model(name: str) -> str:
        return name.replace("AnySole/", "AnySole / ").replace("/", " / ")

    fields = list(rows[0].keys())
    cell_text = [[cell(row[f], f) for f in fields] for row in rows]
    table_text = [[short_model(r["model"])] + [cell(r[f], f) for f in fields[1:]] for r in rows]

    n_cols, n_rows = len(fields), len(table_text)
    fig_width = max(9.0, n_cols * 1.55 + 3.2)
    fig_height = max(2.2, n_rows * 0.42 + 1.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    ax.axis("off")
    table = ax.table(
        cellText=table_text,
        colLabels=fields,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.5)

    header_color = "#2B3A4A"
    for col_idx in range(n_cols):
        cell_obj = table[0, col_idx]
        cell_obj.set_facecolor(header_color)
        cell_obj.set_text_props(color="white", weight="bold", fontsize=9.5)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            cell_obj = table[row_idx + 1, col_idx]
            cell_obj.set_facecolor("#FFFFFF" if row_idx % 2 == 0 else "#F2F4F7")
            cell_obj.set_text_props(color="#1B2430", fontsize=9.5)
            if col_idx == 0:
                cell_obj.set_text_props(color="#1B2430", fontsize=9.5, ha="left")
                cell_obj.get_text().set_ha("left")

    ax.set_title(title, fontsize=12, pad=18, color="#1B2430")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")


def render_mode_overview_png(by_mode_csvs: dict[str, Path], png_path: Path, title: str) -> None:
    """Stack one summary table per mode into a single overview figure.

    Each block holds only same-mode rows; the mode label is the section
    header, so a cross-mode reading is impossible by construction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def short_model(name: str) -> str:
        return name.replace("AnySole/", "AnySole / ").replace("/", " / ")

    def cell(value: Any, field: str) -> str:
        if value == "" or value is None:
            return "-"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return "-"
        return f"{number:.3f}" if field in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3") else f"{number:.2f}"

    blocks = []
    for mode in MODES:
        csv_path = by_mode_csvs.get(mode)
        if csv_path is None or not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if not rows:
            continue
        fields = ["model", "family", *MODE_METRICS[mode]]
        blocks.append((mode, fields, [[short_model(r["model"]), r.get("family", "")] + [cell(r[f], f) for f in MODE_METRICS[mode]] for r in rows]))
    if not blocks:
        print(f"[skip] no mode blocks for {png_path}")
        return

    n_cols = max(len(fields) for _, fields, _ in blocks)
    fig, axes = plt.subplots(len(blocks), 1, figsize=(max(9.0, n_cols * 1.55 + 3.2), sum(0.42 * (len(rows) + 1) + 0.9 for _, _, rows in blocks) + 1.2), dpi=160)
    if len(blocks) == 1:
        axes = [axes]
    for ax, (mode, fields, table_text) in zip(axes, blocks):
        ax.axis("off")
        ax.set_title(f"{mode} — same-mode models", fontsize=12, color="#1B2430", loc="left", pad=10)
        table = ax.table(cellText=table_text, colLabels=fields, cellLoc="center", colLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.0, 1.5)
        for col_idx in range(len(fields)):
            cell_obj = table[0, col_idx]
            cell_obj.set_facecolor("#2B3A4A")
            cell_obj.set_text_props(color="white", weight="bold", fontsize=9.5)
        for row_idx in range(len(table_text)):
            for col_idx in range(len(fields)):
                cell_obj = table[row_idx + 1, col_idx]
                cell_obj.set_facecolor("#FFFFFF" if row_idx % 2 == 0 else "#F2F4F7")
                cell_obj.set_text_props(color="#1B2430", fontsize=9.5)
                if col_idx == 0:
                    cell_obj.set_text_props(color="#1B2430", fontsize=9.5, ha="left")
                    cell_obj.get_text().set_ha("left")
    fig.suptitle(title, fontsize=13, color="#1B2430", y=1.0)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "AnysoleWorkspace/manifests/session_manifest.csv")
    ap.add_argument("--split-csv", type=Path, default=cli_common.DEFAULT_SPLIT_CSV,
                    help="Canonical train/val/test membership table; manifest is metadata only.")
    ap.add_argument("--models-config", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--auto-scan", action="store_true", help="Scan results/ model directories instead of YAML")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", type=Path, default=cli_common.DISPLAY_ROOT / "ResultTest/R2Test_compare")
    ap.add_argument("--force", action="store_true", help="Rebuild and overwrite existing outputs.")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--model-name", "--modal", dest="modal", metavar="MODEL_NAME", default="anysolev1,anysolev1_insole_drift", help="AnySole model name(s), comma-separated; legacy alias: --modal")
    ap.add_argument("--contact-method", default="tactile_abs", help="Contact-label scheme(s), comma-separated; model dir is <model-name>_<contact-method>")
    ap.add_argument("--config-id", default="VT2M,V2M,T2M,V2T", help="AnySole generation task(s), comma-separated")
    ap.add_argument("--modes-config", type=Path, default=Path(__file__).resolve().parent / "models_modes.yaml",
                    help="Mode registry declaring which generation mode each baseline belongs to.")
    ap.add_argument("--by-mode", action="store_true",
                    help="Additionally write per-mode summary blocks (by_mode/<mode>/) and mode_overview.png. "
                         "Rows of different modes never share a block.")
    ap.add_argument("--write-model-metrics", action="store_true",
                    help="Also write <results>/<model>/metrics/<split>_comparison.json with the same canonical metrics.")
    ap.add_argument("--surface-metrics", action="store_true",
                    help="Compute PVE/shape_vertex_std/foot_sliding; this requires loading SMPL surfaces and is slower.")
    args = ap.parse_args()
    modals = cli_common.split_csv_arg(args.modal)
    contact_methods = cli_common.split_csv_arg(args.contact_method)
    config_ids = cli_common.split_csv_arg(args.config_id)
    allowed_configs = {"VT2M", "V2M", "T2M", "V2T"}
    invalid = set(config_ids) - allowed_configs
    if invalid:
        raise SystemExit(f"Unsupported --config-id: {sorted(invalid)}; choices={sorted(allowed_configs)}")
    registry = load_mode_registry(args.modes_config) if args.by_mode else {}
    if args.auto_scan or args.models_config is None:
        models = expand_requested_models(args.results_root, modals, contact_methods, config_ids)
        if args.by_mode:
            registered_names = {
                str(entry.get("name") or "")
                for entries in registry.values()
                for entry in entries
            }
            # The mode registry is the authoritative baseline allow-list.
            # Do not accidentally compare unrelated experiment folders such
            # as singlemodal_eval or ablation dumps found under results/.
            models = [
                model for model in models
                if model["name"].startswith("AnySole/")
                or model["name"] in registered_names
            ]
            existing = {m["name"] for m in models}
            for row in registry_baselines(args.results_root, registry):
                if row["name"] not in existing:
                    models.append(row)
        configs = {"models": models}
    else:
        try:
            import yaml
            configs = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
        except ImportError:
            configs = json.loads(args.models_config.read_text(encoding="utf-8"))
    split_csv = Path(cli_common.resolve_path(args.split_csv, ROOT))
    requested_ids = load_split_ids(split_csv, args.split)
    manifest = read_manifest(args.manifest, args.split, split_csv=split_csv, evaluable_only=True)
    if not manifest: raise SystemExit(f"No '{args.split}' sessions found in {args.manifest}")
    evaluated_ids = [row["session_id"] for row in manifest]
    excluded_ids = [sid for sid in requested_ids if sid not in set(evaluated_ids)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.out_dir / name for name in ("comparison_per_session.csv", "comparison_summary.csv", "comparison_summary.png", "evaluation.log")]
    if cli_common.outputs_ready(outputs) and not args.force:
        print(f"Skip Test2: outputs already exist under {args.out_dir} (use --force to overwrite)")
        return 0
    details, summaries, mode_resolution = [], [], []
    gt_cache: dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]] = {}
    gt_surface_cache: dict[tuple[str, str], dict | None] = {}
    gt_rotation_cache: dict[tuple[str, str], dict] = {}
    contact_cache: dict[tuple[str, str], np.ndarray | None] = {}
    for model in configs.get("models", configs):
        name, root = model["name"], resolve_repo_path(model["prediction_root"], ROOT)
        mode = resolve_mode_for(model, registry) if args.by_mode else str(model.get("mode") or "")
        family = str(model.get("family") or ("main" if name.startswith("AnySole/") else "baseline"))
        model_contact_method = str(model.get("contact_method") or (contact_methods[0] if contact_methods else "tactile_abs"))
        if "," in model_contact_method:
            model_contact_method = cli_common.split_csv_arg(model_contact_method)[0]
        model_rows = []
        for row in manifest:
            sid = row["session_id"]
            pred_path = find_prediction(root, sid, model.get("pattern"), config_id=model.get("config_id"))
            base = {"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "session_id": sid, "subject_id": row.get("subject_id", ""), "action": row.get("action", ""), "split": args.split, "protocol": "", "joint_set": "common19", "status": "ok", "reason": "", "mode": mode, "family": family, "contact_method": model_contact_method}
            try:
                if pred_path is None: raise FileNotFoundError(f"no prediction under {root}")
                if mode == "V2T" or str(model.get("config_id") or "") == "V2T":
                    v2t = load_v2t_archive(pred_path)
                    expected_frames = int(float(row.get("n_frames") or 0))
                    if len(v2t["valid_mask"]) != expected_frames:
                        raise ValueError(
                            f"unified V2T archive has {len(v2t['valid_mask'])} frames; "
                            f"canonical session has {expected_frames}"
                        )
                    vals = v2t_metrics(
                        v2t["pressure_pred"], v2t["pressure_gt"],
                        v2t["contact_pred"], v2t["contact_gt"],
                        v2t["valid_mask"],
                    )
                    base.update(vals)
                    base["protocol"] = "pressure"
                    base["joint_set"] = "per_foot_intensity"
                    base["prediction"] = str(pred_path)
                    details.append(base)
                    model_rows.append(base)
                    continue
                pred, pred_mask, pred_names, protocol = array_from_file(pred_path)
                if name.startswith("AnySole/") and pred_mask is None:
                    raise ValueError(
                        "legacy AnySole motion archive lacks the unified "
                        "joint_xyz_world/valid_mask contract; rerun anysole.eval"
                    )
                if pred_mask is not None:
                    expected_frames = int(float(row.get("n_frames") or 0))
                    if len(pred) != expected_frames or len(pred_mask) != expected_frames:
                        raise ValueError(
                            f"unified motion archive length={len(pred)} mask={len(pred_mask)}; "
                            f"canonical session={expected_frames}"
                        )
                cache_key = (sid, protocol)
                if cache_key not in gt_cache:
                    gt_cache[cache_key] = protocol_gt(row, protocol)
                gt, gt_names = gt_cache[cache_key]
                pred_surface = load_surface(pred_path) if args.surface_metrics else None
                if args.surface_metrics:
                    if cache_key not in gt_surface_cache:
                        gt_surface_cache[cache_key] = protocol_gt_surface(row, protocol)
                    gt_surface = gt_surface_cache[cache_key]
                else:
                    gt_surface = None
                pred_rotation_data = load_rotations(pred_path)
                if cache_key not in gt_rotation_cache:
                    gt_rotation_cache[cache_key] = protocol_gt_rotations(row, protocol)
                gt_rotation_data = gt_rotation_cache[cache_key]
                pred_full, gt_full = pred, gt
                floor = float(np.percentile(gt_full[..., 2], 5))
                eval_fps = float(row.get("target_fps") or args.fps)
                pred_contact = contact_from_joints(pred_full, pred_names, floor, eval_fps)
                contact_key = (sid, model_contact_method)
                if contact_key not in contact_cache:
                    contact_cache[contact_key] = load_contact_gt(
                        row, min(len(pred_full), len(gt_full)), method=model_contact_method
                    )
                gt_contact = contact_cache[contact_key]
                pred = select_common_joints(pred_full, pred_names, protocol)
                gt = select_common_joints(gt_full, gt_names, protocol)
                pred_rotations = select_common_rotations(
                    pred_rotation_data["rotations"], pred_rotation_data["names"], protocol
                )
                gt_rotations = select_common_rotations(
                    gt_rotation_data["rotations"], gt_rotation_data["names"], protocol
                )
                base["protocol"] = protocol
                n_eval = min(len(pred), len(gt), len(pred_mask) if pred_mask is not None else max(len(pred), len(gt)))
                keep = valid_mask(row, n_eval)
                if pred_mask is not None:
                    keep &= pred_mask[:n_eval].astype(bool)
                vals = metrics(
                    pred, gt, keep, eval_fps,
                    pred_rotations=pred_rotations,
                    gt_rotations=gt_rotations,
                    pred_vertices=None if pred_surface is None else pred_surface.get("vertices"),
                    gt_vertices=None if gt_surface is None else gt_surface.get("vertices"),
                    pred_contact=pred_contact,
                    gt_contact=gt_contact,
                )
                base.update(vals); base["prediction"] = str(pred_path)
            except Exception as exc:
                base.update({"status": "missing", "reason": str(exc), "n_valid_frames": 0}); base.update({k: float("nan") for k in METRICS})
            details.append(base); model_rows.append(base)
        s = aggregate(model_rows); s.update({"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "n_sessions": sum(r["status"] == "ok" for r in model_rows), "split_sha256": sha256(split_csv), "manifest_sha256": sha256(args.manifest), "eval_fps": args.fps, "checkpoint": model.get("checkpoint", ""), "mode": mode, "family": family})
        summaries.append(s)
        if args.write_model_metrics:
            write_model_metrics(
                model, root, args.split, split_csv, args.manifest, mode, family,
                model_rows, s, surface_metrics=args.surface_metrics,
            )
        if args.by_mode and not mode:
            mode_resolution.append(f"UNRESOLVED mode -> excluded from mode blocks: {name} (declare it in {args.modes_config})")
    detail_fields = ["model", "variant", "run_name", "session_id", "subject_id", "action", "split", "protocol", "joint_set", "contact_method", "n_valid_frames", *METRICS, "mode", "family", "status", "reason", "prediction"]
    summary_fields = ["model", *METRICS, "mode", "family"]
    for filename, fields, rows in (("comparison_per_session.csv", detail_fields, details), ("comparison_summary.csv", summary_fields, summaries)):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    mode_report = []
    by_mode_csvs: dict[str, Path] = {}
    if args.by_mode:
        by_mode_dir = args.out_dir / "by_mode"
        for mode in MODES:
            rows = [r for r in summaries if r.get("mode") == mode]
            mode_report.append(f"mode[{mode}]: {len(rows)} models ({', '.join(r['model'] for r in rows) or 'none'})")
            mode_fields = ["model", "family", *MODE_METRICS[mode]]
            csv_path = by_mode_dir / mode / "comparison_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=mode_fields); w.writeheader(); w.writerows({k: r.get(k, "") for k in mode_fields} for r in rows)
            render_summary_png(csv_path, by_mode_dir / mode / "comparison_summary.png", f"Test2 {mode} comparison  split={args.split}  fps={args.fps}")
            by_mode_csvs[mode] = csv_path
        render_mode_overview_png(by_mode_csvs, by_mode_dir / "mode_overview.png", f"Test2 comparison by generation mode  split={args.split}  fps={args.fps}")
    (args.out_dir / "evaluation.log").write_text(
        f"split={args.split}\nsplit_csv={split_csv}\nsplit_sha256={sha256(split_csv)}\n"
        f"requested_sessions={len(requested_ids)}\n"
        f"evaluable_sessions={len(manifest)}\n"
        f"excluded_no_valid_frames={','.join(excluded_ids) or 'none'}\n"
        f"models={len(configs.get('models', configs))}\n"
        f"modal={','.join(modals)}\ncontact_method={','.join(contact_methods)}\nconfig_id={','.join(config_ids)}\n"
        f"joint_set=common19\nprotocol_rule=native-protocol GT; explicit semantic-name mapping\n"
        f"surface_metrics={bool(args.surface_metrics)}\n"
        f"modes_config={args.modes_config if args.by_mode else 'disabled'}\n"
        + "\n".join(f"mode_report: {line}" for line in mode_report)
        + ("\n" + "\n".join(mode_resolution) if mode_resolution else "")
        + "\n" + "".join(f"checkpoint[{row['model']}]={row['checkpoint']}\n" for row in summaries),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'comparison_summary.csv'}")
    render_summary_png(
        args.out_dir / "comparison_summary.csv",
        args.out_dir / "comparison_summary.png",
        f"Test2 comparison  split={args.split}  fps={args.fps}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
