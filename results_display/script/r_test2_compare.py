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

from utils import cli_common
from utils.compare_core import (
    COMMON_JOINTS,  # noqa: F401  (re-exported for historical importers)
    METRICS,
    MODES,
    aggregate,
    array_from_file,
    find_prediction,
    load_mode_registry,
    metrics,
    protocol_gt,
    read_manifest,
    resolve_mode_for,
    resolve_repo_path,
    select_common_joints,
    sha256,
    valid_mask,
)

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anysole.types import JOINT_NAMES  # noqa: E402  (historical import surface)
from utils.motion_io import LEGACY_BVH_NAMES, load_motion  # noqa: E402


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
                models.append({
                    "name": f"AnySole/{model_dir}/{config_id}",
                    "variant": config_id,
                    "prediction_root": str(root / "predictions" / "eval_motion"),
                    "pattern": f"{{session_id}}_{config_id}.npz",
                    "checkpoint": str(root / "checkpoints" / "ckpt_last.pt"),
                    "modal": modal,
                    "contact_method": contact_method,
                    "config_id": config_id,
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
        found.append({"name": name, "variant": rel.parts[-1], "prediction_root": str(prediction_root), "pattern": pattern or "", "checkpoint": str(model_root / "checkpoints")})
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
        return f"{number:.3f}" if field in ("Accel_mps2", "Jitter_1e-3_mps2") else f"{number:.2f}"

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
        return f"{number:.3f}" if field in ("Accel_mps2", "Jitter_1e-3_mps2") else f"{number:.2f}"

    blocks = []
    for mode in MODES:
        csv_path = by_mode_csvs.get(mode)
        if csv_path is None or not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if not rows:
            continue
        fields = ["model", "family", *METRICS]
        blocks.append((mode, fields, [[short_model(r["model"]), r.get("family", "")] + [cell(r[f], f) for f in METRICS] for r in rows]))
    if not blocks:
        print(f"[skip] no mode blocks for {png_path}")
        return

    n_cols = len(blocks[0][1])
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
    ap.add_argument("--models-config", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--auto-scan", action="store_true", help="Scan results/ model directories instead of YAML")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", type=Path, default=cli_common.DISPLAY_ROOT / "result/r_test2_compare")
    ap.add_argument("--force", action="store_true", help="Rebuild and overwrite existing outputs.")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--model-name", "--modal", dest="modal", metavar="MODEL_NAME", default="anysolev1,anysolev1_insole_drift", help="AnySole model name(s), comma-separated; legacy alias: --modal")
    ap.add_argument("--contact-method", default="tactile_abs", help="Contact-label scheme(s), comma-separated; model dir is <model-name>_<contact-method>")
    ap.add_argument("--config-id", default="VT2M,V2M,T2M", help="AnySole generation configuration(s), comma-separated")
    ap.add_argument("--modes-config", type=Path, default=Path(__file__).resolve().parent / "models_modes.yaml",
                    help="Mode registry declaring which generation mode each baseline belongs to.")
    ap.add_argument("--by-mode", action="store_true",
                    help="Additionally write per-mode summary blocks (by_mode/<mode>/) and mode_overview.png. "
                         "Rows of different modes never share a block.")
    args = ap.parse_args()
    modals = cli_common.split_csv_arg(args.modal)
    contact_methods = cli_common.split_csv_arg(args.contact_method)
    config_ids = cli_common.split_csv_arg(args.config_id)
    allowed_configs = {"VT2M", "V2M", "T2M"}
    invalid = set(config_ids) - allowed_configs
    if invalid:
        raise SystemExit(f"Unsupported --config-id: {sorted(invalid)}; choices={sorted(allowed_configs)}")
    registry = load_mode_registry(args.modes_config) if args.by_mode else {}
    if args.auto_scan or args.models_config is None:
        models = expand_requested_models(args.results_root, modals, contact_methods, config_ids)
        if args.by_mode:
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
    manifest = read_manifest(args.manifest, args.split)
    if not manifest: raise SystemExit(f"No '{args.split}' sessions found in {args.manifest}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.out_dir / name for name in ("comparison_per_session.csv", "comparison_summary.csv", "comparison_summary.png", "evaluation.log")]
    if cli_common.outputs_ready(outputs) and not args.force:
        print(f"Skip Test2: outputs already exist under {args.out_dir} (use --force to overwrite)")
        return 0
    details, summaries, mode_resolution = [], [], []
    for model in configs.get("models", configs):
        name, root = model["name"], resolve_repo_path(model["prediction_root"], ROOT)
        mode = resolve_mode_for(model, registry) if args.by_mode else str(model.get("mode") or "")
        family = str(model.get("family") or ("main" if name.startswith("AnySole/") else "baseline"))
        model_rows = []
        for row in manifest:
            sid = row["session_id"]
            pred_path = find_prediction(root, sid, model.get("pattern"), config_id=model.get("config_id"))
            base = {"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "session_id": sid, "subject_id": row.get("subject_id", ""), "action": row.get("action", ""), "split": args.split, "protocol": "", "joint_set": "common19", "status": "ok", "reason": "", "mode": mode, "family": family}
            try:
                if pred_path is None: raise FileNotFoundError(f"no prediction under {root}")
                pred, pred_mask, pred_names, protocol = array_from_file(pred_path)
                gt, gt_names = protocol_gt(row, protocol)
                pred = select_common_joints(pred, pred_names, protocol)
                gt = select_common_joints(gt, gt_names, protocol)
                base["protocol"] = protocol
                n_eval = min(len(pred), len(gt), len(pred_mask) if pred_mask is not None else max(len(pred), len(gt)))
                keep = valid_mask(row, n_eval)
                if pred_mask is not None:
                    keep &= pred_mask[:n_eval].astype(bool)
                vals = metrics(pred, gt, keep, args.fps); base.update(vals); base["prediction"] = str(pred_path)
            except Exception as exc:
                base.update({"status": "missing", "reason": str(exc), "n_valid_frames": 0}); base.update({k: float("nan") for k in METRICS})
            details.append(base); model_rows.append(base)
        s = aggregate(model_rows); s.update({"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "n_sessions": sum(r["status"] == "ok" for r in model_rows), "split_sha256": sha256(args.manifest), "manifest_sha256": sha256(args.manifest), "eval_fps": args.fps, "checkpoint": model.get("checkpoint", ""), "mode": mode, "family": family})
        summaries.append(s)
        if args.by_mode and not mode:
            mode_resolution.append(f"UNRESOLVED mode -> excluded from mode blocks: {name} (declare it in {args.modes_config})")
    detail_fields = ["model", "variant", "run_name", "session_id", "subject_id", "action", "split", "protocol", "joint_set", "n_valid_frames", *METRICS, "mode", "family", "status", "reason", "prediction"]
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
            csv_path = by_mode_dir / mode / "comparison_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader(); w.writerows({k: r.get(k, "") for k in summary_fields} for r in rows)
            render_summary_png(csv_path, by_mode_dir / mode / "comparison_summary.png", f"Test2 {mode} comparison  split={args.split}  fps={args.fps}")
            by_mode_csvs[mode] = csv_path
        render_mode_overview_png(by_mode_csvs, by_mode_dir / "mode_overview.png", f"Test2 comparison by generation mode  split={args.split}  fps={args.fps}")
    (args.out_dir / "evaluation.log").write_text(
        f"split={args.split}\nsessions={len(manifest)}\nmodels={len(configs.get('models', configs))}\n"
        f"modal={','.join(modals)}\ncontact_method={','.join(contact_methods)}\nconfig_id={','.join(config_ids)}\n"
        f"joint_set=common19\nprotocol_rule=native-protocol GT; explicit semantic-name mapping\n"
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
