"""Protocol-aware comparison evaluator (R_Test2), native-protocol edition.

评估整改任务 03：删除 common19 作为跨模型数值协议。每个模型按其原生协议
评估——SMPL-24 预测对 SMPL GT、BVH-23 预测对 BVH GT，绝不按数值关节下标对齐
或裁剪成公共关节集。指标适用性由注册表声明的真实输出能力
（``models_modes.yaml`` 的 ``protocol``/``capabilities``/``sources``）加实际
预测文件内容门控：

    missing         预测文件不存在
    invalid         声明能力与文件矛盾（协议不符 / 缺统一契约 / 帧数不符 /
                    声明为真但必要字段缺失）
    not_applicable  模型没有该能力（或来源为 GT 回填/模板重建）——对应指标
                    展示为 ``—``，绝不写 0

contact 不进入任何正式指标集（无 motion 模型训练并显式导出 contact prediction）；
V2T 为三层压力层级（网格/力/CoP），表列只显示 brief 6 键，叶 4 键
（T_mse/T_mae/pressure_force_mae/pressure_cop_error_mean）保留在逐会话明细。
指标公式一律来自 ``utils.compare_core`` / 根 ``metrics.py``
（任务 01 唯一实现），本脚本不再定义任何公式。

输出（``results_display/ResultTest/R2Test_compare/``）：
``comparison_per_session.csv``（逐会话明细，含 protocol/gt_source/metric_reasons）、
``comparison_summary.csv``（模型 × 指标，含 protocol 列）、``comparison_summary.png``、
``evaluation.log``。``--by-mode`` 另写 ``by_mode/<mode>/`` 分块与
``mode_overview.png``；``--write-model-metrics`` 把同一套指标写入
``<model>/metrics/<split>_comparison.json``（schema ``mmvp_native_metrics_v2``）。
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
    CAPABILITY_ALIASES,
    METRICS,
    METRIC_REQUIRES,
    MODES,
    MODE_METRICS,
    PROTOCOL_LABELS,
    aggregate,
    applicable_metrics,
    array_from_file,
    bvh_joints,
    find_prediction,
    load_mode_registry,
    load_split_ids,
    load_v2t_archive,
    metrics,
    normalize_protocol,
    protocol_gt,
    protocol_gt_surface,
    protocol_gt_rotations,
    read_capabilities,
    read_manifest,
    resolve_mode_for,
    resolve_repo_path,
    sha256,
    valid_mask,
    v2t_metrics,
)

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.motion_io import load_rotations, load_surface  # noqa: E402

# Cross-model comparison file schema.  v1 (common19) is superseded and must
# never be read back as a current result; the writer only emits v2.
COMPARISON_SCHEMA_V2 = "mmvp_native_metrics_v2"
LEGACY_SCHEMA_V1 = "mmvp_common_metrics_v1"

# Capability sources allowed to feed public metrics (评估整改 00 §5):
# model_prediction / method_optimization and their deterministic derivations.
OK_SOURCES = ("model_prediction", "method_optimization")
# Source labels are retained for audit only. They do not override the
# generation-capability contract used for metric applicability.
DENIED_SOURCES = (
    "ground_truth_fallback", "template_reconstruction", "method_fitting",
    "derived_for_visualization",
)

# AnySole 主模型能力：模型直接输出 SMPL pose（= 关节旋转）、root
# translation、root rotation；不输出 shape/表面/contact。V2T 行额外输出
# pressure。GT beta 不构成 predicted shape/surface。
ANYSOLE_MOTION_CAPS = {
    "joint_positions": True,
    "direct_joint_rotations": True,
    "root_translation": True,
    "root_rotation": True,
    "predicted_surface": False,
    "predicted_shape": False,
    "pressure_prediction": False,
    "contact_prediction": False,
}
ANYSOLE_V2T_CAPS = {**ANYSOLE_MOTION_CAPS, "pressure_prediction": True}
# 评估资格以各模型生成清单为唯一依据（2026-09-27 用户裁定，见
# z_note/metrics_指标改动说明.md §2）：AnySole 不生成 shape/表面，PVE 一律
# 不适用（此前 pose-only + GT beta 的批准例外已废除）；pressure_toolkit 的
# 拟合表面属其 pipeline 原生生成，PVE 可评估。
ANYSOLE_APPROVED = ()

# Root-orientation forward axis per protocol (both pred and GT rotations of
# one protocol live in the same frame, so the two must match):
#   smpl24: rotations are native SMPL (y-up, body forward +Z)        -> axis 2
#   bvh23:  rotations are the z-up display frame, body forward -Y
#           (verified 2026-09-26 against SMPL GT heading on S13013:
#           median circular diff 4.4 deg; the common 180 deg sign flip
#           cancels exactly in the circular-difference error formulas) -> axis 1
PROTOCOL_FORWARD_AXIS = {"smpl24": 2, "bvh23": 1}


def capability_map(model: dict[str, Any]) -> dict[str, bool]:
    """Normalized capabilities of one comparison row (AnySole rows use the
    built-in declaration; registry rows carry their own)."""
    caps = model.get("capabilities")
    if isinstance(caps, dict) and caps:
        return read_capabilities(caps)
    return {}


def denied_capabilities(entry: dict[str, Any], archive: dict[str, Any] | None = None) -> set[str]:
    """Return provenance findings for audit metadata only.

    The returned set is deliberately not used by ``applicable_for``. Metric
    eligibility comes from generated capabilities and actual solver inputs.
    """
    denied: set[str] = set()
    sources = entry.get("sources")
    if isinstance(sources, dict):
        alias_to_canonical = CAPABILITY_ALIASES
        for field in ("joint_positions", "direct_joint_rotations", "root_translation",
                      "root_rotation", "predicted_surface", "predicted_shape",
                      "pressure_prediction", "contact_prediction"):
            # Registry entries may spell a source under either schema key
            # (e.g. ``surface: template_reconstruction``).
            value = sources.get(field)
            for alias, canonical in alias_to_canonical.items():
                if canonical == field and field not in sources:
                    value = sources.get(alias)
                    break
            if str(value or "").strip().lower() in DENIED_SOURCES:
                denied.add(field)
    meta = archive or {}
    if str(meta.get("provenance_source_type", "")).strip() in DENIED_SOURCES:
        # Archive-level fallback marks the whole payload as not a model output.
        denied.update(
            ("joint_positions", "direct_joint_rotations", "root_translation",
             "root_rotation", "predicted_surface", "predicted_shape",
             "pressure_prediction", "contact_prediction")
        )
    field_to_cap = {
        "root_translation_source": "root_translation",
        "surface_source": "predicted_surface",
        "shape_source": "predicted_shape",
        "pressure_source_type": "pressure_prediction",
    }
    for field, cap in field_to_cap.items():
        if str(meta.get(field, "")).strip() in DENIED_SOURCES:
            denied.add(cap)
    return denied


def applicable_for(caps: dict[str, bool], denied: set[str],
                   approved: Any = ()) -> set[str]:
    """Metrics this row may compute from generated capabilities and GT.

    ``denied`` and ``approved`` remain in the signature for compatibility,
    but provenance labels and historical exceptions do not alter
    applicability.
    """
    del denied, approved
    return applicable_metrics(caps)


def mask_metrics(values: dict[str, Any], applicable: set[str],
                 reasons: dict[str, str], *, reason: str = "not_applicable") -> None:
    """Blank every metric outside the applicable set and record the reason.

    Missing values stay NaN in the row and are written as ``—``/blank by the
    CSV/PNG writers — never as 0.
    """
    for key in METRICS:
        if key not in applicable:
            values[key] = float("nan")
            reasons.setdefault(key, reason)


def archive_meta(path: Path) -> dict[str, str]:
    """Read scalar provenance/string fields of one NPZ without loading arrays."""
    if path.suffix.lower() != ".npz":
        return {}
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in data.files:
            arr = data[key]
            if arr.ndim == 0 and arr.dtype.kind in "US":
                out[key] = str(arr)
        return out


def _csv_value(value: Any) -> Any:
    """CSV cells: NaN/None -> empty (native missing value, rendered as —)."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def anysole_run_dirs(model_dir: Path, variant: str | None) -> list[tuple[str, Path]]:
    """AnySole run subdirs (stacked-hyperparameter naming, 2026-09-24).

    A run dir carries checkpoints/ or predictions/; the flat historical layout
    (model dir itself) is still accepted when no run subdir matches.
    """
    if variant:
        return [(variant, model_dir / variant)]
    if not model_dir.is_dir():
        return []
    runs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and ((p / "checkpoints").is_dir() or (p / "predictions").is_dir())
    )
    return [(p.name, p) for p in runs] or [("", model_dir)]


def expand_anysole_models(results_root: Path, modals: list[str],
                          contact_methods: list[str], config_ids: list[str],
                          variant: str | None = None) -> list[dict[str, Any]]:
    """Expand AnySole modal/contact-method/run/config combinations."""
    models: list[dict[str, Any]] = []
    anysole_root = results_root / "AnySole"
    for modal in modals:
        for contact_method in contact_methods:
            model_dir = anysole_root / cli_common.anysole_model_dir(modal, contact_method)
            if not model_dir.is_dir():
                continue
            for run_name, run_dir in anysole_run_dirs(model_dir, variant):
                suffix = f"/{run_name}" if run_name else ""
                for config_id in config_ids:
                    models.append({
                        "name": f"AnySole/{model_dir.name}{suffix}/{config_id}",
                        "variant": config_id,
                        "run_name": run_name,
                        "prediction_root": str(run_dir / "predictions" / "eval_motion"),
                        "pattern": f"{{session_id}}_{config_id}.npz",
                        "checkpoint": str(run_dir / "checkpoints" / "ckpt_last.pt"),
                        "modal": modal,
                        "contact_method": contact_method,
                        "config_id": config_id,
                        "mode": config_id,
                        "capabilities": ANYSOLE_V2T_CAPS if config_id == "V2T" else ANYSOLE_MOTION_CAPS,
                        "approved": list(ANYSOLE_APPROVED),
                        "sources": {},
                    })
    return models


def registry_baselines(results_root: Path,
                       registry: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """Registry baselines as comparison rows (independent of what is on disk).

    Models without any exported prediction yet still get a row with
    status=missing inside their mode block, so the table tells the reader
    explicitly what was compared and what is not available.
    """
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for entry in registry.get(mode, []):
            name = str(entry.get("name") or "")
            if not name:
                continue
            pred_root = str(entry.get("prediction_root") or "")
            resolved = cli_common.resolve_path(pred_root, results_root) if pred_root else results_root / name / "predictions"
            rows.append({
                "name": name,
                "display_name": str(entry.get("display_name") or name),
                "variant": name,
                "prediction_root": str(resolved),
                "pattern": str(entry.get("pattern") or ""),
                "checkpoint": "",
                "family": str(entry.get("family") or "baseline"),
                "mode": mode,
                "protocol": str(entry.get("protocol") or ""),
                "capabilities": dict(entry.get("capabilities") or {}),
                "sources": dict(entry.get("sources") or {}),
                "approved": [],
            })
    return rows


def _model_result_root(prediction_root: Path) -> Path:
    """Return the owning ``<model>`` directory for metric provenance."""
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
    model: dict[str, Any],
    prediction_root: Path,
    split: str,
    split_csv: Path,
    manifest_path: Path,
    mode: str,
    family: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    capabilities: dict[str, bool],
    denied: set[str],
    applicable: set[str],
    gt_source: str,
    surface_metrics: bool = False,
) -> Path:
    """Persist the same comparison metrics beside each model's results.

    Native model logs remain untouched.  ``<split>_comparison.json`` is the
    cross-model contract: native protocol, capability gating, canonical metric
    names/units, and the exact split/manifest hashes used.  Schema v2 — a
    legacy v1 (common19) file at the same path is reported and superseded,
    never read back as a current result.
    """
    result_root = _model_result_root(prediction_root)
    output = result_root / "metrics" / f"{split}_comparison.json"
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if str(existing.get("schema_version")) == LEGACY_SCHEMA_V1:
                print(f"[legacy] {output}: schema v1 (common19) superseded, rewriting as v2")
        except (ValueError, OSError):
            pass
    metric_rows = []
    for row in rows:
        metric_rows.append({
            key: _json_safe(row.get(key))
            for key in ("session_id", "status", "reason", "n_valid_frames", *METRICS)
        })
    payload = {
        "schema_version": COMPARISON_SCHEMA_V2,
        "model": model.get("name", ""),
        "display_name": model.get("display_name", model.get("name", "")),
        "family": family,
        "mode": mode,
        "split": split,
        "split_csv": str(split_csv),
        "split_sha256": sha256(split_csv),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "evaluation_protocol": summary.get("protocol", ""),
        "capabilities": {
            field: bool(capabilities.get(field, False))
            for field in ("joint_positions", "direct_joint_rotations", "root_translation",
                          "root_rotation", "predicted_surface", "predicted_shape",
                          "pressure_prediction", "contact_prediction")
        },
        "capabilities_denied": sorted(denied),
        "applicable_metrics": sorted(applicable),
        "gt_source": gt_source,
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
    labels, neutral ink; no categorical color coding).  Missing values render
    as —, never as 0."""
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
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return "—"
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
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return "—"
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
        fields = ["model", "protocol", "family", *MODE_METRICS[mode]]
        blocks.append((mode, fields, [[short_model(r["model"]), r.get("protocol", ""), r.get("family", "")] + [cell(r[f], f) for f in MODE_METRICS[mode]] for r in rows]))
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


def _session_time_grid(row: dict[str, str]) -> np.ndarray:
    """Canonical mocap time axis: t_mocap = visual_start_s + i/fps - offset_s."""
    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    return float(row["visual_start_s"]) + np.arange(n, dtype=np.float64) / fps - float(row["offset_s"])


def evaluate_pressure_row(model: dict[str, Any], row: dict[str, str],
                          pred_path: Path, registry: dict[str, list[dict[str, str]]],
                          mode: str) -> dict[str, Any]:
    """One session of a V2T (pressure) row: brief (6) + leaf (4) pressure keys."""
    caps = capability_map(model)
    denied = denied_capabilities(model)
    applicable = applicable_for(caps, denied)
    reasons: dict[str, str] = {}
    base = {
        "model": model["name"], "variant": model.get("variant", ""),
        "run_name": model.get("run_name", ""), "session_id": row["session_id"],
        "subject_id": row.get("subject_id", ""), "action": row.get("action", ""),
        "split": "placeholder", "protocol": "pressure", "gt_source": "",
        "n_valid_frames": 0, "mode": mode, "family": str(model.get("family") or
        ("main" if model["name"].startswith("AnySole/") else "baseline")),
        "status": "ok", "reason": "", "metric_reasons": {}, "prediction": str(pred_path),
    }
    base.update({key: float("nan") for key in METRICS})
    try:
        archive = load_v2t_archive(pred_path)
        expected = int(float(row.get("n_frames") or 0))
        if len(archive["valid_mask"]) != expected:
            raise ValueError(
                f"standard V2T archive length={len(archive['valid_mask'])}; "
                f"canonical session={expected}"
            )
        values = v2t_metrics(archive["pressure_pred"], archive["pressure_gt"],
                             archive["valid_mask"],
                             contact_smpl_pred=archive.get("contact_smpl_pred"),
                             contact_smpl_gt=archive.get("contact_smpl_gt"))
        base.update(values)
        # Pressure keys require generated pressure output; contact-level keys
        # additionally require an explicit contact prediction. Provenance is
        # retained in the row metadata but does not override this contract.
        mask_metrics(base, applicable, reasons)
        # The binarized legacy archive contact fields are never consumed;
        # contact_smpl_mse/bce come from the continuous maps.  contact_f1
        # stays out until a binary contact GT exists (待处理).
        base["metric_reasons"] = json.dumps(reasons, ensure_ascii=False, sort_keys=True)
        base["gt_source"] = "archive embedded pressure_gt (same session grid)"
    except FileNotFoundError as exc:
        base.update({"status": "missing", "reason": str(exc)})
    except Exception as exc:
        base.update({"status": "invalid", "reason": str(exc)})
    return base


def evaluate_motion_row(model: dict[str, Any], row: dict[str, str],
                        pred_path: Path, args: argparse.Namespace, split: str,
                        registry: dict[str, list[dict[str, str]]], mode: str,
                        gt_cache: dict, gt_surface_cache: dict,
                        gt_rotation_cache: dict) -> dict[str, Any]:
    """One session of a motion row: native protocol, capability-gated metrics."""
    caps = capability_map(model)
    archive_info = archive_meta(pred_path) if pred_path is not None else {}
    denied = denied_capabilities(model, archive_info)
    applicable = applicable_for(caps, denied, approved=model.get("approved", ()))
    reasons: dict[str, str] = {}
    base = {
        "model": model["name"], "variant": model.get("variant", ""),
        "run_name": model.get("run_name", ""), "session_id": row["session_id"],
        "subject_id": row.get("subject_id", ""), "action": row.get("action", ""),
        "split": split, "protocol": "", "gt_source": "",
        "n_valid_frames": 0, "mode": mode, "family": str(model.get("family") or
        ("main" if model["name"].startswith("AnySole/") else "baseline")),
        "status": "ok", "reason": "", "metric_reasons": {}, "prediction": str(pred_path),
    }
    base.update({key: float("nan") for key in METRICS})
    if pred_path is None:
        base.update({"status": "missing",
                     "reason": f"no prediction under {model.get('prediction_root', '')}"})
        return base
    try:
        declared = normalize_protocol(str(model.get("protocol") or ""))
        if declared == "pressure":
            raise ValueError("pressure rows use evaluate_pressure_row")
        # Load the prediction in its native protocol (BVH resampled onto the
        # canonical session grid so the frame pairing with GT is exact).
        if pred_path.suffix.lower() == ".bvh":
            pred, pred_names, _parents = bvh_joints(pred_path, row)
            pred_mask = None
            protocol = "bvh23"
        else:
            pred, pred_mask, pred_names, protocol_label = array_from_file(pred_path)
            protocol = normalize_protocol(protocol_label)
            if pred_mask is None:
                raise ValueError(
                    "legacy motion archive lacks the unified "
                    "joint_xyz_world/valid_mask contract"
                )
        if protocol != declared:
            raise ValueError(
                f"declared protocol {declared!r} but prediction is "
                f"{PROTOCOL_LABELS[protocol]!r}"
            )
        expected = int(float(row.get("n_frames") or 0))
        if pred_mask is not None and len(pred_mask) != expected:
            raise ValueError(
                f"unified motion archive mask={len(pred_mask)}; canonical session={expected}"
            )
        gt, gt_names = gt_cache.setdefault((row["session_id"], protocol),
                                           protocol_gt(row, protocol))
        base["protocol"] = PROTOCOL_LABELS[protocol]
        base["gt_source"] = _gt_source_path(row, protocol)
        if not caps.get("joint_positions", False):
            raise ValueError("declared joint_positions=false for a motion row")

        eval_fps = float(row.get("target_fps") or args.fps)
        times = _session_time_grid(row)
        n_eval = min(len(pred), len(gt), expected)
        keep = valid_mask(row, n_eval)
        if pred_mask is not None:
            keep &= pred_mask[:n_eval].astype(bool)

        # Rotations: needed only when root orientation / MPJAE are applicable.
        want_rotations = bool(applicable & {"root_orientation_deg", "root_orientation_drift_deg", "mpjae_deg"})
        pred_rotations = gt_rotations = None
        if want_rotations:
            pred_rotation_data = load_rotations(pred_path)
            cache_key = (row["session_id"], protocol)
            gt_rotation_data = gt_rotation_cache.setdefault(
                cache_key, protocol_gt_rotations(row, protocol)
            )
            pred_rotations = np.asarray(pred_rotation_data["rotations"], dtype=np.float64)
            gt_rotations = np.asarray(gt_rotation_data["rotations"], dtype=np.float64)

        # Surface: only for PVE-applicable models, and only when requested.
        pred_vertices = gt_vertices = None
        public_surface = str(archive_info.get("public_surface_metrics", "true")).strip().lower()
        if ("pve_mm" in applicable and args.surface_metrics
                and public_surface not in {"0", "false", "no"}):
            if protocol != "smpl24":
                reasons["pve_mm"] = "not_applicable: no SMPL surface for bvh23"
            else:
                pred_surface = load_surface(pred_path)
                if pred_surface is None:
                    if caps.get("predicted_surface", False):
                        raise ValueError(
                            "declared predicted_surface=true but archive has no "
                            "surface/vertex contract"
                        )
                    reasons["pve_mm"] = "not_computed: no surface in archive"
                else:
                    pred_vertices = pred_surface.get("vertices")
                    cache_key = (row["session_id"], protocol)
                    if cache_key not in gt_surface_cache:
                        gt_surface_cache[cache_key] = protocol_gt_surface(row, protocol)
                    gt_surface = gt_surface_cache[cache_key]
                    gt_vertices = gt_surface.get("vertices") if gt_surface else None
        elif "pve_mm" in applicable and public_surface in {"0", "false", "no"}:
            reasons["pve_mm"] = "not_applicable: archive public_surface_metrics=false"
        elif "pve_mm" in applicable:
            reasons["pve_mm"] = "not_computed: --surface-metrics off"

        values = metrics(
            pred[:n_eval], gt[:n_eval], keep, eval_fps,
            protocol=protocol,
            names=pred_names,
            pred_rotations=pred_rotations,
            gt_rotations=gt_rotations,
            pred_vertices=pred_vertices,
            gt_vertices=gt_vertices,
            times=times,
            forward_axis=PROTOCOL_FORWARD_AXIS[protocol],
        )
        base.update(values)
        # Post-gate: metrics the capability contract does not permit stay
        # blank with an explicit reason (never silently kept).
        mask_metrics(base, applicable, reasons)
        base["n_valid_frames"] = int(keep.sum())
        for metric in METRICS:
            if not math.isfinite(float(base[metric])) and metric in applicable:
                reasons.setdefault(metric, "missing: solver returned no value")
    except FileNotFoundError as exc:
        base.update({"status": "missing", "reason": str(exc)})
    except Exception as exc:
        base.update({"status": "invalid", "reason": str(exc)})
        base.update({key: float("nan") for key in METRICS})
    base["metric_reasons"] = json.dumps(reasons, ensure_ascii=False, sort_keys=True)
    return base


def _gt_source_path(row: dict[str, str], protocol: str) -> str:
    key = "bvh_path" if protocol == "bvh23" else "smpl_path"
    try:
        return str(resolve_repo_path(row[key], ROOT))
    except (KeyError, ValueError):
        return f"{key}:{row.get(key, '')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.csv")
    ap.add_argument("--split-csv", type=Path, default=cli_common.DEFAULT_SPLIT_CSV,
                    help="Canonical train/val/test membership table; manifest is metadata only.")
    ap.add_argument("--models-config", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--auto-scan", action="store_true",
                    help="Deprecated alias: baselines always come from the mode registry now.")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", type=Path, default=cli_common.DISPLAY_ROOT / "ResultTest/R2Test_compare")
    ap.add_argument("--force", action="store_true", help="Rebuild and overwrite existing outputs.")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--model-name", "--modal", dest="modal", metavar="MODEL_NAME", default="anysolev1,anysolev1_insole_drift", help="AnySole model name(s), comma-separated; legacy alias: --modal")
    ap.add_argument("--contact-method", default="tactile_abs", help="Contact-label scheme(s), comma-separated; model dir is <model-name>_<contact-method>")
    ap.add_argument("--variant", default=None, help="AnySole stacked-hyperparameter run subdir (omit to scan all runs).")
    ap.add_argument("--config-id", default="VT2M,V2M,T2M,V2T", help="AnySole generation task(s), comma-separated")
    ap.add_argument("--modes-config", type=Path, default=Path(__file__).resolve().parent.parent / "models_modes.yaml",
                    help="Mode/protocol/capability registry for baselines.")
    ap.add_argument("--by-mode", action="store_true",
                    help="Additionally write per-mode summary blocks (by_mode/<mode>/) and mode_overview.png. "
                         "Rows of different modes never share a block.")
    ap.add_argument("--write-model-metrics", action="store_true",
                    help="Also write <results>/<model>/metrics/<split>_comparison.json with the same canonical metrics (schema v2).")
    ap.add_argument("--surface-metrics", action="store_true",
                    help="Compute PVE for surface-capable models (SMPL surface loading is slower).")
    args = ap.parse_args()
    modals = cli_common.split_csv_arg(args.modal)
    contact_methods = cli_common.split_csv_arg(args.contact_method)
    config_ids = cli_common.split_csv_arg(args.config_id)
    allowed_configs = {"VT2M", "V2M", "T2M", "V2T"}
    invalid = set(config_ids) - allowed_configs
    if invalid:
        raise SystemExit(f"Unsupported --config-id: {sorted(invalid)}; choices={sorted(allowed_configs)}")
    registry = load_mode_registry(args.modes_config)
    if args.models_config is not None:
        try:
            import yaml
            configs = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
        except ImportError:
            configs = json.loads(args.models_config.read_text(encoding="utf-8"))
        models = [dict(m) for m in configs.get("models", configs)]
        # External model lists must declare protocol/capabilities themselves.
        refused = [m["name"] for m in models if not m.get("capabilities") or not m.get("protocol")]
        for m in models:
            if not m.get("capabilities") or not m.get("protocol"):
                m["refused_reason"] = "no declared protocol/capabilities"
    else:
        models = expand_anysole_models(args.results_root, modals, contact_methods,
                                       config_ids, variant=args.variant)
        models.extend(registry_baselines(args.results_root, registry))
        refused = []
    split_csv = Path(cli_common.resolve_path(args.split_csv, ROOT))
    requested_ids = load_split_ids(split_csv, args.split)
    manifest = read_manifest(args.manifest, args.split, split_csv=split_csv, evaluable_only=True)
    if not manifest:
        raise SystemExit(f"No '{args.split}' sessions found in {args.manifest}")
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
    for model in models:
        name = model["name"]
        if model.get("refused_reason"):
            mode_resolution.append(f"REFUSED {name}: {model['refused_reason']}")
            continue
        root = Path(cli_common.resolve_path(model["prediction_root"], ROOT))
        mode = resolve_mode_for(model, registry) if args.by_mode else str(model.get("mode") or "")
        family = str(model.get("family") or ("main" if name.startswith("AnySole/") else "baseline"))
        if args.by_mode and not mode:
            mode_resolution.append(f"UNRESOLVED mode -> excluded from mode blocks: {name} (declare it in {args.modes_config})")
        model_rows = []
        declared = normalize_protocol(str(model.get("protocol") or ""))
        gt_source = ""
        for row in manifest:
            sid = row["session_id"]
            pred_path = find_prediction(root, sid, model.get("pattern"), config_id=model.get("config_id"))
            if declared == "pressure":
                result = evaluate_pressure_row(model, row, pred_path, registry, mode)
            else:
                result = evaluate_motion_row(model, row, pred_path, args, args.split,
                                             registry, mode, gt_cache, gt_surface_cache,
                                             gt_rotation_cache)
            if result.get("status") == "ok":
                gt_source = result.get("gt_source", "")
            details.append(result)
            model_rows.append(result)
        s = aggregate(model_rows)
        s.update({"model": name, "display_name": model.get("display_name", name),
                  "protocol": _row_protocol(model_rows), "variant": model.get("variant", ""),
                  "run_name": model.get("run_name", ""),
                  "n_sessions": sum(r["status"] == "ok" for r in model_rows),
                  "split_sha256": sha256(split_csv), "manifest_sha256": sha256(args.manifest),
                  "eval_fps": args.fps, "checkpoint": model.get("checkpoint", ""),
                  "mode": mode, "family": family})
        summaries.append(s)
        if args.write_model_metrics and model_rows and model_rows[0].get("protocol"):
            caps = capability_map(model)
            denied = denied_capabilities(model)
            applicable = applicable_for(caps, denied, approved=model.get("approved", ()))
            write_model_metrics(
                model, root, args.split, split_csv, args.manifest, mode, family,
                model_rows, s, caps, denied, applicable, gt_source,
                surface_metrics=args.surface_metrics,
            )
    detail_fields = ["model", "variant", "run_name", "session_id", "subject_id", "action", "split", "protocol", "gt_source", "n_valid_frames", *METRICS, "mode", "family", "status", "reason", "metric_reasons", "prediction"]
    summary_fields = ["model", "protocol", *METRICS, "mode", "family"]
    for filename, fields, rows in (("comparison_per_session.csv", detail_fields, details), ("comparison_summary.csv", summary_fields, summaries)):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: _csv_value(r.get(k, "")) for k in fields} for r in rows)
    mode_report = []
    by_mode_csvs: dict[str, Path] = {}
    if args.by_mode:
        by_mode_dir = args.out_dir / "by_mode"
        for mode in MODES:
            rows = [r for r in summaries if r.get("mode") == mode]
            mode_report.append(f"mode[{mode}]: {len(rows)} models ({', '.join(r['model'] for r in rows) or 'none'})")
            mode_fields = ["model", "protocol", "family", *MODE_METRICS[mode]]
            csv_path = by_mode_dir / mode / "comparison_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=mode_fields)
                w.writeheader()
                w.writerows({k: _csv_value(r.get(k, "")) for k in mode_fields} for r in rows)
            render_summary_png(csv_path, by_mode_dir / mode / "comparison_summary.png", f"Test2 {mode} comparison  split={args.split}  fps={args.fps}")
            by_mode_csvs[mode] = csv_path
        render_mode_overview_png(by_mode_csvs, by_mode_dir / "mode_overview.png", f"Test2 comparison by generation mode  split={args.split}  fps={args.fps}")
    # Per-model protocol/capability/status digest for the log.
    model_lines = []
    for s in summaries:
        n_ok = s.get("n_sessions", 0)
        n_missing = sum(1 for r in details if r.get("model") == s["model"] and r.get("status") == "missing")
        n_invalid = sum(1 for r in details if r.get("model") == s["model"] and r.get("status") == "invalid")
        model_lines.append(
            f"model[{s['model']}] protocol={s.get('protocol','')} mode={s.get('mode','')} "
            f"ok={n_ok} missing={n_missing} invalid={n_invalid}"
        )
    (args.out_dir / "evaluation.log").write_text(
        f"split={args.split}\nsplit_csv={split_csv}\nsplit_sha256={sha256(split_csv)}\n"
        f"manifest={args.manifest}\nmanifest_sha256={sha256(args.manifest)}\n"
        f"requested_sessions={len(requested_ids)}\n"
        f"evaluable_sessions={len(manifest)}\n"
        f"excluded_no_valid_frames={','.join(excluded_ids) or 'none'}\n"
        f"models={len(models)}\n"
        f"modal={','.join(modals)}\ncontact_method={','.join(contact_methods)}\nconfig_id={','.join(config_ids)}\n"
        f"protocol_rule=native-protocol GT (smpl24-native/bvh23-native); no common19\n"
        f"comparison_schema={COMPARISON_SCHEMA_V2}\n"
        f"surface_metrics={bool(args.surface_metrics)}\n"
        f"modes_config={args.modes_config}\n"
        + "\n".join(model_lines)
        + ("\n" + "\n".join(mode_report) if mode_report else "")
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


def _row_protocol(model_rows: list[dict[str, Any]]) -> str:
    """Protocol label of one model's rows (from the first evaluated row)."""
    for row in model_rows:
        if row.get("protocol"):
            return row["protocol"]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
