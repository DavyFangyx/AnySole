"""R_Test4: V2T archive evaluation and generated-vs-GT insole heatmaps.

The default path consumes session-level standardized V2T archives written by
the upstream evaluators.  It never reruns a model.  ``--source infer`` keeps
the former AnySole on-the-fly diagnostic for debugging only.

Three conditioning arms (same names as eval.py ``--config-id``):

    VT2M   real V + real T   reconstruction sanity (upper bound)
    V2M    real V + zero T   the V2T generation itself (headline)
    T2M    zero V + real T   tactile self-reconstruction (input-side sanity)

Outputs under results_display/ResultTest/R4Test_v2t/:

    gif/ or mp4/<session>_<mode>_tgen.{gif,mp4}  GT | generated | |GT-Gen| animation
    cells/<session>_<mode>_cells.{npz,png}      96-cell MAE + static error map
    tgen_summary.csv                            per session x mode MAE/RMSE/corr
    tgen_report.json                            aggregate per mode + V2T summary

Usage (run from the repository root):
    python results_display/script/r_test4_v2t.py
    python results_display/script/r_test4_v2t.py --session S10103
    python results_display/script/r_test4_v2t.py --config-id VT2M,V2M --export-sessions 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from utils import cli_common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger as log  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from anysole.data.dataset import find_session_dir, hrnet_cache_path, session_time_grid  # noqa: E402
from anysole.data.pressure import load_session_pressure, normalize_raw  # noqa: E402
from anysole.eval import _load_model  # noqa: E402
from anysole.infer import _right_pad_windows  # noqa: E402
from anysole.train import load_config, resolve_device  # noqa: E402
from anysole.types import (  # noqa: E402
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    FPS,
    POSE_DIM,
    PRESSURE_CLIP,
    T_RAW_DIM,
    V_FEAT_DIM,
)
from utils.compare_core import (  # noqa: E402
    find_prediction,
    load_mode_registry,
    load_v2t_archive,
    v2t_metrics,
)
from d_test3_contact import (  # noqa: E402
    INFO_FONT,
    INSOLE_H,
    INSOLE_W,
    LABEL_FONT,
    SENSOR_COLS,
    SENSOR_ROWS,
    TITLE_FONT,
    draw_text,
    pressure_to_heatmap,
)

MODE_NAMES = CONFIG_MODE_NAMES
_MODE_TO_CONFIG = dict(zip(CONFIG_MODE_NAMES, (CONFIG_VT, CONFIG_V, CONFIG_T)))

DEFAULT_CKPT = cli_common.RESULTS_ROOT / "AnySole" / "anysolev1_joint_and" / "checkpoints" / "ckpt_last.pt"

# --- Animation layout: 3 columns (GT | Generated | |GT-Gen|) x 2 rows (feet). ---
GAP = 20
ROW_LABEL_W = 52
HEADER_H = 62
CANVAS_W = ROW_LABEL_W + 3 * INSOLE_W + 4 * GAP
CANVAS_H = HEADER_H + 2 * INSOLE_H + 3 * GAP


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--config-id must contain at least one mode")
    unknown = [item for item in modes if item not in _MODE_TO_CONFIG]
    if unknown:
        raise ValueError("Unknown mode(s) %s; choose from VT2M,V2M,T2M" % ",".join(unknown))
    return list(dict.fromkeys(modes))


def canonical_pressure_map(values: np.ndarray) -> np.ndarray:
    """Return one display grid of shape (31,22), two 31x11 feet."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1 and values.size == 96:
        values = values.reshape(2, 4, 12)
    elif values.ndim == 1 and values.size == 31 * 22:
        values = values.reshape(31, 22)
    if values.ndim == 2 and values.shape == (31, 22):
        return values
    if values.ndim == 3 and values.shape == (2, 31, 11):
        return np.concatenate([values[0], values[1]], axis=1)
    if values.ndim == 3 and values.shape == (2, 4, 12):
        from scipy.ndimage import zoom
        values = zoom(values, (1, 31 / 4, 11 / 12), order=1,
                      mode="nearest", prefilter=False)
        return np.concatenate([values[0], values[1]], axis=1)
    raise ValueError(f"unsupported V2T display grid: {values.shape}")


def cells_to_heatmap(cells: np.ndarray, vmax: float) -> Image.Image:
    """Canonical 31x11-per-foot map -> insole heatmap."""
    cells = np.asarray(cells, dtype=np.float32)
    return Image.fromarray(pressure_to_heatmap(np.rot90(cells, k=1), vmax=vmax))


def load_session_inputs(session_id: str, config: dict) -> dict | None:
    """Return V features + normalized GT tactile for one session, or None when inputs are missing."""
    try:
        seq_dir = find_session_dir(Path(config["seq_root"]), session_id)
        meta = json.loads((seq_dir / "align_meta.json").read_text())
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"{session_id}: cannot locate sequence ({exc}); skipped")
        return None
    n_frames = int(meta["n_frames"])
    cache_path = hrnet_cache_path(session_id, Path(config["cache_root"]))
    if not cache_path.is_file():
        log.warning(f"{session_id}: missing HRNet cache {cache_path}; skipped")
        return None
    loaded_v = torch.load(cache_path, map_location="cpu")
    v_feat = loaded_v.float().numpy() if torch.is_tensor(loaded_v) else np.asarray(loaded_v, dtype=np.float32)
    if v_feat.shape != (n_frames, V_FEAT_DIM):
        log.warning(f"{session_id}: V_feat shape {v_feat.shape} != ({n_frames}, {V_FEAT_DIM}); skipped")
        return None
    try:
        pressure = load_session_pressure(meta, session_time_grid(meta))
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"{session_id}: no 48-cell CSV pressure ({exc}); skipped")
        return None
    return {
        "session_id": session_id,
        "meta": meta,
        "n_frames": n_frames,
        "v_feat": v_feat,
        "t_norm": normalize_raw(pressure["T_raw"])[:n_frames],
        "t_phys": np.asarray(pressure["T_phys"], dtype=np.float32)[:n_frames],
    }


def generate_tactile(model, inputs: dict, mode: str, tw: int, device: torch.device, batch_size: int) -> np.ndarray:
    """Tau=0 forwards over padded windows; returns generated T (n_frames, 96, normalized)."""
    config_value = _MODE_TO_CONFIG[mode]
    v_windows = _right_pad_windows(inputs["v_feat"], tw).to(device)
    t_raw_windows = _right_pad_windows(inputs["t_norm"], tw)
    t_phys_windows = _right_pad_windows(inputs["t_phys"], tw)
    if config_value == CONFIG_V:  # tactile zeroed: the V2T generation arm
        t_raw_windows = torch.zeros_like(t_raw_windows)
        t_phys_windows = torch.zeros_like(t_phys_windows)
    if config_value == CONFIG_T:  # vision zeroed
        v_windows = torch.zeros_like(v_windows)
    n_win = v_windows.shape[0]
    parts = []
    with torch.inference_mode():
        for left in range(0, n_win, batch_size):
            right = min(left + batch_size, n_win)
            bsz = right - left
            config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
            tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
            x_in = torch.zeros(bsz, tw, POSE_DIM, device=device)
            out = model(
                v_windows[left:right],
                t_raw_windows[left:right].to(device),
                t_phys_windows[left:right].to(device),
                x_in,
                tau_zero,
                config_id,
                [inputs["session_id"]] * bsz,
            )
            parts.append(out["pressure_hat"].cpu())
    return torch.cat(parts, dim=0).reshape(-1, T_RAW_DIM)[: inputs["n_frames"]].numpy()


def session_metrics(gen_t: np.ndarray, t_norm: np.ndarray) -> dict:
    """MAE/RMSE (normalized 0-1) + per-frame correlation over the 96 cells."""
    diff = gen_t - t_norm
    gen_c = gen_t - gen_t.mean(axis=1, keepdims=True)
    tgt_c = t_norm - t_norm.mean(axis=1, keepdims=True)
    denom = np.sqrt((gen_c * gen_c).sum(axis=1) * (tgt_c * tgt_c).sum(axis=1))
    valid = denom > 1e-8
    corr = np.where(valid, (gen_c * tgt_c).sum(axis=1) / np.maximum(denom, 1e-8), np.nan)
    return {
        "mae_norm": float(np.abs(diff).mean()),
        "rmse_norm": float(np.sqrt(np.square(diff).mean())),
        "corr": float(np.nanmean(corr)),
        "corr_valid_frames": int(valid.sum()),
        "cells": np.abs(diff).mean(axis=0).astype(np.float32),
    }


def render_frame(gt_cells, gen_cells, session_id, mode, frame_idx, n_frames, fps) -> np.ndarray:
    """One animation frame: 2 rows (feet) x 3 columns (GT | Generated | |GT-Gen|)."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, CANVAS_W - 1, CANVAS_H - 1], outline=(48, 48, 56))
    titles = ("GT", "Generated", "|GT-Gen|")
    for col, title in enumerate(titles):
        cx = ROW_LABEL_W + GAP + col * (INSOLE_W + GAP) + INSOLE_W // 2
        draw_text(draw, (cx, 10), title, LABEL_FONT, fill=(220, 220, 230), anchor="mt")
    draw_text(
        draw,
        (CANVAS_W // 2, 38),
        f"{session_id}  {mode}  t={frame_idx / max(fps, 1e-6):.2f}s  {frame_idx}/{n_frames - 1}",
        INFO_FONT,
        fill=(180, 180, 190),
        anchor="mt",
    )
    gt_map = canonical_pressure_map(gt_cells)
    gen_map = canonical_pressure_map(gen_cells)
    panels = (
        (gt_map[:, :11], gen_map[:, :11], np.abs(gen_map[:, :11] - gt_map[:, :11]), "Left"),
        (gt_map[:, 11:], gen_map[:, 11:], np.abs(gen_map[:, 11:] - gt_map[:, 11:]), "Right"),
    )
    for row, (gt48, gen48, err48, label) in enumerate(panels):
        y0 = HEADER_H + row * (INSOLE_H + GAP)
        draw_text(draw, (ROW_LABEL_W // 2, y0 + INSOLE_H // 2), label, TITLE_FONT, fill=(150, 150, 170), anchor="mm")
        for col, cells in enumerate((gt48, gen48, err48)):
            img = cells_to_heatmap(cells * PRESSURE_CLIP, vmax=PRESSURE_CLIP)
            x0 = ROW_LABEL_W + GAP + col * (INSOLE_W + GAP)
            canvas.paste(img, (x0, y0))
    return np.asarray(canvas)


def plot_cell_errors(cells_mae: np.ndarray, out_png: Path, session_id: str, mode: str) -> None:
    """Static per-cell MAE figure: two 4x12 insole grids in raw pressure units."""
    left = cells_mae[:48].reshape(SENSOR_ROWS, SENSOR_COLS) * PRESSURE_CLIP
    right = cells_mae[48:].reshape(SENSOR_ROWS, SENSOR_COLS) * PRESSURE_CLIP
    vmax = max(float(left.max()), float(right.max()), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in ((axes[0], left, "Left foot"), (axes[1], right, "Right foot")):
        im = ax.imshow(data, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{session_id} {mode} per-cell MAE (raw units)")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R_Test4: V2T tactile generation (GT vs generated insoles + error analysis).")
    cli_common.add_common_args(
        parser,
        session=True,
        split_csv=True,
        split=True,
        gen=True,
        fps=True,
        stride=True,
        max_frames=True,
        force=True,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "ResultTest/R4Test_v2t",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "anysole" / "configs" / "v1.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--source", choices=("archives", "infer"), default="archives",
                        help="archives reads standardized V2T outputs; infer is the legacy on-the-fly diagnostic.")
    parser.add_argument("--model-name", "--modal", dest="modal", default="V4B")
    parser.add_argument("--contact-method", default="joint_and")
    parser.add_argument("--modes-config", type=Path, default=SCRIPT_DIR / "models_modes.yaml")
    parser.add_argument("--config-id", type=str, default="V2T",
                        help="archives mode; infer accepts the legacy VT2M,V2M,T2M arms.")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Cap the number of sessions (0 = all).")
    parser.add_argument("--export-sessions", type=int, default=4, help="First N sessions get animations (0 = all).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main_infer(args: argparse.Namespace) -> int:
    if args.config_id.upper() == "V2T":
        args.config_id = "VT2M,V2M,T2M"
    config = load_config(Path(args.config))
    device = resolve_device(args.device)
    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    model = _load_model(checkpoint, config, device)
    tw = int(config["tw"])
    batch_size = int(args.batch_size or config["batch_size"])
    modes = _parse_modes(args.config_id)
    sessions = cli_common.load_evaluable_sessions(args.session, args.split_csv, args.split)
    if args.limit_sessions > 0:
        sessions = sessions[: args.limit_sessions]
    export_n = args.export_sessions if args.export_sessions > 0 else len(sessions)

    out_dir = Path(args.out_dir)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_mode_metrics = {mode: [] for mode in modes}
    processed = 0
    for session_id in sessions:
        inputs = load_session_inputs(session_id, config)
        if inputs is None:
            continue
        processed += 1
        for mode in modes:
            gen_t = generate_tactile(model, inputs, mode, tw, device, batch_size)
            m = session_metrics(gen_t, inputs["t_norm"])
            rows.append(
                {
                    "session": session_id,
                    "mode": mode,
                    "mae_norm": round(m["mae_norm"], 6),
                    "rmse_norm": round(m["rmse_norm"], 6),
                    "corr": round(m["corr"], 6),
                    "corr_valid_frames": m["corr_valid_frames"],
                }
            )
            per_mode_metrics[mode].append(m)
            log.info(
                f"{session_id} {mode}: MAE={m['mae_norm']:.4f} RMSE={m['rmse_norm']:.4f} corr={m['corr']:.4f}"
            )
            npz_path = cells_dir / f"{session_id}_{mode}_cells.npz"
            np.savez_compressed(npz_path, cells_mae=m["cells"], gen_t=gen_t, gt_t=inputs["t_norm"])
            png_path = cells_dir / f"{session_id}_{mode}_cells.png"
            plot_cell_errors(m["cells"], png_path, session_id, mode)

            if processed > export_n:
                continue
            gen_path = cli_common.media_path(out_dir, f"{session_id}_{mode}_tgen", args.gen)
            if cli_common.outputs_ready([gen_path]) and not args.force:
                log.info(f"Skip {session_id} [{mode}]: {gen_path} already exists")
                continue
            n = inputs["n_frames"]
            n_render = n if args.max_frames <= 0 else min(n, args.max_frames)
            gt_cells = inputs["t_norm"]
            frames = [
                render_frame(gt_cells[t], gen_t[t], session_id, mode, t, n, args.fps)
                for t in range(0, n_render, max(args.stride, 1))
            ]
            fps = cli_common.viz_fps(args.fps, args.stride)
            gen_path.parent.mkdir(parents=True, exist_ok=True)
            if args.gen == "gif":
                cli_common.write_gif(frames, gen_path, fps)
            else:
                cli_common.write_mp4(frames, gen_path, fps)
            log.info(f"Wrote {gen_path} ({len(frames)} frames)")

    summary_path = out_dir / "tgen_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session", "mode", "mae_norm", "rmse_norm", "corr", "corr_valid_frames"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {summary_path}")

    aggregate = {
        mode: {
            "sessions": len(per_mode_metrics[mode]),
            "mae_norm_mean": float(np.mean([m["mae_norm"] for m in per_mode_metrics[mode]])) if per_mode_metrics[mode] else None,
            "rmse_norm_mean": float(np.mean([m["rmse_norm"] for m in per_mode_metrics[mode]])) if per_mode_metrics[mode] else None,
            "corr_mean": float(np.mean([m["corr"] for m in per_mode_metrics[mode]])) if per_mode_metrics[mode] else None,
        }
        for mode in modes
    }
    report = {
        "ckpt": str(args.ckpt),
        "modal": str(checkpoint.get("config", {}).get("modal")),
        "modes": modes,
        "sessions_processed": processed,
        "metrics": aggregate,
    }
    if "V2M" in aggregate:
        report["v2t"] = aggregate["V2M"]
    report_path = out_dir / "tgen_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(f"Wrote {report_path}")
    return 0


def plot_archive_error(cells_mae: np.ndarray, out_png: Path, session_id: str, model: str) -> None:
    """Render the canonical 31x11-per-foot absolute error map."""
    values = canonical_pressure_map(cells_mae)
    vmax = max(float(values.max()), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, data, title in ((axes[0], values[:, :11], "Left foot"), (axes[1], values[:, 11:], "Right foot")):
        im = ax.imshow(data, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{session_id} {model} V2T MAE (canonical 31x11 grid)")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def archive_entries(args: argparse.Namespace) -> list[dict[str, str]]:
    registry = load_mode_registry(args.modes_config)
    entries = []
    for modal in cli_common.split_csv_arg(args.modal):
        for contact_method in cli_common.split_csv_arg(args.contact_method):
            model_dir = cli_common.anysole_model_dir(modal, contact_method)
            entries.append({
                "name": f"AnySole/{model_dir}/V2T",
                "prediction_root": str(cli_common.RESULTS_ROOT / "AnySole" / model_dir / "predictions" / "eval_motion"),
                "pattern": "{session_id}_V2T.npz",
                "color": "#50c8ff",
            })
    entries.extend(dict(entry) for entry in registry.get("V2T", []))
    return entries


def main_archives(args: argparse.Namespace) -> int:
    if args.config_id.upper() not in ("V2T", "ALL"):
        raise SystemExit("archive R_Test4 only accepts --config-id V2T")
    sessions = cli_common.load_evaluable_sessions(args.session, args.split_csv, args.split)
    if cli_common.DEFAULT_MANIFEST.suffix.lower() in (".jsonl", ".json"):
        manifest_rows = {
            row["session_id"]: row
            for row in (json.loads(line) for line in cli_common.DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    else:
        with cli_common.DEFAULT_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
            manifest_rows = {row["session_id"]: row for row in csv.DictReader(handle)}
    entries = archive_entries(args)
    out_dir = Path(cli_common.resolve_path(args.out_dir, os.getcwd()))
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_model: dict[str, list[dict]] = {}
    export_count = 0
    for session_id in sessions:
        for entry in entries:
            root = cli_common.resolve_path(entry.get("prediction_root", ""))
            path = find_prediction(root, session_id, entry.get("pattern") or None)
            if path is None:
                rows.append({"model": entry["name"], "session": session_id, "status": "missing"})
                continue
            try:
                archive = load_v2t_archive(path)
                expected_frames = int(float(manifest_rows[session_id]["n_frames"]))
                if len(archive["valid_mask"]) != expected_frames:
                    raise ValueError(
                        f"standard V2T archive length={len(archive['valid_mask'])}; "
                        f"canonical session={expected_frames}"
                    )
                values = v2t_metrics(
                    archive["pressure_pred"], archive["pressure_gt"],
                    archive["contact_pred"], archive["contact_gt"],
                    archive["valid_mask"],
                )
                row = {
                    "model": entry["name"], "session": session_id, "status": "ok",
                    "n_valid_frames": int(values.get("n_valid_frames", 0)),
                    **{key: values.get(key) for key in ("T_mae", "T_rmse", "T_corr", "pressure_force_mae", "pressure_force_rmse", "pressure_force_r2", "pressure_cop_error_left", "pressure_cop_error_right", "pressure_cop_error_mean", "contact_f1", "contact_acc", "contact_recall", "air_recall")},
                }
                # Store canonical cell errors for downstream inspection.  The
                # archive itself remains in its native grid.
                pred_all = archive["pressure_pred"]
                gt_all = archive["pressure_gt"]
                pred_maps = np.stack([canonical_pressure_map(x) for x in pred_all], axis=0)
                gt_maps = np.stack([canonical_pressure_map(x) for x in gt_all], axis=0)
                valid = np.asarray(archive["valid_mask"], dtype=bool)
                cell_error = np.abs(pred_maps - gt_maps)
                if valid.any():
                    cell_error = cell_error[valid].mean(axis=0)
                else:
                    cell_error = np.zeros_like(pred_maps[0])
                safe_model = entry["name"].replace("/", "_")
                np.savez_compressed(
                    cells_dir / f"{session_id}_{safe_model}_cells.npz",
                    cells_mae=cell_error,
                    pressure_pred=pred_maps,
                    pressure_gt=gt_maps,
                    valid_mask=valid,
                )
                plot_archive_error(cell_error, cells_dir / f"{session_id}_{safe_model}_cells.png", session_id, entry["name"])
                if export_count < (args.export_sessions if args.export_sessions > 0 else len(sessions)):
                    n = min(len(pred_maps), args.max_frames) if args.max_frames > 0 else len(pred_maps)
                    frames = [
                        render_frame(archive["pressure_gt"][t], archive["pressure_pred"][t], session_id, "V2T", t, len(pred_maps), args.fps)
                        for t in range(0, n, max(args.stride, 1))
                    ]
                    media = cli_common.media_path(out_dir, f"{session_id}_{safe_model}_V2T", args.gen)
                    media.parent.mkdir(parents=True, exist_ok=True)
                    if args.gen == "gif":
                        cli_common.write_gif(frames, media, cli_common.viz_fps(args.fps, args.stride))
                    else:
                        cli_common.write_mp4(frames, media, cli_common.viz_fps(args.fps, args.stride))
                per_model.setdefault(entry["name"], []).append(row)
                rows.append(row)
            except Exception as exc:
                rows.append({"model": entry["name"], "session": session_id, "status": "invalid", "reason": str(exc)})
        export_count += 1

    fieldnames = ["model", "session", "status", "reason", "n_valid_frames", "T_mae", "T_rmse", "T_corr", "pressure_force_mae", "pressure_force_rmse", "pressure_force_r2", "pressure_cop_error_left", "pressure_cop_error_right", "pressure_cop_error_mean", "contact_f1", "contact_acc", "contact_recall", "air_recall"]
    with (out_dir / "tgen_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    aggregate = {}
    for model, model_rows in per_model.items():
        ok = [row for row in model_rows if row.get("status") == "ok"]
        aggregate[model] = {
            "sessions": len(ok),
            **{key: float(np.average([row[key] for row in ok], weights=[row["n_valid_frames"] for row in ok])) if ok else None for key in ("T_mae", "T_rmse", "T_corr", "pressure_force_mae", "pressure_force_rmse", "pressure_force_r2", "pressure_cop_error_mean", "contact_f1")},
        }
    (out_dir / "tgen_report.json").write_text(json.dumps({"source": "standardized_v2t_archives", "split": args.split, "models": aggregate}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    return main_archives(args) if args.source == "archives" else main_infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
