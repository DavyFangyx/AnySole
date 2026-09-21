"""Protocol-aware comparison evaluator (Test2) for AnySole and BVH baselines.

AnySole predicts native SMPL-24 while Step2Motion predicts Skeleton3/BVH-23.
Those arrays must never be truncated or compared by numeric joint index.  This
evaluator identifies each prediction protocol, selects an explicit 19-joint
semantic intersection, and evaluates against GT expressed in the *same native
protocol* as that prediction.  SMPL is converted to the display/common z-up
coordinate convention by ``motion_io``; BVH remains z-up.

Writes only metrics to ``results_display/r_test2_compare``:
``comparison_per_session.csv`` (full detail), ``comparison_summary.csv``
(model x metrics only) and ``comparison_summary.png`` (table render of the
summary).  Visualization is Test1's job and lives in
``results_display/Test1_visualization``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from utils import cli_common

ROOT = cli_common.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anysole.types import JOINT_NAMES  # noqa: E402
from utils.motion_io import LEGACY_BVH_NAMES, load_motion  # noqa: E402

METRICS = ("MPJPE_mm", "PA_MPJPE_mm", "WMPJPE_mm", "WAMPJPE_mm", "RTE_mm", "Accel_mps2", "Jitter_1e-3_mps2")

# Skeleton3 has one more intermediate spine joint and no SMPL terminal hand
# joints.  The intersection below deliberately excludes ambiguous spine
# levels and the unmatched SMPL hands.  Ordering starts with pelvis because
# metrics() uses joint 0 as the root.
COMMON_JOINTS = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot", "neck", "head",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
SMPL_COMMON_NAMES = {
    name: name for name in COMMON_JOINTS
}
BVH_COMMON_NAMES = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg", "right_knee": "RightLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_foot": "LeftToeBase", "right_foot": "RightToeBase",
    "neck": "Neck", "head": "Head",
    "left_collar": "LeftShoulder", "right_collar": "RightShoulder",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    try:
        out = json.loads(str(value))
        return [int(x) for x in out]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    opener = path.open(encoding="utf-8-sig")
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in opener if line.strip()]
    else:
        rows = list(csv.DictReader(opener))
    selected = []
    for row in rows:
        tags = {x.strip() for x in str(row.get("split_iid", "")).split(",")}
        if split not in tags and str(row.get("split_ood", "")).strip() != split:
            continue
        selected.append(row)
    return selected


def resolve_repo_path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return p
    marker = "/mocap_ori_bvh/"
    normalized = p.as_posix()
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        matches = sorted((ROOT / "AnysoleWorkspace/sources/raw").glob("*/mocap_ori_bvh/" + suffix))
        if matches:
            return matches[0]
    for candidate in (base / p, ROOT / p):
        if candidate.exists():
            return candidate
    return p if p.is_absolute() else ROOT / p


def bvh_joints(path: Path, row: dict[str, str] | None = None) -> np.ndarray:
    """World joints from a BVH; with a manifest row, resampled onto the session grid.

    Without ``row`` the raw BVH is returned as-is.  With ``row`` the motion is
    queried at ``t_mocap = visual_start_s + i/target_fps - offset_s``, the
    mocap<->tactile/video bias compensation used by MotionPRO
    ``prepare_sequences.py`` and ``anysole/data/dataset.py``.  Predictions are
    exported on the session grid, so comparing them frame-by-frame against the
    raw BVH would misalign GT by ``mocap_start_s`` (median ~0.08 s, max ~0.9 s).
    """
    query_t = None
    if row is not None and "visual_start_s" in row and "offset_s" in row:
        n = int(float(row.get("n_frames") or 0))
        if n <= 0:
            raise ValueError("manifest n_frames must be positive for aligned BVH loading")
        fps = float(row.get("target_fps") or 40.0)
        t_grid = float(row["visual_start_s"]) + np.arange(n, dtype=np.float64) / fps
        query_t = t_grid - float(row["offset_s"])
    loaded = load_motion(path, query_t=query_t)
    if loaded["format"] != "bvh":
        raise ValueError(f"expected legacy BVH, got {loaded['format']}: {path}")
    return np.asarray(loaded["joints"], dtype=np.float32)


def _decode_names(values) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v)
                 for v in np.asarray(values).reshape(-1).tolist())


def _infer_protocol(arr: np.ndarray, names: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]]:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"joint array must be (T,J,3), got {arr.shape}")
    if names is not None:
        if len(names) != arr.shape[1]:
            raise ValueError(f"joint_names has {len(names)} entries for J={arr.shape[1]}")
        if set(JOINT_NAMES).issubset(names):
            return "smpl24", names
        if set(LEGACY_BVH_NAMES).issubset(names):
            return "bvh23", names
        raise ValueError(f"unrecognized joint_names protocol: {names}")
    if arr.shape[1] == len(JOINT_NAMES):
        return "smpl24", tuple(JOINT_NAMES)
    if arr.shape[1] == len(LEGACY_BVH_NAMES):
        return "bvh23", tuple(LEGACY_BVH_NAMES)
    raise ValueError(
        f"cannot infer motion protocol from J={arr.shape[1]}; include joint_names"
    )


def array_from_file(path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...], str]:
    if path.suffix.lower() == ".bvh":
        return bvh_joints(path), None, tuple(LEGACY_BVH_NAMES), "bvh23"
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as probe:
            if ({"poses", "trans"}.issubset(probe.files)
                    or {"root_orient", "pose_body", "trans"}.issubset(probe.files)):
                motion = load_motion(path)
                return motion["joints"], None, tuple(motion["names"]), "smpl24"
        data = np.load(path, allow_pickle=False)
        keys = list(data.keys())
        candidates = ("joint_xyz_world", "joints_world", "pred_joints", "joints", "xyz", "pose")
        key = next((k for k in candidates if k in data), None)
        if key is None:
            raise ValueError(f"no joint array in {path}; keys={keys}")
        arr = np.asarray(data[key])
        mask = np.asarray(data["valid_mask"]).reshape(-1) if "valid_mask" in data else None
        names = _decode_names(data["joint_names"]) if "joint_names" in data else None
        protocol, names = _infer_protocol(arr, names)
        return arr, mask, names, protocol
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        protocol, names = _infer_protocol(obj, None)
        return obj, None, names, protocol
    if isinstance(obj, dict):
        for key in ("joint_xyz_world", "joints_world", "pred_joints", "joints", "xyz"):
            if key in obj:
                arr = np.asarray(obj[key])
                names_in = _decode_names(obj["joint_names"]) if "joint_names" in obj else None
                protocol, names = _infer_protocol(arr, names_in)
                mask = np.asarray(obj.get("valid_mask")).reshape(-1) if "valid_mask" in obj else None
                return arr, mask, names, protocol
    raise ValueError(f"unsupported prediction object: {path}")


def select_common_joints(joints: np.ndarray, names: tuple[str, ...], protocol: str) -> np.ndarray:
    """Select the fixed semantic intersection; never align protocols by index."""
    mapping = SMPL_COMMON_NAMES if protocol == "smpl24" else BVH_COMMON_NAMES
    index = {name: i for i, name in enumerate(names)}
    missing = [mapping[key] for key in COMMON_JOINTS if mapping[key] not in index]
    if missing:
        raise ValueError(f"{protocol} missing common joints: {missing}")
    return np.asarray(joints)[:, [index[mapping[key]] for key in COMMON_JOINTS], :]


def protocol_gt(row: dict[str, str], protocol: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if protocol == "bvh23":
        path = resolve_repo_path(row["bvh_path"], ROOT)
        return bvh_joints(path, row), tuple(LEGACY_BVH_NAMES)
    if protocol == "smpl24":
        path = resolve_repo_path(row["smpl_path"], ROOT)
        n = int(float(row["n_frames"]))
        fps = float(row.get("target_fps") or 40.0)
        query_t = (float(row["visual_start_s"])
                   + np.arange(n, dtype=np.float64) / fps
                   - float(row["offset_s"]))
        motion = load_motion(path, query_t=query_t)
        return motion["joints"], tuple(motion["names"])
    raise ValueError(f"unsupported protocol {protocol!r}")


def find_prediction(root: Path, session_id: str, pattern: str | None = None, config_id: str | None = None) -> Path | None:
    if pattern:
        try:
            candidate = root / pattern.format(session_id=session_id, config_id=config_id)
        except (KeyError, IndexError):
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate
    for ext in (".npz", ".npy", ".bvh"):
        exact = root / f"{session_id}{ext}"
        if exact.is_file():
            return exact
    matches = sorted(p for p in root.rglob(f"*{session_id}*") if p.suffix.lower() in (".npz", ".npy", ".bvh"))
    return matches[0] if matches else None


def expand_requested_models(results_root: Path, modals: list[str], config_ids: list[str], contact_methods: list[str]) -> list[dict[str, str]]:
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


def valid_mask(row: dict[str, str], n: int) -> np.ndarray:
    if row.get("valid_frame_indices"):
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray([i for i in parse_list(row["valid_frame_indices"]) if 0 <= i < n], dtype=int)] = True
        return keep
    mask = np.ones(n, dtype=bool)
    for i in parse_list(row.get("fake_frame_indices")):
        if 0 <= i < n:
            mask[i] = False
    return mask


def procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    xs, ys = src.mean(0), dst.mean(0)
    a, b = src - xs, dst - ys
    u, s, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = s.sum() / max((a * a).sum(), 1e-12)
    return scale * (src - xs) @ r + ys


def metrics(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray, fps: float) -> dict[str, float]:
    n = min(len(pred), len(gt), len(keep)); pred, gt, keep = pred[:n], gt[:n], keep[:n]
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = int(keep.sum())
    if not keep.any(): return out
    ppa, gpa = pred - pred[:, :1], gt - gt[:, :1]
    out["MPJPE_mm"] = float(np.linalg.norm(ppa[keep] - gpa[keep], axis=-1).mean() * 1000)
    pa = [np.linalg.norm(procrustes(pred[t], gt[t]) - gt[t], axis=-1).mean() for t in np.flatnonzero(keep)]
    out["PA_MPJPE_mm"] = float(np.mean(pa) * 1000)
    ids = np.flatnonzero(keep)
    if len(ids) >= 2:
        ref = ids[:2]; aligned = pred.copy(); aligned = pred - pred[ref].mean(0) + gt[ref].mean(0)
        err = np.linalg.norm(aligned[keep] - gt[keep], axis=-1).mean()
        out["WMPJPE_mm"] = float(err * 1000)
    if len(ids) >= 1:
        aligned = pred - pred[keep].mean((0, 1)) + gt[keep].mean((0, 1))
        out["WAMPJPE_mm"] = float(np.linalg.norm(aligned[keep] - gt[keep], axis=-1).mean() * 1000)
        out["RTE_mm"] = float(np.linalg.norm((aligned[ids[-1], 0] - gt[ids[-1], 0])) * 1000)
    if n >= 3:
        triple = keep[:-2] & keep[1:-1] & keep[2:]
        ap = (pred[:-2] - 2 * pred[1:-1] + pred[2:]) * fps ** 2
        ag = (gt[:-2] - 2 * gt[1:-1] + gt[2:]) * fps ** 2
        if triple.any():
            out["Accel_mps2"] = float(np.linalg.norm(ap[triple] - ag[triple], axis=-1).mean())
            out["Jitter_1e-3_mps2"] = float(np.linalg.norm(ap[triple], axis=-1).mean() * 1000)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = sum(int(r["n_valid_frames"]) for r in rows)
    for k in METRICS:
        vals = [(float(r[k]), int(r["n_valid_frames"])) for r in rows if math.isfinite(float(r[k]))]
        if vals: out[k] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    vals = [float(r["RTE_mm"]) for r in rows if math.isfinite(float(r["RTE_mm"]))]
    out["RTE_mm"] = float(np.mean(vals)) if vals else float("nan")
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "AnysoleWorkspace/manifests/session_manifest.csv")
    ap.add_argument("--models-config", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--auto-scan", action="store_true", help="Scan results/ model directories instead of YAML")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=cli_common.DISPLAY_ROOT / "result/r_test2_compare")
    ap.add_argument("--force", action="store_true", help="Rebuild and overwrite existing outputs.")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--modal", default="anysolev1,anysolev1_insole_drift", help="AnySole modal(s), comma-separated")
    ap.add_argument("--contact-method", default="tactile_abs", help="Contact-label scheme(s), comma-separated; model dir is <modal>_<contact-method>")
    ap.add_argument("--config-id", default="VT2M,V2M,T2M", help="AnySole generation configuration(s), comma-separated")
    args = ap.parse_args()
    modals = cli_common.split_csv_arg(args.modal)
    contact_methods = cli_common.split_csv_arg(args.contact_method)
    config_ids = cli_common.split_csv_arg(args.config_id)
    allowed_configs = {"VT2M", "V2M", "T2M"}
    invalid = set(config_ids) - allowed_configs
    if invalid:
        raise SystemExit(f"Unsupported --config-id: {sorted(invalid)}; choices={sorted(allowed_configs)}")
    if args.auto_scan or args.models_config is None:
        configs = {"models": expand_requested_models(args.results_root, modals, contact_methods, config_ids)}
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
    details, summaries = [], []
    for model in configs.get("models", configs):
        name, root = model["name"], resolve_repo_path(model["prediction_root"], ROOT)
        model_rows = []
        for row in manifest:
            sid = row["session_id"]
            pred_path = find_prediction(root, sid, model.get("pattern"), config_id=model.get("config_id"))
            base = {"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "session_id": sid, "subject_id": row.get("subject_id", ""), "action": row.get("action", ""), "split": args.split, "protocol": "", "joint_set": "common19", "status": "ok", "reason": ""}
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
        s = aggregate(model_rows); s.update({"model": name, "variant": model.get("variant", ""), "run_name": model.get("run_name", root.name), "n_sessions": sum(r["status"] == "ok" for r in model_rows), "split_sha256": sha256(args.manifest), "manifest_sha256": sha256(args.manifest), "eval_fps": args.fps, "checkpoint": model.get("checkpoint", "")})
        summaries.append(s)
    detail_fields = ["model", "variant", "run_name", "session_id", "subject_id", "action", "split", "protocol", "joint_set", "n_valid_frames", *METRICS, "status", "reason", "prediction"]
    summary_fields = ["model", *METRICS]
    for filename, fields, rows in (("comparison_per_session.csv", detail_fields, details), ("comparison_summary.csv", summary_fields, summaries)):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    (args.out_dir / "evaluation.log").write_text(
        f"split={args.split}\nsessions={len(manifest)}\nmodels={len(configs.get('models', configs))}\n"
        f"modal={','.join(modals)}\ncontact_method={','.join(contact_methods)}\nconfig_id={','.join(config_ids)}\n"
        f"joint_set=common19\nprotocol_rule=native-protocol GT; explicit semantic-name mapping\n"
        + "".join(f"checkpoint[{row['model']}]={row['checkpoint']}\n" for row in summaries),
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
