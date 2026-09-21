"""Test6: dataset self-check — GT FK self-consistency + mean-pose baseline (no model).

Two model-free diagnostics that anchor Test7-Test9:

A. GT self-consistency: FK(GT 6D pose, GT offsets) must reproduce the stored
   kp_gt up to float rounding, and the torch FK used by train/eval must agree
   with the numpy FK used to build the dataset.  Any systematic offset here
   (m vs mm units, hierarchy, resampling) would cap MPJPE far above zero no
   matter how long the model trains.  Native SMPL betas are converted through
   SMPL_NEUTRAL's shapedirs/J_regressor into shape-dependent rest offsets,
   checked by A3 (units/geometry) and A5 (per-subject template consistency:
   same subject must be identical across actions/samples).
   A1 numpy FK(GT) vs stored kp_gt, A2 torch FK(GT) vs stored kp_gt,
   A3 geometry/unit sanity, A4 native SMPL axis-angle roundtrip, A5 offsets.

B. Mean-pose baseline: predict the train-split mean pose on the test split.
   B1 = mean pose + GT root trajectory (pose-only no-op), B2 = fully static
   mean pose.  If the model's MPJPE is close to B1, conditioning is ignored;
   if B1 is clearly below the model's MPJPE, the model is worse than doing
   nothing (a bug, not undertraining).

Outputs under results_display/Test6_dataset_check/:
    fk_selfcheck.json     A1-A5 per-check max errors and geometry stats
    mean_baseline.json    B1/B2 MPJPE (mm), per-joint breakdown, kp std
    mean_pose.npz         train mean pose (valid 6D + kp relative to Hips)
    input_means.npz       train per-timestep V/T means (consumed by Test7/8)

Usage (run from the repository root):
    python results_display/script/test6_dataset_check.py
    python results_display/script/test6_dataset_check.py --session S10103 --limit-sessions 1 --max-windows 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.smpl_io import smpl24_pose6d_to_poses  # noqa: E402
from scipy.spatial.transform import Rotation as SciRotation  # noqa: E402
from anysole.data.dataset import AnySoleDataset, collate_windows, find_session_dir  # noqa: E402
from anysole.geometry import fk_pose6d, fk_pose6d_np, rot6d_to_rotmat, rotmat_to_6d  # noqa: E402
from anysole.train import load_config  # noqa: E402
from anysole.types import (  # noqa: E402
    FOOT_JOINTS,
    JOINT_NAMES,
    JOINT_PROTOCOL_CHECKSUM,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    T_PHYS_DIM,
    T_RAW_DIM,
    V_FEAT_DIM,
)

MEAN_POSE_NPZ = "mean_pose.npz"
INPUT_MEANS_NPZ = "input_means.npz"

# Unit/geometry sanity ranges (meters).  span = max joint distance from Hips
# (leg/trunk length, ~0.9-1.1 m), not full body height.  Values outside these
# are WARNed, not failed: the human decides whether the pipeline is broken.
GEOM_RANGES = {
    "hips_height_mean_m": (0.6, 1.3),
    "span_max_m": (0.7, 1.6),
    "foot_min_height_m": (-0.1, 0.3),
    "offsets_max_m": (0.3, 1.5),
}

DEFAULT_MODEL_METRICS = cli_common.RESULTS_ROOT / "AnySole" / "anysolev1_joint_and" / "metrics" / "test.json"


def build_dataset(mode: str, config: dict, session_ids, contact_method: str) -> AnySoleDataset:
    return AnySoleDataset(
        mode=mode,
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        session_ids=session_ids,
        contact_method=str(contact_method),
    )


def _session_subject(dataset: AnySoleDataset, session_id: str) -> str:
    meta = json.loads((find_session_dir(dataset.seq_root, session_id) / "align_meta.json").read_text())
    return str(meta.get("subject", ""))


def fk_selfcheck(dataset: AnySoleDataset, max_windows: int) -> dict:
    """Run A1-A5 checks; return a JSON-serializable report."""
    a1_max = 0.0
    a4_max = 0.0
    geometry = {name: [] for name in ("hips_height_mean_m", "span_max_m", "foot_min_height_m", "offsets_max_m", "trans_abs_max_m")}
    subject_offsets = {}
    for session in dataset.sessions:
        pose_gt = session["pose_gt"]
        trans_m = session["trans_global"]
        offsets = session["offsets"]
        parents = session["parents"]
        kp_gt = session["kp_gt"]
        # A1: numpy FK (the dataset build path) must reproduce kp_gt exactly.
        kp_re = fk_pose6d_np(pose_gt, trans_m, offsets, parents)
        a1_max = max(a1_max, float(np.abs(kp_re - kp_gt).max()))
        # A4: native SMPL-24 6D <-> axis-angle export roundtrip.
        aa = smpl24_pose6d_to_poses(pose_gt)
        mats = SciRotation.from_rotvec(aa.reshape(-1, N_JOINTS, 3).reshape(-1, 3)).as_matrix()
        pose2 = np.concatenate([mats[..., :, 0], mats[..., :, 1]], axis=-1).reshape(pose_gt.shape).astype(np.float32)
        trans2 = trans_m
        kp2 = fk_pose6d_np(pose2, trans2, offsets, parents)
        a4_max = max(a4_max, float(np.abs(kp2 - kp_gt).max()))
        # A3: geometry/unit sanity in meters.
        kp = kp_gt.astype(np.float64)
        geometry["hips_height_mean_m"].append(float(kp[:, 0, 1].mean()))
        geometry["span_max_m"].append(float(np.linalg.norm(kp - kp[:, 0:1], axis=-1).max()))
        geometry["foot_min_height_m"].append(float(kp[:, FOOT_JOINTS, 1].min()))
        geometry["offsets_max_m"].append(float(np.abs(offsets).max()))
        geometry["trans_abs_max_m"].append(float(np.abs(trans_m).max()))
        # A5: per-subject bone templates (the "betas" analog), grouped by subject.
        subject_offsets.setdefault(_session_subject(dataset, session["session_id"]), []).append(offsets)

    # A2: torch FK (train/eval path) must agree with the stored numpy FK.
    a2_max = 0.0
    n_checked = 0
    for idx in range(min(len(dataset), max_windows)):
        item = dataset[idx]
        kp_t = fk_pose6d(
            item["pose_gt"].unsqueeze(0),
            item["trans_gt"].unsqueeze(0) + item["trans_anchor"].view(1, 1, 3),
            item["offsets"],
            item["parents"],
        )
        a2_max = max(a2_max, float((kp_t - item["kp_gt"]).abs().max()))
        n_checked += 1

    # A5: within-subject template consistency (same person, different
    # actions/samples) — must be identical.  Session ids are
    # S<subject><action><sample>, so templates are grouped by subject.
    within_max = 0.0
    for offsets_list in subject_offsets.values():
        for i in range(len(offsets_list)):
            for j in range(i + 1, len(offsets_list)):
                within_max = max(within_max, float(np.abs(offsets_list[i] - offsets_list[j]).max()))

    warnings = []
    for name, values in geometry.items():
        low, high = min(values), max(values)
        geometry[name] = {"min_m": low, "mean_m": float(np.mean(values)), "max_m": high}
        if name in GEOM_RANGES:
            lo, hi = GEOM_RANGES[name]
            if low < lo or high > hi:
                warnings.append("%s min=%.4f max=%.4f outside [%.2f, %.2f]" % (name, low, high, lo, hi))
    return {
        "A1_np_fk_vs_stored_kp_gt": {"max_m": a1_max, "threshold_m": 1e-6, "pass": a1_max < 1e-6,
                                    "note": "dataset build path; must be exactly 0"},
        "A2_torch_fk_vs_stored_kp_gt": {"max_m": a2_max, "threshold_m": 1e-4, "pass": a2_max < 1e-4,
                                        "note": "train/eval FK path (float32), %d windows checked" % n_checked},
        "A3_geometry_units": {"values": geometry, "warnings": warnings},
        "A4_pose_motion_roundtrip": {"max_m": a4_max, "threshold_m": 2e-3, "pass": a4_max < 2e-3,
                                     "note": "native SMPL-24 axis-angle <-> 6D"},
        "A5_offsets": {
            "within_subject_max_abs_diff_m": within_max,
            "threshold_m": 1e-3,
            "pass": within_max < 1e-3,
            "n_subjects": len(subject_offsets),
            "note": "same subject's template must be identical across actions/samples",
        },
        "n_sessions": len(dataset.sessions),
    }


def collect_train_stats(dataset: AnySoleDataset) -> dict:
    """Per-timestep means over the train windows; the shared source of the
    Test6 mean-pose baseline and the Test7/Test8 'mean input'."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_windows)
    tw = dataset.window_length
    pose_sum = torch.zeros(POSE_DIM)
    kp_abs_sum = torch.zeros(N_JOINTS, 3)
    kp_rel_sum = torch.zeros(N_JOINTS, 3)
    kp_rel_sq = torch.zeros(N_JOINTS, 3)
    v_sum = torch.zeros(tw, V_FEAT_DIM)
    t_sum = torch.zeros(tw, T_RAW_DIM)
    tp_sum = torch.zeros(tw, T_PHYS_DIM)
    n_windows = 0
    n_frames = 0
    for batch in loader:
        pose_sum += batch["pose_gt"].sum(dim=(0, 1))
        kp = batch["kp_gt"]
        rel = kp - kp[:, :, :1]
        kp_abs_sum += kp.sum(dim=(0, 1))
        kp_rel_sum += rel.sum(dim=(0, 1))
        kp_rel_sq += rel.square().sum(dim=(0, 1))
        v_sum += batch["V_feat"].sum(dim=0)
        t_sum += batch["T_raw"].sum(dim=0)
        tp_sum += batch["T_phys"].sum(dim=0)
        n_windows += batch["pose_gt"].shape[0]
        n_frames += batch["pose_gt"].shape[0] * batch["pose_gt"].shape[1]
    # pose/kp sums accumulate over (windows, timesteps) = frames, so they must
    # divide by the frame count; the V/T sums accumulate over windows only
    # (one row per timestep position) and divide by the window count.
    # Dividing the frame sums by n_windows inflated every mean by tw (20x),
    # which was the original B1/B2 blowup (7041 mm instead of ~70 mm).
    frame_div = max(n_frames, 1)
    window_div = max(n_windows, 1)
    mean_6d = (pose_sum / frame_div).numpy().reshape(N_JOINTS, 6)
    # Re-project the arithmetic mean back onto the rotation manifold so the
    # cached mean pose stays a valid 6D rotation when fed into the model.
    valid_6d = rotmat_to_6d(rot6d_to_rotmat(torch.from_numpy(mean_6d))).numpy()
    mean_rel = (kp_rel_sum / frame_div).numpy()
    std = torch.sqrt(torch.clamp(kp_rel_sq / frame_div - (kp_rel_sum / frame_div).square(), min=0.0)).numpy()
    return {
        "mean_pose_6d": valid_6d.astype(np.float32),
        "mean_pose_6d_raw": mean_6d.astype(np.float32),
        "mean_kp_rel": mean_rel.astype(np.float32),
        "mean_kp_abs": (kp_abs_sum / frame_div).numpy().astype(np.float32),
        "kp_std": std.astype(np.float32),
        "V_feat_mean": (v_sum / window_div).numpy().astype(np.float32),
        "T_raw_mean": (t_sum / window_div).numpy().astype(np.float32),
        "T_phys_mean": (tp_sum / window_div).numpy().astype(np.float32),
        "n_windows": n_windows,
        "n_frames": n_frames,
    }


def save_stats(out_dir: Path, stats: dict) -> None:
    np.savez_compressed(
        out_dir / MEAN_POSE_NPZ,
        pose_6d=stats["mean_pose_6d"],
        kp_rel=stats["mean_kp_rel"],
        kp_std=stats["kp_std"],
        motion_protocol=np.asarray(MOTION_PROTOCOL),
        joint_protocol_checksum=np.asarray(JOINT_PROTOCOL_CHECKSUM),
    )
    np.savez_compressed(
        out_dir / INPUT_MEANS_NPZ,
        V_feat=stats["V_feat_mean"],
        T_raw=stats["T_raw_mean"],
        T_phys=stats["T_phys_mean"],
    )


def ensure_mean_pose(npz_path, config: dict, contact_method: str):
    """Load the cached train mean pose, or compute and cache it (Test7)."""
    npz_path = Path(npz_path)
    if npz_path.is_file():
        with np.load(npz_path) as data:
            protocol = str(np.asarray(data.get("motion_protocol", "")))
            checksum = str(np.asarray(data.get("joint_protocol_checksum", "")))
            if protocol == MOTION_PROTOCOL and checksum == JOINT_PROTOCOL_CHECKSUM:
                return data["pose_6d"], data["kp_rel"]
        log.warning(
            "stale mean-pose cache {} has a different motion/tree protocol; rebuilding",
            npz_path,
        )
    log.info("missing {}; computing train-split statistics now", npz_path)
    out_dir = npz_path.parent
    train_dataset = build_dataset("train", config, None, contact_method)
    stats = collect_train_stats(train_dataset)
    save_stats(out_dir, stats)
    return stats["mean_pose_6d"], stats["mean_kp_rel"]


def ensure_input_means(npz_path, config: dict, contact_method: str) -> dict:
    """Load the cached train V/T means, or compute and cache them (Test8)."""
    npz_path = Path(npz_path)
    if npz_path.is_file():
        data = np.load(npz_path)
        return {"V_feat": data["V_feat"], "T_raw": data["T_raw"], "T_phys": data["T_phys"]}
    log.info("missing {}; computing train-split statistics now", npz_path)
    out_dir = npz_path.parent
    train_dataset = build_dataset("train", config, None, contact_method)
    stats = collect_train_stats(train_dataset)
    save_stats(out_dir, stats)
    return {"V_feat": stats["V_feat_mean"], "T_raw": stats["T_raw_mean"], "T_phys": stats["T_phys_mean"]}


def mean_baseline(dataset: AnySoleDataset, stats: dict) -> dict:
    """B1: constant train-mean pose + GT root; B2: fully static mean pose."""
    mean_rel = torch.from_numpy(stats["mean_kp_rel"])
    mean_abs = torch.from_numpy(stats["mean_kp_abs"])
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_windows)
    b1_sum, b2_sum, count = 0.0, 0.0, 0
    joint_sum = np.zeros(N_JOINTS, dtype=np.float64)
    for batch in loader:
        kp = batch["kp_gt"]
        err1 = torch.linalg.vector_norm(mean_rel[None, None] + kp[:, :, :1] - kp, dim=-1)
        err2 = torch.linalg.vector_norm(mean_abs[None, None] - kp, dim=-1)
        b1_sum += float(err1.sum().item())
        b2_sum += float(err2.sum().item())
        joint_sum += err1.sum(dim=(0, 1)).double().numpy()
        count += err1.numel()
    std_mm = float(stats["kp_std"].mean()) * 1000.0
    return {
        "B1_mean_pose_gt_root_MPJPE_mm": b1_sum / count * 1000.0,
        "B2_static_mean_MPJPE_mm": b2_sum / count * 1000.0,
        "per_joint_B1_MPJPE_mm": {JOINT_NAMES[i]: float(joint_sum[i] / (count // N_JOINTS) * 1000.0) for i in range(N_JOINTS)},
        "mean_kp_std_mm": std_mm,
        "train_windows": stats["n_windows"],
        "n_test_joint_samples": count,
    }


def print_summary(fk: dict, baseline: dict) -> None:
    print("== Test6 A: GT FK self-consistency ==")
    for key in ("A1_np_fk_vs_stored_kp_gt", "A2_torch_fk_vs_stored_kp_gt", "A4_pose_motion_roundtrip"):
        check = fk[key]
        status = "PASS" if check["pass"] else "FAIL"
        print("  [%s] %s max=%.3e m (threshold %.3e)  %s" % (status, key, check["max_m"], check["threshold_m"], check["note"]))
    a5 = fk["A5_offsets"]
    print("  [%s] A5_offsets within-subject max=%.3e m (threshold %.3e, %d subjects)  %s"
          % ("PASS" if a5["pass"] else "FAIL", a5["within_subject_max_abs_diff_m"], a5["threshold_m"], a5["n_subjects"], a5["note"]))
    print("  A3 geometry: " + "; ".join("%s [%.3f, %.3f] m" % (k, v["min_m"], v["max_m"]) for k, v in fk["A3_geometry_units"]["values"].items()))
    for warning in fk["A3_geometry_units"]["warnings"]:
        print("  [WARN] %s" % warning)
    print("== Test6 B: mean-pose baseline ==")
    print("  B1 mean pose + GT root : %.1f mm" % baseline["B1_mean_pose_gt_root_MPJPE_mm"])
    print("  B2 static mean pose    : %.1f mm" % baseline["B2_static_mean_MPJPE_mm"])
    print("  GT kp std (signal)     : %.1f mm" % baseline["mean_kp_std_mm"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test6: dataset self-check (GT FK self-consistency + mean-pose baseline).")
    cli_common.add_common_args(
        parser,
        session=True,
        split_csv=True,
        split=True,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "Test6_dataset_check",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "anysole" / "configs" / "v1.yaml"))
    parser.add_argument("--contact-method", type=str, default="tactile_abs",
                        help="Contact-label scheme for dataset loading (labels do not affect Test6 checks).")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Cap the number of test sessions (0 = all).")
    parser.add_argument("--max-windows", type=int, default=64, help="Windows checked by A2 (torch FK).")
    parser.add_argument("--model-metrics", type=str, default=str(DEFAULT_MODEL_METRICS),
                        help="Model metrics JSON for the baseline comparison; '' skips it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    split_csv = Path(cli_common.resolve_path(args.split_csv))
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / name for name in ("fk_selfcheck.json", "mean_baseline.json", "mean_pose.npz", "input_means.npz")]
    if cli_common.outputs_ready(outputs) and not args.force:
        log.info("Skip Test6: outputs already exist under {} (use --force to overwrite)", out_dir)
        return 0

    test_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    if args.limit_sessions:
        test_ids = test_ids[: args.limit_sessions]
    log.info("test sessions ({})", len(test_ids))

    test_dataset = build_dataset("eval", config, test_ids, args.contact_method)
    fk = fk_selfcheck(test_dataset, args.max_windows)

    log.info("computing train-split statistics (mode=train, full split)")
    train_dataset = build_dataset("train", config, None, args.contact_method)
    stats = collect_train_stats(train_dataset)
    save_stats(out_dir, stats)
    baseline = mean_baseline(test_dataset, stats)

    (out_dir / "fk_selfcheck.json").write_text(json.dumps(fk, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "mean_baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(fk, baseline)

    if args.model_metrics:
        metrics_path = Path(cli_common.resolve_path(args.model_metrics))
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            vt2m = (payload.get("metrics") or {}).get("VT2M") or {}
            model_mpjpe = vt2m.get("MPJPE")
            if model_mpjpe is not None:
                b1 = baseline["B1_mean_pose_gt_root_MPJPE_mm"]
                print("  model VT2M MPJPE (%s): %.1f mm" % (metrics_path, model_mpjpe))
                if model_mpjpe > b1:
                    print("  -> model is %.1f mm WORSE than 'mean pose + GT root'; conditioning is not being used (bug territory)" % (model_mpjpe - b1))
                else:
                    print("  -> model beats the mean-pose baseline by %.1f mm (conditioning has some effect)" % (b1 - model_mpjpe))
        else:
            log.warning("model metrics not found: {}", metrics_path)
    log.info("wrote fk_selfcheck.json, mean_baseline.json, mean_pose.npz, input_means.npz under {}", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
