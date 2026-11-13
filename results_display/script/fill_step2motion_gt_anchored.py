#!/usr/bin/env python3
"""Fill GT-anchored trajectory metrics for Step2Motion four-fold comparisons.

Step2Motion has no world anchor (root = cumsum(displacement) from origin), so
the capability gate correctly blanks GlobalMPJPE / traj_ATE / GroundPen in the
anchor-free protocol.  This script adds the same GT-anchored trajectory reads
the main model already reports in its "（GT Anchored）" columns:

  one GT start anchor per session, at the first finite frame, using E9's
  ``align_gt_start``: GT root position + hip-line yaw when both hip vectors are
  non-degenerate, otherwise position only (the archive pads frame 0).  The
  whole predicted sequence is rigidly aligned once (E9 Step2Motion arm uses the
  same function on the same inferred trajectory); height is also translated for
  the 3D reads.  GlobalMPJPE, RootAbs (traj_ATE) and GroundPen are then
  computed on the aligned sequence with the canonical solver.

Only the three trajectory keys are added; every existing per-session metric and
the other summary values are left byte-identical.  The JSON records
``translation_anchor`` / ``gt_anchored_metrics`` provenance.

Usage:
  python results_display/script/fill_step2motion_gt_anchored.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Baselines"))

from Baselines.utils.gt_loading import protocol_gt, valid_mask  # noqa: E402
from Baselines.utils.solver import METRICS, aggregate, metrics  # noqa: E402

sys.path.insert(0, str(ROOT / "results/diagnostics/20261108_traj_sceneocc"))
import trajectory_scene_occlusion as T  # noqa: E402

FOLDS = {
    "fS5S6S11": "crossfit_woS13/fS5S6S11_manifest.jsonl",
    "fS5S11S12": "crossfit_woS13/fS5S11S12_manifest.jsonl",
    "fS6S7": "crossfit_woS13/fS6S7_manifest.jsonl",
    "fS5S6S8": "crossfit_woS13/fS5S6S8_manifest.jsonl",
}
METRIC_KEYS = ("global_mpjpe_mm", "root_ate_mm", "ground_pen_mm")
WINDOW = 20


def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def anchor_start(pred: np.ndarray, gt: np.ndarray, names,
                 sid: str) -> tuple[np.ndarray, int, str]:
    """E9 ``align_gt_start`` in 3D (same S2M arm as the E9 trajectory table).

    E9 works in y-up ``(x, z, -y)``: the root ground position (x,z) and, when
    both the predicted hip line and the GT fit-forward are non-degenerate, the
    hip-line yaw are aligned at the first finite frame.  When the first-frame
    predicted hip vector is zero (the archive pads frame 0), the function falls
    back to translation only — exactly E9's behaviour.  The height (y in y-up)
    is also translated for the 3D GlobalMPJPE / GroundPen reads.
    """
    yup = T.zup_to_yup(np.asarray(pred, dtype=np.float64))
    gt_yup = T.zup_to_yup(np.asarray(gt, dtype=np.float64))
    index = {str(name): i for i, name in enumerate(names)}
    left, right = index.get("left_hip"), index.get("right_hip")
    if left is None or right is None:
        raise ValueError("S2M archive lacks left_hip/right_hip joint names")
    hips = T.zup_to_yup(yup[:, left] - yup[:, right])
    gt_hips = T.gt_hips_from_fit(sid, len(yup))
    first = int(np.where(np.isfinite(yup).all(axis=(1, 2))
                         & np.isfinite(gt_yup).all(axis=(1, 2)))[0][0])
    p0, g0 = yup[first, 0, [0, 2]], gt_yup[first, 0, [0, 2]]
    ph, gh = hips[first], gt_hips[first]
    out = yup.copy()
    if np.linalg.norm(ph) > 1e-6 and np.linalg.norm(gh) > 1e-6:
        a = np.degrees(np.arctan2(ph[0], ph[1]))
        b = np.degrees(np.arctan2(gh[0], gh[1]))
        theta = np.radians(b - a)
        c, s = np.cos(theta), np.sin(theta)
        rel = yup[:, :, [0, 2]] - p0
        out[:, :, 0] = c * rel[:, :, 0] + s * rel[:, :, 1] + g0[0]
        out[:, :, 2] = -s * rel[:, :, 0] + c * rel[:, :, 1] + g0[1]
        mode = "position_yaw"
    else:
        out[:, :, [0, 2]] = yup[:, :, [0, 2]] + (g0 - p0)
        mode = "position_only"
    # 3D extension for GlobalMPJPE / GroundPen: align the pelvis height too.
    out[:, :, 1] = yup[:, :, 1] - yup[first, 0, 1] + gt_yup[first, 0, 1]
    # back to z-up: (x, -z, y)
    aligned = np.stack([out[..., 0], -out[..., 2], out[..., 1]], axis=-1)
    return aligned, 1, mode


def main() -> int:
    updated = 0
    for fold, manifest_rel in FOLDS.items():
        manifest = load_manifest(ROOT / "AnysoleWorkspace/protocol/splits" / manifest_rel)
        path = ROOT / f"results/baselines/Step2Motion/metrics/{fold}_comparison.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        fps = float(payload.get("evaluation_fps") or 40.0)
        n_anchor = 0
        mismatched_keep = []
        for row in payload.get("sessions", []):
            if row.get("status") != "ok":
                continue
            sid = row["session_id"]
            meta = manifest[sid]
            pred_path = (ROOT / "results/baselines/Step2Motion/predictions"
                         / f"gait_model_smpl24_{fold}/eval_motion" / f"{sid}.npz")
            archive = np.load(pred_path, allow_pickle=True)
            pred = np.asarray(archive["joint_xyz_world"], dtype=np.float64)
            pred_mask = np.asarray(archive["valid_mask"], dtype=bool)
            pred_names = tuple(str(x) for x in archive["joint_names"])
            gt, gt_names = protocol_gt(meta, "smpl24")
            n_eval = min(len(pred), len(gt), int(meta["n_frames"]))
            keep = valid_mask(meta, n_eval)
            keep &= pred_mask[:n_eval]
            if int(keep.sum()) != int(row.get("n_valid_frames") or 0):
                mismatched_keep.append((sid, int(keep.sum()), row.get("n_valid_frames")))
            anchored, count, mode = anchor_start(pred[:n_eval], gt[:n_eval],
                                                 pred_names, sid)
            n_anchor += count
            values = metrics(
                anchored, gt[:n_eval], keep, fps,
                protocol="smpl24", names=pred_names,
            )
            for key in METRIC_KEYS:
                row[key] = float(values[key])
            reasons = {}
            try:
                reasons = json.loads(row.get("metric_reasons") or "{}")
            except ValueError:
                pass
            for key in METRIC_KEYS:
                reasons[key] = "gt_anchored: gt_start_position_yaw (E9 S2M arm)"
            row["metric_reasons"] = json.dumps(reasons, ensure_ascii=False, sort_keys=True)
            row["translation_anchor"] = f"gt_start_{mode}"
            row["gt_anchor_count"] = count
        summary = aggregate(payload.get("sessions", []))
        payload.setdefault("summary", {}).update(
            {key: float(summary[key]) for key in METRIC_KEYS})
        payload["summary"]["n_valid_frames"] = int(summary["n_valid_frames"])
        modes = {row.get("translation_anchor") for row in payload.get("sessions", [])
                 if row.get("status") == "ok"}
        payload["translation_anchor"] = sorted(modes)
        payload["gt_anchored_metrics"] = list(METRIC_KEYS)
        payload["gt_anchor_count"] = n_anchor
        payload["gt_anchor_rule"] = (
            "E9 align_gt_start: first finite frame, GT root position + hip-line "
            "yaw when both hip vectors are non-degenerate, otherwise position "
            "only; height aligned for the 3D reads. One anchor per session."
        )
        if mismatched_keep:
            payload["gt_anchor_keep_mismatch"] = mismatched_keep
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        updated += 1
        print(f"[fill] {fold}: n_sessions={payload.get('n_sessions')} "
              f"anchors={n_anchor} global={payload['summary'].get('global_mpjpe_mm'):.3f} "
              f"root_ate={payload['summary'].get('root_ate_mm'):.3f} "
              f"ground_pen={payload['summary'].get('ground_pen_mm'):.3f}"
              + (f" keep_mismatch={mismatched_keep[:3]}" if mismatched_keep else ""))
    print(f"updated {updated} fold comparison files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
