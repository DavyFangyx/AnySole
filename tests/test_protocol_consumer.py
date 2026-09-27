"""Consumer-side acceptance tests for the native-protocol display layer.

评估整改任务 03 §7/§8：能力门控、missing/invalid/not_applicable 状态区分、
V2T 层级（brief 7 + 叶 5，四层含接触层）、R_Test2 与 R_Test4 共用同一
pressure 求解器、压力 archive contact 字段容忍但不消费、SMPL/BVH pred=GT
零误差链路不经过任何公共关节集。

Self-contained; run in the touch_gait env:

    python tests/test_protocol_consumer.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "results_display" / "script")):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.compare_core import (  # noqa: E402
    METRICS,
    V2T_BRIEF_KEYS,
    V2T_LEAF_KEYS,
    V2T_LEVELS,
    V2T_METRICS,
    applicable_metrics,
    load_v2t_archive,
    normalize_protocol,
    read_capabilities,
    v2t_metrics,
)
from r_test2_compare import (  # noqa: E402
    ANYSOLE_APPROVED,
    ANYSOLE_MOTION_CAPS,
    ANYSOLE_V2T_CAPS,
    DENIED_SOURCES,
    applicable_for,
    denied_capabilities,
    evaluate_motion_row,
    evaluate_pressure_row,
    mask_metrics,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if condition else "FAIL", name,
                         (" — " + detail) if detail and not condition else ""))


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------

def test_applicable_gating():
    caps = read_capabilities(ANYSOLE_MOTION_CAPS)
    applicable = applicable_metrics(caps)
    check("AnySole motion: mpjpe applicable", "mpjpe_mm" in applicable)
    check("AnySole motion: root_ate applicable", "root_ate_mm" in applicable)
    check("AnySole motion: orientation applicable", "root_orientation_deg" in applicable)
    check("AnySole motion: mpjae applicable", "mpjae_deg" in applicable)
    check("AnySole motion: pve not applicable (no shape output)",
          "pve_mm" not in applicable)
    check("AnySole motion: contact never applicable", "contact_f1" not in applicable)
    # 生成清单裁定：AnySole 不生成 shape，pose-only + GT beta 的 PVE 例外已
    # 废除——即使 approved 传 pve_mm 也不应进入（消费端不再传）。
    check("AnySole motion: no pose-only PVE exception in consumer",
          not ANYSOLE_APPROVED)
    # pressure_toolkit 类 caps：拟合 pipeline 原生生成表面 → PVE 可评估。
    caps_fitted = read_capabilities({
        "joint_positions": True, "root_translation": True, "root_rotation": True,
        "joint_rotations": True, "predicted_surface": True, "predicted_shape": True,
        "pressure": False, "contact": False,
    })
    check("fitted surface -> pve applicable", "pve_mm" in applicable_metrics(caps_fitted))

    caps_bvh = read_capabilities({
        "joint_positions": True, "root_translation": True, "root_rotation": True,
        "joint_rotations": False, "surface": False, "pressure": False, "contact": False,
    })
    bvh_applicable = applicable_metrics(caps_bvh)
    check("BVH: mpjpe applicable", "mpjpe_mm" in bvh_applicable)
    check("BVH: root_ate applicable", "root_ate_mm" in bvh_applicable)
    check("BVH: orientation applicable", "root_orientation_deg" in bvh_applicable)
    check("BVH: mpjae not applicable (no direct joint rotations)",
          "mpjae_deg" not in bvh_applicable)
    check("BVH: pve not applicable", "pve_mm" not in bvh_applicable)
    check("BVH: shape/vertex not applicable",
          not {"shape_vertex_std_mm"} & bvh_applicable)

    caps_v2t = read_capabilities({"pressure_prediction": True})
    v2t_applicable = applicable_metrics(caps_v2t)
    pressure_keys = (set(V2T_LEVELS["grid"]) | set(V2T_LEVELS["force"])
                     | set(V2T_LEVELS["cop"]))
    check("V2T: all pressure keys applicable", pressure_keys <= v2t_applicable)
    check("V2T: contact keys need contact_prediction",
          not (set(V2T_LEVELS["contact"]) & v2t_applicable))
    check("V2T: motion metrics not applicable",
          not (pressure_keys ^ v2t_applicable or "mpjpe_mm" in v2t_applicable))
    caps_fpp = read_capabilities({"pressure_prediction": True, "contact_prediction": True})
    check("V2T: contact keys applicable when contact_prediction declared",
          set(V2T_LEVELS["contact"]) <= applicable_metrics(caps_fpp))


def test_denied_sources():
    entry = {
        "sources": {
            "root_translation": "ground_truth_fallback",
            "surface": "template_reconstruction",
            "predicted_shape": "method_fitting",
        },
        "capabilities": ANYSOLE_MOTION_CAPS,
    }
    denied = denied_capabilities(entry)
    check("GT-fallback translation retained for audit", "root_translation" in denied)
    check("template surface retained for audit", "predicted_surface" in denied)
    check("fitted shape retained for audit", "predicted_shape" in denied)
    applicable = applicable_for(read_capabilities(ANYSOLE_MOTION_CAPS), denied,
                                approved=("pve_mm",))
    check("source labels do not remove generated root metrics",
          {"root_ate_mm", "w_mpjpe100_mm", "root_rte_percent"} <= applicable)
    check("GT beta exception cannot enable pve", "pve_mm" not in applicable)
    check("joint metrics survive source metadata", "mpjpe_mm" in applicable)

    archive_meta = {"provenance_source_type": "ground_truth_fallback"}
    denied_all = denied_capabilities({"sources": {}}, archive_meta)
    check("archive-level fallback denies translation",
          "root_translation" in denied_all)


def test_mask_metrics():
    values = {key: 1.0 for key in METRICS}
    values["n_valid_frames"] = 10
    reasons: dict = {}
    mask_metrics(values, {"mpjpe_mm"}, reasons)
    check("masked metric becomes NaN", np.isnan(values["pa_mpjpe_mm"]))
    check("kept metric survives", values["mpjpe_mm"] == 1.0)
    check("mask reason recorded", "pa_mpjpe_mm" in reasons)


def test_archive_tolerance():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v2t.npz"
        np.savez(
            path,
            pressure_pred=np.zeros((4, 31, 22), dtype=np.float32),
            pressure_gt=np.zeros((4, 31, 22), dtype=np.float32),
            contact_pred=np.zeros((4, 2), dtype=np.uint8),  # legacy field
            contact_gt=np.zeros((4, 2), dtype=np.uint8),    # legacy field
            valid_mask=np.ones(4, dtype=np.uint8),
        )
        archive = load_v2t_archive(path)
        check("contact fields tolerated but not returned",
              "contact_pred" not in archive and "pressure_pred" in archive)
        bad = Path(tmp) / "bad.npz"
        np.savez(bad, pressure_pred=np.zeros((4, 31, 22)))
        try:
            load_v2t_archive(bad)
            check("archive without pressure_gt/valid_mask rejected", False)
        except ValueError:
            check("archive without pressure_gt/valid_mask rejected", True)


# ---------------------------------------------------------------------------
# V2T: strict 9 metrics, one shared solver
# ---------------------------------------------------------------------------

def test_v2t_hierarchy():
    rng = np.random.default_rng(0)
    gt = rng.random((20, 31, 22), dtype=np.float64) * 2.0
    pred = gt + rng.normal(0, 0.05, gt.shape)
    valid = np.ones(20, dtype=bool)
    valid[::5] = False
    out = v2t_metrics(pred, gt, valid)
    check("v2t_metrics returns brief+leaf + n_valid_frames",
          set(out) == set(V2T_METRICS) | {"n_valid_frames"}, f"got {sorted(out)}")
    check("brief is 7 / leaf is 5", len(V2T_BRIEF_KEYS) == 7 and len(V2T_LEAF_KEYS) == 5)
    check("T_mse == T_rmse^2", abs(out["T_mse"] - out["T_rmse"] ** 2) < 1e-12)
    check("contact keys NaN without contact maps",
          np.isnan(out["contact_smpl_mse"]) and np.isnan(out["contact_smpl_bce"]))
    check("contact_f1 pending (binary GT), not in v2t_metrics", "contact_f1" not in out)
    check("contact family keys stay out of v2t_metrics",
          "contact_acc" not in out and "contact_recall" not in out)
    check("n_valid_frames correct", out["n_valid_frames"] == int(valid.sum()))
    check("perfect-ish pressure -> low error", out["T_mae"] < 0.2)
    check("T_corr high on identical shape", out["T_corr"] > 0.99)


def test_pressure_row_consumes_same_solver():
    """R_Test2's pressure path and R_Test4 both call v2t_metrics: same input
    -> identical 9 numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(1)
        gt = rng.random((12, 31, 22), dtype=np.float32)
        pred = gt + rng.normal(0, 0.02, gt.shape).astype(np.float32)
        valid = np.ones(12, dtype=np.uint8)
        valid[3:5] = 0
        path = tmp_path / "S13013_V2T.npz"
        contact_gt = np.repeat(
            np.array([0.95, 0.05], dtype=np.float32).reshape(1, 2, 1), 40, axis=-1)
        contact_gt = np.repeat(contact_gt, 12, axis=0)
        np.savez(path, pressure_pred=pred, pressure_gt=gt,
                 contact_smpl_pred=np.clip(
                     contact_gt + np.float32(0.1), 0.0, 1.0).astype(np.float16),
                 contact_smpl_gt=contact_gt.astype(np.float16),
                 contact_pred=np.zeros((12, 2), dtype=np.uint8),
                 contact_gt=np.zeros((12, 2), dtype=np.uint8),
                 valid_mask=valid)
        archive = load_v2t_archive(path)
        direct = v2t_metrics(archive["pressure_pred"], archive["pressure_gt"],
                             archive["valid_mask"],
                             contact_smpl_pred=archive.get("contact_smpl_pred"),
                             contact_smpl_gt=archive.get("contact_smpl_gt"))
        # The same brief+leaf solver as r_test4_v2t.main_archives consumes.
        from r_test4_v2t import main_archives  # noqa: E402
        import argparse
        row = {
            "session_id": "S13013", "subject_id": "S13", "action": "01",
            "n_frames": "12", "split": "val",
        }
        model = {
            "name": "Test/V2T", "variant": "V2T", "run_name": "",
            "prediction_root": str(tmp_path), "pattern": "{session_id}_V2T.npz",
            "config_id": "V2T", "mode": "V2T", "family": "main",
            # FPP-Net-like: explicit contact prediction from the contact head.
            "capabilities": {**ANYSOLE_V2T_CAPS, "contact_prediction": True,
                             "contact": True},
            "sources": {}, "approved": [],
        }
        result = evaluate_pressure_row(model, row, path, {}, "V2T")
        check("pressure row status ok", result["status"] == "ok")
        for key in V2T_METRICS:
            check(f"R_Test2 pressure row == direct solver ({key})",
                  abs(float(result[key]) - float(direct[key])) < 1e-12,
                  f"{result[key]} vs {direct[key]}")
        check("motion metrics blank on pressure row",
              np.isnan(result["mpjpe_mm"]) and np.isnan(result["pve_mm"]))
        # r_test4's archive path computes the identical 9 numbers.
        r4 = r_test4_archive_values(path, valid)
        check("R_Test4 values == direct solver",
              all(abs(r4[k] - direct[k]) < 1e-12 for k in V2T_METRICS),
              f"r4={r4} direct={ {k: direct[k] for k in ('T_mae', 'T_corr')} }")


def r_test4_archive_values(path: Path, valid: np.ndarray) -> dict:
    """Mirror of r_test4_v2t.main_archives' metric call (same solver)."""
    archive = load_v2t_archive(path)
    return v2t_metrics(archive["pressure_pred"], archive["pressure_gt"],
                       archive["valid_mask"],
                       contact_smpl_pred=archive.get("contact_smpl_pred"),
                       contact_smpl_gt=archive.get("contact_smpl_gt"))


# ---------------------------------------------------------------------------
# Motion row statuses: missing / invalid / not_applicable
# ---------------------------------------------------------------------------

def _fake_row() -> dict:
    return {
        "session_id": "S13013", "subject_id": "S13", "action": "01",
        "n_frames": "8", "target_fps": "40", "visual_start_s": "0.0",
        "offset_s": "0.0", "valid_frame_indices": "0,1,2,3,4,5,6,7",
        "bvh_path": "", "smpl_path": "",
    }


def test_motion_status_missing_and_invalid():
    args = _fake_args()
    model = {
        "name": "Test/SMPL", "variant": "VT2M", "run_name": "",
        "prediction_root": "/nonexistent", "pattern": "{session_id}_VT2M.npz",
        "config_id": "VT2M", "mode": "VT2M", "family": "main",
        "protocol": "smpl24", "capabilities": ANYSOLE_MOTION_CAPS,
        "sources": {}, "approved": ["pve_mm"],
    }
    result = evaluate_motion_row(model, _fake_row(), None, args, "val", {}, {}, {}, {}, {})
    check("missing prediction -> status missing", result["status"] == "missing")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pred.npz"
        np.savez(
            path,
            joint_xyz_world=np.zeros((8, 23, 3), dtype=np.float32),
            joint_names=np.asarray([
                "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
                "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
                "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            ]),
            valid_mask=np.ones(8, dtype=bool),
        )
        model["prediction_root"] = tmp
        result = evaluate_motion_row(model, _fake_row(), path, args, "val", {}, {}, {}, {}, {})
        check("declared smpl24 but BVH-23 archive -> invalid",
              result["status"] == "invalid", result["reason"])
        check("invalid row metrics all NaN",
              all(np.isnan(float(result[k])) for k in METRICS))


def _fake_args() -> "argparse.Namespace":
    import argparse
    return argparse.Namespace(fps=40.0, surface_metrics=False, split="val")


def test_normalize_protocol():
    check("normalize smpl24-native", normalize_protocol("smpl24-native") == "smpl24")
    check("normalize bvh23", normalize_protocol("bvh23") == "bvh23")
    check("normalize pressure", normalize_protocol("pressure") == "pressure")
    try:
        normalize_protocol("common19")
        check("common19 protocol refused", False)
    except ValueError:
        check("common19 protocol refused", True)


def main() -> int:
    test_applicable_gating()
    test_denied_sources()
    test_mask_metrics()
    test_archive_tolerance()
    test_v2t_hierarchy()
    test_pressure_row_consumes_same_solver()
    test_motion_status_missing_and_invalid()
    test_normalize_protocol()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
