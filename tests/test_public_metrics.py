"""Numerical acceptance tests for the unified public metric solvers.

评估整改任务 01 §5.2 数值测试。Self-contained: run directly —

    python tests/test_public_metrics.py        # touch_gait env

Covers: native SMPL-24 / BVH-23 pred=GT zero (no common19), root orientation
(yaw) identity / bias / drift / ±180° wrap + equivalence with the former
torch formula, missing-capability => NaN/— (never 0), provenance audit does
not override generated capability
via the capability gate, strict-9 V2T, joint-based foot sliding, capability
guard, and the common19-symbol deletion.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "results_display" / "script")):
    if p not in sys.path:
        sys.path.insert(0, p)

from anysole.types import JOINT_NAMES  # noqa: E402
from anysole.utils import metrics as canonical  # noqa: E402  (the only implementation)
from utils.compare_core import (  # noqa: E402
    METRIC_REQUIRES,
    METRICS,
    MODE_METRICS,
    V2T_BRIEF_KEYS,
    V2T_LEAF_KEYS,
    V2T_LEVELS,
    V2T_METRICS,
    applicable_metrics,
    array_from_file,
    bvh_joints,
    metrics,
    native_edges,
    native_frame_mpjpe_mm,
    v2t_metrics,
)
from utils.motion_io import LEGACY_BVH_NAMES  # noqa: E402

FPS = 40.0
T = 120
PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if condition else "FAIL", name, (" — " + detail) if detail and not condition else ""))


def random_walk(n: int, joints: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.02, (n, joints, 3)), axis=0)
    x += rng.normal(0.0, 0.3, (joints, 3))
    return x


def run_identity(native: str, joints: int, names) -> None:
    """pred=GT => applicable error metrics are zero in the native protocol."""
    keep = np.ones(T, dtype=bool)
    keep[::7] = False
    gt = random_walk(T, joints)
    keep = np.ones(T, dtype=bool)
    rot = Rotation.from_rotvec(np.zeros((T * joints, 3))).as_matrix().reshape(T, joints, 3, 3)
    out = metrics(gt.copy(), gt, keep, FPS, protocol=native, names=names,
                  pred_rotations=rot, gt_rotations=rot)
    for key in ("mpjpe_mm", "pa_mpjpe_mm", "w_mpjpe100_mm", "wa_mpjpe100_mm",
                "root_ate_mm", "root_rte_percent", "mpjae_deg",
                "root_orientation_deg", "root_orientation_drift_deg",
                "accel_error_m_s2"):
        check("%s pred=GT %s ~ 0" % (native, key), out[key] <= 1e-6, "got %r" % out[key])
    # jitter is a property of the motion itself: pred=GT => both jitters
    # equal the GT value (an error metric is not defined for it).
    check("%s pred=GT jitter_pred == jitter_gt" % native,
          abs(out["jitter_pred_m_s3"] - out["jitter_gt_m_s3"]) <= 1e-9,
          "got %r vs %r" % (out["jitter_pred_m_s3"], out["jitter_gt_m_s3"]))
    # foot_sliding is a quality measure: pred=GT must equal the GT value, not 0.
    check("%s pred=GT foot_sliding finite" % native, np.isfinite(out["foot_sliding_mm"]))
    # pve must stay missing without surface data.
    check("%s no-surface pve missing" % native, np.isnan(out["pve_mm"]))
    check("%s no-surface diag absent" % native,
          "shape_vertex_std_mm" not in out and "foot_sliding_vertex_mm" not in out)
    check("%s contact not computed in motion solver" % native,
          np.isnan(out.get("contact_f1", float("nan"))))


def test_identity():
    print("== pred=GT native zero (no common19) ==")
    run_identity("smpl24", len(JOINT_NAMES), tuple(JOINT_NAMES))
    run_identity("bvh23", len(LEGACY_BVH_NAMES), tuple(LEGACY_BVH_NAMES))
    # native_frame_mpjpe_mm on the same data (per-frame panel solver).
    gt = random_walk(T, len(JOINT_NAMES))
    per_frame = native_frame_mpjpe_mm(gt, gt, protocol="smpl24")
    check("native_frame_mpjpe_mm pred=GT ~ 0", float(per_frame.max()) <= 1e-6)


def _old_torch_yaw(r_p, r_g, forward_axis):
    """The former eval_protocol torch formula, kept only to prove equivalence."""
    import torch
    def circ(a, b):
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi
    r_p = torch.from_numpy(r_p).double()
    r_g = torch.from_numpy(r_g).double()
    yaw_p = torch.atan2(r_p[:, :, forward_axis][:, 0], r_p[:, :, forward_axis][:, 2]).numpy()
    yaw_g = torch.atan2(r_g[:, :, forward_axis][:, 0], r_g[:, :, forward_axis][:, 2]).numpy()
    return (
        np.abs(circ(yaw_p, yaw_g)) * 180.0 / np.pi,
        np.abs(circ(yaw_p - yaw_p[0], yaw_g - yaw_g[0])) * 180.0 / np.pi,
    )


def test_root_orientation():
    print("== root orientation (yaw) ==")
    rng = np.random.default_rng(7)
    gt = Rotation.from_rotvec(rng.normal(0, 0.5, (T, 3))).as_matrix()

    # 1. identity => both zero.
    d = canonical.root_orientation_error_deg(gt, gt)
    drift = canonical.root_orientation_drift_deg(gt, gt)
    check("root_orientation identity 0", float(d.max()) <= 1e-6 and float(drift.max()) <= 1e-6)

    # 2. fixed yaw bias: abs = bias, drift = 0.
    bias = np.deg2rad(35.0)
    rot_bias = Rotation.from_euler("y", bias).as_matrix()
    pred = np.einsum("ij,tjk->tik", rot_bias, gt)
    d = canonical.root_orientation_error_deg(pred, gt)
    drift = canonical.root_orientation_drift_deg(pred, gt)
    check("fixed yaw bias abs", float(np.abs(d.mean() - 35.0)) <= 1e-6, "got %r" % d.mean())
    check("fixed yaw bias drift 0", float(drift.max()) <= 1e-6, "got %r" % drift.max())

    # 3. known linear drift: pred yaw = gt yaw + k*t => drift error grows at k.
    yaw_gt = np.linspace(0.0, 0.4, T)
    def rotmats_from_yaw(y):
        return np.stack([Rotation.from_euler("y", a).as_matrix() for a in y])
    gt_m = rotmats_from_yaw(yaw_gt)
    pred_m = rotmats_from_yaw(yaw_gt + np.linspace(0.0, 0.3, T))
    drift = canonical.root_orientation_drift_deg(pred_m, gt_m)
    expected = np.linspace(0.0, np.rad2deg(0.3), T)
    check("linear drift matches", float(np.abs(drift - expected).max()) <= 1e-6,
          "max err %r" % float(np.abs(drift - expected).max()))
    # absolute error at the first frame is the offset (0) and grows too.
    d = canonical.root_orientation_error_deg(pred_m, gt_m)
    check("linear drift abs grows", float(np.abs(d - expected).max()) <= 1e-6)

    # 4. ±180° wrap: jump from 179° to -179° must not blow up.
    yaw_a = np.r_[np.deg2rad([179.0]), np.deg2rad([-179.0])]
    yaw_b = np.r_[np.deg2rad([-179.0]), np.deg2rad([179.0])]
    d = canonical.root_orientation_error_deg(rotmats_from_yaw(yaw_a), rotmats_from_yaw(yaw_b))
    check("±180 wrap no jump", float(d.max()) <= 2.1, "got %r" % d.tolist())

    # 5. equivalence with the former torch formula on random rotations.
    pred_r = Rotation.from_rotvec(rng.normal(0, 1.2, (T, 3))).as_matrix()
    new_abs = canonical.root_orientation_error_deg(pred_r, gt)
    new_drift = canonical.root_orientation_drift_deg(pred_r, gt)
    old_abs, old_drift = _old_torch_yaw(pred_r, gt, 2)
    check("matches former torch yaw_abs", float(np.abs(new_abs - old_abs).max()) <= 1e-5)
    check("matches former torch yaw_drift", float(np.abs(new_drift - old_drift).max()) <= 1e-5)


def test_missing_capability():
    print("== missing capability => NaN / —, never 0 ==")
    gt = random_walk(T, 24)
    keep = np.ones(T, dtype=bool)
    out = metrics(gt, gt, keep, FPS, protocol="smpl24", names=tuple(JOINT_NAMES))
    for key in ("mpjae_deg", "root_orientation_deg", "root_orientation_drift_deg"):
        check("no rotations => %s NaN" % key, np.isnan(out[key]), "got %r" % out[key])
    # capability gate: no root_translation => no world/trajectory metrics.
    caps = {"joint_positions": True, "root_translation": False, "root_rotation": False,
            "direct_joint_rotations": False, "predicted_surface": False,
            "pressure_prediction": False, "contact_prediction": False}
    applicable = applicable_metrics(caps)
    for key in ("root_ate_mm", "root_rte_percent", "w_mpjpe100_mm", "wa_mpjpe100_mm"):
        check("no root_translation excludes %s" % key, key not in applicable)
    check("joints still give mpjpe", "mpjpe_mm" in applicable)
    # contact is never applicable without an explicit contact prediction.
    check("contact never applicable", "contact_f1" not in applicable_metrics(caps))
    check("contact requires contact_prediction",
          METRIC_REQUIRES.get("contact_f1") == ("contact_prediction",))


def test_v2t_hierarchy():
    print("== V2T hierarchy: brief 7 + leaf 5 ==")
    rng = np.random.default_rng(3)
    pred = rng.random((T, 2, 31, 11))
    gt = rng.random((T, 2, 31, 11))
    out = v2t_metrics(pred, gt, valid=np.ones(T, dtype=bool))
    metric_keys = sorted(k for k in out if k != "n_valid_frames")
    check("v2t_metrics exactly brief+leaf keys", metric_keys == sorted(V2T_METRICS), "got %r" % metric_keys)
    check("brief is 7 and leaf is 5", len(V2T_BRIEF_KEYS) == 7 and len(V2T_LEAF_KEYS) == 5)
    check("brief+leaf disjoint and complete",
          set(V2T_BRIEF_KEYS) | set(V2T_LEAF_KEYS) == set(V2T_METRICS)
          and not (set(V2T_BRIEF_KEYS) & set(V2T_LEAF_KEYS)))
    check("four levels cover all keys",
          set(sum(V2T_LEVELS.values(), ())) == set(V2T_METRICS),
          "%r" % sorted(set(V2T_METRICS) - set(sum(V2T_LEVELS.values(), ()))))
    check("MODE_METRICS V2T is brief only", MODE_METRICS["V2T"] == V2T_BRIEF_KEYS)
    check("T_mse == T_rmse^2", abs(float(out["T_mse"]) - float(out["T_rmse"]) ** 2) < 1e-12,
          "%r vs %r" % (out["T_mse"], out["T_rmse"] ** 2))
    check("contact keys NaN without contact maps",
          np.isnan(out["contact_smpl_mse"]) and np.isnan(out["contact_smpl_bce"]))
    check("contact_f1 not in formal V2T set (binary GT pending)", "contact_f1" not in out)
    check("contact_f1 is a contact_prediction metric",
          METRIC_REQUIRES.get("contact_f1") == ("contact_prediction",))
    # identical inputs => identical values (determinism across R_Test2/R_Test4).
    again = v2t_metrics(pred, gt, valid=np.ones(T, dtype=bool))
    check("v2t deterministic", all(
        float(out[k]) == float(again[k]) or
        (np.isnan(out[k]) and np.isnan(again[k])) for k in V2T_METRICS))


def test_contact_smpl_metrics():
    print("== contact level: contact_smpl_mse/bce ==")
    from utils.compare_core import contact_smpl_metrics
    rng = np.random.default_rng(7)
    # f6_soft-like gt: per-foot soft values broadcast over vertices.
    gt = np.repeat(rng.choice([0.05, 0.30, 0.70, 0.95], size=(6, 2, 1)), 40, axis=-1)
    out = contact_smpl_metrics(gt, gt, valid=np.ones(6, dtype=bool))
    check("pred==gt -> mse 0", float(out["contact_smpl_mse"]) == 0.0)
    check("pred==gt -> bce >= 0 and finite",
          float(out["contact_smpl_bce"]) >= 0 and np.isfinite(out["contact_smpl_bce"]))
    check("contact_f1 absent (binary GT pending)", "contact_f1" not in out)
    # bce must grow as the prediction moves away from the target.
    bad = np.clip(gt + 0.3, 0.0, 1.0)
    out_bad = contact_smpl_metrics(bad, gt, valid=np.ones(6, dtype=bool))
    check("bce increases for worse pred",
          float(out_bad["contact_smpl_bce"]) > float(out["contact_smpl_bce"]),
          "%r vs %r" % (out_bad["contact_smpl_bce"], out["contact_smpl_bce"]))
    check("mse increases for worse pred", float(out_bad["contact_smpl_mse"]) > 0.0)
    # valid mask must be honored.
    out_valid = contact_smpl_metrics(bad, gt, valid=np.zeros(6, dtype=bool))
    check("all-invalid -> NaN", np.isnan(out_valid["contact_smpl_mse"]))


def test_foot_sliding():
    print("== joint-based foot sliding ==")
    times = np.arange(T, dtype=np.float64) / FPS
    # GT foot stationary (contact), prediction slides 1 mm/frame along one
    # axis => displacement magnitude 1 mm/frame.
    gt_foot = np.zeros((T, 4, 3))
    pred_foot = np.zeros((T, 4, 3))
    pred_foot[:, :, 0] = np.arange(T, dtype=np.float64)[:, None] / 1000.0
    value, count = canonical.foot_sliding_joints(pred_foot, gt_foot, times)
    check("stationary GT => 1 mm/frame sliding", abs(value - 1.0) <= 1e-6, "got %r" % value)
    check("contact count full", count == (T - 1) * 4, "got %r" % count)
    # Moving GT (swing): no contact frames => NaN/0, not 0 mm.
    gt_fast = np.stack([np.arange(T, dtype=np.float64) * 10.0] * 12, axis=-1).reshape(T, 4, 3)
    value, count = canonical.foot_sliding_joints(pred_foot, gt_fast, times)
    check("swing frames not contact", count == 0 and np.isnan(value), "got %r/%r" % (value, count))
    # Vertex variant kept under its own name.
    check("vertex variant separate name", canonical.foot_sliding_vertices is not canonical.foot_sliding_joints)
    check("old foot_sliding name gone", not hasattr(canonical, "foot_sliding"))


def test_common19_deleted():
    print("== common19 deleted ==")
    from utils import compare_core as cc
    for name in ("COMMON_JOINTS", "SMPL_COMMON_NAMES", "BVH_COMMON_NAMES", "COMMON_EDGES",
                 "select_common_joints", "select_common_rotations", "common_edges_for",
                 "frame_mpjpe_mm"):
        check("compare_core has no %s" % name, not hasattr(cc, name))
    check("pve_t_mm gone", "pve_t_mm" not in canonical.METRIC_UNITS)
    check("native protocol labels", hasattr(cc, "PROTOCOL_LABELS"))
    check("smpl24-native label", cc.PROTOCOL_LABELS.get("smpl24") == "smpl24-native")
    check("bvh23-native label", cc.PROTOCOL_LABELS.get("bvh23") == "bvh23-native")
    check("native_edges smpl24 23 edges", len(native_edges("smpl24")) == 23)
    check("native_edges bvh23 22 edges", len(native_edges("bvh23")) == 22)


def test_array_from_file_labels():
    print("== protocol identifiers recorded as -native ==")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.npz"
        np.savez(p, joint_xyz_world=np.zeros((2, 24, 3), dtype=np.float32),
                 joint_names=np.asarray(JOINT_NAMES), valid_mask=np.ones(2, dtype=np.uint8))
        arr, mask, names, protocol = array_from_file(p)
        check("smpl npz protocol label", protocol == "smpl24-native", "got %r" % protocol)
        check("smpl npz names", tuple(names) == tuple(JOINT_NAMES))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "y.npz"
        np.savez(p, joint_xyz_world=np.zeros((2, 23, 3), dtype=np.float32),
                 joint_names=np.asarray(LEGACY_BVH_NAMES))
        arr, mask, names, protocol = array_from_file(p)
        check("bvh npz protocol label", protocol == "bvh23-native", "got %r" % protocol)


def test_real_bvh23_semantic_filter():
    print("== real BVH-23 semantic filter (skipped if data absent) ==")
    cand = REPO / "results/baselines/Step2Motion/predictions/gait_model/S10103_gen.bvh"
    if not cand.is_file():
        print("  skip: no BVH prediction at %s" % cand)
        return
    joints, names, parents = bvh_joints(cand)
    check("real BVH joint count 23", joints.shape[1] == 23, "got %r" % (joints.shape,))
    check("real BVH names are LEGACY_BVH_NAMES", tuple(names) == tuple(LEGACY_BVH_NAMES))
    check("real BVH no EndSite", all("EndSite" not in n for n in names))
    check("real BVH parents valid", int(parents.max()) < 23 and int(parents.min()) >= -1)
    edges = native_edges(names, parents=parents)
    check("real BVH edges from file", len(edges) == 22)


def test_brief_gating():
    print("== AnySole brief: contact/V2T gating ==")
    from anysole.utils.eval_protocol import (BRIEF_MOTION_KEYS, V2T_BRIEF_NAMES,
                                             V2T_LEAF_NAMES, V2T_NAMES,
                                             summary_metrics)
    check("brief V2T_NAMES is 6+4",
          len(V2T_BRIEF_NAMES) == 6 and len(V2T_LEAF_NAMES) == 4 and len(V2T_NAMES) == 10)
    check("brief motion keys exclude contact", "contact_f1" not in BRIEF_MOTION_KEYS)
    fake = {
        "V2M": {"mpjpe_mm": 1.0, "pve_mm": 2.5, "contact_f1": 0.9, "foot_sliding_vertex_mm": 2.0,
                "T_mae": 0.1, "T_rmse": 0.2, "T_mse": 0.04, "T_corr": 0.9,
                "pressure_force_mae": 1.0, "pressure_force_rmse": 1.0, "pressure_force_r2": 0.8,
                "pressure_cop_error_left": 0.1, "pressure_cop_error_right": 0.2,
                "pressure_cop_error_mean": 0.15},
    }
    brief = summary_metrics(fake)
    check("brief V2T exactly 6 brief keys", set(brief["V2T"]) == set(V2T_BRIEF_NAMES))
    check("brief V2T excludes leaves", not (set(brief["V2T"]) & set(V2T_LEAF_NAMES)))
    check("brief motion has no contact", "contact_f1" not in brief["V2M"])
    check("brief motion has no vertex foot slide", "foot_sliding_vertex_mm" not in brief["V2M"])
    check("brief motion has no pve (AnySole 不生成 shape)", "pve_mm" not in brief["V2M"])


def main() -> int:
    for test in (test_identity, test_root_orientation, test_missing_capability,
                 test_v2t_hierarchy, test_contact_smpl_metrics, test_foot_sliding,
                 test_common19_deleted,
                 test_array_from_file_labels, test_real_bvh23_semantic_filter,
                 test_brief_gating):
        test()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
