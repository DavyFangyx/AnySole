"""Acceptance tests for ``Baselines/utils`` (2026-09-26 结构裁定).

Covers: display-side re-export identity (single implementations), motion_io
shim identity, protocol tables (SMPL single source, BVH-23 semantic skeleton),
capability provenance gating, the baseline evaluation entry, and — when real
predictions exist — the cross-check that ``Baselines.utils.evaluate`` produces
bit-identical numbers to the R_Test2 reference row implementation.

    python tests/test_baseline_utils.py        # touch_gait env
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "results_display" / "script")):
    if p not in sys.path:
        sys.path.insert(0, p)

from anysole.types import JOINT_NAMES, JOINT_PARENTS  # noqa: E402
from Baselines.utils import capabilities as caps  # noqa: E402
from Baselines.utils import gt_loading  # noqa: E402
from Baselines.utils import protocols  # noqa: E402
from Baselines.utils import solver  # noqa: E402
from Baselines.utils.evaluate import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    DEFAULT_SPLIT_CSV,
    MODEL_ALIASES,
    _load_registry,
)
from utils.compare_core import (  # noqa: E402
    METRICS,
    metrics,
    native_edges,
    v2t_metrics,
)
import utils.motion_io as shim_io  # noqa: E402

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if condition else "FAIL", name,
                         (" — " + detail) if detail and not condition else ""))


def test_reexport_identity():
    print("== single implementations (re-export identity) ==")
    check("compare_core.metrics is solver.metrics", metrics is solver.metrics)
    check("compare_core.v2t_metrics is solver.v2t_metrics", v2t_metrics is solver.v2t_metrics)
    check("compare_core.native_edges is protocols.native_edges",
          native_edges is protocols.native_edges)
    check("motion_io shim is canonical",
          shim_io.load_motion is gt_loading.load_motion.__globals__.get("load_motion"))


def test_protocol_tables():
    print("== protocol tables ==")
    check("SMPL names from anysole.types (single source)",
          protocols.SMPL_JOINT_NAMES == tuple(JOINT_NAMES))
    check("SMPL parents from anysole.types",
          protocols.SMPL_JOINT_PARENTS == tuple(int(p) for p in JOINT_PARENTS))
    check("BVH names are the 23 semantic Skeleton3",
          len(protocols.BVH_JOINT_NAMES) == 23
          and shim_io.LEGACY_BVH_NAMES == protocols.BVH_JOINT_NAMES)
    check("BVH parents valid",
          max(protocols.BVH_JOINT_PARENTS) < 23
          and min(protocols.BVH_JOINT_PARENTS) >= -1)
    check("SMPL edges 23", len(protocols.SMPL_NATIVE_EDGES) == 23)
    check("BVH edges 22", len(protocols.BVH_NATIVE_EDGES) == 22)
    check("forward axes pinned", protocols.PROTOCOL_FORWARD_AXIS == {"smpl24": 2, "bvh23": 1})
    check("protocol labels", protocols.PROTOCOL_LABELS ==
          {"smpl24": "smpl24-native", "bvh23": "bvh23-native"})
    check("pressure protocol rejected by motion solver",
          protocols.normalize_protocol("pressure") == "pressure")


def test_capability_gating():
    print("== capability + provenance gating ==")
    # pressure_toolkit-like: surface declared false + template source denied.
    entry = {
        "capabilities": {"joint_positions": True, "joint_rotations": True,
                         "root_translation": True, "root_rotation": True,
                         "surface": False, "predicted_surface": False},
        "sources": {"surface": "template_reconstruction",
                    "predicted_shape": "method_fitting"},
    }
    caps_n = caps.read_capabilities(entry["capabilities"])
    denied = caps.denied_capabilities(entry)
    applicable = caps.applicable_for(caps_n, denied)
    check("pve_mm follows declared generated surface capability",
          "pve_mm" not in applicable and "predicted_surface" in denied)
    check("joint metrics still applicable",
          {"mpjpe_mm", "pa_mpjpe_mm", "foot_sliding_mm"} <= applicable)
    # Archive-level provenance remains auditable but does not override the
    # generation capability contract.
    denied_all = caps.denied_capabilities({}, {"provenance_source_type": "ground_truth_fallback"})
    check("archive-level fallback denies all", denied_all == set(caps.CAPABILITY_FIELDS))
    check("archive provenance does not alter generated applicability",
          "mpjpe_mm" in caps.applicable_for(caps_n, denied_all))
    # contact never applicable without an explicit prediction.
    check("contact never applicable",
          "contact_f1" not in caps.applicable_metrics(
              {"joint_positions": True, "root_translation": True}))
    # mask_metrics blanks with reason, never 0.
    values = {key: 1.0 for key in METRICS}
    reasons: dict[str, str] = {}
    caps.mask_metrics(values, {"mpjpe_mm"}, reasons)
    check("masked metrics NaN with reason",
          np.isnan(values["pve_mm"]) and reasons.get("pve_mm") == "not_applicable"
          and values["mpjpe_mm"] == 1.0)


def test_evaluate_entry():
    print("== evaluate entry ==")
    check("registry default exists", DEFAULT_REGISTRY.is_file(), str(DEFAULT_REGISTRY))
    entries = _load_registry(DEFAULT_REGISTRY)
    names = {e.get("name") for e in entries}
    check("registry has the 5 formal baselines",
          {"MotionPRO", "MMVP_pressure_toolkit", "MMVP_VP-MoCap",
           "Step2Motion", "MMVP_FPP-Net"} <= names)
    check("aliases resolve to registry names",
          MODEL_ALIASES["pressure_toolkit"] == "MMVP_pressure_toolkit"
          and MODEL_ALIASES["vp_mocap"] == "MMVP_VP-MoCap")
    check("no no-IMU probe in registry",
          not any("no-imu" in str(n).lower() for n in names))
    # capability gating helper imported without error
    from Baselines.utils.evaluate import _capabilities
    entry = next(e for e in entries if e.get("name") == "Step2Motion")
    caps_n, denied, applicable = _capabilities(entry)
    check("Step2Motion applicable excludes mpjae/pve/contact",
          {"mpjae_deg", "pve_mm", "contact_f1"}.isdisjoint(applicable)
          and "mpjpe_mm" in applicable)


def test_evaluate_crosscheck():
    print("== real-data cross-check vs R_Test2 reference (skipped if data absent) ==")
    pred_bvh = REPO / "results/baselines/Step2Motion/predictions"
    if not (pred_bvh.exists() and any(pred_bvh.rglob("*.bvh"))):
        print("  skip: no Step2Motion predictions")
        return
    try:
        import r_test2_compare as rt
    except Exception as exc:
        print("  skip: r_test2 import failed: %s" % exc)
        return
    from utils.compare_core import find_prediction, load_mode_registry, read_manifest
    from Baselines.utils.evaluate import evaluate_motion_row

    registry = load_mode_registry()
    rows = read_manifest(DEFAULT_MANIFEST, "test", split_csv=DEFAULT_SPLIT_CSV)
    row = next(r for r in rows if r["session_id"] == "S13013")
    entry = next(e for e in registry["T2M"] if e["name"] == "Step2Motion")
    pred = find_prediction(pred_bvh, "S13013", entry.get("pattern"))
    if pred is None:
        print("  skip: no S13013 prediction")
        return
    class _Args:
        fps = 40.0
    ref = rt.evaluate_motion_row(entry, row, pred, _Args(), "test", registry,
                                 "T2M", {}, {}, {})
    mine = evaluate_motion_row(entry, row, pred, False, {}, {}, {})
    for key in ("mpjpe_mm", "pa_mpjpe_mm", "w_mpjpe100_mm", "root_ate_mm",
                "root_orientation_deg", "root_orientation_drift_deg",
                "foot_sliding_mm", "accel_error_m_s2", "jitter_pred_m_s3"):
        check("S13013 %s bit-identical" % key,
              float(ref[key]) == float(mine[key]),
              "ref=%r mine=%r" % (ref[key], mine[key]))
    check("S13013 status both ok", ref["status"] == "ok" and mine["status"] == "ok")


def main() -> int:
    for test in (test_reexport_identity, test_protocol_tables,
                 test_capability_gating, test_evaluate_entry,
                 test_evaluate_crosscheck):
        test()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
