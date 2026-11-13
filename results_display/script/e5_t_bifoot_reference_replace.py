#!/usr/bin/env python3
"""Replace E5 T-only / T2M reference values with the shared BiFoot branch.

The T branch of the main model has no trajectory head; BiFoot is its dedicated
trajectory branch.  E5 used the legacy T2M root head as the T reference.  This
post-process rewrites that reference with the T+BiFoot values:

  * 2D occlusion / time gap: T-only ignores visual input, so the reference is
    the full clean test read (same across all levels);
  * scene occlusion: the reference is computed on the E5 scene session subset
    (session_detail_test.json T2M sessions, n_frames weights).

Protocol: GT first-frame position anchor + world-frame direct displacement
accumulation (the E2 T-block protocol).  `traj_ATE` keeps the table's RootAbs
meaning by using the 3D-RMS variant.  PA / ankle-foot MPJPE are pose metrics
and are left unchanged.

Usage:
  python results_display/script/e5_t_bifoot_reference_replace.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "results" / "diagnostics" / "20261111_variant_sweep"
E5 = REPO / "results_display" / "Etest_crossfit_v8"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = ("V8_Cv2", "V8_Cv4", "V8_Cv5", "V8_Cv6", "V8_Cv7", "V8_Cv8")
E5_DIAG = REPO / "results" / "diagnostics" / "20261107_e5_sceneocc_full"


def load_rows(model, fold):
    path = DIAG / "clean_e2_t_bifoot" / model / ("clean_e2_%s.json" % fold)
    return json.loads(path.read_text(encoding="utf-8"))


def full_agg(model, fold):
    rows = load_rows(model, fold)
    w = [r["n_valid"] for r in rows]
    return {k: float(np.average([r[k] for r in rows], weights=w))
            for k in ("GlobalMPJPE", "traj_ATE_3d_rms",
                      "GroundPen", "FootSliding")}


def scene_agg(model, fold):
    sd = json.loads((E5_DIAG / ("%s_scene" % fold) / "metrics"
                     / "session_detail_test.json").read_text(encoding="utf-8"))
    weights = {r["session_id"]: r["n_frames"]
               for r in sd["conditions"]["T2M"]}
    rows = [r for r in load_rows(model, fold) if r["session"] in weights]
    w = [weights[r["session"]] for r in rows]
    return {k: float(np.average([r[k] for r in rows], weights=w))
            for k in ("GlobalMPJPE", "traj_ATE_3d_rms",
                      "GroundPen", "FootSliding")}


def ms(values):
    a = np.asarray([v for v in values if v is not None], float)
    return "%.1f±%.1f" % (a.mean(), a.std())


def replace_scene_pooled():
    path = (REPO / "results_display" / "Etest_crossfit_v8_archive"
            / "E5_SceneOcc_legacy" / "Full_Table1_SceneOcc_Pooled.csv")
    if not path.is_file():
        print("[e5] skip legacy SceneOcc (archived): %s" % path)
        return
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    cache = {(m, f): scene_agg(m, f) for m in MODELS for f in FOLDS}
    metric_map = {
        "GlobalMPJPE": "GlobalMPJPE",
        "traj_ATE": "traj_ATE_3d_rms",
        "FootSliding": "FootSliding",
        "GroundPen": "GroundPen",
    }
    for row in body:
        if row[idx["条件"]] != "T2M":
            continue
        model, fold = row[idx["模型"]], row[idx["折"]]
        key = metric_map.get(row[idx["指标"]])
        if key is None or (model, fold) not in cache:
            continue
        row[idx["值"]] = "%.3f" % cache[(model, fold)][key]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e5] %s" % path)


def replace_visual_summary():
    clean_vals = {f: full_agg("V8_Cv5", f) for f in FOLDS}
    scene_vals = {f: scene_agg("V8_Cv5", f) for f in FOLDS}
    clean_ms = {k: ms([clean_vals[f][k] for f in FOLDS])
                for k in ("GlobalMPJPE", "traj_ATE_3d_rms",
                          "GroundPen", "FootSliding")}
    scene_ms = {k: ms([scene_vals[f][k] for f in FOLDS])
                for k in ("GlobalMPJPE", "traj_ATE_3d_rms",
                          "GroundPen", "FootSliding")}
    clean_mean = {k: float(np.mean([clean_vals[f][k] for f in FOLDS]))
                  for k in clean_ms}
    scene_mean = {k: float(np.mean([scene_vals[f][k] for f in FOLDS]))
                  for k in scene_ms}

    # Main table: metric x model, only the T-only rows change.
    path = (E5 / "E5_VisualDegradation_Summary"
            / "Main_Table_E5_VisualDegradation_Main.csv")
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        if row[idx["模型"]] != "T-only":
            continue
        metric = row[idx["指标"]]
        if metric == "traj_ATE":
            for col in ("2D 0%", "2D 50%", "2D 90%",
                        "Gap 0%", "Gap 40%", "Gap 60%"):
                row[idx[col]] = "%.1f" % clean_mean["traj_ATE_3d_rms"]
            row[idx["Scene clean"]] = "%.1f" % scene_mean["traj_ATE_3d_rms"]
            row[idx["Scene occl."]] = "%.1f" % scene_mean["traj_ATE_3d_rms"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e5] %s" % path)

    # Full table: degradation mode x subcase x model.
    path = (E5 / "E5_VisualDegradation_Summary"
            / "Main_Table_E5_VisualDegradation_Full.csv")
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    for row in body:
        if row[idx["模型"]] != "T-only":
            continue
        mode = row[idx["退化模式"]]
        vals = scene_ms if mode == "场景遮挡" else clean_ms
        row[idx["GlobalMPJPE (mm)"]] = vals["GlobalMPJPE"]
        row[idx["FootSliding (mm)"]] = vals["FootSliding"]
        row[idx["GroundPen (mm)"]] = vals["GroundPen"]
        row[idx["traj_ATE (mm)"]] = vals["traj_ATE_3d_rms"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([header] + body)
    print("[e5] %s" % path)
    print("  clean T-only:", {k: round(v, 1) for k, v in clean_mean.items()})
    print("  scene T-only:", {k: round(v, 1) for k, v in scene_mean.items()})


def main() -> int:
    replace_scene_pooled()
    replace_visual_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
