#!/usr/bin/env python3
"""Build the MotionPRO adapter_v3_bvh24 input tree from the 2026-10-24 raw.

Why this version exists
-----------------------
The 2026-10-24 mocap delivery replaced the legacy BVH-23 / y-up export with a
BVH24 Blender-world z-up export (``motion_smpl24_blender_world_m.bvh``) and a
matching MoSh++ SMPL fit (``*_smpl_new``).  The frozen ``adapter_v1`` /
``adapter_v2_carpet`` trees were built before that swap, so they carry stale
foot tracks (carpet), stale SMPL supervision (``smpl.npy`` /
``keypoints.npy``) and F6 contact computed from the old BVH.  This builder
regenerates every mocap-dependent input from the current shared facts and
leaves the old trees untouched.

Output (per session)::

    model_inputs/MotionPRO/adapter_v3_bvh24/cam3/<date>/<subject>/<sid>/
        pressure.npz     fixed-world carpet rendered from current BVH24 tracks
        contact.npy      private binary F6 foot-loss weight (cols 6/7)
        smpl.npy         new SMPL fit interpolated on the shared mocap grid
        keypoints.npy    SMPL-24 joints from smpl.npy (native gen_kps recipe)
        frame_id.npy     shared frame id
        color/           symlinks to the shared RGB frames
        feature_hrnet.pth  symlink to adapter_v1 (RGB-only visual feature)
        artifact.json    provenance

Geometry is fitted on the canonical ``splits.csv`` train split only (same
rule as adapter_v2_carpet: evaluation identities never enter the fit).

Usage::

    python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_v3_bvh24 --fit-only
    python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_v3_bvh24 --sessions S11011,S6012
    python -m AnysoleWorkspace.tool.adapters.MotionPRO.build_v3_bvh24
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT / "AnysoleWorkspace"
for path in (ROOT, ROOT / "Baselines/MotionPRO"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402
from lib.util.carpet_v2 import fit_geometry, footprint_bounds, render_carpet  # noqa: E402

VERSION = "adapter_v3_bvh24"
OUTPUT_ROOT = WORKSPACE / "model_inputs" / "MotionPRO" / VERSION
SPLIT_CSV = WORKSPACE / "protocol/splits/default/splits.csv"
V1_ROOT = WORKSPACE / "model_inputs" / "MotionPRO" / "adapter_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_groups() -> dict[str, list[str]]:
    rows = list(csv.DictReader(SPLIT_CSV.open(encoding="utf-8-sig", newline="")))
    return {
        key: sorted({(row.get(key) or "").strip() for row in rows if (row.get(key) or "").strip()})
        for key in ("train", "val", "test")
    }


def output_dir(session: dict, cam: int = 3) -> Path:
    session_dir = Path(session["dir"])
    date, subject = session_dir.parts[-3], session_dir.parts[-2]
    return OUTPUT_ROOT / f"cam{cam}" / date / subject / session["session_id"]


def fit_or_load_geometry(cam: int = 3) -> dict:
    path = OUTPUT_ROOT / f"geometry_cam{cam}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    groups = split_groups()
    bounds = []
    for index, sid in enumerate(groups["train"]):
        session = A.load_shared_session(sid, cam)
        tracks = A.foot_tracks(A._context(session))
        bounds.append(footprint_bounds(tracks))
        if (index + 1) % 20 == 0:
            print(f"fit train footprints {index + 1}/{len(groups['train'])}", flush=True)
    geometry = fit_geometry(bounds, groups["train"], cam)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
    print(f"[geometry] {path} shape_hw={geometry['shape_hw']} entry_hw={geometry['entry_hw']}")
    return geometry


def build_session(session: dict, geometry: dict, cam: int = 3) -> dict:
    sid = session["session_id"]
    out_dir = output_dir(session, cam)
    if (out_dir / "artifact.json").is_file():
        print(f"[skip] {sid}: already built ({out_dir})", flush=True)
        return json.loads((out_dir / "artifact.json").read_text(encoding="utf-8"))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing inputs: {out_dir}")
    out_dir.mkdir(parents=True)
    date, subject = Path(session["dir"]).parts[-3], Path(session["dir"]).parts[-2]
    ctx = A._context(session)
    tracks = A.foot_tracks(ctx)
    left48 = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right48 = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    carpet = render_carpet(tracks, left48, right48, geometry, A._paint_footprint)
    contact = A.soft_f6_contact(ctx)
    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    valid = np.asarray(session["frames"]["valid"], dtype=bool)
    fake = np.asarray(session["frames"]["fake"], dtype=bool)

    np.savez_compressed(
        out_dir / "pressure.npz",
        pressure=carpet.astype(np.float32),
        geometry_sha256=np.asarray(geometry["sha256"]),
        world_origin_xy_m=np.asarray(geometry["origin_xy_m"]),
        cell_m=np.asarray(geometry["cell_m"]),
    )
    np.save(out_dir / "contact.npy", contact)
    np.save(out_dir / "frame_id.npy", frame_id)
    smpl_provenance = A.build_smpl_npy(session, cam, out_dir=out_dir)
    color_stats = A.write_color_symlinks(session, out_dir)

    # RGB-only visual feature is unchanged by the mocap swap: link adapter_v1.
    v1_dir = V1_ROOT / f"cam{cam}" / date / subject / sid
    feature_src = v1_dir / "feature_hrnet.pth"
    if not feature_src.is_file():
        raise FileNotFoundError(feature_src)
    (out_dir / "feature_hrnet.pth").symlink_to(feature_src)

    source_nonzero = np.any(left48 > 0, axis=1) | np.any(right48 > 0, axis=1)
    blank = ~np.any(carpet != 0.0, axis=(1, 2))
    lost = source_nonzero & blank
    artifact = {
        "schema_version": "model_input.motionpro.fixed_world_carpet.v3",
        "producer": "AnysoleWorkspace/tool/adapters/MotionPRO/build_v3_bvh24.py",
        "adapter_version": VERSION,
        "session_id": sid,
        "cam_id": cam,
        "geometry_sha256": geometry["sha256"],
        "geometry_file": str(OUTPUT_ROOT / f"geometry_cam{cam}.json"),
        "source_pressure": str(Path(session["dir"]) / "pressure_48.npz"),
        "source_pressure_sha256": sha256_file(Path(session["dir"]) / "pressure_48.npz"),
        "source_frames_sha256": sha256_file(Path(session["dir"]) / "frames.npz"),
        "source_bvh": session["meta"]["source_files"]["bvh"],
        "source_bvh_sha256": session["meta"].get("source_hashes", {}).get("bvh", ""),
        "source_smpl": session["meta"]["source_files"]["smpl"],
        "source_smpl_sha256": session["meta"].get("source_hashes", {}).get("smpl", ""),
        "smpl_npy": smpl_provenance,
        "color": color_stats,
        "feature_hrnet": {
            "mode": "symlink",
            "path": str(feature_src),
            "sha256": sha256_file(feature_src),
            "reason": "RGB-only frontend; unchanged by the 2026-10-24 mocap swap",
        },
        "stats": {
            "frames": int(len(carpet)),
            "blank_frames": int(blank.sum()),
            "valid_frames": int((valid & ~fake).sum()),
            "source_nonzero_valid_lost": int((lost & valid & ~fake).sum()),
            "contact_rate_left": float(contact[:, 6].mean()),
            "contact_rate_right": float(contact[:, 7].mean()),
        },
    }
    (out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(f"[build] {sid} frames={len(carpet)} blank={int(blank.sum())} "
          f"lost={int(lost.sum())}", flush=True)
    return artifact


def run_gen_kps(cam: int = 3) -> None:
    """Native gen_kps recipe over the new smpl.npy tree (same code path)."""
    baselines = ROOT / "Baselines/MotionPRO"
    if str(baselines) not in sys.path:
        sys.path.insert(0, str(baselines))
    from lib.util.gen_kps import generate_kps  # noqa: E402

    smpl_files = sorted(str(path) for path in
                        (OUTPUT_ROOT / f"cam{cam}").glob("*/*/*/smpl.npy"))
    if not smpl_files:
        raise FileNotFoundError(f"no smpl.npy under {OUTPUT_ROOT / f'cam{cam}'}")
    generate_kps(smpl_files, skip_existing=True)


def install_task_config(geometry: dict, cam: int = 3) -> Path:
    """Write the opt-in task config consumed by hydra (never overwrites)."""
    from omegaconf import OmegaConf

    template = ROOT / f"Baselines/MotionPRO/config/task/imagepressure2smpl_carpet_v2_cam{cam}.yaml"
    cfg = OmegaConf.load(template)
    cfg.name = "imagepressure2smpl_carpet_v3_bvh24"
    cfg.cam_id = cam
    root_uri = f"model-input://MotionPRO/{VERSION}/cam{cam}"
    for key in ("seq_root", "train_data_dir", "eval_data_dir"):
        cfg[key] = root_uri
    cfg.pressure_shape = geometry["shape_hw"]
    cfg.pressure_hw = geometry["entry_hw"]
    cfg.pressure_resize_mode = geometry["resize_mode"]
    cfg.pressure_geometry_sha256 = geometry["sha256"]
    cfg.unified_prediction_dir = f"results://baselines/MotionPRO/predictions/{VERSION}/eval_motion"
    cfg.checkpoint_path = None
    cfg.train_continue = False
    installed = ROOT / f"Baselines/MotionPRO/config/task/imagepressure2smpl_carpet_v3_bvh24_cam{cam}.yaml"
    if installed.exists():
        raise FileExistsError(installed)
    OmegaConf.save(cfg, installed)
    print(f"[config] {installed}")
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--sessions", default="", help="comma list; default all canonical-split sessions")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--install-config", action="store_true")
    parser.add_argument("--skip-kps", action="store_true")
    args = parser.parse_args()

    geometry = fit_or_load_geometry(args.cam)
    if args.fit_only:
        return 0
    groups = split_groups()
    wanted = [s.strip() for s in args.sessions.split(",") if s.strip()] \
        if args.sessions else sorted(set(groups["train"]) | set(groups["val"]) | set(groups["test"]))
    records = []
    for index, sid in enumerate(wanted, start=1):
        session = A.load_shared_session(sid, args.cam)
        records.append(build_session(session, geometry, args.cam))
        if index % 20 == 0:
            print(f"[progress] {index}/{len(wanted)}", flush=True)
    if not args.skip_kps:
        run_gen_kps(args.cam)
    if args.install_config:
        install_task_config(geometry, args.cam)
    summary = {
        "status": "PASS",
        "version": VERSION,
        "geometry": geometry,
        "sessions": len(records),
        "frames": int(sum(r["stats"]["frames"] for r in records)),
        "blank_frames": int(sum(r["stats"]["blank_frames"] for r in records)),
        "source_nonzero_valid_lost": int(sum(r["stats"]["source_nonzero_valid_lost"] for r in records)),
        "contact_rate_left": float(np.mean([r["stats"]["contact_rate_left"] for r in records])),
        "contact_rate_right": float(np.mean([r["stats"]["contact_rate_right"] for r in records])),
    }
    (OUTPUT_ROOT / ("validation_cam%d.json" % args.cam)).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
