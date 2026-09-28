#!/usr/bin/env python
"""T1-A V5 wrapper: run the native MotionPRO visual chain over adapter sessions.

The three upstream scripts are the frozen reference implementations in
``Baselines_old/MotionPRO/lib/util/`` (read-only; this wrapper never edits them,
it only invokes them as subprocesses):

  1. ``gen_bbox.py``           -- mmdet YOLOX-X on ``color/*.jpg``  -> ``bbox.npy`` (T, 8)
  2. ``gen_kps.py``            -- smplx SMPL (CPU) on ``smpl.npy``  -> ``keypoints.npy`` (T, 24, 3)
  3. ``gen_image_feature.py``  -- CLIFF HRNet48 on color+bbox       -> ``feature_hrnet.pth`` (T, 2048)

Two environment facts drive the subprocess setup (verified on this host):

  * The scripts do ``os.chdir(Path(__file__).resolve().parents[2])`` at import
    time, so the cwd is forced to ``Baselines_old/MotionPRO``; the wrapper sets
    it explicitly anyway and exports ``ANYSOLE_WORKSPACE`` so
    ``lib/util/workspace.py`` resolves the same workspace as the adapter.
  * ``gen_bbox.py`` imports mmdet/mmcv, which only exist in the ``bbox_scan``
    conda env; ``gen_kps.py`` and ``gen_image_feature.py`` need smplx/torch,
    which the ``touch_gait`` env has.  The wrapper therefore takes two
    interpreters (``--bbox-python`` / ``--python``).

The scripts walk ``<seq_root>/<date>/<subject>/<session>/`` and read
``color/*.jpg`` + ``smpl.npy`` next to each other, which is exactly the layout
the adapter writes (``write_color_symlinks``).  To run a subset without editing
the scripts, ``--sessions`` builds a shard root of symlinked session dirs and
points the scripts at it; without ``--sessions`` the whole adapter cam root is
processed (the full run -- expect hours: gen_image_feature is the slow step).

This wrapper is smoke-tested on S5091 only; the full 140-session chain is not
run from here automatically.

Usage::

    # smoke, one session, GPU 6
    conda run -n touch_gait python run_visual_chain.py --sessions S5091 --cuda-device 6

    # full camera run (all sessions under the adapter cam3 root)
    conda run -n touch_gait python run_visual_chain.py --cuda-device 6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402

BASELINES_OLD = REPO_ROOT / "Baselines_old"
UTIL_DIR = BASELINES_OLD / "MotionPRO" / "lib" / "util"
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
DEFAULT_TOUCH_PYTHON = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
DEFAULT_BBOX_PYTHON = "/data/fangyuxuan/miniconda3/envs/bbox_scan/bin/python"
STEPS = ("bbox", "kps", "feature")
SCRIPTS = {"bbox": "gen_bbox.py", "kps": "gen_kps.py", "feature": "gen_image_feature.py"}


def session_dir(session_id: str) -> Path:
    return A.model_input_dir(session_id)


def build_shard_root(sessions: list[str], dest: Path) -> Path:
    """Mirror <date>/<subject>/<session> with symlinks to the real session dirs.

    The upstream scripts discover work by walking the sequence root, so a shard
    is the only way to run a subset without patching them.  ``os.walk`` in
    ``findAllFilesWithSpecifiedName`` is called with ``followlinks=True`` and the
    other two scripts glob/list session dirs directly, so links are traversed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for session_id in sessions:
        real = session_dir(session_id)
        if not real.is_dir():
            raise FileNotFoundError(f"{session_id}: {real} is not a directory "
                                    f"(rebuild it with adapter.py --session {session_id})")
        parts = real.parts
        date, subject = parts[-3], parts[-2]
        target = dest / date / subject / real.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(real, target_is_directory=True)
    return dest


def run_step(step: str, seq_root: Path, python: str, cuda_device: str | None,
             mode: str, cam_id: int, dry_run: bool) -> int:
    script = UTIL_DIR / SCRIPTS[step]
    command = [python, str(script), "--cam-id", str(cam_id), "--seq-root", str(seq_root)]
    if mode == "skip":
        command.append("--skip-existing")
    elif mode == "force":
        command.append("--force")
    env = dict(os.environ)
    env["ANYSOLE_WORKSPACE"] = str(WORKSPACE)
    env["PYTHONUNBUFFERED"] = "1"
    if cuda_device is not None and step != "kps":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""  # gen_kps is CPU-only by design
    print(f"\n=== [{step}] {SCRIPTS[step]} ===")
    print("  cwd :", BASELINES_OLD / "MotionPRO")
    print("  env : ANYSOLE_WORKSPACE=%s CUDA_VISIBLE_DEVICES=%r"
          % (env["ANYSOLE_WORKSPACE"], env["CUDA_VISIBLE_DEVICES"]))
    print("  cmd :", " ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(BASELINES_OLD / "MotionPRO"),
                               env=env, check=False)
    if completed.returncode != 0:
        print(f"  !! {SCRIPTS[step]} exited {completed.returncode}")
    return completed.returncode


def verify(sessions: list[str]) -> dict:
    """Check the three chain artifacts next to each session."""
    report: dict[str, dict] = {}
    for session_id in sessions:
        directory = session_dir(session_id)
        entry: dict = {"dir": str(directory)}
        bbox_path = directory / "bbox.npy"
        if bbox_path.is_file():
            bbox = np.load(bbox_path, allow_pickle=True)
            entry["bbox"] = {"shape": list(bbox.shape), "dtype": str(bbox.dtype),
                             "ok": bbox.ndim == 2 and bbox.shape[1] == 8}
        else:
            entry["bbox"] = {"missing": True}
        keypoints_path = directory / "keypoints.npy"
        if keypoints_path.is_file():
            keypoints = np.load(keypoints_path)
            entry["keypoints"] = {"shape": list(keypoints.shape), "dtype": str(keypoints.dtype),
                                  "ok": keypoints.ndim == 3 and keypoints.shape[1:] == (24, 3),
                                  "finite": bool(np.isfinite(keypoints).all())}
        else:
            entry["keypoints"] = {"missing": True}
        feature_path = directory / "feature_hrnet.pth"
        if feature_path.is_file():
            import torch
            feature = torch.load(feature_path, map_location="cpu")
            if isinstance(feature, (list, tuple)):
                feature = torch.cat([t.reshape(t.shape[0], -1) for t in feature], dim=0)
            entry["feature"] = {"shape": list(feature.shape), "dtype": str(feature.dtype),
                                "ok": feature.ndim == 2 and feature.shape[1] == 2048}
        else:
            entry["feature"] = {"missing": True}
        report[session_id] = entry
    return report


def print_verify(report: dict) -> bool:
    ok = True
    print("\n=== verification ===")
    for session_id, entry in report.items():
        print(f"{session_id}: {entry['dir']}")
        for name in ("bbox", "keypoints", "feature"):
            info = entry[name]
            if info.get("missing"):
                print(f"  [--] {name:<10} missing")
                ok = False
                continue
            mark = "ok" if info.get("ok") else "BAD"
            extra = "" if "finite" not in info else f" finite={info['finite']}"
            print(f"  [{mark}] {name:<10} shape={info['shape']} {info['dtype']}{extra}")
            ok = ok and bool(info.get("ok")) and info.get("finite", True)
    return ok


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Adapter-session wrapper for the native MotionPRO visual chain "
                    "(gen_bbox -> gen_kps -> gen_image_feature)")
    parser.add_argument("--seq-root", type=Path, default=None,
                        help="sequence root to process (default: the adapter cam3 root)")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="smoke/subset mode: build a shard root with these sessions")
    parser.add_argument("--cam-id", type=int, default=3)
    parser.add_argument("--steps", default=",".join(STEPS),
                        help="comma-separated subset of bbox,kps,feature")
    parser.add_argument("--python", default=DEFAULT_TOUCH_PYTHON,
                        help="interpreter for gen_kps / gen_image_feature (smplx env)")
    parser.add_argument("--bbox-python", default=DEFAULT_BBOX_PYTHON,
                        help="interpreter for gen_bbox (mmdet env)")
    parser.add_argument("--cuda-device", default=None,
                        help="physical GPU index exported as CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mode", choices=("skip", "force", "none"), default="skip",
                        help="pass --skip-existing / --force to the scripts")
    parser.add_argument("--shard-root", type=Path, default=None,
                        help="where to build the shard root (default: a temp dir)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for step in steps:
        if step not in SCRIPTS:
            raise SystemExit(f"unknown step {step!r} (expected one of {list(SCRIPTS)})")

    cam_root = Path(args.seq_root) if args.seq_root else Path(A.MODEL_INPUT_ROOT)
    if args.sessions:
        shard = args.shard_root or Path(tempfile.mkdtemp(prefix="motionpro_visual_shard_"))
        seq_root = build_shard_root(args.sessions, Path(shard))
        print(f"shard root: {seq_root}")
        for session_id in args.sessions:
            print(f"  {session_id} -> {session_dir(session_id)}")
    else:
        seq_root = cam_root
        print(f"sequence root: {seq_root} (all sessions -- the full run)")

    if not args.verify_only:
        for step in steps:
            python = args.bbox_python if step == "bbox" else args.python
            code = run_step(step, seq_root, python, args.cuda_device, args.mode,
                            args.cam_id, args.dry_run)
            if code != 0:
                return code

    sessions = args.sessions or []
    if not sessions and not args.verify_only:
        print("\n(verification skipped: pass --sessions to verify a subset)")
    report = {"seq_root": str(seq_root), "steps": steps,
              "sessions": args.sessions, "cuda_device": args.cuda_device,
              "artifacts": verify(sessions) if sessions else {}}
    if sessions:
        ok = print_verify(report["artifacts"]) if not args.dry_run else True
        report["passed"] = ok
    else:
        ok = True
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.out}")
    if args.sessions and args.shard_root is None and not args.dry_run:
        print(f"\n(shard root kept for inspection: {seq_root})")
    print(f"\nfull-run command:\n  {DEFAULT_TOUCH_PYTHON} {Path(__file__).resolve()} "
          f"--seq-root {cam_root} --cuda-device <idle-gpu>")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
