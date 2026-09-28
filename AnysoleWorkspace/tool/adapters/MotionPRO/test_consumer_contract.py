#!/usr/bin/env python
"""T1-A V4 consumer read test: the native MotionPRO loader contract on the
private adapter output.

Mirrors the access pattern of ``Baselines/MotionPRO/lib/dataset/image_pressure.py``
(read-only reference, the class ``__init__`` / ``__getitem__`` at lines 34-118)
against a rebuilt session directory:

  * ``smpl.npy``    ``np.load(..., allow_pickle=True).item()`` -> exactly the four
                    keys ``betas`` (10,), ``body_pose`` (T, 69), ``global_orient``
                    (T, 3), ``transl`` (T, 3), with T == len(frame_id.npy).
  * ``pressure.npz``  ``pressure`` (T, 320, 120) float32, values in [0, 255]
                    (the loader divides by 255).
  * ``frame_id.npy``  equal to the shared facts frame index.
  * ``contact.npy``   (T, 10); non-zero columns exactly [6, 7] (private soft-f6),
                    values drawn from the four soft levels {0, 0.05, 0.30, 0.70, 0.95}.
  * ``artifact.json`` declares the virtual carpet and the smpl.npy provenance.

The test then replays the loader's window walk over whole sessions:
``theta = cat([global_orient, body_pose], dim=1)``, ``pressure / 255``,
``contact[:, :10]``, ``keypoints[:, :22]`` and the tail zero-padding for the last
partial window, asserting every window has the loader's declared shapes.

Usage::

    conda run -n touch_gait python test_consumer_contract.py --session S5091
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402

SMPL_KEYS = ("betas", "body_pose", "global_orient", "transl")
CONTACT_NONZERO_COLUMNS = [6, 7]
SOFT_LEVELS = (0.0, 0.05, 0.30, 0.70, 0.95)
PIXEL_MAX = 255.0
WINDOW_LENGTH = 64  # native image_pressure window_length


def check(condition: bool, message: str, failures: list[str]) -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {message}")
    if not condition:
        failures.append(message)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T1-A V4 consumer read test")
    parser.add_argument("--session", default="S5091")
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    failures: list[str] = []
    session = A.load_shared_session(args.session)
    out_dir = A.model_input_dir(args.session)
    shared_frames = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    print(f"session   : {args.session}")
    print(f"input dir : {out_dir}")
    print(f"shared    : {len(shared_frames)} frames "
          f"[{shared_frames.min()}, {shared_frames.max()}]")

    # ---- 1. smpl.npy, exactly the native loader idiom -----------------------
    print("\n[smpl.npy]")
    smpl_path = out_dir / "smpl.npy"
    check(smpl_path.is_file(), f"{smpl_path.name} exists", failures)
    if not smpl_path.is_file():
        print("\nsmpl.npy missing; aborting")
        return 1
    smpl = np.load(smpl_path, allow_pickle=True).item()
    check(isinstance(smpl, dict), "loads as a dict via allow_pickle=True "
                                 ".item() (the native idiom)", failures)
    check(set(smpl) == set(SMPL_KEYS), f"keys are exactly {list(SMPL_KEYS)} "
                                       f"(got {sorted(smpl)})", failures)
    betas = np.asarray(smpl["betas"])
    body_pose = np.asarray(smpl["body_pose"])
    global_orient = np.asarray(smpl["global_orient"])
    transl = np.asarray(smpl["transl"])
    check(betas.shape == (10,), f"betas shape (10,) (got {betas.shape})", failures)
    check(body_pose.ndim == 2 and body_pose.shape[1] == 69,
          f"body_pose (T, 69) (got {body_pose.shape})", failures)
    check(global_orient.ndim == 2 and global_orient.shape[1] == 3,
          f"global_orient (T, 3) (got {global_orient.shape})", failures)
    check(transl.ndim == 2 and transl.shape[1] == 3,
          f"transl (T, 3) (got {transl.shape})", failures)
    frames_smpl = body_pose.shape[0]
    check(global_orient.shape[0] == frames_smpl and transl.shape[0] == frames_smpl,
          "the four keys agree on T", failures)
    for name, array in (("betas", betas), ("body_pose", body_pose),
                        ("global_orient", global_orient), ("transl", transl)):
        check(array.dtype == np.float32, f"{name} dtype float32 (got {array.dtype})",
              failures)
    check(bool(np.isfinite(body_pose).all()) and bool(np.isfinite(transl).all()),
          "body_pose/transl are finite (no NaN/inf)", failures)
    check(float(np.abs(transl).max()) < 1e3,
          f"transl magnitude is metric (max {float(np.abs(transl).max()):.2f} m)",
          failures)
    if body_pose.shape[1] == 69:
        check(bool(np.allclose(body_pose[:, 66:], 0.0, atol=1e-5)),
              "the 3 hand-pose slots of body_pose are zero (69 = 23*3 SMPL layout)",
              failures)

    # ---- 2. frame_id.npy ----------------------------------------------------
    print("\n[frame_id.npy]")
    frame_id = np.load(out_dir / "frame_id.npy")
    n = len(frame_id)
    check(frame_id.shape == shared_frames.shape,
          f"shape {frame_id.shape} == shared frame index shape {shared_frames.shape}",
          failures)
    check(np.array_equal(frame_id.astype(np.int64), shared_frames),
          "values equal the shared facts frame index (and its order)", failures)
    check(frames_smpl == len(frame_id),
          f"smpl T == len(frame_id) ({frames_smpl} vs {len(frame_id)})", failures)

    # ---- 3. pressure.npz ----------------------------------------------------
    print("\n[pressure.npz]")
    pressure = np.load(out_dir / "pressure.npz")["pressure"]
    check(pressure.shape == (len(frame_id), 320, 120),
          f"pressure {pressure.shape} == (T, 320, 120)", failures)
    check(pressure.dtype == np.float32,
          f"float32 (got {pressure.dtype})", failures)
    check(float(pressure.min()) >= 0.0 and float(pressure.max()) <= PIXEL_MAX,
          f"values in [0, 255] (got [{float(pressure.min()):.1f}, "
          f"{float(pressure.max()):.1f}])", failures)
    scaled = pressure / PIXEL_MAX
    check(float(scaled.max()) <= 1.0, "loader's /255 scaling stays in [0, 1]",
          failures)
    check(len(pressure) == len(frame_id), "pressure T == len(frame_id)", failures)

    # ---- 4. contact.npy -----------------------------------------------------
    print("\n[contact.npy]")
    contact = np.load(out_dir / "contact.npy")
    check(contact.shape == (len(frame_id), 10),
          f"contact {contact.shape} == (T, 10)", failures)
    if contact.ndim == 2 and contact.shape[1] == 10:
        nonzero = np.flatnonzero(np.asarray(contact).any(axis=0)).tolist()
        check(nonzero == CONTACT_NONZERO_COLUMNS,
              f"non-zero columns are exactly {CONTACT_NONZERO_COLUMNS} (got {nonzero})",
              failures)
        values = np.unique(np.round(contact.astype(np.float64), 6))
        check(all(float(v) in SOFT_LEVELS for v in values),
              f"values are the soft-f6 levels {list(SOFT_LEVELS)} "
              f"(got {values.tolist()})", failures)
        per_level = {f"{v:g}": int((np.round(contact.astype(np.float64), 6) == v).sum())
                     for v in values}
        print(f"         level histogram: {per_level}")
        for column in CONTACT_NONZERO_COLUMNS:
            column_values = np.unique(
                np.round(contact[:, column].astype(np.float64), 6)).tolist()
            print(f"         column {column}: {column_values}")
        check(len(pressure) == len(contact), "contact T == len(frame_id)", failures)

    # ---- 5. artifact.json provenance ---------------------------------------
    print("\n[artifact.json]")
    manifest = json.loads((out_dir / "artifact.json").read_text(encoding="utf-8"))
    params = manifest.get("parameters", {})
    check(manifest.get("schema_version") == "model_input.motionpro.insole_adapted.v1",
          f"schema_version {manifest.get('schema_version')} (private MotionPRO contract)",
          failures)
    raster = params.get("raster", {})
    check(raster.get("semantic") == "virtual_tactile_carpet",
          f"raster.semantic == virtual_tactile_carpet (got {raster.get('semantic')!r})",
          failures)
    check(raster.get("long_axis", {}).get("extent_m") == 4.0
          and raster.get("wide_axis", {}).get("extent_m") == 1.5,
          f"carpet axes declared 4.0 x 1.5 m (got "
          f"{raster.get('long_axis', {}).get('extent_m')} x "
          f"{raster.get('wide_axis', {}).get('extent_m')})", failures)
    check(raster.get("legacy_raster_flag") is False,
          f"raster.legacy_raster_flag False (got {raster.get('legacy_raster_flag')!r})",
          failures)
    check(bool(raster.get("frame", {}).get("y_up_to_z_up")),
          "raster.frame documents the y-up -> z-up conversion", failures)
    check(params.get("legacy_raster") is False,
          f"legacy_raster False (got {params.get('legacy_raster')!r})", failures)
    carpet = params.get("carpet_geometry", {})
    check(carpet.get("shape") == [320, 120],
          f"carpet_geometry.shape == [320, 120] (got {carpet.get('shape')})", failures)
    check(carpet.get("extent_m") == [4.0, 1.5],
          f"carpet_geometry.extent_m == [4.0, 1.5] (got {carpet.get('extent_m')})",
          failures)
    footprint = params.get("footprint_geometry", {})
    check(footprint.get("rows") == 4 and footprint.get("cols") == 12,
          f"footprint 4 x 12 cells (got {footprint.get('rows')} x "
          f"{footprint.get('cols')})", failures)
    value_convention = params.get("value_convention", {})
    check("1023" in json.dumps(value_convention) and "255" in json.dumps(value_convention),
          f"value_convention records the clip(raw,0,1023)/1023*255 convention "
          f"(got {json.dumps(value_convention)[:120]})", failures)
    smpl_meta = params.get("smpl_npy", {})
    check(bool(smpl_meta.get("source_uri")) and bool(smpl_meta.get("source_hashes_smpl")),
          "smpl.npy provenance: source URI + source hash recorded", failures)
    check(smpl_meta.get("keys") == list(SMPL_KEYS),
          f"smpl_npy.keys == {list(SMPL_KEYS)} (got {smpl_meta.get('keys')})", failures)
    check(smpl_meta.get("shapes", {}).get("body_pose") == [frames_smpl, 69],
          f"smpl_npy.shapes.body_pose == [{frames_smpl}, 69] "
          f"(got {smpl_meta.get('shapes', {}).get('body_pose')})", failures)
    check("interp" in smpl_meta and "out_of_span_frames" in smpl_meta,
          "smpl.npy interpolation method + out-of-span count recorded", failures)
    contact_meta = params.get("contact", {})
    check(contact_meta.get("nonzero_columns") == CONTACT_NONZERO_COLUMNS,
          f"contact.nonzero_columns == {CONTACT_NONZERO_COLUMNS} "
          f"(got {contact_meta.get('nonzero_columns')})", failures)
    check(contact_meta.get("value_set") == [0.05, 0.30, 0.70, 0.95],
          f"contact.value_set == [0.05, 0.3, 0.7, 0.95] "
          f"(got {contact_meta.get('value_set')})", failures)
    links = params.get("color_links", {})
    present = links.get("n_present", 0)
    check(present == n and links.get("n_missing", 1) == 0,
          f"color/ symlink farm complete: {present}/{n} frames linked "
          f"({links.get('n_linked_this_run')} created this run, "
          f"{links.get('n_missing')} missing)", failures)
    color_dir = out_dir / "color"
    check(color_dir.is_dir() and len(list(color_dir.glob("*.jpg"))) == n,
          f"color/ holds {n} jpgs "
          f"(got {len(list(color_dir.glob('*.jpg'))) if color_dir.is_dir() else 'no dir'})",
          failures)

    # ---- 6. replay the loader's window walk --------------------------------
    print(f"\n[loader walk] window_length={args.window_length}")
    keypoints_path = out_dir / "keypoints.npy"
    keypoints = np.load(keypoints_path) if keypoints_path.is_file() else None
    if keypoints is not None:
        check(keypoints.shape[:2] == (n, 24),
              f"keypoints.npy {keypoints.shape} == (T, 24, 3)", failures)
    else:
        print("  [skip] keypoints.npy not present (produced by run_visual_chain.py)")
    theta = np.concatenate([global_orient, body_pose], axis=1)
    check(theta.shape == (n, 72), f"theta = cat([global_orient, body_pose]) "
                                 f"{theta.shape} == (T, 72)", failures)
    windows = 0
    padded_windows = 0
    for start in range(0, n, args.window_length):
        stop = min(start + args.window_length, n)
        size = stop - start
        item = {
            "pressure": pressure[start:stop],
            "theta": theta[start:stop],
            "trans": transl[start:stop],
            "joint": keypoints[start:stop, :22] if keypoints is not None else None,
            "contact": np.asarray(contact)[start:stop, :10],
        }
        if size < args.window_length:
            # the native loader zero-pads the trailing partial window
            padded_windows += 1
            item = {name: (None if array is None else np.concatenate(
                [array, np.zeros((args.window_length - size,) + array.shape[1:],
                                 array.dtype)])) for name, array in item.items()}
        for name, array in item.items():
            if array is None:
                continue
            if array.shape[0] != args.window_length:
                failures.append(f"window {windows} {name} has {array.shape[0]} frames "
                                f"(expected {args.window_length})")
            if name == "pressure":
                if array.shape[1:] != (320, 120):
                    failures.append(f"window {windows} pressure {array.shape[1:]} "
                                    f"!= (320, 120)")
            elif name == "theta" and array.shape[1] != 72:
                failures.append(f"window {windows} theta {array.shape[1]} != 72")
            elif name == "joint" and array.shape[1:] != (22, 3):
                failures.append(f"window {windows} joint {array.shape[1:]} != (22, 3)")
        windows += 1
    check(not any("window" in f for f in failures),
          f"replayed {windows} windows of the native loader without shape errors",
          failures)
    tail = n % args.window_length
    print(f"         tail partial window: {tail if tail else args.window_length} real "
          f"frames + zero padding ({padded_windows} padded window(s))")

    # ---- 7. verdict ---------------------------------------------------------
    report = {
        "session": args.session,
        "input_dir": str(out_dir),
        "frames": n,
        "failures": failures,
        "passed": not failures,
        "smpl_shapes": {k: list(np.asarray(smpl[k]).shape) for k in SMPL_KEYS},
        "contact_levels": [float(v) for v in np.unique(np.round(contact, 6))],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.out}")
    if failures:
        print(f"\nconsumer contract: FAIL ({len(failures)} checks)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nconsumer contract: PASS ({args.session}, {n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
