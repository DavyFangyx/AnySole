#!/usr/bin/env python3
"""Build the raw/ modality link farm (plan: reports/raw_relink_plan_20261001.md §4).

``raw/{rgb,pressure,bvh,smpl}`` become per-modality filtered views over the
upstream tree, shape-preserving with respect to the frozen manifest URIs
(relative paths inside each entry match ``raw://`` URIs segment by segment).
``raw/{calibration,human_masks}`` stay plain symlinks and are managed by
``tool/workspace.py`` (RAW_SYMLINKS), not by this script.

Glob-safety: every level that is hit by a recursive ``**`` glob (raw/smpl by
``build_raw_index.py``, raw/pressure by ``build_manifest.py``) is a real
directory here; symlinks are only leaves (files) or camera directories under
raw/rgb (never ``**``-traversed).

Usage:
    python3 AnysoleWorkspace/tool/relink_raw.py build [--dates ...] [--cam 3] [--dry-run]
    python3 AnysoleWorkspace/tool/relink_raw.py verify
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
RAW = WORKSPACE / "raw"
UP = Path("/data/lizhe/projects/Tactile")
UP_DATA = UP / "1_Data"
UP_MOCAP = UP / "Mocap"

DEFAULT_DATES = ("20260804", "20260807", "20260808", "20260810")
FARM_ENTRIES = ("rgb", "pressure", "bvh", "smpl")


def _rel(target: Path, link: Path) -> str:
    return os.path.relpath(target, link.parent)


def collect(dates: tuple[str, ...], cams: tuple[str, ...]) -> dict[str, list[tuple[Path, Path]]]:
    """Return {entry: [(link path relative to RAW/entry, absolute target), ...]}."""
    items: dict[str, list[tuple[Path, Path]]] = {name: [] for name in FARM_ENTRIES}

    def add(entry: str, rel_link: Path, target: Path) -> None:
        items[entry].append((rel_link, target))

    for date in dates:
        date_dir = UP_DATA / date
        if not date_dir.is_dir():
            print(f"skip date (missing upstream): {date}", file=sys.stderr)
            continue
        for subject_dir in sorted(date_dir.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("S"):
                continue
            for rec in sorted(subject_dir.glob("rec*")):
                if not rec.is_dir():
                    continue
                rel_rec = Path(date) / subject_dir.name / rec.name
                meta = rec / "meta.json"
                if meta.is_file():
                    add("rgb", rel_rec / "meta.json", meta)
                for cam in cams:
                    cam_dir = rec / cam
                    if cam_dir.is_dir():
                        add("rgb", rel_rec / cam, cam_dir)
                    else:
                        print(f"skip missing camera {cam}: {rec}", file=sys.stderr)
                for side in ("left", "right"):
                    csv_path = rec / f"pressure_{side}.csv"
                    if csv_path.is_file():
                        add("pressure", rel_rec / f"pressure_{side}.csv", csv_path)
        bvh_root = date_dir / "mocap_ori_bvh"
        if bvh_root.is_dir():
            rel_root = Path(date) / "mocap_ori_bvh"
            for calib in sorted(bvh_root.glob("*.bvh")):
                add("bvh", rel_root / calib.name, calib)
            for sid_dir in sorted(bvh_root.iterdir()):
                if not sid_dir.is_dir():
                    continue
                for pattern in ("*.bvh", "*.avi"):
                    for motion_file in sorted(sid_dir.glob(pattern)):
                        add("bvh", rel_root / sid_dir.name / motion_file.name, motion_file)
        mmdd = date[4:]
        smpl_root = UP_MOCAP / mmdd / f"{mmdd}smpl" / "mocap_ori_c3d"
        if smpl_root.is_dir():
            for sid_dir in sorted(smpl_root.iterdir()):
                if not sid_dir.is_dir():
                    continue
                npz = sid_dir / "motion_neutral_smpl.npz"
                if npz.is_file():
                    rel_link = Path(mmdd) / f"{mmdd}smpl" / "mocap_ori_c3d" / sid_dir.name / npz.name
                    add("smpl", rel_link, npz)
    return items


def check_manifest_uris() -> tuple[int, list[str]]:
    """Resolve every video/bvh/smpl URI of the manifest against the farm."""
    from workspace import resolve_uri, WorkspacePathError

    manifest = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
    checked, errors = 0, []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("video_path", "bvh_path", "smpl_path"):
            uri = row.get(field)
            if not uri:
                continue
            checked += 1
            try:
                resolve_uri(uri, must_exist=True)
            except WorkspacePathError as exc:
                errors.append(f"{uri} -> {exc}")
    return checked, errors


def build(dry_run: bool, dates: tuple[str, ...] = DEFAULT_DATES, cams: tuple[str, ...] = ("3",),
          report_path: Path | None = None) -> None:
    items = collect(dates, cams)
    summary = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "dates": list(dates),
        "cameras": list(cams),
        "entries": {name: len(links) for name, links in items.items()},
        "manifest_uri_check": None,
    }
    checked, uri_errors = check_manifest_uris()
    summary["manifest_uri_check"] = {"checked": checked, "errors": uri_errors}
    report_path = report_path or WORKSPACE / "reports" / f"raw_farm_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    if dry_run:
        print(f"[dry-run] would link {sum(len(v) for v in items.values())} files/dirs")
        for name in FARM_ENTRIES:
            print(f"[dry-run] {name}: {len(items[name])} links")
        print(f"[dry-run] manifest URIs: {checked - len(uri_errors)}/{checked} resolve")
        for error in uri_errors:
            print(f"[dry-run] MISSING {error}", file=sys.stderr)
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"[dry-run] report: {report_path}")
        return
    staging = RAW / ".farm.new"
    shutil.rmtree(staging, ignore_errors=True)
    for name in FARM_ENTRIES:
        for rel_link, target in items[name]:
            link = staging / name / rel_link
            link.parent.mkdir(parents=True, exist_ok=True)
            # relative target must be computed against the FINAL location
            # (raw/<name>/...), not the staging path (one level deeper).
            link.symlink_to(_rel(target, RAW / name / rel_link))
    backups: list[Path] = []
    try:
        for name in FARM_ENTRIES:
            entry = RAW / name
            if entry.exists() or entry.is_symlink():
                backup = RAW / f"{name}.farm.old"
                if backup.is_symlink() or backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup, ignore_errors=True)
                os.rename(entry, backup)
                backups.append(backup)
            os.rename(staging / name, entry)
    except Exception:
        print("swap failed; .farm.old backups left in place for manual recovery", file=sys.stderr)
        raise
    for backup in backups:
        if backup.is_symlink() or backup.is_file():
            backup.unlink()
        else:
            shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    # authoritative check against the swapped-in farm (pre-swap check above is preflight only)
    checked, uri_errors = check_manifest_uris()
    summary["manifest_uri_check"] = {"checked": checked, "errors": uri_errors}
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"built farm: " + " ".join(f"{name}={len(items[name])}" for name in FARM_ENTRIES))
    print(f"manifest URIs: {checked - len(uri_errors)}/{checked} resolve")
    for error in uri_errors:
        print(f"MISSING {error}", file=sys.stderr)
    print(f"report: {report_path}")


def verify() -> int:
    checked, errors = check_manifest_uris()
    broken = 0
    for path in RAW.rglob("*"):
        if path.is_symlink() and not path.exists():
            broken += 1
            print(f"broken: {path} -> {os.readlink(path)}")
    for name in FARM_ENTRIES:
        entry = RAW / name
        if entry.is_symlink() or not entry.is_dir():
            print(f"missing farm entry: {entry}")
            errors.append(str(entry))
    print(f"verify: manifest {checked - len(errors)}/{checked} resolve; broken links under raw/ = {broken}")
    return 1 if errors or broken else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--dates", default=",".join(DEFAULT_DATES))
    p_build.add_argument("--cam", default="3", help="camera(s) to import under raw/rgb; 'all' or comma list, default 3")
    p_build.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "build":
        dates = tuple(item.strip() for item in args.dates.split(",") if item.strip())
        if args.cam == "all":
            cams = ("1", "2", "3", "4")
        else:
            cams = tuple(item.strip() for item in args.cam.split(",") if item.strip())
            unknown = set(cams) - {"1", "2", "3", "4"}
            if unknown:
                parser.error(f"unknown camera(s): {sorted(unknown)}")
        build(args.dry_run, dates, cams)
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
