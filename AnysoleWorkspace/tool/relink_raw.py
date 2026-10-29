#!/usr/bin/env python3
"""Build the hf-raw modality link farm.

``AnysoleWorkspace/hf-raw/{rgb,bvh,smpl}`` become per-modality filtered views over the
upstream tree, shape-preserving with respect to the frozen manifest URIs
(relative paths inside each entry match ``raw://`` URIs segment by segment).
pressure 不再由本脚本建农场：触觉正典 = AnysoleWorkspace/hf-raw/pressure 数据树
（tactile_v3 清洗产物，见 build_shared.py）。标定正典 = protocol/calibration/，
human_masks 入口 = assets/upstream/，均不在本脚本范围。

Upstream sources (2026-10-24 job onwards):

* bvh: every ``bvh_path`` URI of the session manifest is linked (same link
  name, e.g. ``<sid>/<sid>_Skeleton1.bvh``) onto the per-subject BVH24 export
  ``Mocap/<mmdd>/<mmdd>smpl_new/mocap_ori_c3d/<sid>/motion_smpl24_blender_world_m.bvh``;
* smpl: ``Mocap/<mmdd>/<mmdd>smpl_new/mocap_ori_c3d/<sid>/`` (npz + the four
  validation/conversion jsons), replacing the previous ``<mmdd>smpl`` job.

Glob-safety: every level that is hit by a recursive ``**`` glob (AnysoleWorkspace/hf-raw/smpl
by ``build_raw_index.py``) is a real directory here; symlinks are only leaves
(files) or camera directories under AnysoleWorkspace/hf-raw/rgb (never ``**``-traversed).

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
RAW = WORKSPACE / "hf-raw"
UP = Path("/data/lizhe/projects/Tactile")
UP_DATA = UP / "1_Data"
UP_MOCAP = UP / "Mocap"

DEFAULT_DATES = ("20260804", "20260807", "20260808", "20260810")
FARM_ENTRIES = ("rgb", "bvh", "smpl")


def _rel(target: Path, link: Path) -> str:
    return os.path.relpath(target, link.parent)


def collect(dates: tuple[str, ...], cams: tuple[str, ...]) -> dict[str, list[tuple[Path, Path]]]:
    """Return {entry: [(link path relative to RAW/entry, absolute target), ...]}."""
    items: dict[str, list[tuple[Path, Path]]] = {name: [] for name in FARM_ENTRIES}

    def add(entry: str, rel_link: Path, target: Path) -> None:
        items[entry].append((rel_link, target))

    # bvh: manifest-driven, independent of the date filter.  Every
    # ``bvh_path`` URI keeps its exact link name and points at the per-subject
    # BVH24 export of the 2026-10-24 job.
    manifest = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uri = json.loads(line).get("bvh_path") or ""
        if not uri:
            continue
        # raw://bvh/<yyyymmdd>/mocap_ori_bvh/<sid>/<sid>_Skeleton<N>.bvh
        parts = uri.split("/")
        if len(parts) != 7 or parts[2] != "bvh":
            print(f"skip malformed bvh URI: {uri}", file=sys.stderr)
            continue
        date, sid, fname = parts[3], parts[5], parts[6]
        mmdd = date[4:]
        target = UP_MOCAP / mmdd / f"{mmdd}smpl_new" / "mocap_ori_c3d" / sid / "motion_smpl24_blender_world_m.bvh"
        if target.is_file():
            add("bvh", Path(date) / "mocap_ori_bvh" / sid / fname, target)
        else:
            print(f"skip missing BVH24 upstream: {target}", file=sys.stderr)

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
        mmdd = date[4:]
        smpl_root = UP_MOCAP / mmdd / f"{mmdd}smpl_new" / "mocap_ori_c3d"
        if smpl_root.is_dir():
            for sid_dir in sorted(smpl_root.iterdir()):
                if not sid_dir.is_dir():
                    continue
                for name in (
                    "motion_neutral_smpl.npz",
                    "bvh_validation.json",
                    "conversion_manifest.json",
                    "motion_quality.json",
                    "smpl_validation.json",
                ):
                    source = sid_dir / name
                    if source.is_file():
                        rel_link = Path(mmdd) / f"{mmdd}smpl" / "mocap_ori_c3d" / sid_dir.name / name
                        add("smpl", rel_link, source)
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
