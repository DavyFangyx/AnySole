#!/usr/bin/env python3
"""Initialize, migrate, and validate the centralized baseline workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
RESULTS = REPO_ROOT / "results"
DISPLAY = REPO_ROOT / "resultsdisplay"

SOURCE_LINKS = {
    WORKSPACE / "sources/raw": Path("/data/lizhe/projects/Tactile/1_Data"),
    WORKSPACE / "sources/published": Path("/data/lizhe/projects/Tactile/4_Dataset"),
    WORKSPACE / "sources/calibration_artifacts": Path("/data/lizhe/projects/Tactile/0_Calibration"),
    WORKSPACE / "sources/pressure_washer": Path("../tools/PressureWasher"),
}

# PressureWasher runtime data is kept with the other source-side datasets.
PRESSURE_SOURCE_DIR = WORKSPACE / "sources/PressureWasher"

# Order matters where a child is separated from its former parent.
MIGRATIONS = (
    (REPO_ROOT / "Baselines/Step2Motion/models/gait_model/predictions", RESULTS / "Step2Motion/predictions/gait_model"),
    (REPO_ROOT / "Baselines/Step2Motion/models/gait_model", RESULTS / "Step2Motion/checkpoints/gait_model"),
    (REPO_ROOT / "Baselines/Step2Motion/models/UnderPressure", WORKSPACE / "dependencies/Step2Motion/models/UnderPressure"),
    (REPO_ROOT / "Baselines/Step2Motion/models/dancing", WORKSPACE / "dependencies/Step2Motion/models/dancing"),
    (REPO_ROOT / "Baselines/Step2Motion/models/step2motion", WORKSPACE / "dependencies/Step2Motion/models/step2motion"),
    (REPO_ROOT / "Baselines/Step2Motion/data/gait", WORKSPACE / "derived/Step2Motion/gait"),
    (REPO_ROOT / "Baselines/Step2Motion/data/UnderPressure", WORKSPACE / "dependencies/Step2Motion/data/UnderPressure"),
    (REPO_ROOT / "Baselines/Step2Motion/data/dancing", WORKSPACE / "dependencies/Step2Motion/data/dancing"),
    (REPO_ROOT / "Baselines/Step2Motion/data/exp/viz_compare", DISPLAY / "Step2Motion/gait_model"),
    (REPO_ROOT / "Baselines/Step2Motion/configs/normalizer_dancing.pth", WORKSPACE / "dependencies/Step2Motion/normalizers/normalizer_dancing.pth"),
    (REPO_ROOT / "Baselines/Step2Motion/configs/normalizer_default.pth", WORKSPACE / "dependencies/Step2Motion/normalizers/normalizer_default.pth"),
    (REPO_ROOT / "Baselines/Step2Motion/configs/normalizer_gait.pth", WORKSPACE / "dependencies/Step2Motion/normalizers/normalizer_gait.pth"),
    (REPO_ROOT / "Baselines/Step2Motion/configs/normalizer_step2motion.pth", WORKSPACE / "dependencies/Step2Motion/normalizers/normalizer_step2motion.pth"),
    (REPO_ROOT / "Baselines/MotionPRO/data/sequences", WORKSPACE / "derived/MotionPRO/sequences"),
    (REPO_ROOT / "Baselines/MotionPRO/data/splits", WORKSPACE / "splits/default"),
    (REPO_ROOT / "Baselines/MotionPRO/data/smpl", WORKSPACE / "dependencies/smpl"),
    (REPO_ROOT / "Baselines/MotionPRO/data/cliff_ckpt", WORKSPACE / "dependencies/MotionPRO/cliff_ckpt"),
    (REPO_ROOT / "Baselines/MotionPRO/data/mmdetection", WORKSPACE / "dependencies/MotionPRO/mmdetection"),
    (REPO_ROOT / "Baselines/MotionPRO/data/exp/checkpoint", RESULTS / "MotionPRO/checkpoints"),
    (REPO_ROOT / "Baselines/MotionPRO/data/exp/result", RESULTS / "MotionPRO/metrics"),
    (REPO_ROOT / "Baselines/MotionPRO/data/exp/viz_compare", DISPLAY / "MotionPRO"),
    (REPO_ROOT / "Baselines/MotionPRO/data/tensorboard", RESULTS / "MotionPRO/tensorboard"),
    (REPO_ROOT / "Baselines/MotionPRO/log", RESULTS / "MotionPRO/logs"),
)

LOCAL_DIRS = (
    PRESSURE_SOURCE_DIR,
    WORKSPACE / "derived/pressure_tookit",
    WORKSPACE / "dependencies/pressure_tookit",
    RESULTS / "Step2Motion/checkpoints",
    RESULTS / "Step2Motion/predictions",
    RESULTS / "Step2Motion/metrics",
    RESULTS / "Step2Motion/logs",
    RESULTS / "pressure_tookit",
    DISPLAY / "Step2Motion",
    DISPLAY / "pressure_tookit",
)


def ensure_link(link: Path, target: Path, dry_run: bool) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) != target:
            raise RuntimeError(f"Unexpected link target: {link} -> {os.readlink(link)}")
        return
    if link.exists():
        raise RuntimeError(f"Cannot create source link over existing path: {link}")
    print(f"link {link.relative_to(REPO_ROOT)} -> {target}")
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)


def initialize(dry_run: bool = False) -> None:
    for directory in LOCAL_DIRS:
        print(f"mkdir {directory.relative_to(REPO_ROOT)}")
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)
    for link, target in SOURCE_LINKS.items():
        ensure_link(link, target, dry_run)

    published = Path("/data/lizhe/projects/Tactile/4_Dataset")
    for source in sorted(published.glob("[0-9]" * 8 + "/" + "[0-9]" * 8 + ".json")):
        link = WORKSPACE / "calibration" / source.name
        target = Path("../sources/published") / source.parent.name / source.name
        ensure_link(link, target, dry_run)


def migrate(dry_run: bool = False) -> None:
    initialize(dry_run=dry_run)
    for source, target in MIGRATIONS:
        if not source.exists() and not source.is_symlink():
            print(f"skip missing {source.relative_to(REPO_ROOT)}")
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(
                f"Refusing to overwrite migration target {target.relative_to(REPO_ROOT)}"
            )
        print(f"move {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))


def relink_color_frames(dry_run: bool = False) -> int:
    sequence_root = WORKSPACE / "derived/MotionPRO/sequences"
    physical_raw = Path("/data/lizhe/projects/Tactile/1_Data")
    logical_raw = WORKSPACE / "sources/raw"
    changed = 0
    for link in sequence_root.rglob("color/*"):
        if not link.is_symlink():
            continue
        raw_target = Path(os.readlink(link))
        absolute_target = raw_target if raw_target.is_absolute() else link.parent / raw_target
        try:
            relative_source = absolute_target.resolve().relative_to(physical_raw)
        except ValueError:
            continue
        desired = os.path.relpath(logical_raw / relative_source, link.parent)
        if os.readlink(link) == desired:
            continue
        changed += 1
        if not dry_run:
            link.unlink()
            link.symlink_to(desired)
    print(f"relinked {changed} RGB frames through AnysoleWorkspace/sources/raw")
    return changed


def validate_calibration(path: Path) -> list[str]:
    errors = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return [f"{path}: cannot load JSON: {exc}"]
    cameras = payload.get("cameras", {})
    for cam_id in range(1, 5):
        camera = cameras.get(f"cam{cam_id}", {})
        for key in ("K", "D", "R", "t"):
            if key not in camera:
                errors.append(f"{path}: missing cameras.cam{cam_id}.{key}")
        if len(camera.get("K", [])) != 3 or any(len(row) != 3 for row in camera.get("K", [])):
            errors.append(f"{path}: cameras.cam{cam_id}.K must be 3x3")
        if len(camera.get("R", [])) != 3 or any(len(row) != 3 for row in camera.get("R", [])):
            errors.append(f"{path}: cameras.cam{cam_id}.R must be 3x3")
        if len(camera.get("t", [])) != 3:
            errors.append(f"{path}: cameras.cam{cam_id}.t must have 3 values")
    transform = payload.get("mocap_raw_to_checkerboard_world", {})
    for key in ("input_scale", "R", "t"):
        if key not in transform:
            errors.append(f"{path}: missing mocap_raw_to_checkerboard_world.{key}")
    return errors


def doctor() -> int:
    errors = []
    symlink_count = 0
    for link, _ in SOURCE_LINKS.items():
        if not link.is_symlink():
            errors.append(f"missing source link: {link}")
        elif not link.exists():
            errors.append(f"broken source link: {link} -> {os.readlink(link)}")
    for root in (WORKSPACE, RESULTS, DISPLAY):
        for path in root.rglob("*"):
            if not path.is_symlink():
                continue
            symlink_count += 1
            if not path.exists():
                errors.append(f"broken link: {path} -> {os.readlink(path)}")
    calibrations = sorted((WORKSPACE / "calibration").glob("*.json"))
    if not calibrations:
        errors.append("no calibration summaries found")
    for path in calibrations:
        errors.extend(validate_calibration(path))
    split = WORKSPACE / "splits/default/splits.csv"
    if not split.is_file():
        errors.append(f"missing canonical split: {split}")
    for path in (
        WORKSPACE / "dependencies/smpl/SMPL_NEUTRAL.pkl",
        WORKSPACE / "dependencies/MotionPRO/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
        WORKSPACE / "dependencies/MotionPRO/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
        WORKSPACE / "dependencies/pressure_tookit/depthpro/config.json",
        WORKSPACE / "dependencies/pressure_tookit/depthpro/model.safetensors",
        WORKSPACE / "dependencies/pressure_tookit/depthpro/preprocessor_config.json",
        WORKSPACE / "dependencies/Step2Motion/normalizers/normalizer_gait.pth",
        WORKSPACE / "derived/MotionPRO/sequences/cam3",
        WORKSPACE / "derived/Step2Motion/gait/gait_test.pt",
        RESULTS / "MotionPRO/checkpoints",
        RESULTS / "Step2Motion/checkpoints/gait_model",
        DISPLAY / "MotionPRO",
        DISPLAY / "Step2Motion/gait_model",
    ):
        if not path.exists():
            errors.append(f"missing required path: {path}")
    for source, _ in MIGRATIONS:
        if source.exists() or source.is_symlink():
            errors.append(f"legacy path still exists after migration: {source}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"workspace ok: {len(calibrations)} calibration summaries, "
        f"{symlink_count} valid links"
    )
    print(f"workspace={WORKSPACE}")
    print(f"results={RESULTS}")
    print(f"resultsdisplay={DISPLAY}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "migrate", "relink", "doctor"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "init":
        initialize(dry_run=args.dry_run)
    elif args.command == "relink":
        relink_color_frames(dry_run=args.dry_run)
    else:
        migrate(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
