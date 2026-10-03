#!/usr/bin/env python3
"""Build the versioned, read-only session manifest for the shared data base."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from anysole.types import (  # noqa: E402
    JOINT_NAMES as SMPL24_NAMES,
    JOINT_PARENTS as SMPL24_PARENTS,
    JOINT_PROTOCOL_CHECKSUM as SMPL24_CHECKSUM,
)
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
FIELDS = ("session_id", "subject_id", "action", "trial", "camera", "video_path",
          "pressure_path", "bvh_path", "smpl_path", "target_fps", "n_frames", "visual_start_s",
          "mocap_start_s", "offset_s", "fake_frame_indices", "valid_frame_indices",
          "quality", "eligible_anysole", "eligibility_reason", "joint_checksum",
          "pressure_quality_class", "pressure_quality_reason")

SMPL_COORDINATE_SYSTEM = "SMPL right-handed +X left, +Y up, +Z forward; world motion preserved"
LEGACY_BVH23_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder",
    "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg",
    "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)
LEGACY_BVH23_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21)


def validate_smpl(path: Path) -> list[str]:
    """Return protocol errors for an alleged native SMPL-24 archive."""
    if not path.is_file():
        return []  # reported separately as missing_smpl
    import numpy as np
    errors = []
    try:
        with np.load(path, allow_pickle=True) as data:
            if "poses" in data:
                poses = np.asarray(data["poses"])
            elif all(key in data for key in ("root_orient", "pose_body")):
                poses = np.concatenate([data["root_orient"], data["pose_body"]], axis=-1)
            else:
                return ["invalid_smpl_missing_pose"]
            trans = np.asarray(data["trans"]) if "trans" in data else np.empty((0, 3))
            if poses.ndim != 2 or poses.shape[1] != 72:
                errors.append("invalid_smpl_pose_shape")
            if trans.shape != (poses.shape[0], 3):
                errors.append("invalid_smpl_trans_shape")
            if not np.isfinite(poses).all() or not np.isfinite(trans).all():
                errors.append("invalid_smpl_nonfinite")
            def scalar(key: str) -> str:
                value = np.asarray(data[key]) if key in data else np.asarray("")
                return str(value.reshape(-1)[0]) if value.size else ""
            if scalar("surface_model_type").lower() not in ("", "smpl"):
                errors.append("invalid_smpl_model_type")
            if scalar("gender").lower() not in ("", "neutral"):
                errors.append("invalid_smpl_gender")
            if scalar("model_units").lower() not in ("", "m"):
                errors.append("invalid_smpl_units")
            if scalar("coordinate_system") not in ("", SMPL_COORDINATE_SYSTEM):
                errors.append("invalid_smpl_coordinates")
            if "mocap_frame_rate" in data:
                rate = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
                if not np.isfinite(rate) or rate <= 0:
                    errors.append("invalid_smpl_fps")
            if "root_orient" in data:
                root = np.asarray(data["root_orient"])
                if root.shape != (poses.shape[0], 3) or not np.allclose(poses[:, :3], root, atol=1e-8):
                    errors.append("invalid_smpl_root_redundancy")
            if "pose_body" in data:
                body = np.asarray(data["pose_body"])
                if body.shape not in ((poses.shape[0], 63), (poses.shape[0], 69)):
                    errors.append("invalid_smpl_body_shape")
                elif not np.allclose(poses[:, 3:3 + body.shape[1]], body, atol=1e-8):
                    errors.append("invalid_smpl_body_redundancy")
    except Exception:
        errors.append("invalid_smpl_unreadable")
    return errors


def parse_bvh_header(path: Path) -> tuple[list[str], list[int]]:
    if not path.is_file():
        return [], []
    lines = path.read_text(errors="replace").replace("\r", "").splitlines()
    names, parents, stack = [], [], []
    for line in lines:
        s = line.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            names.append(s.split()[1]); parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
        elif s == "End Site":
            stack.append(-999)
        elif s == "}":
            if stack: stack.pop()
        elif s.startswith("MOTION"):
            break
    return names, parents


def raw_index(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing raw session index: {path}; run build_raw_index.py explicitly"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pressure_files(recording_name: str, pressure_uri: str | None = None) -> tuple[Path, Path] | None:
    root = resolve_uri(pressure_uri or "raw://pressure", must_exist=True)
    if root.is_dir() and (root / "pressure_left.csv").is_file():
        left = [root / "pressure_left.csv"]
        right = [root / "pressure_right.csv"]
    else:
        left = sorted(root.glob(f"**/{recording_name}/pressure_left.csv"))
        right = sorted(root.glob(f"**/{recording_name}/pressure_right.csv"))
    if not left or not right:
        return None
    return left[-1], right[-1]


def pressure_flags(recording_name: str, n_frames: int, pressure_uri: str | None = None) -> tuple[list[int], list[int], str]:
    files = pressure_files(recording_name, pressure_uri)
    if files is None:
        return [], [], "missing_pressure_final"
    import numpy as np

    columns = []
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values:
            return [], [], "empty_pressure_final"
        fake = np.asarray([int(float(row.get("fake", 0))) for row in values], dtype=np.uint8)
        valid = np.asarray([int(float(row.get("valid_mask", 1))) for row in values], dtype=np.uint8)
        columns.append((fake, valid))
    target = np.linspace(0.0, 1.0, n_frames)
    flags = []
    valids = []
    for fake_left, valid_left in columns[:1]:
        fake_right, valid_right = columns[1]
        left_index = np.rint(target * max(len(fake_left) - 1, 0)).astype(int)
        right_index = np.rint(target * max(len(fake_right) - 1, 0)).astype(int)
        fake = np.maximum(fake_left[left_index], fake_right[right_index])
        valid = np.minimum(valid_left[left_index], valid_right[right_index])
        valid = np.minimum(valid, 1 - fake)
        flags = np.flatnonzero(fake).astype(int).tolist()
        valids = np.flatnonzero(valid).astype(int).tolist()
    return flags, valids, ""


def ood_split(subjects: list[str]) -> dict[str, str]:
    """Deterministic subject split: 20% test, preceding 10% validation."""
    unique = sorted(set(subjects), key=lambda x: (int(re.sub(r"\D", "", x) or 0), x))
    n_test = max(1, round(len(unique) * 0.2))
    n_val = max(1, round(len(unique) * 0.1)) if len(unique) > 2 else 0
    test, val = set(unique[-n_test:]), set(unique[-n_test-n_val:-n_test] if n_val else ())
    return {s: ("test" if s in test else "val" if s in val else "train") for s in unique}


def _subject_list(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def join_pressure_quality(rows: list[dict]) -> None:
    """Fill pressure_quality_class/reason from the pressure washer stats table.

    0605d1f registered the A/B/C/D tactile-quality annotation (join key =
    dataset_id == session_id; source = stats/.../missing_pressure_objects.csv,
    definitions in protocol/splits/default/README.md).  A manifest rebuilt
    without this join silently drops the columns, so the join lives here.
    """
    stats_root = WORKSPACE / "work/data_pipeline/pressure_washer/stats"
    candidates = sorted(stats_root.glob("pressure_stats_*/overall/missing_pressure_objects.csv"))
    if not candidates:
        print("join_pressure_quality: no pressure stats table under %s; columns left empty" % stats_root,
              file=sys.stderr)
        return
    stats_path = candidates[-1]
    lookup: dict[str, tuple[str, str]] = {}
    with stats_path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            lookup[record["dataset_id"]] = (record["quality_class"], record["reason"])
    joined = 0
    for row in rows:
        quality = lookup.get(row["session_id"])
        row["pressure_quality_class"] = quality[0] if quality else ""
        row["pressure_quality_reason"] = quality[1] if quality else ""
        joined += quality is not None
    print("join_pressure_quality: %d/%d sessions annotated from %s" % (joined, len(rows), stats_path),
          file=sys.stderr)


def write_splits(rows: list[dict], output: Path, *, exclude: set[str] | None = None,
                 test: set[str] | None = None, val: set[str] | None = None) -> None:
    """Generate a subject split; validation equals test unless ``val`` is given."""
    excluded = exclude or set()
    eligible = [row for row in rows if row.get("eligible_anysole") == "1" and row["subject_id"] not in excluded]
    by_subject = sorted({row["subject_id"] for row in eligible},
                        key=lambda value: (int(re.sub(r"\D", "", value) or 0), value))
    test_subjects = set(test) if test is not None else set(by_subject[-2:])
    unknown = (test_subjects | (val or set())) - set(by_subject)
    if unknown:
        print(f"warning: requested subjects not present in eligible manifest: {sorted(unknown)}")
    val_subjects = set(test_subjects if val is None else val)
    groups = {
        "train": sorted(row["session_id"] for row in eligible if row["subject_id"] not in test_subjects | val_subjects),
        "val": sorted(row["session_id"] for row in eligible if row["subject_id"] in val_subjects),
        "test": sorted(row["session_id"] for row in eligible if row["subject_id"] in test_subjects),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "train", "val", "test"])
        for index in range(max(map(len, groups.values()), default=0)):
            writer.writerow([index,
                             groups["train"][index] if index < len(groups["train"]) else "",
                             groups["val"][index] if index < len(groups["val"]) else "",
                             groups["test"][index] if index < len(groups["test"]) else ""])
    print("wrote %s: train=%d val=%d test=%d; exclude=%s test_subjects=%s val_subjects=%s" %
          (output, len(groups["train"]), len(groups["val"]), len(groups["test"]),
           sorted(excluded), sorted(test_subjects), sorted(val_subjects)))


def build(fps: float, camera: str, output: Path,
          index_path: Path | None = None, split_path: Path | None = None) -> tuple[list[dict], list[str]]:
    index_path = index_path or WORKSPACE / "protocol/manifests/raw_session_index.jsonl"
    rows, issues = [], []
    for meta in raw_index(index_path):
        sid = str(meta["session_id"])
        subject = str(meta["subject_id"])
        n = int(meta["n_frames"])
        bvh = resolve_uri(meta["bvh_uri"])
        smpl = resolve_uri(meta["smpl_uri"]) if meta.get("smpl_uri") else Path("__missing_smpl__")
        recording = resolve_uri(meta["recording_uri"], must_exist=True)
        fake, valid, pressure_issue = pressure_flags(recording.name, n, meta.get("pressure_uri"))
        names, parents = parse_bvh_header(bvh)
        quality = []
        for label, path in (("bvh", bvh), ("smpl", smpl)):
            if not path.is_file():
                quality.append(f"missing_{label}")
        if not recording.is_dir() or not any(recording.glob("3/*.jpg")):
            quality.append("missing_video_source")
        if pressure_issue:
            quality.append(pressure_issue)
        quality.extend(validate_smpl(smpl))
        if tuple(names) != LEGACY_BVH23_NAMES or tuple(parents) != LEGACY_BVH23_PARENTS:
            quality.append("invalid_bvh23_protocol")
        if fps != float(meta.get("target_fps", fps)):
            quality.append("fps_mismatch")
        required_missing = {"missing_bvh", "missing_smpl", "missing_video_source", "missing_pressure_final"}
        eligible = not required_missing.intersection(quality) and not any(
            issue.startswith("invalid_smpl_") or issue.startswith("invalid_bvh23_")
            for issue in quality
        )
        eligibility_reason = "ok" if eligible else ";".join(
            issue for issue in quality if issue.startswith("missing_") or issue.startswith("invalid_")
        )
        pressure_uri = meta.get("pressure_uri", f"raw://pressure/{meta['date']}/{subject}/{recording.name}")
        rows.append({"session_id": sid, "subject_id": subject, "action": meta.get("action", ""),
                     "trial": meta.get("trial", ""), "camera": camera,
                     "video_path": meta["recording_uri"], "pressure_path": pressure_uri,
                     "bvh_path": meta["bvh_uri"], "smpl_path": meta.get("smpl_uri", ""),
                     "target_fps": float(meta.get("target_fps", fps)), "n_frames": n,
                     "visual_start_s": meta.get("visual_start_s", 0.0),
                     "mocap_start_s": meta.get("mocap_start_s", 0.0), "offset_s": meta.get("offset_s", 0.0),
                     "fake_frame_indices": json.dumps(fake), "valid_frame_indices": json.dumps(valid),
                     "quality": "ok" if not quality else ";".join(quality),
                     "eligible_anysole": "1" if eligible else "0", "eligibility_reason": eligibility_reason,
                     "joint_checksum": SMPL24_CHECKSUM if smpl.is_file() else ""})
    join_pressure_quality(rows)
    rows.sort(key=lambda row: row["session_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    jsonl = output.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows, issues


def main() -> int:
    argv = sys.argv[1:]
    command = argv[0] if argv and argv[0] in {"build", "split"} else "build"
    if command == "split":
        p = argparse.ArgumentParser(description="Generate the canonical subject split")
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output", type=Path, default=WORKSPACE / "protocol/splits/default/splits.csv")
        p.add_argument("--exclude", default="", help="subjects to remove, e.g. S4,S5")
        p.add_argument("--test", default="", help="test subjects, e.g. S13,S14; default: last two")
        p.add_argument("--val", default=None, help="validation subjects; default: same subjects as --test")
        a = p.parse_args(argv[1:])
        rows = [json.loads(line) for line in a.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        write_splits(rows, a.output, exclude=_subject_list(a.exclude),
                     test=_subject_list(a.test) if a.test else None,
                     val=_subject_list(a.val) if a.val is not None else None)
        return 0
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, default=40)
    p.add_argument("--camera", default="cam3")
    p.add_argument("--output", type=Path, default=WORKSPACE / "protocol/manifests/session_manifest.csv")
    p.add_argument("--index", type=Path, default=WORKSPACE / "protocol/manifests/raw_session_index.jsonl")
    build_args = argv[1:] if argv and argv[0] == "build" else argv
    a = p.parse_args(build_args)
    rows, _ = build(a.fps, a.camera, a.output, a.index)
    print(f"wrote {a.output} and {a.output.with_suffix('.jsonl')} ({len(rows)} sessions); split file unchanged")
    return 0 if rows else 1


if __name__ == "__main__": raise SystemExit(main())
