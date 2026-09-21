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

WORKSPACE = ROOT / "AnysoleWorkspace"
FIELDS = ("session_id", "subject_id", "action", "trial", "camera", "video_path",
          "pressure_path", "bvh_path", "smpl_path", "target_fps", "n_frames", "visual_start_s",
          "mocap_start_s", "offset_s", "fake_frame_indices", "valid_frame_indices",
          "quality", "eligible_anysole", "eligibility_reason", "split_iid", "split_ood", "joint_checksum")

SMPL_COORDINATE_SYSTEM = "SMPL right-handed +X left, +Y up, +Z forward; world motion preserved"
LEGACY_BVH23_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder",
    "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg",
    "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)
LEGACY_BVH23_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def bvh_for(session_id: str, meta: dict) -> Path:
    recorded = Path(str(meta.get("bvh_path", "")))
    if recorded.is_file():
        return recorded
    date = str(meta.get("date", ""))
    candidates = sorted((WORKSPACE / "sources/raw" / date / "mocap_ori_bvh" / session_id).glob("*.bvh"))
    if candidates:
        return candidates[0]
    raw = Path(str(meta.get("rec_dir", ""))).parent / "mocap_ori_bvh" / session_id
    candidates = sorted(raw.glob("*.bvh")) if raw.is_dir() else []
    return candidates[0] if candidates else recorded


SMPL_ROOTS = tuple(Path(p) for p in (
    "/data/lizhe/projects/Tactile/Mocap/0804",
    "/data/lizhe/projects/Tactile/Mocap/0807",
    "/data/lizhe/projects/Tactile/Mocap/0808",
    "/data/lizhe/projects/Tactile/Mocap/0810",
))


def smpl_for(session_id: str, meta: dict) -> Path:
    recorded = Path(str(meta.get("smpl_path", "")))
    if recorded.is_file():
        return recorded
    for root in SMPL_ROOTS:
        matches = sorted(root.glob(f"**/{session_id}/motion_neutral_smpl.npz"))
        if matches:
            return matches[0]
    return recorded


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


def split_map(split_path: Path) -> dict[str, str]:
    result: dict[str, list[str]] = {}
    if not split_path.is_file():
        return result
    with split_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for split in ("train", "val", "test"):
                sid = (row.get(split) or "").strip()
                if sid:
                    result.setdefault(sid, []).append(split)
    return {sid: ",".join(dict.fromkeys(values)) for sid, values in result.items()}


def ood_split(subjects: list[str]) -> dict[str, str]:
    """Deterministic subject split: 20% test, preceding 10% validation."""
    unique = sorted(set(subjects), key=lambda x: (int(re.sub(r"\D", "", x) or 0), x))
    n_test = max(1, round(len(unique) * 0.2))
    n_val = max(1, round(len(unique) * 0.1)) if len(unique) > 2 else 0
    test, val = set(unique[-n_test:]), set(unique[-n_test-n_val:-n_test] if n_val else ())
    return {s: ("test" if s in test else "val" if s in val else "train") for s in unique}


def write_splits(rows: list[dict], output: Path) -> None:
    """Write the AnySole split consumed by ``AnySoleDataset``.

    A session is eligible only when tactile CSVs, both motion formats and
    video are present.  Splits are subject-disjoint to avoid leaking a
    person's gait between train/validation/test.
    """
    eligible = [row for row in rows if row.get("eligible_anysole") == "1"]
    by_subject = sorted({row["subject_id"] for row in eligible},
                        key=lambda value: (int(re.sub(r"\D", "", value) or 0), value))
    n_test = max(1, round(len(by_subject) * 0.2)) if by_subject else 0
    n_val = max(1, round(len(by_subject) * 0.1)) if len(by_subject) > 2 else 0
    test_subjects = set(by_subject[-n_test:])
    val_subjects = set(by_subject[-n_test - n_val:-n_test]) if n_val else set()
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
    print("wrote %s: train=%d val=%d test=%d (subjects=%s)" %
          (output, len(groups["train"]), len(groups["val"]), len(groups["test"]), by_subject))


def build(fps: float, camera: str, output: Path) -> tuple[list[dict], list[str]]:
    seq_root = WORKSPACE / "derived/MotionPRO/sequences" / camera
    metas = sorted(seq_root.glob("*/*/*/align_meta.json"))
    rows, issues = [], []
    subjects = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sid = str(meta.get("session_id") or meta_path.parent.name)
        subject = str(meta.get("subject") or "")
        subjects.append(subject)
        pressure = meta_path.parent / "pressure.npz"
        fake_path = meta_path.parent / "fake_mask.npy"
        bvh = bvh_for(sid, meta)
        smpl = smpl_for(sid, meta)
        names, parents = parse_bvh_header(bvh)
        fake = []
        if fake_path.is_file():
            import numpy as np
            fake = np.flatnonzero(np.asarray(np.load(fake_path)).reshape(-1) != 0).astype(int).tolist()
        n = int(meta.get("n_frames", 0))
        valid = [i for i in range(n) if i not in set(fake)]
        rec_dir = Path(str(meta.get("rec_dir", "")))
        tactile_left = rec_dir / "pressure_left.csv"
        tactile_right = rec_dir / "pressure_right.csv"
        video_dir = Path(str(meta.get("cam_dir", "")))
        quality = []
        # AnySole consumes the original left/right pressure CSVs below.  The
        # derived pressure.npz is retained as metadata but is not an
        # eligibility requirement (in particular, C3D/derived-only records do
        # not satisfy the tactile requirement).
        for label, path in (("bvh", bvh), ("smpl", smpl), ("video", video_dir)):
            if not Path(path).is_file() and label != "video": quality.append(f"missing_{label}")
            if label == "video" and (not Path(path).is_dir() or not any(Path(path).iterdir())): quality.append("missing_video_source")
        if not tactile_left.is_file() or not tactile_right.is_file(): quality.append("missing_tactile_csv")
        quality.extend(validate_smpl(smpl))
        if tuple(names) != LEGACY_BVH23_NAMES or tuple(parents) != LEGACY_BVH23_PARENTS:
            quality.append("invalid_bvh23_protocol")
        if fps != float(meta.get("target_fps", fps)): quality.append("fps_mismatch")
        required_missing = {"missing_tactile_csv", "missing_bvh", "missing_smpl", "missing_video_source"}
        eligible = not required_missing.intersection(quality) and not any(
            issue.startswith("invalid_smpl_") or issue.startswith("invalid_bvh23_")
            for issue in quality
        )
        eligibility_reason = "ok" if eligible else ";".join(
            issue for issue in quality
            if issue.startswith("missing_") or issue.startswith("invalid_smpl_")
            or issue.startswith("invalid_bvh23_")
        )
        rows.append({"session_id": sid, "subject_id": subject, "action": sid[len(subject):-1] if sid.startswith(subject) else "",
                     "trial": sid[-1:] if sid.startswith(subject) else "", "camera": camera,
                     "video_path": rel(meta.get("cam_dir", "")), "pressure_path": rel(pressure),
                     "bvh_path": rel(bvh), "smpl_path": rel(smpl), "target_fps": float(meta.get("target_fps", fps)),
                     "n_frames": n, "visual_start_s": meta.get("visual_start_s", ""),
                     "mocap_start_s": meta.get("mocap_start_s", ""), "offset_s": meta.get("offset_s", ""),
                     "fake_frame_indices": json.dumps(fake), "valid_frame_indices": json.dumps(valid),
                     "quality": "ok" if not quality else ";".join(quality),
                     "eligible_anysole": "1" if eligible else "0", "eligibility_reason": eligibility_reason,
                     "split_iid": "", "split_ood": "",
                     # AnySole's joint contract is native SMPL-24.  The BVH
                     # header is still validated above because Step2Motion is
                     # a required companion modality, but it must not define
                     # AnySole's protocol checksum.
                     "joint_checksum": SMPL24_CHECKSUM if smpl.is_file() else ""})
    ood = ood_split([row["subject_id"] for row in rows if row["eligible_anysole"] == "1"])
    for row in rows:
        row["split_ood"] = ood.get(row["subject_id"], "unassigned") if row["eligible_anysole"] == "1" else "unassigned"
        row["split_iid"] = row["split_ood"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    jsonl = output.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    split_output = WORKSPACE / "splits/default/splits.csv"
    write_splits(rows, split_output)
    return rows, issues


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--fps", type=float, default=40); p.add_argument("--camera", default="cam3")
    p.add_argument("--output", type=Path, default=WORKSPACE / "manifests/session_manifest.csv")
    a = p.parse_args(); rows, _ = build(a.fps, a.camera, a.output)
    print(f"wrote {a.output} and {a.output.with_suffix('.jsonl')} ({len(rows)} sessions)"); return 0 if rows else 1


if __name__ == "__main__": raise SystemExit(main())
