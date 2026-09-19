#!/usr/bin/env python3
"""Build the versioned, read-only session manifest for the shared data base."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
FIELDS = ("session_id", "subject_id", "action", "trial", "camera", "video_path",
          "pressure_path", "bvh_path", "smpl_path", "target_fps", "n_frames", "visual_start_s",
          "mocap_start_s", "offset_s", "fake_frame_indices", "valid_frame_indices",
          "quality", "split_iid", "split_ood", "joint_checksum")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def checksum(names: list[str], parents: list[int]) -> str:
    payload = json.dumps({"names": names, "parents": parents}, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def build(fps: float, camera: str, output: Path) -> tuple[list[dict], list[str]]:
    seq_root = WORKSPACE / "derived/MotionPRO/sequences" / camera
    split_iid = split_map(WORKSPACE / "splits/default/splits.csv")
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
        quality = []
        for label, path in (("pressure", pressure), ("bvh", bvh), ("smpl", smpl), ("video", meta.get("cam_dir", ""))):
            if not Path(path).is_file() and label != "video": quality.append(f"missing_{label}")
            if label == "video" and not Path(path).exists(): quality.append("missing_video_source")
        if len(names) != 23: quality.append(f"joint_count_{len(names)}")
        if fps != float(meta.get("target_fps", fps)): quality.append("fps_mismatch")
        rows.append({"session_id": sid, "subject_id": subject, "action": sid[len(subject):-1] if sid.startswith(subject) else "",
                     "trial": sid[-1:] if sid.startswith(subject) else "", "camera": camera,
                     "video_path": rel(meta.get("cam_dir", "")), "pressure_path": rel(pressure),
                     "bvh_path": rel(bvh), "smpl_path": rel(smpl), "target_fps": float(meta.get("target_fps", fps)),
                     "n_frames": n, "visual_start_s": meta.get("visual_start_s", ""),
                     "mocap_start_s": meta.get("mocap_start_s", ""), "offset_s": meta.get("offset_s", ""),
                     "fake_frame_indices": json.dumps(fake), "valid_frame_indices": json.dumps(valid),
                     "quality": "ok" if not quality else ";".join(quality),
                     "split_iid": split_iid.get(sid, "unassigned"), "split_ood": "", 
                     "joint_checksum": checksum(names, parents) if names else ""})
    ood = ood_split(subjects)
    for row in rows: row["split_ood"] = ood.get(row["subject_id"], "unassigned")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    jsonl = output.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows, issues


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--fps", type=float, default=40); p.add_argument("--camera", default="cam3")
    p.add_argument("--output", type=Path, default=WORKSPACE / "manifests/session_manifest.csv")
    a = p.parse_args(); rows, _ = build(a.fps, a.camera, a.output)
    print(f"wrote {a.output} and {a.output.with_suffix('.jsonl')} ({len(rows)} sessions)"); return 0 if rows else 1


if __name__ == "__main__": raise SystemExit(main())
