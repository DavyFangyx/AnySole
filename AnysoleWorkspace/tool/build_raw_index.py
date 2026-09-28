#!/usr/bin/env python3
"""Create the source-side session index used by the new manifest builder.

This command is intentionally an explicit migration step.  It can bootstrap
the selected session set from the historical manifest once, but normal
manifest builds read only this index plus ``raw/`` and never scan a model
directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from workspace import resolve_uri

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"


def sid_from_recording(name: str) -> str:
    match = re.search(r"_(S\d+)_([0-9]+)$", name)
    if not match:
        raise ValueError(f"cannot infer session id from recording name: {name}")
    return match.group(1) + match.group(2)


def uri_for(path: Path, root: Path, scheme: str) -> str:
    return f"{scheme}/{path.relative_to(root).as_posix()}"


def find_recording(session_id: str) -> Path:
    matches = []
    for meta in resolve_uri("raw://rgb").glob("*/S*/rec*/meta.json"):
        if sid_from_recording(meta.parent.name) == session_id:
            matches.append(meta.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one raw recording for {session_id}, found {len(matches)}")
    return matches[0]


def build(input_manifest: Path, output: Path) -> int:
    rows = [json.loads(line) for line in input_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = []
    for row in rows:
        sid = row["session_id"]
        recording = find_recording(sid)
        date = recording.parent.name if False else recording.parts[-3]
        subject = recording.parts[-2]
        bvh_matches = sorted((resolve_uri("raw://bvh") / date / "mocap_ori_bvh" / sid).glob("*.bvh"))
        if not bvh_matches:
            raise RuntimeError(f"missing raw BVH for {sid}")
        smpl_matches = sorted(resolve_uri("raw://smpl").glob(f"**/{sid}/motion_neutral_smpl.npz"))
        final_root = WORKSPACE / "work/data_pipeline/pressure_washer/final_fake_marked"
        pressure_dir = final_root / date / subject / recording.name
        item = {
            "schema_version": "raw.session-index.v1",
            "session_id": sid,
            "subject_id": row.get("subject_id") or subject,
            "action": row.get("action", ""),
            "trial": row.get("trial", ""),
            "date": date,
            "camera": row.get("camera", "cam3"),
            "recording_uri": uri_for(recording, resolve_uri("raw://rgb"), "raw://rgb"),
            "bvh_uri": uri_for(bvh_matches[0], resolve_uri("raw://bvh"), "raw://bvh"),
            "smpl_uri": uri_for(smpl_matches[0], resolve_uri("raw://smpl"), "raw://smpl") if smpl_matches else "",
            "pressure_uri": uri_for(pressure_dir, final_root, "work://data_pipeline/pressure_washer/final_fake_marked"),
            "n_frames": int(row["n_frames"]),
            "target_fps": float(row.get("target_fps", 40.0)),
            "visual_start_s": float(row.get("visual_start_s") or 0.0),
            "mocap_start_s": float(row.get("mocap_start_s") or 0.0),
            "offset_s": float(row.get("offset_s") or 0.0),
            "source_manifest": str(input_manifest),
        }
        result.append(item)
    result.sort(key=lambda item: item["session_id"])
    output.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in result), encoding="utf-8")
    print(f"wrote {output} ({len(result)} source sessions)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-manifest", type=Path, required=True,
                        help="one-time migration input; not used by build_manifest")
    parser.add_argument("--output", type=Path,
                        default=WORKSPACE / "protocol/manifests/raw_session_index.jsonl")
    args = parser.parse_args()
    return build(args.from_manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
