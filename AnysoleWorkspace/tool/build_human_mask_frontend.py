#!/usr/bin/env python3
"""Index external SAM3.1 human masks and derive canonical single-person boxes.

Mask PNGs stay at the external read-only source.  This tool writes only the
versioned frame index and bbox arrays consumed by frontends such as CLIFF.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORKSPACE = ROOT / "AnysoleWorkspace"
MASK_ROOT = resolve_uri("raw://human_masks")  # external source registered in workspace.py RAW_SYMLINKS
FRONTEND_ROOT = WORKSPACE / "shared/frontends/human_masks/sam31/v1"
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
FACTS_ROOT = WORKSPACE / "shared/facts/sessions"
TOLERANCE_S = 0.020


def read_manifest() -> dict[str, dict]:
    return {
        row["session_id"]: row
        for row in (json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines())
        if row.get("eligible_anysole") == "1"
    }


def read_mask_rows(session: str, cam: int) -> tuple[list[dict], dict[str, str]]:
    root = MASK_ROOT / session
    index_path = root / "masks_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing mask index: {index_path}")
    # 上游 csv 本身每帧四行（camera=1,2,3,4 齐全），只按目标相机过滤
    # （四相机决策记录 §4-4：外部数据已齐，索引只是未消费）。
    wanted_cams = {str(cam), f"cam{cam}"}
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("camera", "")) in wanted_cams]
    rows.sort(key=lambda row: float(row["visual_time_s"]))
    sources = {"masks_index": sha256_file(index_path)}
    for name in ("segmentation_manifest.json", "failures.csv"):
        path = root / name
        if path.is_file():
            sources[name] = sha256_file(path)
    return rows, sources


def session_output_names(cam: int) -> tuple[str, str]:
    """(bbox, source) filenames for a camera; cam3 keeps the frozen public
    names ``bbox.npz``/``source.json`` (zero change), other cameras get
    per-camera siblings in the same session directory."""
    if cam == 3:
        return "bbox.npz", "source.json"
    return f"bbox_cam{cam}.npz", f"source_cam{cam}.json"


def build_session(session: str, row: dict, cam: int, *, force: bool) -> tuple[int, int, dict[str, str]]:
    facts = next(FACTS_ROOT.glob(f"cam{cam}/*/*/{session}"), None)
    if facts is None:
        raise FileNotFoundError(f"missing shared facts (cam{cam}) for {session}")
    frames = np.load(facts / "frames.npz", allow_pickle=True)
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    visual = np.asarray(frames["visual_time_s"], dtype=np.float64)
    shared_valid = np.asarray(frames["valid"], dtype=np.uint8)
    date, subject = facts.parts[-3], facts.parts[-2]
    output = FRONTEND_ROOT / date / subject / session
    bbox_name, source_name = session_output_names(cam)
    if output.exists() and not force and (output / bbox_name).is_file():
        with np.load(output / bbox_name) as existing:
            return len(existing["frame_id"]), int(existing["valid"].sum()), {}
    mask_rows, source_hashes = read_mask_rows(session, cam)
    if not mask_rows:
        raise ValueError(f"no camera-{cam} mask rows for {session}")
    mask_times = np.asarray([float(item["visual_time_s"]) for item in mask_rows], dtype=np.float64)
    order = np.argsort(mask_times)
    mask_rows_sorted = [mask_rows[i] for i in order]
    mask_times_sorted = mask_times[order]
    bbox_norm = np.zeros((len(frame_ids), 4), dtype=np.float32)
    mask_area = np.zeros(len(frame_ids), dtype=np.int64)
    valid = np.zeros(len(frame_ids), dtype=np.uint8)
    index_rows: list[dict] = []
    for i, time_s in enumerate(visual):
        # Time-based nearest-neighbour join (00 总控 §2.2: session + camera +
        # visual_time_s, tolerance 20ms).  The raw csv ``visual_frame`` index
        # is offset by the session's raw start index and must not be used for
        # pairing; the real residual is recorded as time_error_s so the
        # pressure_toolkit ±20ms contract check can pass on real numbers.
        pos = int(np.searchsorted(mask_times_sorted, time_s))
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(mask_times_sorted)]
        if not candidates:
            continue  # no matchable row -> frame stays invalid, no index row
        best = min(candidates, key=lambda p: abs(mask_times_sorted[p] - time_s))
        error = float(abs(mask_times_sorted[best] - time_s))
        if error > TOLERANCE_S:
            continue  # contract: >20ms -> invalid frame, no fallback row
        source = mask_rows_sorted[best]
        bbox = np.asarray([float(source.get(key, 0.0)) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")], dtype=np.float32)
        mask_area[i] = int(float(source.get("mask_area", 0) or 0))
        bbox_norm[i] = bbox
        # Direct-use mode: the CSV row is authoritative, including zero bbox
        # rows.  Shared validity remains the only frame-level gate here.
        valid[i] = shared_valid[i]
        index_rows.append({
            "session": session, "camera": f"cam{cam}", "frame_id": int(frame_ids[i]),
            "visual_time_s": float(visual[i]), "source_visual_time_s": float(source.get("visual_time_s", 0.0)),
            "time_error_s": error, "image_path": source.get("image_path", ""),
            "mask_path": source.get("mask_path", ""), "mask_area": int(mask_area[i]),
            "bbox_xyxy_norm": bbox_norm[i].tolist(),
            "shared_valid": int(shared_valid[i]), "valid": int(valid[i]),
            "source_status": source.get("status", ""),
        })
    output.mkdir(parents=True, exist_ok=True)
    for name in ("masks_index.csv", "segmentation_manifest.json", "failures.csv"):
        source_path = MASK_ROOT / session / name
        target_path = output / name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        if source_path.is_file():
            target_path.symlink_to(os.path.relpath(source_path, target_path.parent))
    np.savez_compressed(output / bbox_name, frame_id=frame_ids, visual_time_s=visual,
                        bbox_xyxy_norm=bbox_norm, mask_area=mask_area, valid=valid,
                        source_mode=np.asarray("direct_csv_bbox"))
    (output / source_name).write_text(json.dumps({"mask_root": str(MASK_ROOT), "source_hashes": source_hashes}, indent=2) + "\n", encoding="utf-8")
    write_artifact(output, schema_version="shared.frontend.human_mask_bbox.v1", producer="build_human_mask_frontend.py", repository_root=ROOT,
                   parameters={"mask_pipeline": "sam31", "camera": f"cam{cam}", "match": "visual_time_s nearest <=20ms", "bbox_source": "masks_index.csv"},
                   source_artifacts=[f"external://rgb_human_masks/{session}/masks_index.csv"], source_hashes=source_hashes,
                   frame_id_min=0, frame_id_max=len(frame_ids) - 1, frame_count=len(frame_ids),
                   consumers=["CLIFF", "pressure_toolkit"])
    return len(frame_ids), int(valid.sum()), {"index_rows": index_rows, **source_hashes}


def _cam_list(value: str) -> tuple[int, ...]:
    if value.strip() == "all":
        return (1, 2, 3, 4)
    try:
        cams = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError:
        raise SystemExit(f"--cam expects a comma list or 'all', got {value!r}")
    unknown = set(cams) - {1, 2, 3, 4}
    if unknown:
        raise SystemExit(f"unknown camera(s): {sorted(unknown)}")
    if not cams:
        raise SystemExit("--cam list is empty")
    return cams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="", help="comma-separated session IDs; default canonical eligible sessions")
    parser.add_argument("--cam", default="3",
                        help="camera(s) to build mask frontend rows for: 'all' or comma list (1,2,3,4), default 3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cams = _cam_list(args.cam)
    manifest = read_manifest()
    wanted = [item.strip() for item in args.sessions.split(",") if item.strip()] or sorted(manifest)
    FRONTEND_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = FRONTEND_ROOT / "index.jsonl"
    # 2026-10-07 fix: a --sessions subset run used to clobber the whole global
    # index, silently dropping every other session's rows (observed: 52,612 ->
    # 4,504 rows after a 10-session run).  Subset mode now merges: rows for the
    # requested (session, camera) pairs are replaced, all others are preserved.
    rebuilt_pairs = {(session, f"cam{cam}") for session in wanted for cam in cams}
    keep_rows: list[str] = []
    if args.sessions and index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (row.get("session"), row.get("camera")) not in rebuilt_pairs:
                    keep_rows.append(line)
    total = valid_total = 0
    index_handle = index_path.open("w", encoding="utf-8")
    try:
        index_handle.writelines(keep_rows)
        for cam in cams:
            for session in wanted:
                if session not in manifest:
                    raise SystemExit(f"session not eligible or missing from manifest: {session}")
                count, valid, details = build_session(session, manifest[session], cam, force=args.force)
                total += count
                valid_total += valid
                for item in details.get("index_rows", []):
                    index_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                print(f"{session} (cam{cam}): frames={count} valid_bbox={valid}")
    finally:
        index_handle.close()
    write_artifact(FRONTEND_ROOT, schema_version="shared.frontend.human_masks.sam31.v1", producer="build_human_mask_frontend.py", repository_root=ROOT,
                   parameters={"session_count": len(wanted), "camera": ",".join(f"cam{cam}" for cam in cams), "bbox_source": "masks_index.csv",
                               "mode": "direct_csv_bbox", "match": "visual_time_s nearest <=20ms (time_error_s recorded)"},
                   source_artifacts=["external://rgb_human_masks"], frame_id_min=0, frame_id_max=None, frame_count=total,
                   consumers=["CLIFF", "pressure_toolkit"])
    print(f"wrote {index_path}: frames={total} valid_bbox={valid_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
