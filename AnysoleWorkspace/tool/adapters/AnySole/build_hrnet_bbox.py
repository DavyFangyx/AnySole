#!/usr/bin/env python3
"""Build the AnySole HRNet bbox shim from the PoseTransOpt CLIFF mask bboxes.

``anysole/data/extract_hrnet.py`` needs one pixel-space ``bbox.npy`` per
session (row ``idx`` = ``[i, x1, y1, x2, y2, score, 0.99, 0]``, the
MotionPRO ``gen_bbox`` convention) and slices ``box = bbox[idx][1:5]``.  The
only missing input class for the AnySole visual cache is that file: the raw
mocap tree never carried a ``bbox.npy`` and no detector is re-run here.

The bbox source is the already-verified single-person detection of the
PoseTransOpt adapter:

    model-input://PoseTransOpt/adapter_v1/<date>/<subject>/<session>/CLIFF_results.npz
        frame_id       int64[N]     canonical frames (== shared facts frame_id)
        bbox_xyxy_px   float32[N,4] bbox recomputed from the SAM3.1 binary mask,
                                    in mask pixel coordinates (the mask canvas is
                                    the original RGB resolution)
        valid          uint8[N]     1 = mask present/non-empty/within 20ms join

Frames whose CLIFF row is invalid carry the **nearest valid frame's** bbox
(tie -> earlier frame).  CLIFF leaves them zero-filled by contract, and a zero
box would make ``process_image`` fall back to a centre crop; the copy keeps the
crop scale stable across a short dropout while staying visible in the artifact
(``frame_fallback_frames``).

Exception -- a session with **no** valid CLIFF row (2026-09-28: S12102, all 284
frames ``valid=0``/``fake=1`` in shared facts, i.e. zero training windows):
CLIFF has no bbox at all for it, but the SAM3.1 frontend masks of that session
exist and are ``status=ok`` -- CLIFF drops them only because the shared frame
axis is invalid.  Such a session is filled from the frontend index directly
(``shared/frontends/human_masks/sam31/v1/index.jsonl`` -> ``mask_path`` ->
nonzero-mask bbox, the same ``run_cliff.mask_bbox`` rule), so the bbox and the
HRNet cache stay real for real images instead of being fabricated.  Every
exception session is named in the artifact (``sessions_without_valid_cliff``
and ``sessions_exception_source``); a session with neither source stays an
error.

Output (one file per session, plus one directory-level artifact):

    AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox/<session>.npy
    AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox/artifact.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.artifacts import write_artifact  # noqa: E402
from anysole.data.workspace_adapter import load_shared_session  # noqa: E402


FACTS_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
SPLIT_CSV = ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
CLIFF_ROOT = ROOT / "AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1"
MASK_INDEX = ROOT / "AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"
OUTPUT_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox"
PRODUCER = "AnysoleWorkspace/tool/adapters/AnySole/build_hrnet_bbox.py"
SCHEMA_VERSION = "model_input.anysole.v1"
BBOX_SOURCE = "CLIFF_results.npz bbox_xyxy_px (SAM3.1 mask recomputed)"
FALLBACK = "nearest valid frame"
EXCEPTION_SOURCE = ("shared/frontends/human_masks/sam31/v1 mask bbox "
                    "(session has no valid CLIFF row; same SAM3.1 masks)")
SCORE = 1.0


def split_session_ids(split_csv: Path, split: str) -> list:
    """Session ids of one splits.csv column (train/val/test)."""
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and split not in rows[0]:
        raise ValueError("%s missing column %s" % (split_csv, split))
    seen, ids = set(), []
    for row in rows:
        value = (row.get(split) or "").strip()
        if value and value not in seen:
            seen.add(value)
            ids.append(value)
    return ids


def all_session_ids(split_csv: Path = SPLIT_CSV) -> list:
    """Union of the train/val/test columns, in splits.csv order."""
    out = []
    for split in ("train", "val", "test"):
        for session_id in split_session_ids(split_csv, split):
            if session_id not in out:
                out.append(session_id)
    return out


def cliff_path(session_id: str) -> Path:
    matches = sorted(CLIFF_ROOT.glob("*/*/%s/CLIFF_results.npz" % session_id))
    if not matches:
        raise FileNotFoundError(
            "%s: no CLIFF_results.npz under %s" % (session_id, CLIFF_ROOT)
        )
    if len(matches) > 1:
        raise ValueError("%s: ambiguous CLIFF_results.npz %s" % (session_id, matches))
    return matches[0]


def nearest_valid_bbox(valid: np.ndarray, bbox: np.ndarray) -> tuple:
    """Replace invalid rows with the nearest valid frame's bbox.

    Tie -> the earlier frame.  Returns ``(out, fallback_rows)`` where
    ``fallback_rows`` are the invalid frame indices that were filled (always
    all of them, or an error when nothing is valid).
    """
    valid = np.asarray(valid).astype(bool)
    bbox = np.asarray(bbox, dtype=np.float64)
    good = np.flatnonzero(valid)
    if good.size == 0:
        raise ValueError("no valid CLIFF frame to source a bbox from")
    out = bbox.copy()
    bad = np.flatnonzero(~valid)
    if bad.size:
        positions = np.searchsorted(good, bad)
        right = good[np.clip(positions, 0, good.size - 1)]
        left = good[np.clip(positions - 1, 0, good.size - 1)]
        take_left = (bad - left) <= (right - bad)
        nearest = np.where(take_left, left, right)
        out[bad] = bbox[nearest]
    return out, bad


def mask_bbox_px(mask_path: Path):
    """Nonzero-mask bbox in mask pixel coordinates (``run_cliff.mask_bbox`` rule)."""
    from PIL import Image
    if not mask_path.is_file():
        return None
    image = np.asarray(Image.open(mask_path))
    ys, xs = np.nonzero(image > 0)
    if xs.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def frontend_mask_bboxes(session_id: str, n_frames: int) -> tuple:
    """Per-frame mask bbox for a session with no usable CLIFF row.

    The frontend index already carries the canonical ``frame_id`` per mask, so
    rows are addressed directly rather than joined by time.  Returns
    ``(bbox, valid)`` over the canonical frame axis.
    """
    bbox = np.zeros((n_frames, 4), dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)
    if not MASK_INDEX.is_file():
        raise FileNotFoundError("missing SAM3.1 mask index %s" % MASK_INDEX)
    with MASK_INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("session") != session_id:
                continue
            frame = int(row["frame_id"])
            if not 0 <= frame < n_frames:
                raise ValueError("%s: frontend frame_id %d outside [0,%d)"
                                 % (session_id, frame, n_frames))
            box = mask_bbox_px(Path(str(row.get("mask_path", ""))))
            if box is None:
                continue
            bbox[frame] = box
            valid[frame] = True
    if not valid.any():
        raise ValueError("%s: neither CLIFF nor the SAM3.1 frontend has a usable bbox"
                         % session_id)
    return bbox, valid


def build_one(session_id: str, force: bool = False, allow_exception: bool = True) -> dict:
    """Write one session's bbox.npy (stats always computed from the sources, so
    an incremental run still reports honest fallback counts)."""
    output = OUTPUT_ROOT / ("%s.npy" % session_id)
    session = load_shared_session(session_id)
    frames = np.asarray(session["frames"]["frame_id"])
    n_frames = len(frames)

    data = np.load(cliff_path(session_id), allow_pickle=True)
    frame_id = np.asarray(data["frame_id"])
    if not np.array_equal(frame_id, frames):
        raise ValueError(
            "%s: CLIFF frame axis does not match shared facts (%s vs %s)"
            % (session_id, frame_id.shape, frames.shape)
        )
    bbox = np.asarray(data["bbox_xyxy_px"], dtype=np.float64)
    if bbox.shape != (n_frames, 4):
        raise ValueError("%s: CLIFF bbox_xyxy_px shape %s != (%d, 4)"
                         % (session_id, bbox.shape, n_frames))

    valid = np.asarray(data["valid"]).astype(bool)
    source = BBOX_SOURCE
    if not valid.any():
        if not allow_exception:
            raise ValueError("%s: no valid CLIFF frame to source a bbox from" % session_id)
        bbox, valid = frontend_mask_bboxes(session_id, n_frames)
        source = EXCEPTION_SOURCE

    filled, fallback_rows = nearest_valid_bbox(valid, bbox)
    if not np.isfinite(filled).all():
        raise ValueError("%s: non-finite bbox after fallback fill" % session_id)

    stats = {
        "session": session_id,
        "frames": n_frames,
        "valid_frames": int(valid.sum()),
        "fallback_frames": int(fallback_rows.size),
        "bbox_source": source,
    }
    if output.is_file() and not force:
        return dict(stats, status="skip_existing")

    rows = np.zeros((n_frames, 8), dtype=np.float64)
    rows[:, 0] = np.arange(n_frames)
    rows[:, 1:5] = filled
    rows[:, 5] = SCORE
    rows[:, 6] = 0.99
    rows[:, 7] = 0.0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npy")
    np.save(temporary, rows)
    temporary.replace(output)
    return dict(stats, status="written")


def write_dir_artifact(results: list) -> Path:
    """One artifact for the bbox directory (all sessions share one producer)."""
    sessions = [item["session"] for item in results]
    total_frames = int(sum(item["frames"] for item in results))
    fallback_frames = int(sum(item.get("fallback_frames", 0) for item in results))
    exceptions = sorted(item["session"] for item in results
                        if item.get("bbox_source") == EXCEPTION_SOURCE)
    return write_artifact(
        OUTPUT_ROOT,
        schema_version=SCHEMA_VERSION,
        producer=PRODUCER,
        repository_root=ROOT,
        parameters={
            "bbox_source": BBOX_SOURCE,
            "fallback": FALLBACK,
            "sessions": len(sessions),
            "bbox_convention": "[i, x1, y1, x2, y2, score, 0.99, 0] (MotionPRO gen_bbox)",
            "frame_fallback_frames": fallback_frames,
            "frames_total": total_frames,
            "sessions_without_valid_cliff": exceptions,
            "sessions_exception_source": EXCEPTION_SOURCE if exceptions else None,
        },
        source_artifacts=["model-input://PoseTransOpt/adapter_v1"],
        source_hashes={"sessions_covered": len(sessions)},
        frame_id_min=0,
        frame_count=len(sessions),
        consumers=["AnySole"],
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="Build one session only.")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                        help="splits.csv column scope when --session is empty.")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true",
                      help="Keep sessions whose bbox.npy already exists (default).")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true",
                      help="Recompute and overwrite existing bbox.npy files.")
    parser.add_argument("--no-artifact", action="store_true",
                        help="Skip writing the directory artifact.json.")
    parser.add_argument("--no-exception-footer", dest="allow_exception", action="store_false",
                        help="Fail instead of using the SAM3.1 frontend mask bbox for a "
                             "session whose CLIFF rows are all invalid.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.session:
        session_ids = [args.session]
    elif args.split == "all":
        session_ids = all_session_ids(args.split_csv)
    else:
        session_ids = split_session_ids(args.split_csv, args.split)
    if not session_ids:
        raise FileNotFoundError("no sessions selected (split=%s)" % args.split)

    results = []
    for session_id in session_ids:
        item = build_one(session_id, force=args.force, allow_exception=args.allow_exception)
        results.append(item)
        if item["status"] == "written":
            print("bbox %s frames=%d valid=%d fallback=%d source=%s"
                  % (session_id, item["frames"], item["valid_frames"],
                     item["fallback_frames"], item["bbox_source"]))
        else:
            print("skip %s (%s exists)" % (session_id, OUTPUT_ROOT / (session_id + ".npy")))
    if args.session and not args.no_artifact:
        # The directory artifact describes the whole bbox directory; a
        # one-session run must not restate it as a 1-session artifact.
        print("note: --session given, directory artifact.json left untouched")
    elif not args.no_artifact:
        path = write_dir_artifact(results)
        print("artifact %s (%d sessions)" % (path, len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
