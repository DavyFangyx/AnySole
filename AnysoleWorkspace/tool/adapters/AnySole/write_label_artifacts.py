#!/usr/bin/env python3
"""Write the per-method ``artifact.json`` of the AnySole contact-label tree.

One artifact per contact method under

    AnysoleWorkspace/model_inputs/AnySole/adapter_v1/labels/<method>/

covering the 140 canonical sessions produced by
``python -m anysole.data.contact_adapter --split train|val|test --methods all``
(2026-09-28).  Each ``<session>.npz`` carries the shared-frame axis plus its own
provenance; the directory artifact states which method the directory holds and
binds the whole directory with two deterministic hashes over

    sorted("session_id:source_session_hash")   -> sha256 (session binding)
    sorted("session_id:contact_sha256")        -> sha256 (label content)

so a later consumer can detect a partially refilled directory *and* a directory
whose method was swapped -- the session-level hash alone is method-independent
because ``source_session_hash`` hashes the shared session artifact, not the
label.

Run (touch_gait):

    python AnysoleWorkspace/tool/adapters/AnySole/write_label_artifacts.py
    python AnysoleWorkspace/tool/adapters/AnySole/write_label_artifacts.py --method f6_soft
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.artifacts import write_artifact  # noqa: E402


SCHEMA_VERSION = "model_input.anysole.labels.v1"
PRODUCER = "anysole.data.contact_adapter"
ADAPTER_VERSION = "adapter_v1"
LABEL_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/labels"
SPLIT_CSV = ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
SOURCE_ARTIFACTS = ["shared://facts/sessions", "raw://bvh"]
CONSUMERS = ["AnySole"]
NPZ_KEYS = {
    "frame_id", "contact", "valid", "fake", "method",
    "adapter_version", "parameters_json", "source_session", "source_session_hash",
}


def split_session_ids(split_csv: Path = SPLIT_CSV) -> list:
    """Canonical session ids: union of the train/val/test columns, file order."""
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for column in ("train", "val", "test"):
        for row in rows:
            value = (row.get(column) or "").strip()
            if value and value not in out:
                out.append(value)
    return out


def manifest_hash(pairs) -> str:
    """sha256 over the sorted ``session_id:source_session_hash`` lines."""
    payload = "\n".join("%s:%s" % (session_id, digest)
                        for session_id, digest in sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def method_dirs_in(root: Path) -> list:
    return sorted(path for path in root.iterdir()
                  if path.is_dir() and not path.name.startswith("."))


def read_session_record(path: Path, method: str) -> dict:
    """Read one label npz's provenance plus the content hash of its contact array."""
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        if keys != NPZ_KEYS:
            raise ValueError("%s: npz keys %s != %s" % (path, sorted(keys), sorted(NPZ_KEYS)))
        record = json.loads(str(data["parameters_json"]))
        contact = np.ascontiguousarray(np.asarray(data["contact"], dtype=np.float32))
        return {
            "session_id": path.stem,
            "method": str(data["method"]),
            "adapter_version": str(data["adapter_version"]),
            "source_session_hash": str(data["source_session_hash"]),
            "contact_sha256": hashlib.sha256(contact.tobytes()).hexdigest(),
            "constant_value": float(contact.flat[0]) if np.unique(contact).size == 1 else None,
            "semantics": str(record.get("semantics", "")),
        }


def scan_method(method_dir: Path) -> dict:
    """Validate one method directory and collect its hash manifest."""
    method = method_dir.name
    paths = sorted(method_dir.glob("*.npz"))
    if not paths:
        raise ValueError("%s: no label npz" % method_dir)
    records = [read_session_record(path, method) for path in paths]
    for record in records:
        if record["method"] != method:
            raise ValueError("%s: npz declares method %r" % (method_dir, record["method"]))
        if record["adapter_version"] != ADAPTER_VERSION:
            raise ValueError("%s: npz adapter_version %r != %s"
                             % (method_dir, record["adapter_version"], ADAPTER_VERSION))
        if not record["semantics"]:
            raise ValueError("%s: npz parameters_json has no semantics" % method_dir)
    semantics = sorted({record["semantics"] for record in records})
    if len(semantics) != 1:
        raise ValueError("%s: %d distinct semantics strings across sessions"
                         % (method_dir, len(semantics)))
    pairs = [(record["session_id"], record["source_session_hash"]) for record in records]
    contact_pairs = [(record["session_id"], record["contact_sha256"]) for record in records]
    return {
        "method": method,
        "sessions": len(records),
        "semantics": semantics[0],
        "manifest_sha256": manifest_hash(pairs),
        "contact_manifest_sha256": manifest_hash(contact_pairs),
        "session_ids": sorted(record["session_id"] for record in records),
        # Diagnostics only (not written into the artifact): sessions whose
        # contact array is a single constant value carry no per-frame signal.
        "constant_sessions": sorted(
            (record["session_id"], record["constant_value"]) for record in records
            if record["constant_value"] is not None),
    }


def write_method_artifact(method_dir: Path, scan: dict) -> Path:
    return write_artifact(
        method_dir,
        schema_version=SCHEMA_VERSION,
        producer=PRODUCER,
        repository_root=ROOT,
        parameters={
            "contact_method": scan["method"],
            "adapter_version": ADAPTER_VERSION,
            "sessions": scan["sessions"],
            "semantics": scan["semantics"],
            "contact_manifest_sha256": scan["contact_manifest_sha256"],
        },
        source_artifacts=SOURCE_ARTIFACTS,
        source_hashes={
            "sessions_covered": scan["sessions"],
            "sessions_manifest_sha256": scan["manifest_sha256"],
        },
        frame_count=scan["sessions"],
        consumers=CONSUMERS,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append",
                        help="Limit to one method (repeatable); default all.")
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--expected-sessions", type=int, default=None,
                        help="Fail unless each method covers this many sessions.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.label_root)
    if args.method:
        wanted = set(args.method)
        dirs = [path for path in method_dirs_in(root) if path.name in wanted]
        missing = wanted - {path.name for path in dirs}
        if missing:
            raise FileNotFoundError("no label dir for %s under %s" % (sorted(missing), root))
    else:
        dirs = method_dirs_in(root)
    if not dirs:
        raise FileNotFoundError("no method dirs under %s" % root)

    expected = args.expected_sessions
    if expected is None:
        expected = len(split_session_ids())
    canonical = set(split_session_ids())
    for method_dir in dirs:
        scan = scan_method(method_dir)
        gaps = canonical - set(scan["session_ids"])
        extra = set(scan["session_ids"]) - canonical
        if gaps or extra:
            raise ValueError("%s: session set != splits.csv union (missing %s, extra %s)"
                             % (method_dir, sorted(gaps)[:5], sorted(extra)[:5]))
        if scan["sessions"] != expected:
            raise ValueError("%s: %d sessions != expected %d"
                             % (method_dir, scan["sessions"], expected))
        if args.dry_run:
            print("dry-run %s sessions=%d semantics=%r manifest=%s contact=%s"
                  % (method_dir / "artifact.json", scan["sessions"],
                     scan["semantics"][:48], scan["manifest_sha256"][:16],
                     scan["contact_manifest_sha256"][:16]))
            continue
        path = write_method_artifact(method_dir, scan)
        print("wrote %s sessions=%d semantics=%r manifest=%s contact=%s"
              % (path, scan["sessions"], scan["semantics"][:48],
                 scan["manifest_sha256"][:16], scan["contact_manifest_sha256"][:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
