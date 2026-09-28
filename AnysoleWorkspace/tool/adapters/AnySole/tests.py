#!/usr/bin/env python3
"""AnySole adapter acceptance tests: label artifacts + consumer reads.

Run from the repo root in the ``touch_gait`` environment:

    python AnysoleWorkspace/tool/adapters/AnySole/tests.py
    python AnysoleWorkspace/tool/adapters/AnySole/tests.py --method f6_soft

T1 段 (always): per-method ``artifact.json`` contract (method, 140 sessions,
deterministic manifest hashes recomputed from the npz tree), the 140-file
session set against ``protocol/splits/default/splits.csv`` (104/36/36, val ==
test), the npz frame axis against the shared facts (frame_id / valid / fake),
and the AnySole-private loader ``workspace_adapter.load_contact_label`` the
dataset itself calls.

T2 段: the HRNet visual cache check.  It is a stub while the cache is being
built -- it activates per session once ``<cache_root>/<session>.pt`` and its
``write_cache_provenance`` sidecar exist, and becomes a hard check for the
whole set once all 140 sessions are present.  The bbox shim
(``build_hrnet_bbox.py``) is verified whenever the files exist, because it is
the crop source of that cache.

Non-fatal diagnostics are printed with ``note`` / ``warn`` and never fail the
run; they report label-tree properties worth a human look (identical method
pairs, constant-contact sessions).
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

from AnysoleWorkspace.tool.adapters.AnySole import write_label_artifacts as wla  # noqa: E402

LABEL_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/labels"
SPLIT_CSV = ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
FACTS_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
BBOX_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/bbox"
HRNET_CACHE_ROOT = ROOT / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet/cam3"
CLIFF_ROOT = ROOT / "AnysoleWorkspace/model_inputs/PoseTransOpt/adapter_v1"
CREATE_PROVENANCE_SCHEMA = "model_input.anysole.v1"
V_FEAT_DIM = 2051
SAMPLE_SESSIONS = ("S5011", "S10101", "S11073", "S13011", "S14103")
EXPECTED_SPLIT_SIZES = {"train": 104, "val": 36, "test": 36}

_PASSED = 0
_NOTES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED
    if not condition:
        raise AssertionError("%s FAILED: %s" % (name, detail))
    _PASSED += 1
    print("  ok %s" % name)


def note(message: str) -> None:
    _NOTES.append(message)
    print("  note %s" % message)


def warn(message: str) -> None:
    _NOTES.append(message)
    print("  warn %s" % message)


def section(title: str) -> None:
    print("\n== %s" % title)


def split_columns() -> dict:
    """train/val/test session columns of the canonical split file."""
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {column: [row[column].strip() for row in rows if (row.get(column) or "").strip()]
            for column in ("train", "val", "test")}


def shared_frames(session_id: str) -> dict:
    """Canonical frame axis of one shared session."""
    matches = sorted(FACTS_ROOT.glob("*/*/%s/frames.npz" % session_id))
    if len(matches) != 1:
        raise AssertionError("shared facts for %s: %d matches" % (session_id, len(matches)))
    with np.load(matches[0], allow_pickle=True) as data:
        return {"frame_id": np.asarray(data["frame_id"]),
                "valid": np.asarray(data["valid"]),
                "fake": np.asarray(data["fake"]),
                "dir": matches[0].parent}


def test_split_contract() -> dict:
    section("split contract %s" % SPLIT_CSV.name)
    columns = split_columns()
    for column, expected in EXPECTED_SPLIT_SIZES.items():
        check("split %s has %d sessions" % (column, expected),
              len(columns[column]) == expected, "got %d" % len(columns[column]))
        check("split %s has no duplicate session" % column,
              len(set(columns[column])) == len(columns[column]))
    check("val == test (T1 ruling)", columns["val"] == columns["test"])
    check("train/val disjoint", not (set(columns["train"]) & set(columns["val"])))
    union = set(columns["train"]) | set(columns["val"]) | set(columns["test"])
    check("split union has 140 sessions", len(union) == 140, "got %d" % len(union))
    for session_id in SAMPLE_SESSIONS:
        check("sampled session %s is in the split union" % session_id,
              session_id in union)
    return columns


def test_method_artifact(method_dir: Path, union: set) -> dict:
    method = method_dir.name
    section("method %s" % method)
    artifact_path = method_dir / "artifact.json"
    check("artifact.json exists", artifact_path.is_file(), str(artifact_path))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    parameters = artifact.get("parameters", {})
    check("artifact schema_version", artifact.get("schema_version") == wla.SCHEMA_VERSION,
          str(artifact.get("schema_version")))
    check("artifact producer", artifact.get("producer") == wla.PRODUCER)
    check("artifact declares the method",
          parameters.get("contact_method") == method, str(parameters.get("contact_method")))
    check("artifact declares 140 sessions",
          parameters.get("sessions") == 140, str(parameters.get("sessions")))
    check("artifact adapter_version", parameters.get("adapter_version") == wla.ADAPTER_VERSION)
    check("artifact carries a non-empty semantics string",
          bool(str(parameters.get("semantics", "")).strip()))
    check("artifact source_artifacts",
          artifact.get("source_artifacts") == wla.SOURCE_ARTIFACTS,
          str(artifact.get("source_artifacts")))
    check("artifact consumers", artifact.get("consumers") == wla.CONSUMERS)
    check("artifact frame_count == sessions",
          artifact.get("frame_count") == 140, str(artifact.get("frame_count")))
    check("artifact source_hashes.sessions_covered == 140",
          artifact.get("source_hashes", {}).get("sessions_covered") == 140)

    npz_paths = sorted(method_dir.glob("*.npz"))
    session_ids = sorted(path.stem for path in npz_paths)
    check("exactly 140 label npz", len(npz_paths) == 140, "got %d" % len(npz_paths))
    check("label session set == split union", set(session_ids) == union,
          "missing %s extra %s" % (sorted(union - set(session_ids))[:3],
                                   sorted(set(session_ids) - union)[:3]))

    scan = wla.scan_method(method_dir)
    check("manifest hash is reproducible from the npz tree",
          artifact["source_hashes"].get("sessions_manifest_sha256") == scan["manifest_sha256"],
          "%s != %s" % (artifact["source_hashes"].get("sessions_manifest_sha256"),
                        scan["manifest_sha256"]))
    check("contact manifest hash is reproducible",
          parameters.get("contact_manifest_sha256") == scan["contact_manifest_sha256"],
          "%s != %s" % (parameters.get("contact_manifest_sha256"),
                        scan["contact_manifest_sha256"]))
    check("artifact semantics == npz semantics",
          parameters.get("semantics") == scan["semantics"])
    return {"method": method,
            "contact_manifest_sha256": scan["contact_manifest_sha256"],
            "constant_sessions": scan["constant_sessions"],
            "semantics": scan["semantics"]}


def test_samples(method_dir: Path) -> list:
    """Per-method sampled npz contract: 9 keys, frame axis, masks, method."""
    method = method_dir.name
    session_ids = sorted(path.stem for path in method_dir.glob("*.npz"))
    samples = [session_id for session_id in SAMPLE_SESSIONS if session_id in session_ids]
    check("%s: 5 sampled sessions present" % method, len(samples) == len(SAMPLE_SESSIONS),
          "got %s" % samples)
    constants = []
    for session_id in samples:
        path = method_dir / ("%s.npz" % session_id)
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            frame_id = np.asarray(data["frame_id"])
            contact = np.asarray(data["contact"])
            valid = np.asarray(data["valid"])
            fake = np.asarray(data["fake"])
            declared_method = str(data["method"])
            adapter_version = str(data["adapter_version"])
            source_session = str(data["source_session"])
            source_hash = str(data["source_session_hash"])
        frames = shared_frames(session_id)
        expected = frames["frame_id"]
        shared_valid = np.asarray(frames["valid"], dtype=np.uint8)
        shared_fake = np.asarray(frames["fake"], dtype=np.uint8)
        check("%s/%s: npz has the 9 contract keys" % (method, session_id),
              keys == wla.NPZ_KEYS, str(sorted(keys)))
        check("%s/%s: frame_id length == shared length %d" % (method, session_id, len(expected)),
              len(frame_id) == len(expected), "got %d" % len(frame_id))
        check("%s/%s: frame_id first/last == shared first/last (%d/%d)"
              % (method, session_id, int(expected[0]), int(expected[-1])),
              len(frame_id) > 0 and int(frame_id[0]) == int(expected[0])
              and int(frame_id[-1]) == int(expected[-1]))
        check("%s/%s: frame_id == arange(N) == shared frame_id" % (method, session_id),
              np.array_equal(frame_id, np.arange(len(frame_id), dtype=frame_id.dtype))
              and np.array_equal(frame_id, expected))
        check("%s/%s: valid matches shared facts byte-for-byte" % (method, session_id),
              valid.dtype == np.uint8 and valid.tobytes() == shared_valid.tobytes())
        check("%s/%s: fake matches shared facts byte-for-byte" % (method, session_id),
              fake.dtype == np.uint8 and fake.tobytes() == shared_fake.tobytes())
        check("%s/%s: contact shape (N,2) float32 finite" % (method, session_id),
              contact.shape == (len(expected), 2) and contact.dtype == np.float32
              and bool(np.isfinite(contact).all()),
              "%s %s" % (contact.shape, contact.dtype))
        check("%s/%s: method string == dir name" % (method, session_id),
              declared_method == method, declared_method)
        check("%s/%s: adapter_version" % (method, session_id),
              adapter_version == wla.ADAPTER_VERSION, adapter_version)
        check("%s/%s: source_session uri points at the shared session" % (method, session_id),
              source_session.endswith("/%s" % session_id)
              and str(frames["dir"]).endswith(session_id))
        sidecar = frames["dir"] / "artifact.json"
        check("%s/%s: source_session_hash == shared artifact hash" % (method, session_id),
              source_hash == hashlib.sha256(sidecar.read_bytes()).hexdigest())
        if np.unique(contact).size == 1:
            constants.append((session_id, float(contact[0, 0])))
    return constants


def test_loader(method: str, session_id: str = "S10101") -> None:
    """The dataset's own read path: workspace_adapter.load_contact_label."""
    section("consumer loader (anysole.data.workspace_adapter)")
    import anysole.data.workspace_adapter as wa

    check("adapter LABEL_ROOT is the produced tree", Path(wa.LABEL_ROOT) == LABEL_ROOT,
          str(wa.LABEL_ROOT))
    session = wa.load_shared_session(session_id)
    path = wa.label_path(session_id, method)
    check("adapter label_path resolves to the produced npz",
          path == LABEL_ROOT / method / ("%s.npz" % session_id), str(path))
    contact = wa.load_contact_label(session, method)
    with np.load(path, allow_pickle=True) as data:
        on_disk = np.asarray(data["contact"], dtype=np.float32)
    check("load_contact_label returns the on-disk contact array",
          contact.shape == on_disk.shape and np.array_equal(contact, on_disk),
          str(contact.shape))
    check("load_contact_label length == shared frame count",
          contact.shape[0] == len(session["frames"]["frame_id"]))
    check("load_contact_label dtype float32", contact.dtype == np.float32)
    # Second method + second session: the loader is per-(session, method).
    other = wa.load_contact_label(session, "tactile_rel" if method != "tactile_rel" else "f6_soft")
    check("loader reads a second label of the same session",
          other.shape == contact.shape, str(other.shape))


def test_bbox_sample(unioned: set) -> None:
    """The HRNet crop source: bbox shim present for the sampled sessions."""
    section("bbox shim (crop source of the HRNet cache)")
    if not BBOX_ROOT.is_dir():
        warn("bbox root missing (%s): build_hrnet_bbox.py has not been run" % BBOX_ROOT)
        return
    files = sorted(BBOX_ROOT.glob("*.npy"))
    artifact_path = BBOX_ROOT / "artifact.json"
    check("bbox artifact.json exists", artifact_path.is_file(), str(artifact_path))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    check("bbox artifact producer",
          artifact["producer"].endswith("build_hrnet_bbox.py"), str(artifact["producer"]))
    check("bbox artifact consumers", artifact["consumers"] == ["AnySole"])
    check("bbox artifact source_artifacts",
          artifact["source_artifacts"] == ["model-input://PoseTransOpt/adapter_v1"])
    check("bbox artifact declares the nearest-valid fallback",
          "nearest valid frame" in str(artifact["parameters"].get("fallback", "")))
    check("bbox artifact covers every split session",
          {path.stem for path in files} == unioned,
          "missing %s" % sorted(unioned - {path.stem for path in files})[:3])
    for session_id in SAMPLE_SESSIONS:
        path = BBOX_ROOT / ("%s.npy" % session_id)
        check("bbox %s exists" % session_id, path.is_file(), str(path))
        bbox = np.load(path)
        n = len(shared_frames(session_id)["frame_id"])
        check("bbox %s shape (%d,8)" % (session_id, n), bbox.shape == (n, 8), str(bbox.shape))
        check("bbox %s row index column" % session_id,
              np.array_equal(bbox[:, 0], np.arange(n, dtype=bbox.dtype)))
        check("bbox %s score/extra columns" % session_id,
              np.allclose(bbox[:, 5], 1.0) and np.allclose(bbox[:, 6], 0.99)
              and np.allclose(bbox[:, 7], 0.0))
        check("bbox %s valid corners" % session_id,
              np.all(bbox[:, 3] > bbox[:, 1]) and np.all(bbox[:, 4] > bbox[:, 2])
              and np.all(np.isfinite(bbox)))
        cliff = sorted(CLIFF_ROOT.glob("*/*/%s/CLIFF_results.npz" % session_id))[0]
        with np.load(cliff, allow_pickle=True) as data:
            cliff_bbox = np.asarray(data["bbox_xyxy_px"], dtype=np.float64)
            cliff_valid = np.asarray(data["valid"]).astype(bool)
        check("bbox %s matches CLIFF on its valid frames" % session_id,
              np.allclose(bbox[cliff_valid, 1:5], cliff_bbox[cliff_valid]),
              "max diff %.4g" % float(np.abs(bbox[cliff_valid, 1:5] - cliff_bbox[cliff_valid]).max())
              if cliff_valid.any() else "no valid CLIFF frame")
        invalid = np.flatnonzero(~cliff_valid)
        if invalid.size:
            good = np.flatnonzero(cliff_valid)
            # Independent restatement of the documented fallback: every invalid
            # frame carries the bbox of the nearest valid frame (tie -> earlier).
            nearest = good[np.argmin(np.abs(good[None, :] - invalid[:, None]), axis=1)]
            check("bbox %s fills %d invalid frames from the nearest valid frame"
                  % (session_id, invalid.size),
                  np.allclose(bbox[invalid, 1:5], cliff_bbox[nearest]),
                  "max diff %.4g" % float(np.abs(bbox[invalid, 1:5] - cliff_bbox[nearest]).max()))


def test_hrnet_cache_stub(unioned: set) -> None:
    """Stub: activates per session once the HRNet cache exists."""
    section("HRNet cache (stub until the cache is built)")
    if not HRNET_CACHE_ROOT.is_dir():
        warn("HRNet cache root missing (%s): no cache built yet; stub stays inactive"
             % HRNET_CACHE_ROOT)
        return
    import torch
    from anysole.data.dataset import hrnet_cache_path as dataset_hrnet_cache_path
    from anysole.data.workspace_adapter import hrnet_cache_path as adapter_hrnet_cache_path

    caches = sorted(HRNET_CACHE_ROOT.glob("*.pt"))
    check("cache root holds only <session>.pt files",
          all(path.name == "%s.pt" % path.stem for path in caches))
    check("every cached session belongs to the split union",
          {path.stem for path in caches} <= unioned,
          str(sorted({path.stem for path in caches} - unioned)[:3]))
    check("dataset and adapter agree on the cache path",
          dataset_hrnet_cache_path("S13073") == adapter_hrnet_cache_path("S13073") ==
          HRNET_CACHE_ROOT / "S13073.pt")
    for path in caches:
        session_id = path.stem
        feature = torch.load(path, map_location="cpu")
        n = len(shared_frames(session_id)["frame_id"])
        check("cache %s is a (%d,%d) float32 tensor" % (session_id, n, V_FEAT_DIM),
              torch.is_tensor(feature) and tuple(feature.shape) == (n, V_FEAT_DIM)
              and feature.dtype == torch.float32 and bool(torch.isfinite(feature).all()),
              "%s %s" % (tuple(feature.shape) if torch.is_tensor(feature) else type(feature),
                         feature.dtype if torch.is_tensor(feature) else ""))
        sidecar = Path(str(path) + ".artifact.json")
        check("cache %s provenance sidecar exists" % session_id, sidecar.is_file(), str(sidecar))
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        check("cache %s provenance schema" % session_id,
              provenance.get("schema_version") == CREATE_PROVENANCE_SCHEMA,
              str(provenance.get("schema_version")))
        check("cache %s provenance session/shape" % session_id,
              provenance.get("session_id") == session_id
              and provenance.get("shape") == [n, V_FEAT_DIM]
              and provenance.get("frame_count") == n,
              str({k: provenance.get(k) for k in ("session_id", "shape", "frame_count")}))
    if len(caches) == len(unioned):
        check("HRNet cache covers all %d sessions" % len(unioned), True)
    else:
        warn("HRNet cache is partial: %d/%d sessions built; the full-set check "
             "activates when the cache is complete" % (len(caches), len(unioned)))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", help="Limit to one method (repeatable).")
    parser.add_argument("--loader-session", default="S10101",
                        help="Session used for the workspace_adapter loader check.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    columns = test_split_contract()
    union = set(columns["train"]) | set(columns["val"]) | set(columns["test"])

    method_dirs = wla.method_dirs_in(LABEL_ROOT)
    if args.method:
        wanted = set(args.method)
        method_dirs = [path for path in method_dirs if path.name in wanted]
        missing = wanted - {path.name for path in method_dirs}
        if missing:
            raise SystemExit("no label dir for %s" % sorted(missing))
    else:
        check("10 contact methods present", len(method_dirs) == 10,
              str([path.name for path in method_dirs]))

    manifests = {}
    for method_dir in method_dirs:
        info = test_method_artifact(method_dir, union)
        info["sampled_constants"] = test_samples(method_dir)
        manifests[info["method"]] = info

    for method, info in sorted(manifests.items()):
        constant = info["constant_sessions"]
        if constant:
            values = sorted({value for _, value in constant})
            warn("%s: %d/%d sessions have a constant contact array (values %s, e.g. %s)"
                 % (method, len(constant), 140, values, [sid for sid, _ in constant[:3]]))
        if len(info["sampled_constants"]) != len(constant) and info["sampled_constants"]:
            note("%s: sampled-constant sessions %s" % (method, info["sampled_constants"]))

    by_content = {}
    for method, info in manifests.items():
        by_content.setdefault(info["contact_manifest_sha256"], []).append(method)
    duplicates = {digest: names for digest, names in by_content.items() if len(names) > 1}
    if duplicates:
        for digest, names in sorted(duplicates.items()):
            warn("identical contact content: %s (%s)" % (" == ".join(sorted(names)), digest[:12]))
    else:
        note("all %d method directories have distinct contact content" % len(manifests))

    test_loader(sorted(manifests)[0] if manifests else "f6_soft", args.loader_session)
    test_bbox_sample(union)
    test_hrnet_cache_stub(union)

    section("summary")
    print("  %d checks passed, %d notes/warnings" % (_PASSED, len(_NOTES)))
    for message in _NOTES:
        print("    - %s" % message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
