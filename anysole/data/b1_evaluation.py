"""Explicit, read-only B1 final-test derivation on native RGB timestamps.

The frozen training manifest, cache index and statistics are never overwritten.
Only cache metadata is derived; feature tensors retain their original bytes.
Motion uses the B1 cam1 wall-clock mapping for every target camera. Pressure
is linearly sampled from the inherited shared 40 Hz grid with strict support.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from anysole.data.dataset import GroupedViewDataset
from anysole.data.rgb_smpl.alignment import bracket_times
from anysole.data.workspace_adapter import (
    VIEW_CACHE_SCHEMA, VIEW_SCHEMA, load_grouped_manifest, match_rgb_sources,
    read_rgb_sources, semantic_hash, sha256_file,
)

EVALUATION_SCHEMA = "anysole.b1.native-rgb-evaluation.v1"
PRESSURE_GAP_S = .02500001


def verify_test_authorization(path: Path) -> dict:
    """Require a receipt bound to the actual, explicitly authorized plan."""
    receipt = json.loads(Path(path).read_text())
    plan_path = Path(receipt["evaluation_plan_path"])
    if sha256_file(plan_path) != receipt["evaluation_plan_sha256"]:
        raise ValueError("frozen evaluation plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("test_access_authorized") is not True:
        raise ValueError("final-test access has not been authorized")
    return {"evaluation_plan_path": str(plan_path.resolve()),
            "evaluation_plan_sha256": receipt["evaluation_plan_sha256"]}


def pressure_brackets(source_times, target_times, source_foot_valid):
    """Both endpoints must be real and valid; exact samples use their own row."""
    source = np.asarray(source_times, np.float64)
    bracket = bracket_times(source, np.asarray(target_times, np.float64))
    feet = np.asarray(source_foot_valid, bool)
    if feet.shape != (len(source), 2):
        raise ValueError("pressure foot masks differ from the frozen source axis")
    valid = (bracket.valid[:, None] & feet[bracket.lower] & feet[bracket.upper] &
             ((source[bracket.upper] - source[bracket.lower]) <= PRESSURE_GAP_S)[:, None])
    return bracket, valid


def derive_native_record(parent, sources_by_camera, motion_source_times, *, camera=None,
                         teacher=False, rgb_tolerance_ms=25., source_gap_s=.05,
                         timestamp_ties="error"):
    """Pure derivation; callers control I/O and final-test authorization."""
    if teacher == (camera is not None):
        raise ValueError("select either one student camera or the teacher")
    if not teacher and camera not in (1, 2, 3, 4):
        raise ValueError("invalid student camera")
    cameras, anchor = ((1, 2, 4), 1) if teacher else ((camera,), camera)
    sources = sources_by_camera[anchor]
    times = np.asarray([row["time_s"] for row in sources], np.float64)
    frame_ids = np.asarray([row["frame_id"] for row in sources], np.int64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("native RGB timestamps must be finite and nondecreasing")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("native RGB frame numbers must increase")
    if timestamp_ties not in {"error", "smallest_frame"}:
        raise ValueError("unsupported native timestamp tie policy")
    if timestamp_ties == "error" and np.any(np.diff(times) == 0):
        raise ValueError("native RGB timestamps must be unique")
    native_rows = np.flatnonzero(np.r_[True, np.diff(times) > 0])
    excluded_rows = np.flatnonzero(np.r_[False, np.diff(times) == 0])
    times, frame_ids = times[native_rows], frame_ids[native_rows]
    record = copy.deepcopy(parent)
    record["target_frame_id"] = native_rows.tolist()
    record["target_time_s"] = times.tolist()
    record["native_frame_index"] = native_rows.tolist()
    record["native_source_frame_id"] = frame_ids.tolist()
    record["evaluation"] = {"schema_version": EVALUATION_SCHEMA,
        "unit": "teacher" if teacher else f"cam{camera}", "target_camera": anchor,
        "input_cameras": list(cameras), "coordinate_frame": "world",
        "target_axis": "distinct_cached_native_RGB_timestamps_without_label_filtering",
        "timestamp_tie_rule": timestamp_ties,
        "raw_source_count": len(sources), "selected_source_rows": native_rows.tolist(),
        "excluded_duplicate_source_rows": excluded_rows.tolist(),
        "parent_record_sha256": semantic_hash(parent)}
    record["available_allowed_view_count"] = np.asarray([
        match_rgb_sources(times, sources_by_camera[cam], rgb_tolerance_ms)["view_valid"]
        for cam in (1, 2, 4)], bool).sum(0).tolist()
    record["availability_definition"] = "allowed_cam1_2_4_RGB_filename_timestamp_match; not_physical_occlusion"
    clock = parent["clock_mapping"]
    if clock.get("anchor_camera") != 1:
        raise ValueError("B1 final evaluation requires the frozen cam1 clock mapping")
    motion_times = (float(clock["mocap_start_s"]) +
                    (times - float(clock["wall_start_s"])) * float(clock["slope"]))
    st = np.asarray(motion_source_times, np.float64)
    bracket = bracket_times(st, motion_times)
    mv = (bracket.valid & (times >= float(clock["wall_start_s"])) &
          (times <= float(clock["wall_end_s"])) &
          ((st[bracket.upper] - st[bracket.lower]) <= float(parent["motion"]["source_gap_s"])))
    record["motion"].update(time_s=motion_times.tolist(), lower=bracket.lower.tolist(),
        upper=bracket.upper.tolist(), alpha=bracket.alpha.tolist(), valid=mv.tolist())
    pt = np.asarray(parent["target_time_s"], np.float64)
    pb, feet = pressure_brackets(pt, times, parent["pressure"]["foot_valid"])
    record["pressure"].update(row=list(range(len(times))), foot_valid=feet.tolist(),
        fake=(np.asarray(parent["pressure"]["fake"], bool)[pb.lower] |
              np.asarray(parent["pressure"]["fake"], bool)[pb.upper]).tolist(),
        native_sampling={"method": "linear_strict_brackets", "gap_max_s": PRESSURE_GAP_S,
            "source_time_s": pt.tolist(), "source_foot_valid": parent["pressure"]["foot_valid"],
            "lower": pb.lower.tolist(), "upper": pb.upper.tolist(), "alpha": pb.alpha.tolist(),
            "exact": pb.exact.tolist(), "valid": pb.valid.tolist(),
            "parent_selected_camera": parent["pressure"]["selected_camera"],
            "raw_pressure_csv_available": False})
    adjacent = ((np.diff(frame_ids) == 1) & (np.diff(times) > 0) &
                (np.diff(times) <= source_gap_s + 1e-9))
    record["motion_edges"] = (mv[:-1] & mv[1:] & adjacent).tolist()
    # Do not bridge an invalid original pressure sample between native targets.
    source_feet = np.asarray(parent["pressure"]["foot_valid"], bool)
    pe = feet[:-1] & feet[1:] & adjacent[:, None]
    for i in range(len(pe)):
        lo, hi = int(pb.lower[i]), int(pb.upper[i + 1])
        if hi >= lo:
            pe[i] &= source_feet[lo:hi + 1].all(axis=0)
    record["pressure_foot_edges"] = pe.tolist()
    old_views = {v["camera_id"]: v for v in parent["views"]}
    record["views"] = []
    for cam in cameras:
        source = sources_by_camera[cam]
        match = match_rgb_sources(times, source, rgb_tolerance_ms)
        # The anchor is a true native axis, including any frame excluded by labels.
        if cam == anchor:
            match.update(source_row=native_rows.tolist(), source_frame_id=frame_ids.tolist(),
                source_time_s=times.tolist(), time_residual_ms=[0.] * len(times),
                view_valid=[True] * len(times), reuse=[False] * len(times))
        vv, ids = np.asarray(match["view_valid"], bool), np.asarray(match["source_frame_id"])
        ts = np.asarray(match["source_time_s"], np.float64)
        ve = (adjacent & vv[:-1] & vv[1:] & (np.diff(ids) == 1) &
              (np.diff(ts) > 0) & (np.diff(ts) <= source_gap_s + 1e-9))
        directory = (old_views[cam]["directory"] if cam in old_views else
            str(Path(parent["date"]) / "RGB" / parent["method"] / parent["session_id"] / f"camera_{cam}"))
        record["views"].append({"camera_id": cam, "directory": directory, "sources": source,
                                **match, "view_edges": ve.tolist()})
    return record


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_evaluation_data(parent_manifest: Path, dataset_root: Path, feature_cache_root: Path,
                            backup_index: Path, output_dir: Path, authorization: Path) -> dict:
    """Materialize five test-only manifests and a cache index once for all runs.

    RGB filenames are listed to verify source identity; RGB bytes are not read.
    The old cam3 cache entry is accepted only when its original source index,
    method, frontend hashes and parent manifest lineage all agree.
    """
    authorization_info = verify_test_authorization(authorization)
    plan = json.loads(Path(authorization_info["evaluation_plan_path"]).read_text())
    timestamp_ties = "smallest_frame" if plan.get("protocol", {}).get("native_timestamp_ties") else "error"
    root, cache_root, output = Path(dataset_root), Path(feature_cache_root), Path(output_dir)
    records, meta = load_grouped_manifest(Path(parent_manifest))
    if meta.get("protocol", {}).get("b1_version") != "anysole.b1.v1":
        raise ValueError("final-test derivation requires the frozen B1 parent")
    frozen_index_path = cache_root / "index.json"
    frozen_index, old_index = (json.loads(p.read_text()) for p in (frozen_index_path, Path(backup_index)))
    if not frozen_index.get("complete") or frozen_index.get("manifest_sha256") != meta["manifest_sha256"]:
        raise ValueError("frozen B1 feature cache does not bind to the parent manifest")
    if old_index.get("schema_version") != VIEW_CACHE_SCHEMA or not old_index.get("complete"):
        raise ValueError("original four-camera index is incomplete or incompatible")
    if frozen_index.get("frontend_sha256") != old_index.get("frontend_sha256"):
        raise ValueError("frozen and original cache frontend identities differ")
    old_manifest = Path(backup_index).parent / "anysole_manifest.jsonl"
    old_records, old_meta = load_grouped_manifest(old_manifest)
    if old_index.get("manifest_sha256") != old_meta["manifest_sha256"]:
        raise ValueError("original cache index does not bind to its saved manifest")
    old_by_id = {r["recording_id"]: r for r in old_records}
    derived = {name: [] for name in ("cam1", "cam2", "cam3", "cam4", "teacher")}
    entries, audit, exclusions, timestamp_ledger = {}, [], [], []
    for parent in records:
        if "test" not in parent["split_roles"]:
            continue
        if not parent["eligible"]:
            exclusions.append({"recording_id": parent["recording_id"], "reasons": parent["exclusion_reasons"]})
            continue
        rid = parent["recording_id"]
        old = old_by_id[rid]
        sources_by_cam = {}
        for cam in (1, 2, 3, 4):
            view = next(v for v in (old if cam == 3 else parent)["views"] if v["camera_id"] == cam)
            source = read_rgb_sources(root / view["directory"], cam)
            if semantic_hash(source) != semantic_hash(view["sources"]):
                raise ValueError(f"native RGB source index changed: {rid}/cam{cam}")
            key = f"{rid}/cam{cam}"
            entry = (old_index if cam == 3 else frozen_index)["entries"][key]
            if cam != 3 and old_index["entries"].get(key) != entry:
                raise ValueError(f"frozen cache differs from its original four-camera entry: {key}")
            provenance = entry["provenance"]
            expected = {"recording_id": rid, "camera_id": cam, "method": parent["method"],
                        "source_index_sha256": semantic_hash(source)}
            if any(provenance.get(k) != v for k, v in expected.items()):
                raise ValueError(f"cache/source lineage changed: {key}")
            reference = frozen_index["entries"][f"{rid}/cam1"]["provenance"]
            for field in ("asset_hashes", "frontend_sha256", "extractor_sha256", "crop_source_sha256",
                          "frontend_versions", "hrnet_source_commit", "hrnet_source_files", "numeric_backend"):
                if provenance.get(field) != reference.get(field):
                    raise ValueError(f"cache frontend lineage changed: {key}/{field}")
            if not (cache_root / entry["path"]).is_file():
                raise FileNotFoundError(cache_root / entry["path"])
            entries[key] = entry
            sources_by_cam[cam] = source
            times = np.asarray([s["time_s"] for s in source], np.float64)
            duplicate_rows = set(np.flatnonzero(np.r_[False, np.diff(times) == 0]).tolist())
            timestamp_ledger.append({"recording_id": rid, "camera_id": cam,
                "raw_source_count": len(source), "unique_timestamp_count": len(source)-len(duplicate_rows),
                "excluded_duplicate_count": len(duplicate_rows),
                "sources": [{"source_row": i, **s, "selected": i not in duplicate_rows,
                    "exclusion_reason": "duplicate_timestamp_larger_numeric_frame" if i in duplicate_rows else None}
                    for i, s in enumerate(source)]})
        mp = root / parent["motion"]["path"]
        if sha256_file(mp) != parent["motion"]["sha256"]:
            raise ValueError(f"motion source changed: {rid}")
        with np.load(mp, allow_pickle=False) as archive:
            motion_source_times = archive["source_frame_times_s"]
        for name in derived:
            anchor = 1 if name == "teacher" else int(name[-1])
            if not sources_by_cam[anchor]:
                exclusions.append({"recording_id": rid, "unit": name, "reasons": ["no_native_RGB_sources"]})
                continue
            record = derive_native_record(parent, sources_by_cam, motion_source_times,
                camera=None if name == "teacher" else anchor, teacher=name == "teacher",
                rgb_tolerance_ms=meta["protocol"]["rgb_tolerance_ms"],
                source_gap_s=meta["protocol"]["source_gap_ms"] / 1000., timestamp_ties=timestamp_ties)
            derived[name].append(record)
            audit.append({"recording_id": rid, "unit": name, "native_frames": len(record["target_frame_id"]),
                "motion_valid": int(np.asarray(record["motion"]["valid"]).sum()),
                "pressure_valid_foot_frames": np.asarray(record["pressure"]["foot_valid"]).sum(0).tolist(),
                "native_times_sha256": semantic_hash(record["target_time_s"])})
    # Check paths before any write, including a repeated invocation.
    targets = [output / "cache_index.json", output / "coverage.json", output / "native_timestamp_ledger.json"]
    for name in derived:
        targets.extend((output / f"{name}.jsonl", output / f"{name}.meta.json"))
    if any(p.exists() for p in targets):
        raise FileExistsError("derived evaluation outputs already exist; preserve the prior derivation")
    output.mkdir(parents=True, exist_ok=True)
    lineage = {**authorization_info, "parent_manifest_path": str(Path(parent_manifest).resolve()),
        "parent_manifest_sha256": meta["manifest_sha256"],
        "parent_manifest_meta_sha256": sha256_file(Path(parent_manifest).with_suffix(".meta.json")),
        "parent_cache_index_sha256": sha256_file(frozen_index_path),
        "original_four_camera_index_sha256": sha256_file(Path(backup_index)),
        "original_four_camera_manifest_sha256": old_meta["manifest_sha256"],
        "producer_sha256": sha256_file(Path(__file__))}
    manifests = {}
    for name, rows in derived.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for r in rows))
        derived_meta = {"schema_version": VIEW_SCHEMA, "manifest_sha256": sha256_file(path),
            "evaluation_schema": EVALUATION_SCHEMA, "unit": name, "frozen": True,
            "candidate_split_counts": {"train": 0, "val": 0, "test": len(rows)},
            "eligible_split_counts": {"train": 0, "val": 0, "test": len(rows)},
            "original_split_counts": meta["original_split_counts"],
            "protocol": {**meta["protocol"], "evaluation_only": True, "native_rgb_grid": True,
                "window_length": 20, "stride": 10, "cover_tail": True,
                "native_timestamp_ties": timestamp_ties,
                "input_cameras": [1, 2, 4] if name == "teacher" else [int(name[-1])]},
            "lineage": lineage}
        _json(path.with_suffix(".meta.json"), derived_meta)
        manifests[name] = str(path.resolve())
    cache_index = {"schema_version": VIEW_CACHE_SCHEMA, "evaluation_schema": EVALUATION_SCHEMA,
                   "complete": True, "entries": entries, "lineage": lineage}
    _json(output / "cache_index.json", cache_index)
    _json(output / "native_timestamp_ledger.json", {"units": timestamp_ledger, "policy": timestamp_ties})
    missing = [r for r in meta.get("missing_modalities", []) if "test" in r["split_roles"]]
    result = {"schema_version": EVALUATION_SCHEMA, "manifests": manifests,
        "cache_index": str((output / "cache_index.json").resolve()), "lineage": lineage,
        "requested_test_recordings": meta["original_split_counts"]["test"],
        "eligible_test_recordings": meta["eligible_split_counts"]["test"],
        "missing_modalities": missing, "exclusions": exclusions, "native_units": audit,
        "raw_RGB_source_frames": sum(x["raw_source_count"] for x in timestamp_ledger),
        "unique_RGB_timestamps": sum(x["unique_timestamp_count"] for x in timestamp_ledger),
        "excluded_duplicate_timestamp_frames": sum(x["excluded_duplicate_count"] for x in timestamp_ledger),
        "native_timestamp_ledger": str((output / "native_timestamp_ledger.json").resolve()),
        "feature_tensors_recomputed": False, "RGB_bytes_read": False,
        "pressure_sampling": "linear_normalized_reading; strict_real_brackets; no_extrapolation",
        "pressure_gap_max_s": PRESSURE_GAP_S}
    _json(output / "coverage.json", result)
    return result


def load_native_labels(record, dataset_root, smpl_model_path):
    """Use existing SMPL interpolation/FK and frozen normalized pressure values."""
    from anysole.data.smpl_io import load_smpl
    from anysole.utils.geometry import fk_pose6d_np
    root, n = Path(dataset_root), len(record["target_frame_id"])
    for field in ("motion", "pressure"):
        if sha256_file(root / record[field]["path"]) != record[field]["sha256"]:
            raise ValueError(f"{field} source changed for {record['recording_id']}")
    mv = np.asarray(record["motion"]["valid"], bool)
    # Read one valid source pose to recover shape/rest offsets even for a
    # fully unsupported native axis; no unsupported row receives that pose.
    query = np.asarray(record["motion"]["time_s"], np.float64)[mv]
    if not len(query):
        with np.load(root / record["motion"]["path"], allow_pickle=False) as archive:
            query = archive["source_frame_times_s"][:1]
    motion = load_smpl(root / record["motion"]["path"], query,
                       model_path=smpl_model_path, allow_extrapolation=False)
    pose, anchor = np.zeros((n, 144), np.float32), np.zeros((n, 3), np.float32)
    kp = np.zeros((n, 24, 3), np.float32)
    if mv.any():
        pose[mv], anchor[mv] = motion["pose_6d"], motion["trans_m"]
        kp[mv] = fk_pose6d_np(motion["pose_6d"], motion["trans_m"], motion["offsets_m"], motion["parents"])
    sampling = record["pressure"]["native_sampling"]
    with np.load(root / record["pressure"]["path"], allow_pickle=False) as archive:
        pressure_source = np.concatenate([archive["left48"], archive["right48"]], axis=1).astype(np.float32)
    source_valid = np.repeat(np.asarray(sampling["source_foot_valid"], bool), 48, axis=1)
    if pressure_source.shape != source_valid.shape or not np.isfinite(pressure_source[source_valid]).all():
        raise ValueError("frozen source pressure labels are malformed/nonfinite")
    pressure_source = np.where(source_valid, pressure_source, 0)
    normalizer = record["pressure"]["normalizer"]
    if normalizer == "clip_0_1023_div_1023":
        pressure_source = np.clip(pressure_source, 0, 1023) / 1023
    elif normalizer != "identity":
        raise ValueError("unknown frozen pressure normalizer")
    lo, hi = np.asarray(sampling["lower"], int), np.asarray(sampling["upper"], int)
    alpha = np.asarray(sampling["alpha"], np.float64)[:, None]
    pressure = ((1 - alpha) * pressure_source[lo] + alpha * pressure_source[hi]).astype(np.float32)
    pressure = np.where(np.repeat(np.asarray(record["pressure"]["foot_valid"], bool), 48, axis=1), pressure, 0)
    velocity = np.zeros((n, 3), np.float32)
    edges = np.asarray(record["motion_edges"], bool)
    velocity[1:][edges] = (anchor[1:][edges] - anchor[:-1][edges]) / np.diff(np.asarray(record["target_time_s"]))[edges, None]
    return {"pose6d": pose, "root": anchor, "velocity": velocity, "pressure": pressure,
            "betas": motion["betas"], "offsets": motion["offsets_m"], "parents": motion["parents"], "kp_gt": kp}


class NativeRGBEvaluationDataset(GroupedViewDataset):
    """Grouped schema on a derived final-test axis; only frozen features load."""
    def __init__(self, manifest, dataset_root, feature_cache_root, smpl_model_path, *,
                 cache_index, authorization, window_length=20, stride=10):
        auth = verify_test_authorization(authorization)
        self.manifest_path = Path(manifest)
        self.records, self.manifest_meta = load_grouped_manifest(self.manifest_path)
        if (self.manifest_meta.get("evaluation_schema") != EVALUATION_SCHEMA or
                self.manifest_meta["lineage"]["evaluation_plan_sha256"] != auth["evaluation_plan_sha256"]):
            raise ValueError("derived final-test manifest/plan identity mismatch")
        if window_length != 20 or stride != 10:
            raise ValueError("B1 final-test windows are frozen at T20/stride10/tail")
        if any(r["split_roles"] != ["test"] for r in self.records):
            raise ValueError("derived B1 evaluator accepts test-only records")
        self.split, self.b1 = "test", True
        self.dataset_root, self.cache_root = Path(dataset_root), Path(feature_cache_root)
        self.smpl_model_path = Path(smpl_model_path)
        self.window_length, self.stride = window_length, stride
        self.cameras = tuple(self.manifest_meta["protocol"]["input_cameras"])
        self.cache_index = json.loads(Path(cache_index).read_text())
        if (self.cache_index.get("evaluation_schema") != EVALUATION_SCHEMA or
                self.cache_index.get("schema_version") != VIEW_CACHE_SCHEMA or not self.cache_index.get("complete") or
                self.cache_index["lineage"] != self.manifest_meta["lineage"]):
            raise ValueError("derived cache index/manifest lineage mismatch")
        self._features, self._labels, self._camera_geometry = {}, {}, {}
        self.windows = []
        self.coverage = {"recordings": len(self.records), "candidate_windows": 0,
                         "kept_windows": 0, "no_valid_rgb_windows": 0, "tail_rows_without_window": 0}
        for ri, record in enumerate(self.records):
            n = len(record["target_frame_id"])
            starts = list(range(0, max(n - window_length + 1, 0), stride))
            if n and (not starts or starts[-1] != max(n - window_length, 0)):
                starts.append(max(n - window_length, 0))
            views = [(v, self._load_features(record, v)[1]) for v in record["views"]]
            for start in starts:
                end = min(start + window_length, n)
                self.coverage["candidate_windows"] += 1
                active = tuple(v["camera_id"] for v, valid in views if valid[start:end].any())
                if active:
                    self.windows.append((ri, start, end, active))
                    self.coverage["kept_windows"] += 1
                else:
                    self.coverage["no_valid_rgb_windows"] += 1

    def load_labels(self, recording_index):
        if recording_index not in self._labels:
            self._labels[recording_index] = load_native_labels(self.records[recording_index],
                self.dataset_root, self.smpl_model_path)
        return self._labels[recording_index]

    def __getitem__(self, index):
        ri, start, end, _ = self.windows[index]
        self.load_labels(ri)
        item = super().__getitem__(index)
        import torch
        item["meta"]["native_frame_index"] = torch.tensor(self.records[ri]["native_frame_index"][start:end], dtype=torch.long)
        item["meta"]["coordinate_frame"] = "world"
        item["meta"]["evaluation_unit"] = self.manifest_meta["unit"]
        return item

    @property
    def inference_units(self):
        return [{"recording_index": ri, "recording_id": r["recording_id"],
                 "unit": r["evaluation"]["unit"], "camera_id": r["evaluation"]["target_camera"],
                 "n_frames": len(r["target_frame_id"]), "frame_id": r["target_frame_id"],
                 "time_s": r["target_time_s"], "native_source_frame_id": r["native_source_frame_id"]}
                for ri, r in enumerate(self.records)]
