"""E3 complete, synchronized physical windows adapted to the native V7 API.

The A3 source-image associations are reused after content/provenance checks.
Motion/contact/statistics are rebuilt in a new, pinned package.  No padding or
missing-frame imputation is admitted before the model supports attention masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from anysole.data.dataset import collate_windows
from anysole.data.pressure import normalize_raw, physical_tokens
from anysole.data.workspace_adapter import sha256_file
from anysole.types import FPS, JOINT_NAMES, JOINT_PARENTS
from anysole.utils.geometry import fk_pose6d_np, rot6d_to_rotmat_np

SCHEMA = "anysole.e3.data.v1"


def semantic_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify(path: Path, expected: str, what: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"E3 {what} content hash mismatch: {path}")


def load_e3_manifest(data_root: Path, *, verify_sources: bool = True) -> dict:
    root = Path(data_root)
    value = json.loads((root / "data_manifest.json").read_text())
    identity = value.pop("data_hash")
    if value.get("schema") != SCHEMA or semantic_hash(value) != identity:
        raise ValueError("E3 data manifest schema/identity mismatch")
    value["data_hash"] = identity
    for name, row in value["artifacts"].items():
        _verify(root / row["path"], row["sha256"], name)
    if verify_sources:
        for row in value["sources"]:
            _verify(Path(row["path"]), row["sha256"], "source")
    return value


def _feature(entry: dict, view: dict, recording_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(entry["path"])
    _verify(path, entry["sha256"], "RGB cache")
    provenance = entry["provenance"]
    expected = {"recording_id": recording_id, "camera_id": view["camera_id"],
                "source_index_sha256": semantic_hash(view["sources"])}
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError(f"E3 RGB source provenance mismatch: {recording_id}/cam{view['camera_id']}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    values, valid = obj["features"], obj["source_valid"]
    if values.dtype != torch.float32 or valid.dtype != torch.bool:
        raise ValueError("E3 RGB cache dtype mismatch")
    values, valid = values.numpy(), valid.numpy()
    if values.shape != (len(view["sources"]), 2051) or valid.shape != (len(values),):
        raise ValueError("E3 RGB cache/source shape mismatch")
    if not np.isfinite(values[valid]).all() or not np.any(values[valid]):
        raise ValueError("E3 RGB cache is nonfinite or empty")
    return values, valid


def complete_windows(record: dict, source_valid: dict[int, np.ndarray], *,
                     window_length: int, stride: int, cameras: Sequence[int]) -> list[tuple[int, int]]:
    n = len(record["target_frame_id"])
    motion_valid = np.asarray(record["motion"]["valid"], bool)
    # Every target velocity uses the previous physical frame as well.
    valid = motion_valid.copy()
    valid[1:] &= motion_valid[:-1]
    valid &= np.asarray(record["pressure"]["foot_valid"], bool).all(axis=1)
    for cam in cameras:
        view = next((v for v in record["views"] if v["camera_id"] == cam), None)
        if view is None:
            return []
        rows = np.asarray(view["source_row"], np.int64)
        keep = np.asarray(view["view_valid"], bool) & (rows >= 0)
        if np.any(rows[keep] >= len(source_valid[cam])):
            raise ValueError("E3 source row outside visual cache")
        keep[keep] &= source_valid[cam][rows[keep]]
        valid &= keep
    times = np.asarray(record["target_time_s"], np.float64)
    result = []
    for start in range(0, max(n - window_length + 1, 0), stride):
        end = start + window_length
        if valid[start:end].all() and np.allclose(np.diff(times[start:end]), 1 / FPS, atol=1e-8, rtol=0):
            result.append((start, end))
    return result


class E3GroupedDataset(Dataset):
    """One item is one physical window with fixed-order independent RGB views."""

    def __init__(self, data_root: Path, *, split: str = "train", window_length: int = 20,
                 stride: int = 20, cameras: Sequence[int] = (1, 2, 3, 4),
                 t_phys_v2: bool = False, verify_sources: bool = True):
        if split not in {"train", "val", "test"} or not 1 <= stride <= window_length:
            raise ValueError("invalid E3 split/window/stride")
        if not cameras or len(set(cameras)) != len(cameras) or set(cameras) - {1, 2, 3, 4}:
            raise ValueError("E3 cameras must be distinct members of 1..4")
        self.root, self.split = Path(data_root), split
        self.manifest = load_e3_manifest(self.root, verify_sources=verify_sources)
        self.stats = json.loads((self.root / "stats.json").read_text())
        self.records = [json.loads(x) for x in (self.root / "grouped_manifest.jsonl").read_text().splitlines()
                        if x.strip()]
        self.records = [r for r in self.records if r["eligible"] and split in r["split_roles"]]
        self.cache_index = json.loads((self.root / "cache_index.json").read_text())
        self.cameras, self.window_length, self.stride = tuple(cameras), window_length, stride
        self.t_phys_v2 = bool(t_phys_v2)
        self.windows, self._features, self._labels = [], {}, {}
        self.coverage = {"recordings": len(self.records), "candidate_windows": 0, "kept_windows": 0,
                         "excluded_incomplete_windows": 0, "tail_frames": 0}
        for ri, record in enumerate(self.records):
            valid = {}
            for cam in self.cameras:
                view = next(v for v in record["views"] if v["camera_id"] == cam)
                key = f"{record['recording_id']}/cam{cam}"
                self._features[key] = _feature(self.cache_index["entries"][key], view, record["recording_id"])
                valid[cam] = self._features[key][1]
            windows = complete_windows(record, valid, window_length=window_length, stride=stride,
                                       cameras=self.cameras)
            self.windows.extend((ri, start, end, self.cameras) for start, end in windows)
            n = len(record["target_frame_id"])
            candidate = max((n - window_length) // stride + 1, 0)
            self.coverage["candidate_windows"] += candidate
            self.coverage["tail_frames"] += n - ((candidate - 1) * stride + window_length if candidate else 0)
        self.coverage["kept_windows"] = len(self.windows)
        self.coverage["excluded_incomplete_windows"] = self.coverage["candidate_windows"] - len(self.windows)

    def __len__(self):
        return len(self.windows)

    def sample_key(self, index: int) -> str:
        ri, start, end, _ = self.windows[index]
        return f"{self.records[ri]['recording_id']}:{start}:{end}"

    def _label(self, ri: int) -> dict:
        if ri not in self._labels:
            row = self.records[ri]["e3_labels"]
            path = self.root / row["path"]
            _verify(path, row["sha256"], "motion/contact package")
            with np.load(path, allow_pickle=False) as archive:
                self._labels[ri] = {k: archive[k] for k in archive.files}
        return self._labels[ri]

    def __getitem__(self, index: int) -> dict:
        ri, start, end, cameras = self.windows[index]
        record, labels = self.records[ri], self._label(ri)
        def tensor(value):
            return torch.from_numpy(np.array(value, copy=True))
        physical = {name: tensor(labels[name][start:end]) for name in
                    ("T_raw", "pose_gt", "pose_gt_pos", "vel_gt", "kp_gt", "contact_gt")}
        physical["T_phys"] = tensor(physical_tokens(physical["T_raw"].numpy()[:, :48],
            physical["T_raw"].numpy()[:, 48:], t_phys_v2=self.t_phys_v2))
        physical["T_s2m"] = torch.zeros((end - start, 50), dtype=torch.float32)
        for name in ("offsets", "parents", "betas", "root_rot_init"):
            physical[name] = tensor(labels[name])
        anchor = max(start - 1, 0)
        physical["trans_anchor"] = tensor(labels["root"][anchor])
        physical["trans_gt"] = tensor(labels["root"][start:end] - labels["root"][anchor])
        floor = float(labels["floor_y"]) if self.split == "train" else float(self.stats["train_ground_y"])
        physical["floor_y"] = torch.tensor(floor, dtype=torch.float32)
        physical["session_id"] = record["session_id"]
        physical["frame_start"] = torch.tensor(start, dtype=torch.long)
        physical["V_feat"] = torch.zeros((end - start, 2051), dtype=torch.float32)
        views, source_ids, source_times, source_rows = [], [], [], []
        for cam in cameras:
            view = next(v for v in record["views"] if v["camera_id"] == cam)
            rows = np.asarray(view["source_row"][start:end], np.int64)
            feature, _ = self._features[f"{record['recording_id']}/cam{cam}"]
            views.append({**physical, "V_feat": tensor(feature[rows])})
            source_ids.append(view["source_frame_id"][start:end])
            source_times.append(view["source_time_s"][start:end])
            source_rows.append(rows)
        return {"views": views, "physical": physical,
                "meta": {"sample_key": self.sample_key(index), "recording_id": record["recording_id"],
                         "camera_id": torch.tensor(cameras, dtype=torch.long),
                         "target_frame_id": tensor(np.asarray(record["target_frame_id"][start:end], np.int64)),
                         "target_time_s": tensor(np.asarray(record["target_time_s"][start:end], np.float64)),
                         "source_frame_id": tensor(np.asarray(source_ids, np.int64)),
                         "source_time_s": tensor(np.asarray(source_times, np.float64)),
                         "source_feature_row": tensor(np.asarray(source_rows, np.int64)),
                         "valid": torch.ones((len(cameras), end - start), dtype=torch.bool)}}


def e3_grouped_collate(items: list[dict]) -> dict:
    if not items:
        raise ValueError("cannot collate empty E3 grouped batch")
    cameras = items[0]["meta"]["camera_id"]
    if any(not torch.equal(x["meta"]["camera_id"], cameras) for x in items):
        raise ValueError("E3 grouped batch has inconsistent camera identities")
    meta = {key: (torch.stack([x["meta"][key] for x in items]) if torch.is_tensor(value)
                  else [x["meta"][key] for x in items]) for key, value in items[0]["meta"].items()}
    return {"views": [collate_windows([x["views"][c] for x in items]) for c in range(len(cameras))],
            "physical": collate_windows([x["physical"] for x in items]), "meta": meta}


class FlatViewDataset(Dataset):
    """Ordinary V7 samples, with equal exposure of each declared camera."""
    def __init__(self, grouped: E3GroupedDataset):
        self.grouped = grouped

    def __len__(self):
        return len(self.grouped) * len(self.grouped.cameras)

    def __getitem__(self, index):
        return self.grouped[index // len(self.grouped.cameras)]["views"][index % len(self.grouped.cameras)]


def build_e3_data(*, source_manifest: Path, cache_index: Path, split_csv: Path,
                  dataset_root: Path, smpl_model: Path, output: Path) -> dict:
    """Explicit CPU preparation; refuses to overwrite any existing package."""
    from anysole.data.contact_adapter import _f6_soft
    from anysole.data.smpl_io import load_smpl
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E3 data package already exists: {output}")
    source_manifest, cache_index, split_csv = map(Path, (source_manifest, cache_index, split_csv))
    meta = json.loads(source_manifest.with_suffix(".meta.json").read_text())
    _verify(source_manifest, meta["manifest_sha256"], "A3 manifest")
    records = [json.loads(x) for x in source_manifest.read_text().splitlines() if x.strip()]
    cache = json.loads(cache_index.read_text())
    # Canonical split files may have literal CRLF escapes from prior migrations.
    split_text = split_csv.read_text().replace("\\r\\n", "\n")
    split_rows = list(csv.DictReader(split_text.splitlines()))
    splits = {role: {r[role].strip() for r in split_rows if r.get(role, "").strip()}
              for role in ("train", "val", "test")}
    if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("overlapping E3 split identities")
    all_ids = set.union(*splits.values())
    records = [r for r in records if r["session_id"] in all_ids]
    output.mkdir(parents=True)
    (output / "labels").mkdir()
    (output / "splits.csv").write_text(split_text)
    sources = {str(p.resolve()): sha256_file(p) for p in (source_manifest, cache_index, split_csv, smpl_model)}
    old_model = os.environ.get("ANYSOLE_SMPL_MODEL")
    os.environ["ANYSOLE_SMPL_MODEL"] = str(smpl_model)
    feature_entries, moments, pressures, offsets, ground_heights, train_ids = {}, [], [], [], [], []
    coverage = {role: {"recordings": 0, "physical_windows": 0, "rgb_windows": 0, "unique_frames": 0}
                for role in splits}
    source_quality_quarantines = []
    contact_package = {"schema": "anysole.e3.contact.v1", "method": "f6_soft_time_accel_state_smpl24_e3_v1",
                       "algorithm": "existing contact_adapter._f6_soft(enforce_airborne=True)",
                       "coordinates": "current SMPL24 FK metres; permute X,Z,Y only for existing Z-up context",
                       "floor_definition": "original contact_adapter._context all-joint/all-target-frame fifth-percentile height; separate from E3 geometry foot-center ground",
                       "producer_sha256": sha256_file(Path(__file__).with_name("contact_adapter.py")), "sessions": {}}
    changed_motion = 0
    try:
        for record in records:
            record["split_roles"] = [role for role, ids in splits.items() if record["session_id"] in ids]
            if not record["eligible"]:
                continue
            rid = record["recording_id"]
            mpath = dataset_root / record["motion"]["path"]
            ppath = dataset_root / record["pressure"]["path"]
            mhash = sha256_file(mpath)
            changed_motion += mhash != record["motion"]["sha256"]
            record["motion"]["sha256"] = mhash
            sources[str(mpath.resolve())], sources[str(ppath.resolve())] = mhash, sha256_file(ppath)
            _verify(ppath, record["pressure"]["sha256"], "A3 pressure")
            quality_keys = ("root_branch_repair_status", "joint_continuity_status")
            old_quality = {key: record["motion"][key] for key in quality_keys}
            # Motion hashes changed in the source refresh. Quality/coordinate
            # metadata must come from those same current bytes too.
            with np.load(mpath, allow_pickle=False) as archive:
                for key in ("coordinate_system", "model_units", "surface_model_sha256",
                            *quality_keys, "input_data_quality_status", "coordinate_handedness_status"):
                    record["motion"][key] = archive[key].tolist()
            current_quality = {key: record["motion"][key] for key in quality_keys}
            quarantine = any(value not in {"passed", "repaired"} for value in current_quality.values())
            motion = load_smpl(mpath, np.asarray(record["motion"]["time_s"], np.float64))
            valid = np.asarray(record["motion"]["valid"], bool) & motion["query_valid"]
            record["motion"]["valid"] = valid.tolist()
            if not valid.any():
                record["eligible"] = False
                record["exclusion_reasons"].append("current_smpl_has_no_supported_frames")
                continue
            pose, root, off, parents = motion["pose_6d"], motion["trans_m"], motion["offsets_m"], motion["parents"]
            kp = fk_pose6d_np(pose, root, off, parents).astype(np.float32)
            with np.load(ppath, allow_pickle=False) as archive:
                raw = np.concatenate([archive["left48"], archive["right48"]], axis=1).astype(np.float32)
            if raw.shape != (len(pose), 96):
                raise ValueError(f"pressure target axis mismatch: {rid}")
            context = {"pts": kp[..., [0, 2, 1]].astype(np.float64), "names": JOINT_NAMES,
                       # Preserve contact_adapter._context's original all-joint
                       # fifth-percentile floor. This contact-label reference
                       # is separate from the E3 foot-center geometry plane.
                       "floor": float(np.percentile(kp[:, :, 1], 5)),
                       "n": len(pose), "left48": raw[:, :48].astype(np.float64),
                       "right48": raw[:, 48:].astype(np.float64)}
            contact = _f6_soft(context, enforce_airborne=True)
            velocity = np.zeros_like(root)
            dt = np.diff(np.asarray(record["target_time_s"], np.float64))
            velocity[1:] = np.diff(root, axis=0) / dt[:, None]
            root_rot = rot6d_to_rotmat_np(pose.reshape(-1, 24, 6)[:, :1])[:, 0]
            root_rot_init = root_rot[np.flatnonzero(valid)[0]].astype(np.float32)
            position = np.einsum("ij,tpj->tpi", root_rot_init.T, kp[:, 1:] - kp[:, :1]).reshape(-1, 69)
            source_valid = {}
            for view in record["views"]:
                key = f"{rid}/cam{view['camera_id']}"
                entry = cache["entries"][key]
                _, source_valid[view["camera_id"]] = _feature(entry, view, rid)
                feature_entries[key] = entry
                sources[str(Path(entry["path"]).resolve())] = entry["sha256"]
            windows = complete_windows(record, source_valid, window_length=20, stride=20, cameras=(1, 2, 3, 4))
            frame_mask = np.zeros(len(pose), bool)
            for start, end in windows:
                frame_mask[start:end] = True
            floor_y = float(np.percentile(kp[valid][:, [10, 11], 1], 5))
            label_path = Path("labels") / (rid.replace("/", "_") + ".npz")
            np.savez_compressed(output / label_path, T_raw=normalize_raw(raw), pose_gt=pose,
                pose_gt_pos=position.astype(np.float32), root=root, vel_gt=velocity.astype(np.float32),
                kp_gt=kp, contact_gt=contact, offsets=off, parents=parents, betas=motion["betas"],
                root_rot_init=root_rot_init, floor_y=np.asarray(floor_y, np.float32))
            record["e3_labels"] = {"path": str(label_path), "sha256": sha256_file(output / label_path)}
            contact_package["sessions"][rid] = {**record["e3_labels"], "smpl_sha256": mhash,
                "pressure_sha256": record["pressure"]["sha256"], "target_axis_sha256": semantic_hash(record["target_time_s"])}
            if quarantine:
                record["eligible"] = False
                record["exclusion_reasons"].append("untrusted_current_motion_quality")
                row = {"recording_id": rid, "session_id": record["session_id"],
                       "split_roles": record["split_roles"],
                       "state": "REQUIRED_UNSCORED_SOURCE_QUALITY", "old_flags": old_quality,
                       "current_flags": current_quality,
                       "reason": "existing A3 source-quality policy requires passed/repaired motion; current source is retained but not scored",
                       "support_lost": {"physical_windows": len(windows), "rgb_windows": len(windows) * 4,
                                        "unique_frames": int(frame_mask.sum())}}
                record["source_quality_quarantine"] = row
                source_quality_quarantines.append(row)
            for role in record["split_roles"] if not quarantine else ():
                c = coverage[role]
                c["recordings"] += 1
                c["physical_windows"] += len(windows)
                c["rgb_windows"] += len(windows) * 4
                c["unique_frames"] += int(frame_mask.sum())
            if not quarantine and "train" in record["split_roles"] and frame_mask.any():
                moments.append(pose[frame_mask].astype(np.float64))
                pressures.append(normalize_raw(raw[frame_mask]).astype(np.float64))
                offsets.append(off)
                ground_heights.append(kp[frame_mask][:, [10, 11], 1].reshape(-1))
                train_ids.append(rid)
            print(f"E3 prepared {rid}: {len(windows)} complete four-camera windows", flush=True)
    finally:
        if old_model is None:
            os.environ.pop("ANYSOLE_SMPL_MODEL", None)
        else:
            os.environ["ANYSOLE_SMPL_MODEL"] = old_model
    if not moments:
        raise ValueError("no complete E3 train support")
    pose_values, pressure_values = np.concatenate(moments), np.concatenate(pressures)
    stats = {"schema": "anysole.e3.stats.v1", "fit_split": "train", "recordings": train_ids,
             "weighting": "unique physical frames in complete four-camera T20/stride20 windows",
             "pose_count": len(pose_values), "pose_mean": pose_values.mean(axis=0).tolist(),
             "pose_std": np.maximum(pose_values.std(axis=0), .01).tolist(),
             "pressure_count": len(pressure_values), "pressure_cell_mean": np.maximum(pressure_values.mean(axis=0), 1e-4).tolist(),
             "pressure_units": "clip(raw_sensor_0_1023)/1023; same as T_raw and sigmoid readout",
             "mean_offsets": np.mean(offsets, axis=0).tolist(), "parents": list(JOINT_PARENTS),
             "shape_weighting": "equal weight per supported train recording; no evaluation shape input",
             "support_joint_ids": [10, 11], "support_joint_names": [JOINT_NAMES[10], JOINT_NAMES[11]],
             "support_offsets": [[0., 0., 0.], [0., 0., 0.]],
             "train_ground_y": float(np.percentile(np.concatenate(ground_heights), 5)),
             "ground_definition": "SMPL foot-joint center Y fifth percentile, a proxy not a physical sole plane; train recording floor in training, frozen global train floor for evaluation"}
    _json(output / "stats.json", stats)
    _json(output / "contact_package.json", contact_package)
    (output / "grouped_manifest.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    _json(output / "cache_index.json", {"schema": "anysole.e3.cache.v1", "entries": feature_entries})
    artifacts = {name: {"path": name, "sha256": sha256_file(output / name)} for name in
                 ("stats.json", "contact_package.json", "grouped_manifest.jsonl", "cache_index.json", "splits.csv")}
    for record in records:
        if "e3_labels" in record:
            artifacts[record["e3_labels"]["path"]] = record["e3_labels"]
    for role, count in coverage.items():
        count["declared_recordings"] = len(splits[role])
        count["manifest_recordings"] = sum(role in r["split_roles"] for r in records)
        count["required_unscored_source_quality"] = [r["session_id"] for r in source_quality_quarantines if role in r["split_roles"]]
        count["source_quality_support_lost"] = {key: sum(r["support_lost"][key] for r in source_quality_quarantines if role in r["split_roles"])
                                                for key in ("physical_windows", "rgb_windows", "unique_frames")}
    result = {"schema": SCHEMA, "source_a3_manifest": str(source_manifest),
              "split_counts": {k: len(v) for k, v in splits.items()}, "coverage": coverage,
              "declared_split_ids": {role: sorted(ids) for role, ids in splits.items()},
              "missing_source_records": [row for row in meta.get("missing_modalities", []) if row["session_id"] in all_ids],
              "required_unscored_source_quality": source_quality_quarantines,
              "rebuilt_motion_count": changed_motion, "artifacts": artifacts,
              "sources": [{"path": p, "sha256": h} for p, h in sorted(sources.items())],
              "contact_method": contact_package["method"], "complete_window_policy": "all20 target motion/pressure and all4 real RGB valid; no padding; drop incomplete/tail",
              "occlusion_readiness": "clean verified; occ010/occ030 same-image features+detections require explicit extraction before evaluation"}
    result["data_hash"] = semantic_hash(result)
    _json(output / "data_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = build_e3_data(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("data_hash", "coverage", "split_counts", "rebuilt_motion_count")}, indent=2))


if __name__ == "__main__":
    main()
