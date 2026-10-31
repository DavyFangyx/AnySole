"""Scientific data boundaries: complete support, physical normalization, hashes."""
import json

import numpy as np
import pytest
import torch

from anysole.data.e3_grouped import (
    E3GroupedDataset, FlatViewDataset, SCHEMA, complete_windows,
    e3_grouped_collate, load_e3_manifest, semantic_hash,
)
from anysole.data.workspace_adapter import sha256_file
from anysole.types import JOINT_PARENTS


def test_complete_support_never_zero_pads_or_shortens_time():
    n = 61
    record = {"target_frame_id": list(range(n)), "target_time_s": (np.arange(n) / 40).tolist(),
              "motion": {"valid": [True] * n}, "pressure": {"foot_valid": [[True, True]] * n},
              "views": [{"camera_id": c, "source_row": list(range(n)), "view_valid": [True] * n}
                        for c in range(1, 5)]}
    valid = {c: np.ones(n, bool) for c in range(1, 5)}
    assert complete_windows(record, valid, window_length=20, stride=20, cameras=(1, 2, 3, 4)) == [(0, 20), (20, 40), (40, 60)]
    valid[4][20] = False
    assert complete_windows(record, valid, window_length=20, stride=20, cameras=(1, 2, 3, 4)) == [(0, 20), (40, 60)]
    record["motion"]["valid"][39] = False
    # Frame40's velocity depends on unsupported frame39 too.
    assert complete_windows(record, valid, window_length=20, stride=20, cameras=(1, 2, 3, 4)) == [(0, 20)]


@pytest.fixture
def package(tmp_path):
    n = 40
    record = {"session_id": "demo", "recording_id": "date/demo", "eligible": True,
              "split_roles": ["train", "val"], "target_frame_id": list(range(n)),
              "target_time_s": (np.arange(n) / 40).tolist(),
              "motion": {"valid": [True] * n}, "pressure": {"foot_valid": [[True, True]] * n}, "views": []}
    entries = {}
    for c in range(1, 5):
        path = tmp_path / f"cam{c}.pt"
        torch.save({"features": torch.full((n, 2051), float(c)), "source_valid": torch.ones(n, dtype=torch.bool)}, path)
        view = {"camera_id": c, "sources": [{"file": f"{i}.jpg"} for i in range(n)],
                "source_row": list(range(n)), "view_valid": [True] * n,
                "source_frame_id": list(range(n)), "source_time_s": (np.arange(n) / 40).tolist()}
        record["views"].append(view)
        entries[f"date/demo/cam{c}"] = {"path": str(path), "sha256": sha256_file(path),
            "provenance": {"recording_id": "date/demo", "camera_id": c,
                           "source_index_sha256": semantic_hash(view["sources"])}}
    root = np.arange(n * 3).reshape(n, 3).astype(np.float32) / 100
    velocity = np.zeros_like(root)
    velocity[1:] = np.diff(root, axis=0) * 40
    labels = {"T_raw": np.full((n, 96), .2, np.float32), "pose_gt": np.zeros((n, 144), np.float32),
              "pose_gt_pos": np.zeros((n, 69), np.float32), "vel_gt": velocity,
              "kp_gt": np.zeros((n, 24, 3), np.float32), "contact_gt": np.ones((n, 2), np.float32),
              "root": root, "offsets": np.zeros((24, 3), np.float32), "parents": np.array(JOINT_PARENTS),
              "betas": np.zeros(10, np.float32), "root_rot_init": np.eye(3, dtype=np.float32),
              "floor_y": np.array(.123, np.float32)}
    np.savez(tmp_path / "labels.npz", **labels)
    record["e3_labels"] = {"path": "labels.npz", "sha256": sha256_file(tmp_path / "labels.npz")}
    (tmp_path / "stats.json").write_text(json.dumps({"train_ground_y": .456}))
    (tmp_path / "grouped_manifest.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "cache_index.json").write_text(json.dumps({"entries": entries}))
    raw = tmp_path / "raw_source.bin"
    raw.write_bytes(b"immutable raw source")
    manifest = {"schema": SCHEMA, "sources": [{"path": str(raw), "sha256": sha256_file(raw)}],
                "artifacts": {name: {"path": name, "sha256": sha256_file(tmp_path / name)}
                              for name in ("stats.json", "grouped_manifest.jsonl", "cache_index.json", "labels.npz")}}
    manifest["data_hash"] = semantic_hash(manifest)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_four_views_share_one_physical_truth_and_native_anchor(package):
    dataset = E3GroupedDataset(package)
    assert len(dataset) == 2
    batch = e3_grouped_collate([dataset[0], dataset[1]])
    assert len(batch["views"]) == 4
    assert batch["physical"]["T_raw"].shape == (2, 20, 96)
    for cam, view in enumerate(batch["views"], 1):
        assert torch.all(view["V_feat"] == cam)
        assert torch.equal(view["T_raw"], batch["physical"]["T_raw"])
    physical = batch["physical"]
    assert torch.allclose(physical["trans_gt"], physical["vel_gt"].cumsum(1) / 40)
    assert torch.allclose(physical["trans_anchor"][1], torch.tensor([.57, .58, .59]))
    assert batch["meta"]["source_time_s"].dtype == torch.float64
    assert batch["meta"]["camera_id"].tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]
    flat = FlatViewDataset(dataset)
    assert len(flat) == 8
    assert [float(flat[i]["V_feat"][0, 0]) for i in range(8)] == [1, 2, 3, 4] * 2
    # Eval ground never fits the held-out recording geometry.
    assert float(E3GroupedDataset(package, split="val")[0]["physical"]["floor_y"]) == pytest.approx(.456)


def test_original_source_mutation_is_a_hard_failure(package):
    (package / "raw_source.bin").write_bytes(b"changed raw source")
    with pytest.raises(ValueError, match="source content hash mismatch"):
        E3GroupedDataset(package)


def test_label_contact_package_and_manifest_are_hash_pinned(package):
    (package / "labels.npz").write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="labels.npz content hash mismatch"):
        load_e3_manifest(package)


def test_rgb_provenance_cannot_reuse_another_image_source(package):
    value = json.loads((package / "cache_index.json").read_text())
    value["entries"]["date/demo/cam1"]["provenance"]["source_index_sha256"] = "wrong source"
    (package / "cache_index.json").write_text(json.dumps(value))
    manifest = json.loads((package / "data_manifest.json").read_text())
    manifest["artifacts"]["cache_index.json"]["sha256"] = sha256_file(package / "cache_index.json")
    manifest.pop("data_hash")
    manifest["data_hash"] = semantic_hash(manifest)
    (package / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="RGB source provenance mismatch"):
        E3GroupedDataset(package)
