"""Regression tests for pipeline wiring, with synthetic data and no checkpoints."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from AnysoleWorkspace.tool.build_manifest import pressure_flags
from AnysoleWorkspace.tool.build_shared import recording_pressure
from anysole.data.pressure import physical_tokens
from anysole.models.common.embeddings import SharedEmbeddings
from anysole.models.common.encoders import ModalEncoders
from anysole.train import load_config
from anysole.types import JOINT_PARENTS, JOINT_NAMES, POSE_DIM, V_FEAT_DIM, variant_from_config
from anysole.utils import eval_protocol as protocol
from anysole.utils.geometry import fk_pose6d
from anysole.utils.session_motion import (
    PIPELINE_VERSION, pack_motion_windows, prepare_session_sequence, write_motion_archive,
)
from anysole.rho_grid import _corner_check, _write_session_artifacts, _run_cell_inference
from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.tune import parse_val_metrics
from configs.tools.common import effective_variant, sha256_file


def sample_session(n=12):
    rng = np.random.default_rng(9)
    offsets = rng.normal(size=(24, 3)).astype(np.float32) * .05
    offsets[0] = 0
    trans = np.zeros((n, 3), dtype=np.float32)
    trans[:, 0] = np.arange(n) / 40
    return {"session_id": "demo", "V_feat": np.zeros((n, V_FEAT_DIM)),
            "trans_global": trans, "frame_times_s": 1.5 + np.arange(n) / 40,
            "fake_mask": np.zeros(n), "offsets": offsets,
            "parents": np.asarray(JOINT_PARENTS), "betas": np.zeros(10)}


def sample_sequence(session, starts=(0, 8), tw=4):
    ids = np.concatenate([np.arange(start, start + tw) for start in starts])
    pose = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(len(ids), 24)
    trans = torch.tensor(session["trans_global"][ids])
    kp = fk_pose6d(pose[None], trans[None], torch.tensor(session["offsets"]),
                   torch.tensor(session["parents"]))[0]
    seq = {"pred_pose": pose.clone(), "pred_trans": trans.clone(), "gt_pose": pose,
           "gt_trans": trans, "kp_gt": kp, "contact_gt": torch.zeros(len(ids), 2),
           "floor_y": torch.tensor(-1.), "pressure_gt": torch.zeros(len(ids), 96),
           "pressure_pred": None, "offsets": torch.tensor(session["offsets"]),
           "parents": torch.tensor(session["parents"]), "betas": torch.zeros(10)}
    raw = [{"frame_start": torch.tensor(start), "pose_gt": pose[:tw]} for start in starts]
    return seq, raw


def metric_accum(seq, tw=4):
    accum = protocol._Accum()
    contact = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    # Surface diagnostics alone require unavailable SMPL assets. Keep the
    # real joint/FK, temporal, contact and rotation paths under test.
    def vertices(poses, trans, betas):
        return np.repeat(np.asarray(trans)[:, None, :], 6800, axis=1)
    with patch.object(protocol, "pelvis_to_smpl_trans", side_effect=lambda p, t, b: t), \
         patch.object(protocol, "smpl_vertices_from_archive_params", side_effect=vertices):
        protocol._session_metrics(seq, tw, 2, 0, accum, contact)
    return accum


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_erased_pressure_cannot_influence_retained_frames(self):
        torch.manual_seed(7)
        encoder = ModalEncoders(SharedEmbeddings(dim=16, tw=4), dim=16, tw=4,
                                t_encoder="foot_conv").eval()
        raw = np.random.default_rng(4).random((4, 96)).astype(np.float32)
        changed = raw.copy()
        changed[1] += 10
        def tactile(r):
            return torch.tensor(np.concatenate([r, physical_tokens(r[:, :48], r[:, 48:])], axis=1))[None]
        v = torch.randn(1, 4, V_FEAT_DIM)
        mask = torch.tensor([[False, True, False, False]])
        with torch.inference_mode():
            _, original = encoder(v, tactile(raw), torch.tensor([0]), mask_t=mask)
            _, modified = encoder(v, tactile(changed), torch.tensor([0]), mask_t=mask)
            _, dense = encoder(v, tactile(raw), torch.tensor([0]))
            _, no_missing = encoder(v, tactile(raw), torch.tensor([0]), mask_t=torch.zeros_like(mask))
            _, all_missing = encoder(v, tactile(raw), torch.tensor([0]), mask_t=torch.ones_like(mask))
            _, vonly = encoder(v, tactile(raw), torch.tensor([1]))
        torch.testing.assert_close(original, modified, rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(dense, no_missing))
        self.assertTrue(torch.equal(all_missing, vonly))

    def test_manifest_and_facts_share_absolute_pressure_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = Path(directory)
            (rec / "3").mkdir()
            (rec / "3/3_000000.025.jpg").touch()
            (rec / "meta.json").write_text(json.dumps({"started_at_iso": "2026-08-04T00:00:00.000"}))
            for foot in ("left", "right"):
                with (rec / ("pressure_%s.csv" % foot)).open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["t_us", "raw_t_us", "valid_mask", "fake", *map(str, range(1, 49))])
                    for i in range(5):
                        writer.writerow([i * 25000, i * 25000, int(i != 1), int(i == 1), *([i] * 48)])
            files = rec / "pressure_left.csv", rec / "pressure_right.csv"
            expected = recording_pressure(rec, files, 3)
            with patch("AnysoleWorkspace.tool.build_manifest.pressure_files", return_value=files):
                fake, valid, issue = pressure_flags("demo", 3, recording_path=rec)
            self.assertEqual(issue, "")
            self.assertEqual(fake, np.flatnonzero(expected[3]).tolist())
            self.assertEqual(valid, np.flatnonzero(expected[2]).tolist())
            self.assertEqual(fake, [0])  # normalized row mapping would miss this

    def test_packing_preserves_gaps_tail_and_input_arrays(self):
        session = sample_session(13)
        seq, raw = sample_sequence(session)
        before = seq["pred_pose"].clone()
        prepared, packed = prepare_session_sequence(seq, raw, session)
        np.testing.assert_array_equal(prepared["frame_indices"], [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(len(packed["pose"]), 13)
        self.assertFalse(packed["valid"][4:8].any())
        self.assertFalse(packed["valid"][-1])
        self.assertTrue(torch.equal(seq["pred_pose"], before))
        np.testing.assert_allclose(packed["trans"][packed["valid"]], seq["pred_trans"].numpy())

    def test_gaps_do_not_create_motion_derivatives(self):
        session = sample_session()
        seq, raw = sample_sequence(session)
        prepared, _ = prepare_session_sequence(seq, raw, session)
        accum = metric_accum(prepared)
        self.assertLess(accum.means()["jitter_gt_m_s3"], .01)
        self.assertLess(accum.means()["accel_error_m_s2"], 1e-6)
        self.assertEqual(accum.counts["jitter_gt_m_s3"], 2)
        self.assertNotIn("seam_jump_mm", accum.means())

    def test_archives_and_metrics_use_identical_processed_motion(self):
        session = sample_session(8)
        seq, raw = sample_sequence(session, starts=(0, 4))
        seq["pred_trans"][4:, 2] = .1
        prepared, packed = prepare_session_sequence(seq, raw, session)
        self.assertFalse(torch.equal(prepared["pred_trans"], seq["pred_trans"]))
        with tempfile.TemporaryDirectory() as directory, \
             patch("anysole.data.smpl_io.pelvis_to_smpl_trans", side_effect=lambda p, t, b: t), \
             patch("anysole.data.smpl_io.smpl_archive_metadata", return_value={}):
            path = Path(directory) / "motion.npz"
            write_motion_archive(path, session, packed)
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["pred_pelvis_trans"], prepared["pred_trans"].numpy())
                np.testing.assert_array_equal(archive["source_frame_times_s"], session["frame_times_s"])
                self.assertEqual(str(archive["evaluation_pipeline"]), PIPELINE_VERSION)
                saved_joints = archive["joint_xyz_world"]
            direct_joints = fk_pose6d(prepared["pred_pose"][None], prepared["pred_trans"][None],
                                     seq["offsets"], seq["parents"])[0].numpy()
            np.testing.assert_allclose(saved_joints, np.stack([direct_joints[..., 0],
                                      -direct_joints[..., 2], direct_joints[..., 1]], axis=-1))

    def test_rho_archive_keeps_original_grid_and_repr_window_locations(self):
        session = sample_session(13)
        seq, raw = sample_sequence(session)
        _, packed = prepare_session_sequence(seq, raw, session)
        batch = {"frame_start": torch.tensor([0, 8]), "pose_gt": torch.zeros(2, 4, 144)}
        with tempfile.TemporaryDirectory() as directory, \
             patch("anysole.data.smpl_io.pelvis_to_smpl_trans", side_effect=lambda p, t, b: t), \
             patch("anysole.data.smpl_io.smpl_archive_metadata", return_value={}):
            folder = Path(directory)
            _write_session_artifacts(folder, folder, "demo", 100, 100, 0, batch,
                None, None, None, None, {"F": torch.zeros(2, 8, 16)},
                SimpleNamespace(skip_npz=False, skip_repr=False), session=session, packed=packed)
            with np.load(folder / "demo_rhoV100_rhoT100.npz") as archive:
                self.assertEqual(len(archive["poses"]), 13)
                np.testing.assert_array_equal(archive["frame_indices"], np.arange(13))
                self.assertFalse(archive["valid_mask"][4:8].any())
            with np.load(folder / "demo_rhoV100_rhoT100_repr.npz") as archive:
                np.testing.assert_array_equal(archive["window_starts"], [0, 8])

    def test_variants_distinguish_training_conditions_and_same_size_partitions(self):
        base = {"tw": 20, "dropout": .1, "config_probs": [.5, .25, .25], "lr": .0001}
        for change in ({"dropout": .2}, {"config_probs": [0., 1., 0.]}, {"lr": .000100000001}):
            self.assertNotEqual(variant_from_config(base), variant_from_config({**base, **change}))
        self.assertNotEqual(variant_from_config({"part_joints": [[0], [1, 2]]}),
                            variant_from_config({"part_joints": [[0, 1], [2]]}))
        self.assertLessEqual(len(variant_from_config({**base, "traj_deltas": list(range(1000))})), 255)

    def test_scheduler_address_matches_actual_yaml_and_overrides(self):
        spec = {"config": "anysole/configs/v1.yaml", "epochs": 400, "batch_size": 64,
                "train_seed": 3, "train_args": {"tw": 40, "stride": 40, "grad_clip": 5.,
                                                "dropout": 0, "config_probs": [0, 1, 0]}}
        config = load_config(ROOT / spec["config"])
        config.update(spec["train_args"])
        config.update(dropout=0., config_probs=[0., 1., 0.])
        config.update(epochs=400, batch_size=64, seed=3,
                      split_hash=sha256_file(Path(config["split_csv"])))
        self.assertEqual(effective_variant(spec), variant_from_config(config))

    def test_scheduler_partition_names_resolve_like_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition.json"
            groups = [[0], list(range(1, 24))]
            path.write_text(json.dumps({"partition": {"a": [JOINT_NAMES[0]], "b": list(JOINT_NAMES[1:])}}))
            spec = {"train_args": {"part_json": str(path)}}
            config = load_config(ROOT / "anysole/configs/v1.yaml")
            config.update(epochs=800, seed=1, part_joints=groups,
                          split_hash=sha256_file(Path(config["split_csv"])))
            self.assertEqual(effective_variant(spec), variant_from_config(config))

    def test_eval_and_rho_corners_agree_with_real_network(self):
        session = sample_session()
        seq, raw = sample_sequence(session)
        rng = np.random.default_rng(3)
        pressure = rng.random((12, 96)).astype(np.float32)
        phys = physical_tokens(pressure[:, :48], pressure[:, 48:])
        samples = []
        for w, sample in enumerate(raw):
            start = int(sample["frame_start"])
            samples.append({"session_id": "demo", "frame_start": sample["frame_start"],
                "pose_gt": seq["gt_pose"][w * 4:(w + 1) * 4],
                "V_feat": torch.randn(4, V_FEAT_DIM), "T_raw": torch.tensor(pressure[start:start + 4]),
                "T_phys": torch.tensor(phys[start:start + 4]),
                "trans_anchor": torch.tensor(session["trans_global"][max(start - 1, 0)]),
                "trans_gt": seq["gt_trans"][w * 4:(w + 1) * 4] - torch.tensor(session["trans_global"][max(start - 1, 0)]),
                **{key: seq[key][w * 4:(w + 1) * 4] for key in ("kp_gt", "contact_gt")},
                **{key: seq[key] for key in ("floor_y", "offsets", "parents", "betas")}})
        class Dataset:
            sessions = [session]
            valid_windows = [(0, 0, 4), (0, 8, 12)]
            def __len__(self): return len(samples)
            def __getitem__(self, i): return samples[i]
        torch.manual_seed(11)
        model = AnySoleModelV2(d=16, tw=4, pose_layers=1, t_encoder="foot_conv", pose_parts=9).eval()
        args = SimpleNamespace(skip_npz=True, skip_repr=True)
        def vertices(poses, trans, betas):
            return np.repeat(np.asarray(trans)[:, None], 6800, axis=1)
        with patch.object(protocol, "pelvis_to_smpl_trans", side_effect=lambda p, t, b: t), \
             patch.object(protocol, "smpl_vertices_from_archive_params", side_effect=vertices):
            for rv, rt, cid in ((100, 100, 0), (100, 0, 1), (0, 100, 2)):
                evaluated = protocol._evaluate_config(Dataset(), model, torch.device("cpu"), cid,
                    True, None, 50, False, 4, 2)
                grid = _run_cell_inference(model, Dataset(), {"demo": [0, 1]}, torch.device("cpu"),
                    4, 2, rv, rt, 0, 0, False, False, None, None, args)
                self.assertEqual(set(evaluated), set(grid))
                for key in evaluated:
                    self.assertAlmostEqual(evaluated[key], grid[key], places=4, msg=key)

    def test_detail_metric_names_feed_tuning_and_corner_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir()
            detail = {name: {"MPJPE": 3., "PA-MPJPE": 2., "RootOrientation": 1.}
                      for name in ("VT2M", "V2M", "T2M")}
            (root / "metrics/val.json").write_text(json.dumps({"evaluation_pipeline": PIPELINE_VERSION,
                                                               "metrics": detail}))
            self.assertEqual(parse_val_metrics(root), {k: 3. for k in detail})
            args = SimpleNamespace(ckpt=root / "checkpoints/ckpt_last.pt", split="val", limit_sessions=None)
            cells = {"s0": {"rhoV100_rhoT100": {"mpjpe_mm": 3., "pa_mpjpe_mm": 2.}}}
            self.assertEqual(_corner_check(cells, [0], args)["VT2M"]["pa_mpjpe_mm"]["diff"], 0.)


if __name__ == "__main__":
    unittest.main()
