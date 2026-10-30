"""Regression tests runnable with unittest; no additional test dependency."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from anysole.data import contact_adapter, contact_package, workspace_adapter


class CorrectedContactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name)
        self.root = directory / "package"
        self.root.mkdir()
        session_dir = directory / "session"
        session_dir.mkdir()
        (session_dir / "frames.npz").write_bytes(b"clock version 1")
        bvh = directory / "motion.bvh"
        bvh.write_bytes(b"motion version 1")
        frame_ids = np.arange(4)
        np.savez(self.root / "labels.npz", contact=np.ones((4, 2)), frame_id=frame_ids)
        manifest = {"schema": "contact_label_package.v1", "method": "f6_soft", "sessions": {
            "demo": {"source_files": {"frames.npz": contact_package.sha256(session_dir / "frames.npz")},
                     "bvh_path": str(bvh), "bvh_sha256": contact_package.sha256(bvh),
                     "label_file": "labels.npz", "label_sha256": contact_package.sha256(self.root / "labels.npz")}}}
        path = self.root / "label_package.json"
        path.write_text(json.dumps(manifest))
        patcher = patch.object(contact_package, "PACKAGE_SHA256", contact_package.sha256(path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.session = {"session_id": "demo", "dir": session_dir, "frames": {"frame_id": frame_ids}}

    def test_physical_acceleration(self):
        time = np.arange(100, dtype=np.float64) / contact_adapter.FPS
        for speed, acceleration in [(0., 0.), (.1, 0.), (0., 2.)]:
            with self.subTest(speed=speed, acceleration=acceleration):
                points = np.zeros((100, 2, 3), dtype=np.float64)
                points[:, :, 2] = (speed * time + .5 * acceleration * time ** 2)[:, None]
                ctx = {"pts": points, "names": ["LeftFoot", "RightFoot"], "floor": 0., "n": 100}
                actual = contact_adapter._f6_signals(ctx, "left")["az"]
                np.testing.assert_allclose(actual[3:-3], acceleration, atol=1e-8)

    def test_load_returns_an_independent_array(self):
        reader = contact_package.ContactLabelPackage(self.root)
        first = reader.load(self.session)
        first[0, 0] = 0
        self.assertEqual(reader.load(self.session)[0, 0], 1)

    def test_clock_change_rejected_even_with_identical_frame_ids(self):
        (self.session["dir"] / "frames.npz").write_bytes(b"clock version 2")
        with self.assertRaisesRegex(contact_package.ContactPackageError, "changed dependency"):
            contact_package.ContactLabelPackage(self.root).load(self.session)

    def test_missing_label_cannot_trigger_silent_rebuild(self):
        (self.root / "labels.npz").unlink()
        with self.assertRaisesRegex(ValueError, "label missing"):
            contact_package.ContactLabelPackage(self.root).load(self.session)

    def test_manifest_tampering_is_rejected(self):
        path = self.root / "label_package.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(contact_package.ContactPackageError, "hash mismatch"):
            contact_package.ContactLabelPackage(self.root)

    def test_versioned_method_routes_through_strict_package(self):
        dest = self.root.parent / contact_package.CORRECTED_CONTACT_METHOD
        self.root.rename(dest)
        with patch.object(workspace_adapter, "LABEL_ROOT", dest.parent):
            actual = workspace_adapter.load_contact_label(self.session, contact_package.CORRECTED_CONTACT_METHOD)
            self.assertEqual(actual.shape, (4, 2))
            (dest / "label_package.json").unlink()
            with self.assertRaisesRegex(ValueError, "Cannot load pinned"):
                workspace_adapter.load_contact_label(self.session, contact_package.CORRECTED_CONTACT_METHOD)

    def test_registry_has_separate_paths_and_validation_only(self):
        from configs.z_gen.model_registry import MODELS
        from configs.z_gen.common import EXPERIMENTS
        # 当前线（10-29）：V7_CvN = SMPL24 状态法包（当前默认）；
        # V7_CvN_tafix = SMPL24 修正法包参照线——独立 contact 目录、
        # 训练 cell 即评估 cell（val+test），无终评分隔。
        for base in ["V7_Cv%d" % i for i in range(1, 8)]:
            current, corrected = MODELS[base + "_s1"], MODELS[base + "_tafix_s1"]
            self.assertEqual(current["contact_method"], contact_package.SMPL24_STATE_CONTACT_METHOD)
            self.assertEqual(corrected["contact_method"], contact_package.SMPL24_CORRECTED_CONTACT_METHOD)
            self.assertEqual(corrected["name"], current["name"])
            self.assertEqual(corrected["eval_splits"], ["val", "test"])
            self.assertEqual(corrected["train_args"]["final_eval_split"], "none")
            self.assertEqual(corrected["which"], "last")
        self.assertEqual(len(EXPERIMENTS["v7_tafix_train"]["MODEL_IDS"]), 21)


if __name__ == "__main__":
    unittest.main()
