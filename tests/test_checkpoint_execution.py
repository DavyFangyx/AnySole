"""Regression checks for exact checkpoint selection and completed source runs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from configs.tools import common, runner


class CheckpointExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ckpts = self.root / "checkpoints"
        self.ckpts.mkdir()
        self.registry = {"defaults": {}, "models": {"demo": {
            "name": "demo", "which": "last", "train": True, "epochs": 10}}}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, epoch, epochs=10, **config):
        path = self.ckpts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "config": {"epochs": epochs, **config},
                    "model": {"weight": torch.tensor([epoch])}}, path)
        return path

    def resolve(self, which="last", external=None):
        spec = self.registry["models"]["demo"]
        spec["which"] = which
        if external is not None:
            spec["checkpoint"] = str(external)
        with patch.object(common, "registry_ckpt", side_effect=lambda *a, **kw:
                          self.ckpts / ("ckpt_" + kw["which"] + ".pt")):
            return common.checkpoint_path(self.registry, "audit", "demo")

    def materialize(self, force=False):
        spec = self.registry["models"]["demo"]
        def selected(*args):
            path = self.ckpts / ("ckpt_" + spec["which"] + ".pt")
            if not path.is_file():
                raise FileNotFoundError("requested checkpoint missing")
            return path
        with patch.object(runner, "infer_anysole_paths", return_value={"ckpt": self.ckpts / "ckpt_last.pt"}), \
             patch.object(runner, "checkpoint_path", side_effect=selected):
            return runner._materialize_checkpoint(self.registry, "audit", "demo", self.root,
                                                  self.root / "run.log", force)

    def test_last_is_used_even_when_best_exists(self):
        self.save("ckpt_best.pt", 1)
        last = self.save("ckpt_last.pt", 10)
        self.assertEqual(self.resolve(), last)
        with patch.object(runner, "_command_log") as command:
            self.assertEqual(self.materialize(), last)
            command.assert_not_called()

    def test_missing_requested_file_does_not_fall_back(self):
        self.save("ckpt_last.pt", 10)
        with self.assertRaises(FileNotFoundError):
            self.resolve("best")

    def test_explicit_candidate_has_priority(self):
        self.save("ckpt_last.pt", 10)
        candidate = self.save("multi_metric/weights/epoch_000005.pt", 5)
        self.assertEqual(self.resolve(external=candidate), candidate)
        self.assertEqual(runner._assert_training_complete(candidate), 5)

    def test_partial_run_is_rejected_without_starting_training(self):
        self.save("ckpt_last.pt", 3)
        with patch.object(runner, "_command_log") as command:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                self.materialize()
            command.assert_not_called()

    def test_best_without_last_is_not_treated_as_complete(self):
        self.save("ckpt_best.pt", 1)
        with self.assertRaisesRegex(RuntimeError, "without a completed"):
            self.materialize()

    def test_early_best_is_allowed_after_full_source_training(self):
        self.save("ckpt_last.pt", 10)
        best = self.save("ckpt_best.pt", 1)
        self.assertEqual(runner._assert_training_complete(best, 10), 1)

    def test_candidate_configuration_must_match_source(self):
        self.save("ckpt_last.pt", 10, split_csv="new_split.csv")
        best = self.save("ckpt_best.pt", 1, split_csv="old_split.csv")
        with self.assertRaisesRegex(RuntimeError, "config mismatch"):
            runner._assert_training_complete(best)

    def test_training_budget_mismatch_is_reported(self):
        last = self.save("ckpt_last.pt", 10)
        with self.assertRaisesRegex(RuntimeError, "mismatched training"):
            runner._assert_training_complete(last, 20)

    def test_training_exit_failure_is_not_masked_by_a_checkpoint(self):
        self.save("ckpt_last.pt", 3)
        with patch.object(runner, "_train_command", return_value=["not-executed"]), \
             patch.object(runner, "_command_log", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                self.materialize(force=True)

    def test_checkpoint_policy_flags_reach_the_training_command(self):
        command = []
        runner._append_train_args(command, {"train_args": {
            "checkpoint_policy": "multi_metric", "checkpoint_role": "U",
            "checkpoint_gate_v": 40, "checkpoint_gate_t": 90,
            "checkpoint_gate_vt": 40}})
        self.assertEqual(command, ["--checkpoint-policy", "multi_metric",
                                   "--checkpoint-role", "U",
                                   "--checkpoint-gate-v", "40",
                                   "--checkpoint-gate-t", "90",
                                   "--checkpoint-gate-vt", "40"])

    def test_mutated_checkpoint_does_not_publish_a_complete_run(self):
        ckpt = self.save("ckpt_last.pt", 10)
        config = self.root / "eval.yaml"
        config.write_text("contact_method: f6_soft\n")
        output = self.root / "evaluation"
        def initial(*args):
            path = output / "run_manifest.json"
            path.write_text(json.dumps({"status": "running"}))
            return path
        def mutate(*args):
            with ckpt.open("ab") as stream:
                stream.write(b"changed during evaluation")
        with patch.object(runner, "model_root", return_value=output), \
             patch.object(runner, "config_path", return_value=config), \
             patch.object(runner, "_write_initial_manifest", side_effect=initial), \
             patch.object(runner, "_materialize_checkpoint", return_value=ckpt), \
             patch.object(runner, "_formal_eval", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                runner._run_model(self.registry, "audit", "demo")
        self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"], "failed")
        self.assertFalse((output / ".done").exists())


if __name__ == "__main__":
    unittest.main()
