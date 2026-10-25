"""Checkpoint policy regression tests: candidate pool, gates, selection rules.

Run with touch_gait Python: python -m unittest discover -s tests -p test_checkpoint_policy.py
"""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from anysole.checkpoint_review import review
from anysole.checkpoint_candidates import prepare
from anysole.utils.checkpoint_policy import CandidateArchive, DashboardUnavailable, METRICS, infer_role

REPO = Path(__file__).resolve().parents[1]


def dashboard(v=10.0, t=20.0, vt=9.0, gv=100.0, gt=200.0, gvt=90.0):
    return {**{"val/pa_mpjpe/" + k: x for k, x in zip(("V", "T", "VT"), (v, t, vt))},
            **{"val/mpjpe/" + k: x for k, x in zip(("V", "T", "VT"), (gv, gt, gvt))},
            **{"val/tau0_mpjpe/" + k: x for k, x in zip(("V", "T", "VT"), (gv, gt, gvt))}}


def checkpoint(epoch):
    return {"epoch": epoch, "model": {"w": torch.tensor([float(epoch)])},
            "optimizer": {"do_not_duplicate": True}, "config": {"example": True}}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_role_inference_matches_actual_inputs(self):
        for probs, role in (([.5, .25, .25], "U"), ([0, 1, 0], "S_V"),
                            ([0, 0, 1], "S_T"), ([1, 0, 0], "S_VT")):
            self.assertEqual(infer_role({"config_probs": probs}), role)
        with self.assertRaises(ValueError):
            infer_role({"config_probs": [0, 0, 1], "checkpoint_role": "S_VT"})
        with self.assertRaises(ValueError):
            infer_role({"config_probs": [0, float("nan"), 1]})

    def test_gate_passing_epochs_are_all_retained(self):
        # Every epoch passing its role's gates is kept in full — the formal
        # review needs the complete selection domain.
        archive = CandidateArchive(self.root / "archive", "U")
        for epoch in range(1, 31):
            state = archive.update(checkpoint(epoch), dashboard())
        self.assertEqual(len(state["candidates"]), 30)
        self.assertEqual(len(list((archive.root / "weights").glob("*.pt"))), 30)
        self.assertTrue(all(state["candidates"][i]["gate_pass"]
                            for i in state["candidates"]))

    def test_gate_failure_survives_only_as_champion(self):
        archive = CandidateArchive(self.root / "archive", "U")
        state = archive.update(checkpoint(1), dashboard(t=30, gt=200))
        # VT 50 > 40 fails the gate; retained only as the VT Global champion.
        state = archive.update(checkpoint(2), dashboard(t=15, gt=100, vt=50, gvt=80))
        self.assertEqual(set(state["candidates"]), {"epoch_000001", "epoch_000002"})
        self.assertEqual(state["champions"]["VT/GlobalMPJPE"], "epoch_000002")
        self.assertFalse(state["candidates"]["epoch_000002"]["gate_pass"])
        # A later gate-failing epoch beats it on every champion axis: evicted.
        state = archive.update(checkpoint(3), dashboard(t=14, gt=90, vt=45, gvt=70))
        self.assertEqual(set(state["candidates"]), {"epoch_000001", "epoch_000003"})
        self.assertFalse((archive.root / "weights/epoch_000002.pt").exists())
        self.assertTrue((archive.root / "weights/epoch_000003.pt").exists())

    def test_expert_gate_uses_only_its_own_condition(self):
        archive = CandidateArchive(self.root / "archive", "S_T")
        # S_T only checks the T gate (<= 90); V/VT scores are irrelevant.
        state = archive.update(checkpoint(1), dashboard(v=500, vt=500, t=20))
        self.assertTrue(state["candidates"]["epoch_000001"]["gate_pass"])
        # T 95 > 90 fails; still retained as the only (T PA/Global) champion.
        state = archive.update(checkpoint(2), dashboard(v=500, vt=500, t=95, gt=150))
        self.assertFalse(state["candidates"]["epoch_000002"]["gate_pass"])
        self.assertEqual(set(state["candidates"]), {"epoch_000001", "epoch_000002"})

    def test_nonfinite_does_not_mutate_archive(self):
        archive = CandidateArchive(self.root / "archive", "S_T")
        with self.assertRaises(ValueError):
            archive.update(checkpoint(1), dashboard(gt=float("nan")))
        self.assertFalse(archive.root.exists())
        with self.assertRaises(ValueError):
            CandidateArchive(self.root / "other", "U", v_pa_max=-1)

    def test_latest_mode_keeps_only_the_most_recent_gate_passer(self):
        archive = CandidateArchive(self.root / "archive", "U", keep_latest=True)
        state = archive.update(checkpoint(1), dashboard())          # pass -> stored
        state = archive.update(checkpoint(2), dashboard(vt=50, gvt=70))  # VT gate fail -> keep ep1
        self.assertEqual(set(state["candidates"]), {"epoch_000001"})
        self.assertFalse((archive.root / "weights/epoch_000002.pt").exists())
        state = archive.update(checkpoint(3), dashboard(gv=50))     # pass -> replaces ep1
        self.assertEqual(set(state["candidates"]), {"epoch_000003"})
        self.assertFalse((archive.root / "weights/epoch_000001.pt").exists())
        self.assertTrue((archive.root / "weights/epoch_000003.pt").exists())
        self.assertEqual(state["mode"], "latest")
        self.assertEqual(state["champions"], {})

    def test_fresh_directory_is_required(self):
        archive = CandidateArchive(self.root / "archive", "U")
        archive.update(checkpoint(1), dashboard())
        with self.assertRaises(FileExistsError):
            CandidateArchive(self.root / "archive", "U")

    def test_failed_save_never_publishes_missing_checkpoint(self):
        archive = CandidateArchive(self.root / "archive", "S_T")
        state = archive.update(checkpoint(1), dashboard())
        before = (archive.root / "manifest.json").read_bytes()
        def fail(*args):
            raise OSError("simulated disk full")
        with self.assertRaises(OSError):
            archive.update(checkpoint(2), dashboard(gt=100), save=fail)
        self.assertEqual(before, (archive.root / "manifest.json").read_bytes())
        self.assertTrue((archive.root / state["candidates"]["epoch_000001"]["file"]).exists())

    def test_live_train_saving_block_keeps_legacy_outputs(self):
        # Execute the actual training save/validation block with a tiny model;
        # no expensive dataset/training or copy of the implementation is used.
        tree = ast.parse((REPO / "anysole/train.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "epoch")
        start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in n.targets))
        end = next(i for i, n in enumerate(loop.body[start:], start) if isinstance(n, ast.If)
                   and ast.dump(n.test) == ast.dump(ast.parse("wandb_run is not None", mode="eval").body))
        code = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(loop.body[start:end]),
                                                           type_ignores=[])), "live_train_save_block", "exec")
        import os
        for enabled in (False, True):
            out = self.root / ("enabled" if enabled else "legacy")
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            cfg = {"wandb_eval_interval": 1}
            archive = CandidateArchive(out / "multi_metric", "U") if enabled else None
            namespace = {"model": model, "optimizer": optimizer, "config": cfg, "out_dir": out,
                         "torch": torch, "os": os, "json": json, "val_loader": [1], "diffusion": None,
                         "device": "cpu", "occ_mix_kwargs": {}, "best_vt": None,
                         "checkpoint_archive": archive, "_evaluate": lambda *args: dashboard()}
            namespace["DashboardUnavailable"] = DashboardUnavailable
            for epoch in range(3):
                namespace["epoch"] = epoch
                namespace["_evaluate"] = lambda *args, ep=epoch: dashboard(gt=200 + ep, gvt=90 - ep)
                exec(code, namespace)
            self.assertEqual(torch.load(out / "ckpt_last.pt", weights_only=False)["epoch"], 3)
            self.assertEqual(torch.load(out / "ckpt_best.pt", weights_only=False)["epoch"], 3)
            self.assertEqual((out / "multi_metric").exists(), enabled)
            if not enabled:
                self.assertEqual(set(p.name for p in out.iterdir()),
                                 {"ckpt_last.pt", "ckpt_best.pt"})
            else:
                # All three epochs pass the gates: the pool keeps them all.
                self.assertEqual(len(list(archive.root.glob("weights/*.pt"))), 3)
                invalid = dashboard()
                del invalid["val/pa_mpjpe/T"]
                namespace["epoch"] = 3
                namespace["_evaluate"] = lambda *args: invalid
                exec(code, namespace)
                self.assertTrue((archive.root / "skipped_validation.jsonl").exists())
                self.assertEqual(torch.load(out / "ckpt_last.pt", weights_only=False)["epoch"], 4)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def result(self, name, value=100, modes=("V2M", "T2M", "VT2M")):
        folder = self.root / name
        folder.mkdir()
        ckpt = folder / "weight.pt"
        ckpt.write_text(name)
        obj = {"checkpoint": str(ckpt), "modal": "anysolev2", "contact_method": "f6_soft", "split": "val",
               "protocol_seed": 0, "tw": 20, "sample_steps": 50, "forward_axis": "+Z", "robustness": False,
               "metrics": {mode: {metric: value for metric in METRICS} for mode in modes}}
        (folder / "val.json").write_text(json.dumps(obj))
        for kind in ("session", "window"):
            block = ([{"session_id": "S1", "n_frames": 20}] if kind == "session" else
                     {"session_ids": ["S1"], "frame_starts": [0], "n_frames": [20]})
            (folder / (kind + "_detail_val.json")).write_text(json.dumps({
                "checkpoint": str(ckpt), "split": "val", "tw": 20, "contact_method": "f6_soft",
                "conditions": {mode: block for mode in modes}}))
        return folder / "val.json"

    def spec(self, candidate, reference, role="U"):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"split": "val", "experiment": "E1", "rules": None,
            "candidates": [{"id": "candidate", "role": role, "result": str(candidate)}],
            "references": {mode: {"result": str(reference), "mode": mode}
                           for mode in ("V2M", "T2M", "VT2M")}}))
        return path

    def candidate(self, name, v=30.0, t=80.0, vt=31.0):
        result = self.result(name)
        obj = json.loads(result.read_text())
        obj["metrics"]["V2M"]["PA-MPJPE"] = v
        obj["metrics"]["T2M"]["PA-MPJPE"] = t
        obj["metrics"]["VT2M"]["PA-MPJPE"] = vt
        result.write_text(json.dumps(obj))
        return result

    def rules_spec(self, results, role="U", dpa_max=0.5,
                   gates=None, experiment="E2"):
        path = self.root / "manifest_rules.json"
        reference = next(iter(results.values()))
        path.write_text(json.dumps({"split": "val", "experiment": experiment,
            "rules": {"thresholds": {"V": (gates or {}).get("V", 40.0),
                                     "T": (gates or {}).get("T", 90.0),
                                     "VT": (gates or {}).get("VT", 40.0),
                                     "dpa": dpa_max}},
            "candidates": [{"id": name, "role": role, "result": str(r)}
                           for name, r in results.items()],
            "references": {mode: {"result": str(reference), "mode": mode}
                           for mode in ("V2M", "T2M", "VT2M")}}))
        return path

    def test_full_ten_units_one_weight_and_no_automatic_winner(self):
        c, r = self.result("candidate", 90), self.result("reference")
        result = review(self.spec(c, r), self.root / "out")
        state = result["candidates"]["candidate"]
        self.assertIsNone(result["selected_checkpoint"])
        self.assertEqual(result["status"], "comparison_only")
        self.assertEqual(len(state["panels"]), 3)
        self.assertEqual(state["panels"]["T2M"]["numerically_lower_count"], 10)
        self.assertEqual(state["panels"]["T2M"]["metrics"]["RootOrientationDrift"]["unit"], "deg")
        self.assertEqual(state["panels"]["T2M"]["metrics"]["Jitter_pred"]["unit"], "m/s^3")
        self.assertEqual(len((self.root / "out/ten_metrics.csv").read_text().splitlines()), 31)

    def test_selection_picks_best_vt_among_four_threshold_passers(self):
        a = self.candidate("a", v=30, t=80, vt=32)    # dpa +2.0: out of band
        b = self.candidate("b", v=31, t=81, vt=31.5)  # dpa +0.5: in band
        c = self.candidate("c", v=32, t=82, vt=31.2)  # dpa -0.8: in band, best VT
        result = review(self.rules_spec({"a": a, "b": b, "c": c}), self.root / "out")
        self.assertEqual(result["status"], "selection_applied")
        self.assertEqual(result["selected_checkpoint"], str(c.with_name("weight.pt")))
        self.assertTrue("选中：c" in (self.root / "out/comparison.md").read_text())

    def test_gate_failure_is_not_band_member(self):
        a = self.candidate("a", v=30, t=95, vt=31)    # T gate fails
        result = review(self.rules_spec({"a": a}), self.root / "out")
        self.assertEqual(result["status"], "no_qualified_candidate")
        self.assertIsNone(result["selected_checkpoint"])
        self.assertIn("无候选通过全部四个阈值", (self.root / "out/comparison.md").read_text())

    def test_delta_gate_failure_is_not_selected(self):
        a = self.candidate("a", v=30, t=80, vt=33)    # dpa +3.0 > 0.5: delta gate fails
        result = review(self.rules_spec({"a": a}), self.root / "out")
        self.assertEqual(result["status"], "no_qualified_candidate")
        self.assertIsNone(result["selected_checkpoint"])
        self.assertEqual(result["candidates"]["a"]["gate_pass"], False)

    def test_experts_are_compared_but_not_selected(self):
        a = self.candidate("a", v=30, t=20, vt=31)
        result = review(self.rules_spec({"a": a}, role="S_T"), self.root / "out")
        self.assertEqual(result["status"], "no_qualified_candidate")
        self.assertIsNone(result["selected_checkpoint"])

    def test_invalid_rules_are_rejected(self):
        a = self.candidate("a")
        path = self.rules_spec({"a": a})
        obj = json.loads(path.read_text())
        obj["rules"]["thresholds"] = {"V": 40.0, "T": 90.0}  # VT / dpa missing
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "rules.thresholds"):
            review(path, self.root / "out")
        obj["rules"]["thresholds"] = {"V": 40.0, "T": 90.0, "VT": 40.0, "dpa": float("nan")}
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "rules.thresholds"):
            review(path, self.root / "out")

    def test_test_split_is_rejected(self):
        c, r = self.result("candidate"), self.result("reference")
        obj = json.loads(c.read_text()); obj["split"] = "test"; c.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "Only validation"):
            review(self.spec(c, r), self.root / "out")

    def test_mismatched_grid_is_rejected(self):
        c, r = self.result("candidate"), self.result("reference")
        detail_path = r.with_name("window_detail_val.json")
        obj = json.loads(detail_path.read_text()); obj["conditions"]["V2M"]["frame_starts"] = [1]
        # Build independent objects: the serialized fixture copies aliased blocks.
        detail_path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "frame grids"):
            review(self.spec(c, r), self.root / "out")

    def test_missing_metric_and_spliced_input_are_rejected(self):
        c, r = self.result("candidate"), self.result("reference")
        obj = json.loads(c.read_text()); del obj["metrics"]["T2M"]["RootOrientationDrift"]
        c.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "formal metric"):
            review(self.spec(c, r), self.root / "out")
        obj["metrics"].pop("T2M"); c.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "ONE checkpoint"):
            review(self.spec(c, r), self.root / "out")

    def test_large_drift_loss_remains_visible_and_does_not_elect_winner(self):
        c, r = self.result("candidate", 90), self.result("reference")
        obj = json.loads(c.read_text()); obj["metrics"]["T2M"]["RootOrientationDrift"] = 135
        c.write_text(json.dumps(obj))
        result = review(self.spec(c, r), self.root / "out")
        panel = result["candidates"]["candidate"]["panels"]["T2M"]
        self.assertEqual(panel["numerically_lower_count"], 9)
        self.assertEqual(panel["metrics"]["RootOrientationDrift"]["delta"], 35)
        self.assertIsNone(result["selected_checkpoint"])

    def test_pareto_does_not_merge_wins_across_inputs(self):
        a, b, r = self.result("a", 90), self.result("b", 90), self.result("reference", 100)
        oa, ob = json.loads(a.read_text()), json.loads(b.read_text())
        oa["metrics"]["T2M"] = {m: 50 for m in METRICS}
        ob["metrics"]["VT2M"] = {m: 50 for m in METRICS}
        a.write_text(json.dumps(oa)); b.write_text(json.dumps(ob))
        spec = self.spec(a, r); obj = json.loads(spec.read_text())
        obj["candidates"] = [{"id": "a", "role": "U", "result": str(a)},
                             {"id": "b", "role": "U", "result": str(b)}]
        spec.write_text(json.dumps(obj))
        result = review(spec, self.root / "out")
        lists = result["candidate_lists"]["U"]
        self.assertEqual(set(lists["all_input_ten_metric_pareto"]), {"a", "b"})
        self.assertEqual(lists["by_input"]["T2M"], ["a"])
        self.assertEqual(lists["by_input"]["VT2M"], ["b"])

    def test_e5_references_must_come_from_same_checkpoint(self):
        c, r = self.result("candidate", 90), self.result("reference")
        spec = self.spec(c, r)
        obj = json.loads(spec.read_text()); obj["experiment"] = "E5"
        spec.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "same checkpoint"):
            review(spec, self.root / "invalid")
        obj["references"] = {mode: {"same_checkpoint_mode": "V2M" if mode == "VT2M" else mode}
                              for mode in ("V2M", "T2M", "VT2M")}
        spec.write_text(json.dumps(obj))
        result = review(spec, self.root / "valid")
        self.assertEqual(result["candidates"]["candidate"]["panels"]["VT2M"]["reference_mode"], "V2M")
        self.assertIsNone(result["selected_checkpoint"])

    def test_native_evaluation_plan_uses_only_correct_validation_modes(self):
        cfg = self.root / "config.yaml"; cfg.write_text("contact_method: f6_soft\n")
        archive = CandidateArchive(self.root / "checkpoints/multi_metric", "S_T")
        state = archive.update(checkpoint(1), dashboard())
        legacy = self.root / "checkpoints/ckpt_best.pt"
        legacy.write_bytes((archive.root / next(iter(state["candidates"].values()))["file"]).read_bytes())
        before = legacy.read_bytes()
        plan = prepare(archive.root / "manifest.json", cfg, self.root / "plan", device="cpu")
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(len(plan["jobs"]), 1)
        job = plan["jobs"][0]
        self.assertEqual(job["modes"], ["T2M"])
        command = job["command"]
        self.assertEqual(command[command.index("--split") + 1], "val")
        self.assertIn("--no-write-motion", command)
        self.assertEqual(len(job["aliases"]), 2)
        self.assertEqual(legacy.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
