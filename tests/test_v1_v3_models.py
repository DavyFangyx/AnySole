"""New model numbering, isolated structural changes and enumerated training."""
import copy
import io
import unittest
import warnings

import torch

from anysole.eval import _load_model
from anysole.registry import get_builder
from anysole.train import backward_v7_enum, condition_inputs, parse_args
from anysole.types import CONFIG_T, CONFIG_V, JOINT_PROTOCOL_CHECKSUM, MOTION_PROTOCOL
from anysole.utils.losses import compute_losses
from configs.tools.common import load_registry
from configs.tools.runner import _evaluation_splits, _train_command


def config(name):
    entry = get_builder(name)
    return {"model_name": name, "modal": "anysolev2", "d_model": 16, "tw": 4,
            "pose_layers": 1, "dropout": 0., "config_probs": [.5, .25, .25],
            "lambda_pose": 3., "lambda_traj": 1., "lambda_contact": 1.,
            "lambda_air": 2., "traj_deltas": [1],
            "motion_n_joints": 24, "motion_protocol": MOTION_PROTOCOL,
            "joint_protocol_checksum": JOINT_PROTOCOL_CHECKSUM,
            **entry.STRUCTURE, **entry.CONFIG_EXTRA}


def build(name):
    torch.manual_seed(17)
    return get_builder(name).build(config(name)).eval()


def batch():
    torch.manual_seed(19)
    return {"V_feat": torch.randn(2, 4, 2051), "T_raw": torch.rand(2, 4, 96),
            "T_phys": torch.rand(2, 4, 12), "pose_gt": torch.randn(2, 4, 144),
            "vel_gt": torch.randn(2, 4, 3), "trans_gt": torch.zeros(2, 4, 3),
            "contact_gt": torch.tensor([[[1., 0.]] * 4, [[0., 1.]] * 4])}


def forward(model, data, mode):
    cid = torch.full((2,), mode, dtype=torch.long)
    return model(data["V_feat"], data["T_raw"], data["T_phys"], cid)


class NewModelTests(unittest.TestCase):
    def test_v1_is_the_existing_c_architecture(self):
        a, b = build("V1"), build("V7_Cv1")
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        data = batch()
        with torch.no_grad():
            for mode in range(3):
                x, y = forward(a, data, mode), forward(b, data, mode)
                for key in ("x0_hat", "v_hat", "trans_hat"):
                    torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)

    def test_v2_starts_as_v1_and_has_independent_task_gates(self):
        a, b = build("V1"), build("V2")
        data = batch()
        with torch.no_grad():
            x, y = forward(a, data, 0), forward(b, data, 0)
            for key in ("x0_hat", "v_hat"):
                torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
            b.route_root_mlp[-1].bias.add_(2.)
            z = forward(b, data, 0)
        gate = z["route_gate"]
        self.assertEqual(tuple(gate.shape), (2, 4, 10, 1))
        self.assertGreater((gate[:, :, 0] - gate[:, :, 1]).abs().max().item(), .01)
        self.assertGreater((z["v_hat"] - y["v_hat"]).abs().max().item(), 0.)
        self.assertFalse(any(k.startswith("t_only_") for k in b.state_dict()))

    def test_v3_adapters_start_at_identity_and_only_affect_t_only(self):
        a, b = build("V1"), build("V3")
        data = batch()
        with torch.no_grad():
            for mode in range(3):
                x, y = forward(a, data, mode), forward(b, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
            b.t_only_root_adapter[-1].bias.fill_(.2)
            b.t_only_pose_adapter[-1].bias.fill_(.2)
            for mode in (0, CONFIG_V):
                x, y = forward(a, data, mode), forward(b, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
            x, y = forward(a, data, CONFIG_T), forward(b, data, CONFIG_T)
        self.assertGreater((x["x0_hat"] - y["x0_hat"]).abs().max().item(), 0.)
        self.assertGreater((x["v_hat"] - y["v_hat"]).abs().max().item(), 0.)
        self.assertFalse(hasattr(b, "route_root_mlp"))

    def test_adapter_gradients_are_private_to_t_only(self):
        model = build("V3")
        data = batch()
        for mode in range(3):
            model.zero_grad(set_to_none=True)
            out = forward(model, data, mode)
            (out["x0_hat"].square().mean() + out["v_hat"].square().mean()).backward()
            for adapter in (model.t_only_root_adapter, model.t_only_pose_adapter):
                norm = sum(p.grad.abs().sum().item() for p in adapter.parameters() if p.grad is not None)
                if mode == CONFIG_T:
                    self.assertGreater(norm, 0.)
                else:
                    self.assertEqual(norm, 0.)

    def test_missing_inputs_do_not_leak_and_all_gradients_are_finite(self):
        data = batch()
        for name in ("V1", "V2", "V3"):
            model = build(name)
            for mode, keys in ((CONFIG_V, ("T_raw", "T_phys")), (CONFIG_T, ("V_feat",))):
                changed = dict(data)
                changed.update({key: data[key] * 10 for key in keys})
                with torch.no_grad():
                    a, b = forward(model, data, mode), forward(model, changed, mode)
                torch.testing.assert_close(a["x0_hat"], b["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(a["v_hat"], b["v_hat"], rtol=0, atol=0)
            model.zero_grad(set_to_none=True)
            losses = compute_losses(forward(model, data, 0), data, torch.zeros(2, dtype=torch.long), config(name))
            losses["loss"].backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_checkpoint_reload_preserves_each_new_structure(self):
        data = batch()
        for name in ("V1", "V2", "V3"):
            model = build(name)
            stream = io.BytesIO()
            torch.save({"model": model.state_dict(), "config": config(name), "epoch": 1}, stream)
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=False)
            restored = _load_model(checkpoint, config(name), torch.device("cpu"))
            with torch.no_grad():
                for mode in range(3):
                    a, b = forward(model, data, mode), forward(restored, data, mode)
                    torch.testing.assert_close(a["x0_hat"], b["x0_hat"], rtol=0, atol=0)
                    torch.testing.assert_close(a["v_hat"], b["v_hat"], rtol=0, atol=0)


class EnumerationTests(unittest.TestCase):
    def test_streamed_enum_matches_weighted_objective_and_gradients(self):
        model = build("V1_enum")
        reference = copy.deepcopy(model)
        data = batch()
        cfg = config("V1_enum")
        original = {k: v.clone() for k, v in data.items()}
        _, actual = backward_v7_enum(model, data, cfg)
        objective = 0.
        t_probability = .75
        for mode, probability in enumerate(cfg["config_probs"]):
            cid = torch.full((2,), mode, dtype=torch.long)
            v, t, p, s = condition_inputs(data, cid)
            out = reference(v, t, p, cid, T_s2m=s)
            loss = compute_losses(out, data, cid, cfg)
            objective = objective + probability * (3 * loss["L_pose"] + loss["L_traj"])
            objective = objective + probability / t_probability * loss["L_contact"]
        objective.backward()
        torch.testing.assert_close(actual["loss"], objective.detach())
        for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other)
            if p.grad is None or q.grad is None:
                self.assertIsNone(p.grad)
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
        for key in original:
            torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
        self.assertFalse(actual["loss"].requires_grad)

    def test_invalid_enum_probabilities_are_rejected(self):
        model, data = build("V1"), batch()
        for probs in ([0, 0, 0], [.5, -.1, .6], [float("nan"), 0, 1], [1, 0]):
            with self.assertRaises(ValueError):
                backward_v7_enum(model, data, {**config("V1"), "config_probs": probs})

    def test_new_runs_have_matching_recipes_and_no_automatic_test(self):
        registry = load_registry()
        for name in ("V1", "V2", "V3", "V1_enum"):
            spec = registry["models"][name]
            args = parse_args(_train_command(registry, name)[3:])
            self.assertEqual(args.config_probs, [.5, .25, .25])
            self.assertFalse(args.occ_mix)
            self.assertEqual(args.final_eval_split, "none")
            self.assertEqual(args.condition_sampling, "enum" if name == "V1_enum" else "sample")
            self.assertEqual(_evaluation_splits(registry, name), ("val",))
            self.assertEqual(spec["epochs"], 740)
            self.assertTrue(args.checkpoint_all)


torch.set_num_threads(1)
warnings.filterwarnings("ignore", message="enable_nested_tensor")

if __name__ == "__main__":
    unittest.main()
