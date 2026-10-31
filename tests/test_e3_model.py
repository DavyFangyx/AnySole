"""G1 model contracts: shared decode, motion QFACT, physical losses/routing."""
import unittest

import torch

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.models.common.pressure_readout import QueryPressureHead
from anysole.types import CONFIG_T, CONFIG_V, JOINT_PARENTS
from anysole.utils.geometry import e3_support_points
from anysole.utils.losses import (
    backward_e3_losses, compute_losses_v7, contact_height_original,
    e3_parameter_groups, pressure_pair_loss, pressure_supervision,
)


def model(pressure=False, adapters=False):
    torch.manual_seed(41)
    return AnySoleModelV7(
        d=16, tw=4, nhead=8, dropout=0., pose_layers=1,
        reliability_route=True, task_routes=True, t_only_adapters=adapters,
        pressure_head="query_factorized" if pressure else "none",
        pressure_cell_mean=torch.linspace(.01, .3, 96), pressure_seed=7).eval()


def inputs():
    torch.manual_seed(42)
    return torch.randn(2, 4, 2051), torch.rand(2, 4, 96), torch.rand(2, 4, 12)


class E3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_optional_head_preserves_common_initialization_rng_and_motion(self):
        base = model()
        base_rng = torch.get_rng_state()
        pressure = model(True)
        self.assertTrue(torch.equal(base_rng, torch.get_rng_state()))
        state = pressure.state_dict()
        self.assertEqual(set(base.state_dict()), {k for k in state if not k.startswith("pressure_readout.")})
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        base.load_state_dict(base.state_dict(), strict=True)
        v, raw, phys = inputs()
        for condition in range(3):
            cid = torch.full((2,), condition)
            with torch.no_grad():
                a, b = base(v, raw, phys, cid), pressure(v, raw, phys, cid)
            for name in ("x0_hat", "v_hat", "trans_hat", "contact_logits", "z"):
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
            prior = torch.linspace(.01, .3, 96)[None, None].expand(2, 4, 96)
            torch.testing.assert_close(b["pressure_hat"], prior)

    def test_decode_exact_for_all_conditions_and_nonidentity_t_adapters(self):
        net = model(adapters=True)
        with torch.no_grad():
            net.t_only_root_adapter[-1].bias.fill_(.23)
            net.t_only_pose_adapter[-1].bias.fill_(.17)
        v, raw, phys = inputs()
        for cid in (0, CONFIG_V, CONFIG_T):
            out = net(v, raw, phys, torch.full((2,), cid))
            for z in (out["z"], out["z"].flatten(1, 2)):
                decoded = net.decode_z(z)
                for name in ("x0_hat", "v_hat", "trans_hat", "z"):
                    torch.testing.assert_close(out[name], decoded[name], rtol=0, atol=0)
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        z = out["z"].detach().requires_grad_()
        decoded = net.decode_z(z)
        loss = decoded["x0_hat"].square().sum() + decoded["trans_hat"].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_motion_memory_query_time_and_all_missing_masks_are_explicit(self):
        torch.manual_seed(8)
        head = QueryPressureHead("query_direct", dim=16, nhead=8, dropout=0., memory="motion_z")
        memory = torch.randn(2, 3, 10, 16, requires_grad=True)
        valid = torch.tensor([[True] * 5, [False] * 5])
        times = torch.tensor([[100. + i * .025 for i in range(5)]] * 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "explicit memory_valid"):
            head(memory, valid, times)
        a = head(memory, valid, times, torch.ones(2, 3, dtype=torch.bool))["pressure_hat"]
        self.assertEqual(tuple(a.shape), (2, 5, 96))
        self.assertEqual(a[1].abs().sum().item(), 0.)
        shifted = head(memory, valid, times + 9000., torch.ones(2, 3, dtype=torch.bool))["pressure_hat"]
        torch.testing.assert_close(a, shifted, atol=1e-6, rtol=1e-6)
        a.sum().backward()
        self.assertTrue(torch.isfinite(memory.grad).all())
        self.assertGreater(memory.grad[0].abs().sum().item(), 0.)
        self.assertEqual(memory.grad[1].abs().sum().item(), 0.)

    def test_pressure_masked_supervision_pair_and_empty_support(self):
        prediction = torch.arange(4.).view(1, 4, 1, 1).expand(1, 4, 3, 96).clone().requires_grad_()
        pair = pressure_pair_loss(prediction)
        # Six camera pairs: squared differences 1,4,9,1,4,1.
        self.assertEqual(pair["valid_count"], 6 * 3 * 96)
        torch.testing.assert_close(pair["loss"], torch.tensor(20. / 6))
        prediction_bad = torch.full((1, 3, 96), float("nan"), requires_grad=True)
        empty = pressure_supervision(prediction_bad, prediction_bad.detach(), torch.zeros(1, 3, dtype=torch.bool))
        self.assertEqual(empty["valid_count"], 0)
        self.assertEqual(empty["loss"].item(), 0.)
        empty["loss"].backward()
        self.assertTrue(torch.isfinite(prediction_bad.grad).all())
        self.assertEqual(prediction_bad.grad.abs().sum().item(), 0.)
        solo = pressure_pair_loss(prediction[:, :1])
        self.assertEqual(solo["loss"].item(), 0.)

    def test_original_contact_hand_sum_free_h_detached_c_and_view_invariance(self):
        points = torch.zeros(2, 3, 2, 3)
        points[..., 1] = torch.tensor([[[1., 2.]] * 3, [[2., 3.]] * 3])
        points.requires_grad_()
        contact = torch.tensor([[[.5, .25]] * 3] * 2, requires_grad=True)
        ground = torch.tensor([.5, 1.5], requires_grad=True)
        term = contact_height_original(points, contact, ground)
        expected = 2 * 3 * (.5 * .5 ** 2 + .25 * 1.5 ** 2)
        self.assertAlmostEqual(term["raw_sum"].item(), expected)
        self.assertAlmostEqual(term["loss"].item(), expected / 2)
        for views in (1, 2, 4):
            multiview = points[:, None].expand(-1, views, -1, -1, -1)
            actual = contact_height_original(multiview, contact, ground)
            torch.testing.assert_close(actual["loss"], term["loss"])
            torch.testing.assert_close(actual["element_sum"], views * term["raw_sum"])
        term["loss"].backward()
        self.assertIsNone(contact.grad)
        torch.testing.assert_close(ground.grad, torch.tensor([-1.875, -1.875]))
        self.assertGreater(points.grad.abs().sum().item(), 0.)
        no_contact = contact_height_original(points, torch.zeros_like(contact), ground)
        self.assertEqual(no_contact["loss"].item(), 0.)

    def test_contact_masked_nan_and_train_foot_center_proxy_fk_gradients(self):
        pose = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(1, 4, 24).requires_grad_()
        trans = torch.ones(1, 4, 3, requires_grad=True)
        offsets = torch.full((24, 3), .03)
        parents = torch.tensor(JOINT_PARENTS)
        feet = e3_support_points(pose, trans, offsets, parents, torch.zeros(2, 3))
        term = contact_height_original(feet, torch.ones(1, 4, 2), torch.tensor([0.]))
        term["loss"].backward()
        for gradient in (pose.grad, trans.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)
        bad = torch.full((1, 4, 2, 3), float("nan"), requires_grad=True)
        masked = contact_height_original(bad, torch.ones(1, 4, 2), 0., torch.zeros(1, 4, dtype=torch.bool))
        masked["loss"].backward()
        self.assertEqual(masked["loss"].item(), 0.)
        self.assertTrue(torch.isfinite(bad.grad).all())

    def test_pair_route_only_hf_and_supervision_keeps_encoder_gradient(self):
        net = model(True)
        # QFACT starts at a view-independent prior; this deliberately tests
        # a nondegenerate trained readout, rather than demanding initial pair
        # gradients from zero-initialized factor outputs.
        with torch.no_grad():
            torch.nn.init.normal_(net.pressure_readout.output.weight, std=.2)
            torch.nn.init.normal_(net.pressure_readout.total.weight, std=.2)
        v, raw, phys = inputs()
        out = net(v, raw, phys, torch.full((2,), CONFIG_V))
        pair = pressure_pair_loss(out["pressure_hat"].reshape(1, 2, 4, 96))["loss"]
        supervised = pressure_supervision(out["pressure_hat"], torch.zeros(2, 4, 96))["loss"]
        groups = e3_parameter_groups(net)
        admitted = {id(p) for p in groups["HF_motion"]}
        self.assertEqual(len(admitted), len(groups["HF_motion"]))
        self.assertNotIn(id(net.embeddings.modality.weight), admitted)
        parameters = list(net.parameters())
        base = torch.autograd.grad(supervised, parameters, retain_graph=True, allow_unused=True)
        expected_pair = torch.autograd.grad(pair, groups["HF_motion"], retain_graph=True, allow_unused=True)
        self.assertGreater(sum(g.abs().sum().item() for g in expected_pair if g is not None), 0.)
        backward_e3_losses(supervised, pair, net)
        extras = dict(zip((id(p) for p in groups["HF_motion"]), expected_pair))
        encoder_norm = 0.
        for parameter, gradient in zip(parameters, base):
            expected = gradient
            if id(parameter) in extras and extras[id(parameter)] is not None:
                expected = extras[id(parameter)] if expected is None else expected + extras[id(parameter)]
            if expected is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter in net.encoders.v_enc.parameters():
            if parameter.grad is not None:
                encoder_norm += parameter.grad.abs().sum().item()
        self.assertGreater(encoder_norm, 0.)

    def test_bce_effective_batch_weight_override_guard(self):
        net = model()
        v, raw, phys = inputs()
        out = net(v, raw, phys, torch.zeros(2, dtype=torch.long))
        batch = {"pose_gt": out["x0_hat"].detach(), "vel_gt": out["v_hat"].detach(),
                 "contact_gt": torch.zeros(2, 4, 2)}
        for invalid in (-1., float("nan"), [1., 2.]):
            with self.assertRaisesRegex(ValueError, "finite nonnegative scalar"):
                compute_losses_v7(out, batch, torch.zeros(2, dtype=torch.long),
                                  {"traj_deltas": [1], "contact_pos_weight": invalid})
        a = compute_losses_v7(out, batch, torch.zeros(2, dtype=torch.long),
                             {"traj_deltas": [1], "contact_pos_weight": 2.})
        self.assertTrue(torch.isfinite(a["loss"]))

    def test_bce_air_weight_normalizer_preserves_mixed_microbatch_objective(self):
        prediction = {"x0_hat": torch.zeros(2, 4, 144), "v_hat": torch.zeros(2, 4, 3),
                      "trans_hat": torch.zeros(2, 4, 3),
                      "contact_logits": torch.tensor([.8, -.4])[:, None, None].expand(2, 4, 2).clone().requires_grad_()}
        truth = {"pose_gt": prediction["x0_hat"], "vel_gt": prediction["v_hat"],
                 "contact_gt": torch.tensor([1., 0.])[:, None, None].expand(2, 4, 2)}
        config = {"traj_deltas": [1], "lambda_air": 2., "lambda_contact": 1.,
                  "lambda_pose": 0., "lambda_traj": 0., "contact_pos_weight": 1.,
                  "contact_weight_mean": 2.}
        full = compute_losses_v7(prediction, truth, torch.zeros(2, dtype=torch.long), config)["loss"]
        micro = sum(compute_losses_v7({k: v[i:i+1] for k,v in prediction.items()},
                    {k: v[i:i+1] for k,v in truth.items()}, torch.zeros(1, dtype=torch.long),
                    config)["loss"] / 2 for i in range(2))
        # Different reduction grouping can round one float32 addition.
        torch.testing.assert_close(full, micro, rtol=1e-6, atol=1e-7)
        full_gradient = torch.autograd.grad(full, prediction["contact_logits"], retain_graph=True)[0]
        micro_gradient = torch.autograd.grad(micro, prediction["contact_logits"])[0]
        torch.testing.assert_close(full_gradient, micro_gradient, rtol=0, atol=0)
        for invalid in (0., -1., float("nan"), [1., 2.]):
            with self.assertRaisesRegex(ValueError, "finite positive scalar"):
                compute_losses_v7(prediction, truth, torch.zeros(2, dtype=torch.long),
                                  {**config, "contact_weight_mean": invalid})


if __name__ == "__main__":
    unittest.main()
