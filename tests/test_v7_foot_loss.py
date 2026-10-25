"""V7 B: registered weight, total-loss contribution and root-motion gradients."""
import unittest

import torch

from anysole.registry import get_builder
from anysole.train import load_config, parse_args
from anysole.types import CONFIG_V, FPS, JOINT_PARENTS
from anysole.utils.losses import compute_losses
from configs.tools.common import config_path, load_registry
from configs.tools.runner import _train_command


def training_config(model):
    registry = load_registry()
    args = parse_args(_train_command(registry, model)[3:])
    config = load_config(config_path(registry, model))
    entry = get_builder(model)
    config.update(entry.STRUCTURE)
    config.update(entry.CONFIG_EXTRA)
    for key, value in vars(args).items():
        if key.startswith("lambda_") and value is not None:
            config[key] = value
    return config


def isolated_motion():
    # Synthetic control: constant orientation and 0.5 m/frame translation.
    # Identical outputs/targets make the A/B difference isolate the foot term.
    pose = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(1, 4, 24).requires_grad_()
    velocity = torch.zeros(1, 4, 3)
    velocity[..., 0] = 20.
    velocity.requires_grad_()
    trans = velocity.cumsum(1) / FPS
    logits = torch.zeros(1, 4, 2, requires_grad=True)
    offsets = torch.full((1, 24, 3), .03)
    offsets[:, 0] = 0.
    out = {"x0_hat": pose, "v_hat": velocity, "trans_hat": trans,
           "contact_logits": logits, "z_root": torch.zeros(1, 4, 1, 16)}
    batch = {"pose_gt": pose.detach(), "vel_gt": velocity.detach(),
             "trans_gt": trans.detach(), "trans_anchor": torch.zeros(1, 3),
             "offsets": offsets, "parents": torch.tensor(JOINT_PARENTS).view(1, 24),
             "contact_gt": torch.tensor([[[1., 0.]] * 4])}
    return out, batch


class V7FootLossTests(unittest.TestCase):
    def setUp(self):
        self.a = training_config("V7_Av1")
        self.b = training_config("V7_Bv1")
        self.cid = torch.zeros(1, dtype=torch.long)

    def test_registered_recipe_records_the_weight_without_cli_aliasing(self):
        self.assertEqual(self.b.get("lambda_foot"), 1.)
        self.assertEqual(self.a.get("lambda_foot", 0.), 0.)
        registry = load_registry()
        args = parse_args(_train_command(registry, "V7_Bv1")[3:])
        self.assertIsNone(args.lambda_foot_res)

    def test_b_minus_a_equals_the_weighted_foot_loss(self):
        out, batch = isolated_motion()
        a = compute_losses(out, batch, self.cid, self.a)
        b = compute_losses(out, batch, self.cid, self.b)
        self.assertGreater(b["L_foot"].item(), 0.)
        delta = b["loss"] - a["loss"]
        torch.testing.assert_close(delta, self.b["lambda_foot"] * b["L_foot"])
        self.assertGreater(delta.item(), 0.)
        print("B raw_L_foot=%.6f total_B_minus_A=%.6f" % (b["L_foot"].item(), delta.item()))

    def test_weight_zero_and_fractional_weights_control_the_total(self):
        out, batch = isolated_motion()
        a = compute_losses(out, batch, self.cid, self.a)
        for weight in (0., .5, 1.):
            with self.subTest(weight=weight):
                b = compute_losses(out, batch, self.cid, {**self.b, "lambda_foot": weight})
                torch.testing.assert_close(b["loss"] - a["loss"], weight * b["L_foot"])

    def test_missing_weight_is_rejected_instead_of_silently_disabling_b(self):
        out, batch = isolated_motion()
        config = dict(self.b)
        config.pop("lambda_foot", None)
        with self.assertRaisesRegex(ValueError, "explicit lambda_foot"):
            compute_losses(out, batch, self.cid, config)

    def test_total_adds_root_gradient_while_contact_weights_stay_detached(self):
        out, batch = isolated_motion()
        a = compute_losses(out, batch, self.cid, self.a)
        b = compute_losses(out, batch, self.cid, self.b)
        ga = torch.autograd.grad(a["loss"], out["v_hat"], retain_graph=True)[0]
        gb = torch.autograd.grad(b["loss"], out["v_hat"], retain_graph=True)[0]
        gf = torch.autograd.grad(b["L_foot"], out["v_hat"], retain_graph=True)[0]
        contact = torch.autograd.grad(b["L_foot"], out["contact_logits"], allow_unused=True)[0]
        self.assertGreater((gb - ga).abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(gb).all())
        torch.testing.assert_close(gb - ga, self.b["lambda_foot"] * gf)
        self.assertIsNone(contact)
        print("B added_root_gradient=%.6f foot_contact_gradient=detached" % (gb - ga).abs().sum().item())

    def test_v_only_receives_no_foot_or_t_contact_supervision(self):
        out, batch = isolated_motion()
        cid = torch.full((1,), CONFIG_V)
        a = compute_losses(out, batch, cid, self.a)
        b = compute_losses(out, batch, cid, self.b)
        self.assertEqual(b["L_foot"].item(), 0.)
        self.assertEqual(b["L_contact"].item(), 0.)
        torch.testing.assert_close(b["loss"], a["loss"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
