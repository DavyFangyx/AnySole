"""V3_Bv1_IMU (raw108_imu) structural tests.

Run with the torch env:
  /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m unittest \
      discover -s tests -p "test_v3b_imu.py" -v
"""
import unittest

import numpy as np
import torch

from anysole.data.tactile_s2m import build_t_s2m
from anysole.models.common.model_v2 import AnySoleModelV2


def make_model(tactile_input):
    return AnySoleModelV2(
        d=64, tw=4, nhead=4, dropout=0.0, pose_layers=1,
        t_encoder="foot_conv", tactile_input=tactile_input,
        tactile_direct=False, no_imu=False, v_input="hrnet",
        f2_repr=False, pose_parts=9, soft_parts=True, gate="sigma",
    )


class TestV3BIMU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = make_model("raw108").eval()
        torch.manual_seed(0)
        self.imu = make_model("raw108_imu").eval()
        missing, unexpected = self.imu.load_state_dict(self.base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith("encoders.imu_") for k in missing))
        self.b, self.tw = 2, 4
        self.v = torch.randn(self.b, self.tw, 2051)
        self.tr = torch.randn(self.b, self.tw, 96)
        self.tp = torch.randn(self.b, self.tw, 12)
        self.ts = torch.randn(self.b, self.tw, 50)

    def _forward(self, model, config_id, ts=None):
        cid = torch.full((self.b,), config_id, dtype=torch.long)
        with torch.no_grad():
            return model(self.v, self.tr, self.tp, cid,
                         T_s2m=self.ts if ts is None else ts)

    def test_concat_projection_is_identity_at_init(self):
        w = self.imu.encoders.imu_fuse.proj.weight
        self.assertTrue(torch.equal(w[:, :64], torch.eye(64)))
        self.assertEqual(float(w[:, 64:].abs().max()), 0.0)
        self.assertEqual(float(self.imu.encoders.imu_fuse.proj.bias.abs().max()), 0.0)

    def test_step0_equals_base(self):
        for config_id in (0, 1):  # 0 = VT, 1 = V (CONFIG_V)
            a = self._forward(self.base, config_id)
            b = self._forward(self.imu, config_id)
            self.assertTrue(torch.equal(a["x0_hat"], b["x0_hat"]))
            self.assertTrue(torch.equal(a["trans_hat"], b["trans_hat"]))

    def test_imu_changes_vt_but_not_v_only(self):
        with torch.no_grad():
            self.imu.encoders.imu_fuse.proj.weight[:, 64:].normal_(0.0, 0.05)
        a = self._forward(self.base, 0)
        b = self._forward(self.imu, 0)
        self.assertGreater(float((a["x0_hat"] - b["x0_hat"]).abs().max()), 0.0)
        # V-only replaces the whole T stream with the null token after the
        # concat, so the IMU path must not leak into V2M.
        a_v = self._forward(self.base, 1)
        b_v = self._forward(self.imu, 1)
        self.assertTrue(torch.equal(a_v["x0_hat"], b_v["x0_hat"]))

    def test_imu_gradient_reaches_encoder(self):
        self.imu.train()
        cid = torch.zeros(self.b, dtype=torch.long)

        def loss_once():
            out = self.imu(self.v, self.tr, self.tp, cid, T_s2m=self.ts)
            return out["x0_hat"].square().mean() + out["trans_hat"].square().mean()

        # Step 0: the concat projection's IMU half is zero, so the gradient
        # lands on that half but not yet on the IMU encoder.  One optimizer
        # step opens the path; the second backward must reach the encoder.
        loss_once().backward()
        fuse = self.imu.encoders.imu_fuse.proj.weight.grad
        self.assertGreater(float(fuse[:, 64:].abs().sum()), 0.0)
        with torch.no_grad():
            self.imu.encoders.imu_fuse.proj.weight[:, 64:] -= 0.1 * fuse[:, 64:]
        self.imu.zero_grad(set_to_none=True)
        loss_once().backward()
        grad = sum(p.grad.abs().sum().item()
                   for p in self.imu.encoders.imu_enc.parameters()
                   if p.grad is not None)
        self.assertGreater(grad, 0.0)


class TestTactileS2MIMUColumns(unittest.TestCase):
    def test_facts_columns_land_in_expected_slots(self):
        n = 3

        def grid(offset):
            return (np.arange(n * 3, dtype=np.float32) + offset).reshape(n, 3)

        facts = {
            "left_acc_g": grid(0.0),
            "left_gyro_rad_s": grid(100.0),
            "right_acc_g": grid(200.0),
            "right_gyro_rad_s": grid(300.0),
        }
        t = build_t_s2m(np.zeros((n, 96), np.float32), imu_facts=facts)
        self.assertEqual(t.shape, (n, 50))
        # Right foot has no Y flip: raw facts land verbatim in 41:47.
        np.testing.assert_allclose(t[:, 41:44], facts["right_acc_g"])
        np.testing.assert_allclose(t[:, 44:47], facts["right_gyro_rad_s"])
        # Left foot keeps the legacy Y flip; X/Z pass through.
        np.testing.assert_allclose(t[:, 16], facts["left_acc_g"][:, 0])
        np.testing.assert_allclose(t[:, 18], facts["left_acc_g"][:, 2])
        np.testing.assert_allclose(t[:, 17], -facts["left_acc_g"][:, 1])
        np.testing.assert_allclose(t[:, 19], facts["left_gyro_rad_s"][:, 0])
        np.testing.assert_allclose(t[:, 21], facts["left_gyro_rad_s"][:, 2])


if __name__ == "__main__":
    unittest.main()
