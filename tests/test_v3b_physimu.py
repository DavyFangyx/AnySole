"""V3_Bv1_PhysIMU (raw96_phys_imu, fusion method 2) structural tests.

Run with the torch env:
  /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m unittest \
      discover -s tests -p "test_v3b_physimu.py" -v
"""
import unittest

import torch

from anysole.models.v3_3b_physimu import build


class TestPhysIMU(unittest.TestCase):
    def setUp(self):
        self.model = build(dict(d_model=64, tw=4, dropout=0.0, pose_layers=1)).eval()
        self.b, self.tw = 2, 4
        self.v = torch.randn(self.b, self.tw, 2051)
        self.tr = torch.randn(self.b, self.tw, 96)
        self.tp = torch.randn(self.b, self.tw, 12)
        self.ts = torch.randn(self.b, self.tw, 50)

    def _forward(self, config_id, tp=None, ts=None):
        cid = torch.full((self.b,), config_id, dtype=torch.long)
        with torch.no_grad():
            return self.model(self.v, self.tr,
                              self.tp if tp is None else tp, cid,
                              T_s2m=self.ts if ts is None else ts)

    def test_footconv_raw_only_accepts_96(self):
        m = self.model.encoders.t_enc
        self.assertTrue(m.raw_only)
        self.assertEqual(m.feat_mlp[0].in_features, 2)
        out = m(self.tr)
        self.assertEqual(tuple(out.shape), (self.b, self.tw, 64))

    def test_fusion_shapes(self):
        fuse = self.model.encoders.phys_imu_fuse
        self.assertEqual(tuple(fuse.phys_proj.weight.shape), (64, 128))
        self.assertEqual(tuple(fuse.out_proj.weight.shape), (64, 128))

    def test_forward_shapes(self):
        out = self._forward(0)
        self.assertEqual(tuple(out["x0_hat"].shape), (self.b, self.tw, 144))
        self.assertEqual(tuple(out["trans_hat"].shape), (self.b, self.tw, 3))

    def test_v_only_does_not_leak_imu_or_phys(self):
        a = self._forward(1)
        b = self._forward(1, tp=torch.randn_like(self.tp),
                          ts=torch.randn_like(self.ts))
        self.assertTrue(torch.equal(a["x0_hat"], b["x0_hat"]))
        self.assertTrue(torch.equal(a["trans_hat"], b["trans_hat"]))

    def test_imu_and_phys_change_vt(self):
        with torch.no_grad():
            self.model.encoders.phys_imu_enc.mlp[0].weight.normal_(0.0, 0.05)
            self.model.encoders.t_phys_enc.mlp[0].weight.normal_(0.0, 0.05)
            self.model.encoders.phys_imu_fuse.phys_proj.weight.normal_(0.0, 0.05)
        a = self._forward(0)
        b = self._forward(0, tp=torch.randn_like(self.tp),
                          ts=torch.randn_like(self.ts))
        self.assertGreater(float((a["x0_hat"] - b["x0_hat"]).abs().max()), 0.0)

    def test_gradients_reach_both_mlps(self):
        self.model.train()
        cid = torch.zeros(self.b, dtype=torch.long)
        out = self.model(self.v, self.tr, self.tp, cid, T_s2m=self.ts)
        loss = out["x0_hat"].square().mean() + out["trans_hat"].square().mean()
        loss.backward()
        for name in ("t_phys_enc", "phys_imu_enc"):
            module = getattr(self.model.encoders, name)
            grad = sum(p.grad.abs().sum().item()
                       for p in module.parameters() if p.grad is not None)
            self.assertGreater(grad, 0.0, name)


if __name__ == "__main__":
    unittest.main()
