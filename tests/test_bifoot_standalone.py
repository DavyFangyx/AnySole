"""Standalone BiFoot-CrossAttn instance tests (no S2M repo import).

Run with the torch env:
  /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m unittest \
      discover -s tests -p "test_bifoot_standalone.py" -v
"""
import os
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VT_DIR = os.path.join(ROOT, "results", "diagnostics",
                      "20261111_variant_sweep")
sys.path.insert(0, VT_DIR)

from bifoot_standalone import (  # noqa: E402
    BiFootCrossAttn, BiFootLength, BiFootDecoupled, Normalizer, TransformerTranslation, build_from_config,
    load_standalone, save_standalone)

CKPT = os.path.join(VT_DIR, "ckpt", "bifoot_standalone_fS5S6S11.pt")
CONFIG = dict(c_dim=50, input_T=100, output_dim=72, hidden_dim=256,
              n_hidden=1, n_heads=4, d_ff=512, dropout=0.1, d=256,
              heads=4, tac_dim=6)


class TestLocalHead(unittest.TestCase):
    def test_blank_head_shapes(self):
        m = TransformerTranslation(c_dim=50, input_T=100, output_dim=72,
                                   hidden_dim=256, n_hidden=1, n_heads=4,
                                   d_ff=512, dropout=0.1, is_prior=False)
        c = torch.zeros(2, 100, 50)
        y = m(torch.zeros(2, 100, 69), c)
        self.assertEqual(tuple(y.shape), (2, 100, 3))

    def test_bi_foot_zero_init_equals_s2m(self):
        s2m = TransformerTranslation(c_dim=50, input_T=100, output_dim=72,
                                     hidden_dim=256, n_hidden=1, n_heads=4,
                                     d_ff=512, dropout=0.1, is_prior=False)
        model = BiFootCrossAttn(s2m)
        s2m.eval(); model.eval()
        torch.manual_seed(0)
        c = torch.randn(2, 100, 50)
        tl = torch.randn(2, 100, 6)
        tr = torch.randn(2, 100, 6)
        with torch.no_grad():
            y_s2m = s2m(torch.zeros(2, 100, 69), c)
            y_bf = model(c, tl, tr)
        self.assertEqual(float((y_s2m - y_bf).abs().max()), 0.0)

    def test_state_keys_match_checkpoint(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        model = build_from_config(ck["config"])
        # strict load must succeed
        model.load_state_dict(ck["state"], strict=True)
        self.assertEqual(set(model.state_dict()), set(ck["state"]))

    def test_length_zero_init_and_channel_invariants(self):
        s2m = TransformerTranslation(c_dim=50,input_T=100,output_dim=72,
                                     hidden_dim=256,n_hidden=1,n_heads=4,
                                     d_ff=512,dropout=0.1,is_prior=False)
        model=BiFootLength(s2m).eval(); s2m.eval()
        c=torch.randn(2,100,50); tl=torch.randn(2,100,6); tr=torch.randn(2,100,6)
        with torch.no_grad():
            base=s2m(torch.zeros(2,100,69),c); y=model(c,tl,tr)
        self.assertEqual(float((base-y).abs().max()),0.0)
        with torch.no_grad(): model.log_scale.bias.fill_(0.4); y=model(c,tl,tr)
        self.assertTrue(torch.equal(base[...,1],y[...,1]))
        cross=base[...,0]*y[...,2]-base[...,2]*y[...,0]
        self.assertLess(float(cross.abs().max()),1e-6)
        dot=base[...,0]*y[...,0]+base[...,2]*y[...,2]
        self.assertTrue(bool((dot>=-1e-7).all()))

    def test_decoupled_zero_init_and_tactile_cannot_rotate(self):
        s2m=TransformerTranslation(c_dim=50,input_T=100,output_dim=72,
                                   hidden_dim=256,n_hidden=1,n_heads=4,
                                   d_ff=512,dropout=0.1,is_prior=False)
        model=BiFootDecoupled(s2m).eval();s2m.eval()
        c=torch.randn(2,100,50);t1=torch.randn(2,100,6);t2=torch.randn(2,100,6)
        with torch.no_grad():
            base=s2m(torch.zeros(2,100,69),c);y0=model(c,t1,t2)
        self.assertLess(float((base-y0).abs().max()),1e-7)
        with torch.no_grad():
            model.length_head[-1].weight.normal_(0,0.1)
            model.direction_head[-1].bias.fill_(0.2)
            y1=model(c,t1,t2);y2=model(c,-t1,-t2)
        cross=y1[...,0]*y2[...,2]-y1[...,2]*y2[...,0]
        dot=y1[...,0]*y2[...,0]+y1[...,2]*y2[...,2]
        self.assertLess(float(cross.abs().max()),1e-5)
        self.assertTrue(bool((dot>=-1e-7).all()))
        self.assertTrue(torch.equal(y1[...,1],y2[...,1]))

    def test_length_physical_scaling_with_nonzero_mean(self):
        cfg=dict(CONFIG,model_type='length',physical_scale=True)
        m=build_from_config(cfg).eval()
        mean=torch.tensor([0.004,0.001,-0.003])
        std=torch.tensor([0.002,0.007,0.011])
        m.set_output_stats(mean,std)
        c=torch.randn(1,100,50);tl=torch.randn(1,100,6);tr=torch.randn(1,100,6)
        with torch.no_grad():
            base=m.s2m(torch.zeros(1,100,69),c)
            self.assertTrue(torch.equal(base,m(c,tl,tr)))
            m.log_scale.bias.fill_(0.4)
            y=m(c,tl,tr)*std+mean
            expected=base*std+mean
            expected[..., [0,2]]*=torch.exp(torch.tensor(0.4))
        self.assertTrue(torch.allclose(y,expected,atol=1e-8,rtol=1e-6))
        self.assertTrue(torch.equal(y[...,1],expected[...,1]))
        norm=Normalizer(mean,std,torch.zeros(50),torch.ones(50))
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'physical.pt')
            save_standalone(path,m,norm,cfg)
            reloaded,_,_=load_standalone(path,'cpu')
            with torch.no_grad():
                self.assertTrue(torch.equal(m(c,tl,tr),reloaded(c,tl,tr)))


class TestStandaloneCheckpoint(unittest.TestCase):
    def test_strict_load_and_norm(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        model, norm, meta = load_standalone(CKPT, "cpu")
        self.assertEqual(model.training, False)
        self.assertEqual(tuple(model.s2m.input_emb[0].weight.shape), (256, 12))
        self.assertEqual(tuple(norm.insoles_mean.shape), (50,))
        self.assertTrue(torch.equal(norm.insoles_mean,
                                   ck["norm_state"]["insoles_mean"]))
        self.assertIn("fold", meta)

    def test_save_load_roundtrip(self):
        model, norm, meta = load_standalone(CKPT, "cpu")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.pt")
            save_standalone(p, model, norm, CONFIG, meta)
            m2, n2, meta2 = load_standalone(p, "cpu")
        self.assertEqual(set(model.state_dict()), set(m2.state_dict()))
        for k in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[k],
                                        m2.state_dict()[k]))
        self.assertTrue(torch.equal(norm.poses_mean, n2.poses_mean))
        self.assertEqual(meta, meta2)

    def test_strict_load_rejects_missing_key(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        state = dict(ck["state"])
        state.pop(next(iter(state)))
        model = build_from_config(ck["config"])
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)

    def test_normalizer_methods(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        norm = Normalizer.from_state(ck["norm_state"])
        x = torch.randn(4, 50)
        y = norm.normalize_insole(x)
        x2 = y * norm.insoles_std + norm.insoles_mean
        self.assertTrue(torch.allclose(x, x2, atol=1e-3))
        p = torch.randn(4, 72)
        q = norm.normalize_poses(p)
        p2 = norm.denormalize_poses(q)
        self.assertTrue(torch.allclose(p, p2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
