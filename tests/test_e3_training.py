"""Grouped update equivalence and exact continuation, not configuration snapshots."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from anysole.models.common.model_v7 import AnySoleModelV7
from anysole.train import (backward_e3_grouped, e3_atomic_checkpoint, e3_epoch_indices,
                           e3_restore_rng, e3_rng_state, validate_e3_config)
from anysole.types import JOINT_PARENTS


def fixture():
    torch.manual_seed(617)
    net = AnySoleModelV7(d=16, tw=4, dropout=0., pose_layers=1,
                         reliability_route=True, task_routes=True,
                         pressure_head="query_factorized", pressure_cell_mean=[.1]*96).eval()
    with torch.no_grad():
        # Escape QFACT's intentionally view-independent prior initialization.
        for name, parameter in net.pressure_readout.named_parameters():
            if "out" in name:
                parameter.add_(torch.randn_like(parameter) * .01)
    physical = {"pose_gt": torch.tensor([1.,0.,0.,0.,1.,0.]).repeat(4,4,24),
                "T_raw": torch.rand(4,4,96), "T_phys": torch.rand(4,4,12),
                "V_feat": torch.zeros(4,4,2051), "contact_gt": torch.rand(4,4,2),
                "trans_anchor": torch.zeros(4,3), "trans_gt": torch.zeros(4,4,3),
                "vel_gt": torch.zeros(4,4,3), "offsets": torch.full((4,24,3),.03),
                "parents": torch.tensor(JOINT_PARENTS).repeat(4,1), "floor_y": torch.zeros(4)}
    views = [{**physical,"V_feat":torch.randn(4,4,2051)} for _ in range(4)]
    config = {"recipe":"UP_G1", "phase_id":"G1_original", "v7":True,
              "config_probs":[.5,.25,.25], "lambda_contact":1., "lambda_air":2.,
              "lambda_pressure":1., "lambda_pressure_pair":1., "lambda_contact_geometry":1.,
              "support_offsets":[[0.,0.,0.]]*2, "traj_deltas":[2],
              "contact_pos_weight":float(((1-physical["contact_gt"].mean())/physical["contact_gt"].mean()).clamp(.05,20)),
              "contact_weight_mean":float((1+2*(1-physical["contact_gt"])).mean())}
    return net, {"physical":physical,"views":views}, config


def sliced(group,start,end):
    return {"physical":{k:v[start:end] for k,v in group["physical"].items()},
            "views":[{k:v[start:end] for k,v in view.items()} for view in group["views"]]}


class E3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_microbatch_and_whole_batch_update_match(self):
        full, group, config = fixture()
        split = copy.deepcopy(full)
        scaler = torch.amp.GradScaler("cuda",enabled=False)
        backward_e3_grouped(full,group,config,scaler)
        for start in (0,2):
            backward_e3_grouped(split,sliced(group,start,start+2),config,scaler,.5)
        for (name,a),(_,b) in zip(full.named_parameters(),split.named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-6,msg=name)
        for net in (full,split):
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            torch.optim.SGD(net.parameters(),lr=1e-3).step()
        for a,b in zip(full.parameters(),split.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

    def test_optimizer_rng_and_sample_position_resume(self):
        net,group,config=fixture()
        optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        backward_e3_grouped(net,group,config,scaler)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        random.seed(2); np.random.seed(3)
        with tempfile.TemporaryDirectory(dir="/private/Tactile/.ai/tmp") as directory:
            path=Path(directory)/"resume.pt"
            e3_atomic_checkpoint(path,{"model":net.state_dict(),"optimizer":optimizer.state_dict(),
                                      "rng":e3_rng_state(),"sample_cursor":4})
            saved=torch.load(path,weights_only=False)
        expected=(random.random(),float(np.random.rand()),torch.rand(3))
        backward_e3_grouped(net,group,config,scaler); optimizer.step()
        resumed,_,_=fixture()
        resumed.load_state_dict(saved["model"],strict=True)
        opt2=torch.optim.Adam(resumed.parameters(),lr=1e-4); opt2.load_state_dict(saved["optimizer"])
        e3_restore_rng(saved["rng"])
        actual=(random.random(),float(np.random.rand()),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]); self.assertTrue(torch.equal(expected[2],actual[2]))
        backward_e3_grouped(resumed,group,config,scaler); opt2.step()
        for a,b in zip(net.parameters(),resumed.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(e3_epoch_indices(17,1,4)[saved["sample_cursor"]:],e3_epoch_indices(17,1,4)[4:])

    def test_amp_teacher_does_not_detach_shared_encoders(self):
        net,group,config=fixture()
        net.train()
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            backward_e3_grouped(net,group,config,scaler)
        parameters=dict(net.named_parameters())
        for name in ("encoders.v_enc.proj.weight","encoders.t_enc.conv.0.weight",
                     "v_temporal.layers.0.self_attn.in_proj_weight"):
            gradient=parameters[name].grad
            self.assertIsNotNone(gradient,name)
            self.assertTrue(torch.isfinite(gradient).all(),name)
            self.assertGreater(float(gradient.norm()),0.,name)

    def test_contract_rejects_up_tta_missing_geometry_and_online(self):
        _,_,config=fixture()
        config.update(condition_sampling="enum",final_eval_split="none",wandb_mode="offline",
                      pressure_head="query_factorized",inference_mode="direct",batch_size=4,
                      microbatch_size=2,epochs=2,tw=4)
        validate_e3_config(config)
        for change in ({"inference_mode":"tta"},{"lambda_contact_geometry":0.},
                       {"wandb_mode":"online"},{"recipe":"UQ"},{"lambda_foot":1.}):
            with self.assertRaises(ValueError): validate_e3_config({**config,**change})


if __name__ == "__main__":
    unittest.main()
