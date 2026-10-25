"""V5 input/loss availability: gradients, native training integration, legacy.
Run: python -m unittest discover -s tests -p test_v5_conditioning.py
"""
import ast,copy,tempfile,unittest
from pathlib import Path
from unittest import mock
import torch
from anysole.types import CONFIG_V,CONFIG_T,CONFIG_VT
from anysole.utils.losses import compute_losses_v5_1
from anysole.utils.v5_conditioning import (availability_masks,apply_availability,
    conditioning_policy,require_fresh_output)

REPO=Path(__file__).resolve().parents[1]

def fixture(n=3):
    torch.manual_seed(17);tw=5
    batch={'pose_gt':torch.zeros(n,tw,12),'vel_gt':torch.zeros(n,tw,3),
        'contact_gt':torch.full((n,tw,2),.3),'phase_gt':torch.zeros(n,tw,2,2),
        'phase_mask':torch.ones(n,tw,2,dtype=torch.bool),'cop_gt':torch.zeros(n,tw,2,2),
        'cop_mask':torch.ones(n,tw,2,dtype=torch.bool),'T_phys':torch.ones(n,tw,12),
        'strike_gt':torch.zeros(n,tw,2,dtype=torch.long),'strike_mask':torch.ones(n,tw,2,dtype=torch.bool)}
    def leaf(shape):return (torch.randn(*shape)*.1+.7).requires_grad_()
    out={k:leaf((n,tw,12)) for k in ('x0_hat','x0_v','x0_prior')}
    out.update({k:leaf((n,tw,3)) for k in ('v_hat','trans_hat','v_hat_v','trans_v','v_prior','trans_prior')})
    out.update({k:leaf((n,tw,4)) for k in ('tcop','gtcop')})
    out['strike_logits']=leaf((n,tw,6));out['contact_logits']=leaf((n,tw,2))
    out['z']={k:leaf((n,tw,2)) for k in ('logit_gV','logit_gT','logit_poe')}
    out['z'].update({k:leaf((n,tw,8)) for k in ('mu_gV','mu_gT')})
    weights={'v5_conditioning':'availability_v1','lambda_pose':3.,'lambda_traj':1.,
        'lambda_prior_pose':3.,'lambda_prior_traj':1.,'lambda_contact':1.,'lambda_phase':1.,'lambda_cop':1.,
        'lambda_tcop':1.,'lambda_strike':1.,'lambda_gtcop':1.,'lambda_cond':2.,'traj_deltas':[1,2]}
    return out,batch,weights

def normgrad(x,row=None):
    if x.grad is None:return 0.
    return float(torch.linalg.vector_norm(x.grad if row is None else x.grad[row]))

class AvailabilityTests(unittest.TestCase):
    def test_training_cli_records_the_explicit_recipe(self):
        from anysole.train import parse_args
        self.assertIsNone(parse_args([]).v5_conditioning)
        self.assertEqual(parse_args(['--v5-conditioning','availability_v1']).v5_conditioning,'availability_v1')

    def test_conditions_and_all_structured_dropouts(self):
        cid=torch.tensor([CONFIG_V,CONFIG_T,CONFIG_VT,CONFIG_VT,CONFIG_VT,CONFIG_V])
        b={'occ_v_zeroed':torch.tensor([0,0,0,1,0,0]),'occ_t_zeroed':torch.tensor([0,0,0,0,1,0])}
        inputs=tuple(torch.ones(6,5,d) for d in (7,8,12,50))
        actual,m=apply_availability(inputs,cid,b,torch.tensor([0,0,0,0,0,1]))
        self.assertEqual(m['v_available'].tolist(),[True,False,True,False,True,False])
        self.assertEqual(m['t_available'].tolist(),[False,True,True,True,False,False])
        for x,mask in zip(actual,(m['v_available'],m['t_available'],m['t_available'],m['t_available'])):
            self.assertTrue(torch.equal(x.sum((1,2))>0,mask))
        self.assertTrue(all(torch.all(x==1) for x in inputs))

    def test_observed_zero_pressure_is_not_a_missing_modality(self):
        cid=torch.tensor([CONFIG_VT]);inputs=(torch.ones(1,5,7),)+tuple(torch.zeros(1,5,d) for d in (8,12,50))
        _,m=apply_availability(inputs,cid,{},torch.tensor([False]))
        self.assertTrue(bool(m['t_available'][0]))

    def test_bad_masks_and_policies_fail(self):
        cid=torch.tensor([CONFIG_VT])
        for bad in (torch.tensor([2]),torch.tensor([.5]),torch.tensor([[0]])):
            with self.assertRaises(ValueError):availability_masks(cid,{'occ_t_zeroed':bad})
        with self.assertRaises(ValueError):availability_masks(torch.tensor([999]))
        with self.assertRaises(ValueError):conditioning_policy({'v5_conditioning':'typo'})
        self.assertEqual(conditioning_policy({}),'legacy')

    def test_fresh_output_guard_preserves_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';require_fresh_output(p);self.assertFalse(p.exists())
            p.mkdir();require_fresh_output(p);(p/'ckpt_last.pt').write_text('old')
            with self.assertRaises(FileExistsError):require_fresh_output(p)
            self.assertEqual((p/'ckpt_last.pt').read_text(),'old')

    def test_native_training_block_gates_the_same_inputs_and_masks(self):
        tree=ast.parse((REPO/'anysole/train.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name)
            and n.test.id=='v5_1_mode' and any(isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='v_drop' for t in c.targets) for c in n.body))
        wrapped=ast.Module(body=[ast.For(target=ast.Name(id='_',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),body=[node],orelse=[])],type_ignores=[])
        code=compile(ast.fix_missing_locations(wrapped),'native_v5_training_block','exec')
        cid=torch.tensor([CONFIG_V,CONFIG_T,CONFIG_VT,CONFIG_VT,CONFIG_VT,CONFIG_V])
        inputs=tuple(torch.ones(6,5,d) for d in (7,8,12,50));batch={'occ_t_zeroed':torch.tensor([0,0,0,0,1,0])}
        ns={'torch':torch,'config':{'v_cond_dropout':.5},'v5_1_mode':True,'v5_availability':True,
            'batch_size':6,'device':'cpu','batch':batch,'config_id':cid,'v_kw':{},'apply_availability':apply_availability,
            'availability_counts':{k:0 for k in ('windows','V','T','VT','neither')},
            **dict(zip(('v_feat','t_raw','t_phys','t_s2m'),inputs))}
        with mock.patch('torch.rand',return_value=torch.tensor([.9,.9,.9,.1,.9,.1])):exec(code,ns)
        self.assertEqual(batch['v_available'].tolist(),[True,False,True,False,True,False])
        self.assertEqual(batch['t_available'].tolist(),[False,True,True,True,False,False])
        self.assertEqual(ns['availability_counts']['neither'],1)
        self.assertTrue(torch.equal(ns['t_raw'].sum((1,2))>0,batch['t_available']))

class LossTests(unittest.TestCase):
    def test_gradients_follow_all_three_conditions_and_extra_dropouts(self):
        o,b,w=fixture(6);cid=torch.tensor([CONFIG_V,CONFIG_T,CONFIG_VT,CONFIG_VT,CONFIG_VT,CONFIG_V])
        b['v_zero']=torch.tensor([0,0,0,1,0,1]);b['occ_t_zeroed']=torch.tensor([0,0,0,0,1,0])
        losses=compute_losses_v5_1(o,b,cid,w,cond_tau=.5);losses['loss'].backward()
        expected={
            'x0_v':[1,0,1,0,1,0],'x0_hat':[0,0,1,0,0,0],'x0_prior':[0,1,1,1,0,0],
            'v_hat_v':[1,0,0,0,1,0],'v_hat':[0,0,1,0,0,0],
            'v_prior':[0,1,1,1,0,0],'contact_logits':[0,1,1,1,0,0],
            'tcop':[0,1,1,1,0,0],'gtcop':[0,1,1,1,0,0],'strike_logits':[0,1,1,1,0,0]}
        for k,mask in expected.items():
            self.assertEqual([int(normgrad(o[k],i)>0) for i in range(6)],mask,k)
        for k,mask in [('mu_gV',[1,0,1,0,1,0]),('mu_gT',[0,1,1,1,0,0]),
                       ('logit_gV',[1,0,1,0,1,0]),('logit_gT',[0,1,1,1,0,0]),('logit_poe',[0,0,1,0,0,0])]:
            self.assertEqual([int(normgrad(o['z'][k],i)>0) for i in range(6)],mask,k)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))

    def test_all_observed_VT_matches_legacy_loss_and_gradients(self):
        a,b,w=fixture();c=copy.deepcopy(a);cid=torch.full((3,),CONFIG_VT)
        old=compute_losses_v5_1(a,b,cid,dict(w,v5_conditioning='legacy'),cond_tau=.5)
        new=compute_losses_v5_1(c,b,cid,w,cond_tau=.5)
        self.assertEqual(old.keys(),new.keys())
        for k in old:self.assertTrue(torch.equal(old[k],new[k]),k)
        old['loss'].backward();new['loss'].backward()
        for k in a:
            if isinstance(a[k],dict):
                for z in a[k]:self.assertTrue(torch.equal(a[k][z].grad,c[k][z].grad),z)
            elif a[k].grad is not None:self.assertTrue(torch.equal(a[k].grad,c[k].grad),k)

    def test_missing_T_has_no_T_or_fused_supervision(self):
        o,b,w=fixture();cid=torch.full((3,),CONFIG_V)
        loss=compute_losses_v5_1(o,b,cid,w,cond_tau=.5)
        for k in ('L_prior_pose','L_prior_traj','L_tcop','L_strike','L_gtcop','L_cond'):
            self.assertEqual(float(loss[k]),0.,k)
        loss['loss'].backward()
        for k in ('x0_prior','v_prior','trans_prior','x0_hat','v_hat','trans_hat','contact_logits'):
            self.assertEqual(normgrad(o[k]),0.,k)
        self.assertGreater(normgrad(o['v_hat_v']),0.)
        self.assertGreater(normgrad(o['trans_v']),0.)

    def test_missing_V_has_no_visual_or_fused_supervision(self):
        o,b,w=fixture();cid=torch.full((3,),CONFIG_T)
        loss=compute_losses_v5_1(o,b,cid,w,cond_tau=.5)
        self.assertEqual(float(loss['L_pose']),0.);self.assertEqual(float(loss['L_traj']),0.)
        loss['loss'].backward()
        for k in ('x0_v','x0_hat','v_hat','trans_hat','v_hat_v','trans_v'):
            self.assertEqual(normgrad(o[k]),0.,k)
        self.assertGreater(normgrad(o['x0_prior']),0.)
        self.assertGreater(normgrad(o['v_prior']),0.)

    def test_all_missing_zero_loss_and_gradients(self):
        o,b,w=fixture();cid=torch.full((3,),CONFIG_VT)
        b.update(v_zero=torch.ones(3,dtype=torch.bool),t_zero=torch.ones(3,dtype=torch.bool))
        loss=compute_losses_v5_1(o,b,cid,w,cond_tau=.5)
        self.assertTrue(all(float(v)==0 for v in loss.values()))
        loss['loss'].backward()
        for k in o:
            for value in (o[k].values() if isinstance(o[k],dict) else [o[k]]):self.assertEqual(normgrad(value),0.)

    def test_unavailable_contact_labels_do_not_change_visual_class_weight(self):
        o,b,w=fixture();p=copy.deepcopy(o);c=copy.deepcopy(b);cid=torch.tensor([CONFIG_VT,CONFIG_V,CONFIG_T])
        c['contact_gt'][2]=.95
        la=compute_losses_v5_1(o,b,cid,w)['L_contact'];lb=compute_losses_v5_1(p,c,cid,w)['L_contact']
        ga=torch.autograd.grad(la,o['z']['logit_gV'])[0];gb=torch.autograd.grad(lb,p['z']['logit_gV'])[0]
        self.assertTrue(torch.equal(ga,gb))

    def test_missing_visual_trajectory_readout_is_rejected(self):
        o,b,w=fixture();o['v_hat_v']=None
        with self.assertRaises(ValueError):compute_losses_v5_1(o,b,torch.full((3,),CONFIG_V),w)

if __name__=='__main__':unittest.main()
