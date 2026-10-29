"""Guard current-source labels against motion changes and historical reuse."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from anysole.data import contact_package as p, workspace_adapter as w, contact_adapter as adapter

class Smpl24ContactReleaseTests(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.base=Path(tmp.name);self.root=self.base/'package';self.root.mkdir();self.sessiondir=self.base/'session';self.sessiondir.mkdir()
  frames=self.sessiondir/'frames.npz';frames.write_bytes(b'unchanged frame grid')
  self.bvh=self.base/'motion.bvh';self.bvh.write_bytes(b'current BVH')
  self.smpl=self.base/'motion.npz';self.smpl.write_bytes(b'current SMPL')
  ids=np.arange(4);np.savez(self.root/'labels.npz',frame_id=ids,contact=np.ones((4,2),np.float32))
  row={'source_files':{'frames.npz':p.sha256(frames)},'bvh_path':str(self.bvh),'bvh_sha256':p.sha256(self.bvh),'bvh_resolved_path':str(self.bvh.resolve()),'smpl_path':str(self.smpl),'smpl_sha256':p.sha256(self.smpl),'smpl_resolved_path':str(self.smpl.resolve()),'label_file':'labels.npz','label_sha256':p.sha256(self.root/'labels.npz')}
  manifest={'schema':'contact_label_package.v2','method':p.SMPL24_STATE_CONTACT_METHOD,'sessions':{'demo':row}}
  path=self.root/'label_package.json';path.write_text(json.dumps(manifest))
  for obj,name,value in [(p,'SMPL24_STATE_PACKAGE_SHA256',p.sha256(path)),(w,'shared_bvh_path',lambda session:self.bvh),(w,'shared_smpl_path',lambda session:self.smpl)]:
   patcher=patch.object(obj,name,value);patcher.start();self.addCleanup(patcher.stop)
  self.session={'session_id':'demo','dir':self.sessiondir,'frames':{'frame_id':ids}}
 def reader(self):return p.ContactLabelPackage(self.root,method=p.SMPL24_STATE_CONTACT_METHOD)
 def test_current_sources_load(self):self.assertEqual(self.reader().load(self.session).shape,(4,2))
 def test_smpl_content_change_rejected(self):
  self.smpl.write_bytes(b'different SMPL')
  with self.assertRaisesRegex(p.ContactPackageError,'SMPL'):self.reader().load(self.session)
 def test_active_source_repoint_even_with_identical_bytes_rejected(self):
  other=self.base/'other.npz';other.write_bytes(self.smpl.read_bytes())
  with patch.object(w,'shared_smpl_path',return_value=other):
   with self.assertRaisesRegex(p.ContactPackageError,'SMPL'):self.reader().load(self.session)
 def test_bvh_content_change_rejected(self):
  self.bvh.write_bytes(b'different BVH')
  with self.assertRaisesRegex(p.ContactPackageError,'BVH'):self.reader().load(self.session)
 def test_same_frame_count_changed_grid_rejected(self):
  (self.sessiondir/'frames.npz').write_bytes(b'a different frame grid')
  with self.assertRaisesRegex(p.ContactPackageError,'changed dependency'):self.reader().load(self.session)
 def test_new_package_cannot_load_as_historical_method(self):
  with self.assertRaises(p.ContactPackageError):p.ContactLabelPackage(self.root,method=p.STATE_CONTACT_METHOD)
 def test_registry_current_screen_has_seven_unified_models(self):
  from configs.z_gen.model_registry import MODELS
  from configs.z_gen.common import EXPERIMENTS
  ids=EXPERIMENTS['v7_statefix_smpl24_train']['MODEL_IDS'];self.assertEqual(len(ids),7)
  for i,key in enumerate(ids,1):
   m=MODELS[key];self.assertEqual(m['name'],'V7_Cv%d'%i);self.assertEqual(m['which'],'last');self.assertEqual(m['contact_method'],p.SMPL24_STATE_CONTACT_METHOD);self.assertEqual(m['eval_splits'],['val']);self.assertEqual(m['train_args']['smpl_support_policy'],'exclude')
 def test_cli_default_option(self):
  with patch.object(adapter,'build_one',return_value=[]) as make:
   adapter.main(['--session','demo']);self.assertIsNone(make.call_args.kwargs['label_method'])
 def test_cli_explicit_option(self):
  with patch.object(adapter,'build_one',return_value=[]) as make:
   adapter.main(['--session','demo','--label-method','probe']);self.assertEqual(make.call_args.kwargs['label_method'],'probe')

if __name__=='__main__':unittest.main()
