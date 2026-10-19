"""Bounded CPU protocol/entrypoint oracles, not evidence of learned pressure."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anysole.utils import eval_protocol as p
from anysole import infer,eval as evaluation,train
from anysole.types import JOINT_PARENTS
from anysole.utils.geometry import fk_pose6d


@pytest.fixture(autouse=True,scope='module')
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def window(ids=None,velocity=1.,angle=0.,recording='r',camera=1):
    ids=np.arange(5) if ids is None else np.asarray(ids)
    n=len(ids);c,s=np.cos(angle),np.sin(angle)
    rotation=np.tile(np.array([[c,-s,0],[s,c,0],[0,0,1.]]),(n,24,1,1))
    return {'recording_id':recording,'camera_id':camera,'subject':'S13','frame_id':ids,
        'time_s':ids/40.,'valid':np.ones(n,bool),'view_edges':np.ones(max(n-1,0),bool),
        'motion_edges':np.ones(max(n-1,0),bool),'pressure_edges':np.ones((max(n-1,0),96),bool),
        'source_frame_id':ids.copy(),'source_time_s':ids/40.,'reuse':np.zeros(n,bool),'time_residual_ms':np.zeros(n),
        'rotation':rotation,'pressure':np.ones((n,96))*.2,'velocity':np.tile([velocity,0,0],(n,1)),
        'motion_valid':np.ones(n,bool),'pressure_valid':np.ones((n,96),bool),
        'gt_joints':np.zeros((n,24,3)),'gt_pressure':np.ones((n,96))*.1,
        'gt_pose6d':np.tile([1.,0,0,0,1,0],(n,24)), 'gt_root':np.zeros((n,3)), 'gt_betas':np.ones(10)}


def test_overlap_so3_velocity_before_integration_and_single_axis():
    a=window(velocity=1.,angle=0.); b=window(velocity=3.,angle=np.pi/2)
    series=p.aggregate_visual_windows([a,b])[('r',1)]
    assert len(series['frame_id'])==5 and np.all(series['overlap_count']==2)
    np.testing.assert_allclose(series['trajectory'][:,0],np.arange(5)*.05)
    rot=series['rotation']
    np.testing.assert_allclose(rot@rot.swapaxes(-2,-1),np.broadcast_to(np.eye(3),rot.shape),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot),1.,atol=1e-12)
    np.testing.assert_allclose(rot[0,0,0,0],np.sqrt(.5),atol=1e-12)
    assert series['trajectory'][0].tolist()==[0,0,0]


@pytest.mark.parametrize('kind',['missing_target','invalid_rgb','reuse','source_gap','label_gap'])
def test_dynamic_segments_never_bridge_gaps(kind):
    w=window()
    if kind=='missing_target':
        w=window([0,1,3,4,5])
    elif kind=='invalid_rgb':
        w['valid'][2]=False; w['pressure'][2]=np.nan
    elif kind=='reuse':
        w['source_frame_id'][2]=w['source_frame_id'][1];w['reuse'][2]=True
    elif kind=='source_gap':
        w['source_time_s'][2:]+=.1
    else:
        w['motion_edges'][1]=False
    s=p.aggregate_visual_windows([w])[('r',1)]
    if kind=='label_gap':
        assert not s['motion_edges'][1]
    else:
        assert s['segment_id'][2]!=s['segment_id'][1]
        if s['valid'][2]:
            np.testing.assert_array_equal(s['trajectory'][2],0)
    assert len(s['frame_id'])==5


def test_duplicate_metadata_disagreement_rejected_and_reflection_corrected():
    a=window();b=window();b['gt_pressure'][0,0]=.9
    with pytest.raises(ValueError,match='disagree'):
        p.aggregate_visual_windows([a,b])
    reflected=np.diag([1.,1.,-1.])
    np.testing.assert_allclose(np.linalg.det(p.project_so3(reflected)),1)


def test_pressure_numeric_unclamped_and_true_temporal_negative_correlation():
    w=window();pattern=np.linspace(.01,.02,96)
    w['gt_pressure']=np.arange(1,6)[:,None]*pattern
    w['pressure']=np.arange(5,0,-1)[:,None]*pattern
    item=p.aggregate_visual_windows([w])[('r',1)]
    metrics=p.visual_pressure_metrics(item)
    assert metrics['pressure_spatial_corr']['value']==pytest.approx(1)
    assert metrics['left_temporal_corr']['value']==pytest.approx(-1)
    assert metrics['right_temporal_corr']['value']==pytest.approx(-1)
    assert metrics['left_peak_timing_abs_ms']['value']==pytest.approx(100)
    delta=w['pressure']-w['gt_pressure']
    assert metrics['pressure_mae']['value']==pytest.approx(np.abs(delta).mean())
    assert metrics['pressure_rmse']['value']==pytest.approx(np.sqrt(np.square(delta).mean()))
    assert metrics['left_low_load_fraction']['value'] is None
    assert 'train-fitted' in metrics['left_contact_f1']['reason']
    w['pressure'][:]=-2
    metrics=p.visual_pressure_metrics(p.aggregate_visual_windows([w])[('r',1)])
    assert metrics['pressure_out_of_range_fraction']['value']==1
    assert metrics['pressure_mae']['value']>2
    assert metrics['left_temporal_corr']['value'] is None


def test_pressure_frozen_contact_geometry_and_low_load():
    item=p.aggregate_visual_windows([window()])[('r',1)]
    policy={'pressure_eval':{'fit_split':'train','fitted_from_train_sha256':'fixture',
        'low_load_threshold_normalized':[5,5],'contact_threshold_normalized':[1,1],
        'sensor_xy_grid':np.tile(np.stack([np.arange(48),np.zeros(48)],-1),(2,1,1)).tolist()}}
    metrics=p.visual_pressure_metrics(item,policy)
    assert metrics['left_contact_f1']['value']==1
    assert metrics['left_cop_error_grid']['value']==pytest.approx(0,abs=1e-12)
    assert metrics['left_low_load_fraction']['value']==1
    policy['pressure_eval']['fit_split']='test'
    assert p.visual_pressure_metrics(item,policy)['left_contact_f1']['value'] is None


def test_constant_float32_gt_does_not_claim_missing_temporal_prediction():
    item=p.aggregate_visual_windows([window(np.arange(23))])[('r',1)]
    item['gt_pressure']=np.full((23,96),np.float32(.1),dtype=np.float32)
    item['pressure']=np.linspace(.02,.4,23)[:,None]*np.ones((1,96))
    item['gt_pressure'][19:]=np.arange(1,5,dtype=np.float32)[:,None]*.02
    item['edges'][18]=False
    # Identical float32 readings can have a nonzero float32 std due to
    # mean-rounding. They carry no defined temporal correlation target.
    constant=item['gt_pressure'][:19].reshape(-1,2,48).sum(-1)[:,0]
    assert constant.std()>1e-12 and constant.std(dtype=np.float64)==0
    metrics=p.visual_pressure_metrics(item)
    for foot in ('left','right'):
        value=metrics[f'{foot}_temporal_corr']
        assert value['eligible_count']==4
        assert value['count']==4
        assert value['missing_prediction_count']==0
        assert value['value']==pytest.approx(1.)


def test_constant_prediction_on_dynamic_gt_still_counts_as_missing():
    item=p.aggregate_visual_windows([window()])[('r',1)]
    item['gt_pressure']=np.arange(1,6,dtype=np.float32)[:,None]*np.ones((1,96),np.float32)*.01
    item['pressure']=np.full((5,96),.1)
    value=p.visual_pressure_metrics(item)['right_temporal_corr']
    assert value['value'] is None
    assert value['eligible_count']==5
    assert value['missing_prediction_count']==5


def test_beta_zero_selector_and_full_rows_share_formulas_and_coverage():
    w=window();offsets=torch.arange(72).reshape(24,3).float()/200;offsets[0]=0
    pose=torch.tensor(w['gt_pose6d'],dtype=torch.float32)
    w['gt_joints']=fk_pose6d(pose[None],torch.zeros(1,5,3),offsets,torch.tensor(JOINT_PARENTS))[0].numpy()
    series=p.aggregate_visual_windows([w]);a=p.visual_selection_metrics(series,offsets)
    b=p.visual_selection_metrics(series,offsets,prefix='eval')
    rows=p.visual_session_rows(series,offsets)
    assert a['val/coverage_hash']==b['eval/coverage_hash']
    assert a['val/mpjpe']<1e-7 and rows[0]['metrics']['mpjpe_mm']['value']<1e-4
    assert a['val/pressure_rmse']==pytest.approx(rows[0]['metrics']['pressure_rmse']['value'])
    altered=p.visual_selection_metrics(series,offsets*2)
    assert altered['val/mpjpe']>.1 and a['val/coverage_hash']==altered['val/coverage_hash']
    # GT beta is stored exclusively as evaluator truth; never selects pred offsets.
    w['gt_betas']*=100
    unchanged=p.visual_selection_metrics(p.aggregate_visual_windows([w]),offsets)
    assert unchanged==a
    assert rows[0]['metrics']['root_ate_mm']['value'] is None


def test_motion_temporal_counts_follow_whole_source_and_label_segments():
    w=window(np.arange(8));w['motion_edges'][2]=False
    item=p.aggregate_visual_windows([w])[('r',1)]
    offsets=torch.arange(72).reshape(24,3).float()/200;offsets[0]=0
    metrics=p.visual_motion_metrics(item,offsets)
    assert metrics['accel_error_m_s2']['count']==4 # 3-frame +5-frame =>1+3
    assert metrics['jitter_pred_m_s3']['count']==2
    assert metrics['w_mpjpe100_mm']['value'] is None


class RGBOracle(nn.Module):
    def forward_visual(self,features,valid,times):
        assert features.shape[1:]==(20,2051) and valid.dtype==torch.bool
        assert torch.all(torch.diff(times)>0)
        n,t=features.shape[:2]
        return {'x0_hat':features.new_tensor([1,0,0,0,1,0]).repeat(n,t,24),
                'pressure_hat':features[...,:1].expand(n,t,96),'v_hat':features.new_ones(n,t,3)}


def test_rgb_target_grid_is_not_irregular_source_axis_and_short_tail_padding():
    sources=[{'frame_id':i,'time_s':t,'file':f'{i}.png'} for i,t in enumerate([4.,4.025,4.1])]
    values=np.ones((3,2051),np.float32)
    series,matching=infer.visual_feature_windows(RGBOracle(),values,[True]*3,sources,{},torch.device('cpu'))
    item=series[('rgb_input',1)]
    np.testing.assert_allclose(item['time_s'],[4,4.025,4.05,4.075,4.1])
    assert len(matching['source_row'])==5 and item['valid'].sum()==5
    assert item['reuse'].any() and len(np.unique(item['segment_id']))>=3


def test_rgb_meta_and_reader_refuse_gt_access_and_unknown_fps(tmp_path,monkeypatch):
    import cv2
    directory=tmp_path/'rgb';directory.mkdir()
    for i in range(3):
        cv2.imwrite(str(directory/f'{i}.png'),np.zeros((8,8,3),np.uint8))
    metadata=tmp_path/'rgb.json'
    metadata.write_text(json.dumps({'schema_version':'anysole.rgb-only.v1','motion_path':'forbidden.npz'}))
    with pytest.raises(ValueError,match='forbidden'):
        infer.load_rgb_only_meta(metadata)
    metadata.write_text(json.dumps({'schema_version':'anysole.rgb-only.v1','source_time_s':[1,1.025,1.075]}))
    meta=infer.load_rgb_only_meta(metadata)
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert not any(word in str(path).lower() for word in ('motion','pressure','calibration','session.json'))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    rows=list(infer.iter_rgb_input(directory,meta))
    assert [r['time_s'] for r,rgb in rows]==[1,1.025,1.075]
    with pytest.raises(ValueError,match='assume-fps'):
        list(infer.iter_rgb_input(directory,{}))
    assert list(infer.iter_rgb_input(directory,{},assume_fps=20))[2][0]['time_s']==.1
    physical=tmp_path/'physical';physical.mkdir()
    cv2.imwrite(str(physical/'frame10_3_000000.000.jpg'),np.zeros((8,8,3),np.uint8))
    cv2.imwrite(str(physical/'frame11_3_000000.025.jpg'),np.zeros((8,8,3),np.uint8))
    physical_meta={}
    physical_rows=list(infer.iter_rgb_input(physical,physical_meta))
    assert physical_meta['camera_id']==3 and [r['frame_id'] for r,rgb in physical_rows]==[10,11]


def test_prediction_export_contains_no_gt_beta_or_window_anchor(tmp_path,monkeypatch):
    from anysole.data import smpl_io
    rest=np.zeros((24,3));rest[0]=[1,2,3]
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda *a,**k:rest)
    series=p.aggregate_visual_windows([window()])
    paths=p.export_visual_series(series,tmp_path/'predictions',{'smpl_model_path':'authorized_fixture'}, {'checkpoint_sha256':'x'})
    with np.load(paths[0]['path'],allow_pickle=False) as archive:
        assert np.array_equal(archive['betas'],np.zeros(10))
        assert not any(k.startswith('gt_') for k in archive.files)
        np.testing.assert_allclose(archive['trans'][0],-rest[0])
        assert np.all(archive['trajectory'][0]==0)
        meta=json.loads(str(archive['metadata_json']))
        assert meta['pressure_main']=='unclamped_linear' and meta['pred_beta_source']=='zero'
    with pytest.raises(FileExistsError):
        p.export_visual_series(series,tmp_path/'predictions',{'smpl_model_path':'fixture'}, {})


def test_recording_cluster_bootstrap_keeps_cameras_together():
    rows=[]
    for recording,values in [('a',[0,100]),('b',[100,0])]:
        for camera,value in enumerate(values,1):
            rows.append({'recording_id':recording,'camera_id':camera,'subject':'S13',
                         'metrics':{'oracle':{'value':value,'reason':None}}})
    result=p.summarize_visual_rows(rows,bootstrap_samples=300)
    metric=result['metrics']['oracle']
    assert metric['value']==50 and metric['recording_cluster_ci95']==[50,50]
    assert metric['recording_clusters']==2


def report(seed=1,experiment='base_time',shard_index=0,shard_count=1):
    rows=[]
    for recording in ('a','b'):
        rows.append({'recording_id':recording,'camera_id':3,'subject':'S13','metrics':{
            'mpjpe_mm':{'value':10+seed,'count':5,'unit':'mm','reason':None,'support_hash':'same'}}})
    return {'schema_version':p.VISUAL_EVAL_PROTOCOL,'manifest_sha256':'m','split':'test','protocol_sha256':'p',
            'stats_sha256':'s','smpl_runtime_model_hash':'z','checkpoint_sha256':'c','seed':seed,'experiment':experiment,
            'rows':rows,'predictions':[],'shard_index':shard_index,'shard_count':shard_count,'expected_recordings':['a','b']}


def test_shard_merge_rejects_missing_duplicate_and_checkpoint_mismatch():
    a=report(shard_count=2);b=report(shard_index=1,shard_count=2)
    a['rows']=a['rows'][:1];b['rows']=b['rows'][1:]
    merged=p.merge_visual_shards([a,b],bootstrap_samples=0)
    assert len(merged['rows'])==2 and merged['shard_index'] is None
    for bad in ([a],[a,a]):
        with pytest.raises(ValueError,match='shards'):
            p.merge_visual_shards(bad)
    b['checkpoint_sha256']='other'
    with pytest.raises(ValueError,match='checkpoint'):
        p.merge_visual_shards([a,b])


def test_three_seed_summary_pairs_only_identical_metric_supports():
    reports=[report(seed=s,experiment=e) for e in ('base_legacy','base_time','base_pool','view_consist') for s in (1,2,3)]
    reports.extend(report(seed=1,experiment=e) for e in ('pose_pair','press_pair','pair_shuffle'))
    for item in reports:
        if item['experiment']=='view_consist':
            for row in item['rows']:
                row['metrics']['mpjpe_mm']['value']-=2
    result=p.aggregate_visual_reports(reports,baseline='base_time',bootstrap_samples=10)
    assert len(reports)==15 and result['formal_matrix_complete']
    assert result['experiments']['base_time']['mpjpe_mm']['mean']==12
    assert result['experiments']['base_time']['mpjpe_mm']['std']==1
    assert result['paired_differences']['view_consist-base_time']['mpjpe_mm']['value']==-2
    assert result['experiments']['pose_pair']['mpjpe_mm']['std'] is None
    assert result['experiments']['pose_pair']['mpjpe_mm']['seed_count']==1
    assert result['paired_differences']['pose_pair-base_time']['mpjpe_mm']['paired_seeds']==[1]
    reports[-1]['rows'][0]['metrics']['mpjpe_mm']['support_hash']='different'
    with pytest.raises(ValueError,match='coverage'):
        p.aggregate_visual_reports(reports)
    with pytest.raises(ValueError,match='15 runs'):
        p.aggregate_visual_reports([report()])
    with pytest.raises(ValueError,match='seeds'):
        p.aggregate_visual_reports([report()],require_full_matrix=False)


@pytest.mark.parametrize('head',['legacy','time_pe'])
def test_checkpoint_load_requires_saved_effective_head_pe_and_stats(tmp_path,head):
    from anysole.registry import get_builder
    config=get_builder('visual').effective_config({'pressure_head':head,'seed':0})
    model=get_builder('visual').build(config)
    checkpoint={'schema_version':'anysole.visual-training.v1','config':config,'semantic_hash':train.visual_run_identity(config),
        'stats':{'pose_mean':[0.]*144,'pose_std':[1.]*144},'model':model.state_dict()}
    path=tmp_path/'checkpoint.pt';torch.save(checkpoint,path)
    loaded,saved,stats,_=p.load_visual_student(path)
    assert saved['pressure_time_pe_enabled']==(head=='time_pe') and not loaded.training
    del loaded
    checkpoint['config']['pressure_time_pe_enabled']=head!='time_pe'
    checkpoint['semantic_hash']=train.visual_run_identity(checkpoint['config']);torch.save(checkpoint,path)
    with pytest.raises(ValueError,match='contradicts'):
        p.load_visual_student(path)
    checkpoint['config']=config.copy();checkpoint['config'].pop('pressure_head')
    checkpoint['semantic_hash']=train.visual_run_identity(checkpoint['config']);torch.save(checkpoint,path)
    with pytest.raises(ValueError,match='missing saved'):
        p.load_visual_student(path)


def test_new_cli_help_and_cpu_resource_path(capsys,tmp_path):
    for function,args in ((evaluation.main,['--checkpoint','x','--help']),(infer.main,['--checkpoint','x','--help'])):
        with pytest.raises(SystemExit) as exc:
            function(args)
        assert exc.value.code==0
    text=capsys.readouterr().out
    assert '--manifest' in text and '--rgb-meta' in text and '--input' in text
    with p.visual_gpu_slot('cpu',None,'fixture') as resource:
        assert resource['slot'] is None
    with pytest.raises(ValueError,match='resource-plan'):
        with p.visual_gpu_slot('cuda:0',None,'fixture'):
            pass


def test_pure_rgb_cli_end_to_end_refuses_all_legacy_gt_readers(tmp_path,monkeypatch):
    """Routing/export oracle with mocked student/frontend, never a real demo."""
    import cv2
    import types
    from anysole.data import extract_hrnet,smpl_io
    from anysole.data.workspace_adapter import sha256_file
    root=tmp_path/'pure_rgb';root.mkdir()
    image=root/'frames';image.mkdir()
    for i in range(3):
        cv2.imwrite(str(image/f'{i}.png'),np.full((8,8,3),i,np.uint8))
    meta=root/'rgb.json';meta.write_text(json.dumps({'schema_version':'anysole.rgb-only.v1','source_time_s':[0,.025,.05]}))
    checkpoint=root/'student.pt';checkpoint.write_text('explicit synthetic IO oracle')
    config={'smpl_model_path':str(root/'smpl.pkl'),'hrnet_checkpoint':str(root/'hrnet.pt'),
            'detector_checkpoint':str(root/'detector.pt'),'hrnet_config':str(root/'hrnet.yaml'),
            'hrnet_module_root':str(root/'module'),'detector_version':'fixture',
            'preprocessing_version':'fixture','cache_schema':'fixture','structure_hash':'fixture'}
    for key in ('smpl_model_path','hrnet_checkpoint','detector_checkpoint','hrnet_config'):
        Path(config[key]).write_text('owned fixture asset')
    for key in ('hrnet_checkpoint','detector_checkpoint'):
        config[key+'_hash']=sha256_file(Path(config[key]))
    monkeypatch.setattr(p,'load_visual_student',lambda *a,**k:(RGBOracle(),config,{},{}))
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda *a,**k:np.zeros((24,3)))
    monkeypatch.setattr(extract_hrnet,'load_encoder',lambda *a,**k:None)
    class Frontend:
        def __init__(self,*a):
            pass
        def prepare(self,rgb):
            return (np.zeros((3,256,192),np.float32),np.zeros(3,np.float32),{'source':'RGB pixels only'})
        def encode_prepared(self,rows):
            return torch.ones(len(rows),2051)*.2
    monkeypatch.setattr(extract_hrnet,'RGBFeatureFrontend',Frontend)
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(__version__='fixture',YOLO=lambda *a:None))
    def forbidden(*a,**k):
        raise AssertionError('GT reader accessed by pure RGB CLI')
    monkeypatch.setattr(infer,'load_shared_session',forbidden)
    monkeypatch.setattr(infer,'load_session_pressure',forbidden)
    monkeypatch.setattr(infer,'load_smpl',forbidden)
    monkeypatch.setattr(infer,'shared_smpl_path',forbidden)
    result=infer.main(['--checkpoint',str(checkpoint),'--input',str(image),'--rgb-meta',str(meta),
                       '--output',str(root/'output'),'--device','cpu'])
    assert result==0
    report=json.loads((root/'output/inference.json').read_text())
    assert report['GT_access']=='none' and report['predicted_target_frames']==3
    assert report['decoded_frames']==3 and report['lookahead_seconds']==.475
    assert set(report['timing_seconds'])=={'decode','detection','crop','hrnet','student','total'}
    with np.load(report['predictions'][0]['path'],allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['betas'],0)
        assert archive['pressure'].shape==(3,96) and archive['valid'].all()
        assert not any(name.startswith('gt_') for name in archive.files)


def test_full_eval_cli_uses_shared_val_oracle_and_keeps_unlabelled_rgb(tmp_path,monkeypatch):
    from anysole.data import dataset as module,smpl_io
    from anysole.data.workspace_adapter import sha256_file
    # Reuse the runtime's public synthetic grouped fixture, not a data source.
    from test_view_training import ToyGroups
    item=ToyGroups((2,))[0]
    offsets=item['labels']['offsets']
    for name in ('source_frame_id','source_time_s','reuse','time_residual_ms'):
        value=(torch.arange(20) if name=='source_frame_id' else torch.arange(20,dtype=torch.float64)/40
               if name=='source_time_s' else torch.zeros(20,dtype=torch.bool) if name=='reuse' else torch.zeros(20))
        item['meta'][name]=value.repeat(2,1)
    item['visual']['V_feat'][...,:1]=.1
    item['masks']['motion_valid'][10:]=False
    item['masks']['pressure_valid'][10:]=False
    class Dataset:
        def __init__(self,*args,**kwargs):
            assert kwargs['cover_tail'] is True and kwargs['cameras']==(1,2,3,4)
            self.split=kwargs['split'];self.windows=[(0,0,20,(1,2))]
            self.records=[{'recording_id':'toy_0'}];self.manifest_meta={'manifest_sha256':'fixture'}
        def __len__(self):
            return len(self.windows)
        def __getitem__(self,index):
            return copy.deepcopy(item)
    config={'smpl_model_path':str(tmp_path/'smpl.pkl'),'dataset_root':'unused',
            'feature_cache_root':'unused','tw':20,'structure_hash':'fixture','seed':0,'experiment':'base_time'}
    Path(config['smpl_model_path']).write_text('explicit fixture current SMPL')
    checkpoint=tmp_path/'student.pt';checkpoint.write_text('explicit fixture checkpoint')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(p,'load_visual_student',lambda *a,**k:(RGBOracle(),config,{},{}))
    monkeypatch.setattr(p,'beta_zero_visual_offsets',lambda *a:offsets)
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda *a,**k:np.zeros((24,3)))
    monkeypatch.setattr(module,'GroupedViewDataset',Dataset)
    original=p.collect_visual_windows
    def collect(model,dataset,device,**kwargs):
        records=dataset.records;del dataset.records
        try:
            return original(model,dataset,device,**kwargs)
        finally:
            dataset.records=records
    monkeypatch.setattr(p,'collect_visual_windows',collect)
    result=evaluation.main(['--checkpoint',str(checkpoint),'--manifest',str(manifest),'--output',str(tmp_path/'eval'),
                            '--bootstrap-samples','0'])
    assert result==0
    report=json.loads((tmp_path/'eval/eval.json').read_text())
    assert report['selection_metrics']['eval/prediction_frames']==40
    assert report['selection_metrics']['eval/motion_frames']==20
    dataset=Dataset(split='val',cover_tail=True,cameras=(1,2,3,4))
    config['validation_cameras']=[1,2,3,4]
    validation,_=p.evaluate_visual_dataset(RGBOracle(),dataset,config,torch.device('cpu'),offsets=offsets)
    for metric in ('mpjpe','pressure_rmse','coverage_hash'):
        assert report['selection_metrics']['eval/'+metric]==validation['val/'+metric]
    assert len(report['rows'])==2 and len(report['predictions'])==2
    run=tmp_path/'trained_run';run.mkdir();(run/'last.pt').write_text('preserved existing run')
    assert evaluation.main(['--checkpoint',str(checkpoint),'--manifest',str(manifest),
        '--output',str(run/'eval.json'),'--bootstrap-samples','0'])==0
    assert (run/'eval.json').is_file() and (run/'last.pt').read_text()=='preserved existing run'


def test_pressure_prediction_failure_cannot_drop_hard_public_samples():
    w=window([0,1]);w['pressure'][0]=.1
    w['pressure'][1,:48]=0;w['pressure'][1,48:]=.2
    good=p.visual_pressure_metrics(p.aggregate_visual_windows([w])[('r',1)])['left_ratio_mae']
    assert good['value']==pytest.approx(.25) and good['count']==2
    w['pressure'][1]=-1
    bad=p.visual_pressure_metrics(p.aggregate_visual_windows([w])[('r',1)])['left_ratio_mae']
    assert bad['value'] is None and bad['count']==1
    assert bad['eligible_count']==2 and bad['missing_prediction_count']==1
    assert bad['support_hash']!=good['support_hash']
    assert bad['eligible_support_hash']==good['eligible_support_hash']
    rows=[{'recording_id':'r','camera_id':1,'subject':'S13','metrics':{'left_ratio_mae':bad}}]
    summary=p.summarize_visual_rows(rows,bootstrap_samples=10)['metrics']['left_ratio_mae']
    assert summary['value'] is None and summary['recording_cluster_ci95'] is None
    w=window();w['gt_pressure']=np.arange(1,6)[:,None]*np.ones((1,96))*.01
    w['pressure'][:]=.1
    metrics=p.visual_pressure_metrics(p.aggregate_visual_windows([w])[('r',1)])
    assert metrics['left_temporal_corr']['value'] is None
    assert metrics['left_temporal_corr']['missing_prediction_count']==5
    assert metrics['left_peak_timing_abs_ms']['missing_prediction_count']==1


def test_cop_invalid_prediction_is_na_on_gt_fixed_contact_set():
    w=window();w['pressure'][-1]=-1
    policy={'pressure_eval':{'fit_split':'train','fitted_from_train_sha256':'fixture',
        'contact_threshold_normalized':[1,1],'sensor_xy_grid':np.zeros((2,48,2)).tolist()}}
    metric=p.visual_pressure_metrics(p.aggregate_visual_windows([w])[('r',1)],policy)['left_cop_error_grid']
    assert metric['value'] is None and metric['count']==4 and metric['eligible_count']==5
    assert metric['missing_prediction_count']==1


def test_gt_shape_oracle_stays_outside_main_metrics(monkeypatch):
    from anysole.data import smpl_io
    item=p.aggregate_visual_windows([window()])[('r',1)]
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda data,**k:np.arange(72).reshape(24,3)/100*data['betas'][0])
    main=p.visual_motion_metrics(item,torch.arange(72).reshape(24,3).float()/200)
    first=p.visual_gt_shape_diagnostic(item,'owned_fixture')
    item['gt_betas']*=2
    second=p.visual_gt_shape_diagnostic(item,'owned_fixture')
    assert first['excluded_from_main_metrics'] and second['excluded_from_main_metrics']
    assert first['gt_beta_mpjpe_mm']['value']!=second['gt_beta_mpjpe_mm']['value']
    assert main==p.visual_motion_metrics(item,torch.arange(72).reshape(24,3).float()/200)


def test_manifest_templates_preserve_all_cameras_unobserved_axis_and_gt_denominator():
    from test_view_training import ToyGroups
    from anysole.data.dataset import grouped_collate
    item=ToyGroups((2,))[0]
    item['meta'].update(subject='S13',action='walk',date='fixture')
    item['visual']['view_valid'][0,7]=False
    item['visual']['view_edges'][0,6:8]=False
    item['masks']['motion_valid'][-3:]=False
    item['masks']['pressure_valid'][-3:]=False
    for name in ('source_frame_id','source_time_s','reuse','time_residual_ms'):
        value=(torch.arange(20) if name=='source_frame_id' else torch.arange(20,dtype=torch.float64)/40
               if name=='source_time_s' else torch.zeros(20,dtype=torch.bool) if name=='reuse' else torch.zeros(20))
        item['meta'][name]=value.repeat(2,1)
    record={'recording_id':'toy_0','subject':'S13','action':'walk','date':'fixture','target_frame_id':list(range(20)),
        'target_time_s':(np.arange(20)/40).tolist(),'motion':{'valid':item['masks']['motion_valid'].tolist()},
        'pressure':{'foot_valid':item['masks']['pressure_valid'][:,[0,48]].tolist()},
        'motion_edges':item['masks']['motion_edges'].tolist(),
        'pressure_foot_edges':item['masks']['pressure_edges'][:,[0,48]].tolist(),'views':[]}
    for camera in (1,2):
        record['views'].append({'camera_id':camera,**{name:item['meta'][name][camera-1].numpy().tolist()
            for name in ('source_frame_id','source_time_s','reuse','time_residual_ms')}})
    class Dataset:
        split='val';records=[record];_labels={0:{k:v.numpy() for k,v in item['labels'].items()}}
        def __len__(self):
            return 2
        def __getitem__(self,index):
            return copy.deepcopy(item)
    windows,truth,_=p.collect_visual_windows(RGBOracle(),Dataset(),torch.device('cpu'))
    series=p.aggregate_visual_windows(windows)
    assert len(series)==4 and all(len(value['frame_id'])==20 for value in series.values())
    assert series[('toy_0',3)]['valid'].sum()==0 and series[('toy_0',1)]['valid'].sum()==19
    result=p.visual_selection_metrics(series,item['labels']['offsets'],truth)
    assert result['val/prediction_frames']==39 and result['val/gt_motion_frames']==4*17
    assert len(series[('toy_0',1)]['time_s'])==20


def test_artifact_verifier_rejects_prediction_hash_tampering(tmp_path,monkeypatch):
    from anysole.data import smpl_io
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda *a,**k:np.zeros((24,3)))
    entries=p.export_visual_series(p.aggregate_visual_windows([window()]),tmp_path/'predictions',
        {'smpl_model_path':'fixture'},{'checkpoint_sha256':'checkpoint'})
    report={'rows':[{'recording_id':'r','camera_id':1}],'checkpoint_sha256':'checkpoint','predictions':entries}
    p.verify_visual_report_artifacts(report)
    entries[0]['sha256']='tampered'
    with pytest.raises(ValueError,match='hash mismatch'):
        p.verify_visual_report_artifacts(report)


def test_physical_date_session_id_export_is_flat_and_preserves_original_id(tmp_path,monkeypatch):
    from anysole.data import smpl_io
    monkeypatch.setattr(smpl_io,'_rest_joints',lambda *a,**k:np.zeros((24,3)))
    windows=[window(recording='20260804/S5011'),window(recording='20260808/S11012')]
    entries=p.export_visual_series(p.aggregate_visual_windows(windows),tmp_path/'predictions',
        {'smpl_model_path':'fixture'},{'checkpoint_sha256':'c'})
    assert [Path(e['path']).name for e in entries]==['20260804_S5011_cam1.npz','20260808_S11012_cam1.npz']
    for entry in entries:
        assert Path(entry['path']).parent==tmp_path/'predictions'
        with np.load(entry['path'],allow_pickle=False) as archive:
            assert json.loads(str(archive['metadata_json']))['recording_id']==entry['recording_id']
    for recording in ('/20260804/S5011','../S5011','20260804/../S5011','20260804/S5011/extra','20260804\\S5011'):
        with pytest.raises(ValueError,match='unsafe'):
            p.export_visual_series(p.aggregate_visual_windows([window(recording=recording)]),tmp_path/'invalid',
                                   {'smpl_model_path':'fixture'},{})


@pytest.mark.parametrize('frozen',[True,False])
def test_production_checkpoint_restores_sigma_before_immediate_eval(tmp_path,frozen):
    """All six Python flags affect saved validation, beyond tensor state_dict."""
    from anysole.registry import get_builder
    config=get_builder('visual').effective_config({'pressure_head':'time_pe','seed':1})
    model=get_builder('visual').build(config).eval()
    assert sum(parameter.numel() for parameter in model.parameters())==18451344
    model.pose_head.set_sigma_frozen(frozen)
    generator=torch.Generator().manual_seed(1478)
    features=torch.randn(1,20,2051,generator=generator)
    valid=torch.ones(1,20,dtype=torch.bool)
    times=torch.arange(20,dtype=torch.float64)[None]/40
    with torch.no_grad():
        expected=model.forward_visual(features,valid,times)
    checkpoint={'schema_version':'anysole.visual-training.v1','config':config,
        'semantic_hash':train.visual_run_identity(config),'model':model.state_dict(),
        'stats':{'pose_mean':[0.]*144,'pose_std':[1.]*144},
        'model_runtime':{'sigma_frozen':frozen},
        'training':{'successful_updates':1,'sigma_frozen':frozen},'schedule':{'sigma_frozen':frozen}}
    path=tmp_path/'saved_student.pt';torch.save(checkpoint,path)
    loaded,*_=p.load_visual_student(path,'cpu')
    assert all(layer.sigma_frozen is frozen for layer in loaded.pose_head.decoder.layers)
    assert not loaded.training
    with torch.no_grad():
        actual=loaded.forward_visual(features,valid,times)
    assert actual['sigma_frozen'] is frozen
    numeric={}
    for name in ('x0_hat','pressure_hat','v_hat','trans_hat','gate_sigma'):
        assert torch.equal(expected[name],actual[name]),name
        numeric[name+'_max_abs']=float((expected[name]-actual[name]).abs().max())
    evidence={'production_parameters':18451344,'saved_sigma_frozen':frozen,
              'all_layer_flags':[layer.sigma_frozen for layer in loaded.pose_head.decoder.layers],
              'immediate_eval_exact':True,**numeric}
    (tmp_path/'sigma_roundtrip.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps(evidence,sort_keys=True))
    del loaded
    # Backward compatibility uses saved training/schedule for earlier checkpoints.
    del checkpoint['model_runtime'];torch.save(checkpoint,path)
    compatibility,*_=p.load_visual_student(path,'cpu')
    assert all(layer.sigma_frozen is frozen for layer in compatibility.pose_head.decoder.layers)
    del compatibility
    for fields,pattern in (({'model_runtime':{'sigma_frozen':not frozen}},'disagree'),
                           ({'model_runtime':{'sigma_frozen':int(frozen)}},'boolean'),
                           ({'training':{'successful_updates':1},'schedule':{}},'missing sigma-frozen')):
        invalid={**checkpoint,**fields};torch.save(invalid,path)
        with pytest.raises(ValueError,match=pattern):
            p.load_visual_student(path,'cpu')


def _cpu_resource_plan(tmp_path,*,n=3,eval_cap=1,version='cpu-resources.v1',gpus=(0,)):
    from configs.tools.runner import plan_hash
    from test_view_training import make_synthetic_cpu_resource_plan
    valid_n=n if type(n) is int and n>0 else 3
    valid_cap=eval_cap if type(eval_cap) is int and 1<=eval_cap<=valid_n else 1
    valid_version=version if isinstance(version,str) and version.strip() else 'cpu-resources.v1'
    path,plan=make_synthetic_cpu_resource_plan(tmp_path,n=valid_n,eval_cap=valid_cap,
                                               gpus=gpus,version=valid_version)
    # Invalid-field cases keep a complete synthetic provenance chain and a new
    # plan hash, so they test the intended resource field rather than stale hash.
    plan['resources'].update(runs_per_gpu=n,eval_runs_per_gpu=eval_cap,version=version)
    plan['base_config']['control_dir']=str(tmp_path/'control')
    plan['plan_hash']=plan_hash(plan);path.write_text(json.dumps(plan))
    return path,plan


def _cpu_resource_checker(monkeypatch):
    from configs.tools import runner
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'check_visual_gpu_owners',lambda *args:None)
    assert not torch.cuda.is_initialized()
    return runner


@pytest.mark.parametrize('cap',[None,0,-1,True,1.5,'1',4])
def test_eval_guard_rejects_unmeasured_eval_cap_before_lease(tmp_path,monkeypatch,cap):
    _cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path,eval_cap=cap)
    with pytest.raises(ValueError,match='(?i)eval|cap|concurrency'):
        with p.visual_gpu_slot('cuda:0',path,'invalid_eval_cap'):
            pytest.fail('invalid eval cap admitted')
    assert not (tmp_path/'control/gpu_slots.json').exists()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('n',[None,0,True,1.5,'3'])
def test_eval_guard_rejects_unmeasured_total_cap(tmp_path,monkeypatch,n):
    _cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path,n=n)
    with pytest.raises(ValueError,match='(?i)concurrency|runs_per_gpu|resource'):
        with p.visual_gpu_slot('cuda:0',path,'invalid_total_cap'):
            pytest.fail('invalid total cap admitted')
    assert not (tmp_path/'control/gpu_slots.json').exists()


@pytest.mark.parametrize('version',[None,'',' ',1,True])
def test_eval_guard_rejects_unmeasured_resource_version(tmp_path,monkeypatch,version):
    _cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path,version=version)
    with pytest.raises(ValueError,match='(?i)version'):
        with p.visual_gpu_slot('cuda:0',path,'invalid_resource_version'):
            pytest.fail('invalid resource version admitted')
    assert not (tmp_path/'control/gpu_slots.json').exists()


def test_eval_guard_enforces_eval_cap_with_empty_slots_and_releases(tmp_path,monkeypatch):
    runner=_cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path)
    ledger=runner.SlotLedger(tmp_path/'control/gpu_slots.json')
    with p.visual_gpu_slot('cuda:0',path,'first_eval') as first:
        assert first['slot']=='0:0' and first['eval_runs_per_gpu']==1
        assert first['resource_version']=='cpu-resources.v1'
        with pytest.raises(RuntimeError,match='defer evaluation'):
            with p.visual_gpu_slot('cuda:0',path,'second_eval'):
                pytest.fail('two evals exceeded measured cap despite empty slots')
        active=[v for v in json.loads(ledger.path.read_text())['slots'].values() if v['state']=='active']
        assert len(active)==1 and active[0]['job_class']=='eval'
    with p.visual_gpu_slot('cuda:0',path,'after_release'):
        pass
    assert all(v['state']=='released' for v in json.loads(ledger.path.read_text())['slots'].values())
    assert not torch.cuda.is_initialized()


def test_eval_guard_shares_total_slots_with_training_and_owns_exception_release(tmp_path,monkeypatch):
    runner=_cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path)
    ledger=runner.SlotLedger(tmp_path/'control/gpu_slots.json');slots=[(0,s) for s in range(3)]
    leases=[(ledger.claim(slots,identity,job_class='train',resource_version='cpu-resources.v1'),identity)
            for identity in ('train_A','train_B')]
    try:
        with pytest.raises(RuntimeError,match='body failed'):
            with p.visual_gpu_slot('cuda:0',path,'eval_failure') as resource:
                assert resource['slot']=='0:2'
                assert sum(v['state']=='active' for v in json.loads(ledger.path.read_text())['slots'].values())==3
                raise RuntimeError('body failed')
        data=json.loads(ledger.path.read_text())['slots']
        assert all(data[key]['state']=='active' for key,_ in leases)
        assert data['0:2']['state']=='released' and data['0:2']['exit_code']==1
        with p.visual_gpu_slot('cuda:0',path,'eval_after_failure'):
            pass
    finally:
        for key,identity in leases:
            ledger.release(key,identity,0)
    assert not torch.cuda.is_initialized()


def test_eval_guard_matches_per_card_envelope_from_multicard_training_request(tmp_path,monkeypatch):
    runner=_cpu_resource_checker(monkeypatch)
    path,plan=_cpu_resource_plan(tmp_path,gpus=tuple(range(7)))
    ledger=runner.SlotLedger(tmp_path/'control/gpu_slots.json')
    slots=runner.visual_slot_assignments(list(range(7)),3)
    key=ledger.claim(slots,'multicard_training',job_class='train',resource_version='cpu-resources.v1')
    try:
        assert key=='0:0'
        with p.visual_gpu_slot('cuda:0',path,'same_card_evaluation') as resource:
            assert resource['slot']=='0:1'
            data=json.loads(ledger.path.read_text())['slots']
            for entry in data.values():
                assert entry['approved_slots']==[0,1,2] and entry['total_slots_per_gpu']==3
    finally:
        ledger.release(key,'multicard_training',0)
    assert not torch.cuda.is_initialized()


def _cpu_live_resource_entry(identity,*,job_type='eval',version='cpu-resources.v1',starttime=None):
    import os
    return {'gpu':0,'slot':0,'state':'active','pid':os.getpid(),
            'starttime':train.process_starttime(os.getpid()) if starttime is None else starttime,
            'semantic_hash':identity,'job_type':job_type,'resource_version':version,
            'approved_slots':[0,1,2],'total_slots_per_gpu':3}


def test_eval_guard_counts_legacy_alias_and_ignores_only_precise_stale_lease(tmp_path,monkeypatch):
    runner=_cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path)
    ledger=runner.SlotLedger(tmp_path/'control/gpu_slots.json')
    with ledger.transaction() as data:
        data['slots']['0:0']=_cpu_live_resource_entry('legacy_eval',job_type='n_calibration_eval')
    with pytest.raises(RuntimeError,match='defer evaluation'):
        with p.visual_gpu_slot('cuda:0',path,'next_eval'):
            pytest.fail('legacy eval alias bypassed cap')
    with ledger.transaction() as data:
        data['slots']['0:0']['starttime']='not-this-process-starttime'
    with p.visual_gpu_slot('cuda:0',path,'after_stale') as resource:
        assert resource['slot']=='0:0'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind',['unknown_type','different_version','missing_version','outside_slot_set'])
def test_eval_guard_rejects_unclassified_or_incompatible_live_lease(tmp_path,monkeypatch,kind):
    runner=_cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path)
    ledger=runner.SlotLedger(tmp_path/'control/gpu_slots.json')
    entry=_cpu_live_resource_entry('existing_live',job_type='train')
    if kind=='unknown_type':entry['job_type']='unclassified_live_type'
    elif kind=='different_version':entry['resource_version']='other-resources.v1'
    elif kind=='missing_version':entry.pop('resource_version')
    else:entry['slot']=3
    key=f"0:{entry['slot']}"
    with ledger.transaction() as data:data['slots'][key]=entry.copy()
    with pytest.raises((ValueError,RuntimeError),match='class|type|version|slot|resource'):
        with p.visual_gpu_slot('cuda:0',path,'must_reject'):
            pytest.fail('incompatible live resource was ignored')
    assert json.loads(ledger.path.read_text())['slots'][key]==entry
    assert not torch.cuda.is_initialized()


def _cpu_racing_eval_guard(plan,identity,ready,start,results,release):
    """Independent CPU process; only NVML inspection is stubbed."""
    import os
    from configs.tools import runner
    torch.set_num_threads(1)
    runner.check_visual_gpu_owners=lambda *args:None
    assert not torch.cuda.is_initialized()
    ready.put(os.getpid())
    if not start.wait(30):raise TimeoutError('CPU race start timed out')
    try:
        with p.visual_gpu_slot('cuda:0',plan,identity) as resource:
            results.put({'status':'claimed','pid':os.getpid(),'slot':resource['slot'],
                         'cuda_initialized':torch.cuda.is_initialized()})
            if not release.wait(30):raise TimeoutError('CPU race release timed out')
    except RuntimeError as exc:
        if 'defer evaluation' not in str(exc):raise
        results.put({'status':'denied','pid':os.getpid(),'error':str(exc),
                     'cuda_initialized':torch.cuda.is_initialized()})
    assert not torch.cuda.is_initialized()


def test_eval_guard_atomic_limit_across_independent_processes(tmp_path,monkeypatch):
    import multiprocessing
    _cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path)
    ctx=multiprocessing.get_context('spawn')
    ready=ctx.Queue();results=ctx.Queue();start=ctx.Event();release=ctx.Event()
    children=[ctx.Process(target=_cpu_racing_eval_guard,args=(str(path),f'CPU_eval_{i}',ready,start,results,release))
              for i in range(2)]
    try:
        for child in children:child.start()
        assert {ready.get(timeout=30) for _ in children}=={child.pid for child in children}
        start.set();rows=[results.get(timeout=30) for _ in children]
        assert sorted(row['status'] for row in rows)==['claimed','denied']
        assert all(row['cuda_initialized'] is False for row in rows)
        winner=next(row for row in rows if row['status']=='claimed')
        active=[v for v in json.loads((tmp_path/'control/gpu_slots.json').read_text())['slots'].values()
                if v['state']=='active']
        assert len(active)==1 and active[0]['pid']==winner['pid']
        assert active[0]['starttime']==train.process_starttime(winner['pid'])
    finally:
        release.set()
        for child in children:
            child.join(timeout=15)
            if child.is_alive():child.terminate();child.join(timeout=5)
    assert all(child.exitcode==0 for child in children)
    assert all(v['state']=='released' for v in json.loads((tmp_path/'control/gpu_slots.json').read_text())['slots'].values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind',['stale_hash_caps','wrong_schema','missing_measurement',
    'wrong_measurement_hash','wrong_measurement_schema','incomplete_mixed',
    'unauthorized_measurement','referenced_config_drift'])
def test_eval_guard_binding_rejects_before_nvml_or_claim(tmp_path,monkeypatch,kind):
    import hashlib
    runner=_cpu_resource_checker(monkeypatch)
    path,plan=_cpu_resource_plan(tmp_path,n=2)
    def save_candidate(candidate):
        report_path=Path(plan['resources']['measurement']['path'])
        report_path.write_text(json.dumps(candidate))
        plan['resources']['measurement']['sha256']=hashlib.sha256(report_path.read_bytes()).hexdigest()
    if kind=='stale_hash_caps':
        # The original independent counterexample: keep hash/version while
        # increasing n and eval cap, without a new measured envelope.
        plan['resources'].update(runs_per_gpu=3,eval_runs_per_gpu=2)
    else:
        if kind=='wrong_schema':plan['schema']='not.anysole.visual-plan.v1'
        elif kind=='missing_measurement':plan['resources'].pop('measurement')
        elif kind=='wrong_measurement_hash':plan['resources']['measurement']['sha256']='0'*64
        else:
            candidate=json.loads(Path(plan['resources']['measurement']['path']).read_text())
            if kind=='wrong_measurement_schema':candidate['schema']='not.a.measured.resource-candidate'
            elif kind=='incomplete_mixed':candidate['mixed_evaluation_passed']=False
            elif kind=='unauthorized_measurement':
                authorization=Path(candidate['authorization']['path'])
                permit=json.loads(authorization.read_text());permit['measurement_authorized']=False
                authorization.write_text(json.dumps(permit))
                candidate['authorization']['sha256']=hashlib.sha256(authorization.read_bytes()).hexdigest()
            else:
                final_config=Path(candidate['final_config']['path'])
                final_config.write_text(final_config.read_text()+'\n# SYNTHETIC_CPU_ONLY_DRIFT\n')
            save_candidate(candidate)
        plan['plan_hash']=runner.plan_hash(plan)
    path.write_text(json.dumps(plan));calls=[]
    def forbidden(name):
        def call(*args,**kwargs):
            calls.append(name)
            raise AssertionError('resource validation happened after '+name)
        return call
    monkeypatch.setattr(runner,'check_visual_gpu_owners',forbidden('NVML'))
    monkeypatch.setattr(runner.SlotLedger,'claim',forbidden('claim'))
    with pytest.raises((ValueError,PermissionError,OSError)):
        with p.visual_gpu_slot('cuda:0',path,'rejected_binding'):
            pytest.fail('tampered/unmeasured resource plan admitted')
    assert calls==[]
    assert not (tmp_path/'control/gpu_slots.json').exists()
    assert not torch.cuda.is_initialized()


def test_eval_guard_uses_real_shared_binding_helper_before_nvml_and_claim(tmp_path,monkeypatch):
    runner=_cpu_resource_checker(monkeypatch)
    path,_=_cpu_resource_plan(tmp_path);events=[]
    original_validate=runner.validate_visual_resource_plan;original_claim=runner.SlotLedger.claim
    def validate(plan):
        events.append('validate')
        return original_validate(plan)
    def claim(ledger,*args,**kwargs):
        events.append('claim')
        assert kwargs['job_class']=='eval' and kwargs['max_job_class_per_gpu']==1
        return original_claim(ledger,*args,**kwargs)
    monkeypatch.setattr(runner,'validate_visual_resource_plan',validate)
    monkeypatch.setattr(runner,'check_visual_gpu_owners',lambda *args:events.append('NVML_STUB'))
    monkeypatch.setattr(runner.SlotLedger,'claim',claim)
    with p.visual_gpu_slot('cuda:0',path,'validated_binding'):
        assert events==['validate','NVML_STUB','claim']
    assert not torch.cuda.is_initialized()
