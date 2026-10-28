"""Physical-time/grouped contracts, using controlled files rather than datasets."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anysole.data import workspace_adapter as adapter
from anysole.data.dataset import GroupedViewDataset, grouped_collate
from anysole.data.smpl_io import load_smpl
from anysole.types import assert_visual_batch_shapes


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _imported(tmp_path):
    root=tmp_path/'data'; metadata=tmp_path/'pressure'; metadata.mkdir()
    date='20260101'; n=30
    split=tmp_path/'splits.csv'; split.write_text('train,val,test\nS5011,S13011,S13011\n')
    align=[]; selections=[]; sessions=[]
    for sid,subject in (('S5011','S5'),('S13011','S13')):
        pp=root/date/'Tactile'/sid/'pressure_48.npz'; pp.parent.mkdir(parents=True)
        np.savez(pp,frame_id=np.arange(n),left48=np.ones((n,48),np.float32)*1023,
                 right48=np.ones((n,48),np.float32)*2046,valid=np.ones(n,np.uint8),fake=np.zeros(n,np.uint8),
                 unit=np.asarray('source_sensor_units'),alignment_method=np.asarray('nearest_values_absolute_t_us'))
        mp=root/date/'Mocap'/sid/'motion_neutral_smpl.npz';mp.parent.mkdir(parents=True)
        trans=np.zeros((100,3));trans[:,0]=np.arange(100)*.01
        np.savez(mp,poses=np.zeros((100,72)),trans=trans,betas=np.zeros(10),
                 source_frame_times_s=np.arange(100)/120,mocap_frame_rate=120.,
                 coordinate_system=np.asarray('SMPL right-handed +X left, +Y up, +Z forward; world motion preserved'),
                 model_units=np.asarray('m'),surface_model_sha256=np.asarray('source-model'),
                 root_branch_repair_status=np.asarray('passed'),joint_continuity_status=np.asarray('passed'),
                 fit_valid=np.ones((100,53),bool),unneeded_object=np.asarray([{'unsafe':'not loaded'}],object))
        for cam in range(1,5):
            directory=root/date/'RGB'/'method_1_offset'/sid/f'camera_{cam}';directory.mkdir(parents=True)
            for i in range(n):
                clock=10000+i/40;hh=int(clock//3600);mm=int(clock%3600//60);ss=clock%60
                (directory/f'frame{i}_{cam}_{hh:02d}{mm:02d}{ss:06.3f}.jpg').write_bytes(b'explicit fixture RGB')
        anchor=root/date/'RGB_MoCap_time_alignment'/f'{sid}.csv';anchor.parent.mkdir(exist_ok=True)
        with anchor.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['标记','视觉时间(s)','视觉帧','动捕时间(s)','动捕帧'])
            w.writerow(['start',0,0,0,0]);w.writerow(['end',(n-1)/40,n-1,(n-1)/40,87])
        pa={'camera':'cam1','fps':40.,'frame_count':n,'first_frame_wall_s':10000.,
            'out_of_span_frames_left':[0,0],'out_of_span_frames_right':[1,0],
            'left_source_sha256':'left-source','right_source_sha256':'right-source'}
        align.append({'session':f'{date}/{subject}/{sid}','selected_camera':'cam1','cameras':[pa]})
        selections.append({'session':f'{date}/{subject}/{sid}','selected_camera':'cam1','selected_sha256':_hash(pp),
                           'candidates':[{'camera':'cam1','sha256':_hash(pp),'frame_count':n,'video_first_wall_s':10000.}]})
        sessions.append([date,subject,sid,'cam1',n])
    (root/date/f'{date}.json').write_text(json.dumps({'pixel_coordinates':'raw_distorted'}))
    with (metadata/'sessions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['date','subject','session','selected_camera','frame_count']);w.writerows(sessions)
    (metadata/'alignment_audit.json').write_text(json.dumps(align))
    (metadata/'selection_audit.json').write_text(json.dumps({'sessions':selections}))
    config={'dataset_root':str(root),'split_csv':str(split),'pressure_metadata_root':str(metadata)}
    manifest=root/'anysole_manifest.jsonl'
    meta=adapter.build_grouped_manifest(config,manifest)
    return root,manifest,meta


def _rest(data,model_path=None):
    result=np.zeros((24,3));result[:,0]=np.arange(24)*.1+.5
    return result


def _cache(tmp_path,manifest,cameras=(1,2,3,4),invalid_row=None):
    root=tmp_path/'cache';root.mkdir(exist_ok=True)
    records,_=adapter.load_grouped_manifest(manifest)
    entries={}
    for record in records:
        for view in record['views']:
            if view['camera_id'] not in cameras: continue
            n=len(view['sources']);features=torch.arange(1,n+1)[:,None].expand(n,2051).float().clone()
            valid=torch.ones(n,dtype=torch.bool)
            if invalid_row is not None:
                valid[invalid_row]=False;features[invalid_row]=float('nan')
            path=root/f"{record['session_id']}_cam{view['camera_id']}.pt"
            torch.save({'features':features,'source_valid':valid},path)
            entries[f"{record['recording_id']}/cam{view['camera_id']}"]={'path':path.name,'sha256':_hash(path),
                'provenance':{'recording_id':record['recording_id'],'camera_id':view['camera_id'],
                              'method':record['method'],'source_index_sha256':adapter.semantic_hash(view['sources'])}}
    (root/'index.json').write_text(json.dumps({'schema_version':adapter.VIEW_CACHE_SCHEMA,'entries':entries}))
    return root


def test_imported_provenance_bom_roles_and_no_pressure_dependency(tmp_path,monkeypatch):
    root,manifest,meta=_imported(tmp_path)
    records,_=adapter.load_grouped_manifest(manifest)
    assert meta['candidate_split_counts']=={'train':1,'val':1,'test':1}
    assert next(r for r in records if r['session_id']=='S13011')['split_roles']==['val','test']
    r=next(r for r in records if r['session_id']=='S5011')
    assert r['pressure']['selected_camera']=='cam1' and r['clock_mapping']['anchor_camera']==3
    assert r['target_time_s'][1]==pytest.approx(10000.025)
    assert r['motion']['time_s'][1]==pytest.approx(.025)
    assert r['pressure']['foot_valid'][0]==[True,False]
    assert r['pressure']['foot_valid'][1]==[True,True]
    (tmp_path/'pressure').rename(tmp_path/'unavailable_source_metadata')
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    labels=adapter.load_grouped_labels(r,root,tmp_path/'unused.pkl')
    assert labels['pressure'][0,:48].tolist()==[1.]*48
    assert labels['pressure'][0,48:].tolist()==[0.]*48
    assert labels['pressure'][1].tolist()==[1.]*96
    assert labels['root'][0,0]==pytest.approx(.5)
    assert labels['velocity'][1,0]==pytest.approx(1.2)
    assert np.isfinite(labels['pose6d']).all()
    # Builder never rewrites the old roles or overwrites an existing index implicitly.
    with pytest.raises(FileExistsError): adapter.build_grouped_manifest({},manifest)


def test_nearest_real_time_ties_outside_and_numeric_order(tmp_path):
    for name in ('frame10_1_120000.050.jpg','frame2_1_120000.000.jpg','frame3_1_120000.000.jpg'):
        (tmp_path/name).write_bytes(b'rgb')
    source=adapter.read_rgb_sources(tmp_path,1)
    assert [r['frame_id'] for r in source]==[2,3,10]
    result=adapter.match_rgb_sources(np.asarray([43199.999,43200.,43200.025,43200.05,43200.051]),source,25)
    assert result['source_frame_id']==[2,2,2,10,10]
    assert result['view_valid']==[False,True,True,True,False]
    assert result['reuse'][2]


def test_fake_and_each_foot_source_span():
    result=adapter.source_pressure_mask(np.asarray([1,1,0,1,1]),np.asarray([0,0,1,0,0]),
        {'out_of_span_frames_left':[1,0],'out_of_span_frames_right':[0,2]})
    assert result.tolist()==[[False,True],[True,True],[False,False],[True,False],[True,False]]
    with pytest.raises(ValueError):
        adapter.source_pressure_mask(np.ones(2),np.ones(2),{'out_of_span_frames_left':[0,0],'out_of_span_frames_right':[0,0]})


@pytest.mark.parametrize('count',[1,2,3,4])
def test_real_cache_mapping_missing_nan_masks_and_read_only(tmp_path,monkeypatch,count):
    root,manifest,meta=_imported(tmp_path)
    cameras=tuple(range(1,count+1));cache=_cache(tmp_path,manifest,cameras,invalid_row=2)
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    before={str(p.relative_to(tmp_path)):_hash(p) for p in tmp_path.rglob('*') if p.is_file()}
    ds=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',cameras=cameras,cover_tail=True)
    assert len(ds)==2
    item=ds[0]
    assert item['visual']['V_feat'].shape==(count,20,2051)
    assert not item['visual']['view_valid'][:,2].any()
    assert not item['visual']['V_feat'][:,2].any() # explicitly masked missing detection, not a valid fake cache
    assert not item['visual']['view_edges'][:,1:3].any()
    batch=grouped_collate([item,ds[1]])
    assert assert_visual_batch_shapes(batch)==(2,count,20)
    after={str(p.relative_to(tmp_path)):_hash(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


def test_cache_required_and_hash_checked(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path)
    with pytest.raises(FileNotFoundError,match='real RGB'):
        GroupedViewDataset(manifest,root,tmp_path/'missing',tmp_path/'unused.pkl')
    cache=_cache(tmp_path,manifest)
    index=json.loads((cache/'index.json').read_text());entry=index['entries']['20260101/S5011/cam1']
    (cache/entry['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='provenance'):
        GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl')


def test_real_grouped_dataset_test_role_is_rejected_before_model_forward(tmp_path,monkeypatch):
    from anysole.utils.eval_protocol import evaluate_visual_dataset
    root,manifest,_=_imported(tmp_path);cache=_cache(tmp_path,manifest)
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    dataset=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',split='test',cover_tail=True)
    class Sentinel(torch.nn.Module):
        def forward_visual(self,*args,**kwargs):
            raise AssertionError('test-role Dataset reached model forward')
    with pytest.raises(ValueError,match='test role'):
        evaluate_visual_dataset(Sentinel(),dataset,{'validation_cameras':[1,2,3,4]},
                                torch.device('cpu'),offsets=torch.zeros(24,3))
    assert dataset.split=='test'
    for split in ('train','val'):
        other=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',split=split)
        assert other.split==split


def test_collate_camera_and_tail_padding_keeps_time(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path);cache=_cache(tmp_path,manifest)
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    a=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',cameras=(1,))[0]
    b=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl')[0]
    # A genuine short inference item: retain first 5 targets, not their indexes compressed.
    for section in ('visual','masks','meta','labels'):
        for key,value in a[section].items():
            if not torch.is_tensor(value) or key in {'betas','offsets','parents','camera_id'}:continue
            if key in {'V_feat','view_valid','source_frame_id','source_time_s','source_feature_row','reuse','padding','time_residual_ms'}:
                a[section][key]=value[:,:5]
            elif key=='view_edges':a[section][key]=value[:,:4]
            else:a[section][key]=value[:4 if key in {'motion_edges','pressure_edges'} else 5]
    batch=grouped_collate([a,b]);assert_visual_batch_shapes(batch)
    assert batch['meta']['padding'][0,1:].all()
    assert batch['visual']['target_time_s'][0,5]==pytest.approx(10000.125)
    assert not batch['masks']['motion_valid'][0,5:].any()
    assert_visual_batch_shapes(grouped_collate([a]))


def test_train_stats_ignore_heldout_values(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path);model=tmp_path/'model.pkl';model.write_bytes(b'explicit fixture')
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    original=adapter.load_grouped_labels
    def only_train(record,*args):
        assert record['split_roles']==['train']
        return original(record,*args)
    monkeypatch.setattr(adapter,'load_grouped_labels',only_train)
    first=adapter.fit_grouped_stats(manifest,root,model)
    assert first['pose_count']==30 and first['velocity_count']==29
    heldout=root/'20260101/Mocap/S13011/motion_neutral_smpl.npz'
    heldout.write_bytes(b'heldout may be unavailable while fitting train statistics')
    second=adapter.fit_grouped_stats(manifest,root,model)
    assert first==second


def test_strict_smpl_query_rejects_extrapolation_without_marker_pickle(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path)
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    path=root/'20260101/Mocap/S5011/motion_neutral_smpl.npz'
    with pytest.raises(ValueError,match='outside'):
        load_smpl(path,np.asarray([-.01]),allow_extrapolation=False)
    result=load_smpl(path,np.asarray([.0125]),allow_extrapolation=False)
    assert result['model_trans_m'][0,0]==pytest.approx(.015)


def test_all_fake_recording_is_excluded_by_quality_and_sha_mismatch_fails(tmp_path):
    root,manifest,_=_imported(tmp_path)
    pp=root/'20260101/Tactile/S13011/pressure_48.npz'
    with np.load(pp,allow_pickle=False) as z: values={k:z[k] for k in z.files}
    values['valid'][:]=0;values['fake'][:]=1
    values['alignment_method']=np.asarray('nearest_values_normalized_t_us')
    np.savez(pp,**values)
    config={'dataset_root':str(root),'split_csv':str(tmp_path/'splits.csv'),'pressure_metadata_root':str(tmp_path/'pressure')}
    with pytest.raises(ValueError,match='provenance'):
        adapter.build_grouped_manifest(config,root/'bad.jsonl')
    selected=tmp_path/'pressure/selection_audit.json';obj=json.loads(selected.read_text())
    entry=next(r for r in obj['sessions'] if r['session'].endswith('/S13011'))
    entry['selected_sha256']=_hash(pp);entry['candidates'][0]['sha256']=_hash(pp)
    selected.write_text(json.dumps(obj))
    adapter.build_grouped_manifest(config,root/'revised.jsonl')
    records,meta=adapter.load_grouped_manifest(root/'revised.jsonl')
    bad=next(r for r in records if r['session_id']=='S13011')
    assert not bad['eligible'] and 'no_trustworthy_pressure' in bad['exclusion_reasons']
    assert meta['eligible_split_counts']=={'train':1,'val':0,'test':0}


def test_so3_interpolation_and_duplicate_source_dynamic_edges(tmp_path):
    from anysole.data.rgb_smpl.alignment import bracket_times,interpolate_axis_angle
    source=np.asarray([[[0.,0.,0.]],[[0.,0.,np.pi]]])
    sampled=interpolate_axis_angle(source,bracket_times(np.asarray([0.,1.]),np.asarray([.5])))
    assert sampled[0,0,2]==pytest.approx(np.pi/2)
    root,manifest,_=_imported(tmp_path)
    records,_=adapter.load_grouped_manifest(manifest)
    r=records[0];view=r['views'][0]
    # Image identity changes, not merely regular target times, govern dynamics.
    sources=[{'file':'a','frame_id':1,'time_s':0.},{'file':'b','frame_id':2,'time_s':.075}]
    matched=adapter.match_rgb_sources(np.arange(4)/40,sources,25)
    assert matched['reuse'][1]
    assert matched['source_frame_id'][0]==matched['source_frame_id'][1]


def test_missing_whole_camera_is_not_a_missing_feature_cache(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path)
    records,meta=adapter.load_grouped_manifest(manifest)
    for record in records:
        view=record['views'][3];n=len(record['target_frame_id'])
        view['sources']=[];view.update(adapter.match_rgb_sources(np.asarray(record['target_time_s']),[],25))
        view['view_edges']=[False]*(n-1)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    meta['manifest_sha256']=_hash(manifest)
    manifest.with_suffix('.meta.json').write_text(json.dumps(meta))
    cache=_cache(tmp_path,manifest,cameras=(1,2,3))
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    ds=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl')
    assert ds[0]['visual']['V_feat'].shape==(3,20,2051)


def test_frontend_crop_geometry_and_rgb_only_selection():
    from types import SimpleNamespace
    from anysole.data.extract_hrnet import RGBFeatureFrontend
    class Detector:
        def predict(self,**kwargs):
            assert kwargs['classes']==[0] and 'source' in kwargs
            boxes=SimpleNamespace(xyxy=torch.tensor([[20.,10.,80.,110.],[0.,0.,10.,10.]]),conf=torch.tensor([.9,.99]))
            return [SimpleNamespace(boxes=boxes)]
    class Encoder(torch.nn.Module):
        def forward(self,x):return x.mean((2,3)).mean(1,keepdim=True).expand(len(x),2048)
    frontend=RGBFeatureFrontend(Encoder(),Detector(),torch.device('cpu'))
    rgb=np.full((120,160,3),128,np.uint8)
    prepared=frontend.prepare(rgb)
    assert prepared[0].shape==(3,256,192)
    assert prepared[2]['bbox_xyxy']==[20.,10.,80.,110.]
    assert np.asarray(prepared[2]['crop_affine_pixel_centers']).shape==(2,3)
    feature=frontend.encode_prepared([prepared])
    assert feature.shape==(1,2051) and torch.isfinite(feature).all()


def test_all_invalid_views_are_counted_and_never_forwarded(tmp_path):
    root,manifest,_=_imported(tmp_path);cache=_cache(tmp_path,manifest)
    index_path=cache/'index.json';index=json.loads(index_path.read_text())
    for key,entry in index['entries'].items():
        if '/S5011/' not in key:continue
        path=cache/entry['path'];obj=torch.load(path,weights_only=True)
        obj['source_valid'][:]=False;obj['features'][:]=float('nan')
        torch.save(obj,path);entry['sha256']=_hash(path)
    index_path.write_text(json.dumps(index))
    ds=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',cover_tail=True)
    assert len(ds)==0 and ds.coverage['no_valid_rgb_windows']==2
    assert ds.coverage['zero_supervision_windows']==0


def test_evaluation_keeps_rgb_without_ground_truth_and_training_filters_it(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path)
    records,meta=adapter.load_grouped_manifest(manifest)
    for record in records:
        n=len(record['target_frame_id'])
        record['motion']['valid']=[False]*n
        record['pressure']['foot_valid']=[[False,False] for _ in range(n)]
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    meta['manifest_sha256']=_hash(manifest)
    manifest.with_suffix('.meta.json').write_text(json.dumps(meta))
    cache=_cache(tmp_path,manifest)
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints',_rest)
    train=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl')
    assert len(train)==0 and train.coverage['zero_supervision_windows']==1
    for split,cover_tail,expected in (('train',True,2),('val',False,1),('test',True,2)):
        dataset=GroupedViewDataset(manifest,root,cache,tmp_path/'unused.pkl',split=split,cover_tail=cover_tail)
        assert len(dataset)==expected
        assert dataset.coverage['zero_supervision_windows']==expected
        assert dataset.coverage['no_valid_rgb_windows']==0
        item=dataset[0]
        assert item['visual']['view_valid'].any()
        assert not item['masks']['motion_valid'].any()
        assert not item['masks']['pressure_valid'].any()
        assert_visual_batch_shapes(grouped_collate([item]))


def _mock_extraction(tmp_path,monkeypatch,root,manifest):
    import sys
    from types import SimpleNamespace
    from anysole.data import extract_hrnet as frontend
    class Encoder(torch.nn.Module):
        def forward(self,x):return torch.ones((len(x),2048))
    class Detector:
        def predict(self,**kwargs):
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[20.,10.,80.,110.]]),conf=torch.tensor([.9])))]
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda _:Detector()))
    monkeypatch.setattr(frontend,'load_encoder',lambda *a,**k:Encoder())
    monkeypatch.setattr(frontend,'read_rgb',lambda path:np.full((120,160,3),128,np.uint8))
    assets=tmp_path/'assets';assets.mkdir()
    for name in ('yaml','encoder.pt','detector.pt'):(assets/name).write_bytes(b'fixture asset')
    return {'dataset_root':str(root),'manifest_path':str(manifest),'feature_cache_root':str(tmp_path/'cache'),
            'hrnet_module_root':str(assets),'hrnet_config':str(assets/'yaml'),
            'hrnet_checkpoint':str(assets/'encoder.pt'),'detector_checkpoint':str(assets/'detector.pt'),
            'cache_min_free_bytes':0},frontend


def test_parallel_shards_and_unique_atomic_merge(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    root,manifest,_=_imported(tmp_path);config,frontend=_mock_extraction(tmp_path,monkeypatch,root,manifest)
    ids=['20260101/S5011','20260101/S13011'];paths=[tmp_path/f'shard{i}.json' for i in range(2)]
    def run(i):
        cfg={**config,'cache_recording_ids':[ids[i]],'cache_index_output':str(paths[i]),'cache_worker_id':f'worker{i}'}
        return frontend.extract_grouped(cfg,device=torch.device('cpu'),batch_size=8)
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(run,range(2)))
    assert [r['entries'] for r in results]==[4,4]
    main=Path(config['feature_cache_root'])/'index.json'
    assert not main.exists() # workers never publish a partial canonical index
    result=frontend.merge_cache_shards(manifest,root,Path(config['feature_cache_root']),paths)
    assert result['coverage']['units']==8 and result['coverage']['valid_sources']==240
    assert json.loads(main.read_text())['complete']
    before=_hash(main)
    shard=json.loads(paths[1].read_text());shard['entries'].update(json.loads(paths[0].read_text())['entries'])
    paths[1].write_text(json.dumps(shard))
    with pytest.raises(ValueError,match='duplicate'):frontend.merge_cache_shards(manifest,root,Path(config['feature_cache_root']),paths)
    assert _hash(main)==before


def test_shard_recovery_skips_only_committed_hash_valid_units(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path);config,frontend=_mock_extraction(tmp_path,monkeypatch,root,manifest)
    shard=tmp_path/'shard.json';state=tmp_path/'state.json'
    config.update(cache_recording_ids=['20260101/S5011'],cache_index_output=str(shard),cache_worker_state=str(state),cache_worker_id='resume-owner')
    original=frontend.RGBFeatureFrontend.prepare;calls=0
    def interrupted(self,rgb):
        nonlocal calls
        calls+=1
        if calls==31:raise RuntimeError('controlled interruption')
        return original(self,rgb)
    monkeypatch.setattr(frontend.RGBFeatureFrontend,'prepare',interrupted)
    with pytest.raises(RuntimeError,match='controlled'):frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)
    partial=json.loads(shard.read_text());assert len(partial['entries'])==1 and not partial['complete']
    assert json.loads(state.read_text())['status']=='failed'
    committed=next(iter(partial['entries'].values()));path=Path(config['feature_cache_root'])/committed['path']
    signature=(_hash(path),path.stat().st_mtime_ns)
    monkeypatch.setattr(frontend.RGBFeatureFrontend,'prepare',original)
    frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)
    assert (_hash(path),path.stat().st_mtime_ns)==signature
    assert json.loads(state.read_text())['attempt_sources']==90
    assert json.loads(shard.read_text())['complete']
    bad=json.loads(shard.read_text());next(iter(bad['entries'].values()))['sha256']='incorrect'
    shard.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='hash mismatch'):frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)


def test_shard_merge_rejects_incomplete_and_wrong_feature_hash(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path);config,frontend=_mock_extraction(tmp_path,monkeypatch,root,manifest)
    shard=tmp_path/'all.json';config.update(cache_index_output=str(shard),cache_worker_id='one-writer')
    frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)
    good=json.loads(shard.read_text());bad={**good,'complete':False};shard.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='incomplete'):frontend.merge_cache_shards(manifest,root,Path(config['feature_cache_root']),[shard])
    next(iter(good['entries'].values()))['sha256']='incorrect';shard.write_text(json.dumps(good))
    with pytest.raises(ValueError,match='hash mismatch'):frontend.merge_cache_shards(manifest,root,Path(config['feature_cache_root']),[shard])
    assert not (Path(config['feature_cache_root'])/'index.json').exists()


def test_cache_locks_reject_live_owner_and_recover_only_own_dead_owner(tmp_path):
    from anysole.data.extract_hrnet import cache_writer_lock
    lock=tmp_path/'unit.lock'
    with cache_writer_lock(lock,'same'):
        with pytest.raises(FileExistsError):
            with cache_writer_lock(lock,'same'):pass
    lock.write_text(json.dumps({'pid':999999999,'start_time_ticks':1,'owner':'same'}))
    with cache_writer_lock(lock,'same'):assert lock.exists()
    assert not lock.exists()
    lock.write_text(json.dumps({'pid':999999999,'start_time_ticks':1,'owner':'other'}))
    with pytest.raises(FileExistsError):
        with cache_writer_lock(lock,'same'):pass


def test_single_writer_default_index_remains_compatible(tmp_path,monkeypatch):
    root,manifest,_=_imported(tmp_path);config,frontend=_mock_extraction(tmp_path,monkeypatch,root,manifest)
    config['cache_worker_state']=str(tmp_path/'state.json')
    first=frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)
    assert first['entries']==8 and Path(first['index'])==Path(config['feature_cache_root'])/'index.json'
    digest=_hash(first['index'])
    second=frontend.extract_grouped(config,device=torch.device('cpu'),batch_size=8)
    assert _hash(second['index'])==digest and json.loads((tmp_path/'state.json').read_text())['attempt_sources']==0


def _b1_imported(tmp_path, monkeypatch):
    root, manifest, _ = _imported(tmp_path)
    split = tmp_path / 'splits.csv'
    split.write_text('train,val,test\nS5011,S13011,\n')
    from scipy.spatial.transform import Rotation
    a = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    cameras = {f'cam{c}': {'R': Rotation.from_euler('y', angle, degrees=True).as_matrix().tolist(),
                't': [float(c * 10), 5., 400.], 'K': [[1000., 0., 800.], [0., 1000., 600.], [0., 0., 1.]]}
               for c, angle in ((1, 0), (2, -90), (4, 30))}
    calibration = {'pixel_coordinates': 'raw_distorted', 'image_size_px': [1600, 1200],
                   'cameras': cameras, 'mocap_raw_to_checkerboard_world':
                   {'input_scale': .1, 'R': a.tolist(), 't': [10., 20., 30.]}}
    (root / '20260101/20260101.json').write_text(json.dumps(calibration))
    metadata = tmp_path / 'pressure'
    alignment = json.loads((metadata / 'alignment_audit.json').read_text())
    for recording in alignment:
        for camera in recording['cameras']:
            camera['pressure_epoch_wall_s'] = 9999.99
    (metadata / 'alignment_audit.json').write_text(json.dumps(alignment))
    original = adapter.read_rgb_sources
    accessed = []
    def allowed_sources(directory, camera):
        assert camera in (1, 2, 4)
        accessed.append(camera)
        return original(directory, camera)
    monkeypatch.setattr(adapter, 'read_rgb_sources', allowed_sources)
    config = {'b1_version': 'anysole.b1.v1', 'dataset_root': str(root), 'split_csv': str(split),
              'pressure_metadata_root': str(metadata), 'clock_anchor_camera': 1}
    meta = adapter.build_grouped_manifest(config, manifest, overwrite=True)
    return root, manifest, meta, config, calibration, accessed


def test_b1_manifest_filters_cam3_before_io_and_enforces_disjoint_roles(tmp_path, monkeypatch):
    root, manifest, meta, config, _, accessed = _b1_imported(tmp_path, monkeypatch)
    records, _ = adapter.load_grouped_manifest(manifest)
    assert accessed == [1, 2, 4] * 2
    assert meta['original_split_counts'] == {'train': 1, 'val': 1, 'test': 0}
    assert all(len(r['split_roles']) == 1 and r['eligible'] for r in records)
    assert all(r['clock_mapping']['anchor_camera'] == 1 for r in records)
    assert all(not r['clock_mapping']['cam3_image_or_feature_access'] for r in records)
    assert all([v['camera_id'] for v in r['views']] == [1, 2, 4] for r in records)
    assert adapter.grouped_window_inventory(manifest)['splits']['train']['cam3_windows'] == 0
    (tmp_path / 'splits.csv').write_text('train,val,test\nS5011,S13011,S13011\n')
    with pytest.raises(ValueError, match='disjoint'):
        adapter.build_grouped_manifest(config, root / 'bad.jsonl')


def test_b1_composed_calibration_roundtrip_and_static_camera_collation(tmp_path, monkeypatch):
    root, manifest, meta, _, cal, _ = _b1_imported(tmp_path, monkeypatch)
    from anysole.utils.geometry import calibrated_camera_geometry_np
    from anysole.data.rgb_smpl.coordinates import gt_smpl_to_checkerboard, checkerboard_to_camera
    geometry = calibrated_camera_geometry_np(cal)
    points = np.asarray([[.1, .2, .3], [-.4, 1., 2.]])
    for i, camera in enumerate((1, 2, 4)):
        rotation, translation = geometry['camera_R_cw'][i], geometry['camera_t_cw_m'][i]
        actual = points @ rotation.T + translation
        expected = checkerboard_to_camera(gt_smpl_to_checkerboard(points, cal), cal['cameras'][f'cam{camera}'])
        np.testing.assert_allclose(actual, expected, atol=5e-7)
        np.testing.assert_allclose((actual - translation) @ rotation, points, atol=5e-7)
    np.testing.assert_allclose(geometry['camera_geometry'][0, :9].reshape(3, 3), np.eye(3), atol=1e-7)
    np.testing.assert_allclose(geometry['camera_geometry'][0, 9:12], 0, atol=1e-7)
    cache = _cache(tmp_path, manifest, cameras=(1, 2, 4))
    index = json.loads((cache / 'index.json').read_text())
    index.update(manifest_sha256=meta['manifest_sha256'], complete=True)
    (cache / 'index.json').write_text(json.dumps(index))
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints', _rest)
    one = GroupedViewDataset(manifest, root, cache, tmp_path/'unused.pkl', cameras=(1,))[0]
    all_views = GroupedViewDataset(manifest, root, cache, tmp_path/'unused.pkl', cameras=(1,2,4))[0]
    batch = grouped_collate([one, all_views])
    assert batch['visual']['camera_R_cw'].shape == (2,3,3,3)
    assert batch['visual']['camera_geometry'].shape == (2,3,16)
    assert batch['visual']['reference_R_cw'].shape == (2,3,3)
    # Group truth stays in world coordinates; root rotation is still identity.
    np.testing.assert_array_equal(one['labels']['pose6d'][0,:6], [1,0,0,0,1,0])
    torch.testing.assert_close(one['labels']['pressure'], all_views['labels']['pressure'])
    with pytest.raises(ValueError, match='cam3'):
        GroupedViewDataset(manifest, root, cache, tmp_path/'unused.pkl', cameras=(3,))
    cal['cameras']['cam1']['R'] = np.diag([-1.,1.,1.]).tolist()
    with pytest.raises(ValueError, match='right-handed'):
        calibrated_camera_geometry_np(cal)


def test_b1_statistics_fit_camera_root_and_train_only_pressure(tmp_path, monkeypatch):
    root, manifest, _, _, calibration, _ = _b1_imported(tmp_path, monkeypatch)
    records, meta = adapter.load_grouped_manifest(manifest)
    record = next(r for r in records if r['split_roles'] == ['train'])
    path = root / record['pressure']['path']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ('left48', 'right48'):
        arrays[key] = np.zeros((30, 48), np.float32)
        arrays[key][15:] = 1023.
    np.savez(path, **arrays)
    record['pressure']['sha256'] = _hash(path)
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    meta['manifest_sha256'] = _hash(manifest)
    manifest.with_suffix('.meta.json').write_text(json.dumps(meta))
    model = tmp_path / 'model.pkl'
    model.write_bytes(b'explicit model fixture')
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints', _rest)
    first = adapter.fit_grouped_stats(manifest, root, model)
    from anysole.utils.geometry import calibrated_camera_geometry_np, rotmat_to_6d_np
    rotations = calibrated_camera_geometry_np(calibration)['camera_R_cw']
    np.testing.assert_allclose(first['pose_mean'][:6], rotmat_to_6d_np(rotations).mean(0), atol=1e-7)
    assert first['pose_count'] == 90 and first['velocity_count'] == 87
    assert first['statistics_cameras'] == [1, 2, 4]
    pressure = first['pressure_train_statistics']
    assert pressure['recordings'] == ['20260101/S5011']
    assert all(0 < p < 1 for p in pressure['contact_rate'])
    assert all(p > 0 for p in pressure['pressure_log_total_scale'])
    (root / '20260101/Mocap/S13011/motion_neutral_smpl.npz').write_bytes(b'heldout unavailable')
    (root / '20260101/Tactile/S13011/pressure_48.npz').write_bytes(b'heldout unavailable')
    assert adapter.fit_grouped_stats(manifest, root, model) == first
