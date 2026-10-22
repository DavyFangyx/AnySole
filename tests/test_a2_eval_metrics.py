"""Independent numerical fixtures for the A2 evaluation additions."""
import copy

import numpy as np
import pytest

from anysole.utils.a2eval_metrics import (consensus_variance,match_events,pressure_diagnostics,
    fit_train_thresholds,file_sha,verify_thresholds)
from anysole.utils.a2eval_interventions import build_manifest,apply_intervention


def item(n=12):
    pattern=np.linspace(.01,.03,96)
    gt=np.arange(n)[:,None]*pattern
    return {'pressure':gt.copy(),'gt_pressure':gt,'valid':np.ones(n,bool),
        'pressure_valid':np.ones((n,96),bool),'frame_id':np.arange(n),'time_s':np.arange(n)/40,
        'edges':np.ones(n-1,bool),'pressure_edges':np.ones((n-1,96),bool)}


def thresholds():
    return {'active_threshold_q25':[.05]*96,'highload_threshold_q90':[.15]*96,
        'contact_threshold_normalized':[2,2]}


def stats():
    return {'pressure_eval':{'fit_split':'train','fitted_from_train_sha256':'fixture',
        'contact_threshold_normalized':[2,2],'low_load_threshold_normalized':[1,1]}}


def test_centered_mse_removes_cell_offset_and_amplitude_ratio_is_unbiased():
    x=item();x['pressure']+=np.arange(96)[None]/100
    m=pressure_diagnostics(x,thresholds(),stats())
    assert m['pressure_centered_mse']['value']==pytest.approx(0,abs=1e-28)
    assert m['pressure_temporal_std_ratio']['value']==pytest.approx(1)
    assert m['pressure_mae']['value']>0


def test_centered_mse_detects_temporal_inversion_despite_equal_means():
    x=item();x['pressure']=x['gt_pressure'][::-1].copy()
    m=pressure_diagnostics(x,thresholds(),stats())
    expected=np.square(x['pressure']-x['gt_pressure']).mean()
    assert m['pressure_centered_mse']['value']==pytest.approx(expected)
    assert m['left_temporal_corr']['value']==pytest.approx(-1)


def test_constant_gt_and_no_activity_remain_explicit_na():
    x=item();x['gt_pressure'][:]=0;x['pressure'][:]=.2
    m=pressure_diagnostics(x,thresholds(),stats())
    assert m['pressure_temporal_std_ratio']['value'] is None
    assert m['left_temporal_corr']['value'] is None
    assert m['pressure_active_rmse']['value'] is None
    assert m['left_onset_event_f1']['value'] is None


def test_active_rmse_and_no_prediction_clipping():
    x=item();x['pressure'][:]=-2
    m=pressure_diagnostics(x,thresholds(),stats())
    keep=x['gt_pressure']>.05
    assert m['pressure_active_rmse']['value']==pytest.approx(np.sqrt(np.square(x['pressure'][keep]-x['gt_pressure'][keep]).mean()))
    assert m['pressure_out_of_range_fraction']['value']==1


def test_equal_view_decomposition_matches_hand_computation():
    y=np.zeros((2,3));z=np.stack([y+1,y+3]);mask=np.ones_like(y,bool)
    r=consensus_variance(z,y,mask)
    assert r['mse']==5 and r['consensus_error_mse']==4 and r['cross_view_variance']==1
    mask[0,0]=False
    assert consensus_variance(z,y,mask)['valid_cells']==5


def test_event_matching_maximizes_pairs_before_minimizing_error():
    # Nearest-greedy p0->g1 would discard p1. The optimum matches both.
    assert match_events([.09,.18],[0,.10])==[(0,0),(1,1)]
    assert match_events([0],[.100001])==[]
    assert match_events([0],[.1])==[(0,0)]


def test_event_transitions_and_missing_gap_censoring():
    x=item(16);x['gt_pressure'][:]=0;x['pressure'][:]=0
    x['gt_pressure'][4:9]=.1;x['pressure'][5:10]=.1
    m=pressure_diagnostics(x,thresholds(),stats())
    assert m['left_onset_event_f1']['value']==1
    assert m['left_onset_event_time_mae_ms']['value']==pytest.approx(25)
    assert m['left_offset_event_time_mae_ms']['value']==pytest.approx(25)
    x['edges'][3]=False
    m=pressure_diagnostics(x,thresholds(),stats())
    assert m['left_onset_event_f1']['value'] is None
    assert m['left_onset_event_false']['value']==1


def test_events_never_match_across_a_gap_or_manufacture_boundary_transitions():
    x=item(16);x['gt_pressure'][:]=0;x['pressure'][:]=0
    x['gt_pressure'][4:]=.1;x['pressure'][6:]=.1;x['edges'][4]=False
    m=pressure_diagnostics(x,thresholds(),stats())
    assert m['left_onset_event_f1']['value']==0
    assert m['left_onset_event_missed']['value']==1
    assert m['left_offset_event_f1']['value'] is None


def unit(recording,camera=1,action='walk',n=8):
    valid=np.ones(n,bool);valid[2]=False
    return {'recording':recording,'camera':camera,'action':action,'features':np.arange(n*2051,dtype=np.float32).reshape(n,2051),
        'valid':valid,'time_s':np.arange(n)/40,'frame_id':np.arange(n)}


def test_shared_interventions_touch_only_valid_rgb_and_are_reproducible():
    units={'r1/cam1':unit('r1'),'r2/cam1':unit('r2',n=11)}
    units['r2/cam1']['features']+=100000
    original=copy.deepcopy(units);manifest=build_manifest(units)
    assert manifest==build_manifest(units)
    for kind in ('correct','repeat-first','time-shuffle','wrong-recording'):
        transformed=apply_intervention(units,manifest,kind)
        for key,u in units.items():
            np.testing.assert_array_equal(transformed[key][~u['valid']],u['features'][~u['valid']])
            for field in u:np.testing.assert_array_equal(u[field],original[key][field])
    x=apply_intervention(units,manifest,'repeat-first')['r1/cam1']
    np.testing.assert_array_equal(x[units['r1/cam1']['valid']],np.broadcast_to(x[0],(7,2051)))
    wrong=apply_intervention(units,manifest,'wrong-recording')['r1/cam1']
    assert wrong[0,0]>=100000


def test_wrong_recording_needs_a_distinct_recording_and_identity_is_checked():
    units={'r1/cam1':unit('r1'),'r1/cam2':unit('r1',2)}
    manifest=build_manifest(units)
    assert apply_intervention(units,manifest,'wrong-recording')['r1/cam1'] is None
    units['r1/cam1']['time_s'][0]=100
    with pytest.raises(ValueError,match='identity changed'):apply_intervention(units,manifest,'correct')


def test_train_thresholds_use_unique_frames_positive_quantiles_and_refuse_tampering(tmp_path):
    # Four identical cameras must not replicate physical frames in counts.
    raw=np.broadcast_to(np.array([0,100,200,400,800],np.float32)[:,None],(5,96)).copy()
    path=tmp_path/'p.npz';np.savez(path,left48=raw[:,:48],right48=raw[:,48:])
    record={'recording_id':'train','eligible':True,'split_roles':['train'],'target_time_s':(np.arange(5)/40).tolist(),
        'pressure':{'path':'p.npz','sha256':file_sha(path),'normalizer':'clip_0_1023_div_1023','foot_valid':np.ones((5,2),bool).tolist()},
        'pressure_foot_edges':np.ones((4,2),bool).tolist(),
        'views':[{'view_valid':[True]*5,'view_edges':[True]*4} for _ in range(4)]}
    policy={'pressure_eval':{'fit_split':'train','fitted_from_train_sha256':'fixture','contact_threshold_normalized':[10,10]}}
    r=fit_train_thresholds([record],tmp_path,policy)
    normalized=(raw/np.float32(1023)).astype(np.float64)
    assert r['pressure_valid_count']==[5]*96
    assert r['positive_count']==[4]*96
    assert r['active_threshold_q25'][0]==pytest.approx(np.quantile(normalized[1:,0],.25))
    assert r['pressure_cell_mean'][0]==pytest.approx(normalized[:,0].mean())
    verify_thresholds(r)
    r['pressure_cell_mean'][0]+=1
    with pytest.raises(ValueError,match='identity mismatch'):verify_thresholds(r)
    record['split_roles']=['train','test']
    with pytest.raises(ValueError,match='overlaps'):fit_train_thresholds([record],tmp_path,policy)
