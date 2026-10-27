"""A2 pressure diagnostics and train-only calibration, independent of training.

No dev labels or prediction-dependent statistics are accepted by the fitter.
Pressure is the original float32 clip(raw, 0, 1023)/1023 representation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def thresholds_digest(value):
    """Hash every field except the two explicitly self-referential hash fields."""
    payload=json.loads(json.dumps(value,allow_nan=False))
    payload.pop('thresholds_sha256',None)
    payload.get('science_config',{}).pop('pressure_train_statistics_hash',None)
    return digest(payload)


def finalize_thresholds(value):
    value['thresholds_sha256']=thresholds_digest(value)
    value['science_config']['pressure_train_statistics_hash']=value['thresholds_sha256']
    return value


def verify_thresholds(value):
    expected=thresholds_digest(value)
    if value.get('thresholds_sha256')!=expected or value.get('science_config',{}).get('pressure_train_statistics_hash')!=expected:
        raise ValueError('train thresholds identity mismatch')
    return value


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def load_pressure(record, dataset_root):
    path = Path(dataset_root)/record['pressure']['path']
    if file_sha(path) != record['pressure']['sha256']:
        raise ValueError('pressure source changed: '+str(path))
    with np.load(path, allow_pickle=False) as archive:
        value = np.concatenate([archive['left48'], archive['right48']], axis=1).astype(np.float32)
    valid = np.repeat(np.asarray(record['pressure']['foot_valid'], bool), 48, axis=1)
    if value.shape != valid.shape or not np.isfinite(value[valid]).all():
        raise ValueError('invalid pressure labels')
    value = np.where(valid, value, 0)
    if record['pressure']['normalizer'] == 'clip_0_1023_div_1023':
        value = np.clip(value, 0, 1023)/1023
    elif record['pressure']['normalizer'] != 'identity':
        raise ValueError('unknown pressure normalizer')
    return value, valid


def fit_train_thresholds(records, dataset_root, frozen_stats):
    """Fit once per unique physical train frame, with any valid RGB camera.

    q25/q90 are per-cell positive-value linear quantiles. Adjacent differences
    use both valid ends, 25ms target intervals, and trusted label/source edges.
    Contact uses the unchanged, already train-fitted sum48 threshold.
    """
    policy = frozen_stats['pressure_eval']
    if policy.get('fit_split') != 'train' or not policy.get('fitted_from_train_sha256'):
        raise ValueError('contact policy must be frozen from train')
    values, masks, diffs, diffmasks, support, sources = [], [], [], [], [], []
    train = sorted((r for r in records if r['eligible'] and 'train' in r['split_roles']), key=lambda r:r['recording_id'])
    if not train:
        raise ValueError('no train recordings')
    # Refuse a mixed-role fit: this would silently fit on the development panel.
    if any(set(r['split_roles']) & {'val', 'test'} for r in train):
        raise ValueError('train fit overlaps dev/test roles')
    for record in train:
        value, valid = load_pressure(record, dataset_root)
        any_view = np.asarray([v['view_valid'] for v in record['views']], bool).any(0)
        mask = valid & any_view[:, None]
        times = np.asarray(record['target_time_s'], np.float64)
        label_edges = np.repeat(np.asarray(record['pressure_foot_edges'], bool),48,axis=1)
        rgb_edges = np.asarray([v['view_edges'] for v in record['views']], bool).any(0)
        edge = label_edges & rgb_edges[:,None] & mask[:-1] & mask[1:] & np.isclose(np.diff(times)[:,None], .025, rtol=0, atol=1e-7)
        values.append(value); masks.append(mask)
        diffs.append(np.abs(np.diff(value.astype(np.float64), axis=0))); diffmasks.append(edge)
        support.append({'recording_id':record['recording_id'], 'mask_sha256':array_sha(mask), 'edge_sha256':array_sha(edge)})
        sources.append({'recording_id':record['recording_id'], 'sha256':record['pressure']['sha256']})
    value = np.concatenate(values).astype(np.float64); valid = np.concatenate(masks)
    differences = np.concatenate(diffs); difference_valid = np.concatenate(diffmasks)
    positive = valid & (value > 0)
    q25, q90, mean, median, positive_count = [], [], [], [], []
    for cell in range(96):
        x = value[positive[:,cell],cell]; allx = value[valid[:,cell],cell]
        if not len(allx):
            raise ValueError('train cell has no valid labels')
        positive_count.append(len(x)); mean.append(float(allx.mean())); median.append(float(np.median(allx)))
        q25.append(float(np.quantile(x,.25, method='linear')) if len(x) else None)
        q90.append(float(np.quantile(x,.90, method='linear')) if len(x) else None)
    threshold = np.asarray([np.inf if q is None else q for q in q25])
    active = valid & (value > threshold)
    pi = float(active.sum()/valid.sum())
    alpha = min(4., (1-pi)/max(pi,.01))
    contact_threshold = np.asarray(policy['contact_threshold_normalized'], np.float64)
    contact_rate, pos_weight, contact_count, huber, diff_huber = [], [], [], [], []
    for foot in range(2):
        s = slice(foot*48,(foot+1)*48)
        keep = valid[:,s].all(-1)
        contacts = value[:,s].sum(-1)[keep] > contact_threshold[foot]
        if not len(contacts) or not contacts.any() or contacts.all():
            raise ValueError('contact BCE needs both train classes')
        rate = float(contacts.mean())
        contact_rate.append(rate); pos_weight.append((1-rate)/rate); contact_count.append(len(contacts))
        x = value[:,s][positive[:,s]]
        if not len(x):
            raise ValueError('foot has no positive train values')
        huber.append(max(.001,float(np.median(x))))
        dx = differences[:,s][difference_valid[:,s]]
        diff_huber.append(max(.001,float(np.quantile(dx,.75,method='linear'))) if len(dx) else None)
    result = {'schema':'anysole.a2.train-thresholds.v1', 'fit_split':'train',
        'frame_weighting':'unique_physical_target_frame_with_any_valid_RGB',
        'recordings':[r['recording_id'] for r in train], 'train_support_sha256':digest(support),
        'train_pressure_sources_sha256':digest(sources), 'train_sources':sources,
        'pressure_cell_mean':mean, 'pressure_cell_median':median,
        'pressure_initial_mean':np.maximum(mean,1e-4).tolist(),
        'pressure_valid_count':valid.sum(0).tolist(), 'positive_count':positive_count,
        'active_threshold_q25':q25, 'highload_threshold_q90':q90,
        'no_positive_cells':[i for i,n in enumerate(positive_count) if n==0],
        'active_fraction':pi, 'active_alpha':alpha, 'active_weight_normalizer':1+alpha*pi,
        'huber_delta_per_foot':huber, 'diff_huber_delta_per_foot':diff_huber,
        'diff_valid_count_per_cell':difference_valid.sum(0).tolist(),
        'contact_threshold_normalized':contact_threshold.tolist(),
        'contact_rate':contact_rate,'contact_pos_weight':pos_weight,'contact_valid_count':contact_count,
        'contact_policy':policy, 'numpy_version':np.__version__,
        'definitions':{'active':'GT > per-cell positive q25; no-positive cells never active',
            'highload':'GT > per-cell positive q90; no-positive cells never highload',
            'diff':'both ends valid, same40Hz target interval, any RGB source edge and label edge',
            'contact_pos_weight':'negative_train_count / positive_train_count',
            'huber':'median positive train cell reading per foot, minimum .001',
            'diff_huber':'q75 valid adjacent absolute train cell differences per foot, minimum .001'}}
    result['science_config'] = {'pressure_cell_mean':result['pressure_initial_mean'],
        'pressure_contact_rate':contact_rate, 'pressure_contact_threshold':contact_threshold.tolist(),
        'pressure_contact_pos_weight':pos_weight, 'pressure_active_threshold':q25,
        'pressure_active_alpha':alpha, 'pressure_active_normalizer':1+alpha*pi,
        'pressure_huber_delta':huber, 'pressure_diff_delta':diff_huber}
    result['hash_definition']='SHA256 sorted compact JSON; exclude thresholds_sha256 and science_config.pressure_train_statistics_hash only'
    return finalize_thresholds(result)


def fit_train_root_prior(records,dataset_root,smpl_model_path):
    """Equal-recording mean first valid train pelvis, in native SMPL metres.

    The first target with trustworthy motion and any valid RGB is used once.
    SMPL shape/rest-pelvis correction is exactly the supervised data reader's.
    """
    from anysole.data.smpl_io import load_smpl
    entries=[]
    for r in sorted(records,key=lambda r:r['recording_id']):
        if not r['eligible'] or 'train' not in r['split_roles']:continue
        if set(r['split_roles'])&{'val','test'}:raise ValueError('train root prior overlaps dev/test')
        keep=np.asarray(r['motion']['valid'],bool)&np.asarray([v['view_valid'] for v in r['views']],bool).any(0)
        ix=np.flatnonzero(keep)
        if not len(ix):raise ValueError('train recording lacks valid starting root')
        first=int(ix[0]);path=Path(dataset_root)/r['motion']['path']
        if file_sha(path)!=r['motion']['sha256']:raise ValueError('train motion source changed')
        label=load_smpl(path,np.asarray([r['motion']['time_s'][first]]),model_path=Path(smpl_model_path),allow_extrapolation=False)
        entries.append({'recording':r['recording_id'],'target_frame':int(r['target_frame_id'][first]),
            'motion_time_s':r['motion']['time_s'][first],'source_sha256':r['motion']['sha256'],
            'pelvis_root_m':label['trans_m'][0].astype(np.float64).tolist()})
    if not entries:raise ValueError('no train root starts')
    result={'schema':'anysole.a2.train-root-prior.v1','fit_split':'train','coordinate':'native SMPL +Y up, metres',
        'rule':'one first motion-valid/any-RGB-valid target per train recording; equal-recording mean; native shape rest-pelvis correction',
        'root_prior_m':np.mean([e['pelvis_root_m'] for e in entries],axis=0).tolist(),'entries':entries,
        'smpl_model_sha256':file_sha(smpl_model_path)}
    result['prior_sha256']=digest(result)
    return result


def metric(value, support, unit, *, reason=None, eligible=None):
    support = np.asarray(support).tolist()
    eligible = support if eligible is None else np.asarray(eligible).tolist()
    finite = value is not None and np.isfinite(value)
    return {'value':float(value) if finite else None, 'count':len(support), 'unit':unit,
        'support_hash':digest(support), 'eligible_count':len(eligible),'eligible_support_hash':digest(eligible),
        'reason':None if finite else (reason or 'no valid support')}


EVENT_METRIC_VERSION='anysole.a2.events.v2-roundoff1ns'


def match_events(predicted, truth, tolerance=.100):
    """Monotone one-to-one maximum-cardinality, minimum-total-time matching."""
    predicted, truth = np.asarray(predicted,float), np.asarray(truth,float)
    n,m = len(predicted),len(truth)
    # DP stores (number matches, negative cost); lexicographic optimum.
    count = np.zeros((n+1,m+1),int); cost = np.zeros((n+1,m+1)); move = np.zeros((n+1,m+1),np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options = [(count[i-1,j],cost[i-1,j],1),(count[i,j-1],cost[i,j-1],2)]
            dt = abs(predicted[i-1]-truth[j-1])
            # Absolute seconds-of-day lose precision when subtracted. Keep the
            # physical100ms boundary, with only1ns numerical roundoff allowance.
            if dt <= tolerance+1e-9:
                options.append((count[i-1,j-1]+1,cost[i-1,j-1]-dt,3))
            number, negative, direction = max(options,key=lambda x:(x[0],x[1]))
            count[i,j],cost[i,j],move[i,j] = number,negative,direction
    pairs=[];i,j=n,m
    while i and j:
        if move[i,j] == 3:
            pairs.append((i-1,j-1));i-=1;j-=1
        elif move[i,j] == 1:i-=1
        else:j-=1
    return list(reversed(pairs))


def pressure_diagnostics(item, thresholds, frozen_stats,*,event_records=None):
    """Preserve every canonical40Hz metric and add explicit A2 diagnostics."""
    from anysole.utils.eval_protocol import visual_pressure_metrics
    result = visual_pressure_metrics(item, frozen_stats)
    pred,gt = np.asarray(item['pressure'],np.float64),np.asarray(item['gt_pressure'],np.float64)
    valid = np.asarray(item['pressure_valid'],bool)&np.asarray(item['valid'],bool)[:,None]
    ids,times = np.asarray(item['frame_id']),np.asarray(item['time_s'],np.float64)
    if not np.isfinite(pred[valid]).all():
        raise ValueError('nonfinite valid pressure prediction')
    for suffix,key in (('active','active_threshold_q25'),('highload','highload_threshold_q90')):
        limit=np.asarray([np.inf if x is None else x for x in thresholds[key]])
        mask=valid&(gt>limit)
        support=np.argwhere(mask)
        d=pred[mask]-gt[mask]
        result['pressure_'+suffix+'_rmse']=metric(np.sqrt(np.mean(d*d)) if len(d) else None,support,'normalized_sensor_reading')
        result['gt_'+suffix+'_fraction']=metric(float(mask.sum()/valid.sum()) if valid.any() else None,np.argwhere(valid),'fraction')
    for name,mask in (('zero',valid&(gt==0)),):
        d=pred[mask]-gt[mask]
        result['pressure_'+name+'_rmse']=metric(np.sqrt(np.mean(d*d)) if len(d) else None,np.argwhere(mask),'normalized_sensor_reading')
        result['gt_'+name+'_fraction']=metric(float(mask.sum()/valid.sum()) if valid.any() else None,np.argwhere(valid),'fraction')
    for cell in range(96):
        take=valid[:,cell];d=pred[take,cell]-gt[take,cell]
        result[f'cell{cell:03d}_pressure_mae']=metric(np.abs(d).mean() if len(d) else None,ids[take],'normalized_sensor_reading')
        result[f'cell{cell:03d}_pressure_rmse']=metric(np.sqrt(np.square(d).mean()) if len(d) else None,ids[take],'normalized_sensor_reading')
    centered,amplitude,center_support,amplitude_support = [],[],[],[]
    for cell in range(96):
        take=valid[:,cell]
        if take.any():
            a,b=pred[take,cell],gt[take,cell]
            centered.extend(((a-a.mean())-(b-b.mean()))**2)
            center_support.extend(np.column_stack((ids[take],np.full(take.sum(),cell))).tolist())
            if b.std()>1e-12:
                amplitude.append(a.std()/b.std());amplitude_support.append(cell)
    result['pressure_centered_mse']=metric(np.mean(centered) if centered else None,center_support,'normalized_sensor_reading_squared')
    result['pressure_temporal_std_ratio']=metric(np.mean(amplitude) if amplitude else None,amplitude_support,'ratio',reason='GT constant on all cells')
    result['gt_dynamic_cell_fraction']=metric(len(amplitude_support)/96,list(range(96)),'fraction')
    for foot,name in enumerate(('left','right')):
        s=slice(foot*48,(foot+1)*48);keep=valid[:,s].all(-1)
        cellmask=valid[:,s];d=pred[:,s][cellmask]-gt[:,s][cellmask]
        result[name+'_pressure_mae']=metric(np.abs(d).mean() if len(d) else None,np.argwhere(cellmask),'normalized_sensor_reading')
        result[name+'_pressure_rmse']=metric(np.sqrt(np.square(d).mean()) if len(d) else None,np.argwhere(cellmask),'normalized_sensor_reading')
        for suffix,key in (('active','active_threshold_q25'),('highload','highload_threshold_q90')):
            limit=np.asarray([np.inf if x is None else x for x in thresholds[key]])[s]
            take=cellmask&(gt[:,s]>limit)
            delta=pred[:,s][take]-gt[:,s][take]
            result[name+'_pressure_'+suffix+'_rmse']=metric(np.sqrt(np.square(delta).mean()) if len(delta) else None,np.argwhere(take),'normalized_sensor_reading')
        if keep.any():
            a,b=pred[keep,s].sum(-1),gt[keep,s].sum(-1)
            result[name+'_load_temporal_std_ratio']=metric(a.std()/b.std() if b.std()>1e-12 else None,ids[keep],'ratio',reason='GT total load constant')
        pp=pred[:,s].sum(-1)>thresholds['contact_threshold_normalized'][foot]
        gg=gt[:,s].sum(-1)>thresholds['contact_threshold_normalized'][foot]
        tp=int((pp&gg&keep).sum());fp=int((pp&~gg&keep).sum());fn=int((~pp&gg&keep).sum());tn=int((~pp&~gg&keep).sum())
        for key,value in (('contact_precision',tp/(tp+fp) if tp+fp else None),('contact_recall',tp/(tp+fn) if tp+fn else None),
                          ('air_recall',tn/(tn+fp) if tn+fp else None),
                          ('balanced_accuracy',.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None)):
            result[name+'_'+key]=metric(value,ids[keep],'fraction')
        edge=np.asarray(item['edges'],bool)&np.asarray(item['pressure_edges'],bool)[:,s].all(-1)&keep[:-1]&keep[1:]
        # Events are observed transitions. Segment starts/ends are censored,
        # never manufactured events at missing data or recording boundaries.
        edge &= np.isclose(np.diff(times),.025,rtol=0,atol=1e-7)
        event_totals={'matched':0,'predicted':0,'truth':0,'signed':[],'pair_support':[],'gt_support':[]}
        for event,before,after in (('onset',False,True),('offset',True,False)):
            pi=np.flatnonzero(edge&(pp[:-1]==before)&(pp[1:]==after))+1
            gi=np.flatnonzero(edge&(gg[:-1]==before)&(gg[1:]==after))+1
            # Events cannot be matched through a source/label gap.
            segments=np.r_[0,np.cumsum(~edge)]
            pairs=[]
            for segment in sorted(set(segments[pi])|set(segments[gi])):
                pis=np.flatnonzero(segments[pi]==segment);gis=np.flatnonzero(segments[gi]==segment)
                pairs.extend((int(pis[a]),int(gis[b])) for a,b in match_events(times[pi[pis]],times[gi[gis]]))
            matched=len(pairs);den=len(pi)+len(gi)
            prefix=name+'_'+event+'_event_'
            result[prefix+'f1']=metric(2*matched/den if len(gi) and den else None,ids[gi],'fraction',reason='no GT events')
            result[prefix+'missed']=metric(len(gi)-matched,ids[gi],'events')
            result[prefix+'false']=metric(len(pi)-matched,ids[pi],'events')
            signed=[(times[pi[a]]-times[gi[b]])*1000 for a,b in pairs]
            pair_support=[[int(ids[pi[a]]),int(ids[gi[b]])] for a,b in pairs]
            if event_records is not None:
                matched_pred={a:b for a,b in pairs};matched_gt={b:a for a,b in pairs}
                for b,index in enumerate(gi):
                    a=matched_gt.get(b)
                    event_records.append({'foot':name,'event':event,'status':'matched' if a is not None else 'missed',
                        'gt_frame':int(ids[index]),'gt_time_s':float(times[index]),
                        'pred_frame':int(ids[pi[a]]) if a is not None else None,
                        'pred_time_s':float(times[pi[a]]) if a is not None else None,
                        'error_ms':float((times[pi[a]]-times[index])*1000) if a is not None else None})
                for a,index in enumerate(pi):
                    if a not in matched_pred:event_records.append({'foot':name,'event':event,'status':'false',
                        'gt_frame':None,'gt_time_s':None,'pred_frame':int(ids[index]),'pred_time_s':float(times[index]),'error_ms':None})
            result[prefix+'time_mae_ms']=metric(np.abs(signed).mean() if signed else None,pair_support,'ms')
            result[prefix+'time_bias_ms']=metric(np.mean(signed) if signed else None,pair_support,'ms')
            event_totals['matched']+=matched;event_totals['predicted']+=len(pi);event_totals['truth']+=len(gi)
            event_totals['signed'].extend(signed)
            event_totals['pair_support'].extend([[event,*p] for p in pair_support])
            event_totals['gt_support'].extend([[event,int(i)] for i in ids[gi]])
        den=event_totals['truth']+event_totals['predicted'];matched=event_totals['matched']
        result[name+'_event_f1']=metric(2*matched/den if event_totals['truth'] and den else None,event_totals['gt_support'],'fraction',reason='no GT events')
        result[name+'_event_time_mae_ms']=metric(np.abs(event_totals['signed']).mean() if event_totals['signed'] else None,event_totals['pair_support'],'ms')
    return result


def consensus_variance(predictions, truth, mask):
    """Equal-view identity on a declared common valid support."""
    z,y,mask=np.asarray(predictions,np.float64),np.asarray(truth,np.float64),np.asarray(mask,bool)
    if z.ndim!=3 or z.shape[1:]!=y.shape or mask.shape!=y.shape:
        raise ValueError('expected views x time x cells on common support')
    if not mask.any() or not np.isfinite(z[:,mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('empty or nonfinite common support')
    center=z.mean(0)
    mse=float(np.square(z-y[None])[:,mask].mean())
    consensus=float(np.square(center-y)[mask].mean())
    variance=float(np.square(z-center[None])[:,mask].mean())
    if not np.isclose(mse,consensus+variance,atol=1e-12,rtol=1e-12):
        raise AssertionError('consensus identity failed')
    return {'mse':mse,'consensus_error_mse':consensus,'cross_view_variance':variance,
        'valid_cells':int(mask.sum()),'support_hash':array_sha(mask),'views':len(z)}
