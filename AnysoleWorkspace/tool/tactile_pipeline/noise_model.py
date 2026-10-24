"""Isolated-impulse treatment only; no adaptive smoothing.

y = underlying loading + baseline + short-scale residual + isolated impulse.
Second differences provide a residual-scale proxy, not observed sensor truth.
No baseline is inferred from standing minima or transferred between recordings.
"""
import numpy as np

SETTINGS = {'model': 'robust short-scale residual proxy for isolated-impulse thresholds only', 'scale': '1.4826 * MAD(second_difference) / sqrt(6), strict temporal segments', 'impulse_threshold': 'max(20, 6*sigma) AND above positive-load envelope if >=10 positive modeled samples', 'neighbor_agreement': 'abs(previous-next)<=max(20,3*sigma)', 'load_envelope': 'max(2*median_positive, median_positive+6*MAD_positive); protect ordinary-amplitude sparse load pulses', 'edge_guard': 'protect large foot-total transitions when >=3 channels coherently change beyond max(20,4*sigma); also protect broad nonzero-channel activation/deactivation', 'smoothing': 'disabled; no adaptive or other continuous smoothing executed', 'no_new_activation': True, 'no_clock_policy': 'retain values; do not assume sampling frequency', 'boundaries': 'do not modify segment endpoints or invalid/gap/ambiguous rows', 'baseline': 'handled separately by unchanged airborne baseline core', 'primary_candidate': 'impulse_only', 'alternatives': ['raw_preserved'], 'parameter_selection': 'same v3 impulse thresholds and guards; no new tuning; only smoothing removed', 'limitations': 'residual includes real motion; impulses may include real impacts; offline future-neighbor access', 'status': 'no_smoothing_revision', 'physical_truth_available': False, 'production_promoted': False, 'physical_noise_distribution': 'not identified; no fitted Gaussian model or Gaussian efficacy claim', 'residual_scale_interpretation': 'Inherited MAD/sqrt(6) scale is a fixed threshold heuristic, not a measured sensor noise distribution'}

def segments(ids):
    for label in np.unique(ids[ids>=0]):
        rows=np.flatnonzero(ids==label)
        if len(rows)>=5:yield rows

def mad(x,axis=0):
    return 1.4826*np.median(np.abs(x-np.median(x,axis=axis,keepdims=True)),axis=axis)

def estimate(values,segment_ids):
    blocks=list(segments(segment_ids));diffs=[]
    for rows in blocks:diffs.append(np.diff(values[rows],n=2,axis=0))
    if not diffs:return np.zeros(48),0
    d=np.concatenate(diffs)
    return mad(d)/np.sqrt(6),len(d)

def process(values,segment_ids):
    original=np.asarray(values,dtype=float)
    impulse=original.copy()
    sigma,scale_samples=estimate(original,segment_ids)
    eligible=np.concatenate(list(segments(segment_ids))) if any(True for _ in segments(segment_ids)) else np.array([],int)
    envelope=np.zeros(48)
    if len(eligible):
        for c in range(48):
            positive=original[eligible,c];positive=positive[positive>0]
            if len(positive)>=10:
                center=float(np.median(positive))
                envelope[c]=max(2*center,center+6*float(mad(positive)))
    reason=np.zeros(original.shape,np.uint8)
    modeled_rows=np.zeros(len(original),bool)
    edge_rows=np.zeros(len(original),bool)
    for rows in segments(segment_ids):
        x=original[rows];modeled_rows[rows]=True
        total=x.sum(1);foot_delta=np.diff(total)
        edge_threshold=max(50.,4*float(mad(foot_delta)))
        changes=np.diff(x,axis=0)
        significant=np.abs(changes)>np.maximum(20,4*sigma)
        direction=np.sign(foot_delta)
        coherent=(significant&(np.sign(changes)==direction[:,None])).sum(1)>=3
        active_threshold=np.maximum(20,4*sigma)
        activations=((x[1:]>active_threshold)!=(x[:-1]>active_threshold)).sum(1)>=3
        foot_edge=((np.abs(foot_delta)>edge_threshold)&coherent)|activations
        edge=np.zeros(len(rows),bool);edge[1:]|=foot_edge;edge[:-1]|=foot_edge
        edge_rows[rows]=edge
        prev=x[:-2];center=x[1:-1];nxt=x[2:]
        neighbor=(prev+nxt)/2
        isolated=((center-prev)*(center-nxt)>0)&(np.abs(center-neighbor)>np.maximum(20,6*sigma))
        isolated &= np.abs(prev-nxt)<=np.maximum(20,3*sigma)
        isolated &= ~edge[1:-1,None]
        # Never turn recorded zero into newly invented positive contact.
        isolated &= center>0
        isolated &= center>envelope[None,:]
        filtered=x.copy();filtered[1:-1]=np.where(isolated,neighbor,center)
        impulse[rows]=filtered;reason[rows[1:-1]]|=isolated.astype(np.uint8)
    return dict(impulse_only=impulse,sigma=sigma,
        scale_samples=scale_samples,load_envelope48=envelope,change_reason=reason,modeled_rows=modeled_rows,edge_rows=edge_rows)
