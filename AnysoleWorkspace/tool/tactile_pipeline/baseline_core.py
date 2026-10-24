"""Portable core extracted without changing processing functions."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def mad(x,axis=0):return 1.4826*np.median(abs(x-np.median(x,axis=axis,keepdims=True)),axis=axis)

def fit_channel_baseline(raw,anchors,eligible):
    n=len(raw);fit=np.unique(np.concatenate([a['fit_rows'] for a in anchors])).astype(int) if anchors else np.array([],int)
    if not len(fit):return np.zeros(48),[],fit
    x=raw[fit];med=np.median(x,axis=0);total=float(np.median(x.sum(1)))
    order=np.argsort(med)[::-1];top6=order[:6]
    if total<=100 or med[top6].sum()<.6*max(med.sum(),1):return np.zeros(48),[],fit
    fraction=(x>0).mean(0);scale=mad(x)
    stable=(med>=20)&(fraction>=.8)&(scale<=np.maximum(10,.35*med))
    # Single human center: use nearby raw samples only to test channel stability,
    # not as additional labelled air observations or zero-calibration values.
    if len(fit)==1:
        center=fit[0];near=np.arange(max(0,center-2),min(n,center+3));near=near[eligible[near]]
        if len(near)<3:return np.zeros(48),[],fit
        local=raw[near];stable&=(local>0).mean(0)>=.8
        stable&=mad(local)<=np.maximum(10,.2*med)
    keep=[int(c) for c in top6 if stable[c]]
    baseline=np.zeros(48)
    if keep:baseline[keep]=.9*np.quantile(x[:,keep],.1,axis=0)
    return baseline,keep,fit

def make_baseline(raw,segment,eligible,anchors,events):
    b0,channels,fit=fit_channel_baseline(raw,anchors,eligible)
    constant=np.broadcast_to(b0,raw.shape).copy();piece=constant.copy();applied=[]
    # A step correction requires airborne observations on BOTH sides of the boundary.
    # Unlabelled persistent steps remain candidates; they may be actual foot loading.
    for c in channels:
        b=float(b0[c]);candidates=[]
        for a in anchors:
            # Existing AI interval evidence is weaker than human confirmation and
            # stays explicitly tiered; no extra surrounding human air labels inferred.
            if a['tier'] not in ('AI_reviewed_interval_not_human','synthetic_known_air_interval'):continue
            allrows=np.unique(a['fit_rows']+a['holdout_rows']).astype(int)
            if len(allrows)<12:continue
            half=len(allrows)//2;pre=raw[allrows[:half],c];post=raw[allrows[half:],c]
            delta=float(np.median(post)-np.median(pre))
            residual=np.r_[pre-np.median(pre),post-np.median(post)]
            scatter=float(mad(residual))
            if abs(delta)<=max(20,4*scatter):continue
            lo=int(allrows[max(2,len(allrows)//4)]);hi=int(allrows[min(len(allrows)-3,3*len(allrows)//4)])
            dd=np.diff(raw[lo-1:hi+1,c]);boundary=lo+int(np.argmax(dd*np.sign(delta)))
            if not eligible[boundary] or boundary<=allrows[1] or boundary>=allrows[-2]:continue
            local_pre=raw[max(int(allrows[0]),boundary-3):boundary,c]
            local_post=raw[boundary:min(int(allrows[-1])+1,boundary+3),c]
            if min(len(local_pre),len(local_post))<3:continue
            local_shift=float(np.median(local_post)-np.median(local_pre))
            # A difference between interval halves alone may be gradual drift or
            # mixed fluctuations. Require an actual sustained local edge.
            if local_shift*np.sign(delta)<max(20,.5*abs(delta)):continue
            candidates.append(dict(session=a.get('session',''),side=a.get('side',''),channel=c+1,
                boundary_row0=boundary,shift=delta,anchor_id=a['anchor_id'],scatter=scatter,
                tier=a['tier'],air_pre_rows=half,air_post_rows=len(allrows)-half))
        filtered=[]
        for e in sorted(candidates,key=lambda e:abs(float(e['shift'])),reverse=True):
            if not any(abs(int(e['boundary_row0'])-int(old['boundary_row0']))<16 for old in filtered):filtered.append(e)
        if not filtered:continue
        rr=np.arange(len(raw));reference=int(fit[np.argmin(abs(raw[fit,c]-b/.9))]);offset=np.zeros(len(rr))
        for e in sorted(filtered,key=lambda e:int(e['boundary_row0'])):offset[rr>=int(e['boundary_row0'])]+=float(e['shift'])
        offset-=offset[reference]
        piece[:,c]=np.clip(b+offset,0,2*b)
        for e in filtered:
            applied.append(dict(session=e['session'],side=e['side'],channel=c+1,
                boundary_row0=int(e['boundary_row0']),estimated_baseline_jump=float(e['shift']),
                reference_row0=reference,reference_baseline=b,calibration_anchor_id=e['anchor_id'],
                pre_air_samples=e['air_pre_rows'],post_air_samples=e['air_post_rows'],
                within_state_scatter=e['scatter'],evidence=e['tier'],
                confidence='airborne_interval_supported_signal_model_not_physical_fault_certification'))
    constant[~eligible]=0;piece[~eligible]=0
    return constant,piece,applied,b0,fit
