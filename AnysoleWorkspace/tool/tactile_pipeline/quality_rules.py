"""Native pressure audit rules; statistical flags never change measurements."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CELL_FLAGS = {'INVALID_VALUE':1, 'HAMPEL_CANDIDATE':2, 'WHOLE_RECORD_ZERO':4,
 'WHOLE_RECORD_CONSTANT_POSITIVE':8, 'PERSISTENT_POSITIVE_FLOOR':16,
 'MODEL_1023_CLIPPING_RISK':32, 'LEGACY_CHANNEL_ADVISORY':64,
 'IDENTICAL_DYNAMIC_CHANNEL':128}
ROW_FLAGS = {'INVALID_PRESSURE':1, 'NO_WALL_CLOCK':2, 'CONFLICTING_TIME_BATCH':4,
 'BACKWARD_TIME':8, 'GAP_ADJACENT':16}

def clock_flags(t, v):
    n=len(t);amb=np.zeros(n,bool);back=np.zeros(n,bool);gap=np.zeros(n,bool)
    finite=np.isfinite(t)
    if n>1:
        dt=np.diff(t);pair=finite[:-1]&finite[1:]
        back[1:]=pair&(dt<0)
        g=pair&(dt>.3);gap[1:]|=g;gap[:-1]|=g
    # Group globally by receive timestamp, even if repeated non-contiguously.
    ids=np.flatnonzero(finite)
    if len(ids):
        order=ids[np.argsort(t[ids],kind='stable')]
        for block in np.split(order,np.flatnonzero(np.diff(t[order])!=0)+1):
            if len(block)>1 and not np.all(v[block]==v[block[0]]):amb[block]=True
    return amb,back,gap

def hampel(v,t):
    n=len(v);out=np.zeros(v.shape,bool)
    if n<5:return out
    windows=sliding_window_view(v,5,axis=0)
    med=np.median(windows,axis=-1)
    mad=1.4826*np.median(abs(windows-med[:,:,None]),axis=-1)
    candidate=abs(v[2:-2]-med)>np.maximum(20,3*mad)
    valid=np.isfinite(windows).all(axis=-1)&(windows>=0).all(axis=-1)
    if np.isfinite(t).any():
        tw=sliding_window_view(t,5);d=np.diff(tw,axis=1)
        valid &= (np.isfinite(tw).all(1)&(d>=0).all(1)&(d<=.3).all(1))[:,None]
    out[2:-2]=candidate&valid
    return out

def longest_run(x):
    if not len(x):return 0
    edges=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1,len(x)]
    return int(np.diff(edges).max())

def analyze(v,t,wall,legacy):
    n=len(v);valid=np.isfinite(v)&(v>=0)
    amb,back,gap=clock_flags(t,v)
    spikes=hampel(v,t)
    floor=np.full(48,np.nan);p95=np.full(48,np.nan)
    zero=np.zeros(48,bool);constant=np.zeros(48,bool);persistent=np.zeros(48,bool)
    if n:
        floor=np.nanmin(np.where(valid,v,np.nan),axis=0)
        p95=np.nanquantile(np.where(valid,v,np.nan),.95,axis=0)
        zero=valid.all(0)&(v==0).all(0)
        constant=valid.all(0)&(v==v[0]).all(0)&(v[0]>0)
        persistent=(floor>=20)&(floor>=.8*p95)
    duplicate=np.zeros(48,bool)
    if n>=5:
        dynamic=valid.all(0)&np.any(v!=v[0],axis=0)
        ids=np.flatnonzero(dynamic)
        if len(ids):
            _,inverse,counts=np.unique(v[:,ids].T,axis=0,return_inverse=True,return_counts=True)
            duplicate[ids]=counts[inverse]>1
    clip=valid&(v>1023)
    cell=np.zeros(v.shape,np.uint16)
    for mask,bit in [(~valid,1),(spikes,2),(zero[None,:],4),
        (constant[None,:],8),(persistent[None,:],16),(clip,32),
        (legacy[None,:],64),(duplicate[None,:],128)]:
        cell |= np.broadcast_to(mask,v.shape).astype(np.uint16)*bit
    rf=np.zeros(n,np.uint16)
    value_row=valid.all(1)
    for mask,bit in [(~value_row,1),(~np.isfinite(wall),2),(amb,4),(back,8),(gap,16)]:rf|=mask.astype(np.uint16)*bit
    align=value_row&np.isfinite(wall)&np.isfinite(t)&~amb&~back
    temporal=align&~gap
    segments=np.full(n,-1,np.int64);last=-2;seg=-1
    for i in np.flatnonzero(temporal):
        if i!=last+1 or t[i]<=t[last] or t[i]-t[last]>.3:seg+=1
        segments[i]=seg;last=i
    return dict(cell_valid=valid,cell_review_flags=cell,row_quality_flags=rf,
        row_value_usable=value_row,row_alignment_eligible=align,
        row_temporal_eligible=temporal,temporal_segment_id=segments,
        ambiguous_receive_batch=amb,backward_time_step=back,receive_gap_adjacent=gap,
        statistical_spike_candidate=spikes,channel_floor=floor,channel_p95=p95,
        channel_all_zero=zero,channel_constant_positive=constant,
        channel_persistent_floor=persistent,channel_identical_dynamic=duplicate)
