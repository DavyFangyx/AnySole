"""Portable core extracted without changing processing functions."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def scan_steps(x,t,segment,session,side):
    n=len(x);w=8;events=[]
    # Detection includes row-order-only blocks but correction later requires trusted clock.
    blocks=[]
    if np.any(segment>=0):
        for s in np.unique(segment[segment>=0]):blocks.append((np.flatnonzero(segment==s),True))
    elif n:blocks=[(np.arange(n),False)]
    for rows,clock in blocks:
        if len(rows)<2*w+8:continue
        block=x[rows]
        d2=np.diff(block,n=2,axis=0);sigma=1.4826*np.median(abs(d2-np.median(d2,axis=0)),axis=0)/np.sqrt(6)
        sw=sliding_window_view(block,w,axis=0)
        quant=np.quantile(sw,[.1,.5,.9],axis=-1)
        count=len(block)-2*w+1
        left=quant[:,:count];right=quant[:,w:w+count];shift=right-left;delta=shift[1]
        spread=np.max(abs(shift-delta[None,:,:]),axis=0)
        good=(abs(delta)>np.maximum(50,8*sigma))&(spread<=np.maximum(15,.3*abs(delta)))
        for c in np.flatnonzero(good.any(0)):
            candidates=np.flatnonzero(good[:,c]);order=candidates[np.argsort(abs(delta[candidates,c]))[::-1]]
            kept=[]
            for j in order:
                if any(abs(int(j)-k)<w for k in kept):continue
                kept.append(int(j));boundary=int(j+w);global_row=int(rows[boundary]);dv=float(delta[j,c])
                before=block[j:j+w,c];after=block[boundary:boundary+w,c]
                # Confirm a new level through the following additional window.
                post2=block[boundary+w:min(len(block),boundary+3*w),c]
                persistent=len(post2)>=w and abs(float(np.median(post2))-float(np.median(after)))<=max(20,.3*abs(dv))
                other=np.delete(block, c, axis=1)
                pre_other=other[j:j+w].sum(1);post_other=other[boundary:boundary+w].sum(1)
                other_delta=abs(float(np.median(post_other)-np.median(pre_other)))
                other_level=max(float(np.median(pre_other)),float(np.median(post_other)),1)
                independent_channel=other_delta<=max(100,.15*other_level)
                active_pre=float(np.median((other[j:j+w]>0).sum(1)))
                active_post=float(np.median((other[boundary:boundary+w]>0).sum(1)))
                edge_row=global_row
                # Refine exact change using local first differences of this channel.
                lo=max(int(rows[0])+1,global_row-w//2);hi=min(int(rows[-1]),global_row+w//2)
                if hi>=lo:
                    dd=np.diff(x[lo-1:hi+1,c]);signed=dd*np.sign(dv)
                    edge_row=lo+int(np.argmax(signed))
                tail_end=min(len(x),edge_row+24)
                evidence='persistent_single_channel_translation' if persistent and independent_channel else 'loading_or_step_ambiguous'
                events.append(dict(session=session,side=side,channel=int(c+1),boundary_row0=edge_row,
                    candidate_window_center_row0=global_row,shift=dv,pre_q10=float(np.quantile(before,.1)),
                    pre_median=float(np.median(before)),post_q10=float(np.quantile(after,.1)),post_median=float(np.median(after)),
                    quantile_translation_spread=float(spread[j,c]),persistent_following_window=persistent,
                    other_channels_stable=independent_channel,other_channels_median_delta=other_delta,
                    active_other_channels_pre=active_pre,active_other_channels_post=active_post,
                    trusted_time=clock,signal_evidence=evidence,physical_fault_confirmed=False))
    return events
