import numpy as np, cv2, json, os, glob
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc.reshape(3,1),K,D)
UV=u.reshape(len(J),-1,2)
N=len(J)
# --- DIRECTION: pick the fastest-motion frames and print proj vs kp at several lags ---
kc=np.median(kp[:,H],axis=1); v=np.linalg.norm(np.gradient(kc,axis=0),axis=1)
cand=np.where(gate&(np.arange(N)>30)&(np.arange(N)<N-30))[0]
fast=cand[np.argsort(-v[cand])][:3]
print('=== direction check on the 3 fastest frames ===')
print('  (all joints in image px; PIDX = SMPL joint idx, HIDX = HALPE idx)')
for i in sorted(fast.tolist()):
    print(' frame %d  kp centroid v=%.1f px/frame  proj pelvis=(%.0f,%.0f)'%(i,v[i],UV[i,0,0],UV[i,0,1]))
    for jn,pi,hi in (('pelvis',0,19),('L_ankle',7,15),('R_ankle',8,16)):
        row='   %-8s proj=(%6.1f,%6.1f)  '%(jn,UV[i,pi,0],UV[i,pi,1])
        for L in (0,10):
            row+=' kp[i+%+d]=(%6.1f,%6.1f) d=%5.1f  '%(L,kp[i+L,hi,0],kp[i+L,hi,1],np.hypot(UV[i,pi,0]-kp[i+L,hi,0],UV[i,pi,1]-kp[i+L,hi,1]))
        print(row)
# --- INTRA-SESSION CONSTANCY: split into thirds ---
print('\n=== intra-session constancy (thirds) ===')
def kp_at(delta):
    f=np.arange(N)+delta; lo=np.clip(np.floor(f).astype(int),0,N-1); hi=np.clip(lo+1,0,N-1); w=(f-lo)[:,None,None]
    return kp[lo]*(1-w)+kp[hi]*w,(f>=0)&(f<N-1)
for lo,hi,nm in ((0,181,'frames 0-180'),(181,362,'181-361'),(362,543,'362-542')):
    seg=np.zeros(N,bool); seg[lo:hi]=True
    best=None
    for d in np.arange(-2,20.01,0.5):
        K2,ok0=kp_at(d); ok=gate&ok0&seg
        if ok.sum()<15: continue
        O=UV[ok][:,S]-K2[ok][:,H]; dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
        a=np.median(np.hypot(dx,dy))
        if best is None or a<best[1]: best=(d,a,ok.sum())
    # value at lag 0 for the same segment
    K2,ok0=kp_at(0); ok=gate&ok0&seg; O=UV[ok][:,S]-K2[ok][:,H]
    a0=np.median(np.hypot(np.median(O[...,0],axis=1),np.median(O[...,1],axis=1)))
    print('  %-12s gate=%3d  lag*=%+5.1f f  align %.1f -> %.1f px'%(nm,best[2],best[0],a0,best[1]))
# --- session-level summary from the multi scan ---
res=json.load(open('/tmp/r2-scratch/multi_lag.json'))
interior=[r for r in res if -4<r['lag']<20]
clean=[r for r in interior if r['align']<12]
print('\n=== multi-session (140) ===')
print('  sessions with an INTERIOR lag*: %d / 140'%len(interior))
print('  of those, align* <12 px (a time shift really fixes them): %d'%len(clean))
print('  their lag*: median %+.1f  IQR [%+.1f,%+.1f]  min %+.1f max %+.1f'%(
  np.median([r['lag'] for r in clean]),np.percentile([r['lag'] for r in clean],25),np.percentile([r['lag'] for r in clean],75),
  min(r['lag'] for r in clean),max(r['lag'] for r in clean)))
print('  their align0 median %.1f -> align* median %.1f'%(np.median([r['align0'] for r in clean]),np.median([r['align'] for r in clean])))
print('  sessions already clean at lag 0 (align0<=8px): %d'%sum(1 for r in res if r['align0']<=8))
print('  S5091:',[ (r['lag'],round(r['align0'],1),round(r['align'],1)) for r in res if 'S5091' in r['label']])
