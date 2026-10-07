import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
rv,_=cv2.Rodrigues(Rc)
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
UV=np.empty(J.shape[:2]+(2,))
for i in range(len(Pc)):
    u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D); UV[i]=u[:,0,:]

# ---------- 1. pelvis trajectory cross-correlation (projected SMPL pelvis vs RTMPose hip mid) ----------
proj_pelvis = UV[:,0]                       # SMPL joint 0
kp_pelvis   = 0.5*(kp[:,H[1]]+kp[:,H[2]])   # l_hip, r_hip midpoint
def xcorr(a,b,maxlag=12):
    out={}
    for L in range(-maxlag,maxlag+1):
        j=np.arange(len(a))+L; ok=gate&(j>=0)&(j<len(a))
        # use only pairs both gate-valid; correlation of the raw (not detrended) signals
        x=proj_pelvis[ok,0]; y=a[ok,0] if False else None
        out[L]=(ok.sum(),)
    return out
print('=== 1. pelvis x/y trajectory cross-correlation, proj@i vs kp@i+L (gate frames) ===')
for axis,name in ((0,'x (image horizontal)'),(1,'y (image vertical)')):
    print(' axis',name)
    rows=[]
    for L in range(-10,11):
        j=np.arange(543)+L; ok=gate&(j>=0)&(j<543)
        a=proj_pelvis[ok,axis]; b=kp_pelvis[j[ok],axis]
        c=np.corrcoef(a,b)[0,1]
        d=np.median(np.abs(a-b))
        rows.append((L,c,d,ok.sum()))
    best=max(rows,key=lambda r:r[1])
    for L,c,d,n in rows:
        if abs(L)<=6: print('   L=%+3d  r=%+.4f  median|err| %5.1f px  n=%d'%(L,c,d,n))
    print('   PEAK at L=%+d (r=%.4f) ; r(L=0)=%.4f'%(best[0],best[1],[c for L,c,_,_ in rows if L==0][0]))

# ---------- 2. shape residual vs lag (limb motion, translation-insensitive) ----------
print('\n=== 2. shape residual (frame-2D-offset removed) vs lag ===')
for L in range(-6,7):
    j=np.arange(543)+L; ok=gate&(j>=0)&(j<543)
    O=UV[ok][:,S]-kp[j[ok]][:,H]
    dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
    R=O.copy(); R[...,0]-=dx[:,None]; R[...,1]-=dy[:,None]
    r=np.linalg.norm(R,axis=2)
    o=np.hypot(dx,dy)
    print('  L=%+3d  shape-residual median %5.1f  p90 %5.1f px | align-offset median %5.1f px'%(L,np.median(r),np.percentile(r,90),np.median(o)))

# ---------- 3. step-phase: pelvis image height (y) detrended, cross-correlation ----------
print('\n=== 3. detrended pelvis-y (step bounce) cross-correlation ===')
def detrend(sig,mask,w=41):
    idx=np.where(mask)[0]; v=sig[idx]; out=np.empty_like(v)
    for i in range(len(v)):
        lo=max(0,i-w//2); hi=min(len(v),i+w//2+1)
        out[i]=v[i]-np.median(v[lo:hi])
    return idx,out
idx_a,da=detrend(proj_pelvis[:,1],gate)
idx_b,db=detrend(kp_pelvis[:,1],gate)
for L in range(-10,11):
    # frames present in both detrended sets, shifted
    j=idx_a+L
    common=np.isin(idx_b,j)
    if common.sum()<50: continue
    ai=np.where(common)[0]; pos=np.searchsorted(idx_b,j[ai])
    a=da[ai]; b=db[pos]
    print('  L=%+3d  r=%+.4f  n=%d  amp(proj) %.1f px'%(L,np.corrcoef(a,b)[0,1],common.sum(),np.std(a)))
