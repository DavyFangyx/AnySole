import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
S=np.array(S); H=np.array(H)
rv,_=cv2.Rodrigues(Rc)
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
UV=np.empty(J.shape[:2]+(2,))
for i in range(len(Pc)):
    u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D); UV[i]=u[:,0,:]
N=543
# per-frame gate for the shifted index: use the intersection of the valid gate frames at i and i+delta
def kp_at(delta):
    f=np.arange(N)+delta
    lo=np.clip(np.floor(f).astype(int),0,N-1); hi=np.clip(lo+1,0,N-1); w=(f-lo)[:,None,None]
    return kp[lo]*(1-w)+kp[hi]*w, (f>=0)&(f<N-1)
print('=== sub-frame lag sweep: proj@i vs interpolated kp@i+delta ===')
print('  delta |  frame-align offset | shape residual |  joint RMS')
res=[]
for delta in np.arange(-2,20.01,0.25):
    K2,ok0=kp_at(delta)
    ok=gate&ok0
    O=UV[ok][:,S]-K2[ok][:,H]
    dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
    R=O.copy(); R[...,0]-=dx[:,None]; R[...,1]-=dy[:,None]
    res.append((delta,np.median(np.hypot(dx,dy)),np.median(np.linalg.norm(R,axis=2)),np.sqrt((O**2).sum(2).mean())))
res=np.array(res)
for d,a,s,r in res:
    if abs(d*4-round(d*4))<1e-9 and (d*4)%4==0: print('  %+5.2f |  %6.1f px          |  %6.1f px      | %6.1f'%(d,a,s,r))
best_align=res[np.argmin(res[:,1])]; best_shape=res[np.argmin(res[:,2])]; best_rms=res[np.argmin(res[:,3])]
print('  ARGMIN frame-align offset: delta=%+.2f frames (%.0f ms) -> %.1f px'%(best_align[0],best_align[0]*25,best_align[1]))
print('  ARGMIN shape residual    : delta=%+.2f frames (%.0f ms) -> %.1f px'%(best_shape[0],best_shape[0]*25,best_shape[2]))
print('  ARGMIN joint RMS         : delta=%+.2f frames (%.0f ms) -> %.1f px'%(best_rms[0],best_rms[0]*25,best_rms[3]))
# --- per-joint optimum (does every joint agree?) ---
print('\n=== per-joint sub-frame optimum (min median joint error) ===')
names=['pelvis','L_hip','R_hip','L_knee','R_knee','L_ankle','R_ankle','L_foot','R_foot','head','neck','L_sho','R_sho','L_elb','R_elb','L_wri','R_wri']
for k in range(len(S)):
    vals=[]
    for delta in np.arange(-2,20.01,0.25):
        K2,ok0=kp_at(delta); ok=gate&ok0
        e=np.linalg.norm(UV[ok][:,S[k]]-K2[ok][:,H[k]],axis=1)
        vals.append(np.median(e))
    vals=np.array(vals); d=res[:,0][np.argmin(vals)]
    print('   %-8s best delta %+5.2f f (%4.0f ms)  err@best %5.1f px   err@0 %5.1f'%(names[k],d,d*25,vals.min(),vals[np.argmin(np.abs(res[:,0]))]))
