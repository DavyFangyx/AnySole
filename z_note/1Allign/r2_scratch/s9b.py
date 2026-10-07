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
O=UV[:,S]-kp[:,H]                                  # (543,17,2) proj - kp
dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
g=gate
print('=== A. per-frame 2D alignment offset (gate n=%d) ==='%g.sum())
print('  dx  median %+.1f  IQR [%+.1f,%+.1f]  |dx|med %.1f  p90|dx| %.1f'%(np.median(dx[g]),np.percentile(dx[g],25),np.percentile(dx[g],75),np.median(np.abs(dx[g])),np.percentile(np.abs(dx[g]),90)))
print('  dy  median %+.1f  IQR [%+.1f,%+.1f]  |dy|med %.1f  p90|dy| %.1f'%(np.median(dy[g]),np.percentile(dy[g],25),np.percentile(dy[g],75),np.median(np.abs(dy[g])),np.percentile(np.abs(dy[g]),90)))
res=O.copy(); res[...,0]-=dx[:,None]; res[...,1]-=dy[:,None]
r=np.linalg.norm(res,axis=2)
print('  shape residual (after per-frame 2D offset): median %.1f  p90 %.1f px  [gate]'%(np.median(r[g]),np.percentile(r[g],90)))
# --- time-offset regression: offset_px ~ 2x2 matrix * velocity_px_per_frame + const
kc=np.median(kp[:,H],axis=1)                       # kp joint centroid (543,2)
vel=np.gradient(kc,axis=0)                         # px/frame
m=g.copy(); m[0]=False; m[-1]=False
idx=np.where(m)[0]
X=np.stack([vel[idx,0],vel[idx,1],np.ones(len(idx))],axis=1)
Y=np.stack([dx[idx],dy[idx]],axis=1)
sol,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
c=sol[0:2,:].T; b=sol[2]
ev=np.linalg.eigvals(c)
print('\n=== B. time-offset regression  offset_px = C @ v_px_frame + b ===')
print('  C = [[%+.3f,%+.3f],[%+.3f,%+.3f]]  eigenvalues %s'%(c[0,0],c[0,1],c[1,0],c[1,1],np.round(ev,3).tolist()))
dt=-0.5*np.trace(c).real
print('  -> offset = C @ v  =>  implied time offset dt = -trace(C)/2 = %+.2f frames = %+.0f ms @40Hz'%(dt,dt*25))
print('     (eig(C) = %s ; pure-scalar model would need C = -dt*I)'%np.round(ev,3).tolist())
print('  b = (%+.1f, %+.1f) px'%(b[0],b[1]))
# --- lag sweep on per-frame offset magnitude
print('\n=== C. lag sweep: project@i vs kp@i+L (gate) ===')
for L in range(-4,5):
    j=idx+L; ok=(j>=0)&(j<543)
    o=UV[idx[ok]][:,S]-kp[j[ok]][:,H]
    d=np.hypot(np.median(o[...,0],axis=1),np.median(o[...,1],axis=1))
    print('  L=%+d n=%3d  median|offset| %6.1f px   median dx %+6.1f dy %+6.1f'%(L,ok.sum(),np.median(d),np.median(np.median(o[...,0],axis=1)),np.median(np.median(o[...,1],axis=1))))
# --- how much motion is there between adjacent frames? (can a 1-3 frame lag matter?)
sp=np.linalg.norm(vel[idx],axis=1)
print('\n=== D. kp centroid image speed (px/frame @40Hz): median %.1f  p90 %.1f  max %.1f'%(np.median(sp),np.percentile(sp,90),sp.max()))
print('  => 1 frame = %.1f px median motion; the lag branch can be discriminated'%np.median(sp))
