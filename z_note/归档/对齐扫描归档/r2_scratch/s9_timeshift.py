import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
rv,_=cv2.Rodrigues(Rc)
def uv_of(P):
    Pc=(Rm@(100.0*P.reshape(-1,3)).T).T.reshape(P.shape)+tm;o=np.empty(Pc.shape[:2]+(2,))
    for i in range(len(Pc)):
        u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D);o[i]=u[:,0,:]
    return o
UV=uv_of(J)
# per-frame 2D offset (componentwise median = robust 2D alignment error), gate frames
O=UV[:,S]-kp[:,H]                     # (543,17,2)
dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
g=gate
print('=== per-frame 2D alignment offset (gate n=%d) ==='%g.sum())
print('  dx: median %+.1f px  p25 %+.1f  p75 %+.1f  |dx| med %.1f  |dx|<20px %.0f%%'%(
 np.median(dx[g]),np.percentile(dx[g],25),np.percentile(dx[g],75),np.median(np.abs(dx[g])),100*(np.abs(dx[g])<20).mean()))
print('  dy: median %+.1f px  p25 %+.1f  p75 %+.1f  |dy| med %.1f  |dy|<20px %.0f%%'%(
 np.median(dy[g]),np.percentile(dy[g],25),np.percentile(dy[g],75),np.median(np.abs(dy[g])),100*(np.abs(dy[g])<20).mean()))
print('  per-frame |offset| (px): median %.1f  p90 %.1f'%(np.median(np.hypot(dx[g],dy[g])),np.percentile(np.hypot(dx[g],dy[g]),90)))
# residual AFTER removing the per-frame 2D offset = pose/shape scatter
res=O.copy(); res[...,0]-=dx[:,None]; res[...,1]-=dy[:,None]
r=np.linalg.norm(res,axis=2)
print('  shape residual after per-frame 2D offset: median %.1f px  p90 %.1f  (gate frames)'%(np.median(r[g]),np.percentile(r[g],90)))
# regress the per-frame offset on the kp image velocity -> time offset in frames
kp_c=np.median(kp[:,H],axis=1)          # (543,2) centroid of matched joints
v=np.gradient(kp_c,axis=0)*1.0          # px per frame
m=g[1:-1]&gate[:-2]&gate[2:]
A=np.stack([v[m,0],v[m,1],np.ones(m.sum())],axis=1)
sol,_,_,_=np.linalg.lstsq(A,np.stack([dx[1:-1][m],dy[1:-1][m]],axis=1),rcond=None)
print('\n=== time-offset regression  offset_px = a*v_px_per_frame + b ===')
print('  a = (%.3f, %.3f)  -> implied time offset %+.2f / %+.2f frames (40Hz), mean %+.2f frames = %+.1f ms'%(
  sol[0,0],sol[0,1],sol[0,0],sol[0,1],0.5*(sol[0,0]+sol[0,1]),0.5*(sol[0,0]+sol[0,1])*25))
print('  b = (%+.1f, %+.1f) px residual constant offset'%(sol[2,0],sol[2,1]))
# direct lag sweep on the per-frame alignment offset (median |offset|)
print('\n=== lag sweep on per-frame median offset magnitude (gate frames) ===')
for L in range(-4,5):
    j=np.arange(543)+L; ok=g&(j>=0)&(j<543)
    o=UV[ok][:,S]-kp[j[ok]][:,H]
    d=np.hypot(np.median(o[...,0],axis=1),np.median(o[...,1],axis=1))
    print('  lag %+d n=%3d  median|offset| %6.1f px'%(L,ok.sum(),np.median(d)))
# and with per-frame 2D affine (scale) allowed
print('\n=== frame-0 (standing) per-joint alignment in detail ===')
o=UV[0][S]-kp[0][H]
print('  offsets px (proj - kp):',np.round(o[:8],1).tolist(),' median dx %+.1f dy %+.1f'%(np.median(o[:,0]),np.median(o[:,1])))
