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
N=543
# ---- 3D pelvis velocity (world frame, from raw mocap) ----
pel=Pc[:,0,:]                       # cm in checkerboard world
v3=np.gradient(pel,axis=0)*40.0     # cm/s
sp3=np.linalg.norm(v3[:,:2],axis=1) # horizontal speed cm/s
O=UV[:,S]-kp[:,H]                   # proj - kp
dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
# kp 2D image velocity of the joint centroid
kc=np.median(kp[:,H],axis=1); v2=np.gradient(kc,axis=0)*40.0   # px/s
print('=== 1. speed groups ===')
groups={'stationary (<0.15 m/s)':(sp3<15),'slow (0.15-0.6)':(sp3>=15)&(sp3<60),'walking (>0.6 m/s)':(sp3>=60)}
for nm,m in groups.items():
    m=m&gate&(np.arange(N)>2)&(np.arange(N)<N-3)
    if m.sum()<5: print('  %-22s n=%d (too few)'%(nm,m.sum())); continue
    print('  %-22s n=%3d  |align offset| median %5.1f px   dx %+6.1f dy %+5.1f   kp speed %5.1f px/s'%(
      nm,m.sum(),np.median(np.hypot(dx[m],dy[m])),np.median(dx[m]),np.median(dy[m]),np.median(np.linalg.norm(v2[m],axis=1))))
print()
# ---- 2. direction sign test: does the offset flip with the walking direction? ----
print('=== 2. offset vs horizontal 3D motion direction ===')
vv=v3[:,:2]; good=(sp3>60)&gate&(np.arange(N)>2)&(np.arange(N)<N-3)
ux=vv[:,0]/np.maximum(sp3,1e-9); uy=vv[:,1]/np.maximum(sp3,1e-9)
print('  n walking frames=%d ; fraction moving +x(world)=%.2f'%(good.sum(),(ux[good]>0).mean()))
for sgn,nm in ((+1,'walking +x world'),(-1,'walking -x world')):
    m=good&(np.sign(ux)==sgn)
    if m.sum()<10: print('   %-18s n=%d'%(nm,m.sum())); continue
    print('   %-18s n=%3d  dx median %+6.1f px  dy %+5.1f  |offset| %5.1f px'%(nm,m.sum(),np.median(dx[m]),np.median(dy[m]),np.median(np.hypot(dx[m],dy[m]))))
# same in image space using the kp velocity direction (independent of mocap)
print('  -- using kp image velocity direction instead (2D-only) --')
sg=np.sign(v2[:,0]); good2=(np.abs(v2[:,0])>40)&gate
for sgn in (+1,-1):
    m=good2&(sg==sgn)
    if m.sum()<10: print('   image moving %+d x  n=%d'%(sgn,m.sum())); continue
    print('   kp moving %+d x-image  n=%3d  dx median %+6.1f  dy %+5.1f'%(sgn,m.sum(),np.median(dx[m]),np.median(dy[m])))
# ---- 3. linear fit offset = -dt * kp_image_velocity ; report dt and R^2 ----
print('\n=== 3. offset vs kp image velocity (2D-only), per-frame median offset ===')
m=gate&(np.arange(N)>1)&(np.arange(N)<N-2)
for ax,nm in ((0,'dx'),(1,'dy')):
    y=dx if ax==0 else dy
    vk=v2[m,ax]
    A=np.stack([vk,np.ones(m.sum())],axis=1)
    sol,res,_,_=np.linalg.lstsq(A,y[m],rcond=None)
    pred=A@sol; ss=1-((y[m]-pred)**2).sum()/((y[m]-y[m].mean())**2).sum()
    print('  %s = %+.4f * v_kp  %+.1f px   R^2=%.3f   (implied dt=%+.2f frames)'%(nm,sol[0],sol[1],ss,-sol[0]*40))
# ---- 4. same but only high-|velocity| frames (trend-free) ----
hi=(np.abs(v2[:,0])>150)&gate
print('  high-speed subset n=%d : dx vs vx slope implies dt=%s'%(hi.sum(),
  round(-np.polyfit(v2[hi,0],dx[hi],1)[0]*40,2)))
