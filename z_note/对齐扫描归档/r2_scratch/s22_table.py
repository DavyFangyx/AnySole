import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K'],float);D=np.array(cam['D'],float);Rc=np.array(cam['R'],float);tc=np.array(cam['t'],float).reshape(3,1)
T=cal['mocap_raw_to_checkerboard_world']; Rm=np.array(T['R'],float);tm0=np.array(T['t'],float)
J=np.load(MI+'keypoints.npy').astype(float); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
rv=cv2.Rodrigues(Rc)[0]
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm0
u,_=cv2.projectPoints(Pc.reshape(-1,1,3),rv,tc,K,D); UV=u.reshape(len(J),-1,2)
N=len(J)
def kp_at(d):
    f=np.arange(N)+d; lo=np.clip(np.floor(f).astype(int),0,N-1); hi=np.clip(lo+1,0,N-1); w=(f-lo)[:,None,None]
    return kp[lo]*(1-w)+kp[hi]*w,(f>=0)&(f<N-1)
# joint mapping table
M=[('pelvis',0,19),('L_hip',1,11),('R_hip',2,12),('L_knee',4,13),('R_knee',5,14),('L_ankle',7,15),('R_ankle',8,16),
   ('L_foot',10,24),('R_foot',11,25),('neck',12,18),('head',15,17),('L_shoulder',16,5),('R_shoulder',17,6),
   ('L_elbow',18,7),('R_elbow',19,8),('L_wrist',20,9),('R_wrist',21,10)]
# mirror controls: swap L/R on the HALPE side
MM=[('L_ankle',7,'R_ank',16),('R_ankle',8,'L_ank',15),('L_foot',10,'R_heel',25),('R_foot',11,'L_heel',24),
    ('L_shoulder',16,'R_sho',6),('R_shoulder',17,'L_sho',5),('L_knee',4,'R_knee',14),('R_knee',5,'L_knee',13)]
print('=== S5091 per-joint reprojection error, gate frames n=%d ==='%gate.sum())
print('%-11s %7s %7s %7s %7s %7s'%('joint','@lag0','@lag+9.5','p90@9.5','max@9.5','mirror@9.5'))
K2,ok0=kp_at(9.5); ok=gate&ok0
rows=[]
for nm,pi,hi in M:
    e0=np.linalg.norm(UV[gate][:,pi]-kp[gate][:,hi],axis=1)
    e1=np.linalg.norm(UV[ok][:,pi]-K2[ok][:,hi],axis=1)
    rows.append((nm,np.median(e0),np.median(e1),np.percentile(e1,90),e1.max()))
e0all=np.concatenate([[np.linalg.norm(UV[gate][:,pi]-kp[gate][:,hi],axis=1)] for _,pi,hi in M])
e1all=np.concatenate([[np.linalg.norm(UV[ok][:,pi]-K2[ok][:,hi],axis=1)] for _,pi,hi in M])
for nm,m0,m1,p90,mx in rows: print('%-11s %7.1f %7.1f %7.1f %7.1f'%(nm,m0,m1,p90,mx))
print('%-11s %7.1f %7.1f %7.1f %7.1f   <- all 17 pairs x %d frames'%('ALL',np.median(e0all),np.median(e1all),np.percentile(e1all,90),e1all.max(),ok.sum()))
def frameoff(UVx,kpx,okx):
    dx=np.median(np.stack([np.median((UVx[okx][:,pi]-kpx[okx][:,hi])[:,0]) for _,pi,hi in M]))
    dy=np.median(np.stack([np.median((UVx[okx][:,pi]-kpx[okx][:,hi])[:,1]) for _,pi,hi in M]))
    return np.hypot(dx,dy)
print('\n  mean@0 %.1f  mean@9.5 %.1f  |  per-frame align offset @0 %.1f px -> @9.5 %.1f px'%(
  e0all.mean(),e1all.mean(),frameoff(UV,kp,gate),frameoff(UV,K2,ok)))
print('\n=== L/R mirror controls @lag+9.5 (correct pairing vs swapped L<->R landmark) ===')
for nm,pi,hnm,hi in MM:
    e=np.median(np.linalg.norm(UV[ok][:,pi]-K2[ok][:,hi],axis=1))
    print('   %-11s vs %-7s = %6.1f px   (vs correct-side %.1f px)'%(nm,hnm,e,[r[2] for r in rows if r[0]==nm][0]))
print('\n=== stationary vs moving (from smpl.npy 3D speed) ===')
pel=Pc[:,0,:]; sp3=np.linalg.norm(np.gradient(pel,axis=0)[:,:2],axis=1)*40
K2b,okb=kp_at(0)
for nm,m in [('stationary <0.15 m/s',(sp3<15)&gate&(np.arange(N)>2)&(np.arange(N)<N-3)),
             ('slow 0.15-0.6 m/s',(sp3>=15)&(sp3<60)&gate&(np.arange(N)>2)&(np.arange(N)<N-3)),
             ('walking >0.6 m/s',(sp3>=60)&gate&(np.arange(N)>2)&(np.arange(N)<N-3))]:
    O=UV[m][:, [x[1] for x in M]]-K2b[m][:, [x[2] for x in M]]
    dx=np.median(O[...,0],1); dy=np.median(O[...,1],1)
    print('   %-20s n=%3d  |align offset| median %5.1f px (dx %+6.1f dy %+5.1f)'%(nm,m.sum(),np.median(np.hypot(dx,dy)),np.median(dx),np.median(dy)))
