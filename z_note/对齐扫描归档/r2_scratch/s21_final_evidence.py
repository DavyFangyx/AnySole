import numpy as np, cv2, json, os, glob
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
RAW=WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz'
print('########## A. chain fidelity: smpl.npy vs raw mocap npz ##########')
raw=dict(np.load(RAW,allow_pickle=True))
def show(k,v):
    a=np.asarray(v); print('  raw[%s] shape=%s dtype=%s'%(k,a.shape,a.dtype))
for k in raw:
    if np.asarray(raw[k]).size>12: show(k,raw[k])
    else: print('  raw[%s] = %s'%(k,str(np.asarray(raw[k]).ravel()[:6])))
S=np.load(MI+'smpl.npy',allow_pickle=True).item()
print('  smpl.npy keys=',sorted(S.keys()))
print('  shapes:',{k:np.asarray(S[k]).shape for k in S})
poses=np.asarray(raw['poses'],float); trans=np.asarray(raw['trans'],float); betas=np.asarray(raw['betas'],float)
st=np.asarray(raw['source_frame_times_s'],float)
fid=np.load(MI+'frame_id.npy'); frames=dict(np.load(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/frames.npz',allow_pickle=True))
mt=np.asarray(frames['mocap_time_s'],float); vt=np.asarray(frames['visual_time_s'],float)
print('  frames.npz keys=',sorted(frames.keys()))
tgt=np.clip(mt,st[0],st[-1])
def interp(col):
    return np.stack([np.interp(tgt,st,col[:,j]) for j in range(col.shape[1])],1)
go=interp(poses[:,:3]); bp=interp(poses[:,3:72])
# adapter rule: global_orient=poses[:,:3]; body_pose=poses[:,3:] -> 69 cols (SMPL 24 joints -> 23*3=69)
bp69=interp(poses[:,3:72]) if poses.shape[1]>=72 else interp(poses[:,3:])
tr=interp(trans)
print('  maxabs(smpl.npy - raw np.interp) : body_pose %.2e  global_orient %.2e  transl %.2e  betas %.2e'%(
  np.abs(np.asarray(S['body_pose'],float)-bp69).max(),np.abs(np.asarray(S['global_orient'],float)-go).max(),
  np.abs(np.asarray(S['transl'],float)-tr).max(),np.abs(np.asarray(S['betas'],float)-betas[:10]).max()))
print('  float32 round-trip: body_pose %.2e global_orient %.2e transl %.2e'%(
  np.abs(np.asarray(S['body_pose'],np.float32).astype(float)-bp69).max(),np.abs(np.asarray(S['global_orient'],np.float32).astype(float)-go).max(),
  np.abs(np.asarray(S['transl'],np.float32).astype(float)-tr).max()))
print('  source: n_frames %d  times [%.3f..%.3f]  dt median %.9f  fps %.6f'%(len(st),st[0],st[-1],np.median(np.diff(st)),1/np.median(np.diff(st))))
print('  grid  : smpl.npy T=%d  keypoints=%s  frames.npz=%d  color=%d'%(len(S['transl']),np.load(MI+'keypoints.npy').shape,len(mt),len(os.listdir(MI+'color'))))
print('  out_of_span frames: %d ; first query t=%.6f s = %.3f source frames (ms off 120Hz grid: %.3f)'%(int(((mt<st[0])|(mt>st[-1])).sum()),mt[0],(mt[0]-st[0])/np.median(np.diff(st)),1000*((mt[0]-st[0])/np.median(np.diff(st))%1)*np.median(np.diff(st))))
print('  clock relation: visual-mocap = %+.6f s constant=%s ; visual[0]=%.6f mocap[0]=%.6f'%(np.median(vt-mt),bool(np.allclose(vt-mt,np.median(vt-mt))),vt[0],mt[0]))
print('  session.json offset_s=%s'%json.load(open(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/session.json')).get('offset_s'))
print('  frame_id 0..%d monotone=%s  visual dt median %.6f'%(fid.max(),bool(np.all(np.diff(fid)>0)),np.median(np.diff(vt))))

print('\n########## B. keypoints.npy == SMPL forward(smpl.npy) ##########')
import smplx, torch
sm=smplx.create(WS+'/assets/third_party/smpl/SMPL_NEUTRAL.pkl',model_type='smpl')
with torch.no_grad():
    r=sm(betas=torch.tensor(np.asarray(S['betas'],np.float32)).unsqueeze(0),
         body_pose=torch.tensor(np.asarray(S['body_pose'],np.float32)).reshape(-1,23,3),
         global_orient=torch.tensor(np.asarray(S['global_orient'],np.float32)).unsqueeze(1),
         transl=torch.tensor(np.asarray(S['transl'],np.float32)))
    J=r.joints[:,:24].numpy()
KP=np.load(MI+'keypoints.npy')
print('  smplx pkg ; fwd joints %s vs keypoints.npy %s ; maxabs diff %.3e m'%(smplx.__version__,J.shape,KP.shape,np.abs(J-KP).max()))
print('  (float32 storage) maxabs diff of .astype(float32): %.3e'%np.abs(J.astype(np.float32)-KP.astype(np.float32)).max())

print('\n########## C. geometry / projection chain ##########')
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K'],float);D=np.array(cam['D'],float);Rc=np.array(cam['R'],float);tc=np.array(cam['t'],float).reshape(3,1)
T=cal['mocap_raw_to_checkerboard_world']; Rm=np.array(T['R'],float);tm0=np.array(T['t'],float)
print('  calib: cam3 fx=%.4f fy=%.4f cx=%.4f cy=%.4f  D=%s  image_size=%s'%(K[0,0],K[1,1],K[0,2],K[1,2],np.round(D.ravel(),5).tolist(),cam.get('image_size_px')))
print('  mocap_raw_to_checkerboard_world: input_scale=%s  t=%s (unit cm)  |t|=%.2f cm'%(T.get('input_scale'),np.round(tm0,4).tolist(),np.linalg.norm(tm0)))
print('  world_to_camera: %s ; pixel_coordinates=%s'%(cal.get('world_to_camera'),cal.get('pixel_coordinates')))
Jd=np.asarray(KP,float)
def proj(P_scale,perm=None,order='xyz'):
    P=Jd.copy()
    if perm=='c3d': P=P[...,[1,2,0]]*np.array([-1,1,1])   # raw_c3d_to_smpl style
    Pc=(Rm@(P_scale*P.reshape(-1,3)).T).T.reshape(P.shape)+tm0
    u,_=cv2.projectPoints(np.ascontiguousarray(Pc.reshape(-1,1,3)),cv2.Rodrigues(Rc)[0],tc,K,D)
    return u.reshape(P.shape[0],P.shape[1],2)
def stats(UV,tag,bbox=None):
    inimg=((UV[...,0]>=0)&(UV[...,0]<1624)&(UV[...,1]>=0)&(UV[...,1]<1240)).all()
    h=UV[:,:,1].max(1)-UV[:,:,1].min(1)
    s='  %-28s in-image %s  cloudH median %5.1f px'%(tag,'100%' if inimg else '%.0f%%'%(100*((UV[...,0]>=0)&(UV[...,0]<1624)&(UV[...,1]>=0)&(UV[...,1]<1240)).mean()),np.median(h))
    if bbox is not None:
        C=np.stack([UV[:,:,0].min(1),UV[:,:,1].min(1),UV[:,:,0].max(1),UV[:,:,1].max(1)],1)
        x1=np.maximum(C[:,0],bbox[:,1]);y1=np.maximum(C[:,1],bbox[:,2]);x2=np.minimum(C[:,2],bbox[:,3]);y2=np.minimum(C[:,3],bbox[:,4])
        inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None); ua=(C[:,2]-C[:,0])*(C[:,3]-C[:,1])+(bbox[:,3]-bbox[:,2])*(bbox[:,4]-bbox[:,1])-inter
        s+='  IoU vs YOLOX %.3f'%np.median(inter/ua)
    print(s)
bb=np.load(MI+'bbox.npy')
UV_true=proj(100.0); stats(UV_true,'CORRECT  s=100, no perm',bb)
stats(proj(10.0),'control s=10 (cm as m)',bb)
stats(proj(1.0),'control s=1 (m as m)',bb)
stats(proj(1000.0),'control s=1000',bb)
# z-up-first control
Pz=np.stack([Jd[...,0],-Jd[...,2],Jd[...,1]],-1)
Pc=(Rm@(100.0*Pz.reshape(-1,3)).T).T.reshape(Pz.shape)+tm0
u,_=cv2.projectPoints(np.ascontiguousarray(Pc.reshape(-1,1,3)),cv2.Rodrigues(Rc)[0],tc,K,D); stats(u.reshape(Jd.shape),'control z-up-first',bb)
# camera height above the mocap floor plane (scale-invariant check)
Rw2c=cv2.Rodrigues(Rc)[0]; Cw=(-Rw2c.T@tc).ravel()   # camera centre in checkerboard world (cm)
Jz=(Rm@(100.0*Jd.reshape(-1,3)).T).T.reshape(Jd.shape)+tm0
floor=np.percentile(Jz.reshape(-1,3)[:,2],5)   # vertical axis is z in the checkerboard world
print('  camera centre in cb-world: %s ; camera height above mocap floor plane (p5 z): %.1f cm ; foot z p1=%.1f p5=%.1f'%(
  np.round(Cw,1).tolist(),Cw[2]-floor,np.percentile(Jz[...,2],1),floor))
# axis/world-frame cross-check vs raw BVH
bvh=glob.glob(WS+'/raw/bvh/20260804/mocap_ori_bvh/S5091/*.bvh')
print('  raw BVH present: %s'%[os.path.basename(b) for b in bvh])
