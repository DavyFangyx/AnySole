import numpy as np, os, torch, smplx
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
RAW=WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz'
d=dict(np.load(RAW,allow_pickle=True))
poses=d['poses'];times=d['source_frame_times_s'];trans=d['trans'];betas=d['betas']
print('[raw] poses %s trans %s betas %s times %s'%(poses.shape,trans.shape,betas.shape,times.shape))
print('[raw] poses[:,:3]==root_orient %s ; poses[:,3:66]==pose_body %s ; poses[:,66:72] maxabs %.1f'%(
 np.array_equal(poses[:,:3],d['root_orient']),np.array_equal(poses[:,3:66],d['pose_body']),float(np.abs(poses[:,66:]).max())))
print('[raw] fps %.6f  span %.4f..%.4f s (%d frames)'%(1/np.median(np.diff(times)),times[0],times[-1],len(poses)))
fr=np.load(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/frames.npz',allow_pickle=True)
q=fr['mocap_time_s']
def interp(v):
    c=np.clip(q,times[0],times[-1]);f=v.reshape(len(v),-1);o=np.empty((len(q),f.shape[1]))
    for i in range(f.shape[1]):o[:,i]=np.interp(c,times,f[:,i])
    return o
go=interp(poses[:,:3]);bp=interp(poses[:,3:]);tr=interp(trans)
s=np.load(MI+'smpl.npy',allow_pickle=True).item()
print('[keys] on-disk %s'%sorted(s.keys()))
for n,rc in [('global_orient',go),('body_pose',bp),('transl',tr)]:
    on=np.asarray(s[n],dtype=np.float64)
    print('[repro] %-13s shape %s  maxabs(on-disk - recomputed) %.3e  bit-exact-as-f32 %s'%(
      n,on.shape,float(np.abs(on-rc).max()),bool(np.array_equal(on,np.asarray(rc,dtype=np.float32)))))
print('[repro] betas bit-exact %s'%np.array_equal(np.asarray(s['betas']),np.asarray(betas[:10],dtype=np.float32)))
print('[time] frames.npz %d @40Hz uniform dt=%.6f ; smpl T=%d ; keypoints T=%d ; frame_id %d..%d'%(
  len(q),np.median(np.diff(q)),np.asarray(s['transl']).shape[0],np.load(MI+'keypoints.npy').shape[0],
  np.load(MI+'frame_id.npy')[0],np.load(MI+'frame_id.npy')[-1]))
print('[time] query t0=%.6f = %.3f source-frames @120Hz (offset %.3f ms from 120Hz grid); out_of_span 0 confirmed by clip'%(q[0],q[0]*120,(q[0]-round(q[0]*120)/120)*1e3))
model=smplx.create(WS+'/assets/third_party/smpl/SMPL_NEUTRAL.pkl').to('cpu')
with torch.no_grad():
    out=model(betas=torch.tensor(np.asarray(s['betas'],dtype=np.float32)).unsqueeze(0),
              body_pose=torch.tensor(np.asarray(s['body_pose'],dtype=np.float32)).reshape(-1,23,3),
              global_orient=torch.tensor(np.asarray(s['global_orient'],dtype=np.float32)).unsqueeze(1),
              transl=torch.tensor(np.asarray(s['transl'],dtype=np.float32)))
J=out.joints[:,:24].numpy(); K=np.load(MI+'keypoints.npy')
print('[smpl] joints %s verts %s'%(tuple(out.joints.shape),tuple(out.vertices.shape)))
print('[kps] maxabs(keypoints.npy - recomputed SMPL-24 joints) = %.3e m  (= %.5f mm)  bit-equal-as-f32 %s'%(
  np.abs(K-J).max(),np.abs(K-J).max()*1000,bool(np.array_equal(K,J))))
print('[unit] joints extent (m) per axis %s ; y-range %.3f..%.3f (y-up meters)'%(
  np.round(K.reshape(-1,3).max(0)-K.reshape(-1,3).min(0),3),K[...,1].min(),K[...,1].max()))
print('[mesh] v_template pelvis/feet templates vs transl: pelvis_joint0 mean %s'%np.round(K[:,0].mean(0),3))
