import numpy as np, os, sys, torch, smplx
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
RAW=WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz'

# --- raw contract (adapter load_smpl_source checks) ---
d=dict(np.load(RAW,allow_pickle=True))
poses=d['poses']; times=d['source_frame_times_s']; trans_raw=d['trans']; betas_raw=d['betas']
print('[raw] poses',poses.shape,'times',times.shape,'trans',trans_raw.shape,'betas',betas_raw.shape)
print('[raw] poses[:,:3]==root_orient:',np.array_equal(poses[:,:3],d['root_orient']))
print('[raw] poses[:,3:66]==pose_body:',np.array_equal(poses[:,3:66],d['pose_body']))
print('[raw] poses[:,66:72] max abs:',float(np.abs(poses[:,66:]).max()))
print('[raw] fps=%.6f  t0=%.6f t-1=%.6f'%(1/np.median(np.diff(times)),times[0],times[-1]))
# keep xyz-timestamp key if present
for k in d:
    if 'timestamp' in k.lower() or 'time' in k.lower():
    

# --- reproduce smpl.npy via adapter interp ---
fr=np.load(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/frames.npz',allow_pickle=True)
q=fr['mocap_time_s']
def interp(vals):
    c=np.clip(q,times[0],times[-1]); f=vals.reshape(len(vals),-1)
    out=np.empty((len(q),f.shape[1]))
    for c_ in range(f.shape[1]): out[:,c_]=np.interp(c,times,f[:,c_])
    return out
go=interp(poses[:,:3]); bp=interp(poses[:,3:]); tr=interp(trans_raw)
s=np.load(MI+'smpl.npy',allow_pickle=True).item()
for name,recomputed in [('global_orient',go),('body_pose',bp),('transl',tr)]:
    on=np.asarray(s[name],dtype=np.float64)
    print('[repro] %-14s shape %s  on-disk=recomputed maxabs %.3e  f32-cast-exact %s'%(
        name,on.shape,float(np.abs(on-recomputed).max()),
        bool(np.array_equal(on,np.asarray(recomputed,dtype=np.float32)))))
print('[repro] betas exact:',np.array_equal(np.asarray(s['betas']),np.asarray(betas_raw[:10],dtype=np.float32)))
print('[repro] t0=%.6f p0 offset_s=%.6f (%.3f ms)'%(q[0],q[0]-times[0],(q[0]-times[0])*1e3))

# --- SMPL forward ---
pkl=WS+'/assets/third_party/smpl/SMPL_NEUTRAL.pkl'
print('[smpl] pkl exists',os.path.exists(pkl))
model=smplx.create(pkl).to('cpu')
betas=torch.tensor(np.asarray(s['betas'],dtype=np.float32)).unsqueeze(0)
body=torch.tensor(np.asarray(s['body_pose'],dtype=np.float32)).reshape(-1,23,3)
go_t=torch.tensor(np.asarray(s['global_orient'],dtype=np.float32)).unsqueeze(1)
tr_t=torch.tensor(np.asarray(s['transl'],dtype=np.float32))
with torch.no_grad(): out=model(betas=betas,body_pose=body,global_orient=go_t,transl=tr_t)
print('[smpl] joints',tuple(out.joints.shape),'vertices',tuple(out.vertices.shape))
J=out.joints[:,:24].numpy()
K=np.load(MI+'keypoints.npy')
print('[kps] on-disk',K.shape,'J[:, :24]',J.shape)
print('[kps] maxabs diff vs recomputed joints: %.3e (meters)  -> mm %.4f'%(np.abs(K-J).max(),np.abs(K-J).max()*1000))
print('[kps] allclose atol 1e-6:',bool(np.allclose(K,J,atol=1e-6)))
# joint ranges -> unit + up axis
for nm,idx in [('pelvis',0),('L_foot',10),('R_foot',11),('head',15),('L_wrist',20)]:
    print('  joint %-8s idx%2d mean=%s'%(nm,idx,np.round(K[:,idx].mean(0),4)))
allj=K.reshape(-1,3)
print('[unit] joints global min/max per axis:',np.round(allj.min(0),3),np.round(allj.max(0),3))
print('[unit] extent per axis (m):',np.round(allj.max(0)-allj.min(0),3))
