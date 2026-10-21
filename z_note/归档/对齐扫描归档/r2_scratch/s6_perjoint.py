import numpy as np, cv2, json, glob, os
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
KP=WS+'/shared/frontends/rtmpose_halpe26/v1/20260804/S5/S5091/keypoints/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64)
Xcb=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)
rv,_=cv2.Rodrigues(Rc); UV=np.empty(J.shape[:2]+(2,))
for i in range(len(J)):
    u,_=cv2.projectPoints(Xcb[i],rv,tc.reshape(3,1),K,D); UV[i]=u[:,0,:]
# RTMPose sidecars
kp=np.stack([np.load(KP+'%06d.npy'%i,allow_pickle=True).item()['keypoints'] for i in range(543)])
sc=np.stack([np.load(KP+'%06d.npy'%i,allow_pickle=True).item()['keypoint_scores'] for i in range(543)])
fid=[np.load(KP+'%06d.npy'%i,allow_pickle=True).item()['frame_id'] for i in range(543)]
print('[join] sidecar frame_id == index for all 543:', all(int(f)==i for i,f in enumerate(fid)))
print('[rgb names] first/last:',os.path.basename(sorted(glob.glob(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/rgb/*'))[0]),os.path.basename(sorted(glob.glob(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/rgb/*'))[-1]))
HALPE=['nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder','L_elbow','R_elbow','L_wrist','R_wrist','L_hip','R_hip','L_knee','R_knee','L_ankle','R_ankle','head','neck','hip','R_big_toe','R_small_toe','R_heel','L_big_toe','L_small_toe','L_heel']
MAP=[(0,19,'pelvis<->hip'),(1,11,'L_hip'),(2,12,'R_hip'),(4,13,'L_knee'),(5,14,'R_knee'),(7,15,'L_ankle'),(8,16,'R_ankle'),
     (10,25,'L_foot<->L_heel'),(11,22,'R_foot<->R_heel'),(15,17,'head'),(12,18,'neck'),(16,5,'L_shoulder'),(17,6,'R_shoulder'),
     (18,7,'L_elbow'),(19,8,'R_elbow'),(20,9,'L_wrist'),(21,10,'R_wrist'),(10,23,'L_foot<->L_bigtoe'),(11,20,'R_foot<->R_bigtoe')]
err=np.stack([np.linalg.norm(UV[:,s]-kp[:,h],axis=1) for s,h,_ in MAP],axis=1)  # (543,npair)
fmed=np.median(err,axis=1)
print('\n=== per-frame median 2D reprojection error over %d mapped joints (px) ==='%len(MAP))
print('  percentiles 5/25/50/75/95: %.1f / %.1f / %.1f / %.1f / %.1f'%tuple(np.percentile(fmed,[5,25,50,75,95])))
for t in (20,30,50,100):
    print('  frames with per-frame median < %3d px: %3d/543 (%.0f%%)'%(t,(fmed<t).sum(),100*(fmed<t).mean()))
sel=fmed<50
print('  selected subset (frame-median<50px) n=%d  contiguous runs: %d'%(sel.sum(),1+int(np.sum(np.diff(sel.astype(int))==1))))
print('\n=== per-joint error table (px) — subset n=%d ==='%sel.sum())
print('  %-16s %7s %7s %7s %7s %7s'%('smpl24 <-> halpe26','mean','median','p90','max','kp_score'))
for pi,(s,h,nm) in enumerate(MAP):
    e=err[sel,pi]
    print('  %-16s %7.1f %7.1f %7.1f %7.1f %7.3f'%(nm,e.mean(),np.median(e),np.percentile(e,90),e.max(),np.median(sc[sel,h])))
print('\n  overall (all pairs, subset): mean %.1f  median %.1f  p90 %.1f'%(err[sel].mean(),np.median(err[sel]),np.percentile(err[sel],90)))
print('  overall (all pairs, ALL 543 frames): median %.1f  p90 %.1f'%(np.median(err),np.percentile(err,90)))
# vertical/scale sanity: kp body height vs projected body height
h_kp=kp[:,[17,15]].max(1)[:,1]-kp[:,[15,16,25,22]].min(1)[:,1]
h_pj=UV[:,[15]].max(1)[:,1]-UV[:,[7,8,10,11]].min(1)[:,1]
print('\n  body height px (head->lowest ankle/heel): kp median %.1f  proj median %.1f  ratio %.3f (subset)'%(
  np.median(h_kp[sel]),np.median(h_pj[sel]),np.median(h_kp[sel]/h_pj[sel])))
# lag sweep with per-joint error
print('\n=== lag sweep (mean per-joint error on subset, px) ===')
for lag in range(-3,4):
    idx=np.arange(543); j=idx+lag; ok=sel&(j>=0)&(j<543)
    e=np.linalg.norm(UV[ok][:,:,:][:,[s for s,_,_ in MAP]]-kp[j[ok]][:,[h for _,h,_ in MAP]],axis=1)
    print('  lag %+d  n=%3d  mean %.1f  median %.1f'%(lag,ok.sum(),e.mean(),np.median(e)))
np.save('/tmp/r2-scratch/err.npy',err); np.save('/tmp/r2-scratch/sel.npy',sel); np.save('/tmp/r2-scratch/UV.npy',UV); np.save('/tmp/r2-scratch/kp.npy',kp)
