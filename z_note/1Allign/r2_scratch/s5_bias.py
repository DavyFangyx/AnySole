import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64)
b=np.load(MI+'bbox.npy');X1,Y1,X2,Y2=b[:,1],b[:,2],b[:,3],b[:,4]
rv,_=cv2.Rodrigues(Rc)
def proj(P):
    uv=np.empty(P.shape[:2]+(2,))
    for i in range(len(P)):
        u,_=cv2.projectPoints(P[i],rv,tc.reshape(3,1),K,D); uv[i]=u[:,0,:]
    return uv
def chain(delta=np.zeros(3),s=100.0): return (Rm@(s*(J.reshape(-1,3)+delta)).T).T.reshape(J.shape)
uv=proj(chain()); subj=np.load('/tmp/r2-scratch/subj.npy')
dx=uv[...,0]-((X1+X2)/2)[:,None]; dy=uv[...,1]-((Y1+Y2)/2)[:,None]
names=['pelvis','L_hip','R_hip','spine1','L_knee','R_knee','spine2','L_ankle','R_ankle','spine3','L_foot','R_foot','neck','L_collar','R_collar','head','L_shoulder','R_shoulder','L_elbow','R_elbow','L_wrist','R_wrist','L_hand','R_hand']
print('=== per-joint median offset vs silhouette-box centre (subject frames, px) ===')
for i,n in enumerate(names):
    print('  %2d %-11s dx %+7.1f  dy %+7.1f  (px; + = right/down)'%(i,n,np.median(dx[subj,i]),np.median(dy[subj,i])))
pel=uv[subj,0]
print('\nhull L margin (proj_x1-box_x1) med %+.1f | R margin (box_x2-proj_x2) med %+.1f | hull-out-right frac %.2f'%(
 np.median(uv[subj,:,0].min(1)-X1[subj]),np.median(X2[subj]-uv[subj,:,0].max(1)),float((uv[subj,:,0].max(1)>X2[subj]).mean())))
# global world-offset fit (3 params) minimising hull-centre vs box-centre
from scipy.optimize import minimize
def cost(d):
    u=proj(chain(d)); return float(np.sum((u[subj,:,0].min(1)+u[subj,:,0].max(1))/2-((X1+X2)/2)[subj])**2
                     +np.sum((u[subj,:,1].min(1)+u[subj,:,1].max(1))/2-((Y1+Y2)/2)[subj])**2)
r=minimize(cost,np.zeros(3),method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1.0,maxiter=4000,maxfev=4000))
dopt=r.x; print('\nbest-fit constant world offset (raw y-up frame, m): %s  |d|=%.3f m  cost %.3e -> %.3e'%(np.round(dopt,4),np.linalg.norm(dopt),r.fun if False else cost(np.zeros(3)),r.fun))
u2=proj(chain(dopt)); dx2=u2[...,0]-((X1+X2)/2)[:,None]
print('after offset: |dx| med %.1f px  IQR %.1f..%.1f  |dy| med %.1f px'%(
 np.median(np.abs(dx2[subj])),np.percentile(dx2[subj],25),np.percentile(dx2[subj],75),np.median(np.abs(u2[...,1]-((Y1+Y2)/2)[:,None])[subj])))
Zc=(Rc@chain().reshape(-1,3).T).T.reshape(J.shape)+tc; Z=Zc[:,:,2]
print('\n=== is the bias an image-space constant or a metric offset? ===')
for tag,m in [('all subject',subj)]:
    x=1.0/Z[m].mean(1)
    print('  corr(dx_hullcentre, 1/Z)=%.3f   corr(dx, yaw=global_orient[1])=%.3f'%(
      np.corrcoef(np.median(dx[m],axis=1),x)[0,1], np.corrcoef(np.median(dx[m],axis=1),np.load(MI+'smpl.npy',allow_pickle=True).item()['global_orient'][:,1][m])[0,1]))
go=np.load(MI+'smpl.npy',allow_pickle=True).item()['global_orient']
print('  Z_cam median %.2f m  (subject frames)'%np.median(Z[subj]))
print('\n=== per-frame IoU on subject frames ===')
iou=np.load('/tmp/r2-scratch/iou.npy')
print('  median %.3f  q25 %.3f  q75 %.3f  frac>0.5 %.2f  frac>0.3 %.2f'%(np.median(iou[subj]),np.percentile(iou[subj],25),np.percentile(iou[subj],75),(iou[subj]>0.5).mean(),(iou[subj]>0.3).mean()))
