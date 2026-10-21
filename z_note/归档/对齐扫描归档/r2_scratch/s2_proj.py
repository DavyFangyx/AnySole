import numpy as np, cv2, json, torch, smplx
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json'))
cam=cal['cameras']['cam3']; K=np.array(cam['K']); D=np.array(cam['D']); Rc=np.array(cam['R']); tc=np.array(cam['t'])
mb=cal['mocap_raw_to_checkerboard_world']; Rm=np.array(mb['R']); tm=np.array(mb['t']); sc=mb['input_scale']
d=np.load(WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz',allow_pickle=True)
Rconv=np.asarray(d['raw_c3d_to_smpl'])[:3,:3]   # X_smpl = Rconv @ X_raw
J=np.load(MI+'keypoints.npy').astype(np.float64)  # (543,24,3) meters, SMPL frame
b=np.load(MI+'bbox.npy'); B=b[:,1:5]; S=b[:,5]
print('K fx=%.4f fy=%.4f cx=%.4f cy=%.4f  |t|=%.3f  scale=%.4f'%(K[0,0],K[1,1],K[0,2],K[1,2],np.linalg.norm(tc),sc))
def project(P):  # P (T,N,3) -> (T,N,2) pixel, distortion applied
    rv,_=cv2.Rodrigues(Rc); out=np.empty(P.shape[:2]+(2,))
    for i in range(P.shape[0]):
        uv,_=cv2.projectPoints(P[i].astype(np.float64),rv,tc.reshape(3,1),K,D)
        out[i]=uv[:,0,:]
    return out
cands={
 'A raw=SMPL_m, s=0.1'      : Rm@(sc*J.reshape(-1,3)).T,
 'B raw=SMPL_m, s=100'      : Rm@(100.0*J.reshape(-1,3)).T,
 'C raw=c3d_mm, s=0.1'      : Rm@(sc*(1000.0*(Rconv.T@J.reshape(-1,3).T))).T.reshape(-1,3).T*0+ (Rm@(sc*(1000.0*(Rconv.T@J.reshape(-1,3).T)))).T.reshape(-1,3).T if False else (Rm@(sc*1000.0*(Rconv.T@J.reshape(-1,3).T))).T.reshape(-1,3).T,
 'D raw=c3d, s=0.1 (no mm)': (Rm@(sc*(Rconv.T@J.reshape(-1,3).T))).T.reshape(-1,3).T,
}
# normalize C: shape (3, T*24) -> (T,24,3)
cands={k:(np.asarray(v).T.reshape(J.shape) if np.asarray(v).shape[0]==3 else np.asarray(v)) for k,v in cands.items()}
print('\ncand | Zcam med(m) | in-image%% | projH med(px) | boxH med(px) | IoU med | ctr-err/H med | joints-in-bbox%%')
for name,Xcb in cands.items():
    Xc=(Rc@Xcb.reshape(-1,3).T).T+tc
    Z=Xc[:,2].reshape(J.shape[:2])
    uv=project(Xcb)
    x1=uv[...,0].min(1); x2=uv[...,0].max(1); y1=uv[...,1].min(1); y2=uv[...,1].max(1)
    inimg=((x1>0)&(x2<1624)&(y1>0)&(y2<1240)).mean()
    iou=[]; ctr=[]
    for i in range(len(uv)):
        ix1,iy1,ix2,iy2=max(x1[i],B[i,0]),max(y1[i],B[i,1]),min(x2[i],B[i,2]),min(y2[i],B[i,3])
        aw=max(0,ix2-ix1); ah=max(0,iy2-iy1); inter=aw*ah
        ua=(x2[i]-x1[i])*(y2[i]-y1[i])+(B[i,2]-B[i,0])*(B[i,3]-B[i,1])-inter
        iou.append(inter/ua if ua>0 else 0)
        ctr.append(np.hypot((x1[i]+x2[i])/2-(B[i,0]+B[i,2])/2,(y1[i]+y2[i])/2-(B[i,1]+B[i,3])/2)/max(1e-6,(y2[i]-y1[i])))
    iou=np.array(iou); ctr=np.array(ctr)
    inside=((uv[...,0]>B[:,None,0])&(uv[...,0]<B[:,None,2])&(uv[...,1]>B[:,None,1])&(uv[...,1]<B[:,None,3])).mean()
    print('%-26s %8.3f %8.1f %10.1f %10.1f %8.3f %10.3f %8.1f'%(
        name,np.median(Z),100*inimg,np.median(y2-y1),np.median(B[:,3]-B[:,2]),np.median(iou),np.median(ctr),100*inside))
print('\nprojH(t) trace for best cand:')
for name in ['B raw=SMPL_m, s=100']:
    uv=project(cands[name]); print(' frame: projH  boxH  projCx boxCx  IoU')
    for i in range(0,543,60):
        print('  %3d  %6.1f %6.1f %7.1f %7.1f  %.2f'%(i,uv[i,:,1].max()-uv[i,:,1].min(),B[i,3]-B[i,2],(uv[i,:,0].min()+uv[i,:,0].max())/2,(B[i,0]+B[i,2])/2,0))
    np.save('/tmp/r2-scratch/uv_B.npy',uv)
