import numpy as np, cv2, json, sys
sys.path.insert(0,'/data/fangyuxuan/projects/gait')
from AnysoleWorkspace.tool._bvh_aligner_pose import parse_bvh_aligner
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
mb=cal['mocap_raw_to_checkerboard_world'];Rm=np.array(mb['R']);tm=np.array(mb['t']);sc=mb['input_scale']
Kj=np.load(MI+'keypoints.npy').astype(np.float64)
b=np.load(MI+'bbox.npy');X1,Y1,X2,Y2=b[:,1],b[:,2],b[:,3],b[:,4];Bw=X2-X1;Bh=Y2-Y1
to_zup=lambda X: np.stack([X[...,0],-X[...,2],X[...,1]],axis=-1)
rv,_=cv2.Rodrigues(Rc)
def proj(Xcb):
    uv=np.empty(Xcb.shape[:2]+(2,))
    for i in range(len(Xcb)):
        u,_=cv2.projectPoints(Xcb[i],rv,tc.reshape(3,1),K,D); uv[i]=u[:,0,:]
    return uv
# camera height above the mocap floor plane (floor = y=0 in the y-up mocap frame)
Cw=-Rc.T@tc   # camera centre in the world(checkerboard) frame, cm
def cam_height(fr_conv_inv, s):
    # world = Rm @ (s*X) + tm ; floor plane normal in the raw frame n0, point p0 -> world
    n0=fr_conv_inv['n']; p0=fr_conv_inv['p']
    nw=Rm@n0; pw=Rm@(s*p0)+tm
    return float(nw@(Cw-pw)/np.linalg.norm(nw))
cands={
 'J y-up m, s=100 (mm->cm)'   : (Rm@(100.0*Kj.reshape(-1,3)).T).T.reshape(Kj.shape),
 'J z-up m, s=100'            : (Rm@(100.0*to_zup(Kj).reshape(-1,3)).T).T.reshape(Kj.shape),
 'J y-up m, s=0.1 (as-written)':(Rm@(0.1*Kj.reshape(-1,3)).T).T.reshape(Kj.shape),
 'J y-up m, s=1'              : (Rm@(1.0*Kj.reshape(-1,3)).T).T.reshape(Kj.shape),
 'J y-up m, s=1000'           : (Rm@(1000.0*Kj.reshape(-1,3)).T).T.reshape(Kj.shape),
}
# floor plane in the raw y-up frame: y=0, normal (0,1,0), point origin
floor={'n':np.array([0.,1.,0.]),'p':np.array([0.,0.,0.])}
print('cam3 centre in checkerboard world (cm):',np.round(Cw,2),' |Cw|=%.1f cm'%np.linalg.norm(Cw))
print('\n%-30s %9s %8s %8s %8s %9s %9s %8s'%('candidate','Zcam_med','in-img%','IoU_med','ctr/H','Hratio','camH_m','subj%'))
res={}
for name,Xcb in cands.items():
    uv=proj(Xcb); Xc=(Rc@Xcb.reshape(-1,3).T).T+tc; Z=Xc[:,2].reshape(Kj.shape[:2])
    px1,px2=uv[...,0].min(1),uv[...,0].max(1); py1,py2=uv[...,1].min(1),uv[...,1].max(1)
    inimg=((px1>0)&(px2<1624)&(py1>0)&(py2<1240)).mean()
    ix1=np.maximum(px1,X1);ix2=np.minimum(px2,X2);iy1=np.maximum(py1,Y1);iy2=np.minimum(py2,Y2)
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    iou=inter/((px2-px1)*(py2-py1)+Bw*Bh-inter+1e-9)
    ctr=np.hypot((px1+px2)/2-(X1+X2)/2,(py1+py2)/2-(Y1+Y2)/2)/Bh
    subj=iou>0.25
    res[name]=dict(uv=uv,iou=iou,subj=subj,ctr=ctr)
    ch=cam_height(floor, {'J y-up m, s=100 (mm->cm)':100.0,'J z-up m, s=100':100.0,'J y-up m, s=0.1 (as-written)':0.1,'J y-up m, s=1':1.0,'J y-up m, s=1000':1000.0}[name])
    print('%-30s %9.1f %8.1f %8.3f %8.3f %9.3f %9.2f %8.0f'%(name,np.median(Z),100*inimg,np.median(iou),np.median(ctr),
        np.median((py2-py1)/Bh),ch,100*subj.mean()))
best='J y-up m, s=100 (mm->cm)'
uv=res[best]['uv']; subj=res[best]['subj']
print('\n=== %s : subject-frame detail (n=%d of 543) ==='%(best,subj.sum()))
px1,px2=uv[...,0].min(1),uv[...,0].max(1);py1,py2=uv[...,1].min(1),uv[...,1].max(1)
print('joint-hull H/proj vs box H: median ratio %.3f  (IQR %.3f-%.3f)'%(
  np.median((py2-py1)[subj]/Bh[subj]),np.percentile((py2-py1)[subj]/Bh[subj],25),np.percentile((py2-py1)[subj]/Bh[subj],75)))
print('centre dx px: median %+.1f (IQR %+.1f..%+.1f)   dy px: median %+.1f'%(
  np.median(((px1+px2)/2-(X1+X2)/2)[subj]),np.percentile(((px1+px2)/2-(X1+X2)/2)[subj],25),np.percentile(((px1+px2)/2-(X1+X2)/2)[subj],75),
  np.median(((py1+py2)/2-(Y1+Y2)/2)[subj])))
print('proj box bottom y2 - box y2 px: median %+.1f ;  proj top y1 - box y1: median %+.1f'%(
  np.median((py2-Y2)[subj]),np.median((py1-Y1)[subj])))
print('joints inside detected box: %.1f%% (subject frames)  %.1f%% (all)'%(
  100*((uv[...,0]>X1[:,None])&(uv[...,0]<X2[:,None])&(uv[...,1]>Y1[:,None])&(uv[...,1]<Y2[:,None]))[subj].mean(),
  100*((uv[...,0]>X1[:,None])&(uv[...,0]<X2[:,None])&(uv[...,1]>Y1[:,None])&(uv[...,1]<Y2[:,None])).mean()))
# lag sweep on subject frames: projCx vs boxCx and full 2D centre
print('\n=== time-lag sweep (subject frames only) ===')
pcx=(px1+px2)/2; pcy=(py1+py2)/2; bcx=(X1+X2)/2; bcy=(Y1+Y2)/2
for lag in range(-3,4):
    sl=slice(max(0,-lag),543-max(0,lag)); t=slice(max(0,lag),543-max(0,-lag))
    m=subj[t]
    if m.sum()<10: continue
    print(' lag %+d  n=%3d  |dCx| med %6.1f px  |dCy| med %6.1f px  corr(Cx) %.4f'%(
      lag,m.sum(),np.median(np.abs(pcx[t][m]-bcx[sl][m])),np.median(np.abs(pcy[t][m]-bcy[sl][m])),
      np.corrcoef(pcx[t][m],bcx[sl][m])[0,1]))
# overlays
for fi in [0,60,150,300,450]:
    im=cv2.imread(MI+'color/%06d.jpg'%fi)
    for (u,v) in uv[fi]:
        cv2.circle(im,(int(round(u)),int(round(v))),7,(0,0,255),-1)
        cv2.circle(im,(int(round(u)),int(round(v))),9,(255,255,255),2)
    cv2.rectangle(im,(int(X1[fi]),int(Y1[fi])),(int(X2[fi]),int(Y2[fi])),(0,255,0),3)
    for s_,e_ in [(0,1),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19)]:
        cv2.line(im,(int(uv[fi,s_,0]),int(uv[fi,s_,1])),(int(uv[fi,e_,0]),int(uv[fi,e_,1])),(255,0,255),2)
    cv2.imwrite('/tmp/r2-scratch/overlay_%03d.jpg'%fi,im)
print('\noverlays -> /tmp/r2-scratch/overlay_*.jpg ; IoU per frame saved')
np.save('/tmp/r2-scratch/uv_best.npy',uv); np.save('/tmp/r2-scratch/subj.npy',subj); np.save('/tmp/r2-scratch/iou.npy',res[best]['iou'])
