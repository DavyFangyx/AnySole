import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); sc=np.load('/tmp/r2-scratch/kp.npy')
gate=np.load('/tmp/r2-scratch/gate.npy')
rv,_=cv2.Rodrigues(Rc)
def chain(P): return (Rm@(100.0*P.reshape(-1,3)).T).T.reshape(P.shape)+tm
def uv_of(P):
    Pc=chain(P);o=np.empty(Pc.shape[:2]+(2,))
    for i in range(len(Pc)):
        u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D);o[i]=u[:,0,:]
    return o
UV=uv_of(J)
HN=['nose','l_eye','r_eye','l_ear','r_ear','l_sho','r_sho','l_elb','r_elb','l_wri','r_wri','l_hip','r_hip','l_kne','r_kne','l_ank','r_ank','head','neck','hip','l_btoe','r_btoe','l_stoe','r_stoe','l_heel','r_heel']
MAP=[(0,19,'SMPL0 pelvis <-> hip19'),(1,11,'SMPL1 L_hip'),(2,12,'SMPL2 R_hip'),(4,13,'SMPL4 L_knee'),(5,14,'SMPL5 R_knee'),
     (7,15,'SMPL7 L_ankle'),(8,16,'SMPL8 R_ankle'),(10,24,'SMPL10 L_foot<->l_heel'),(11,25,'SMPL11 R_foot<->r_heel'),
     (10,20,'SMPL10 L_foot<->l_btoe'),(11,21,'SMPL11 R_foot<->r_btoe'),(15,17,'SMPL15 head<->head17'),(12,18,'SMPL12 neck'),
     (16,5,'SMPL16 L_shoulder'),(17,6,'SMPL17 R_shoulder'),(18,7,'SMPL18 L_elbow'),(19,8,'SMPL19 R_elbow'),
     (20,9,'SMPL20 L_wrist'),(21,10,'SMPL21 R_wrist')]
e=np.stack([np.linalg.norm(UV[:,s]-kp[:,h],axis=1) for s,h,_ in MAP],axis=1)
print('=== A. per-joint 2D reprojection error, gate frames n=%d (px) ==='%gate.sum())
print('  %-26s %7s %8s %7s %8s %8s'%('pair','mean','median','p90','max','kp_score'))
for i,(s,h,nm) in enumerate(MAP):
    print('  %-26s %7.1f %8.1f %7.1f %8.1f %8.3f'%(nm,e[gate,i].mean(),np.median(e[gate,i]),np.percentile(e[gate,i],90),e[gate,i].max(),np.median(sc[gate,h,1] if sc.ndim==3 else 0)))
print('  %-26s %7.1f %8.1f %7.1f %8.1f'%('ALL 19 pairs',e[gate].mean(),np.median(e[gate]),np.percentile(e[gate],90),e[gate].max()))
print('  ALL 543 frames x 19 pairs: median %.1f  p90 %.1f'%(np.median(e),np.percentile(e,90)))
print('\n=== B. mirror / L-R control (median px, gate frames) ===')
for a,b,nm in [(10,24,'L_foot vs l_heel'),(11,25,'R_foot vs r_heel'),(10,25,'L_foot vs r_heel (MIRROR ctrl)'),(11,24,'R_foot vs l_heel (MIRROR ctrl)'),
               (16,5,'L_shoulder vs l_sho'),(16,6,'L_shoulder vs r_sho (MIRROR ctrl)'),(7,15,'L_ankle vs l_ank'),(7,16,'L_ankle vs r_ank (MIRROR ctrl)')]:
    print('  %-32s %6.1f'%(nm,np.median(np.linalg.norm(UV[gate,a]-kp[gate,b],axis=1))))
print('\n=== C. time-lag sweep (gate frames, per-joint median px) ===')
for lag in range(-3,4):
    j=np.arange(543)+lag; ok=gate&(j>=0)&(j<543)
    ee=np.stack([np.linalg.norm(UV[ok,s]-kp[j[ok],h],axis=1) for s,h,_ in MAP],axis=1)
    print('  lag %+d  n=%3d  median %6.1f  mean %6.1f'%(lag,ok.sum(),np.median(ee),ee.mean()))
print('\n=== D. scale: kp body height vs projected body height (gate frames, px) ===')
h_kp=kp[gate][:,[15,16,20,21,24,25],1].max(1)-kp[gate][:,17,1]
h_pj=UV[gate][:,[7,8,10,11],1].max(1)-UV[gate][:,15,1]
print('  kp(head-top->lowest foot) median %.1f ; proj(head joint->lowest foot joint) median %.1f ; ratio %.3f'%(np.median(h_kp),np.median(h_pj),np.median(h_kp/h_pj)))
print('\n=== E. bias after the (negligible) best constant correction ===')
print('  dx median %+.1f px ; dy median %+.1f px ; |dx|<20px frac %.2f ; |dy|<20px frac %.2f'%(
  np.median((UV[gate][:, [s for s,_,_ in MAP],0]-kp[gate][:,[h for _,h,_ in MAP],0])),
  np.median((UV[gate][:, [s for s,_,_ in MAP],1]-kp[gate][:,[h for _,h,_ in MAP],1])),
  float((np.abs(UV[gate][:, [s for s,_,_ in MAP],0]-kp[gate][:,[h for _,h,_ in MAP],0])<20).mean()),
  float((np.abs(UV[gate][:, [s for s,_,_ in MAP],1]-kp[gate][:,[h for _,h,_ in MAP],1])<20).mean())))
print('\n=== F. control candidates (all 543 frames) ===')
b=np.load(MI+'bbox.npy');X1,Y1,X2,Y2=b[:,1],b[:,2],b[:,3],b[:,4]
tozup=lambda X:np.stack([X[...,0],-X[...,2],X[...,1]],axis=-1)
Rconv=np.asarray(np.load(WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz',allow_pickle=True)['raw_c3d_to_smpl'])[:3,:3]
cands={'CORRECT J(m),s=100':(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape),
       's=0.1 (as-written)':(Rm@(0.1*J.reshape(-1,3)).T).T.reshape(J.shape),
       's=1':(Rm@(1.0*J.reshape(-1,3)).T).T.reshape(J.shape),
       'z-up first, s=100':(Rm@(100.0*tozup(J).reshape(-1,3)).T).T.reshape(J.shape),
       'Rconv permuted, s=100':(Rm@(100.0*(Rconv.T@J.reshape(-1,3).T)).T).T.reshape(J.shape)}
print('  %-24s %8s %9s %8s'%('candidate','in-img%','IoU_med','hull/boxH'))
for nm,Xcb in cands.items():
    P=uv_of.__wrapped__ if False else None
    Pc=Xcb;o=np.empty(Pc.shape[:2]+(2,))
    for i in range(len(Pc)):
        u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D);o[i]=u[:,0,:]
    x1,x2=o[...,0].min(1),o[...,0].max(1);y1,y2=o[...,1].min(1),o[...,1].max(1)
    inimg=((x1>0)&(x2<1624)&(y1>0)&(y2<1240)).mean()
    ix1=np.maximum(x1,X1);ix2=np.minimum(x2,X2);iy1=np.maximum(y1,Y1);iy2=np.minimum(y2,Y2)
    iou=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)/((x2-x1)*(y2-y1)+(X2-X1)*(Y2-Y1)-np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)+1e-9)
    print('  %-24s %8.1f %9.3f %8.3f'%(nm,100*inimg,np.median(iou),np.median((y2-y1)/(Y2-Y1))))
