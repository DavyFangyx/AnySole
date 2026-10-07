import numpy as np, sys
sys.path.insert(0,'/data/fangyuxuan/projects/gait')
from AnysoleWorkspace.tool._bvh_aligner_pose import parse_bvh_aligner
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
bvh=parse_bvh_aligner(WS+'/raw/bvh/20260804/mocap_ori_bvh/S5091/S5091_Skeleton3.bvh',trim_leading_seconds=0.0)
names=list(bvh['names']); js=bvh['joints'].astype(np.float64)
fr=np.load(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/frames.npz',allow_pickle=True); q=fr['mocap_time_s']
src=np.arange(len(js))*bvh['frame_time']
pts=np.empty((len(q),)+js.shape[1:])
for j in range(js.shape[1]):
    for a in range(3): pts[:,j,a]=np.interp(np.clip(q,src[0],src[-1]),src,js[:,j,a])
assert np.ptp(pts[0],axis=0).max()>5.0; pts*=0.01
K=np.load(MI+'keypoints.npy').astype(np.float64)
to_zup=lambda X: np.stack([X[...,0],-X[...,2],X[...,1]],axis=-1)
# BVH variants
bvh_zup=pts                              # adapter carpet frame (m, z-up)
bvh_yup=np.stack([pts[...,0],pts[...,2],-pts[...,1]],axis=-1)  # inverse of (x,-z,y) -> raw y-up m
iF={'L':names.index('LeftFoot'),'R':names.index('RightFoot')}; iT={'L':names.index('LeftToeBase'),'R':names.index('RightToeBase')}
iH=names.index('Hips'); iHead=names.index('Head')
print('=== per-frame MEDIAN distance (mm) SMPL joint vs BVH joint ===')
rows=[('SMPL y-up (as stored)', K), ('SMPL->z-up (x,-z,y)', to_zup(K))]
cols=[('BVH z-up (carpet)', bvh_zup), ('BVH y-up (raw)', bvh_yup)]
for cn,C in rows:
    for rn,Rf in cols:
        d={}
        for side,si in (('L',10),('R',11)):
            d['foot%s'%side]=np.linalg.norm(C[:,si]-Rf[:,iF[side]],axis=1)
        d['footL_vs_FootR(cross)']=np.linalg.norm(C[:,10]-Rf[:,iF['R']],axis=1)
        d['pelvis_vs_Hips']=np.linalg.norm(C[:,0]-Rf[:,iH],axis=1)
        d['head15_vs_Head']=np.linalg.norm(C[:,15]-Rf[:,iHead],axis=1)
        print('%-24s vs %-20s '%(cn,rn)+' '.join('%s=%.0f'%(k,np.median(v)*1000) for k,v in d.items()))
print()
print('=== absolute positions, frame 0 (m) ===')
print('SMPL pelvis      ',np.round(K[0,0],3),'  SMPL L_foot',np.round(K[0,10],3),' R_foot',np.round(K[0,11],3))
print('SMPL->zup pelvis ',np.round(to_zup(K)[0,0],3),'  L_foot',np.round(to_zup(K)[0,10],3))
print('BVH  zup Hips    ',np.round(bvh_zup[0,iH],3),'  L_foot',np.round(bvh_zup[0,iF['L']],3),' L_toe',np.round(bvh_zup[0,iT['L']],3))
print('BVH  yup Hips    ',np.round(bvh_yup[0,iH],3),'  L_foot',np.round(bvh_yup[0,iF['L']],3))
print()
print('=== mean over 543 frames (m) ==='); 
print('SMPL->zup pelvis mean',np.round(to_zup(K)[:,0].mean(0),3),' BVH zup Hips mean',np.round(bvh_zup[:,iH].mean(0),3))
print('SMPL->zup Lfoot mean ',np.round(to_zup(K)[:,10].mean(0),3),' BVH zup Lfoot mean',np.round(bvh_zup[:,iF['L']].mean(0),3))
print('SMPL->zup Rfoot mean ',np.round(to_zup(K)[:,11].mean(0),3),' BVH zup Rfoot mean',np.round(bvh_zup[:,iF['R']].mean(0),3))
# carpet painting cross-check: feet in carpet frame vs painted cells
P=np.load(MI+'pressure.npz')['pressure']
painted=(P>0).any(0)   # (320,120)
print('\ncarpet painted cells %d of %d'%(painted.sum(),painted.size))
# carpet coords: x_i = i*1.25cm+0.625cm (i=0..119), y_j = -(j*1.25cm+0.625cm) (j=0..319)
def cell_of(X):  # X (T,3) m, z-up carpet frame
    i=np.floor((X[:,0]-0.00625)/0.0125).astype(int); j=np.floor((-X[:,1]-0.00625)/0.0125).astype(int)
    return i,j
for tag,arr,idx in [('SMPL Lfoot',to_zup(K),10),('SMPL Rfoot',to_zup(K),11),('BVH Lfoot',bvh_zup,iF['L']),('BVH Rfoot',bvh_zup,iF['R'])]:
    i,j=cell_of(arr[:,idx]); ok=(i>=0)&(i<120)&(j>=0)&(j<320)
    hit=(ok&painted[np.clip(j,0,319),np.clip(i,0,119)]).mean()
    print('  %-11s in-bounds %3d/543  lands-on-painted %.1f%%'%(tag,ok.sum(),100*hit))
