import numpy as np, sys
sys.path.insert(0,'/data/fangyuxuan/projects/gait')
from AnysoleWorkspace.tool._bvh_aligner_pose import parse_bvh_aligner
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
bvh=parse_bvh_aligner(WS+'/raw/bvh/20260804/mocap_ori_bvh/S5091/S5091_Skeleton3.bvh',trim_leading_seconds=0.0)
J=parse_bvh_aligner.__doc__
js=bvh['joints'].astype(np.float64); names=list(bvh['names'])
print('[bvh] frames %d joints %d fps %.4f frame_time %.6f trim %d'%(len(js),js.shape[1],bvh['fps'],bvh['frame_time'],bvh['trim_frames']))
print('[bvh] n_frames/fps = %.4f s'%(len(js)/bvh['fps']))
print('[bvh] raw units check: frame0 ptp=%.3f -> cm? %s'%(np.ptp(js[0],axis=0).max(), np.ptp(js[0],axis=0).max()>5.0))
print('[bvh] joint names:',names)
for nm in ['Hips','LeftFoot','LeftToeBase','RightFoot','RightToeBase','Head']:
    if nm in names:
        i=names.index(nm); print('  %-14s idx%2d mean over frames %s'%(nm,i,np.round(js[:,i].mean(0),4)))
# time grid: src_t = np.arange(n)*frame_time ; query
fr=np.load(WS+'/shared/facts/sessions/cam3/20260804/S5/S5091/frames.npz',allow_pickle=True)
q=fr['mocap_time_s']
src=np.arange(len(js))*bvh['frame_time']
print('[bvh] src span %.4f..%.4f  query span %.4f..%.4f  query_out_of_span=%d'%(src[0],src[-1],q[0],q[-1],int(((q<src[0])|(q>src[-1])).sum())))
pts=np.empty((len(q),)+js.shape[1:])
for j in range(js.shape[1]):
    for a in range(3): pts[:,j,a]=np.interp(np.clip(q,src[0],src[-1]),src,js[:,j,a])
if np.ptp(pts[0],axis=0).max()>5.0: pts*=0.01
print('[bvh] after scale: frame0 z-range %.3f..%.3f (m, z-up)'%(pts[0,:,2].min(),pts[0,:,2].max()))
print('[bvh] floor(p5 z)=%.4f m'%np.percentile(pts[:,:,2],5))

# SMPL joints -> candidate frames
K=np.load(MI+'keypoints.npy').astype(np.float64)      # (543,24,3) m, y-up (SMPL/marker frame)
to_zup=lambda X: np.column_stack([X[...,0],-X[...,2],X[...,1]])
cands={'SMPL as-is (y-up m)':K, 'SMPL->z-up (x,-z,y)':to_zup(K)}
# BVH reference joints (z-up m) and y-up variant
bvh_z=pts; bvh_y=np.column_stack([pts[...,0],pts[...,2],-pts[...,1]])
refs={'BVH z-up (adapter carpet frame)':bvh_z,'BVH y-up':bvh_y}
SMPL_FOOT={'L':10,'R':11}
BVH_FOOT={'L':names.index('LeftFoot'),'R':names.index('RightFoot')}
BVH_TOE={'L':names.index('LeftToeBase'),'R':names.index('RightToeBase')}
print('\n[axis test] per-side foot RMSE (m) and mm, median over 543 frames')
print('%-32s %-32s %8s %8s %8s'%('smpl cand','bvh ref','L mm','R mm','crossedL'))
for cn,C in cands.items():
    for rn,Rf in refs.items():
        for tgt,sm in [('foot',SMPL_FOOT)]:
            eL=np.linalg.norm(C[:,sm['L']]-Rf[:,BVH_FOOT['L']],axis=1)
            eR=np.linalg.norm(C[:,sm['R']]-Rf[:,BVH_FOOT['R']],axis=1)
            eX=np.linalg.norm(C[:,sm['L']]-Rf[:,BVH_FOOT['R']],axis=1)
            print('%-32s %-32s %8.1f %8.1f %8.1f'%(cn,rn,np.median(eL)*1000,np.median(eR)*1000,np.median(eX)*1000))
        # toe
        eT=np.linalg.norm(C[:,SMPL_FOOT['L']]-Rf[:,BVH_TOE['L']],axis=1)
        print('%-32s %-32s  ankle->toe L median %.1f mm'%(cn,rn,np.median(eT)*1000))
