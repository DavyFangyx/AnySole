import numpy as np, json, os
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
RAW=WS+'/raw/smpl/0804/0804smpl/mocap_ori_c3d/S5091/motion_neutral_smpl.npz'
raw=dict(np.load(RAW,allow_pickle=True))
print('=== raw npz metadata ===')
for k in sorted(raw):
    a=np.asarray(raw[k])
    if a.dtype.kind in 'US' or a.size<=12: print('  %-32s %s'%(k,str(a.ravel()[:4])[:120]))
    else: print('  %-32s shape=%s dtype=%s'%(k,a.shape,a.dtype))
