import numpy as np, cv2, json, os, glob
exec(open('/tmp/r2-scratch/s15_multi.py').read().split("res=[]")[0])   # reuse sweep machinery
def sweep_wide(mi,fe,cal_path,label,lo=-30,hi=30):
    cam=json.load(open(cal_path))['cameras']['cam3']
    K=np.array(cam['K'],dtype=np.float64); D=np.array(cam['D'],dtype=np.float64)
    Rc=np.array(cam['R'],dtype=np.float64); tc=np.array(cam['t'],dtype=np.float64).reshape(3,1)
    cf=json.load(open(cal_path)); Rm=np.array(cf['mocap_raw_to_checkerboard_world']['R']); tm=np.array(cf['mocap_raw_to_checkerboard_world']['t'])
    J=np.load(mi+'keypoints.npy').astype(np.float64); fid=np.load(mi+'frame_id.npy')
    kp=frontend_kp(fe,len(J),fid); valid=~np.isnan(kp[:,0,0])
    Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
    u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc,K,D); UV=u.reshape(len(J),-1,2)
    gate=np.zeros(len(J),bool); bp=mi+'bbox.npy'
    bb=np.load(bp)
    for i in range(len(J)):
        if not valid[i]: continue
        x1,y1,x2,y2=bb[i,1:5]; k=kp[i][H]
        gate[i]=np.isfinite(k).all() and (k[:,0]>=x1-10).all() and (k[:,0]<=x2+10).all() and (k[:,1]>=y1-10).all() and (k[:,1]<=y2+10).all()
    N=len(J); g=gate&valid
    def kp_at(d):
        f=np.arange(N)+d; lo2=np.clip(np.floor(f).astype(int),0,N-1); hi2=np.clip(lo2+1,0,N-1); w=(f-lo2)[:,None,None]
        return kp[lo2]*(1-w)+kp[hi2]*w,(f>=0)&(f<N-1)
    rows=[]
    for d in np.arange(lo,hi+0.01,1.0):
        K2,ok0=kp_at(d); ok=g&ok0
        if ok.sum()<20: continue
        O=UV[ok][:,S]-K2[ok][:,H]; dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
        R=O.copy(); R[...,0]-=dx[:,None]; R[...,1]-=dy[:,None]
        rows.append((d,np.median(np.hypot(dx,dy)),np.median(np.linalg.norm(R,axis=2)),ok.sum()))
    rows=np.array(rows)
    i=int(np.argmin(rows[:,1]))
    return dict(label=label,n=int(g.sum()),N=N,lag=rows[i,0],align=rows[i,1],shape=rows[i,2],align0=rows[np.argmin(np.abs(rows[:,0]))][1])
prev=json.load(open('/tmp/r2-scratch/multi_lag.json'))
targets=[r['label'] for r in prev if r['align0']>15]
print('widening sweep for %d sessions with align0>15 px ...'%len(targets))
out=[]
for lab in targets:
    date,subj,sess=lab.split('/')
    mi=f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}/{sess}/'
    fe=f'{WS}/shared/frontends/rtmpose_halpe26/v1/{date}/{subj}/{sess}/keypoints/'
    r=sweep_wide(mi,fe,f'{WS}/protocol/calibration/{date}.json',lab)
    out.append(r)
    print('%-22s gate=%4d align0 %6.1f -> align* %6.1f at lag %+5.1f f (%+5.0f ms)  shape %.1f->%.1f'%(lab,r['n'],r['align0'],r['align'],r['lag'],r['lag']*25,0,0))
json.dump(out,open('/tmp/r2-scratch/wide_lag.json','w'),indent=1)
print('\n=== widened summary ===')
w=np.array([[r['lag'],r['align0'],r['align']] for r in out])
good=w[w[:,2]<12]
print('of %d, those a time shift FIXES (align*<12px): %d'%(len(w),len(good)))
print('  their lag*: ',sorted(good[:,0].tolist()))
print('  lags in ms:',sorted((good[:,0]*25).astype(int).tolist()))
print('the rest (%d) keep align*>12px -> NOT a time shift (2D tracking/data issue)'%(len(w)-len(good)))
