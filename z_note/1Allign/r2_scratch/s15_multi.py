import numpy as np, cv2, json, os, glob, sys
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
S=np.array(S); H=np.array(H)
def frontend_kp(fe, n_target, frame_ids):
    files=sorted(glob.glob(fe+'*.npy'))
    kps={}
    for f in files:
        d=np.load(f,allow_pickle=True).item(); kps[int(d['frame_id'])]=np.asarray(d['keypoints'],dtype=np.float64)
    out=np.full((n_target,26,2),np.nan)
    for i,fid in enumerate(frame_ids):
        if int(fid) in kps: out[i]=kps[int(fid)]
    return out
def sweep(mi,fe,cal_path,label):
    cam=json.load(open(cal_path))['cameras']['cam3']
    K=np.array(cam['K'],dtype=np.float64); D=np.array(cam['D'],dtype=np.float64)
    Rc=np.array(cam['R'],dtype=np.float64); tc=np.array(cam['t'],dtype=np.float64).reshape(3,1)
    cf=json.load(open(cal_path))
    Rm=np.array(cf['mocap_raw_to_checkerboard_world']['R']); tm=np.array(cf['mocap_raw_to_checkerboard_world']['t'])
    J=np.load(mi+'keypoints.npy').astype(np.float64)
    fid=np.load(mi+'frame_id.npy')
    kp=frontend_kp(fe,len(J),fid)
    valid=~np.isnan(kp[:,0,0])
    Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
    u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc,K,D)
    UV=u.reshape(J.shape[0],-1,2)
    # gate: kp cloud inside YOLOX bbox
    gate=np.zeros(len(J),bool)
    bp=mi+'bbox.npy'
    if os.path.exists(bp):
        bb=np.load(bp)
        for i in range(len(J)):
            if not valid[i]: continue
            x1,y1,x2,y2=bb[i,1:5]; k=kp[i][H]
            gate[i]=np.isfinite(k).all() and (k[:,0]>=x1-10).all() and (k[:,0]<=x2+10).all() and (k[:,1]>=y1-10).all() and (k[:,1]<=y2+10).all()
    else: gate=valid.copy()
    N=len(J); g=gate&valid
    def kp_at(delta):
        f=np.arange(N)+delta
        lo=np.clip(np.floor(f).astype(int),0,N-1); hi=np.clip(lo+1,0,N-1); w=(f-lo)[:,None,None]
        return kp[lo]*(1-w)+kp[hi]*w,(f>=0)&(f<N-1)
    rows=[]
    for d in np.arange(-4,20.01,0.5):
        K2,ok0=kp_at(d); ok=g&ok0
        if ok.sum()<20: continue
        O=UV[ok][:,S]-K2[ok][:,H]
        dx=np.median(O[...,0],axis=1); dy=np.median(O[...,1],axis=1)
        R=O.copy(); R[...,0]-=dx[:,None]; R[...,1]-=dy[:,None]
        rows.append((d,np.median(np.hypot(dx,dy)),np.median(np.linalg.norm(R,axis=2)),ok.sum()))
    rows=np.array(rows)
    if len(rows)==0: return None
    i=int(np.argmin(rows[:,1]))
    return dict(label=label,n=int(g.sum()),N=N,lag=rows[i,0],align=rows[i,1],shape=rows[i,2],
                align0=rows[np.argmin(np.abs(rows[:,0]))][1],shape0=rows[np.argmin(np.abs(rows[:,0]))][2])
res=[]
for date in sorted(os.listdir(WS+'/model_inputs/MotionPRO/adapter_v1/cam3')):
    for subj in sorted(os.listdir(WS+f'/model_inputs/MotionPRO/adapter_v1/cam3/{date}')):
        for sess in sorted(os.listdir(WS+f'/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}')):
            mi=f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}/{sess}/'
            fe=f'{WS}/shared/frontends/rtmpose_halpe26/v1/{date}/{subj}/{sess}/keypoints/'
            calp=f'{WS}/protocol/calibration/{date}.json'
            if not os.path.exists(mi+'keypoints.npy'): print('skip (no keypoints.npy)',sess); continue
            if not os.path.isdir(fe): print('skip (no frontend)',date,subj,sess); continue
            r=sweep(mi,fe,calp,f'{date}/{subj}/{sess}')
            if r: res.append(r); print('%-22s N=%4d gate=%4d | lag*=%+5.1f f (%+4.0f ms) align %.1f->%.1f shape %.1f->%.1f'%(
                r['label'],r['N'],r['n'],r['lag'],r['lag']*25,r['align0'],r['align'],r['shape0'],r['shape']))
print('\n=== summary ===')
lags=np.array([r['lag'] for r in res]); print('n_sessions=%d  lag* mean %+.2f frames (%+.0f ms) median %+.2f  min %+.2f max %+.2f'%(
  len(res),lags.mean(),lags.mean()*25,np.median(lags),lags.min(),lags.max()))
print('all positive:', bool((lags>0).all()))
np.save('/tmp/r2-scratch/multi_lag.npy',np.array([[r['lag'],r['align0'],r['align'],r['shape0'],r['shape'],r['n'],r['N']] for r in res]))
json.dump(res,open('/tmp/r2-scratch/multi_lag.json','w'),indent=1)
