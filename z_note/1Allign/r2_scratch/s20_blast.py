import numpy as np, cv2, json, os
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
prev=json.load(open('/tmp/r2-scratch/multi_lag.json')); wide=json.load(open('/tmp/r2-scratch/wide_lag.json'))
w={r['label']:r for r in wide}
rows=[]
for r in prev:
    lab=r['label']; date,subj,sess=lab.split('/')
    mi=f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}/{sess}/'
    try:
        J=np.load(mi+'keypoints.npy').astype(float)
        cal=json.load(open(f'{WS}/protocol/calibration/{date}.json')); cam=cal['cameras']['cam3']
        K=np.array(cam['K'],float);D=np.array(cam['D'],float);Rc=np.array(cam['R'],float);tc=np.array(cam['t'],float).reshape(3,1)
        Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
        Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
        u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc,K,D); UV=u.reshape(len(J),-1,2)
        c=UV.mean(1); v=np.linalg.norm(np.gradient(c,axis=0),axis=1)*40
        sp=np.median(v)
        lag=r['lag'] if r['lag']>-4 else (w.get(lab,{}).get('lag',r['lag']))
        al=r['align0']
        rows.append((lab,sp,al,lag,al/max(sp/40,1e-9)))
    except Exception as e: print('err',lab,e)
rows.sort(key=lambda x:-x[4])
print('%-24s %8s %8s %8s %8s'%('session','imgspd','align0','lag*','align0/spd'))
print('(imgspd=median projected image speed px/s ; align0 px ; lag* frames ; last col ~ implied |lag| in frames)')
for lab,sp,al,lag,imp in rows[:14]:
    print('%-24s %8.1f %8.1f %+8.1f %8.1f'%(lab,sp,al,lag,imp))
print('...')
tru=[r for r in rows if r[1]>120]     # >3 px/frame of real motion
print('\n=== sessions with real image motion (>120 px/s = 3 px/frame) : n=%d ==='%len(tru))
good=[r for r in tru if r[2]<=8]
print('  of these, align0<=8 px (genuinely synced): n=%d (%.0f%%)'%(len(good),100*len(good)/len(tru)))
print('  misaligned (>8 px): n=%d ; align0 median %.1f px, implied |lag| median %.1f frames (%.0f ms)'%(
   len(tru)-len(good),np.median([r[2] for r in tru if r[2]>8]),np.median([r[4] for r in tru if r[2]>8]),np.median([r[4] for r in tru if r[2]>8])*25))
al=np.array([r[2] for r in tru]); 
print('  align0 distribution: p25 %.1f median %.1f p75 %.1f p90 %.1f max %.1f'%(np.percentile(al,25),np.median(al),np.percentile(al,75),np.percentile(al,90),al.max()))
lagv=np.array([r[3] for r in rows if r[1]>120])
print('  measured lag* (all with motion): median %+.1f  p10 %+.1f  p90 %+.1f  min %+.1f max %+.1f frames'%(
  np.median(lagv),np.percentile(lagv,10),np.percentile(lagv,90),lagv.min(),lagv.max()))
print('  |lag*| <=2 frames (well synced): %d / %d'%(int((np.abs(lagv)<=2).sum()),len(lagv)))
