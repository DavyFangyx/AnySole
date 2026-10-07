import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
rv,_=cv2.Rodrigues(Rc)
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
UV=np.empty(J.shape[:2]+(2,))
for i in range(len(Pc)):
    u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D); UV[i]=u[:,0,:]
N=543
# ---- stride period from the raw 3D mocap pelvis height (clean, no 2D noise) ----
pel_z=J[:,0,1]  # SMPL y-up: pelvis height in meters
d=np.diff(pel_z)
print('=== stride structure (from raw smpl.npy, y-up meters) ===')
print('  pelvis height: mean %.3f m  std %.3f m  min %.3f max %.3f'%(pel_z.mean(),pel_z.std(),pel_z.min(),pel_z.max()))
x=pel_z-pel_z.mean(); ac=np.correlate(x,x,'full')[N-1:]; ac/=ac[0]
pk=[l for l in range(8,120) if ac[l]>ac[l-1] and ac[l]>=ac[l+1]]
print('  autocorr peaks (lags):',pk[:6],' values',np.round(ac[pk[:6]],3).tolist())
print('  -> step/stride period ~',pk[:3],'frames =',[round(p/40*1000) for p in pk[:3]],'ms')
# ---- same for kp pelvis y and proj pelvis y ----
kp_pel=0.5*(kp[:,11]+kp[:,12])[:,1]; pr_pel=UV[:,0,1]
for nm,sig in (('kp pelvis_y',kp_pel),('proj pelvis_y',pr_pel)):
    y=sig[gate]-sig[gate].mean(); n=len(y); a=np.correlate(y,y,'full')[n-1:]; a/=a[0]
    ps=[l for l in range(6,120) if a[l]>a[l-1] and a[l]>=a[l+1]]
    print('  %s: autocorr peaks %s (values %s)  std %.1f px'%(nm,ps[:5],np.round(a[ps[:5]],3).tolist(),sig[gate].std()))
# ---- wide-lag cross-correlation of RAW (mean removed) traces, no high-pass ----
def xc(a,b,mask,maxlag=60):
    a=a-a[mask].mean(); b=b-b[mask].mean(); out=[]
    for L in range(-maxlag,maxlag+1):
        j=np.arange(N)+L; ok=mask&(j>=0)&(j<N)
        out.append((L,np.corrcoef(a[ok],b[j[ok]])[0,1],ok.sum()))
    return out
print('\n=== wide-lag cross-correlation (raw, mean-removed) proj@i vs kp@i+L ===')
for nm,a,b in (('pelvis_y',pr_pel,kp_pel),('pelvis_x',UV[:,0,0],0.5*(kp[:,11]+kp[:,12])[:,0]),
               ('L_ankle_y',UV[:,7,1],kp[:,15,1]),('R_ankle_y',UV[:,8,1],kp[:,16,1])):
    rows=xc(a,b,gate)
    best=max(rows,key=lambda r:r[1])
    print('  %-10s peak L=%+d r=%.3f | r(0)=%+.3f  r(+5)=%+.3f r(+10)=%+.3f r(-5)=%+.3f r(-10)=%+.3f'%(
        nm,best[0],best[1],[r for L,r,_ in rows if L==0][0],[r for L,r,_ in rows if L==5][0],
        [r for L,r,_ in rows if L==10][0],[r for L,r,_ in rows if L==-5][0],[r for L,r,_ in rows if L==-10][0]))
    if nm=='pelvis_y':
        print('    curve:',[(L,round(r,2)) for L,r,_ in rows if L%5==0])
