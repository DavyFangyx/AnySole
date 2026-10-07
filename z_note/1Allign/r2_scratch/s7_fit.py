import numpy as np, cv2, json
from scipy.optimize import minimize
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy')
MAP=[(0,19),(1,11),(2,12),(4,13),(5,14),(7,15),(8,16),(15,17),(12,18),(16,5),(17,6),(18,7),(19,8),(20,9),(21,10)]
S=[s for s,_ in MAP]; H=[h for _,h in MAP]
rv,_=cv2.Rodrigues(Rc)
def chain(P): return (Rm@(100.0*P.reshape(-1,3)).T).T.reshape(P.shape)+tm
def uv_of(P):
    Pc=chain(P); out=np.empty(Pc.shape[:2]+(2,))
    for i in range(len(Pc)):
        u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D); out[i]=u[:,0,:]
    return out
def jerr(P,idx=None):
    if idx is None: idx=slice(None)
    u=uv_of(P)[idx][:,S]; return u-kp[idx][:,H]
def cost(par,model):
    if model=='none': P=J
    elif model=='delta': P=J+par[:3]
    elif model=='yaw':
        a=par[0]; c,s=np.cos(a),np.sin(a); Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        P=(Ry@J.reshape(-1,3).T).T.reshape(J.shape)
    elif model=='delta+yaw':
        a=par[3]; c,s=np.cos(a),np.sin(a); Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        P=((Ry@J.reshape(-1,3).T).T.reshape(J.shape))+par[:3]
    e=jerr(P); return float(np.mean(np.sum(e**2,axis=2)))
# subject frames: RTMPose kp in YOLOX box (2D-only gate, independent of our chain)
b=np.load(MI+'bbox.npy'); inside=((kp[:,:,0]>b[:,1,None])&(kp[:,:,0]<b[:,3,None])&(kp[:,:,1]>b[:,2,None])&(kp[:,:,1]<b[:,4,None])).mean(1)
gate=inside>0.8
print('gate (kp cloud inside YOLOX box >80%%): %d/543 frames'%gate.sum())
np.save('/tmp/r2-scratch/gate.npy',gate)
for model,p0 in [('none',np.zeros(1)),('delta',np.zeros(3)),('yaw',np.zeros(1)),('delta+yaw',np.zeros(4))]:
    if model=='none':
        e=jerr(J)[gate]; rms=np.sqrt(np.mean(np.sum(e**2,axis=2))); med=np.median(np.linalg.norm(e,axis=2))
        print('%-10s  no fit: per-joint RMS %.1f px  median %.1f px'%(model,rms,med)); continue
    f=lambda p: cost(p,model)
    r=minimize(f,p0,method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-3,maxiter=6000,maxfev=6000))
    e=jerr((J+r.x[:3]) if model=='delta' else (J if model=='yaw' else J+r.x[:3]))[gate]
    d=np.linalg.norm(e,axis=2)
    extra=''
    if model in ('delta',): extra=' delta_m=%s |d|=%.3f m'%(np.round(r.x[:3],4),np.linalg.norm(r.x[:3]))
    if model=='yaw': extra=' yaw=%.3f deg'%np.degrees(r.x[0])
    if model=='delta+yaw': extra=' delta_m=%s |d|=%.3f m yaw=%.3f deg'%(np.round(r.x[:3],4),np.linalg.norm(r.x[:3]),np.degrees(r.x[3]))
    print('%-10s  RMS %.1f px  median %.1f px  p90 %.1f  cost %.4g -> %.4g%s'%(model,np.sqrt(np.mean(d**2)),np.median(d),np.percentile(d,90),cost(p0,model),r.fun,extra))
# per-joint after delta+yaw
r=minimize(lambda p: cost(p,'delta+yaw'),np.zeros(4),method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-3,maxiter=6000,maxfev=6000))
P=(J+r.x[:3]); a=r.x[3]; c,s=np.cos(a),np.sin(a); Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]]); P=((Ry@J.reshape(-1,3).T).T.reshape(J.shape))+r.x[:3]
e=np.linalg.norm(uv_of(P)[gate][:,S]-kp[gate][:,H],axis=2)
names=['pelvis','L_hip','R_hip','L_knee','R_knee','L_ankle','R_ankle','head','neck','L_sho','R_sho','L_elb','R_elb','L_wri','R_wri']
print('\nper-joint after delta+yaw (gate frames, px):')
for i,n in enumerate(names): print('  %-8s median %6.1f  mean %6.1f'%(n,np.median(e[:,i]),e[:,i].mean()))
print('\nvertical-only residual (dy) after correction: median %.1f px'%np.median(np.abs((uv_of(P)[gate][:,S]-kp[gate][:,H])[:,:,1])))
print('horizontal (dx) after correction: median %+.1f px'%np.median((uv_of(P)[gate][:,S]-kp[gate][:,H])[:,:,0]))
# scale check: kp body height vs projected body height on gate frames
h_kp=kp[gate][:,[15,16,20,22,23,25],1].max(1)-kp[gate][:,17,1]
Hp=uv_of(P)[gate]; h_pj=Hp[:,[7,8,10,11],1].max(1)-Hp[:,15,1]
print('\nscale: kp height (head-top->lowest foot kp) median %.1f px ; projected (head joint->lowest foot joint) median %.1f px ; ratio %.3f'%(np.median(h_kp),np.median(h_pj),np.median(h_kp/h_pj)))
