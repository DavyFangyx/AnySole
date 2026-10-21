import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); kp=np.load('/tmp/r2-scratch/kp.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
H={'nose':0,'l_hip':11,'r_hip':12,'l_knee':13,'r_knee':14,'l_ank':15,'r_ank':16,'l_bigtoe':20,'r_bigtoe':21,'l_heel':24,'r_heel':25,'l_sho':5,'r_sho':6}
rv,_=cv2.Rodrigues(Rc)
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
UV=np.empty(J.shape[:2]+(2,))
for i in range(len(Pc)):
    u,_=cv2.projectPoints(np.ascontiguousarray(Pc[i]),rv,tc.reshape(3,1),K,D); UV[i]=u[:,0,:]
N=543
def hp(sig,w=21):
    k=np.ones(w)/w; sm=np.convolve(sig,k,mode='same')
    # fix edges: use 'valid'-like normalization
    ones=np.convolve(np.ones_like(sig),k,mode='same')
    return sig-sm/ones
def xcorr_hp(a,b,mask,maxlag=12):
    ah=hp(a); bh=hp(b); out=[]
    for L in range(-maxlag,maxlag+1):
        j=np.arange(N)+L; ok=mask&(j>=0)&(j<N)
        x=ah[ok]; y=bh[j[ok]]
        if ok.sum()<40: continue
        r=np.corrcoef(x,y)[0,1]; out.append((L,r,ok.sum()))
    return out
SIG={
 'pelvis_y (SMPL j0 vs hip-mid)': (UV[:,0,1], 0.5*(kp[:,H['l_hip']]+kp[:,H['r_hip']])[:,1]),
 'L_ankle_y (SMPL j7 vs l_ank)':  (UV[:,7,1], kp[:,H['l_ank'],1]),
 'R_ankle_y (SMPL j8 vs r_ank)':  (UV[:,8,1], kp[:,H['r_ank'],1]),
 'L_foot_y  (SMPL j10 vs l_heel)':(UV[:,10,1],kp[:,H['l_heel'],1]),
 'L_ankle_x (SMPL j7 vs l_ank)':  (UV[:,7,0], kp[:,H['l_ank'],0]),
 'head_y    (SMPL j15 vs nose)':  (UV[:,15,1], kp[:,H['nose'],1]),
}
print('=== phase cross-correlation of HIGH-PASSED (21-frame moving-mean removed) trajectories ===')
print('    L = number of frames the kp is advanced;  peak at L* means proj@i matches kp@i+L*')
allpk=[]
for name,(a,b) in SIG.items():
    rows=xcorr_hp(a,b,gate)
    Ls=np.array([r[0] for r in rows]); rs=np.array([r[1] for r in rows])
    i=int(np.argmax(rs)); L0=Ls[i]
    # parabolic sub-frame
    if 0<i<len(rs)-1:
        d=(rs[i-1]-rs[i+1])/(2*(rs[i-1]-2*rs[i]+rs[i+1]))
    else: d=0.0
    print('  %-34s best L=%+d (%+.2f sub-frame) r=%.3f | r(L=0)=%+.3f  r(+1)=%+.3f r(-1)=%+.3f  n=%d'%(
        name,L0,L0+d,rs[i],rs[np.where(Ls==0)[0][0]],rs[np.where(Ls==1)[0][0]],rs[np.where(Ls==-1)[0][0]],rows[0][2]))
    allpk.append((name,L0+d,rs[i]))
print('\n  sampled curve for L_ankle_y / pelvis_y:')
for name in ['pelvis_y (SMPL j0 vs hip-mid)','L_ankle_y (SMPL j7 vs l_ank)']:
    rows=xcorr_hp(*SIG[name],gate)
    print('   ',name)
    print('     ',[(L,round(r,3)) for L,r,_ in rows if abs(L)<=8])
print('\n  peaks:',[(n,round(p,2),round(r,3)) for n,p,r in allpk])
