import numpy as np, cv2, json
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
MI=WS+'/model_inputs/MotionPRO/adapter_v1/cam3/20260804/S5/S5091/'
cal=json.load(open(WS+'/protocol/calibration/20260804.json')); cam=cal['cameras']['cam3']
K=np.array(cam['K']);D=np.array(cam['D']);Rc=np.array(cam['R']);tc=np.array(cam['t'])
Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
J=np.load(MI+'keypoints.npy').astype(np.float64); bb=np.load(MI+'bbox.npy'); gate=np.load('/tmp/r2-scratch/gate.npy')
Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc.reshape(3,1),K,D)
UV=u.reshape(J.shape[0],-1,2)
N=len(J)
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
print('=== YOLOX-detector cross-check (independent of the RTMPose frontend) ===')
print('  lag | median IoU(proj joint-hull, YOLOX box) | median boxH/cloudH | n')
rows=[]
for L in range(-2,21):
    j=np.arange(N)+L; ok=gate&(j>=0)&(j<N)
    A=[];B=[]
    for i in np.where(ok)[0]:
        p=UV[i]; c=[p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()]
        b=bb[j[i],1:5]
        A.append(iou(c,b)); B.append((b[3]-b[1])/(c[3]-c[1]))
    A=np.array(A);B=np.array(B)
    rows.append((L,np.median(A),np.median(B),ok.sum()))
    if L%2==0 or L in (9,11): print('  %+3d | %6.3f | %6.3f | %d'%(L,np.median(A),np.median(B),ok.sum()))
r=np.array(rows); print('  ARGMAX IoU at lag %+d (%.3f) ; at lag 0 %.3f'%(r[np.argmax(r[:,1]),0],r[:,1].max(),r[np.argmin(np.abs(r[:,0]))][1]))
# restricted to YOLOX-box-height correlation (box is a body proxy in the image)
print('\n=== box height (y2-y1) vs projected cloud height over lag, HIGH-PASSED (21f) ===')
bh=bb[:,4]-bb[:,3]
ch=np.array([UV[i][:,1].max()-UV[i][:,1].min() for i in range(N)])
def hp(s,w=21):
    k=np.ones(w)/w; return s-np.convolve(s,k,'same')/np.convolve(np.ones_like(s),k,'same')
a=hp(bh); b=hp(ch)
for L in range(-10,21,1):
    j=np.arange(N)+L; ok=gate&(j>=0)&(j<N)
    if ok.sum()<40: continue
    print('  L=%+3d r=%+.3f n=%d'%(L,np.corrcoef(a[ok],b[j[ok]])[0,1],ok.sum()))
