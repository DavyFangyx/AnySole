import numpy as np, cv2, json, os, glob
WS='/data/fangyuxuan/projects/gait/AnysoleWorkspace'
S=[0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]; H=[19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10]
S=np.array(S); H=np.array(H)
def load_kp(fe,target_fid):
    d={}
    for f in glob.glob(fe+'*.npy'):
        o=np.load(f,allow_pickle=True).item(); d[int(o['frame_id'])]=np.asarray(o['keypoints'],float)
    out=np.full((len(target_fid),26,2),np.nan)
    for i,fi in enumerate(target_fid):
        if int(fi) in d: out[i]=d[int(fi)]
    return out
def cloud_bb(UV):
    return np.stack([UV[:,:,0].min(1),UV[:,:,1].min(1),UV[:,:,0].max(1),UV[:,:,1].max(1)],1)
def iou_vec(c,b):
    x1=np.maximum(c[:,0],b[:,0]); y1=np.maximum(c[:,1],b[:,1]); x2=np.minimum(c[:,2],b[:,2]); y2=np.minimum(c[:,3],b[:,3])
    iw=np.clip(x2-x1,0,None); ih=np.clip(y2-y1,0,None); inter=iw*ih
    ua=(c[:,2]-c[:,0])*(c[:,3]-c[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return np.where(ua>0,inter/np.maximum(ua,1e-9),0)
def run(date,subj,sess):
    mi=f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}/{sess}/'
    fe=f'{WS}/shared/frontends/rtmpose_halpe26/v1/{date}/{subj}/{sess}/keypoints/'
    cal=json.load(open(f'{WS}/protocol/calibration/{date}.json')); cam=cal['cameras']['cam3']
    K=np.array(cam['K'],float);D=np.array(cam['D'],float);Rc=np.array(cam['R'],float);tc=np.array(cam['t'],float).reshape(3,1)
    Rm=np.array(cal['mocap_raw_to_checkerboard_world']['R']);tm=np.array(cal['mocap_raw_to_checkerboard_world']['t'])
    J=np.load(mi+'keypoints.npy').astype(float); fid=np.load(mi+'frame_id.npy'); bb=np.load(mi+'bbox.npy')
    kp=load_kp(fe,fid); N=len(J)
    Pc=(Rm@(100.0*J.reshape(-1,3)).T).T.reshape(J.shape)+tm
    u,_=cv2.projectPoints(Pc.reshape(-1,1,3),cv2.Rodrigues(Rc)[0],tc,K,D); UV=u.reshape(N,-1,2)
    C=cloud_bb(UV); B=bb[:,1:5]
    out=[]
    for L in range(-25,26):
        j=np.arange(N)+L; ok=(j>=0)&(j<N)
        y=iou_vec(C[ok],B[j[ok]])
        out.append((L,np.median(y)))
    o=np.array(out); pk=o[np.argmax(o[:,1]),0]
    return pk,o
print('session              IoU-peak lag (frames/ms)   IoU@0    IoU@peak   (independent YOLOX detector)')
for date,subj,sess,known in [('20260804','S5','S5091',9.5),('20260804','S7','S7031',9.0),('20260804','S6','S6092',8.5),
                             ('20260810','S13','S13093',-14.0),('20260808','S11','S11101',-17.0),('20260810','S13','S13082',-12.0),
                             ('20260804','S5','S5011',None),('20260810','S14','S14013',None)]:
    try:
        pk,o=run(date,subj,sess)
        i0=np.argmin(np.abs(o[:,0]))
        print('%-8s %-7s %-6s  %+6.1f (%+5.0f ms)   %.3f    %.3f    jt-sweep lag*=%s'%(
            date,subj,sess,pk,pk*25,o[i0,1],o[np.argmax(o[:,1]),1],known))
    except Exception as e: print(date,subj,sess,'ERR',e)
