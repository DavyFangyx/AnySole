"""S1: shared sweep machinery (verbatim copy of /tmp/r2-scratch/s15_multi.py sweep(),
   parameterised by lag range/step so the same estimator can run wide)."""
import numpy as np, cv2, json, os, glob

WS = '/data/fangyuxuan/projects/gait/AnysoleWorkspace'
S = np.array([0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21])
H = np.array([19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10])
# SMPL joint index for the 17 mapped pairs (for per-joint work)
SMPL_PAIRS = [0,1,2,4,5,7,8,10,11,12,15,16,17,18,19,20,21]
HALPE_PAIRS = [19,11,12,13,14,15,16,24,25,18,17,5,6,7,8,9,10]

def frontend_kp(fe, n_target, frame_ids):
    files = sorted(glob.glob(fe + '*.npy'))
    kps = {}
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        kps[int(d['frame_id'])] = np.asarray(d['keypoints'], dtype=np.float64)
    out = np.full((n_target, 26, 2), np.nan)
    for i, fid in enumerate(frame_ids):
        if int(fid) in kps:
            out[i] = kps[int(fid)]
    return out

def project(cal_path, J):
    cam = json.load(open(cal_path))['cameras']['cam3']
    K = np.array(cam['K'], dtype=np.float64); D = np.array(cam['D'], dtype=np.float64)
    Rc = np.array(cam['R'], dtype=np.float64); tc = np.array(cam['t'], dtype=np.float64).reshape(3, 1)
    cf = json.load(open(cal_path))
    Rm = np.array(cf['mocap_raw_to_checkerboard_world']['R']); tm = np.array(cf['mocap_raw_to_checkerboard_world']['t'])
    Pc = (Rm @ (100.0 * J.reshape(-1, 3)).T).T.reshape(J.shape) + tm
    u, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), cv2.Rodrigues(Rc)[0], tc, K, D)
    return u.reshape(J.shape[0], -1, 2)

def load_session(date, subj, sess, root=WS):
    mi = f'{root}/model_inputs/MotionPRO/adapter_v1/cam3/{date}/{subj}/{sess}/'
    fe = f'{root}/shared/frontends/rtmpose_halpe26/v1/{date}/{subj}/{sess}/keypoints/'
    calp = f'{root}/protocol/calibration/{date}.json'
    if not os.path.exists(mi + 'keypoints.npy'): return None
    if not os.path.isdir(fe): return None
    J = np.load(mi + 'keypoints.npy').astype(np.float64)
    fid = np.load(mi + 'frame_id.npy')
    kp = frontend_kp(fe, len(J), fid)
    valid = ~np.isnan(kp[:, 0, 0])
    UV = project(calp, J)
    N = len(J)
    bb = np.load(mi + 'bbox.npy') if os.path.exists(mi + 'bbox.npy') else None
    if bb is not None:
        gate = np.zeros(N, bool)
        for i in range(N):
            if not valid[i]: continue
            x1, y1, x2, y2 = bb[i, 1:5]; k = kp[i][H]
            gate[i] = np.isfinite(k).all() and (k[:, 0] >= x1 - 10).all() and (k[:, 0] <= x2 + 10).all() \
                      and (k[:, 1] >= y1 - 10).all() and (k[:, 1] <= y2 + 10).all()
    else:
        gate = valid.copy()
    return dict(J=J, fid=fid, kp=kp, valid=valid, UV=UV, N=N, bb=bb, gate=gate)

def sweep(se, label, lo=-4, hi=20, step=0.5, min_ok=20):
    J, kp, valid, UV, N, gate = se['J'], se['kp'], se['valid'], se['UV'], se['N'], se['gate']
    g = gate & valid
    def kp_at(delta):
        f = np.arange(N) + delta
        lo_i = np.clip(np.floor(f).astype(int), 0, N - 1); hi_i = np.clip(lo_i + 1, 0, N - 1)
        w = (f - lo_i)[:, None, None]
        return kp[lo_i] * (1 - w) + kp[hi_i] * w, (f >= 0) & (f < N - 1)
    rows = []
    for d in np.arange(lo, hi + step / 2.0, step):
        K2, ok0 = kp_at(d); ok = g & ok0
        if ok.sum() < min_ok: continue
        O = UV[ok][:, S] - K2[ok][:, H]
        dx = np.median(O[..., 0], axis=1); dy = np.median(O[..., 1], axis=1)
        R = O.copy(); R[..., 0] -= dx[:, None]; R[..., 1] -= dy[:, None]
        rows.append((d, np.median(np.hypot(dx, dy)), np.median(np.linalg.norm(R, axis=2)), ok.sum()))
    rows = np.array(rows)
    if len(rows) == 0: return None
    i = int(np.argmin(rows[:, 1]))
    z = int(np.argmin(np.abs(rows[:, 0])))
    return dict(label=label, n=int(g.sum()), N=N, lag=rows[i, 0], align=rows[i, 1], shape=rows[i, 2],
                align0=rows[z, 1], shape0=rows[z, 2], curves=rows)

def sessions(root=WS, dates=None):
    base = root + '/model_inputs/MotionPRO/adapter_v1/cam3'
    for date in sorted(os.listdir(base)):
        if dates and date not in dates: continue
        for subj in sorted(os.listdir(base + '/' + date)):
            for sess in sorted(os.listdir(f'{base}/{date}/{subj}')):
                yield date, subj, sess
