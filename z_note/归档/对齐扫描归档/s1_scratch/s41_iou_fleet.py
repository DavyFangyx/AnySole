"""S1 task2: YOLOX IoU lag for all 140 sessions (free gate = independent of RTMPose),
   plus per-segment (3 thirds) lag for the segment-stability check."""
import sys, json, numpy as np, time
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import WS, load_session
from s40_iou import box_iou, cloud_box, iou_lag

W = 1624.0; Hh = 1240.0

def iou_curve(date, subj, sess, lo=-30, hi=30, seg=None, root=WS):
    se = load_session(date, subj, sess, root=root)
    if se is None: return None
    N, UV, bb = se['N'], se['UV'], se['bb']
    cb = np.array([cloud_box(UV[i]) for i in range(N)])
    inside = ((cb[:, 0] > -40) & (cb[:, 1] > -40) & (cb[:, 2] < W + 40) & (cb[:, 3] < Hh + 40))
    g = inside
    if seg is not None:
        a, b = int(N * seg[0]), int(N * seg[1])
        m = np.zeros(N, bool); m[a:b] = True; g = g & m
    rows = []
    for L in range(lo, hi + 1):
        j = np.arange(N) + L
        ok = g & (j >= 0) & (j < N)
        if ok.sum() < 30: continue
        A = np.array([box_iou(cb[i], bb[j[i], 1:5]) for i in np.where(ok)[0]])
        rows.append((L, np.median(A), int(ok.sum())))
    if not rows: return None
    return np.array(rows)

def peak(r):
    k = int(np.argmax(r[:, 1]))
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1, 1], r[k, 1], r[k + 1, 1]
        den = (y0 - 2 * y1 + y2)
        sub = r[k, 0] + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
        sub = float(np.clip(sub, r[k, 0] - 1, r[k, 0] + 1))
    else:
        sub = float(r[k, 0])
    return float(r[k, 0]), sub, float(r[k, 1]), int(r[k, 2]), float(r[np.argmin(np.abs(r[:, 0]))][1])

t0 = time.time()
out = {}
for date, subj, sess in [(x.split('/')[0], x.split('/')[1], x.split('/')[2]) for x in sorted(
        [f'{d}/{s}/{ss}' for d in sorted(__import__('os').listdir(WS + '/model_inputs/MotionPRO/adapter_v1/cam3'))
         for s in sorted(__import__('os').listdir(f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{d}'))
         for ss in sorted(__import__('os').listdir(f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{d}/{s}'))])]:
    lab = f'{date}/{subj}/{sess}'
    r = iou_curve(date, subj, sess)
    if r is None:
        print('SKIP', lab); continue
    pi, sub, ioum, ng, at0 = peak(r)
    rec = dict(label=lab, lag_iou=pi, lag_iou_sub=sub, iou_max=ioum, iou_at0=at0, n_used=ng, N=int(r[0, 2]))
    for si, (nm, seg) in enumerate([('seg1', (0, .34)), ('seg2', (.33, .67)), ('seg3', (.66, 1.0))]):
        rs = iou_curve(date, subj, sess, seg=seg)
        if rs is not None:
            _, s2, _, n2, a2 = peak(rs)
            rec[nm + '_lag'] = s2; rec[nm + '_n'] = n2
    out[lab] = rec
    print('%-22s IoU lag %+6.2f (int %+4d, IoU %.3f, at0 %.3f, n=%3d)  | seg %+.1f/%+.1f/%+.1f (max-min %.1f)  [%.0fs]' % (
        lab, sub, pi, ioum, at0, ng, rec.get('seg1_lag', np.nan), rec.get('seg2_lag', np.nan), rec.get('seg3_lag', np.nan),
        np.nanmax([rec.get(k + '_lag', np.nan) for k in ('seg1', 'seg2', 'seg3')]) -
        np.nanmin([rec.get(k + '_lag', np.nan) for k in ('seg1', 'seg2', 'seg3')]), time.time() - t0))
json.dump(out, open('/tmp/s1-scratch/iou_fleet.json', 'w'), indent=1)
print('\ndone %d sessions in %.0fs' % (len(out), time.time() - t0))
