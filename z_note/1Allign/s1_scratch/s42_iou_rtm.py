"""S1 task2: YOLOX IoU lag with the RTMPose gate (the setting s16_iou_lag.py actually used),
   for all 140 sessions; also re-do the cloud-vs-box geometry diagnostic with correct
   bbox column order [frame_id, x1, y1, x2, y2, score, iou, cls]."""
import sys, json, numpy as np, time
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import WS, load_session
from s40_iou import box_iou, cloud_box

W, Hh = 1624.0, 1240.0

def curve(date, subj, sess, gate='rtm', lo=-30, hi=30, seg=None, min_ok=30):
    se = load_session(date, subj, sess)
    if se is None: return None
    N, UV, bb = se['N'], se['UV'], se['bb']
    cb = np.array([cloud_box(UV[i]) for i in range(N)])
    inside = ((cb[:, 0] > -40) & (cb[:, 1] > -40) & (cb[:, 2] < W + 40) & (cb[:, 3] < Hh + 40))
    g = se['gate'] & se['valid'] if gate == 'rtm' else (inside & se['gate'] & se['valid'] if gate == 'both' else inside)
    if seg is not None:
        m = np.zeros(N, bool); m[int(N * seg[0]):int(N * seg[1])] = True; g = g & m
    rr = []
    for L in range(lo, hi + 1):
        j = np.arange(N) + L
        ok = g & (j >= 0) & (j < N)
        if ok.sum() < min_ok: continue
        A = np.array([box_iou(cb[i], bb[j[i], 1:5]) for i in np.where(ok)[0]])
        rr.append((L, np.median(A), int(ok.sum())))
    return (np.array(rr), cb, bb, g) if rr else None

def peak(r):
    k = int(np.argmax(r[:, 1]))
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1, 1], r[k, 1], r[k + 1, 1]; den = y0 - 2 * y1 + y2
        sub = r[k, 0] + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
        sub = float(np.clip(sub, r[k, 0] - 1, r[k, 0] + 1))
    else: sub = float(r[k, 0])
    return float(r[k, 0]), sub, float(r[k, 1]), int(r[k, 2]), float(r[np.argmin(np.abs(r[:, 0]))][1])

t0 = time.time(); out = {}; geo = {}
for date, subj, sess in sorted([(d, s, ss) for d in sorted(__import__('os').listdir(WS + '/model_inputs/MotionPRO/adapter_v1/cam3'))
                                for s in sorted(__import__('os').listdir(f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{d}'))
                                for ss in sorted(__import__('os').listdir(f'{WS}/model_inputs/MotionPRO/adapter_v1/cam3/{d}/{s}'))]):
    lab = f'{date}/{subj}/{sess}'
    z = curve(date, subj, sess, gate='rtm')
    if z is None: print('SKIP', lab); continue
    r, cb, bb, g = z
    pi, sub, ioum, ng, at0 = peak(r)
    rec = dict(label=lab, lag_iou=pi, lag_iou_sub=sub, iou_max=ioum, iou_at0=at0, n_used=ng)
    z2 = curve(date, subj, sess, gate='both')
    if z2 is not None:
        _, s2, m2, n2, a2 = peak(z2[0]); rec.update(lag_iou_both=s2, iou_max_both=m2, iou_at0_both=a2, n_both=n2)
    for nm, sg in [('seg1', (0, .34)), ('seg2', (.33, .67)), ('seg3', (.66, 1.0))]:
        zz = curve(date, subj, sess, gate='rtm', seg=sg)
        if zz is not None:
            _, s3, _, n3, _ = peak(zz[0]); rec[nm + '_lag'] = s3; rec[nm + '_n'] = n3
    out[lab] = rec
    # geometry diagnostic, CORRECT column order
    boxH = bb[:, 4] - bb[:, 2]; cH = cb[:, 3] - cb[:, 1]
    geo[lab] = dict(gate_frac=float(g.mean()), boxH=float(np.median(boxH)), cloudH=float(np.median(cH)),
                    ratio=float(np.median(cH) / np.median(boxH)),
                    dcx=float(np.median((cb[:, 0] + cb[:, 2]) / 2 - (bb[:, 1] + bb[:, 3]) / 2)),
                    dcy=float(np.median((cb[:, 1] + cb[:, 3]) / 2 - (bb[:, 2] + bb[:, 4]) / 2)))
    print('%-22s gate=%-6s n=%3d  IoU lag %+6.2f (int %+4d, max %.3f, at0 %.3f)  boxH %6.1f cloudH %6.1f r %4.2f  dc(%+5.0f,%+5.0f) [%.0fs]' % (
        lab, 'rtm', ng, sub, pi, ioum, at0, geo[lab]['boxH'], geo[lab]['cloudH'], geo[lab]['ratio'],
        geo[lab]['dcx'], geo[lab]['dcy'], time.time() - t0))
json.dump(out, open('/tmp/s1-scratch/iou_fleet_rtm.json', 'w'), indent=1)
json.dump(geo, open('/tmp/s1-scratch/iou_geo.json', 'w'), indent=1)
print('\ndone %d in %.0fs' % (len(out), time.time() - t0))
