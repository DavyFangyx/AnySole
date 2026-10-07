"""S1 task2: independent cross-check of the visual<->mocap lag using the YOLOX detector
   boxes (bbox.npy) instead of the RTMPose frontend.  Two independent statistics:
     (1) IoU( projected joint-hull box , YOLOX person box ) maximised over lag
     (2) normalised cross-correlation of high-passed cloud height vs YOLOX box height
   Both use only bbox.npy + keypoints.npy + calibration: the frontend keypoints never enter,
   except for the optional 'rtm' gate, which is reported separately so the two methods can be
   compared on identical frame subsets.
"""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import WS, load_session, project

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def cloud_box(P, pad=0.0):
    x1, y1 = P[:, 0].min(), P[:, 1].min(); x2, y2 = P[:, 0].max(), P[:, 1].max()
    h = (y2 - y1) * pad
    return [x1 - h, y1 - h, x2 + h, y2 + h]

def hp(s, w=21):
    k = np.ones(w) / w
    return s - np.convolve(s, k, 'same') / np.convolve(np.ones_like(s), k, 'same')

def iou_lag(date, subj, sess, lo=-30, hi=30, pad=0.0, gate='free', root=WS):
    se = load_session(date, subj, sess, root=root)
    if se is None: return None
    N, UV, bb = se['N'], se['UV'], se['bb']
    if bb is None: return None
    W = 1624.0; Hh = 1240.0
    cb = np.array([cloud_box(UV[i], pad) for i in range(N)])
    inside = ((cb[:, 0] > -40) & (cb[:, 1] > -40) & (cb[:, 2] < W + 40) & (cb[:, 3] < Hh + 40))
    if gate == 'free':
        g = inside
    elif gate == 'rtm':
        g = se['gate'] & se['valid']
    elif gate == 'both':
        g = inside & se['gate'] & se['valid']
    ch = cb[:, 3] - cb[:, 1]
    bh = bb[:, 4] - bb[:, 3]
    rows = []
    for L in range(lo, hi + 1):
        j = np.arange(N) + L
        ok = g & (j >= 0) & (j < N)
        if ok.sum() < 30: continue
        A = np.array([box_iou(cb[i], bb[j[i], 1:5]) for i in np.where(ok)[0]])
        rows.append((L, np.median(A), ok.sum()))
    if not rows: return None
    r = np.array(rows)
    k = int(np.argmax(r[:, 1]))
    y0, y1, y2 = (r[k - 1, 1], r[k, 1], r[k + 1, 1]) if 0 < k < len(r) - 1 else (r[k, 1], r[k, 1], r[k, 1])
    den = (y0 - 2 * y1 + y2)
    sub = r[k, 0] + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    sub = float(np.clip(sub, r[k, 0] - 1, r[k, 0] + 1))
    # height correlation peak
    a = hp(bh); c = hp(ch)
    cc = []
    for L in range(lo, hi + 1):
        j = np.arange(N) + L
        ok = g & (j >= 0) & (j < N)
        if ok.sum() < 30: continue
        cc.append((L, np.corrcoef(a[ok], c[j[ok]])[0, 1], ok.sum()))
    cc = np.array(cc)
    kk = int(np.argmax(cc[:, 1])) if len(cc) else 0
    argmax_iou = int(r[k, 0]); argmax_cc = int(cc[kk, 0]) if len(cc) else None
    return dict(label=f'{date}/{subj}/{sess}', n_gate=int(g.sum()), N=N, gate=gate,
                lag_iou=float(r[k, 0]), lag_iou_sub=sub, iou_max=float(r[k, 1]),
                iou_at0=float(r[np.argmin(np.abs(r[:, 0]))][1]),
                lag_cc=float(argmax_cc) if argmax_cc is not None else None,
                cc_max=float(cc[kk, 1]) if len(cc) else None,
                cc_at0=float(cc[np.argmin(np.abs(cc[:, 0]))][1]) if len(cc) else None)

if __name__ == '__main__':
    print('=== self-test on the R2 anchor S5091 (RTMPose reprojection gave lag +9.5) ===')
    for g in ['free', 'rtm', 'both']:
        r = iou_lag('20260804', 'S5', 'S5091', gate=g)
        print('  gate=%-5s n=%3d  IoU argmax lag %+6.1f (sub %+.2f, IoU %.3f, at0 %.3f)   CC argmax lag %+5.0f (r %.3f, at0 %.3f)' % (
            g, r['n_gate'], r['lag_iou'], r['lag_iou_sub'], r['iou_max'], r['iou_at0'], r['lag_cc'], r['cc_max'], r['cc_at0']))
