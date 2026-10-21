"""S1 task2c: segment stability, both estimators, for sessions with |lag|>10 f.
   RTMPose reprojection restricted to frame thirds + YOLOX IoU restricted to the same thirds."""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import WS, load_session, sweep

rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))
cands = sorted([l for l, r in rt.items() if abs(r['lag_eff']) > 10])
print('sessions with |lag_eff|>10 f: %d' % len(cands))

def sweep_sub(se, lo, hi, sub):
    """ver batim sweep() but restricted to the frame subset 'sub'."""
    J, kp, valid, UV, N, gate = se['J'], se['kp'], se['valid'], se['UV'], se['N'], se['gate']
    g = np.zeros(N, bool); g[sub] = True
    g = g & gate & valid
    def kp_at(delta):
        f = np.arange(N) + delta
        lo_i = np.clip(np.floor(f).astype(int), 0, N - 1); hi_i = np.clip(lo_i + 1, 0, N - 1)
        w = (f - lo_i)[:, None, None]
        return kp[lo_i] * (1 - w) + kp[hi_i] * w, (f >= 0) & (f < N - 1)
    S = np.array([0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]); H = np.array([19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10])
    rows = []
    for d in np.arange(lo, hi + 0.25, 0.5):
        K2, ok0 = kp_at(d); ok = g & ok0
        if ok.sum() < 20: continue
        O = UV[ok][:, S] - K2[ok][:, H]
        dx = np.median(O[..., 0], axis=1); dy = np.median(O[..., 1], axis=1)
        rows.append((d, np.median(np.hypot(dx, dy)), ok.sum()))
    rows = np.array(rows)
    if len(rows) == 0: return None
    k = int(np.argmin(rows[:, 1]))
    return float(rows[k, 0]), float(rows[k, 1]), int(rows[k, 2])

# take 5 spread across the |lag|>10 population (largest few + a couple mid / opposite-sign)
pick = []
pos = [l for l in cands if rt[l]['lag_eff'] > 10]
neg = [l for l in cands if rt[l]['lag_eff'] < -10]
ser = sorted(neg, key=lambda l: rt[l]['lag_eff'])
pick += ser[:2] + ser[len(ser) // 2:len(ser) // 2 + 2] + pos[:1]
print('picked %d: %s' % (len(pick), pick))
print()
print('%-22s | %-28s | %-28s' % ('session', 'RTMPose reprojection (0-33/33-67/67-100%)', 'YOLOX IoU (same thirds)'))
rtspans, iospans = [], []
for lab in pick:
    date, subj, sess = lab.split('/')
    se = load_session(date, subj, sess)
    N = se['N']
    thirds = [(0, N // 3), (N // 3, 2 * N // 3), (2 * N // 3, N)]
    a = []
    for i, j in thirds:
        sub = np.zeros(N, bool); sub[i:j] = True
        z = sweep_sub(se, -30, 30, sub)
        a.append(z[0] if z else np.nan)
    b = [io[lab].get('seg%d_lag' % k, np.nan) for k in (1, 2, 3)]
    rtspans.append(np.nanmax(a) - np.nanmin(a)); iospans.append(np.nanmax(b) - np.nanmin(b))
    print('%-22s | full %+6.1f  segs %+6.1f %+6.1f %+6.1f  span %5.1f | segs %+6.1f %+6.1f %+6.1f  span %5.1f' % (
        lab, rt[lab]['lag_eff'], a[0], a[1], a[2], np.nanmax(a) - np.nanmin(a), b[0], b[1], b[2], np.nanmax(b) - np.nanmin(b)))
rs = np.array(rtspans); isp = np.array(iospans)
print()
print('RTMPose segment spans: median %.2f f  max %.2f  <=2f: %d/%d' % (np.median(rs), rs.max(), (rs <= 2).sum(), len(rs)))
print('YOLOX   segment spans: median %.2f f  max %.2f  <=2f: %d/%d' % (np.median(isp), isp.max(), (isp <= 2).sum(), len(isp)))
json.dump({'pick': pick, 'rtm_spans': rtspans, 'iou_spans': iospans}, open('/tmp/s1-scratch/segments.json', 'w'), indent=1)
