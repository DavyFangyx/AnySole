"""S1 task2c (proper): segment stability on the WELL-POSED sessions (|lag|>10 f, CI<=2 f, n>=120),
   so each third has enough gated frames. Reports per-third frame counts too."""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import load_session
from s60_segments import sweep_sub

rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))
S = np.array([0,1,2,4,5,7,8,10,11,15,12,16,17,18,19,20,21]); H = np.array([19,11,12,13,14,15,16,24,25,17,18,5,6,7,8,9,10])

def sweep_sub2(se, lo, hi, sub, min_ok=25):
    J, kp, valid, UV, N, gate = se['J'], se['kp'], se['valid'], se['UV'], se['N'], se['gate']
    g = np.zeros(N, bool); g[sub] = True; g = g & gate & valid
    def kp_at(delta):
        f = np.arange(N) + delta
        lo_i = np.clip(np.floor(f).astype(int), 0, N - 1); hi_i = np.clip(lo_i + 1, 0, N - 1)
        w = (f - lo_i)[:, None, None]
        return kp[lo_i] * (1 - w) + kp[hi_i] * w, (f >= 0) & (f < N - 1)
    rows = []
    for d in np.arange(lo, hi + 0.25, 0.5):
        K2, ok0 = kp_at(d); ok = g & ok0
        if ok.sum() < min_ok: continue
        O = UV[ok][:, S] - K2[ok][:, H]
        dx = np.median(O[..., 0], axis=1); dy = np.median(O[..., 1], axis=1)
        rows.append((d, np.median(np.hypot(dx, dy)), int(ok.sum())))
    rows = np.array(rows)
    if len(rows) == 0: return None
    k = int(np.argmin(rows[:, 1]))
    return float(rows[k, 0]), float(rows[k, 1]), int(rows[k, 2])

pool = [l for l, r in rt.items() if abs(r['lag_eff']) > 10 and r['ci_half'] <= 2.0 and r['n'] >= 120]
pool.sort(key=lambda l: rt[l]['lag_eff'])
print('well-posed |lag|>10 sessions (CI<=2f, n>=120): %d' % len(pool))
pick = pool[:2] + pool[len(pool) // 2 - 1:len(pool) // 2 + 1] + pool[-1:]
print('picked 5:', pick)
print()
print('%-22s %6s %6s %7s %s' % ('session', 'lag*', 'CI', 'align0', 'per-third RTMPose lag (frames) + n_gated'))
rs, isp = [], []
for lab in pick:
    date, subj, sess = lab.split('/')
    se = load_session(date, subj, sess)
    N = se['N']
    th = [(0, N // 3), (N // 3, 2 * N // 3), (2 * N // 3, N)]
    a, an = [], []
    for i, j in th:
        sub = np.zeros(N, bool); sub[i:j] = True
        z = sweep_sub2(se, -30, 30, sub)
        a.append(z[0] if z else np.nan); an.append(z[2] if z else 0)
    b = [io[lab].get('seg%d_lag' % k, np.nan) for k in (1, 2, 3)]
    bn = [io[lab].get('seg%d_n' % k, 0) for k in (1, 2, 3)]
    r = rt[lab]
    span_r = np.nanmax(a) - np.nanmin(a)
    span_i = np.nanmax(b) - np.nanmin(b)
    if not np.isnan(span_r): rs.append(span_r)
    if not np.isnan(span_i): isp.append(span_i)
    print('%-22s %+6.1f %+6.1f %7.2f  RTM %+6.1f %+6.1f %+6.1f (span %4.1f, n %d/%d/%d) | IoU %+6.1f %+6.1f %+6.1f (span %4.1f)' % (
        lab, r['lag_eff'], r['ci_half'], r['align0'], a[0], a[1], a[2], span_r, an[0], an[1], an[2], b[0], b[1], b[2], span_i))
print()
rs = np.array(rs); isp = np.array(isp)
print('RTMPose per-third span: median %.2f f  min %.2f  max %.2f  ; <=2f %d/%d ; <=3f %d/%d' % (
    np.median(rs), rs.min(), rs.max(), (rs <= 2).sum(), len(rs), (rs <= 3).sum(), len(rs)))
print('YOLOX   per-third span: median %.2f f  min %.2f  max %.2f  ; <=2f %d/%d' % (
    np.median(isp), isp.min(), isp.max(), (isp <= 2).sum(), len(isp)))
print()
print('=== monotonic trend check: is the drift monotone across the three thirds? ===')
for lab in pick:
    date, subj, sess = lab.split('/'); se = load_session(date, subj, sess); N = se['N']
    th = [(0, N // 3), (N // 3, 2 * N // 3), (2 * N // 3, N)]
    a = []
    for i, j in th:
        m = np.zeros(N, bool); m[i:j] = True
        z = sweep_sub2(se, -30, 30, m); a.append(z[0] if z else np.nan)
    mono = 'n/a'
    if not any(np.isnan(a)): mono = 'monotone' if (a[0] <= a[1] <= a[2] or a[0] >= a[1] >= a[2]) else 'non-monotone'
    print('  %-22s %+6.1f %+6.1f %+6.1f  %s' % (lab, a[0], a[1], a[2], mono))
json.dump({'pick': pick, 'rtm_spans': rs.tolist(), 'iou_spans': isp.tolist()}, open('/tmp/s1-scratch/segments2.json', 'w'), indent=1)
