"""S1 task1: definitive fleet lag table + confidence classes + CSV.
Class definitions (fixed BEFORE looking at outcomes, thresholds taken from 15_R2 doc):
  A 已对齐      : align0 <= 8 px            (at lag 0 already at the ~4.5px static floor; no shift needed)
  B 可平移修复  : align0 > 8 px  AND align* < 12 px   (a per-session scalar shift reaches the floor)
  C 修不动      : align0 > 8 px  AND align* >= 12 px  (shift does not reach the floor)
  D 证据不足    : gate n < 100 OR lag CI half-width > 4.0 frames (curve cannot constrain the shift)
Lag CI: the set {d : align(d) <= align* + 1.0 px}; half-width reported.
"""
import sys, json, csv, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import *

full = json.load(open('/tmp/s1-scratch/full_lag.json'))
rows = []
for r in full:
    date, subj, sess = r['label'].split('/')
    se = load_session(date, subj, sess)
    rr = sweep(se, r['label'], lo=-30, hi=30, step=0.5)
    c = rr['curves']; d, al, sh, okc = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    i = int(np.argmin(al))
    band = al <= al[i] + 1.0
    lo, hi = d[band].min(), d[band].max()
    # sub-frame parabolic refinement about the discrete min
    if 0 < i < len(al) - 1:
        y0, y1, y2 = al[i - 1], al[i], al[i + 1]
        den = (y0 - 2 * y1 + y2)
        sub = d[i] + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
        sub = float(np.clip(sub, d[i] - 0.5, d[i] + 0.5))
    else:
        sub = float(d[i])
    lag_sub = float(np.clip(round(sub * 2) / 2.0, -30, 30))
    rows.append(dict(label=r['label'], date=date, subj=subj, sess=sess,
                     lag_pt=float(d[i]), lag_sub=lag_sub, lag_ms=lag_sub * 25.0,
                     ci_lo=float(lo), ci_hi=float(hi), ci_half=float((hi - lo) / 2.0),
                     align0=float(r['align0']), align=float(al[i]),
                     gain=float(r['align0'] - al[i]), shape0=float(r['shape0']), shape=float(sh[i]),
                     n=int(r['n']), N=int(r['N']), frac=float(r['n'] / r['N']), ok_curve=int(okc[i])))

def classify(x):
    if x['n'] < 100 or x['ci_half'] > 4.0: return 'D'
    if x['align0'] <= 8.0: return 'A'
    return 'B' if x['align'] < 12.0 else 'C'

for x in rows:
    x['cls'] = classify(x)
    x['lag_eff'] = 0.0 if x['cls'] == 'A' else x['lag_sub']   # A-class needs no shift
    x['lag_eff_ms'] = x['lag_eff'] * 25.0
json.dump(rows, open('/tmp/s1-scratch/lag_fleet.json', 'w'), indent=1)

with open('/tmp/s1-scratch/lag_fleet.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['session', 'lag_frames', 'lag_ms', 'lag_eff_frames', 'lag_eff_ms', 'align0_px', 'align_star_px',
                'gain_px', 'n_gate', 'N', 'gate_frac', 'ci_lo_f', 'ci_hi_f', 'ci_half_f', 'conf_class'])
    for x in sorted(rows, key=lambda r: (r['cls'], -abs(r['lag_eff']))):
        w.writerow([x['label'], '%.1f' % x['lag_sub'], '%+.0f' % x['lag_ms'], '%.1f' % x['lag_eff'],
                    '%+.0f' % x['lag_eff_ms'], '%.2f' % x['align0'], '%.2f' % x['align'], '%.2f' % x['gain'],
                    x['n'], x['N'], '%.3f' % x['frac'], '%.1f' % x['ci_lo'], '%.1f' % x['ci_hi'],
                    '%.1f' % x['ci_half'], x['cls']])

print('=== FINAL FLEET TABLE (140 sessions, wide range -30..+30 f, 0.5 f step) ===')
import collections
cnt = collections.Counter(x['cls'] for x in rows)
print('class counts: ', dict(sorted(cnt.items())),
      '  A+B (fixable/fine) = %d (%.1f%%)' % (cnt['A'] + cnt['B'], 100.0 * (cnt['A'] + cnt['B']) / len(rows)))
for cl in 'ABCD':
    sub = [x for x in rows if x['cls'] == cl]
    if not sub: continue
    print('\n--- class %s (n=%d) ---' % (cl, len(sub)))
    for x in sorted(sub, key=lambda r: -abs(r['lag_eff'])):
        print('  %-22s lag %+6.1f f (%+6.0f ms)  align0 %6.1f -> %5.2f (gain %5.1f)  n=%3d/%3d (%.0f%%) CI +-%.1f f' % (
            x['label'], x['lag_eff'], x['lag_eff_ms'], x['align0'], x['align'], x['gain'],
            x['n'], x['N'], 100 * x['frac'], x['ci_half']))

lags = np.array([x['lag_eff'] for x in rows])
print('\n=== effective correction lag (A set to 0) ===')
print('n=%d mean %+.2f median %+.2f  min %+.1f max %+.1f   |lag|>10f: %d   |lag|>0.5f: %d' % (
    len(lags), lags.mean(), np.median(lags), lags.min(), lags.max(),
    (np.abs(lags) > 10).sum(), (np.abs(lags) > 0.5).sum()))
print('positive (image needs EARLIER mocap i.e. GT is ahead): %d ; negative: %d ; zero: %d' % (
    (lags > 0.5).sum(), (lags < -0.5).sum(), (np.abs(lags) <= 0.5).sum()))
for q in [0, 5, 25, 50, 75, 95, 100]:
    print('  p%-3d %+7.2f f  (%+6.0f ms)' % (q, np.percentile(lags, q), np.percentile(lags, q) * 25))
