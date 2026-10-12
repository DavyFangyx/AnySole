"""S1: classify the fleet into A/B/C/D, write lag_fleet.csv, curve-quality diagnostics."""
import sys, json, csv, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import *

full = json.load(open('/tmp/s1-scratch/full_lag.json'))

print('=== align0 / align* distribution (140 sessions, wide range -30..30 step 0.5) ===')
a0 = np.array([r['align0'] for r in full]); a = np.array([r['align'] for r in full])
lag = np.array([r['lag'] for r in full]); n = np.array([r['n'] for r in full]); N = np.array([r['N'] for r in full])
for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    print('  p%-3d align0 %7.2f   align* %6.2f' % (q, np.percentile(a0, q), np.percentile(a, q)))
print()
print('align0 bins:  <=2 :%d  2-5:%d  5-8:%d  8-12:%d  12-15:%d  15-25:%d  25-50:%d  >50:%d' % (
    (a0 <= 2).sum(), ((a0 > 2) & (a0 <= 5)).sum(), ((a0 > 5) & (a0 <= 8)).sum(),
    ((a0 > 8) & (a0 <= 12)).sum(), ((a0 > 12) & (a0 <= 15)).sum(), ((a0 > 15) & (a0 <= 25)).sum(),
    ((a0 > 25) & (a0 <= 50)).sum(), (a0 > 50).sum()))
print('align* bins:  <=4 :%d  4-8:%d  8-12:%d  12-20:%d  >20:%d' % (
    (a <= 4).sum(), ((a > 4) & (a <= 8)).sum(), ((a > 8) & (a <= 12)).sum(),
    ((a > 12) & (a <= 20)).sum(), (a > 20).sum()))
print()
print('=== curve quality diagnostics ===')
diag = {}
for r in full:
    lab = r['label']
    date, subj, sess = lab.split('/')
    se = load_session(date, subj, sess)
    rr = sweep(se, lab, lo=-30, hi=30, step=0.5)
    c = rr['curves']  # d, align, shape, ok
    d, al = c[:, 0], c[:, 1]
    i = int(np.argmin(al))
    # in-boundary minima? is the argmin at the edge of the scanned window?
    edge = (i == 0) or (i == len(al) - 1)
    # number of local minima with align within 1.5 px of the global min (multi-modality)
    within = al < al[i] + 1.5
    # count contiguous runs of "within 1.5px of min"
    runs = int(np.sum(np.diff(np.concatenate([[0], within.astype(int), [0]])) == 1))
    # curvature / sharpness: rise of align 3 frames either side
    lo3 = al[max(0, i - 6)]; hi3 = al[min(len(al) - 1, i + 6)]
    sharp = max(lo3, hi3) - al[i]
    diag[lab] = dict(edge=edge, nruns=runs, sharp=float(sharp), nmin=int(within.sum()),
                     align=float(al[i]), lag=float(d[i]), n=int(c[i, 3]), align0=float(al[np.argmin(np.abs(d))]))
json.dump(diag, open('/tmp/s1-scratch/curve_diag.json', 'w'), indent=1)
nedge = sum(1 for v in diag.values() if v['edge'])
nflat = sum(1 for v in diag.values() if v['sharp'] < 2.0)
nmult = sum(1 for v in diag.values() if v['nruns'] > 1)
print('  argmin at scan edge (-30 or +30): %d  -> range still insufficient' % nedge)
print('  flat curve (align rises <2px within +-3 frames of min): %d' % nflat)
print('  multi-modal (>=2 disjoint sub-minimal runs within 1.5px): %d' % nmult)
print('  shallow+multimodal (ambiguous lag): %d' % sum(1 for v in diag.values() if v['sharp'] < 2.0 and v['nruns'] > 1))
print()
print('  flat/ambiguous sessions (sharp<2px):')
for k, v in sorted(diag.items(), key=lambda kv: kv[1]['sharp'])[:20]:
    print('    %-22s lag %+6.1f align* %5.2f sharp %5.2f nruns %d edge %s' % (k, v['lag'], v['align'], v['sharp'], v['nruns'], v['edge']))
