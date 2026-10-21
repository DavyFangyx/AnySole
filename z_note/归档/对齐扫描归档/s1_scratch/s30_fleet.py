"""S1 task1: definitive full-fleet sweep, wide lag range, 0.5-frame step.
   Also replicates the narrow (-4..20) and wide (-30..30 step 1.0) old settings
   so json-vs-rerun agreement can be checked per session."""
import sys, json, numpy as np, time
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import *

t0 = time.time()
full, narrow, wideint = [], [], []
for date, subj, sess in sessions():
    lab = f'{date}/{subj}/{sess}'
    se = load_session(date, subj, sess)
    if se is None:
        print('SKIP', lab); continue
    rf = sweep(se, lab, lo=-30, hi=30, step=0.5)
    rn = sweep(se, lab, lo=-4, hi=20, step=0.5)
    rw = sweep(se, lab, lo=-30, hi=30, step=1.0)
    for r in (rf, rn, rw): r.pop('curves')
    full.append(rf); narrow.append(rn); wideint.append(rw)
    print('%-22s N=%4d gate=%4d | lag*=%+6.1f f (%+5.0f ms) align0 %6.1f -> align* %5.1f  shape0 %5.1f -> %5.1f (%.0fs)' % (
        lab, rf['N'], rf['n'], rf['lag'], rf['lag'] * 25, rf['align0'], rf['align'], rf['shape0'], rf['shape'], time.time() - t0))

json.dump(full, open('/tmp/s1-scratch/full_lag.json', 'w'), indent=1)
json.dump(narrow, open('/tmp/s1-scratch/narrow_lag.json', 'w'), indent=1)
json.dump(wideint, open('/tmp/s1-scratch/wideint_lag.json', 'w'), indent=1)
print('\n=== done %d sessions in %.0fs ===' % (len(full), time.time() - t0))
lags = np.array([r['lag'] for r in full])
print('lag* mean %+.2f median %+.2f min %+.2f max %+.2f  |  |lag|>10: %d' % (
    lags.mean(), np.median(lags), lags.min(), lags.max(), (np.abs(lags) > 10).sum()))
