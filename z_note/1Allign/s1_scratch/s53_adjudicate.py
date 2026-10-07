"""S1: adjudicate every RTMPose-vs-IoU disagreement with the residual curve itself.
   If align(IoU_lag) is not better than align(lag0), the IoU peak is spurious for pairing
   purposes (it fits box size/position wobble rather than the pose)."""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import load_session, sweep

cmp_ = json.load(open('/tmp/s1-scratch/final_cmp.json'))
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}

print('%-22s %-4s %7s %7s %7s | %8s %8s %8s | verdict' % ('session', 'cls', 'RTM*', 'IoU', 'Δ', 'al@0', 'al@RTM*', 'al@IoU'))
adj = {'iou_spurious': 0, 'rtm_confirmed': 0, 'neither': 0, 'iou_better': 0}
out = []
for x in sorted(cmp_, key=lambda y: (y['cls'], -abs(y['d']))):
    if abs(x['d']) <= 2.0: continue
    if not (x['ioumax'] >= 0.50 and x['prom'] >= 0.02 and x['n_iou'] >= 60): continue
    lab = x['label']; date, subj, sess = lab.split('/')
    se = load_session(date, subj, sess)
    r = sweep(se, lab, lo=-30, hi=30, step=0.5)
    c = r['curves']; d, al = c[:, 0], c[:, 1]
    def at(v): return float(al[np.argmin(np.abs(d - v))])
    a0, aR, aI = at(0.0), at(x['rtm']), at(x['iou'])
    if aI >= a0 - 0.5 and aR < aI - 1.0:
        v = 'IoU peak spurious (does not reduce the pose residual)'; adj['iou_spurious'] += 1
    elif aR < aI - 1.0 and aI < a0 - 1.0:
        v = 'both reduce residual; lag ambiguous'; adj['rtm_confirmed'] += 1
    elif aI < aR - 1.0:
        v = 'IoU lag better than RTM'; adj['iou_better'] += 1
    else:
        v = 'neither clearly better'; adj['neither'] += 1
    out.append(dict(label=lab, cls=x['cls'], rtm=x['rtm'], iou=x['iou'], a0=a0, aR=aR, aI=aI, verdict=v, prom=x['prom']))
    print('%-22s %-4s %+7.1f %+7.2f %+7.2f | %8.2f %8.2f %8.2f | %s' % (
        lab, x['cls'], x['rtm'], x['iou'], x['d'], a0, aR, aI, v))
print('\n adjudication of the %d discriminating disagreements: %s' % (len(out), adj))
print(' => IoU peak fails to reduce the pose residual in %d/%d (%.0f%%) of them' % (
    adj['iou_spurious'], len(out), 100 * adj['iou_spurious'] / max(1, len(out))))
json.dump(out, open('/tmp/s1-scratch/adjudication.json', 'w'), indent=1)
print()
print('=== per-class, IoU peak examined ===')
for cl in 'ABCD':
    s = [o for o in out if o['cls'] == cl]
    if s: print('  class %s: n=%d  IoU spurious %d (%.0f%%)' % (cl, len(s), sum(1 for o in s if o['verdict'].startswith('IoU peak spurious')), 100*sum(1 for o in s if o['verdict'].startswith('IoU peak spurious'))/len(s)))
