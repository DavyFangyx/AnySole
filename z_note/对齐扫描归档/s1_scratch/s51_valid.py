"""S1: IoU-estimate validity gating + honest pass rates + why the invalid ones fail."""
import json, numpy as np, sys
sys.path.insert(0, '/tmp/s1-scratch')
from sweep_lib import WS, load_session
from s40_iou import cloud_box
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet.json'))

print('=== why the IoU method returns lag at +-30 (degenerate): the projected cloud vs the boxes ===')
W, Hh = 1624.0, 1240.0
degenerate = []
diag = {}
for lab in sorted(io):
    date, subj, sess = lab.split('/')
    se = load_session(date, subj, sess)
    N, UV, bb = se['N'], se['UV'], se['bb']
    cb = np.array([cloud_box(UV[i]) for i in range(N)])
    inside = ((cb[:, 0] > -40) & (cb[:, 1] > -40) & (cb[:, 2] < W + 40) & (cb[:, 3] < Hh + 40))
    # per-frame IoU at lag 0 to see whether the cloud ever lands on a box
    a = np.array([(max(0, min(cb[i, 2], bb[i, 2]) - max(cb[i, 0], bb[i, 1])) *
                   max(0, min(cb[i, 3], bb[i, 3]) - max(cb[i, 1], bb[i, 2]))) for i in range(N)])
    inter = np.array([max(0.0, min(cb[i, 2], bb[i, 2]) - max(cb[i, 0], bb[i, 1])) *
                      max(0.0, min(cb[i, 3], bb[i, 3]) - max(cb[i, 1], bb[i, 2])) for i in range(N)])
    ua = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1]) + (bb[:, 4] - bb[:, 3]) * (bb[:, 2] - bb[:, 1]) - inter
    iou0 = inter / np.maximum(ua, 1e-9)
    diag[lab] = dict(inside_frac=float(inside.mean()), iou0_med=float(np.median(iou0)),
                     iou0_p90=float(np.percentile(iou0, 90)), cloudH=float(np.median(cb[:, 3] - cb[:, 1])),
                     boxH=float(np.median(bb[:, 4] - bb[:, 3])),
                     cx=float(np.median((cb[:, 0] + cb[:, 2]) / 2)), cy=float(np.median((cb[:, 1] + cb[:, 3]) / 2)),
                     bx=float(np.median((bb[:, 1] + bb[:, 3]) / 2)), by=float(np.median((bb[:, 2] + bb[:, 4]) / 2)))
json.dump(diag, open('/tmp/s1-scratch/iou_diag.json', 'w'), indent=1)

print('  IoUmax<0.10 (no overlap signal at any lag): %d sessions' % sum(1 for l in io if io[l]['iou_max'] < 0.10))
print('  IoUmax<0.30: %d ; 0.30-0.50: %d ; >=0.50: %d' % (
    sum(1 for l in io if io[l]['iou_max'] < 0.30), sum(1 for l in io if 0.30 <= io[l]['iou_max'] < 0.50),
    sum(1 for l in io if io[l]['iou_max'] >= 0.50)))
bad = sorted([l for l in io if io[l]['iou_max'] < 0.30], key=lambda l: io[l]['iou_max'])
print('\n  degenerate sessions (IoUmax<0.30), n=%d:' % len(bad))
for l in bad:
    d = diag[l]
    print('    %-22s IoUmax %.3f  inside %.2f  cloudH %6.1f boxH %6.1f (ratio %4.2f)  cloud_ctr (%6.0f,%6.0f) box_ctr (%6.0f,%6.0f)  off (%5.0f,%5.0f)' % (
        l, io[l]['iou_max'], d['inside_frac'], d['cloudH'], d['boxH'], d['cloudH'] / max(d['boxH'], 1e-9),
        d['cx'], d['cy'], d['bx'], d['by'], d['cx'] - d['bx'], d['cy'] - d['by']))

print('\n=== honest pass rate: only IoU estimates with a genuine signal ===')
def iou_valid(l): return io[l]['iou_max'] >= 0.50 and io[l]['n_used'] >= 100
d = []
for lab, r in rt.items():
    d.append((lab, r['lag_eff'], io[lab]['lag_iou_sub'], r['cls'], r['ci_half'], io[lab]['iou_max'],
              io[lab]['iou_at0'], r['align0'], r['align']))
allv = [x for x in d if iou_valid(x[0])]
print('sessions with a usable IoU estimate: %d/140' % len(allv))
for thr, nm in [(2, '|Δ|<=2f'), (3, '|Δ|<=3f')]:
    k = sum(1 for x in allv if abs(x[1] - x[2]) <= thr)
    print('  ALL usable: %s %d/%d (%.0f%%)  median|Δ| %.2f' % (nm, k, len(allv), 100 * k / len(allv),
                                                               np.median([abs(x[1] - x[2]) for x in allv])))
conf = [x for x in allv if x[3] in 'BC' and x[4] <= 2.0]
k = sum(1 for x in conf if abs(x[1] - x[2]) <= 2)
print('  B/C class AND CI<=2f AND usable IoU: |Δ|<=2f %d/%d (%.0f%%)  median|Δ| %.2f' % (
    k, len(conf), 100 * k / max(1, len(conf)), np.median([abs(x[1] - x[2]) for x in conf]) if conf else np.nan))
for x in sorted(conf, key=lambda y: -abs(y[1] - y[2]))[:10]:
    print('      %-22s RTM %+6.1f IoU %+6.2f Δ %+6.2f  (IoUmax %.3f at0 %.3f)' % (x[0], x[1], x[2], x[2] - x[1], x[5], x[6]))
