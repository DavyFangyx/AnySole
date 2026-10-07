"""S1 task2: RTMPose-reprojection lag  vs  YOLOX IoU lag  (independent 2D source)."""
import json, numpy as np, collections
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet.json'))
rows = list(rt.values())

print('=== A) fleet-wide agreement: RTMPose lag_eff vs YOLOX IoU lag ===')
d = []
for r in rows:
    lab = r['label']
    if lab not in io: continue
    d.append((lab, r['lag_eff'], io[lab]['lag_iou_sub'], r['cls'], r['align0'], r['align'], r['ci_half'],
              io[lab]['iou_max'], io[lab]['iou_at0'], io[lab]['n_used']))
d.sort(key=lambda x: -abs(x[1] - x[2]))
agree = [x for x in d if abs(x[1] - x[2]) <= 2.0]
print('n=%d with both estimates' % len(d))
print('  |Δ| <= 2 f : %d (%.1f%%)   |Δ| <= 3 f : %d (%.1f%%)   |Δ| <= 4 f : %d (%.1f%%)' % (
    len(agree), 100 * len(agree) / len(d),
    sum(1 for x in d if abs(x[1] - x[2]) <= 3), 100 * sum(1 for x in d if abs(x[1] - x[2]) <= 3) / len(d),
    sum(1 for x in d if abs(x[1] - x[2]) <= 4), 100 * sum(1 for x in d if abs(x[1] - x[2]) <= 4) / len(d)))
print('  median |Δ| %.2f f   mean |Δ| %.2f f' % (np.median([abs(x[1] - x[2]) for x in d]), np.mean([abs(x[1] - x[2]) for x in d])))
print('  sign agreement (both same sign or both ~0): %d/%d' % (
    sum(1 for x in d if (x[1] > 0.5 and x[2] > 0.5) or (x[1] < -0.5 and x[2] < -0.5) or (abs(x[1]) <= 0.5 and abs(x[2]) <= 2)),
    len(d)))
print('\n  worst 25 disagreements:')
for x in d[:25]:
    print('    %-22s RTM %+6.1f  IoU %+6.2f  Δ %+6.2f | cls %s align0 %6.1f align* %5.2f CI+-%.1f IoUmax %.3f at0 %.3f n=%d' % (
        x[0], x[1], x[2], x[2] - x[1], x[3], x[4], x[5], x[6], x[7], x[8], x[9]))

print('\n=== B) agreement restricted to CONFIDENT reprojection cases (B/C class, CI<=2f) ===')
conf = [x for x in d if x[3] in 'BC' and x[6] <= 2.0]
print('n=%d ; |Δ|<=2f: %d (%.1f%%) ; median |Δ| %.2f' % (
    len(conf), sum(1 for x in conf if abs(x[1] - x[2]) <= 2), 100 * sum(1 for x in conf if abs(x[1] - x[2]) <= 2) / max(1, len(conf)),
    np.median([abs(x[1] - x[2]) for x in conf]) if conf else np.nan))
for x in sorted(conf, key=lambda y: -abs(y[1] - y[2]))[:15]:
    print('    %-22s RTM %+6.1f  IoU %+6.2f  Δ %+6.2f | align0 %6.1f align* %5.2f' % (x[0], x[1], x[2], x[2] - x[1], x[4], x[5]))

print('\n=== C) agreement by class ===')
for cl in 'ABCD':
    s = [x for x in d if x[3] == cl]
    if not s: continue
    ok = sum(1 for x in s if abs(x[1] - x[2]) <= 2)
    print('  class %s n=%3d  |Δ|<=2f %3d (%.0f%%)  median |Δ| %.2f' % (
        cl, len(s), ok, 100 * ok / len(s), np.median([abs(x[1] - x[2]) for x in s])))

print('\n=== D) IoU prominence: does the IoU curve actually discriminate? ===')
prom = np.array([x[7] - x[8] for x in d])
print('  IoU at peak minus IoU at lag0: median %.3f  >0.02: %d  <=0.01: %d' % (
    np.median(prom), (prom > 0.02).sum(), (prom <= 0.01).sum()))

print('\n=== E) C-class (shift cannot reach the floor) re-tested with YOLOX IoU ===')
for r in rows:
    if r['cls'] != 'C': continue
    lab = r['label']; q = io.get(lab)
    print('  %-22s RTM lag %+6.1f align0 %6.1f -> align* %5.2f (CI +-%.1f, n=%d/%d)' % (
        lab, r['lag_eff'], r['align0'], r['align'], r['ci_half'], r['n'], r['N']))
    print('  %-22s IoU lag %+6.2f  IoUmax %.3f  IoU@0 %.3f (Δ%.3f)  n=%d  segs %s' % (
        '', q['lag_iou_sub'], q['iou_max'], q['iou_at0'], q['iou_max'] - q['iou_at0'], q['n_used'],
        [round(q.get(k + '_lag', float('nan')), 1) for k in ('seg1', 'seg2', 'seg3')]))

print('\n=== F) segment stability (IoU method) for |RTMPose lag| > 10 f ===')
seg = [x for x in d if abs(x[1]) > 10]
print('n=%d sessions with |lag_eff|>10' % len(seg))
stab = 0
for x in seg:
    q = io[x[0]]
    sv = [q.get(k + '_lag', np.nan) for k in ('seg1', 'seg2', 'seg3')]
    span = np.nanmax(sv) - np.nanmin(sv)
    ok = span <= 2.0
    stab += ok
    print('    %-22s RTM %+6.1f | IoU segs %+6.1f %+6.1f %+6.1f  span %5.1f %s' % (
        x[0], x[1], sv[0], sv[1], sv[2], span, 'STABLE' if ok else 'DRIFT'))
print('  of %d, span<=2f: %d (%.0f%%)' % (len(seg), stab, 100 * stab / len(seg)))
json.dump([{'label': x[0], 'lag_rtm': x[1], 'lag_iou': x[2], 'cls': x[3], 'delta': x[2] - x[1],
            'iou_max': x[7], 'iou_at0': x[8]} for x in d], open('/tmp/s1-scratch/compare_rtm_iou.json', 'w'), indent=1)
