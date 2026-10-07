"""S1 task2b: independent sampling verification (~10 sessions: both ends + middle),
   plus reconciliation of the old report's '38 aligned / 47 fixable / 6 unfixable' counts."""
import json, numpy as np
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))
rows = sorted(rt.values(), key=lambda r: r['lag_eff'])

print('=== reconciliation with 15_R2 doc counts ===')
a0 = np.array([r['align0'] for r in rows]); al = np.array([r['align'] for r in rows])
print('  old: align0<=8 -> 38 ; this run (wide range): %d' % (a0 <= 8).sum())
old38 = {r['label'] for r in rows if r['align0'] <= 8}
oldm = {r['label']: r for r in json.load(open('/tmp/r2-scratch/multi_lag.json'))}
old38m = {l for l, r in oldm.items() if r['align0'] <= 8}
print('  sets identical to old multi_lag align0<=8 ? %s  (sym-diff %s)' % (old38 == old38m, sorted(old38 ^ old38m)))
big = [r for r in rows if r['align0'] > 15]
print('  old: align0>15 -> 53 ; this run: %d' % len(big))
fixm = [r for r in big if r['align'] < 12]; unf = [r for r in big if r['align'] >= 12]
print('  old: of the 53, 47 fixed by a shift, 6 not ; this run (wide range): %d fixed, %d not' % (len(fixm), len(unf)))
print('  the %d not fixed by a shift at align0>15 (old \"6\"口径):' % len(unf))
for r in sorted(unf, key=lambda r: -r['align0']):
    q = io[r['label']]
    print('    %-22s lag %+6.1f  align0 %6.1f -> %5.2f  n=%3d/%3d(%.0f%%) gate  IoU %+6.2f (max %.3f prom %.3f)' % (
        r['label'], r['lag_eff'], r['align0'], r['align'], r['n'], r['N'], 100 * r['frac'], q['lag_iou_sub'], q['iou_max'], q['iou_max'] - q['iou_at0']))

print('\n=== task 2b: sampling verification, ~10 sessions from both ends and the middle ===')
neg = sorted([r for r in rows if r['lag_eff'] < -6], key=lambda r: r['lag_eff'])[:3]
pos = sorted([r for r in rows if r['lag_eff'] > 6], key=lambda r: -r['lag_eff'])[:3]
mid = sorted([r for r in rows if abs(r['lag_eff']) <= 3], key=lambda r: abs(r['lag_eff']))[:4]
pick = neg + pos + mid
print('  picked %d: %s' % (len(pick), [r['label'] for r in pick]))
print()
print('%-22s %-3s %8s %8s %7s %7s %7s  %s' % ('session', 'cls', 'RTM*', 'IoU', 'Δ', 'IoUmax', 'prom', 'verdict'))
ok = 0; okd = 0; nd = 0
for r in pick:
    q = io[r['label']]
    d = q['lag_iou_sub'] - r['lag_eff']
    usable = q['iou_max'] >= 0.50 and (q['iou_max'] - q['iou_at0']) >= 0.02 and q['n_used'] >= 60
    v = ('PASS' if abs(d) <= 2 else 'FAIL') if usable else 'IoU-not-discriminative'
    ok += (abs(d) <= 2) and usable
    nd += usable
    print('%-22s %-3s %+8.1f %+8.2f %+7.2f %7.3f %7.3f  %s' % (
        r['label'], r['cls'], r['lag_eff'], q['lag_iou_sub'], d, q['iou_max'], q['iou_max'] - q['iou_at0'], v))
print()
print('  pass rate on the sampled 10: %d/10 ; among those where the IoU curve is discriminative: %d/%d' % (ok, ok, nd))

print('\n=== fleet-level cross-check (headline) ===')
u = [r for r in rows if io[r['label']]['iou_max'] >= 0.50 and (io[r['label']]['iou_max'] - io[r['label']]['iou_at0']) >= 0.02 and io[r['label']]['n_used'] >= 60]
bc = [r for r in u if r['cls'] in 'BC' and r['ci_half'] <= 2.0]
k = sum(1 for r in bc if abs(io[r['label']]['lag_iou_sub'] - r['lag_eff']) <= 2)
print('  B/C & CI<=2f & discriminative IoU: n=%d  |Δ|<=2f %d (%.0f%%)  median|Δ| %.2f' % (
    len(bc), k, 100 * k / max(1, len(bc)), np.median([abs(io[r['label']]['lag_iou_sub'] - r['lag_eff']) for r in bc])))
ab = [r for r in u]
k2 = sum(1 for r in ab if abs(io[r['label']]['lag_iou_sub'] - r['lag_eff']) <= 2)
print('  all classes, discriminative IoU:  n=%d  |Δ|<=2f %d (%.0f%%)' % (len(ab), k2, 100 * k2 / max(1, len(ab))))
