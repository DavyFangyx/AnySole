"""S1: FINAL lag table + classes + CSV + statistics."""
import json, csv, numpy as np, collections
rows = [r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))]
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))

def classify(x):
    """Decision rule, fixed before reading outcomes:
       1) align0 <= 8 px  -> A 已对齐 (at the 4.5 px static floor within tolerance)
       2) n_gate < 100 or gate_frac < 0.20 -> D 证据不足 (too few gated frames)
       3) gain = align0 - align* < 3 px     -> D 证据不足 (nothing for a shift to remove)
       4) lag CI half-width > 4 f           -> D 证据不足 (curve cannot pin the lag)
       5) align* < 12 px                    -> B 可平移修复
       6) else                              -> C 修不动"""
    if x['align0'] <= 8.0: return 'A', ''
    if x['n'] < 100: return 'D', 'n_gate<100'
    if x['frac'] < 0.20: return 'D', 'gate_frac<0.20'
    if x['align0'] - x['align'] < 3.0: return 'D', 'gain<3px'
    if x['ci_half'] > 4.0: return 'D', 'lag_CI>+-4f'
    if x['align'] < 12.0: return 'B', ''
    return 'C', 'align*>=12px'

for x in rows:
    x['cls'], x['flag'] = classify(x)
    x['lag_eff'] = 0.0 if x['cls'] == 'A' else x['lag_sub']
    x['lag_eff_ms'] = x['lag_eff'] * 25.0
    q = io[x['label']]
    x['lag_iou'] = q['lag_iou_sub']; x['iou_max'] = q['iou_max']; x['iou_prom'] = q['iou_max'] - q['iou_at0']
    x['iou_disc'] = bool(q['iou_max'] >= 0.50 and (q['iou_max'] - q['iou_at0']) >= 0.02 and q['n_used'] >= 60)

with open('/tmp/s1-scratch/lag_fleet.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['session', 'conf_class', 'lag_frames', 'lag_ms', 'lag_eff_frames', 'lag_eff_ms',
                'align0_px', 'align_star_px', 'gain_px', 'lag_CI_half_f', 'n_gate', 'N', 'gate_frac',
                'lag_iou_frames', 'iou_max', 'iou_prominence', 'iou_discriminative', 'flag'])
    for x in sorted(rows, key=lambda r: (r['cls'], -abs(r['lag_eff']))):
        w.writerow([x['label'], x['cls'], '%.1f' % x['lag_sub'], '%+.0f' % x['lag_ms'], '%.1f' % x['lag_eff'],
                    '%+.0f' % x['lag_eff_ms'], '%.2f' % x['align0'], '%.2f' % x['align'], '%.2f' % x['gain'],
                    '%.1f' % x['ci_half'], x['n'], x['N'], '%.3f' % x['frac'], '%.2f' % x['lag_iou'],
                    '%.3f' % x['iou_max'], '%.3f' % x['iou_prom'], int(x['iou_disc']), x['flag']])
json.dump(rows, open('/tmp/s1-scratch/lag_fleet_final.json', 'w'), indent=1)

print('=== FINAL CLASSES (n=%d) ===' % len(rows))
c = collections.Counter(x['cls'] for x in rows)
print('  A 已对齐 %d | B 可平移修复 %d | C 修不动 %d | D 证据不足 %d' % (c['A'], c['B'], c['C'], c['D']))
print('  A+B = %d (%.1f%%) ; A+B+C (有明确结论) = %d (%.1f%%)' % (
    c['A'] + c['B'], 100 * (c['A'] + c['B']) / len(rows), c['A'] + c['B'] + c['C'], 100 * (c['A'] + c['B'] + c['C']) / len(rows)))
print('  D breakdown:', dict(collections.Counter(x['flag'] for x in rows if x['cls'] == 'D')))
print()
print('=== lag distribution (effective correction; A class = 0 by definition) ===')
for cl in 'ABCD':
    v = np.array([x['lag_eff'] for x in rows if x['cls'] == cl])
    if len(v): print('  %s n=%3d  min %+6.1f  p25 %+6.1f  median %+6.1f  p75 %+6.1f  max %+6.1f  mean %+6.2f' % (
        cl, len(v), v.min(), np.percentile(v, 25), np.median(v), np.percentile(v, 75), v.max(), v.mean()))
allv = np.array([x['lag_eff'] for x in rows])
print('  ALL n=%d  min %+.1f  p5 %+.1f  p25 %+.1f  median %+.1f  p75 %+.1f  p95 %+.1f  max %+.1f  mean %+.2f' % (
    len(allv), allv.min(), np.percentile(allv, 5), np.percentile(allv, 25), np.median(allv),
    np.percentile(allv, 75), np.percentile(allv, 95), allv.max(), allv.mean()))
print('  |lag|>10 f: %d (%.0f%%)  |lag|>5 f: %d  |lag|<=2 f: %d' % (
    (np.abs(allv) > 10).sum(), 100 * (np.abs(allv) > 10).sum() / len(allv), (np.abs(allv) > 5).sum(), (np.abs(allv) <= 2).sum()))
print('  positive (image LATE) %d ; negative (image EARLY) %d ; ~0 %d' % (
    (allv > 0.5).sum(), (allv < -0.5).sum(), (np.abs(allv) <= 0.5).sum()))
print('  ms range: %+.0f .. %+.0f' % (allv.min() * 25, allv.max() * 25))
print()
print('=== per-date ===')
byd = collections.defaultdict(list)
for x in rows: byd[x['date']].append(x)
for d in sorted(byd):
    v = np.array([x['lag_eff'] for x in byd[d]]); cc = collections.Counter(x['cls'] for x in byd[d])
    print('  %s n=%3d  median %+6.1f  sd %5.2f  range [%+.1f, %+.1f]  A%d B%d C%d D%d' % (
        d, len(v), np.median(v), v.std(ddof=1), v.min(), v.max(), cc['A'], cc['B'], cc['C'], cc['D']))
print()
print('=== agreement (headline) ===')
d2 = [x for x in rows if x['iou_disc']]
print('  IoU discriminative: %d/140' % len(d2))
k = sum(1 for x in d2 if abs(x['lag_iou'] - x['lag_eff']) <= 2)
print('  all discriminative: |Δ|<=2f %d/%d (%.0f%%)  median|Δ| %.2f' % (k, len(d2), 100 * k / len(d2), np.median([abs(x['lag_iou'] - x['lag_eff']) for x in d2])))
h = [x for x in d2 if x['cls'] in 'BC' and x['ci_half'] <= 2.0]
k2 = sum(1 for x in h if abs(x['lag_iou'] - x['lag_eff']) <= 2)
print('  B/C & CI<=2f & discriminative: |Δ|<=2f %d/%d (%.0f%%)  median|Δ| %.2f' % (k2, len(h), 100 * k2 / len(h), np.median([abs(x['lag_iou'] - x['lag_eff']) for x in h])))
hh = [x for x in d2 if abs(x['lag_eff']) > 10 and x['ci_half'] <= 2.0]
k3 = sum(1 for x in hh if abs(x['lag_iou'] - x['lag_eff']) <= 2)
print('  |lag|>10 & CI<=2f & discriminative: |Δ|<=2f %d/%d (%.0f%%)' % (k3, len(hh), 100 * k3 / len(hh)))
print()
print('=== A-class sessions whose curve argmin is far from 0 (flag for transparency) ===')
for x in sorted([y for y in rows if y['cls'] == 'A'], key=lambda y: -abs(y['lag_sub'])):
    if abs(x['lag_sub']) >= 5 and x['gain'] >= 2:
        print('    %-22s align0 %5.2f -> align* %5.2f at lag %+.1f (CI+-%.1f) [lag_eff forced to 0]' % (
            x['label'], x['align0'], x['align'], x['lag_sub'], x['ci_half']))
