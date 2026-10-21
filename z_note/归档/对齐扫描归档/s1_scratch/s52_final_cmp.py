"""S1: final RTMPose-vs-YOLOXIoU comparison on the rtm-gated IoU estimates."""
import json, numpy as np
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))
geo = json.load(open('/tmp/s1-scratch/iou_geo.json'))

rows = []
for lab, r in rt.items():
    q = io[lab]
    rows.append(dict(label=lab, cls=r['cls'], rtm=r['lag_eff'], iou=q['lag_iou_sub'], d=q['lag_iou_sub'] - r['lag_eff'],
                     ioumax=q['iou_max'], iou_at0=q['iou_at0'], prom=q['iou_max'] - q['iou_at0'],
                     ci=r['ci_half'], align0=r['align0'], align=r['align'], n_rtm=r['n'],
                     n_iou=q['n_used'], gate_frac=geo[lab]['gate_frac'],
                     seg=[q.get('seg%d_lag' % k, np.nan) for k in (1, 2, 3)]))

print('=== IoU estimate quality with the rtm gate (the s16 setting) ===')
mx = np.array([x['ioumax'] for x in rows]); pr = np.array([x['prom'] for x in rows])
print('  IoUmax: median %.3f  <0.30: %d  <0.50: %d  >=0.50: %d' % (np.median(mx), (mx < .30).sum(), (mx < .50).sum(), (mx >= .50).sum()))
print('  prominence(IoUmax-IoU@0): median %.3f  >=0.02: %d  <0.02: %d' % (np.median(pr), (pr >= .02).sum(), (pr < .02).sum()))
usable = [x for x in rows if x['ioumax'] >= 0.50 and x['n_iou'] >= 60]
disc = [x for x in usable if x['prom'] >= 0.02]
print('  usable(IoUmax>=0.50 & n>=60): %d ; of those with prominence>=0.02: %d' % (len(usable), len(disc)))

def rate(sub, nm):
    if not sub: print('  %s: n=0' % nm); return
    for thr in (2, 3):
        k = sum(1 for x in sub if abs(x['d']) <= thr)
        print('  %-46s |Δ|<=%df: %3d/%3d (%.0f%%)  median|Δ| %.2f  mean|Δ| %.2f' % (
            nm, thr, k, len(sub), 100 * k / len(sub), np.median([abs(x['d']) for x in sub]), np.mean([abs(x['d']) for x in sub])))

print('\n=== agreement rates ===')
rate(usable, 'usable IoU (all classes)')
rate(disc, 'usable + discriminative IoU (all classes)')
for cl in 'ABCD':
    rate([x for x in disc if x['cls'] == cl], 'class %s, usable+discriminative' % cl)
print()
rate([x for x in disc if x['cls'] in 'BC' and x['ci'] <= 2.0], 'B/C class & CI<=2f & usable IoU  <-- headline cross-check')
rate([x for x in disc if x['cls'] in 'BC'], 'B/C class & usable IoU')
rate([x for x in disc if x['cls'] in 'AD'], 'A/D class & usable IoU')

print('\n=== disagreements in the headline group ===')
h = [x for x in disc if x['cls'] in 'BC' and x['ci'] <= 2.0 and abs(x['d']) > 2]
for x in sorted(h, key=lambda y: -abs(y['d'])):
    print('    %-22s RTM %+6.1f  IoU %+6.2f  Δ %+6.2f | align0 %6.1f align* %5.2f IoUmax %.3f prom %.3f n=%d' % (
        x['label'], x['rtm'], x['iou'], x['d'], x['align0'], x['align'], x['ioumax'], x['prom'], x['n_iou']))

print('\n=== task 2a: C class (4 sessions, shift never reaches the floor) re-tested with IoU ===')
for x in [y for y in rows if y['cls'] == 'C']:
    lab = x['label']
    print('  %-22s RTM %+6.1f (align0 %6.1f -> align* %5.2f, CI+-%.1f, n=%d)' % (lab, x['rtm'], x['align0'], x['align'], x['ci'], x['n_rtm']))
    print('  %-22s IoU %+6.2f  IoUmax %.3f  IoU@0 %.3f (prom %.3f)  n_iou %d  segs [%s]  gate_frac %.2f' % (
        '', x['iou'], x['ioumax'], x['iou_at0'], x['prom'], x['n_iou'],
        ', '.join('%+.1f' % s for s in x['seg']), x['gate_frac']))

print('\n=== task 2a: D class, can IoU resolve them? ===')
dd = [x for x in rows if x['cls'] == 'D']
res = [x for x in dd if x['ioumax'] >= 0.50 and x['prom'] >= 0.02 and x['n_iou'] >= 60]
un = [x for x in dd if not (x['ioumax'] >= 0.50 and x['prom'] >= 0.02 and x['n_iou'] >= 60)]
print('  D n=%d: resolved by IoU (IoUmax>=0.50, prom>=0.02, n>=60) = %d ; still unresolved = %d' % (len(dd), len(res), len(un)))
print('  resolved D (now with an independent lag):')
for x in sorted(res, key=lambda y: -abs(y['iou'])):
    print('    %-22s RTM %+6.1f (CI+-%.1f%s)  IoU %+6.2f (max %.3f prom %.3f n=%d)' % (
        x['label'], x['rtm'], x['ci'], ', n=%d' % x['n_rtm'] if x['n_rtm'] < 100 else '', x['iou'], x['ioumax'], x['prom'], x['n_iou']))
print('  still unresolved D (IoU also fails):')
for x in sorted(un, key=lambda y: y['ioumax']):
    print('    %-22s RTM %+6.1f (CI+-%.1f, n=%d/%d)  IoU %+6.2f IoUmax %.3f prom %.3f n_iou %d gate_frac %.2f' % (
        x['label'], x['rtm'], x['ci'], x['n_rtm'], rt[x['label']]['N'], x['iou'], x['ioumax'], x['prom'], x['n_iou'], x['gate_frac']))
json.dump(rows, open('/tmp/s1-scratch/final_cmp.json', 'w'), indent=1)
