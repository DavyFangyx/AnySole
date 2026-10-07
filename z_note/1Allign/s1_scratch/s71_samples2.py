"""S1 task2b (proper): sampling verification from the RELIABLE population (CI<=2f, n>=120),
   two ends + middle; plus the naive absolute-extremes sample for contrast."""
import json, numpy as np
rt = {r['label']: r for r in json.load(open('/tmp/s1-scratch/lag_fleet.json'))}
io = json.load(open('/tmp/s1-scratch/iou_fleet_rtm.json'))
rows = sorted(rt.values(), key=lambda r: r['lag_eff'])

def line(r):
    q = io[r['label']]
    d = q['lag_iou_sub'] - r['lag_eff']
    prom = q['iou_max'] - q['iou_at0']
    usable = q['iou_max'] >= 0.50 and prom >= 0.02 and q['n_used'] >= 60
    return d, prom, usable, ('PASS' if abs(d) <= 2 else 'FAIL') if usable else 'IoU 不可判别'

def run(pick, title):
    print('=== %s ===' % title)
    print('%-22s %-3s %8s %8s %7s %8s %7s  %s' % ('session', 'cls', 'RTM*', 'IoU', 'Δ', 'IoUmax', 'prom', '判据'))
    ok = tot = 0; oku = totu = 0
    for r in pick:
        q = io[r['label']]
        d, prom, usable, v = line(r)
        tot += 1; ok += (abs(d) <= 2)
        if usable: totu += 1; oku += (abs(d) <= 2)
        print('%-22s %-3s %+8.1f %+8.2f %+7.2f %8.3f %7.3f  %s' % (
            r['label'], r['cls'], r['lag_eff'], q['lag_iou_sub'], d, q['iou_max'], prom, v))
    print('  通过率（|Δ|<=2f，全部样本）: %d/%d' % (ok, tot))
    print('  通过率（仅 IoU 可判别样本）: %d/%d' % (oku, totu))
    print()
    return ok, tot, oku, totu

rel = [r for r in rows if r['ci_half'] <= 2.0 and r['n'] >= 120 and r['frac'] >= 0.20]
rel.sort(key=lambda r: r['lag_eff'])
print('可靠总体 (CI<=2f & n>=120 & gate_frac>=0.20): %d sessions, lag 范围 %+.1f .. %+.1f' % (
    len(rel), rel[0]['lag_eff'], rel[-1]['lag_eff']))
neg = rel[:3]; pos = rel[-3:]; mid = sorted(rel, key=lambda r: abs(r['lag_eff']))[:3]
pickRel = neg + mid + pos
print('抽 9 个(两端各 3 + 中间 3): %s' % [r['label'] for r in pickRel])
print()
a = run(pickRel, 'task 2b 抽样复核 A：从可靠总体两端+中间各抽 3（共 9）')

neg2 = sorted([r for r in rows if r['lag_eff'] < -6], key=lambda r: r['lag_eff'])[:3]
pos2 = sorted([r for r in rows if r['lag_eff'] > 6], key=lambda r: -r['lag_eff'])[:3]
mid2 = sorted([r for r in rows if abs(r['lag_eff']) <= 3], key=lambda r: abs(r['lag_eff']))[:4]
b = run(neg2 + pos2 + mid2, 'task 2b 抽样复核 B：不加可靠度筛，取 |lag| 绝对两端 + 中间（对照）')

rows2 = [r for r in rows if abs(r['lag_eff']) > 10]
strong = [r for r in rows2 if r['ci_half'] <= 2.0 and io[r['label']]['iou_max'] - io[r['label']]['iou_at0'] >= 0.02]
print('=== |lag|>10 且两法均可判别的全部 session（%.0f%% 通过率检查的完整子集）===' % 0)
print('%d sessions' % len(strong))
k = sum(1 for r in strong if abs(io[r['label']]['lag_iou_sub'] - r['lag_eff']) <= 2)
print('  |Δ|<=2f: %d/%d (%.0f%%)' % (k, len(strong), 100 * k / max(1, len(strong))))
for r in sorted(strong, key=lambda r: -abs(r['lag_eff'])):
    d, prom, usable, v = line(r)
    print('    %-22s RTM %+6.1f IoU %+6.2f Δ %+5.2f prom %.3f  %s' % (r['label'], r['lag_eff'], io[r['label']]['lag_iou_sub'], d, prom, v))
