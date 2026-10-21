import json, numpy as np, collections
rows = json.load(open('/tmp/s1-scratch/lag_fleet.json'))
a0 = np.array([r['align0'] for r in rows]); al = np.array([r['align'] for r in rows])
gn = np.array([r['gain'] for r in rows]); ci = np.array([r['ci_half'] for r in rows])
n = np.array([r['n'] for r in rows]); fr = np.array([r['frac'] for r in rows])

print('=== joint table: rows = align0 bin, cols = gain bin (counts) ===')
gb = [0, 1, 3, 6, 12, 1e9]; gname = ['<1', '1-3', '3-6', '6-12', '>12']
ab = [0, 5, 8, 12, 15, 25, 1e9]; aname = ['<=5', '5-8', '8-12', '12-15', '15-25', '>25']
print('%-8s' % 'align0', ''.join('%8s' % g for g in gname), '  | total  CI<=4f  n>=100')
for i in range(len(ab) - 1):
    m = (a0 > ab[i]) & (a0 <= ab[i + 1])
    print('%-8s' % aname[i], ''.join('%8d' % int(((gn > gb[j]) & (gn <= gb[j + 1]) & m).sum()) for j in range(len(gb) - 1)),
          '  | %5d  %5d  %5d' % (m.sum(), (m & (ci <= 4)).sum(), (m & (n >= 100)).sum()))

print('\n=== CI half-width distribution by align0 bin ===')
for i in range(len(ab) - 1):
    m = (a0 > ab[i]) & (a0 <= ab[i + 1])
    if m.sum() > 0:
        c = ci[m]
        print('  align0 %-8s n=%3d  CI<=2f:%3d  2-4f:%3d  4-8f:%3d  >8f:%3d' % (
            aname[i], m.sum(), (c <= 2).sum(), ((c > 2) & (c <= 4)).sum(), ((c > 4) & (c <= 8)).sum(), (c > 8).sum()))

print('\n=== per-DATE structure of the correction lag (BC sessions = mispaired & shift works) ===')
byd = collections.defaultdict(list)
for r in rows:
    if r['align0'] > 8 and r['gain'] >= 3 and r['n'] >= 100 and r['frac'] >= 0.20:
        byd[r['date']].append(r['lag_eff'])
for d in sorted(byd):
    v = np.array(byd[d])
    print('  %s n=%2d  median %+6.1f  mean %+6.1f  sd %5.1f  range [%+.1f, %+.1f]  span %5.1f f' % (
        d, len(v), np.median(v), v.mean(), v.std(ddof=1) if len(v) > 1 else 0, v.min(), v.max(), v.max() - v.min()))

print('\n=== per-DATE+SUBJECT ===')
bys = collections.defaultdict(list)
for r in rows:
    if r['align0'] > 8 and r['gain'] >= 3 and r['n'] >= 100 and r['frac'] >= 0.20:
        bys[(r['date'], r['subj'])].append(r['lag_eff'])
for k in sorted(bys):
    v = np.array(bys[k])
    if len(v) > 1:
        print('  %s %-5s n=%d  lags %s  span %.1f  %s' % (k[0], k[1], len(v), sorted(v), v.max() - v.min(),
                                                          'CONSTANT' if v.max() - v.min() <= 2 else 'VARIES'))

print('\n=== lag_eff vs align0 sign groups ===')
mis = [r for r in rows if r['align0'] > 8 and r['gain'] >= 3]
p = [r for r in mis if r['lag_eff'] > 0.5]; q = [r for r in mis if r['lag_eff'] < -0.5]
print('mispaired-and-fixable n=%d  positive lag (image LATE, GT ahead) %d  negative lag (image EARLY) %d' % (len(mis), len(p), len(q)))
print('  positive lags:', sorted(r['lag_eff'] for r in p))
print('  negative lags (median %.1f):' % np.median([r['lag_eff'] for r in q]), sorted(r['lag_eff'] for r in q)[:20], '...')
