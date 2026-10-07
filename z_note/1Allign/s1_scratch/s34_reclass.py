"""S1 task1: final classification (reordered) + per-date/subject clustering + flags."""
import json, csv, numpy as np, collections
rows = json.load(open('/tmp/s1-scratch/lag_fleet.json'))

def classify(x):
    """Order matters: an already-at-floor session needs no shift regardless of where a
       flat curve's argmin happens to sit, so A is tested first."""
    if x['n'] < 100:            return 'D', 'n_gate<100'
    if x['frac'] < 0.20:        return 'D', 'gate_frac<0.20'
    if x['align0'] <= 8.0:      return 'A', ''
    if x['ci_half'] > 4.0:      return 'D', 'lag_CI>+-4f'
    if x['align'] < 12.0:       return 'B', ''
    return 'C', 'align*>12px'

for x in rows:
    x['cls'], x['flag'] = classify(x)
    x['lag_eff'] = 0.0 if x['cls'] == 'A' else x['lag_sub']
    x['lag_eff_ms'] = x['lag_eff'] * 25.0
json.dump(rows, open('/tmp/s1-scratch/lag_fleet.json', 'w'), indent=1)

with open('/tmp/s1-scratch/lag_fleet.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['session', 'lag_frames', 'lag_ms', 'lag_eff_frames', 'lag_eff_ms', 'align0_px', 'align_star_px',
                'gain_px', 'n_gate', 'N', 'gate_frac', 'ci_lo_f', 'ci_hi_f', 'ci_half_f', 'conf_class', 'flag'])
    for x in sorted(rows, key=lambda r: (r['cls'], -abs(r['lag_eff']))):
        w.writerow([x['label'], '%.1f' % x['lag_sub'], '%+.0f' % x['lag_ms'], '%.1f' % x['lag_eff'],
                    '%+.0f' % x['lag_eff_ms'], '%.2f' % x['align0'], '%.2f' % x['align'], '%.2f' % x['gain'],
                    x['n'], x['N'], '%.3f' % x['frac'], '%.1f' % x['ci_lo'], '%.1f' % x['ci_hi'],
                    '%.1f' % x['ci_half'], x['cls'], x['flag']])

cnt = collections.Counter(x['cls'] for x in rows)
print('=== FINAL CLASS COUNTS (140) ===')
print(dict(sorted(cnt.items())))
for cl in 'ABCD':
    sub = [x for x in rows if x['cls'] == cl]
    print('\n--- %s (n=%d) ---' % (cl, len(sub)))
    for x in sorted(sub, key=lambda r: -abs(r['lag_eff'])):
        print('  %-22s lag_eff %+6.1f f (%+6.0f ms)  align0 %6.1f -> %5.2f  n=%3d/%3d (%.0f%%) CI +-%.1f f  %s' % (
            x['label'], x['lag_eff'], x['lag_eff_ms'], x['align0'], x['align'], x['n'], x['N'],
            100 * x['frac'], x['ci_half'], x['flag']))

print('\n=== per-DATE structure (is the offset a per-day constant or per-session?) ===')
byd = collections.defaultdict(list)
for x in rows:
    if x['cls'] in 'BC':
        byd[x['date']].append(x['lag_eff'])
for d in sorted(byd):
    v = np.array(byd[d])
    print('  %s  n=%2d  median %+6.1f f  mean %+6.1f  sd %5.1f  min %+6.1f  max %+6.1f  span %5.1f f' % (
        d, len(v), np.median(v), v.mean(), v.std(ddof=1) if len(v) > 1 else 0, v.min(), v.max(), v.max() - v.min()))
print('\n=== per-DATE+SUBJECT structure ===')
bys = collections.defaultdict(list)
for x in rows:
    if x['cls'] in 'BC':
        bys[(x['date'], x['subj'])].append(x['lag_eff'])
multi = {k: v for k, v in bys.items() if len(v) > 1}
same = sum(1 for k, v in multi.items() if max(v) - min(v) <= 2.0)
print('  (date,subject) groups with >1 BC session: %d ; of those with within-group span <=2 f: %d' % (len(multi), same))
for k in sorted(multi):
    v = np.array(multi[k])
    print('    %s %-5s n=%d lags %s  span %.1f' % (k[0], k[1], len(v), sorted(v), v.max() - v.min()))
