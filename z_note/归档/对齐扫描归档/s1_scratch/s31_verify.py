"""S1 task1: json-vs-rerun agreement for the three sweep settings."""
import json, numpy as np
old_m = {r['label']: r for r in json.load(open('/tmp/r2-scratch/multi_lag.json'))}
old_w = {r['label']: r for r in json.load(open('/tmp/r2-scratch/wide_lag.json'))}
new_n = {r['label']: r for r in json.load(open('/tmp/s1-scratch/narrow_lag.json'))}
new_w = {r['label']: r for r in json.load(open('/tmp/s1-scratch/wideint_lag.json'))}
new_f = {r['label']: r for r in json.load(open('/tmp/s1-scratch/full_lag.json'))}

def cmp(a, b, keys, name, labels=None):
    labels = labels or sorted(a)
    bad = []
    for lab in labels:
        if lab not in b: bad.append((lab, 'MISSING', None)); continue
        for k in keys:
            va, vb = a[lab][k], b[lab][k]
            if isinstance(va, float):
                if not np.isclose(va, vb, rtol=0, atol=1e-9): bad.append((lab, k, va, vb))
            elif va != vb: bad.append((lab, k, va, vb))
    print('%-42s sessions=%3d  fields=%s  mismatches=%d' % (name, len(labels), ','.join(keys), len(bad)))
    for x in bad[:10]: print('     ', x)
    return bad

print('=== replication of OLD multi_lag.json (range -4..20, step 0.5) ===')
cmp(old_m, new_n, ['lag', 'align', 'align0', 'shape', 'shape0', 'n', 'N'], 'multi_lag.json vs rerun(narrow)')
print()
print('=== replication of OLD wide_lag.json (range -30..30, step 1.0, 53 sessions) ===')
cmp(old_w, new_w, ['lag', 'align', 'align0', 'n', 'N'], 'wide_lag.json vs rerun(wide,int)', sorted(old_w))
print()

print('=== per-session lag deltas (multi vs wide settings), all 140 ===')
d = []
for lab in sorted(new_f):
    d.append((lab, new_n[lab]['lag'], new_w[lab]['lag'], new_f[lab]['lag'],
              old_m[lab]['lag'], old_w.get(lab, {}).get('lag')))
d.sort(key=lambda x: -abs(x[3] - x[4]))
print('%d sessions | multi-settings(clipped) differs from wide-settings by >0.5f: %d' % (
    len(d), sum(1 for x in d if abs(x[3] - x[4]) > 0.5)))
print('sessions where old multi lag == -4.0 (lower boundary):', sum(1 for x in d if x[4] == -4.0))
print('sessions where old multi lag == +20.0 (upper boundary):', sum(1 for x in d if x[4] == 20.0))
print('\ntop 15 |new_full - old_multi| (these are the boundary-clipped ones):')
for x in d[:15]:
    print('  %-22s old_multi %+6.1f  new_narrow %+6.1f  new_wide_int %+6.1f  NEW_full %+6.1f' % (x[0], x[4], x[1], x[2], x[3]))
json.dump([{'label': x[0], 'lag_narrow': x[1], 'lag_wideint': x[2], 'lag_full': x[3],
            'old_multi': x[4], 'old_wide': x[5]} for x in d], open('/tmp/s1-scratch/replication_deltas.json', 'w'), indent=1)
