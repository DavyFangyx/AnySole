"""Portable pressure reader: corrected values are explicit, original arrays retained.

v3: original48 = true raw; fault_repaired48 = fault-cell burst repair
(fault_cell.py) applied on raw rows before any baseline estimation;
denoised48/impulse_only48 = isolated-impulse step on the repaired input;
processed48 = baseline_step_corrected output (values48 default).
"""
from pathlib import Path
import csv,hashlib
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4)
def index(root):
    with (Path(root)/'file_manifest.csv').open(encoding='utf-8-sig') as f:return {(r['session'],r['side']):r for r in csv.DictReader(f)}

def load_pressure(root,session,side,*,version='baseline_step_corrected',verify_hash=True):
    key={'raw_preserved':'original48','fault_repaired':'fault_repaired48','denoised':'denoised48','impulse_only':'impulse_only48','constant_baseline':'constant_corrected48','baseline_step_corrected':'processed48'}
    if version not in key:raise ValueError('unknown version')
    root=Path(root);r=index(str(root.resolve()))[session,side];p=root/r['relative_npz']
    if verify_hash and hashlib.sha256(p.read_bytes()).hexdigest()!=r['npz_sha256']:raise ValueError('hash mismatch')
    with np.load(p,allow_pickle=False) as z:out={k:z[k] for k in z.files}
    out['values48']=out[key[version]]
    out['pressure_value_valid']=np.isfinite(out['original48']).all(1)&(out['original48']>=0).all(1)
    out.update(metadata=r,version=version,missing=len(out['values48'])==0)
    return out
