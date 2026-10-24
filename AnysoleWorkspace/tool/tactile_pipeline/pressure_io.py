"""Portable pressure reader: corrected values are explicit, original arrays retained.

v3: original48 = true raw; fault_repaired48 = fault-cell burst repair
(fault_cell.py) applied on raw rows before any baseline estimation;
denoised48/impulse_only48 = isolated-impulse step on the repaired input;
processed48 = baseline_step_corrected output (values48 default).

2026-10-24 定妆照：无 file_manifest.csv 索引——dataset 树本身即目录
（{date}/{subject}/{session}/{side}.npz，路径即 session+side 映射）。
行级质量包（{side}_{t_us,raw_t_us,valid,fake}）已由生产端烤进各 npz，
消费端不依赖任何生产侧中间件。
"""
from pathlib import Path
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4)
def index(root):
    table = {}
    for p in sorted((Path(root)/'dataset').glob('*/*/*/*.npz')):
        table[(p.parent.name, p.stem)] = {'session': p.parent.name, 'side': p.stem,
                                          'relative_npz': str(p.relative_to(root))}
    return table

def load_pressure(root,session,side,*,version='baseline_step_corrected'):
    key={'raw_preserved':'original48','fault_repaired':'fault_repaired48','denoised':'denoised48','impulse_only':'impulse_only48','constant_baseline':'constant_corrected48','baseline_step_corrected':'processed48'}
    if version not in key:raise ValueError('unknown version')
    root=Path(root);r=index(str(root.resolve()))[session,side];p=root/r['relative_npz']
    with np.load(p,allow_pickle=False) as z:out={k:z[k] for k in z.files}
    out['values48']=out[key[version]]
    out['pressure_value_valid']=np.isfinite(out['original48']).all(1)&(out['original48']>=0).all(1)
    out.update(metadata=r,version=version,missing=len(out['values48'])==0)
    return out
