"""Verify the complete package and all delivered pressure/CSV/change-log files.

v3: the value-change chain is raw -> fault_repaired (fault_cell_changes) ->
impulse_only (impulse_changes) -> processed (changes); every change appears
in exactly one log. original48 remains the true raw array.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from pressure_io import load_pressure

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2]/'work'/'data_pipeline'/'tactile_v3');args=ap.parse_args();root=args.package_root.resolve()
    manifest_path=root/'MANIFEST.sha256.json';manifest_files=0
    code_root=Path(__file__).resolve().parent
    if manifest_path.exists():
        entries=json.loads(manifest_path.read_text())
        for rel,expected in entries.items():
            # 代码/数据拆分后：configs/ 与 pipeline/ 归 code 目录（pipeline/
            # 前缀去掉——模块现在在 code 根），其余在 data 根。
            if rel.startswith('configs/'):
                base, tail = code_root, rel
            elif rel.startswith('pipeline/'):
                base, tail = code_root, rel[len('pipeline/'):]
            else:
                base, tail = root, rel
            path=(base/tail).resolve();assert path.is_relative_to(base) and sha(path)==expected,rel
        manifest_files=len(entries)
    originals=read(root/'inputs/raw_manifest.csv')
    for r in originals:
        assert sha(root/r['relative_raw_csv'])==r['raw_csv_sha256']
        if r['relative_metadata']:assert sha(root/r['relative_metadata'])==r['metadata_sha256']
    values=modified=rawmodified=empty=impulses=faultvalues=0;files=read(root/'data/file_manifest.csv')
    for r in files:
        z=load_pressure(root/'data',r['session'],r['side']);raw=z['original48'];repaired=z['fault_repaired48'];den=z['denoised48'];out=z['values48'];b=z['estimated_baseline48'];ok=z['row_alignment_eligible'];n=len(raw)
        assert out.shape==(int(r['rows']),48) and np.isfinite(out).all() and (out>=0).all()
        assert np.array_equal(den,z['impulse_only48'])
        assert np.array_equal(out,np.maximum(den-b,0)) and np.array_equal(out[~ok],den[~ok])
        assert int(r['fault_burst_values'])==np.count_nonzero(repaired!=raw)
        assert sha(root/'data'/r['relative_csv'])==r['csv_sha256']
        assert sha(root/'data'/r['relative_changes'])==r['changes_sha256']
        assert sha(root/'data'/r['relative_impulse_changes'])==r['impulse_changes_sha256']
        assert sha(root/'data'/r['relative_fault_cell_changes'])==r['fault_cell_changes_sha256']
        with np.load(root/'data'/r['relative_fault_cell_changes'],allow_pickle=False) as log:
            fr,fc=np.nonzero(repaired!=raw)
            order=np.argsort(fr*48+fc);fr,fc=fr[order],fc[order]
            lr,lc=log['row0'],log['channel']-1
            lorder=np.argsort(lr*48+lc);lr,lc=lr[lorder],lc[lorder]
            assert np.array_equal(fr,lr) and np.array_equal(fc,lc)
            assert np.array_equal(raw[fr,fc],log['before'][lorder]) and np.array_equal(repaired[fr,fc],log['after'][lorder])
        faultvalues+=len(fr)
        with np.load(root/'data'/r['relative_impulse_changes'],allow_pickle=False) as log:
            ir,ic=np.nonzero(den!=repaired)
            assert np.array_equal(ir,log['row0']) and np.array_equal(ic+1,log['channel'])
            assert np.array_equal(repaired[ir,ic],log['before']) and np.array_equal(den[ir,ic],log['after'])
        impulses+=len(ir)
        with np.load(root/'data'/r['relative_changes'],allow_pickle=False) as log:
            rr,cc=np.nonzero(out!=den);assert np.array_equal(rr,log['row0']) and np.array_equal(cc+1,log['channel'])
            assert np.array_equal(den[rr,cc],log['before']) and np.array_equal(out[rr,cc],log['after'])
        values+=raw.size;modified+=np.count_nonzero(out!=den);rawmodified+=np.count_nonzero(out!=raw);empty+=n==0
    expected=json.loads((root/'reports/full_rebuild_validation.json').read_text())
    assert (len(files),values,modified,rawmodified,empty,impulses,faultvalues)==(890,9888960,expected['baseline_modified_values'],expected['total_changed_from_raw'],64,expected['isolated_impulse_modified_values'],expected['fault_burst_modified_values'])
    assert expected['adaptive_smoothing_enabled'] is False and expected['adaptive_smoothing_modified_values']==0
    print(json.dumps(dict(status='PASS',package_manifest_files=manifest_files,raw_foot_files=len(originals),
        processed_foot_files=len(files),pressure_values=int(values),baseline_modified_values=int(modified),
        total_changed_from_raw=int(rawmodified),empty_files=int(empty),isolated_impulse_modified_values=int(impulses),
        fault_burst_modified_values=int(faultvalues),adaptive_smoothing_enabled=False,external_paths_required=False)))

if __name__=='__main__':main()