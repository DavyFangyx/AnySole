"""Offline end-to-end replay from package-native raw CSV and original metadata.

v3: fault-cell burst repair (fault_cell.py) runs on the raw rows BEFORE
analyze/process/baseline, so bursts never enter the airborne baseline.
original48 keeps the true raw array; fault_repaired48 is the repair input
to the unchanged v2 treatments (isolated impulse / airborne baseline /
supported step). Every value change appears in exactly one change log:
fault_cell_changes (raw -> fault_repaired), impulse_changes
(fault_repaired -> impulse_only), changes (impulse_only -> processed).
"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,shutil
import numpy as np
from quality_rules import analyze
from noise_model import process
from baseline_core import make_baseline
from fault_cell import repair as repair_fault_bursts
from detector import scan_steps

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def inside(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('relative input path escapes package')
    return p
def same(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    return np.array_equal(a,b,equal_nan=True) if np.issubdtype(a.dtype,np.inexact) else np.array_equal(a,b)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package-root',type=Path,
                    default=Path(__file__).resolve().parents[2]/'work'/'data_pipeline'/'tactile_v3')
    ap.add_argument('--output',type=Path,required=True,help='new directory outside this package')
    ap.add_argument('--verify-reference',action='store_true',help='compare every rebuilt array and CSV with shipped data')
    ap.add_argument('--scan',action='store_true',help='also rerun full-population statistical step screening')
    args=ap.parse_args();root=args.package_root.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('output exists; choose a fresh directory')
    if out.is_relative_to(root):raise ValueError('output must be outside immutable package')
    inventory=read(root/'inputs/raw_manifest.csv')
    anchors=collections.defaultdict(list)
    for a in json.loads((Path(__file__).resolve().parent/'configs'/'calibration_anchors.json').read_text()):anchors[a['session'],a['side']].append(a)
    # Shipped data/ is itself the rebuild output; only needed as a reference
    # comparison when --verify-reference is requested (post-shipping check).
    reference={(r['session'],r['side']):r for r in read(root/'data/file_manifest.csv')} if args.verify_reference else {}
    out.mkdir(parents=True);files=[];models=[];steps=[];screening=[];refcount=0;modified=0;rawmodified=0;impulses=0;total=0;denfiles=0;faultvalues=0
    for ix,r in enumerate(inventory):
        sid,side=r['session'],r['side'];src=inside(root,r['relative_raw_csv']);assert sha(src)==r['raw_csv_sha256']
        with src.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        assert fields and all(str(c) in fields for c in range(1,49))
        x=np.array([[float(a[str(c)]) for c in range(1,49)] for a in rows],dtype=float).reshape(-1,48);n=len(x)
        assert n==int(r['rows'])
        # v3: fault-cell burst repair on raw rows, before any baseline estimation.
        xraw=x
        x,burstmask,(fr,fc,fb,fa)=repair_fault_bursts(xraw)
        faultcells=','.join(map(str,sorted({int(c) for c in fc}))) if len(fr) else ''
        faultmax=float(fb.max()) if len(fr) else float('nan')
        native=np.array([float(a['t_us'])/1e6 for a in rows],float) if 't_us' in fields else np.full(n,np.nan)
        epoch=np.nan
        if r['relative_metadata']:
            metap=inside(root,r['relative_metadata']);assert sha(metap)==r['metadata_sha256']
            iso=json.loads(metap.read_text()).get('started_at_iso','')
            if iso:
                date=datetime.datetime.fromisoformat(iso);epoch=date.hour*3600+date.minute*60+date.second+date.microsecond/1e6
        wall=native+epoch if np.isfinite(epoch) else np.full(n,np.nan)
        legacy=np.zeros(48,bool);legacyids=json.loads(r['legacy_advisory_channels0'])
        assert all(0<=c<48 for c in legacyids);legacy[legacyids]=True
        q=analyze(x,native,wall,legacy)
        a=process(x,q['temporal_segment_id']);den=a['impulse_only']
        aa=anchors.get((sid,side),[])
        for anchor in aa:
            fit=np.asarray(anchor['fit_rows'],int);assert np.all((fit>=0)&(fit<n)) and q['row_alignment_eligible'][fit].all()
        constant,b,applied,b0,fit=make_baseline(x,q['temporal_segment_id'],q['row_alignment_eligible'],aa,[])
        fixed=np.maximum(den-constant,0);processed=np.maximum(den-b,0)
        assert np.isfinite(processed).all() and (processed>=0).all()
        assert np.array_equal(processed[~q['row_alignment_eligible']],den[~q['row_alignment_eligible']])
        rel=Path('dataset')/r['date']/r['subject']/sid/(side+'.npz');p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(original48=xraw,fault_repaired48=x,impulse_only48=den,denoised48=den,constant_corrected48=fixed,processed48=processed,
            estimated_baseline48=b,constant_baseline48=constant,source_row0=np.arange(n),
            native_receive_time_s=native,wall_time_day_s=wall,
            row_alignment_eligible=q['row_alignment_eligible'],row_temporal_eligible=q['row_temporal_eligible'],
            temporal_segment_id=q['temporal_segment_id'],cell_valid=q['cell_valid'],
            cell_review_flags=q['cell_review_flags'],row_quality_flags=q['row_quality_flags'],
            schema_version=np.asarray('native_faultrepair_impulse_airborne_baseline_step.v3'))
        np.savez_compressed(p,**payload)
        fl=Path('fault_cell_changes')/(sid+'_'+side+'.npz');(out/fl).parent.mkdir(exist_ok=True)
        np.savez_compressed(out/fl,row0=fr,channel=fc,before=fb,after=fa)
        ir,ic=np.nonzero(den!=x);il=Path('impulse_changes')/(sid+'_'+side+'.npz');(out/il).parent.mkdir(exist_ok=True)
        np.savez_compressed(out/il,row0=ir,channel=ic+1,before=x[ir,ic],after=den[ir,ic])
        rr,cc=np.nonzero(processed!=den);cl=Path('changes')/(sid+'_'+side+'.npz');(out/cl).parent.mkdir(exist_ok=True)
        np.savez_compressed(out/cl,row0=rr,channel=cc+1,before=den[rr,cc],after=processed[rr,cc],baseline=b[rr,cc])
        csvrel=Path('processed_csv')/r['date']/r['subject']/sid/(side+'.csv');csvout=out/csvrel;csvout.parent.mkdir(parents=True,exist_ok=True)
        if not n:shutil.copy2(src,csvout)
        else:
            for row,c in np.argwhere(xraw!=processed):rows[int(row)][str(int(c)+1)]=format(processed[row,c],'.17g')
            write(csvout,rows,fields)
        if args.verify_reference:
            expected=reference[sid,side];expectedp=inside(root/'data',expected['relative_npz'])
            assert sha(expectedp)==expected['npz_sha256']
            with np.load(expectedp,allow_pickle=False) as z:
                assert set(z.files)==set(payload),(sid,side,'keys')
                for key,v in payload.items():assert same(v,z[key]),(sid,side,key,'reference mismatch')
            assert sha(csvout)==expected['csv_sha256'],(sid,side,'CSV reference mismatch')
            refcount+=1
        if args.scan:screening.extend(scan_steps(x,native,q['temporal_segment_id'],sid,side))
        for c in np.flatnonzero(b0>0):models.append(dict(session=sid,side=side,channel=int(c+1),baseline=float(b0[c]),fit_rows='|'.join(map(str,fit)),evidence_tiers='|'.join(sorted({a['tier'] for a in aa}))))
        steps.extend(applied)
        modified+=len(rr);rawmodified+=np.count_nonzero(xraw!=processed);total+=x.size;impulses+=np.count_nonzero(a['impulse_only']!=x);denfiles+=bool(np.any(x!=den));faultvalues+=len(fr)
        files.append(dict(session=sid,side=side,date=r['date'],subject=r['subject'],action=r['action'],trial=r['trial'],rows=n,
            baseline_channels=int((b0>0).sum()),baseline_changed_values=len(rr),total_changed_from_raw=int(np.count_nonzero(xraw!=processed)),
            modeled_step_boundaries=len(applied),fault_burst_values=len(fr),fault_cells=faultcells,fault_raw_max=faultmax,
            relative_npz=str(rel),npz_sha256=sha(p),relative_csv=str(csvrel),csv_sha256=sha(csvout),
            relative_changes=str(cl),changes_sha256=sha(out/cl),relative_impulse_changes=str(il),impulse_changes_sha256=sha(out/il),
            relative_fault_cell_changes=str(fl),fault_cell_changes_sha256=sha(out/fl)))
        if (ix+1)%100==0:print('rebuilt foot files',ix+1,flush=True)
    write(out/'file_manifest.csv',files)
    write(out/'channel_baselines.csv',models,fields=list(models[0]) if models else ['session','side','channel','baseline'])
    write(out/'applied_step_models.csv',steps,fields=list(steps[0]) if steps else ['session','side','channel','boundary_row0'])
    if args.scan:write(out/'step_candidates.csv',screening,fields=list(screening[0]) if screening else ['session','side','channel','boundary_row0'])
    receipt=dict(status='PASS',raw_foot_files=len(files),recordings=len({r['session'] for r in files}),pressure_values=int(total),
        denoised_files=int(denfiles),isolated_impulse_modified_values=int(impulses),baseline_channels=len(models),
        baseline_corrected_files=sum(r['baseline_changed_values']>0 for r in files),baseline_modified_values=int(modified),
        fault_burst_modified_values=int(faultvalues),fault_affected_files=sum(r['fault_burst_values']>0 for r in files),
        total_changed_from_raw=int(rawmodified),supported_airborne_steps=len(steps),
        reference_files_exact_arrays_and_csv=int(refcount),full_scan_enabled=args.scan,screen_candidates=len(screening),
        no_source_files_written=True,external_server_paths_required=False,raw_meta_and_frozen_calibration_inputs_included=True,
        numpy_version=np.__version__,physical_zero_certified=False,adaptive_smoothing_enabled=False,adaptive_smoothing_modified_values=0)
    (out/'rebuild_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
