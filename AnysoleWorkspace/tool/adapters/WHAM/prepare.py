"""Rebuild clean-start control inputs only; never launch inference or delete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3

from .compact import short_name
from .constraints import build_annotation_index, read_index_metadata
from .dataset import build_inventory, write_manifest
from .manifest_index import build_index, file_sha256, load_index, read_unit
from .pipeline import fingerprint, parser as pipeline_parser, plan, root_check
from .runner import _atomic_write_json


def coverage(manifest, index, idx):
    frames = 0
    with sqlite3.connect(f'file:{Path(index).resolve()}?mode=ro',uri=True) as db:
        for row in idx['units']:
            unit = read_unit(manifest,row['unit_index'],idx)
            ids = [int(f['frame']) for f in unit['frames']]
            if len(set(ids)) != len(ids):
                raise ValueError('duplicate source frames: '+row['unit_id'])
            found = set()
            for start in range(0,len(ids),900):
                batch = ids[start:start+900]
                marks = ','.join('?' for _ in batch)
                found.update(r[0] for r in db.execute(
                    f'SELECT frame FROM annotations WHERE date=? AND session=? AND camera=? AND frame IN ({marks})',
                    (unit['date'],unit['session'],int(unit['camera'].split('_')[1]),*batch)))
            if found != set(ids):
                raise ValueError('missing target annotation: '+row['unit_id'])
            frames += len(ids)
    return dict(units=len(idx['units']),frames=frames,status='passed')


def prepare(root, certificate):
    root = root_check(root)
    cert = json.loads(Path(certificate).read_text())
    if cert.get('status') != 'passed' or cert.get('output_layout') != 'compact-v1':
        raise ValueError('compact-output validation certificate is required')
    if root.exists():
        extra = [p.name for p in root.iterdir() if p.name != '.run' and
                 not re.fullmatch(r'\d{8}_m[12]_S\d+_c[1-9]\d*',p.name)]
        if extra:
            raise ValueError('result root contains legacy/unrelated files; preserve/review before cleanup: '+str(extra))
    # Check validated code/model/runtime and Parquet sources before rebuilding.
    for path,expected in {**cert['asset_hashes'],**cert['annotation_sources']}.items():
        if file_sha256(path) != expected:
            raise ValueError('validated asset/source changed: '+path)
    work = root/'.run'
    work.mkdir(parents=True,exist_ok=True)
    manifest,roi = work/'full.jsonl',work/'roi.sqlite'
    if not manifest.exists():
        inventory = build_inventory(cert['dataset_root'],strict=True)
        write_manifest(manifest,inventory)
        build_index(manifest)
    idx = load_index(manifest,verify_hash=True)
    if idx['manifest_sha256'] != cert['manifest_sha256']:
        raise ValueError('rebuilt manifest differs from validated corpus')
    if not roi.exists():
        build_annotation_index(cert['dataset_root'],roi)
    meta = read_index_metadata(roi)
    if cert.get('roi_sha256') and file_sha256(roi) != cert['roi_sha256']:
        raise ValueError('ROI index differs from validated content')
    if {r['path']:r['sha256'] for r in meta['sources']} != cert['annotation_sources']:
        raise ValueError('ROI index sources differ from validated annotations')
    args = pipeline_parser().parse_args(['full-run','--manifest',str(manifest),
                 '--result-root',str(root),'--annotation-index',str(roi),'--compact-output',
                 '--viz-workers',str(cert['viz_workers'])])
    args.gpus = list(range(7))
    run_plan = plan(args,idx)
    names = [short_name(r) for r in idx['units']]
    if len(names) != len(set(names)):
        raise ValueError('compact directory collision')
    digest,config,_ = fingerprint(args,idx)
    actual = {p:h for p,h in config['files'].items() if p != str(roi.resolve())}
    if actual != cert['asset_hashes']:
        raise ValueError('prepared runtime asset inventory differs from validation')
    old = json.loads((work/'config.json').read_text()) if (work/'config.json').exists() else None
    if old and old['config_hash'] != digest:
        raise ValueError('existing run configuration differs; do not overwrite')
    previous = json.loads((work/'ready.json').read_text()) if (work/'ready.json').exists() else None
    valid_coverage = (previous and previous.get('manifest_sha256') == idx['manifest_sha256'] and
                      previous.get('roi_sha256') == file_sha256(roi) and previous.get('coverage',{}).get('status') == 'passed')
    checked = previous['coverage'] if valid_coverage else coverage(manifest,roi,idx)
    result = dict(status='passed',config_hash=digest,manifest_sha256=idx['manifest_sha256'],
                  roi_sha256=file_sha256(roi),coverage=checked,plan=run_plan,
                  required_free_bytes=cert['required_free_bytes'],viz_workers=cert['viz_workers'],
                  inference_started=False)
    _atomic_write_json(work/'ready.json',result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-root',required=True)
    p.add_argument('--certificate',required=True)
    args = p.parse_args(argv)
    print(json.dumps(prepare(args.result_root,args.certificate),ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
