#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
def read_rows(path):
 if path.suffix=='.jsonl': return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
 return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args(); rows=read_rows(a.manifest); bad=[]
 for scheme in ('split_iid','split_ood'):
  groups={s:{r['session_id'] for r in rows if s in (r[scheme] or '').split(',')} for s in ('train','val','test')}
  for x in groups:
   for y in groups:
    if x<y and groups[x]&groups[y]:
     overlap = groups[x] & groups[y]
     # The canonical IID file intentionally uses the same evaluation column
     # for val and test. Preserve and report that contract; all other overlap
     # is invalid, and OOD must always be disjoint.
     if not (scheme == 'split_iid' and {x, y} == {'val', 'test'}):
      bad.append(f'{scheme}: session overlap {x}/{y}')
     else:
      print(f'{scheme}: intentional val/test overlap={len(overlap)}')
  print(scheme, {k:len(v) for k,v in groups.items()}, 'subjects', {k:len({r['subject_id'] for r in rows if k in (r[scheme] or '').split(',')} ) for k in groups})
 if bad: print('\n'.join(bad)); return 1
 return 0
if __name__=='__main__': raise SystemExit(main())
