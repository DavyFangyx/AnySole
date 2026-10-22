"""Immutable RGB-only interventions shared by all frozen A2 model runs."""
from __future__ import annotations

import numpy as np

from .a2eval_metrics import array_sha, digest

KINDS = ('correct','repeat-first','time-shuffle','wrong-recording')


def build_manifest(units, *, seed=1, window=20, stride=10):
    """No labels accepted: donor matching reads recording/camera/action only.

    Donors must differ in recording. Prefer same camera+action, then same camera,
    then same action, then any other recording. Equal candidates sort by ID.
    A donor's valid RGB rows are linearly rank-mapped to recipient valid rows.
    """
    entries=[]
    for key,u in sorted(units.items()):
        n=len(u['valid']);take=np.flatnonzero(u['valid'])
        if u['features'].shape!=(n,2051):
            raise ValueError('expected features [time,2051]')
        rng=np.random.default_rng(int(digest({'seed':seed,'unit':key})[:16],16))
        shuffle=take.copy();rng.shuffle(shuffle)
        donors=[(k,v) for k,v in units.items() if v['recording']!=u['recording'] and np.any(v['valid'])]
        donors.sort(key=lambda kv:(kv[1]['camera']!=u['camera'],kv[1]['action']!=u['action'],kv[0]))
        donor_key,donor=donors[0] if donors else (None,None)
        donor_rows=np.flatnonzero(donor['valid']) if donor is not None else np.array([],int)
        mapping=donor_rows[np.rint(np.linspace(0,len(donor_rows)-1,len(take))).astype(int)] if len(take) and len(donor_rows) else []
        starts=list(range(0,max(n-window+1,0),stride));tail=max(n-window,0)
        if not starts or starts[-1]!=tail:starts.append(tail)
        entries.append({'unit':key,'recording':u['recording'],'camera':u['camera'],'action':u['action'],
            'valid_rows':take.tolist(),'first_valid_row':int(take[0]) if len(take) else None,
            'shuffle_rows':shuffle.tolist(),'donor_unit':donor_key,'donor_rows':np.asarray(mapping).tolist(),
            'valid_sha256':array_sha(np.asarray(u['valid'],bool)),'time_sha256':array_sha(u['time_s']),
            'frame_sha256':array_sha(u['frame_id']),'features_sha256':array_sha(u['features']),
            'windows':[{'start':s,'stop':min(s+window,n)} for s in starts],
            'wrong_status':'available' if donor_key else 'N/A: no other recording with valid RGB'})
    result={'schema':'anysole.a2.rgb-interventions.v1','seed':seed,'window':window,'stride':stride,
        'policy':'recording-level fixed valid-position permutation and donor mapping; overlapping windows share exact RGB; labels/time/valid unchanged',
        'entries':entries}
    result['manifest_sha256']=digest(result)
    return result


def apply_intervention(units, manifest, kind):
    if kind not in KINDS:raise ValueError('unknown RGB intervention')
    result={}
    for entry in manifest['entries']:
        key=entry['unit'];u=units[key]
        for field,value in (('valid_sha256',np.asarray(u['valid'],bool)),('time_sha256',u['time_s']),
                            ('frame_sha256',u['frame_id']),('features_sha256',u['features'])):
            if array_sha(value)!=entry[field]:raise ValueError('intervention source identity changed')
        x=np.array(u['features'],copy=True);take=np.asarray(entry['valid_rows'],int)
        if kind=='repeat-first' and len(take):x[take]=x[entry['first_valid_row']]
        elif kind=='time-shuffle':x[take]=x[np.asarray(entry['shuffle_rows'],int)]
        elif kind=='wrong-recording':
            donor=entry['donor_unit']
            if donor is None:
                result[key]=None
                continue
            x[take]=units[donor]['features'][np.asarray(entry['donor_rows'],int)]
        result[key]=x
    return result


def dataset_units(dataset,*,cameras=None):
    units={}
    for record in dataset.records:
        n=len(record['target_frame_id'])
        for view in record['views']:
            if view['camera_id'] not in (dataset.cameras if cameras is None else cameras):continue
            source,valid=dataset._load_features(record,view)
            x=np.zeros((n,2051),np.float32);rows=np.asarray(view['source_row'],int)
            x[valid]=source[rows[valid]]
            key=record['recording_id']+f"/cam{view['camera_id']}"
            units[key]={'features':x,'valid':valid,'recording':record['recording_id'],'camera':view['camera_id'],
                'action':record['action'],'time_s':np.asarray(record['target_time_s'],np.float64),
                'frame_id':np.asarray(record['target_frame_id'],np.int64)}
    return units


class IntervenedDataset:
    """Replace RGB tensors after ordinary dataset loading; metadata untouched."""
    def __init__(self,dataset,features):self.dataset,self.features=dataset,features
    def __getattr__(self,name):return getattr(self.dataset,name)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        import torch
        item=self.dataset[index]
        ri,start,end,cameras=self.dataset.windows[index]
        rid=self.dataset.records[ri]['recording_id']
        visual=dict(item['visual']);value=visual['V_feat'].clone()
        for c,cam in enumerate(cameras):
            transformed=self.features[rid+f'/cam{cam}']
            if transformed is None:raise ValueError('wrong-recording donor unavailable; declare subset before inference')
            value[c,:end-start]=torch.from_numpy(transformed[start:end])
        visual['V_feat']=value
        return {**item,'visual':visual}
