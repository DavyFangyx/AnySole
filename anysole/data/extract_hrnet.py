"""Extract per-frame CLIFF-HRNet features and bbox info into AnySole caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from anysole.data.crop import CROP_IMG_HEIGHT, CROP_IMG_WIDTH, process_image
from anysole.data.dataset import find_session_dir, hrnet_cache_path
from anysole.data.workspace_adapter import load_shared_session, write_cache_provenance
from anysole.types import (
    CLIFF_CKPT,
    HRNET_YAML,
    MOTIONPRO_ROOT,
    HRNET_CACHE_ROOT,
    SEQ_ROOT,
    V_FEAT_DIM,
    WORKSPACE_ROOT,
)

HRNET_FEAT_DIM = 2048
BBOX_INFO_DIM = 3
CLIFF_FOCAL_LENGTH = 608.0

ENCODER_PREFIX = "module.encoder."


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2051-d AnySole visual features (HRNet 2048 + CLIFF bbox_info 3)."
    )
    parser.add_argument("--cam-id", type=int, default=3)
    parser.add_argument("--grouped", action="store_true", help="Extract real RGB sources from the grouped manifest")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--hrnet-module-root", type=Path)
    parser.add_argument("--hrnet-config", type=Path)
    parser.add_argument("--hrnet-checkpoint", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--cameras", default="1,2,3,4")
    parser.add_argument("--recording-list", type=Path, help="Explicit JSON list of date/session IDs for one shard")
    parser.add_argument("--index-output", type=Path, help="Independent shard index; defaults to the single-writer main index")
    parser.add_argument("--worker-state", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--merge-shards", nargs="+", type=Path, help="Validate complete shards and atomically publish main index")
    parser.add_argument("--session", type=str, default=None, help="Only extract this session id.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"],
                        help="Split column scope when --session is empty ('all' = every session under the seq root).")
    parser.add_argument("--split-csv", type=str, default=str(WORKSPACE_ROOT / "protocol/splits/default/splits.csv"),
                        help="splits.csv path used by --split.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions whose cache exists.")
    mode.add_argument("--force", "--overwrite", dest="force", action="store_true", help="Recompute and overwrite existing caches.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--seq-root", type=Path, default=None)
    parser.add_argument("--bbox-root", type=Path, default=None,
                        help="Optional private/shared frontend bbox root; defaults to the session dir.")
    parser.add_argument("--occlude-lower", type=float, default=None,
                        help="Occlude the bottom fraction (0-1) of the person bbox before HRNet "
                             "(V-degradation experiment); default cache root becomes "
                             "cam{id}_occ_lower{FFF}.")
    parser.add_argument("--occlude-fill", choices=("black", "blur"), default="black",
                        help="Fill for the occluded strip (default black).")
    parser.add_argument("--limit-sessions", type=int, default=None)
    return parser.parse_args(argv)


def occlusion_cache_dirname(cam_id: int, frac: float) -> str:
    """Cache dir name for one occlusion level: cam{id}_occ_lower{FFF} with
    FFF = round(frac * 100) zero-padded (0.1 -> cam3_occ_lower010)."""
    return "cam%d_occ_lower%03d" % (int(cam_id), int(round(frac * 100)))


def resolve_roots(
    cam_id: int,
    cache_root: Optional[Path] = None,
    seq_root: Optional[Path] = None,
    occlude_lower: Optional[float] = None,
) -> tuple:
    resolved_seq_root = seq_root or SEQ_ROOT.parent / ("cam%d" % int(cam_id))
    if cache_root is not None:
        resolved_cache_root = cache_root
    elif occlude_lower is not None:
        resolved_cache_root = HRNET_CACHE_ROOT.parent / occlusion_cache_dirname(cam_id, occlude_lower)
    else:
        resolved_cache_root = HRNET_CACHE_ROOT.parent / ("cam%d" % int(cam_id))
    return resolved_seq_root, resolved_cache_root


def list_session_dirs(seq_root: Path, session_id: Optional[str]) -> List[Path]:
    if session_id:
        return [find_session_dir(seq_root, session_id)]
    dirs = sorted(path for path in seq_root.glob("*/*/*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError("No sequence dirs under %s" % seq_root)
    return dirs


def split_session_ids(split_csv: Path, split: str) -> set:
    """Session ids of one splits.csv column (train/val/test)."""
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[split].strip() for row in rows if row[split].strip()}


def select_session_dirs(seq_root: Path, args: argparse.Namespace) -> List[Path]:
    """Session dirs for the requested scope: explicit ``--session`` wins,
    else the ``--split`` column filters (default 'all' = every session)."""
    dirs = list_session_dirs(seq_root, args.session)
    if not args.session and args.split != "all":
        wanted = split_session_ids(Path(args.split_csv), args.split)
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def load_encoder(device: torch.device, module_root: Optional[Path] = None,
                 config_path: Optional[Path] = None,
                 checkpoint_path: Optional[Path] = None) -> torch.nn.Module:
    motionpro_root = str(module_root or MOTIONPRO_ROOT)
    if motionpro_root not in sys.path:
        sys.path.insert(0, motionpro_root)

    from lib.model.backbone.hrnet.cls_hrnet import HighResolutionNet
    from lib.model.backbone.hrnet.hrnet_config import cfg
    from lib.model.backbone.hrnet.hrnet_config import update_config

    cfg = cfg.clone()
    update_config(cfg, str(config_path or HRNET_YAML))
    model = HighResolutionNet(cfg)
    ckpt = torch.load(str(checkpoint_path or CLIFF_CKPT), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    enc_state = {}
    for key, value in state.items():
        for prefix in (ENCODER_PREFIX, "encoder."):
            if key.startswith(prefix):
                enc_state[key[len(prefix):]] = value
                break
    if not enc_state:
        raise KeyError("CLIFF checkpoint missing %s weights: %s" % (ENCODER_PREFIX, CLIFF_CKPT))
    model.load_state_dict(enc_state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


class RGBFeatureFrontend:
    """The same RGB-only detector/crop/HRNet path for extraction and inference."""

    def __init__(self, encoder: torch.nn.Module, detector, device: torch.device):
        self.encoder, self.detector, self.device = encoder, detector, device
        self.previous_box = None

    def reset(self):
        self.previous_box = None

    def select_box(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        result = self.detector.predict(source=rgb[:, :, ::-1], classes=[0], conf=.25,
                                       verbose=False, device=str(self.device))[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        if not len(boxes):
            return None
        area = np.maximum(boxes[:,2]-boxes[:,0],0)*np.maximum(boxes[:,3]-boxes[:,1],0)
        score = np.zeros(len(boxes))
        if self.previous_box is not None:
            prev=self.previous_box
            intersection=np.maximum(np.minimum(boxes[:,2:],prev[2:])-np.maximum(boxes[:,:2],prev[:2]),0).prod(axis=1)
            union=area+(prev[2]-prev[0])*(prev[3]-prev[1])-intersection
            score=intersection/np.maximum(union,1e-6)
        # Track by IoU when possible; otherwise largest person, confidence,
        # then lexicographic box coordinates. This rule never reads GT.
        tracked=score.max()>=.1
        candidates=range(len(boxes))
        chosen=min(candidates,key=lambda i: (-(score[i] if tracked else area[i]),
                                             -confidence[i],*boxes[i].tolist()))
        self.previous_box=boxes[chosen].astype(np.float32)
        return self.previous_box.copy()

    def prepare(self, rgb: np.ndarray):
        box=self.select_box(rgb)
        if box is None:
            return None
        norm,center,scale,ul,br,_=process_image(rgb,box,crop_height=CROP_IMG_HEIGHT,crop_width=CROP_IMG_WIDTH)
        center=np.asarray(center,np.float32)
        height,width=rgb.shape[:2]
        size=np.float32(scale)*200
        bbox_info=np.asarray([(center[0]-width/2)/CLIFF_FOCAL_LENGTH*2.8,
                              (center[1]-height/2)/CLIFF_FOCAL_LENGTH*2.8,
                              (size-.24*CLIFF_FOCAL_LENGTH)/(.06*CLIFF_FOCAL_LENGTH)],np.float32)
        sx,sy=CROP_IMG_WIDTH/float(br[0]-ul[0]),CROP_IMG_HEIGHT/float(br[1]-ul[1])
        affine=[[sx,0,sx*(.5-float(ul[0]))-.5],[0,sy,sy*(.5-float(ul[1]))-.5]]
        return np.asarray(norm,np.float32),bbox_info,{"bbox_xyxy":box.tolist(),
            "center":center.tolist(),"scale":float(scale),"crop_ul":np.asarray(ul).tolist(),
            "crop_br":np.asarray(br).tolist(),"crop_affine_pixel_centers":affine,
            "image_shape":[height,width],"crop_shape":[CROP_IMG_HEIGHT,CROP_IMG_WIDTH]}

    def encode_prepared(self, prepared: list) -> torch.Tensor:
        if not prepared:
            return torch.empty((0,V_FEAT_DIM),dtype=torch.float32)
        batch=torch.from_numpy(np.stack([p[0] for p in prepared])).to(self.device)
        with torch.inference_mode():
            values=self.encoder(batch)
        if values.shape!=(len(prepared),HRNET_FEAT_DIM) or not torch.isfinite(values).all():
            raise ValueError("HRNet produced invalid features")
        return torch.cat((values.detach().cpu().float(),torch.from_numpy(np.stack([p[1] for p in prepared]))),dim=1)


def process_identity() -> dict:
    rest=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    return {"pid":os.getpid(),"pgid":os.getpgid(0),"start_time_ticks":int(rest[19])}


def process_is_alive(identity: dict) -> bool:
    try:
        rest=Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return rest[0]!='Z' and int(rest[19])==identity['start_time_ticks']
    except (FileNotFoundError,KeyError,ValueError):
        return False


def atomic_cache_json(path: Path, value: dict) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,sort_keys=True,separators=(',',':'),allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


@contextmanager
def cache_writer_lock(path: Path, owner: str):
    """Only the same owner may recover its provably dead prior process."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    token={**process_identity(),"owner":owner}
    try:
        descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        prior=json.loads(path.read_text())
        if prior.get('owner')!=owner or process_is_alive(prior):
            raise FileExistsError(f"cache writer lock has another/live owner: {path}")
        path.unlink()
        descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as handle:
        json.dump(token,handle)
    try:
        yield token
    finally:
        if path.exists() and json.loads(path.read_text())==token:
            path.unlink()


def validate_cache_entry(record: dict, view: dict, entry: dict, dataset_root: Path,
                         cache_root: Path, *, frontend_sha256: Optional[str] = None,
                         verify_images: bool = True) -> dict:
    from anysole.data.workspace_adapter import semantic_hash,sha256_file
    root=Path(cache_root).resolve();path=(root/entry['path']).resolve()
    if not path.is_relative_to(root) or sha256_file(path)!=entry['sha256']:
        raise ValueError('cache file path/hash mismatch')
    p=entry['provenance'];n=len(view['sources'])
    expected={'recording_id':record['recording_id'],'camera_id':view['camera_id'],
              'method':record['method'],'source_index_sha256':semantic_hash(view['sources']),
              'source_count':n}
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('cache source identity/row mismatch')
    if frontend_sha256 is not None and p.get('frontend_sha256')!=frontend_sha256:
        raise ValueError('cache frontend version mismatch')
    if len(p['source_image_sha256'])!=n or len(p['source_geometry'])!=n:
        raise ValueError('cache image/crop provenance count mismatch')
    obj=torch.load(path,map_location='cpu',weights_only=True)
    values,valid=obj['features'],obj['source_valid']
    if values.dtype!=torch.float32 or valid.dtype!=torch.bool or values.shape!=(n,V_FEAT_DIM) or valid.shape!=(n,):
        raise ValueError('cache feature dtype/shape mismatch')
    if not torch.isfinite(values[valid]).all() or not (values[valid].abs().sum(1)>0).all():
        raise ValueError('nonfinite or zero valid RGB feature')
    if int(valid.sum())!=p['source_valid_count']:
        raise ValueError('cache source-valid count mismatch')
    for i,source in enumerate(view['sources']):
        if verify_images and sha256_file(Path(dataset_root)/view['directory']/source['file'])!=p['source_image_sha256'][i]:
            raise ValueError('source RGB image hash changed')
        geometry=p['source_geometry'][i]
        if valid[i] and (geometry is None or geometry.get('crop_shape')!=[CROP_IMG_HEIGHT,CROP_IMG_WIDTH]):
            raise ValueError('valid feature has no matching crop provenance')
    return {'sources':n,'valid_sources':int(valid.sum()),'invalid_sources':int((~valid).sum())}


def merge_cache_shards(manifest: Path, dataset_root: Path, cache_root: Path,
                       shard_paths: list[Path], *, verify_images: bool = True) -> dict:
    """Only publish the canonical index after complete independent validation."""
    from anysole.data.workspace_adapter import load_grouped_manifest,VIEW_CACHE_SCHEMA,sha256_file
    records,meta=load_grouped_manifest(manifest)
    expected={f"{r['recording_id']}/cam{v['camera_id']}":(r,v) for r in records if r['eligible']
              for v in r['views'] if v['sources']}
    entries={};frontend=None
    for path in shard_paths:
        shard=json.loads(Path(path).read_text())
        if shard.get('schema_version')!=VIEW_CACHE_SCHEMA or shard.get('manifest_sha256')!=meta['manifest_sha256'] or not shard.get('complete'):
            raise ValueError('incomplete shard or schema/manifest mismatch')
        for key,entry in shard['entries'].items():
            if key in entries:
                raise ValueError(f'duplicate cache unit across shards: {key}')
            if key not in expected:
                raise ValueError(f'unexpected cache unit: {key}')
            current=entry['provenance'].get('frontend_sha256')
            if not current or (frontend is not None and frontend!=current):
                raise ValueError('mixed frontend versions in cache shards')
            frontend=current;entries[key]=entry
    if set(entries)!=set(expected):
        raise ValueError(f'incomplete corpus cache: missing {len(set(expected)-set(entries))} units')
    summary={'units':len(entries),'sources':0,'valid_sources':0,'invalid_sources':0,'by_camera':{}}
    for key in sorted(entries):
        r,v=expected[key]
        unit=validate_cache_entry(r,v,entries[key],dataset_root,cache_root,
                                  frontend_sha256=frontend,verify_images=verify_images)
        for name in ('sources','valid_sources','invalid_sources'):summary[name]+=unit[name]
        camera=str(v['camera_id']);row=summary['by_camera'].setdefault(camera,{'units':0,'sources':0,'valid_sources':0,'invalid_sources':0})
        row['units']+=1
        for name in ('sources','valid_sources','invalid_sources'):row[name]+=unit[name]
    index={'schema_version':VIEW_CACHE_SCHEMA,'manifest_sha256':meta['manifest_sha256'],
           'frontend_sha256':frontend,'complete':True,'coverage':summary,'entries':entries}
    path=Path(cache_root)/'index.json'
    with cache_writer_lock(Path(cache_root)/'.merge.lock','complete-corpus-merge'):
        previous=sha256_file(path) if path.exists() else None
        atomic_cache_json(path,index)
    return {'index':str(path),'index_sha256':sha256_file(path),'previous_index_sha256':previous,
            'frontend_sha256':frontend,'manifest_sha256':meta['manifest_sha256'],'coverage':summary}


def extract_grouped(config: dict, *, device: torch.device, cameras=(1,2,3,4),
                    session: Optional[str] = None, limit_sessions: Optional[int] = None,
                    batch_size: int = 32, overwrite: bool = False) -> dict:
    import inspect
    from importlib.metadata import version
    from anysole.data.workspace_adapter import (load_grouped_manifest, semantic_hash,
        sha256_file, VIEW_CACHE_SCHEMA)
    cache_root=Path(config["feature_cache_root"])
    os.environ.setdefault("YOLO_CONFIG_DIR",str(cache_root/".ultralytics"))
    from ultralytics import YOLO
    required=("hrnet_module_root","hrnet_config","hrnet_checkpoint","detector_checkpoint")
    for field in required:
        if not config.get(field) or not Path(config[field]).exists():
            raise FileNotFoundError(f"Missing actual frontend asset {field}={config.get(field)}")
    records,meta=load_grouped_manifest(Path(config["manifest_path"]))
    selected=[r for r in records if r["eligible"] and (session is None or r["session_id"]==session)]
    allowlist=config.get('cache_recording_ids')
    if allowlist is not None:
        if len(allowlist)!=len(set(allowlist)) or set(allowlist)-{r['recording_id'] for r in selected}:
            raise ValueError('duplicate/unknown recording allowlist')
        selected=[r for r in selected if r['recording_id'] in set(allowlist)]
    if limit_sessions is not None:
        selected=selected[:limit_sessions]
    if not selected:
        raise ValueError("no eligible grouped recordings selected")
    root,cache_root=Path(config["dataset_root"]),Path(config["feature_cache_root"])
    cache_root.mkdir(parents=True,exist_ok=True)
    index_path=Path(config.get('cache_index_output',cache_root/'index.json'))
    owner=str(config.get('cache_worker_id',f'single:{index_path.resolve()}'))
    state_path=Path(config['cache_worker_state']) if config.get('cache_worker_state') else None
    code_hash=sha256_file(Path(__file__))
    if config.get('cache_expected_code_sha256',code_hash)!=code_hash:
        raise ValueError('extractor changed after cache plan freeze')
    state={**process_identity(),'owner':owner,'device':str(device),'status':'running',
           'manifest_sha256':meta['manifest_sha256'],'extractor_sha256':code_hash,
           'total_units':sum(bool(v['sources']) and v['camera_id'] in cameras for r in selected for v in r['views']),
           'total_sources':sum(len(v['sources']) for r in selected for v in r['views'] if v['camera_id'] in cameras),
           'completed_units':[],'failed_units':[],'completed_sources':0,'current_unit':None,'current_source_row':0,
           'attempt_sources':0,'started_utc':datetime.now(timezone.utc).isoformat()}
    started=time.monotonic();last_pulse=0.
    def pulse(force=False):
        nonlocal last_pulse
        now=time.monotonic()
        if force or now-last_pulse>=10:
            if sha256_file(Path(__file__))!=code_hash:
                raise RuntimeError('extractor changed during frozen cache execution')
            free=shutil.disk_usage(cache_root).free
            if free<int(config.get('cache_min_free_bytes',10*1024**3)):
                raise RuntimeError(f'cache disk reserve exhausted: {free}')
            state.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=now-started,
                         disk_free_bytes=free,attempt_sources_per_s=state['attempt_sources']/max(now-started,1e-6))
            if state_path:atomic_cache_json(state_path,state)
            last_pulse=now
    with cache_writer_lock(index_path.with_name(index_path.name+'.lock'),owner):
        index=json.loads(index_path.read_text()) if index_path.exists() else {"schema_version":VIEW_CACHE_SCHEMA,"entries":{}}
        if index.get('manifest_sha256',meta['manifest_sha256'])!=meta['manifest_sha256']:
            raise ValueError('resume shard manifest mismatch')
        index.update(manifest_sha256=meta['manifest_sha256'],complete=False,owner=owner)
        atomic_cache_json(index_path,index);pulse(True)
        try:
            return _extract_grouped_units(config,device,cameras,batch_size,overwrite,selected,root,cache_root,
                                          index_path,index,state,pulse,code_hash)
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}',exit_code=1)
            if state['current_unit']:state['failed_units'].append(state['current_unit'])
            if state_path:atomic_cache_json(state_path,state)
            raise


def _extract_grouped_units(config,device,cameras,batch_size,overwrite,selected,root,cache_root,
                           index_path,index,state,pulse,code_hash):
        import inspect
        from importlib.metadata import version
        from ultralytics import YOLO
        from anysole.data.workspace_adapter import semantic_hash,sha256_file,VIEW_CACHE_SCHEMA
        encoder=load_encoder(device,Path(config["hrnet_module_root"]),Path(config["hrnet_config"]),Path(config["hrnet_checkpoint"]))
        frontend=RGBFeatureFrontend(encoder,YOLO(str(config["detector_checkpoint"])),device)
        asset_hashes={k:sha256_file(Path(config[k])) for k in ("hrnet_config","hrnet_checkpoint","detector_checkpoint")}
        module_dir=Path(inspect.getfile(type(encoder))).parent
        implementation={str(p.relative_to(module_dir)):sha256_file(p) for p in sorted(module_dir.rglob('*.py'))}
        frontend_spec={"schema_version":VIEW_CACHE_SCHEMA,"asset_hashes":asset_hashes,
                    "extractor_sha256":code_hash,"crop_source_sha256":sha256_file(Path(__file__).with_name('crop.py')),
                    "hrnet_source_commit":config.get("hrnet_source_commit"),"hrnet_source_files":implementation,
                    "frontend_versions":{"torch":str(torch.__version__),"numpy":np.__version__,
                                         "opencv":cv2.__version__,"ultralytics":version('ultralytics')},
                    "encoder_batch_size":batch_size,"feature_dtype":"float32",
                    "numeric_backend":{"cudnn_benchmark":torch.backends.cudnn.benchmark,
                                       "cudnn_deterministic":torch.backends.cudnn.deterministic,
                                       "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
                                       "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32},
                    "crop_shape":[CROP_IMG_HEIGHT,CROP_IMG_WIDTH],"bbox_info_focal_length":CLIFF_FOCAL_LENGTH,
                    "person_selection":"IoU>=0.1 else largest area; confidence and box tie breaks"}
        frontend_digest=semantic_hash(frontend_spec)
        index['frontend_sha256']=frontend_digest;state['frontend_sha256']=frontend_digest;pulse(True)
        for record in selected:
            for view in record["views"]:
                if view["camera_id"] not in cameras or not view["sources"]:
                    continue
                key=f"{record['recording_id']}/cam{view['camera_id']}"
                state.update(current_unit=key,current_source_row=0);pulse(True)
                source_hash=semantic_hash(view["sources"])
                identity={"schema_version":VIEW_CACHE_SCHEMA,"recording_id":record["recording_id"],
                    "camera_id":view["camera_id"],"method":record["method"],"source_index_sha256":source_hash,
                    **frontend_spec,"frontend_sha256":frontend_digest}
                if key in index["entries"] and not overwrite:
                    entry=index["entries"][key]
                    if any(entry["provenance"].get(k)!=v for k,v in identity.items()):
                        raise ValueError(f"existing frontend cache differs: {key}")
                    if sha256_file(cache_root/entry["path"])!=entry["sha256"]:
                        raise ValueError(f"existing frontend cache hash mismatch: {key}")
                    state['completed_units'].append(key);state['completed_sources']+=len(view['sources']);pulse(True)
                    continue
                unit_lock=cache_writer_lock(cache_root/'.unit_locks'/f"{record['date']}_{record['session_id']}_cam{view['camera_id']}.lock",state['owner'])
                # A unit and its independent shard index have one writer.
                unit_lock.__enter__()
                try:
                    _extract_one_unit(frontend,record,view,key,identity,index,index_path,root,cache_root,
                                      batch_size,state,pulse)
                finally:
                    unit_lock.__exit__(None,None,None)
        index['complete']=True;atomic_cache_json(index_path,index)
        state.update(status='completed',current_unit=None,current_source_row=0,exit_code=0);pulse(True)
        return {"entries":len(index["entries"]),"manifest_sha256":index['manifest_sha256'],"index":str(index_path)}


def _extract_one_unit(frontend,record,view,key,identity,index,index_path,root,cache_root,batch_size,state,pulse):
                from anysole.data.workspace_adapter import semantic_hash,sha256_file
                frontend.reset()
                n=len(view["sources"])
                features=torch.zeros((n,V_FEAT_DIM),dtype=torch.float32)
                source_valid=torch.zeros(n,dtype=torch.bool)
                geometry,image_hashes,pending,rows=[],[],[],[]
                for row,source in enumerate(view["sources"]):
                    path=root/view["directory"]/source["file"]
                    image_hashes.append(sha256_file(path))
                    prepared=frontend.prepare(read_rgb(path))
                    geometry.append(None if prepared is None else prepared[2])
                    state['attempt_sources']+=1;state['current_source_row']=row+1;pulse()
                    if prepared is not None:
                        pending.append(prepared);rows.append(row)
                    if len(pending)>=batch_size or row==n-1:
                        if pending:
                            features[rows]=frontend.encode_prepared(pending)
                            source_valid[rows]=True
                        pending,rows=[],[]
                provenance={**identity,"source_image_sha256":image_hashes,"source_geometry":geometry,
                            "source_valid_count":int(source_valid.sum()),"source_count":n,
                            "quality":"observed" if source_valid.any() else "no_person_detected_all_sources"}
                digest=semantic_hash(provenance)
                name=f"{record['date']}_{record['session_id']}_cam{view['camera_id']}_{digest[:12]}.pt"
                target=cache_root/name
                temporary=target.with_suffix('.tmp')
                torch.save({"features":features,"source_valid":source_valid},temporary)
                os.replace(temporary,target)
                index["entries"][key]={"path":name,"sha256":sha256_file(target),"provenance":provenance}
                atomic_cache_json(index_path,index)
                state['completed_units'].append(key);state['completed_sources']+=n;pulse(True)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Failed to read image %s" % path)
    return bgr[:, :, ::-1]


def occlude_lower_strip(rgb: np.ndarray, box, frac: float, fill: str) -> np.ndarray:
    """Mask the full-width rows at/under (1 - frac) of the person bbox height.

    The mask is anchored to the per-frame bbox (the same box that drives
    ``process_image``); bbox and bbox_info stay untouched on purpose -- the
    person is still detected, only the lower body is unreadable.
    """
    y1, y2 = float(box[1]), float(box[3])
    cut = int(round(y1 + (1.0 - frac) * (y2 - y1)))
    if cut <= 0 or cut >= rgb.shape[0] - 1:
        return rgb
    out = rgb.copy()
    if fill == "black":
        out[cut:] = 0
    else:  # blur: keep local color statistics, remove texture
        out[cut:] = cv2.GaussianBlur(rgb, (0, 0), 24)[cut:]
    return out


def session_frame_paths(seq_dir: Path, n_frames: int) -> List[Path]:
    color_dir = seq_dir / ("rgb" if (seq_dir / "rgb").is_dir() else "color")
    paths = [color_dir / ("%06d.jpg" % idx) for idx in range(n_frames)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "%s expected %d JPGs under color/; missing e.g. %s" % (seq_dir, n_frames, missing[0])
        )
    return paths


def extract_session(
    seq_dir: Path,
    cache_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    bbox_root: Optional[Path] = None,
    occlude_lower: Optional[float] = None,
    occlude_fill: str = "black",
) -> None:
    if (seq_dir / "session.json").is_file():
        meta = json.loads((seq_dir / "session.json").read_text())
        n_frames = int(meta["frame_count"])
    else:
        meta = json.loads((seq_dir / "align_meta.json").read_text())
        n_frames = int(meta["n_frames"])
    image_paths = session_frame_paths(seq_dir, n_frames)
    bbox_path = (bbox_root / f"{seq_dir.name}.npy") if bbox_root else seq_dir / "bbox.npy"
    if not bbox_path.is_file():
        raise FileNotFoundError(f"Missing AnySole bbox frontend {bbox_path}")
    bbox = np.load(bbox_path)
    if bbox.shape[0] != n_frames:
        raise ValueError("%s bbox.npy length %s != n_frames %d" % (seq_dir, bbox.shape, n_frames))

    chunks = []
    batch = []
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            rgb = read_rgb(image_path)
            box = bbox[idx][1:5]
            if occlude_lower is not None:
                rgb = occlude_lower_strip(rgb, box, occlude_lower, occlude_fill)
            norm_img, center, scale, _, _, _ = process_image(
                rgb, box, crop_height=CROP_IMG_HEIGHT, crop_width=CROP_IMG_WIDTH
            )
            norm_img = np.asarray(norm_img, dtype=np.float32)
            if tuple(norm_img.shape) != (3, CROP_IMG_HEIGHT, CROP_IMG_WIDTH):
                raise ValueError(
                    "%s frame %d crop shape %s != (3, %d, %d)"
                    % (seq_dir.name, idx, norm_img.shape, CROP_IMG_HEIGHT, CROP_IMG_WIDTH)
                )
            img_height, img_width = rgb.shape[:2]
            center = np.asarray(center, dtype=np.float32)
            bbox_size = np.float32(scale) * 200.0
            bbox_info = np.asarray(
                [
                    (center[0] - img_width / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                    (center[1] - img_height / 2.0) / CLIFF_FOCAL_LENGTH * 2.8,
                    (bbox_size - 0.24 * CLIFF_FOCAL_LENGTH)
                    / (0.06 * CLIFF_FOCAL_LENGTH),
                ],
                dtype=np.float32,
            )
            batch.append((norm_img, bbox_info))
            if len(batch) < batch_size and idx + 1 < n_frames:
                continue
            x = torch.from_numpy(np.stack([item[0] for item in batch], axis=0)).to(
                device=device, dtype=torch.float32
            )
            xf = model(x)
            if xf.ndim != 2 or xf.shape[1] != HRNET_FEAT_DIM:
                raise ValueError(
                    "HRNet output shape %s != (B, %d)" % (tuple(xf.shape), HRNET_FEAT_DIM)
                )
            bbox_info = torch.from_numpy(np.stack([item[1] for item in batch], axis=0))
            features = torch.cat(
                [xf.detach().cpu().to(dtype=torch.float32), bbox_info.float()], dim=1
            )
            if features.shape[1] != V_FEAT_DIM:
                raise ValueError(
                    "AnySole visual feature shape %s != (B, %d)"
                    % (tuple(features.shape), V_FEAT_DIM)
                )
            chunks.append(features)
            batch = []

    feat = torch.cat(chunks, dim=0)
    if tuple(feat.shape) != (n_frames, V_FEAT_DIM):
        raise ValueError(
            "%s feature shape %s != (%d, %d)" % (seq_dir.name, tuple(feat.shape), n_frames, V_FEAT_DIM)
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat.contiguous(), cache_path)
    if (seq_dir / "session.json").is_file():
        parameters = {"frontend": "CLIFF-HRNet", "feature_dim": V_FEAT_DIM}
        if occlude_lower is not None:
            parameters["occlude_lower"] = occlude_lower
            parameters["occlude_fill"] = occlude_fill
        write_cache_provenance(
            cache_path,
            session=load_shared_session(seq_dir.name, root=seq_dir.parents[2]),
            kind="extract_hrnet",
            shape=tuple(feat.shape),
            parameters=parameters,
        )
    print("saved %s %s" % (cache_path, tuple(feat.shape)))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.grouped:
        from anysole.data.workspace_adapter import read_view_config
        config=read_view_config(args.config) if args.config else {}
        for field in ("manifest","dataset_root","cache_root","hrnet_module_root","hrnet_config",
                      "hrnet_checkpoint","detector_checkpoint"):
            value=getattr(args,field,None)
            if value is not None:
                config[{"manifest":"manifest_path","cache_root":"feature_cache_root"}.get(field,field)]=str(value)
        if args.merge_shards:
            result=merge_cache_shards(Path(config['manifest_path']),Path(config['dataset_root']),
                                      Path(config['feature_cache_root']),args.merge_shards)
            print(json.dumps(result))
            return 0
        for field,key in (("index_output","cache_index_output"),("worker_state","cache_worker_state"),
                          ("worker_id","cache_worker_id"),("expected_code_sha256","cache_expected_code_sha256")):
            value=getattr(args,field)
            if value is not None:config[key]=str(value)
        if args.recording_list:
            config['cache_recording_ids']=json.loads(args.recording_list.read_text())
        cameras=tuple(int(x) for x in args.cameras.split(','))
        if set(cameras)-{1,2,3,4}:
            raise ValueError("--cameras must contain only 1,2,3,4")
        device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        result=extract_grouped(config,device=device,cameras=cameras,session=args.session,
                               limit_sessions=args.limit_sessions,batch_size=args.batch_size,overwrite=args.force)
        print(json.dumps(result))
        return 0
    seq_root, cache_root = resolve_roots(args.cam_id, args.cache_root, args.seq_root,
                                         occlude_lower=args.occlude_lower)
    session_dirs = select_session_dirs(seq_root, args)
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_dirs = session_dirs[: args.limit_sessions]
    if not session_dirs:
        raise FileNotFoundError("No session dirs match split=%s under %s" % (args.split, seq_root))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    model = None
    for seq_dir in session_dirs:
        cache_path = hrnet_cache_path(seq_dir.name, cache_root)
        if cache_path.is_file() and not args.force:
            print("skip %s (%s exists)" % (seq_dir.name, cache_path))
            continue
        if model is None:
            print("load encoder on %s" % device)
            model = load_encoder(device)
        print("extract %s" % seq_dir)
        extract_session(seq_dir, cache_path, model, device, int(args.batch_size), args.bbox_root,
                        args.occlude_lower, args.occlude_fill)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
