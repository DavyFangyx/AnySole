"""Short public outputs; private caches survive until verified publication."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .manifest_index import file_sha256
from .runner import _atomic_write_json, unit_root_for

LAYOUT = 'compact-v1'


def short_name(row):
    date, method, session, camera = row['unit_id'].split('/')
    methods = {'method_1_offset': 'm1', 'method_2_projection': 'm2'}
    if not re.fullmatch(r'\d{8}', date) or method not in methods or not re.fullmatch(r'S\d+', session):
        raise ValueError('unsupported compact unit identity')
    match = re.fullmatch(r'camera_([1-9]\d*)', camera)
    if not match:
        raise ValueError('unsupported compact camera identity')
    return f'{date}_{methods[method]}_{session}_c{int(match[1])}'


def configure(args):
    if getattr(args, 'compact_output', False) and not getattr(args, '_compact_configured', False):
        args.public_root = str(Path(args.result_root).resolve())
        args.result_root = str(Path(args.public_root) / '.run')
        args._compact_configured = True


def active(root):
    marker = Path(root) / 'layout.json'
    return marker.is_file() and json.loads(marker.read_text()).get('layout') == LAYOUT


def directory(root, row):
    root = Path(root).resolve()
    if root.name != '.run':
        raise ValueError('compact work root must be .run')
    return root.parent / short_name(row)


def state(root, row):
    path = directory(root, row) / 'state.json'
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.partial')
    with source.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    if file_sha256(source) != file_sha256(temporary):
        raise ValueError('compact copy hash mismatch')
    os.replace(temporary, destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inference_valid(root, row, digest, mock=False):
    record = state(root, row)
    if not record or record.get('config_hash') != digest or record.get('unit_id') != row['unit_id']:
        return False
    try:
        evidence = record['inference']
        p = directory(root, row) / 'result.pkl'
        if evidence['status'] != 'succeeded' or evidence['frames'] != row['frame_count'] or file_sha256(p) != evidence['sha256']:
            return False
        if mock:
            return True
        from .validation import validate_prediction
        validate_prediction(p, row['frame_count'])
        return (evidence['track_id'] == 0 and evidence['tracking_verified'] and
                evidence['roi_frames'] == row['frame_count'] and evidence['roi_sha256'] and
                '--target_bboxes' in evidence['command'] and
                '--estimate_local_only' in evidence['command'] and '--calib' not in evidence['command'])
    except (KeyError, OSError, ValueError, TypeError):
        return False


def visualization_valid(root, row, digest, mock=False):
    if not inference_valid(root, row, digest, mock):
        return False
    try:
        record = state(root, row)['visualization']
        p = directory(root, row) / 'overlay.mp4'
        if record['status'] != 'succeeded' or file_sha256(p) != record['sha256']:
            return False
        if mock:
            return True
        from .overlay import PROJECTION, verify_video
        return (record['projection'] == PROJECTION and record['joint_count'] == 24 and
                record['video']['frames'] == row['frame_count'] and
                record['prediction_sha256'] == state(root, row)['inference']['sha256'] and
                verify_video(p, record['video']) == record['video'])
    except (KeyError, OSError, ValueError, TypeError):
        return False


def publish_inference(args, row, digest):
    root = Path(args.result_root)
    work = unit_root_for(root, row, row['unit_index'])
    destination = directory(root, row)
    from .validation import prediction_path
    source = work / 'wham_output/mock_result.json' if args.mock else prediction_path(work)
    atomic_copy(source, destination / 'result.pkl')
    terminal = json.loads((work / 'status.json').read_text())
    evidence = dict(status='succeeded', frames=row['frame_count'], sha256=file_sha256(destination / 'result.pkl'),
                    attempt=terminal.get('attempt', 1))
    if not args.mock:
        roi = json.loads((work / 'input/target_bboxes.json').read_text())
        evidence.update(track_id=0, tracking_verified=True, roi_frames=roi['frame_count'],
                        roi_sha256=file_sha256(work / 'input/target_bboxes.json'),
                        command=json.loads((work / 'command.json').read_text())['command'])
    record = dict(layout=LAYOUT, unit_id=row['unit_id'], unit_index=row['unit_index'],
                  config_hash=digest, inference=evidence, visualization=dict(status='pending'))
    _atomic_write_json(destination / 'state.json', record)


def publish_visualization(args, row, digest):
    root = Path(args.result_root)
    work = unit_root_for(root, row, row['unit_index'])
    destination = directory(root, row)
    if not inference_valid(root, row, digest, args.mock):
        publish_inference(args, row, digest)
    source = work / 'visualization' / ('mock-overlay.json' if args.mock else 'smpl_joints_overlay.mp4')
    atomic_copy(source, destination / 'overlay.mp4')
    record = state(root, row)
    evidence = dict(status='succeeded', sha256=file_sha256(destination / 'overlay.mp4'))
    if not args.mock:
        meta = json.loads((work / 'visualization/metadata.json').read_text())
        evidence.update(video=meta['video'], joint_count=meta['joint_count'],
                        projection=meta['projection_policy'], prediction_sha256=meta['prediction_sha256'],
                        input_video_sha256=meta['video_sha256'])
    record['visualization'] = evidence
    _atomic_write_json(destination / 'state.json', record)
    if not visualization_valid(root, row, digest, args.mock):
        raise ValueError('published compact output validation failed; caches preserved')
    # Delete only this new layout's generated, verified unit cache. Never touch
    # source images, prior legacy runs, failures, or another unit's directory.
    expected = (root / 'units' / work.name).resolve()
    if work.is_symlink() or work.resolve() != expected or root not in expected.parents:
        raise ValueError('unsafe compact cache path')
    shutil.rmtree(work)


def failed(args, row, digest, outcome):
    record = state(args.result_root,row) or dict(layout=LAYOUT, unit_id=row['unit_id'],
             unit_index=row['unit_index'], config_hash=digest,
             inference=dict(status='pending'), visualization=dict(status='pending'))
    record[outcome['stage']] = dict(status='failed', attempt=outcome.get('attempt'),
                                  error=outcome.get('error'), failure_kind=outcome.get('kind'))
    _atomic_write_json(directory(args.result_root,row)/'state.json',record)


def verify_roi(args,row):
    from .constraints import write_target_bbox_sidecar
    from .manifest_index import read_unit
    unit = read_unit(args.manifest,row['unit_index'])
    with tempfile.TemporaryDirectory(prefix='.roi-audit-',dir=args.result_root) as folder:
        path = write_target_bbox_sidecar(args.annotation_index,unit,folder)
        return file_sha256(path) == state(args.result_root,row)['inference']['roi_sha256']


def tidy(root):
    root = Path(root).resolve()
    units = root/'units'
    if units.is_dir() and not any(units.iterdir()):
        units.rmdir()
    configs = root/'.worker-config'
    if configs.is_dir() and not configs.is_symlink() and configs.resolve().parent == root:
        shutil.rmtree(configs)


def prepare_overlay(args, row):
    """Recreate input video after cache cleanup, without rerunning WHAM."""
    from .manifest_index import unit_manifest
    from .dataset import unit_at
    from .runner import prepare_wham_video
    work = unit_root_for(args.result_root, row, row['unit_index'])
    video = work / 'input/sequence.mp4'
    if not video.exists():
        manifest = unit_manifest(args.manifest, row['unit_index'])
        unit = unit_at(manifest, row['unit_index'])
        prepare_wham_video(manifest, unit, work, fps=args.fps, ffmpeg_bin=args.ffmpeg_bin,
                           log_path=work / 'run.log')
    return video, directory(args.result_root, row) / 'result.pkl'
