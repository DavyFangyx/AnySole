"""Atomic byte-offset JSONL index; each worker reads only its selected line."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .dataset import InventoryError, _atomic_write_bytes


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def index_path(path):
    return Path(str(path) + '.index.json')


def build_index(path):
    path = Path(path).resolve()
    original = identity(path)
    rows, seen, methods, sessions = [], set(), {}, set()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            digest.update(line)
            unit = json.loads(line)
            uid = unit['unit_id']
            if uid in seen:
                raise InventoryError('duplicate unit_id: ' + uid)
            seen.add(uid)
            n = int(unit['frame_count'])
            if n != len(unit['frames']) or n <= 0:
                raise InventoryError('invalid frame_count: ' + uid)
            rows.append(dict(unit_index=len(rows), offset=offset, length=len(line),
                             unit_id=uid, frame_count=n,
                             line_sha256=hashlib.sha256(line).hexdigest()))
            m = methods.setdefault(unit['method'], dict(units=0, frames=0))
            m['units'] += 1
            m['frames'] += n
            sessions.add((unit['date'], unit['session']))
    if identity(path) != original:
        raise InventoryError('manifest changed while indexing')
    result = dict(schema_version=1, manifest=str(path), manifest_identity=original,
                  manifest_sha256=digest.hexdigest(), units=rows,
                  unit_count=len(rows), frame_count=sum(r['frame_count'] for r in rows),
                  methods=methods, session_count=len(sessions))
    _atomic_write_bytes(index_path(path), (json.dumps(result, sort_keys=True) + '\n').encode())
    return result


def load_index(path, verify_hash=False):
    path = Path(path).resolve()
    value = json.loads(index_path(path).read_text())
    if value['manifest'] != str(path) or value['manifest_identity'] != identity(path):
        raise InventoryError('manifest/index identity mismatch')
    if verify_hash and file_sha256(path) != value['manifest_sha256']:
        raise InventoryError('manifest hash mismatch')
    return value


def read_unit(path, number, index=None):
    idx = index or load_index(path)
    if identity(path) != idx['manifest_identity']:
        raise InventoryError('manifest changed')
    if number < 0 or number >= len(idx['units']):
        raise InventoryError('unit index out of range')
    row = idx['units'][number]
    with Path(path).open('rb') as f:
        f.seek(row['offset'])
        line = f.read(row['length'])
    if hashlib.sha256(line).hexdigest() != row['line_sha256']:
        raise InventoryError('unit line hash mismatch')
    unit = json.loads(line)
    if unit['unit_id'] != row['unit_id']:
        raise InventoryError('unit identity mismatch')
    return unit


def unit_manifest(path, number):
    idx = load_index(path)
    unit = read_unit(path, number, idx)
    keys = ('schema_id', 'schema_version', 'dataset_root', 'source_tree',
            'selection', 'complete', 'summary', 'issues')
    value = {k: unit[k] for k in keys if k in unit}
    value['indexed_manifest_sha256'] = idx['manifest_sha256']
    value['units'] = [None] * number + [unit]
    return value
