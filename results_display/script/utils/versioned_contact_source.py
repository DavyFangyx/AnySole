"""Display-only reader with explicit, exact-content relocation of pinned BVH sources.

Default behavior remains the training reader's strict dependency validation.
An explicit source root can recover the very same original BVH bytes after a
link farm was repointed. No package manifest, labels or shared facts are changed.
"""
from pathlib import Path
import json
import numpy as np

from anysole.data.contact_package import ContactLabelPackage, ContactPackageError, sha256
from anysole.data.workspace_adapter import LABEL_ROOT


def bvh_time_support(path, query):
    frames, dt = None, None
    with Path(path).open() as stream:
        for line in stream:
            value = line.strip()
            if value.startswith('Frames:'):
                frames = int(value.split(':', 1)[1])
            elif value.startswith('Frame Time:'):
                dt = float(value.split(':', 1)[1])
                break
    if not frames or not dt or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid original BVH time coverage')
    q = np.asarray(query, dtype=np.float64)
    return np.isfinite(q) & (q >= -1e-9) & (q <= (frames - 1) * dt + 1e-9)


def package_info(method, candidate_root=None):
    if candidate_root is None:
        package = ContactLabelPackage(LABEL_ROOT / method, method=method)
        return package.root, package.manifest, package.expected_sha256
    root = Path(candidate_root).resolve()
    index_path = root / 'candidate_index.json'
    digest = (root / 'candidate_index.sha256').read_text().strip()
    if sha256(index_path) != digest:
        raise ContactPackageError('Candidate index hash mismatch')
    row = json.loads(index_path.read_text())['methods'][method]
    package_root = root / method
    manifest_path = package_root / 'label_package.json'
    if sha256(manifest_path) != row['manifest_sha256']:
        raise ContactPackageError('Candidate manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'contact_label_candidate.v1' or manifest.get('method') != method:
        raise ContactPackageError('Invalid candidate schema/method')
    return package_root, manifest, row['manifest_sha256']


def load_versioned_contact(session, method, source_bvh_root=None, candidate_root=None):
    if candidate_root is None and source_bvh_root is None:
        package = ContactLabelPackage(LABEL_ROOT / method, method=method)
        return package.load(session), {'source_relocated': False}
    package_root, manifest, _digest = package_info(method, candidate_root)
    if candidate_root and source_bvh_root:
        raise ContactPackageError('Choose either current candidates or original source recovery')
    sid = session['session_id']
    record = manifest['sessions'].get(sid)
    if record is None:
        raise ContactPackageError('Session missing from pinned package: ' + sid)
    if candidate_root:
        source = Path(record['bvh_path'])
    else:
        root = Path(source_bvh_root).resolve()
        source = (root / str(session['meta']['date']) / 'mocap_ori_bvh' / sid /
                  Path(record['bvh_path']).name).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ContactPackageError('Recovered BVH escapes explicit source root') from error
    for name, expected in record['source_files'].items():
        path = Path(session['dir']) / name
        if not path.is_file() or sha256(path) != expected:
            raise ContactPackageError(sid + ' changed shared source: ' + name)
    if not source.is_file() or sha256(source) != record['bvh_sha256']:
        raise ContactPackageError(sid + ' recovered original BVH hash mismatch')
    path = (package_root / record['label_file']).resolve()
    try:
        path.relative_to(package_root)
    except ValueError as error:
        raise ContactPackageError('Label file escapes pinned package') from error
    if not path.is_file() or sha256(path) != record['label_sha256']:
        raise ContactPackageError(sid + ' label missing or modified')
    with np.load(path, allow_pickle=False) as data:
        contact = np.asarray(data['contact'], dtype=np.float32)
        ids = np.asarray(session['frames']['frame_id'])
        if contact.shape != (len(ids), 2) or not np.array_equal(data['frame_id'], ids):
            raise ContactPackageError(sid + ' label/frame axis mismatch')
        if not np.isfinite(contact).all() or not np.isin(contact, [0, 1]).all():
            raise ContactPackageError(sid + ' invalid binary contact values')
    return contact.copy(), {'source_relocated': not bool(candidate_root), 'original_BVH': str(source),
        'original_BVH_sha256': record['bvh_sha256'], 'manifest_BVH_path': record['bvh_path'],
        'policy': ('current-derived-BVH candidate; not promoted to training' if candidate_root else
                   'explicit display audit relocation; content hash must equal immutable manifest')}
