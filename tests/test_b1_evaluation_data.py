"""Native-grid derivation contracts; all files are controlled synthetic fixtures."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anysole.data import b1_evaluation as evaluation
from anysole.data import workspace_adapter as adapter
from anysole.data.dataset import grouped_collate
from test_view_data import _imported, _cache, _hash, _rest


def _write_manifest(path, rows, meta):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    meta['manifest_sha256'] = _hash(path)
    path.with_suffix('.meta.json').write_text(json.dumps(meta))


def _fixture(tmp_path, monkeypatch):
    root, manifest, meta = _imported(tmp_path)
    rows, _ = adapter.load_grouped_manifest(manifest)
    for row in rows:
        row['split_roles'] = ['train'] if row['session_id'] == 'S5011' else ['test']
        row['clock_mapping']['anchor_camera'] = 1
    meta['candidate_split_counts'] = {'train': 1, 'val': 0, 'test': 1}
    meta['eligible_split_counts'] = {'train': 1, 'val': 0, 'test': 1}
    meta['original_split_counts'] = {'train': 1, 'val': 0, 'test': 2}
    meta['missing_modalities'] = [{'recording_id': '20260101/S14011', 'session_id': 'S14011',
        'split_roles': ['test'], 'reasons': ['missing_Mocap']}]
    calibration = {'pixel_coordinates': 'raw_distorted', 'image_size_px': [1600, 1200],
        'cameras': {f'cam{c}': {'R': np.eye(3).tolist(), 't': [0., 0., 400.],
            'K': [[1000., 0., 800.], [0., 1000., 600.], [0., 0., 1.]]} for c in (1, 2, 3, 4)},
        'mocap_raw_to_checkerboard_world': {'input_scale': .1, 'R': np.eye(3).tolist(), 't': [0., 0., 0.]}}
    calpath = root / '20260101/20260101.json'
    calpath.write_text(json.dumps(calibration))
    for row in rows:
        row['calibration']['sha256'] = _hash(calpath)
    _write_manifest(manifest, rows, meta)
    cache = _cache(tmp_path, manifest)
    index_path = cache / 'index.json'
    index = json.loads(index_path.read_text())
    index.update(complete=True, manifest_sha256=_hash(manifest))
    index_path.write_text(json.dumps(index))
    backup = tmp_path / 'backup'; backup.mkdir()
    (backup / 'anysole_manifest.jsonl').write_bytes(manifest.read_bytes())
    (backup / 'anysole_manifest.meta.json').write_bytes(manifest.with_suffix('.meta.json').read_bytes())
    (backup / 'index.json').write_bytes(index_path.read_bytes())
    for row in rows:
        row['views'] = [v for v in row['views'] if v['camera_id'] != 3]
    meta['protocol']['b1_version'] = 'anysole.b1.v1'
    meta['frozen'] = True
    _write_manifest(manifest, rows, meta)
    index['entries'] = {k: v for k, v in index['entries'].items() if not k.endswith('/cam3')}
    index['manifest_sha256'] = _hash(manifest)
    index_path.write_text(json.dumps(index))
    plan = tmp_path / 'plan.json'; plan.write_text(json.dumps({'test_access_authorized': True}))
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps({'evaluation_plan_path': str(plan), 'evaluation_plan_sha256': _hash(plan)}))
    monkeypatch.setattr('anysole.data.smpl_io._rest_joints', _rest)
    return root, manifest, cache, backup / 'index.json', auth


def test_authorization_precedes_all_dataset_io(tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{}')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'evaluation_plan_path': str(plan), 'evaluation_plan_sha256': _hash(plan)}))
    with pytest.raises(ValueError, match='not been authorized'):
        evaluation.prepare_evaluation_data(tmp_path/'absent', tmp_path, tmp_path, tmp_path, tmp_path/'out', receipt)
    plan.write_text('{"test_access_authorized":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluation.verify_test_authorization(receipt)


def test_native_derivation_independent_camera_time_and_teacher_inputs(tmp_path, monkeypatch):
    root, manifest, cache, backup, auth = _fixture(tmp_path, monkeypatch)
    rows, _ = adapter.load_grouped_manifest(manifest)
    parent = next(r for r in rows if r['split_roles'] == ['test'])
    original, _ = adapter.load_grouped_manifest(backup.parent/'anysole_manifest.jsonl')
    original_parent = next(r for r in original if r['split_roles'] == ['test'])
    sources = {v['camera_id']: copy.deepcopy(v['sources']) for v in original_parent['views']}
    for value in sources[2]:
        value['time_s'] += .005
    student = evaluation.derive_native_record(parent, sources, np.arange(100)/120, camera=2)
    teacher = evaluation.derive_native_record(parent, sources, np.arange(100)/120, teacher=True)
    assert student['target_time_s'][0] == pytest.approx(10000.005)
    assert teacher['target_time_s'][0] == pytest.approx(10000.)
    assert [v['camera_id'] for v in student['views']] == [2]
    assert [v['camera_id'] for v in teacher['views']] == [1, 2, 4]
    assert student['motion']['time_s'][0] == pytest.approx(.005)
    assert student['pressure']['native_sampling']['alpha'][0] == pytest.approx(.2)
    assert student['views'][0]['source_row'] == list(range(30))
    assert student['available_allowed_view_count'][1] == 3
    assert student['pressure']['foot_valid'][-1] == [False, False]
    assert parent['target_time_s'][0] == 10000.
    sources[2][1]['time_s'] = sources[2][0]['time_s']
    with pytest.raises(ValueError, match='unique'):
        evaluation.derive_native_record(parent, sources, np.arange(100)/120, camera=2)
    selected = evaluation.derive_native_record(parent, sources, np.arange(100)/120,
        camera=2, timestamp_ties='smallest_frame')
    assert selected['target_frame_id'][:3] == [0, 2, 3]
    assert selected['views'][0]['source_row'][:3] == [0, 2, 3]
    assert selected['evaluation']['excluded_duplicate_source_rows'] == [1]
    assert not selected['views'][0]['view_edges'][0]
    assert len(selected['views'][0]['sources']) == 30


def test_pressure_bracket_masks_exact_gaps_and_each_foot():
    source = np.asarray([0., .025, .05, .1])
    feet = np.asarray([[1, 1], [1, 0], [1, 1], [1, 1]], bool)
    b, valid = evaluation.pressure_brackets(source, [-.001, 0., .0125, .025, .05, .075, .101], feet)
    assert valid.tolist() == [[False, False], [True, True], [True, False], [True, False],
                              [True, True], [False, False], [False, False]]
    assert b.exact[3]
    with pytest.raises(ValueError, match='strictly increasing'):
        evaluation.pressure_brackets([0., 0.], [0.], feet[:2])


def test_derived_cache_lineage_tail_coverage_and_readonly(tmp_path, monkeypatch):
    root, manifest, cache, backup, auth = _fixture(tmp_path, monkeypatch)
    protected = [manifest, manifest.with_suffix('.meta.json'), cache/'index.json', backup]
    before = {str(p): _hash(p) for p in protected}
    result = evaluation.prepare_evaluation_data(manifest, root, cache, backup, tmp_path/'out', auth)
    assert result['requested_test_recordings'] == 2
    assert result['eligible_test_recordings'] == 1
    assert result['missing_modalities'][0]['reasons'] == ['missing_Mocap']
    assert set(result['manifests']) == {'cam1', 'cam2', 'cam3', 'cam4', 'teacher'}
    for name, path in result['manifests'].items():
        ds = evaluation.NativeRGBEvaluationDataset(path, root, cache, tmp_path/'unused.pkl',
            cache_index=result['cache_index'], authorization=auth)
        assert len(ds) == 2
        assert ds.windows[-1][1:3] == (10, 30)
        assert ds.inference_units[0]['n_frames'] == 30
        item = ds[0]
        expected_cameras = [1, 2, 4] if name == 'teacher' else [int(name[-1])]
        assert item['meta']['camera_id'].tolist() == expected_cameras
        assert item['visual']['V_feat'].shape == (len(expected_cameras), 20, 2051)
        assert item['meta']['coordinate_frame'] == 'world'
        assert item['masks']['motion_valid'].all()
        assert item['labels']['pressure'][1].tolist() == [1.]*96
        assert item['labels']['root'][1, 0] == pytest.approx(.53)
        assert item['labels']['velocity'][1, 0] == pytest.approx(1.2)
        assert grouped_collate([item, ds[1]])['meta']['native_frame_index'].shape == (2, 20)
    assert before == {str(p): _hash(p) for p in protected}
    with pytest.raises(FileExistsError, match='already exist'):
        evaluation.prepare_evaluation_data(manifest, root, cache, backup, tmp_path/'out', auth)


def test_cam3_cache_requires_saved_source_lineage(tmp_path, monkeypatch):
    root, manifest, cache, backup, auth = _fixture(tmp_path, monkeypatch)
    index = json.loads(backup.read_text())
    index['entries']['20260101/S13011/cam3']['provenance']['source_index_sha256'] = 'wrong'
    backup.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='cache/source lineage'):
        evaluation.prepare_evaluation_data(manifest, root, cache, backup, tmp_path/'out', auth)
    assert not (tmp_path/'out').exists()


def test_features_are_verified_and_short_native_axis_is_padded(tmp_path, monkeypatch):
    root, manifest, cache, backup, auth = _fixture(tmp_path, monkeypatch)
    result = evaluation.prepare_evaluation_data(manifest, root, cache, backup, tmp_path/'out', auth)
    path = Path(result['manifests']['cam3'])
    rows, meta = adapter.load_grouped_manifest(path)
    record = rows[0]
    # Keep a short derived native axis while retaining the original source table.
    for key in ('target_frame_id', 'target_time_s', 'native_frame_index', 'native_source_frame_id', 'available_allowed_view_count'):
        record[key] = record[key][:5]
    for key in ('time_s', 'lower', 'upper', 'alpha', 'valid'):
        record['motion'][key] = record['motion'][key][:5]
    for key in ('row', 'foot_valid'):
        record['pressure'][key] = record['pressure'][key][:5]
    for key in ('lower', 'upper', 'alpha', 'exact', 'valid'):
        record['pressure']['native_sampling'][key] = record['pressure']['native_sampling'][key][:5]
    record['motion_edges'] = record['motion_edges'][:4]
    record['pressure_foot_edges'] = record['pressure_foot_edges'][:4]
    for view in record['views']:
        for key in ('source_row', 'source_frame_id', 'source_time_s', 'time_residual_ms', 'view_valid', 'reuse'):
            view[key] = view[key][:5]
        view['view_edges'] = view['view_edges'][:4]
    _write_manifest(path, rows, meta)
    ds = evaluation.NativeRGBEvaluationDataset(path, root, cache, tmp_path/'unused.pkl',
        cache_index=result['cache_index'], authorization=auth)
    assert len(ds) == 1
    item = ds[0]; batch = grouped_collate([item])
    assert batch['visual']['V_feat'].shape == (1, 1, 20, 2051)
    assert batch['meta']['padding'][0, 0, 5:].all()
    entry = json.loads((cache/'index.json').read_text())['entries']['20260101/S13011/cam1']
    (cache/entry['path']).write_bytes(b'corruption')
    with pytest.raises(ValueError, match='provenance'):
        evaluation.NativeRGBEvaluationDataset(result['manifests']['cam1'], root, cache, tmp_path/'unused.pkl',
            cache_index=result['cache_index'], authorization=auth)
