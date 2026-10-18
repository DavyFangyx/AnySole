"""CPU controller tests with explicit synthetic IO, not real-data pilot evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole import train
from anysole.data.dataset import grouped_collate
from anysole.types import JOINT_PARENTS, validate_view_config
from anysole.utils.geometry import fk_pose6d
from configs.tools import runner

ARTIFACTS = Path('/private/Tactile/.ai/runtime/tasks/active/TASK-20261016-1513-anysole-view-consistency/artifacts/runtime')


@pytest.fixture(autouse=True, scope='module')
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyVisual(nn.Module):
    """Dropout makes RNG restoration observable, with the exact V-only API."""
    def __init__(self, dropout=.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Dropout(dropout))
        self.pose = nn.Linear(8, 144)
        self.pressure = nn.Linear(8, 96)
        self.velocity = nn.Linear(8, 3)

    def forward_visual(self, features, valid, timestamps):
        h = self.trunk(features[..., :3])
        identity = h.new_tensor([1, 0, 0, 0, 1, 0]).repeat(24)
        return {'x0_hat': identity + .05*self.pose(h), 'pressure_hat': self.pressure(h),
                'v_hat': self.velocity(h), 'sigma_frozen': False}


class ToyGroups:
    """Masks/GT are synthetic and remain outside the student's input."""
    split = 'train'

    def __init__(self, cameras=(4, 2, 3, 1), zero=False, nan=False):
        self.cameras, self.zero, self.nan = cameras, zero, nan
        self.windows = [(g, 0, 20, tuple(range(1, c+1))) for g, c in enumerate(cameras)]

    def __len__(self):
        return len(self.cameras)

    def sample_key(self, i):
        return f'toy_{i}:0:20'

    def __getitem__(self, g):
        c, t = self.cameras[g], 20
        generator = torch.Generator().manual_seed(41+g)
        pose = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(t, 24)
        root = torch.zeros(t, 3)
        root[:, 0] = torch.arange(t)/20
        offsets = torch.arange(72).float().reshape(24, 3)/200
        offsets[0] = 0
        parents = torch.tensor(JOINT_PARENTS)
        motion = torch.full((t,), not self.zero, dtype=torch.bool)
        pressure = torch.full((t, 96), not self.zero, dtype=torch.bool)
        features = torch.randn(c, t, 2051, generator=generator)
        if self.nan:
            features[0, 0, 0] = float('nan')
        return {'visual': {'V_feat': features, 'view_valid': torch.ones(c, t, dtype=torch.bool),
                           'view_edges': torch.ones(c, t-1, dtype=torch.bool),
                           'target_time_s': torch.arange(t, dtype=torch.float64)/40},
                'labels': {'pose6d': pose, 'pressure': torch.full((t, 96), .1*(g+1)),
                           'velocity': torch.tensor([2., 0, 0]).expand(t, -1).clone(), 'root': root,
                           'offsets': offsets, 'parents': parents, 'betas': torch.zeros(10),
                           'kp_gt': fk_pose6d(pose[None], root[None], offsets, parents)[0]},
                'masks': {'motion_valid': motion, 'pressure_valid': pressure,
                          'velocity_valid': motion.clone(), 'motion_edges': motion[:-1].clone(),
                          'pressure_edges': pressure[:-1].clone()},
                'meta': {'camera_id': torch.arange(1, c+1), 'sample_key': self.sample_key(g),
                         'recording_id': f'toy_{g}', 'target_frame_id': torch.arange(t),
                         'window_length': 20}}


def config(root, **overrides):
    value = yaml.safe_load((Path(__file__).resolve().parents[1]/'anysole/configs/view_consistency.yaml').read_text())
    value.update(stage='pilot', experiment='base_pool', device='cpu', num_workers=0, amp=False,
                 lambda_mv=0., lambda_sigma=0., lambda_assign=0., target_updates=4,
                 target_rgb_windows=40, effective_batch_rgb_windows=10, microbatch_rgb_windows=10,
                 validation_every_updates=1, checkpoint_every_updates=1, selection_m_ref=1.,
                 selection_p_ref=.5, wandb_mode='disabled', heartbeat_sec=.05,
                 disk_min_free_bytes=1, control_dir=str(root/'control'),
                 _test_stats={'pose_mean': [0.]*144, 'pose_std': [1.]*144})
    value.update(overrides)
    return value


def student():
    torch.manual_seed(991)
    return TinyVisual()


def validation(model, dataset, config, device):
    # Deterministic numeric selector fixture; production evaluate_visual tested separately.
    magnitude = float(next(model.parameters()).detach().square().mean())
    return {'val/mpjpe': magnitude, 'val/pressure_rmse': magnitude*2,
            'val/pressure_mae': magnitude, 'val/motion_frames': 80,
            'val/pressure_cells': 80*96, 'val/coverage_hash': 'fixed_toy_domain'}, None


def run_fixture(root, *, stop=None, resume=None, cfg=None, groups=None, model=None):
    return train.train_visual_run(cfg or config(root), root/'run', resume=resume,
                                 stop_after_updates=stop, dataset=groups or ToyGroups(),
                                 validation_dataset=ToyGroups(), model=model or student(),
                                 validation_fn=validation)


def assert_state_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_joint_selector_pareto_score_tie_coverage_and_unfrozen():
    selector = train.JointSelector(1., 1.)
    def metrics(m, p, coverage='x'):
        return {'val/mpjpe': m, 'val/pressure_rmse': p, 'val/motion_frames': 2,
                'val/pressure_cells': 192, 'val/coverage_hash': coverage}
    assert selector.consider(metrics(.4, .8), 1)
    assert not selector.consider(metrics(.5, .9), 2)
    assert selector.consider(metrics(.3, .9), 3)  # equal score, lower motion wins
    assert not selector.consider(metrics(.3, .9), 4)  # earlier update wins
    assert selector.best_step == 3
    assert len(selector.candidates) == 3
    with pytest.raises(ValueError, match='coverage'):
        selector.consider(metrics(.2, .2, 'changed'), 5)
    with pytest.raises(ValueError, match='nonfinite'):
        selector.consider(metrics(float('nan'), .2), 5)
    assert not train.JointSelector().consider(metrics(.1, .1), 1)


def test_identity_excludes_operations_and_is_not_self_referential(tmp_path):
    cfg = config(tmp_path)
    changed = {**cfg, 'run_dir': '/elsewhere', 'run_id': 'x', 'plan_hash': 'a'*64,
               'semantic_hash': 'b'*64, 'device': 'cuda:3', 'wandb_id': 'remote', 'pid': 1}
    assert train.visual_run_identity(cfg) == train.visual_run_identity(changed)
    assert train.visual_run_identity(cfg) != train.visual_run_identity({**cfg, 'lambda_pose': 4.})
    assert train.visual_run_identity(cfg) != train.visual_run_identity({**cfg, 'stage': 'formal'})


def test_group_budget_variable_c_resume_and_no_model_rng():
    sampler = train.GroupBudgetSampler([4, 2, 3, 1], ['a', 'b', 'c', 'd'], 3)
    before = torch.get_rng_state()
    for _ in range(5):
        assert sum(sampler.weights[i] for i in sampler.next_update(8)) == 8
    torch.testing.assert_close(torch.get_rng_state(), before)
    state = copy.deepcopy(sampler.state_dict())
    restored = train.GroupBudgetSampler([4, 2, 3, 1], ['a', 'b', 'c', 'd'], 3)
    restored.load_state_dict(state)
    assert sampler.next_update(8) == restored.next_update(8)
    with pytest.raises(ValueError, match='cannot pack'):
        train.GroupBudgetSampler([3], ['a'], 1).next_update(8)
    with pytest.raises(ValueError, match='split'):
        train.microbatch_groups([0], [4], 3)


def test_process_lease_and_slot_ledger_duplicate_and_stale(tmp_path):
    path = tmp_path/'writer.lock'
    with train.ProcessLease(path, 'x'):
        with pytest.raises((BlockingIOError, RuntimeError)):
            with train.ProcessLease(path, 'x'):
                pass
    path.write_text(json.dumps({'status': 'active', 'pid': 999999999, 'starttime': 'stale'}))
    with train.ProcessLease(path, 'x'):
        pass
    ledger = runner.SlotLedger(tmp_path/'slots.json')
    slot = ledger.claim([(0, 0)], 'x')
    with pytest.raises(RuntimeError, match='already'):
        ledger.claim([(0, 0)], 'x')
    assert ledger.claim([(0, 0)], 'y') is None
    ledger.release(slot, 'x', 0)
    assert ledger.claim([(0, 0)], 'y') == slot


def test_continuous_vs_atomic_resume_exact_all_training_state(tmp_path):
    continuous, split = tmp_path/'continuous', tmp_path/'split'
    whole = run_fixture(continuous)
    interrupted = run_fixture(split, stop=2)
    assert interrupted['state'] == 'paused' and interrupted['successful_updates'] == 2
    resumed = run_fixture(split, resume=split/'run/last.pt')
    assert whole['state'] == resumed['state'] == 'pilot_completed'
    assert resumed['formal_complete'] is False
    a = torch.load(continuous/'run/last.pt', weights_only=False)
    b = torch.load(split/'run/last.pt', weights_only=False)
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'sampler', 'selector', 'training', 'rng', 'schedule', 'stats'):
        assert_state_equal(a[key], b[key])
    assert a['training']['rgb_windows'] == 40 and a['training']['successful_updates'] == 4
    left = train.GroupBudgetSampler([4, 2, 3, 1], [ToyGroups().sample_key(i) for i in range(4)], 1)
    right = copy.deepcopy(left)
    left.load_state_dict(a['sampler']); right.load_state_dict(b['sampler'])
    assert left.next_update(8) == right.next_update(8)
    rows = [json.loads(line) for line in (split/'run/metrics.jsonl').read_text().splitlines()]
    assert [row['history_step'] for row in rows] == list(range(4))
    assert [row['progress/update'] for row in rows] == [1, 2, 3, 4]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(split/'run/tb')).Reload()
    assert [point.step for point in events.Scalars('progress/update')] == [1, 2, 3, 4]


@pytest.mark.parametrize('failure', ('zero', 'nan'))
def test_failures_do_not_consume_success_budget(tmp_path, failure):
    with pytest.raises((ValueError, FloatingPointError)):
        run_fixture(tmp_path, groups=ToyGroups(zero=failure == 'zero', nan=failure == 'nan'))
    record = json.loads((tmp_path/'run/run.json').read_text())
    assert record['state'] == 'failed' and record['exit_code'] == 1
    assert record['successful_updates'] == record['rgb_windows'] == 0
    assert not runner.verified_visual_completion(tmp_path/'run', record['semantic_hash'])


def test_bad_checkpoint_schema_hash_and_partial_file_rejected(tmp_path):
    run_fixture(tmp_path, stop=1)
    good = torch.load(tmp_path/'run/last.pt', weights_only=False)
    for i, (key, value) in enumerate([('schema_version', 'wrong'), ('semantic_hash', '0'*64)]):
        bad = copy.deepcopy(good); bad[key] = value
        torch.save(bad, tmp_path/'bad.pt')
        with pytest.raises(ValueError, match='schema/semantic'):
            run_fixture(tmp_path, resume=tmp_path/'bad.pt', cfg=config(tmp_path, recovery_note='synthetic corruption rejection check'))
    (tmp_path/'bad.pt').write_bytes(b'PK\x03')
    with pytest.raises(ValueError, match='corrupt'):
        run_fixture(tmp_path, resume=tmp_path/'bad.pt', cfg=config(tmp_path, recovery_note='synthetic corruption rejection check'))


def test_candidate_plan_has_fifteen_identities_and_cannot_launch(tmp_path):
    template = Path(__file__).resolve().parents[1]/'anysole/configs/view_consistency.yaml'
    plan = runner.create_visual_candidate_plan(template, tmp_path/'plan.json')
    assert len(plan['runs']) == len({row['semantic_hash'] for row in plan['runs']}) == 15
    assert plan['plan_hash'] == runner.plan_hash(plan)
    legacy = runner.resolve_visual_plan_run(plan, plan['runs'][0])
    assert legacy['pressure_head'] == 'legacy' and legacy['pressure_time_pe_enabled'] is False
    assert runner.run_visual_plan(tmp_path/'plan.json', dry_run=True) == 0
    with pytest.raises(ValueError, match='candidate/unmeasured'):
        runner.run_visual_plan(tmp_path/'plan.json')


def test_formal_rejects_unfrozen_and_any_test_truncation(tmp_path):
    cfg = config(tmp_path, stage='formal')
    with pytest.raises(ValueError):
        validate_view_config(cfg, formal=True)
    with pytest.raises(ValueError, match='test IO or stop'):
        train.train_visual_run(cfg, tmp_path/'run', dataset=ToyGroups())


def test_complete_formal_schema_accepts_only_full_reference_budget(tmp_path):
    cfg = config(tmp_path, stage='formal', plan_frozen=True, resource_frozen=True,
                 resource_version='synthetic_schema_fixture', runs_per_gpu=1, selection_coverage_hash='0'*64)
    cfg.update(train.reference_visual_budget(1765))
    for name in ('manifest_hash', 'split_hash', 'stats_hash', 'frontend_hash', 'frontend_runtime_hash',
                 'source_hash', 'plan_hash', 'cache_index_hash', 'stats_file_hash',
                 'hrnet_checkpoint_hash', 'detector_checkpoint_hash', 'smpl_runtime_model_hash'):
        cfg[name] = '0'*64  # schema-only fixture, never launch or claim a trained run
    cfg = train.get_builder('visual').effective_config(cfg)
    validate_view_config(cfg, formal=True)
    for key, value in [('target_updates', 4), ('target_rgb_windows', 4*256),
                       ('reference_epochs', 200), ('resource_frozen', False),
                       ('selection_p_ref', 1.e-10), ('stop_after_updates', 2)]:
        with pytest.raises(ValueError):
            validate_view_config({**cfg, key: value}, formal=True)


def test_variable_c_controller_accumulation_matches_merged_parameters(tmp_path):
    groups = ToyGroups()
    full = grouped_collate([groups[i] for i in range(4)])
    micros = [grouped_collate([groups[i] for i in (0, 1)]), grouped_collate([groups[i] for i in (2, 3)])]
    torch.manual_seed(5)
    a, b = TinyVisual(dropout=0), None
    b = copy.deepcopy(a)
    cfg = config(tmp_path)
    optimizers = [torch.optim.SGD(model.parameters(), lr=.001) for model in (a, b)]
    results = []
    for model, batches, opt in ((a, [full], optimizers[0]), (b, micros, optimizers[1])):
        results.append(train.visual_accumulation_step(model, batches, cfg, opt,
                       torch.amp.GradScaler('cuda', enabled=False), torch.device('cpu'), 0, 4))
    assert results[0]['counts'] == results[1]['counts']
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def test_lambda_zero_shuffle_does_not_call_shuffle_rng(tmp_path, monkeypatch):
    cfg = config(tmp_path, experiment='pair_shuffle', pair_shuffle=True, lambda_mv=.2)
    model = TinyVisual(dropout=0)
    batch = grouped_collate([ToyGroups()[i] for i in (0, 1)])
    def forbidden(*args, **kwargs):
        raise AssertionError('lambda=0 must not execute pair RNG')
    monkeypatch.setattr(train, 'shuffle_pair_groups', forbidden)
    result = train.visual_accumulation_step(model, [batch], cfg,
        torch.optim.SGD(model.parameters(), lr=.001), torch.amp.GradScaler('cuda', enabled=False),
        torch.device('cpu'), 0, 20)
    assert result['lambda_mv'] == 0 and result['shuffle_maps'] == [None]


def test_common_microbatch_packing_supports_distinct_key_derangement():
    chunks = train.microbatch_groups([0, 1, 2, 3], [4, 2, 3, 1], 6,
                                    keys=['a', 'b', 'c', 'd'], require_pairs=True)
    assert chunks == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match='derangement'):
        train.microbatch_groups([0, 1], [4, 2], 4, keys=['a', 'b'], require_pairs=True)
    with pytest.raises(ValueError, match='derangement'):
        train.microbatch_groups([0, 1, 1], [4, 2], 8, keys=['a', 'b'], require_pairs=True)


def test_production_val_deduplicates_uses_given_beta_zero_offsets_and_preserves_rng(tmp_path):
    class RepeatedVal(ToyGroups):
        split = 'val'
        def __init__(self):
            super().__init__((2, 2))
        def __getitem__(self, i):
            item = super().__getitem__(0)
            item['visual']['V_feat'][..., 0] = .1
            return item
    class Oracle(nn.Module):
        def forward_visual(self, features, valid, times):
            n, t = features.shape[:2]
            return {'x0_hat': features.new_tensor([1, 0, 0, 0, 1, 0]).repeat(n, t, 24),
                    'pressure_hat': features[..., :1].expand(-1, -1, 96),
                    'v_hat': features.new_zeros(n, t, 3)}
    dataset, model = RepeatedVal(), Oracle().train()
    offsets = dataset[0]['labels']['offsets']
    rng = train.capture_training_rng(torch.device('cpu'))
    result, plot = train.evaluate_visual(model, dataset, config(tmp_path), torch.device('cpu'), offsets=offsets)
    assert result['val/motion_frames'] == 40 and result['val/pressure_cells'] == 40*96
    assert result['val/prediction_frames'] == 40 and result['val/gt_motion_frames'] == 80
    assert result['val/mpjpe'] < 1e-7 and result['val/pressure_rmse'] == 0
    assert result['val/pred_beta_source'] == 'zero' and model.training
    assert_state_equal(rng, train.capture_training_rng(torch.device('cpu')))
    altered, _ = train.evaluate_visual(model, dataset, config(tmp_path), torch.device('cpu'), offsets=offsets*2)
    assert altered['val/mpjpe'] > .1 and altered['val/coverage_hash'] == result['val/coverage_hash']
    dataset.split = 'test'
    with pytest.raises(ValueError, match='test role'):
        train.evaluate_visual(model, dataset, config(tmp_path), torch.device('cpu'), offsets=offsets)

    class MissingTruthVal(ToyGroups):
        split = 'val'
        def __init__(self):
            super().__init__((2, 2))
        def __getitem__(self, i):
            item = super().__getitem__(i)
            item['visual']['V_feat'][..., 0] = .1*(i+1)
            if i == 1:
                item['masks']['motion_valid'][:] = False
                item['masks']['pressure_valid'][:] = False
            return item
    retained, _ = train.evaluate_visual(model, MissingTruthVal(), config(tmp_path),
                                       torch.device('cpu'), offsets=offsets)
    assert retained['val/prediction_frames'] == 80 and retained['val/motion_frames'] == 40


def test_fixed_train_val_pressure_motion_media_write_tensorboard(tmp_path):
    cfg = config(tmp_path)
    item = ToyGroups()[0]
    plot = train.visual_sample_plot(student(), item, cfg, torch.device('cpu'), split='train')
    assert plot['prediction'].shape == (20, 96) and plot['trajectory_m'].shape == (20, 3)
    assert np.isfinite(plot['motion_error_m']).all() and np.all(plot['trajectory_m'][0] == 0)
    log_dir = tmp_path/'media'; log_dir.mkdir()
    logger = train.VisualLogger(log_dir, cfg)
    logger.log({'progress/update': 1}, plot=[plot])
    logger.close()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(log_dir/'tb')).Reload()
    assert 'train/left_pressure_display_clip01_truth_prediction' in events.Tags()['images']
    assert 'train/motion_error_m_u1' in events.Tags()['scalars']


def test_grouped_cli_rejects_silent_legacy_overrides(tmp_path):
    args = train.parse_args(['--seed', '2', '--run-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='unsupported CLI overrides'):
        train._visual_main(args, config(tmp_path))


def test_reference_budget_uses_measured_cam3_and_frozen_common_batch():
    assert train.reference_visual_budget(1765)['target_updates'] == 740*7
    assert train.reference_visual_budget(1765, drop_last=True)['target_updates'] == 740*6
    changed = train.reference_visual_budget(1765, 240)
    assert changed['target_updates'] == 740*8 and changed['target_rgb_windows'] == 740*8*240
    with pytest.raises(ValueError, match='zero'):
        train.reference_visual_budget(1, 256, drop_last=True)
    refs = train.visual_selection_references({'val/mpjpe': 0., 'val/pressure_rmse': .3,
        'val/motion_frames': 2, 'val/pressure_cells': 192, 'val/coverage_hash': 'common'})
    assert refs['selection_m_ref'] == 1.e-8 and refs['selection_p_ref'] == .3


def test_actual_model_checkpoint_saved_effective_structure_rebuild(tmp_path):
    entry = train.get_builder('visual')
    cfg = entry.effective_config(config(tmp_path))
    torch.manual_seed(15)
    original = entry.build(cfg)
    checkpoint = {'config': original.effective_config, 'model': original.state_dict()}
    train.atomic_checkpoint(tmp_path/'actual_model.pt', checkpoint)
    saved = torch.load(tmp_path/'actual_model.pt', map_location='cpu', weights_only=False)
    rebuilt = entry.build(saved['config'])
    rebuilt.load_state_dict(saved['model'], strict=True)
    assert rebuilt.effective_config['pressure_time_pe_enabled'] is True
    assert rebuilt.effective_config['structure_hash'] == cfg['structure_hash']
    for left, right in zip(original.parameters(), rebuilt.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_gpu_owner_guard_refuses_unknown_process_without_signals(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setitem(sys.modules, 'pynvml', SimpleNamespace(
        nvmlInit=lambda: calls.append('init'), nvmlShutdown=lambda: calls.append('shutdown'),
        nvmlDeviceGetHandleByIndex=lambda gpu: gpu,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=999999999)]))
    with pytest.raises(RuntimeError, match='outside this task slot ledger'):
        runner.check_visual_gpu_owners([0], runner.SlotLedger(tmp_path/'slots.json'))
    assert calls == ['init', 'shutdown']


def test_wandb_fallback_metadata_records_actual_segment_id_and_sync_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id='actual_offline_segment', dir=str(tmp_path/'offline-run/files'),
                               define_metric=lambda *a, **k: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        Api=lambda **kwargs: SimpleNamespace(api_key=None), Settings=lambda **kwargs: kwargs, init=init))
    logger = train.VisualLogger(tmp_path, config(tmp_path, wandb_mode='online'))
    state = logger.state_dict(); logger.close()
    assert state['wandb_id'] == 'actual_offline_segment' and state['wandb_logical_id']
    assert state['offline_dir'] == str(tmp_path/'offline-run')
    assert state['offline_segments'] == [str(tmp_path/'offline-run')]
    assert state['wandb_url'] is None and state['online_fallback_reason'] == 'authentication_unavailable'
    assert calls[0]['mode'] == 'offline' and calls[0]['id'] is None


def test_worker_watchdog_uses_exact_pid_starttime_and_bounded_startup(tmp_path):
    cfg = {'heartbeat_timeout_sec': 30, 'worker_startup_timeout_sec': 90}
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 100., cfg, now=150.) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 100., cfg, now=200.) == 'heartbeat_stale'
    train.atomic_json(tmp_path/'heartbeat.json', {'pid': 42, 'starttime': 'start', 'unix_time': 190.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 100., cfg, now=200.) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'other_pid_generation', 100., cfg, now=200.) == 'heartbeat_stale'


def test_worker_watchdog_retains_fresh_pulse_across_one_read_gap_and_recovers(tmp_path, monkeypatch):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}
    original_cfg = copy.deepcopy(cfg)
    cache = {}
    path = tmp_path/'heartbeat.json'
    train.atomic_json(path, {'pid': 42, 'starttime': 'start', 'unix_time': 999.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=999., heartbeat_state=cache) is None
    original_read = Path.read_text
    def missing(file, *args, **kwargs):
        if file == path:raise FileNotFoundError('SYNTHETIC_CPU_ONLY read race')
        return original_read(file, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', missing)
        assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1000., heartbeat_state=cache) is None
        assert cache['last_valid_unix_time'] == 999.
    train.atomic_json(path, {'pid': 42, 'starttime': 'start', 'unix_time': 1001.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1001., heartbeat_state=cache) is None
    assert cache['last_valid_unix_time'] == 1001. and cfg == original_cfg


@pytest.mark.parametrize('damage', ['missing', 'bad_json', 'static', 'wrong_pid', 'wrong_starttime', 'list'])
def test_worker_watchdog_seen_pulse_expires_at_300_without_new_valid_time(tmp_path, damage):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}
    cache = {}; path = tmp_path/'heartbeat.json'
    train.atomic_json(path, {'pid': 42, 'starttime': 'start', 'unix_time': 999.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=999., heartbeat_state=cache) is None
    if damage == 'missing':path.unlink()
    elif damage == 'bad_json':path.write_text('{')
    elif damage == 'list':path.write_text('[]')
    elif damage == 'wrong_pid':train.atomic_json(path, {'pid': 43, 'starttime': 'start', 'unix_time': 1299.})
    elif damage == 'wrong_starttime':train.atomic_json(path, {'pid': 42, 'starttime': 'reused', 'unix_time': 1299.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1299., heartbeat_state=cache) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1299.01, heartbeat_state=cache) == 'heartbeat_stale'
    assert cache['last_valid_unix_time'] == 999.


@pytest.mark.parametrize('timestamp', [True, float('nan'), float('inf'), float('-inf'), '1000', 100000.])
def test_worker_watchdog_invalid_timestamp_cannot_extend_seen_deadline(tmp_path, timestamp):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}; cache = {}
    path = tmp_path/'heartbeat.json'
    train.atomic_json(path, {'pid': 42, 'starttime': 'start', 'unix_time': 999.})
    runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=999., heartbeat_state=cache)
    # Deliberately inject invalid/nonfinite external JSON. The native writer
    # already rejects NaN/Inf, so using it would test the writer, not this guard.
    path.write_text(json.dumps({'pid': 42, 'starttime': 'start', 'unix_time': timestamp}))
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1000., heartbeat_state=cache) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, now=1300., heartbeat_state=cache) == 'heartbeat_stale'
    assert cache['last_valid_unix_time'] == 999.


def test_worker_watchdog_no_valid_first_pulse_keeps_startup_boundary(tmp_path):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}; cache = {}
    train.atomic_json(tmp_path/'heartbeat.json', {'pid': 42, 'starttime': 'start', 'unix_time': 999999.})
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 100., cfg, now=1000., heartbeat_state=cache) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 100., cfg, now=1000.01, heartbeat_state=cache) == 'heartbeat_stale'
    assert cache['last_valid_unix_time'] is None


def test_worker_watchdog_cache_is_per_child_and_resets_on_pid_generation(tmp_path):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}
    first, second = {}, {}; path = tmp_path/'heartbeat.json'
    train.atomic_json(path, {'pid': 42, 'starttime': 'old', 'unix_time': 999.})
    runner.visual_worker_heartbeat(tmp_path, 42, 'old', 0., cfg, now=999., heartbeat_state=first)
    train.atomic_json(path, {'pid': 43, 'starttime': 'other', 'unix_time': 1200.})
    runner.visual_worker_heartbeat(tmp_path, 43, 'other', 1000., cfg, now=1200., heartbeat_state=second)
    assert first['last_valid_unix_time'] == 999. and second['last_valid_unix_time'] == 1200.
    runner.visual_worker_heartbeat(tmp_path, 42, 'new', 1000., cfg, now=1200., heartbeat_state=first)
    assert first['last_valid_unix_time'] is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'new', 1000., cfg, now=1900., heartbeat_state=first) is None
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'new', 1000., cfg, now=1900.01, heartbeat_state=first) == 'heartbeat_stale'


def test_worker_watchdog_default_now_is_sampled_after_read(tmp_path, monkeypatch):
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}; cache = {}; clock = [1000.]
    original_read = Path.read_text
    def read(file, *args, **kwargs):
        if file == tmp_path/'heartbeat.json':
            clock[0] = 1000.01
            return json.dumps({'pid': 42, 'starttime': 'start', 'unix_time': 1000.01})
        return original_read(file, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read); monkeypatch.setattr(runner.time, 'time', lambda: clock[0])
    assert runner.visual_worker_heartbeat(tmp_path, 42, 'start', 0., cfg, heartbeat_state=cache) is None
    assert cache['last_valid_unix_time'] == 1000.01


def test_worker_watchdog_actual_runner_cache_initialization_and_call(tmp_path, monkeypatch):
    """Execute the actual child's init and watchdog call with CPU-only process IO."""
    import ast, inspect
    from types import SimpleNamespace
    tree = ast.parse(inspect.getsource(runner.run_visual_plan))
    initialize = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == '_view_heartbeat_state' for t in n.targets))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'visual_worker_heartbeat')
    def execute(nodes, namespace):exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_runner_CPU_path', 'exec'), namespace)
    cfg = {'heartbeat_timeout_sec': 300, 'worker_startup_timeout_sec': 900}
    process = SimpleNamespace(pid=42, _view_starttime='start', _view_started_at=0., _view_config=cfg)
    namespace = {'process':process, 'root':tmp_path, 'visual_worker_heartbeat':runner.visual_worker_heartbeat}
    execute([initialize], namespace)
    train.atomic_json(tmp_path/'heartbeat.json', {'pid':42, 'starttime':'start', 'unix_time':999.})
    clock = [999.]; monkeypatch.setattr(runner.time, 'time', lambda: clock[0])
    execute([call], namespace); assert namespace['reason'] is None
    (tmp_path/'heartbeat.json').unlink(); clock[0] = 1000.
    execute([call], namespace)
    assert namespace['reason'] is None and process._view_heartbeat_state['last_valid_unix_time'] == 999.
    clock[0] = 1300.; execute([call], namespace); assert namespace['reason'] == 'heartbeat_stale'
    other = SimpleNamespace(pid=43, _view_starttime='next', _view_started_at=1300., _view_config=cfg)
    namespace['process'] = other; execute([initialize, call], namespace)
    assert namespace['reason'] is None and other._view_heartbeat_state is not process._view_heartbeat_state


def test_pilot_subset_keeps_predeclared_order_and_refuses_unknown_or_formal():
    from types import SimpleNamespace
    obj = SimpleNamespace(windows=[('a',), ('b',), ('c',)])
    obj.sample_key = lambda i: obj.windows[i][0]
    train.restrict_pilot_windows(obj, {'stage': 'pilot', 'pilot_train_sample_keys': ['c', 'a']})
    assert obj.windows == [('c',), ('a',)]
    with pytest.raises(ValueError, match='absent'):
        train.restrict_pilot_windows(obj, {'stage': 'pilot', 'pilot_train_sample_keys': ['missing']})
    with pytest.raises(ValueError, match='only pilot'):
        train.restrict_pilot_windows(obj, {'stage': 'formal', 'pilot_train_sample_keys': ['a']})


@pytest.mark.parametrize('frozen', (False, True))
def test_checkpoint_preserves_sigma_functional_flag_before_any_next_update(tmp_path, frozen):
    from types import SimpleNamespace
    cfg = train.get_builder('visual').effective_config(config(tmp_path))
    original = train.get_builder('visual').build(cfg).eval()
    original.pose_head.set_sigma_frozen(frozen)
    optimizer = torch.optim.Adam(original.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    sampler = train.GroupBudgetSampler([4, 4], ['a', 'b'], 1)
    selector = train.JointSelector()
    checkpoint = train._visual_checkpoint(original, optimizer, scheduler, scaler, sampler, selector,
        cfg, {'successful_updates': 1, 'sigma_frozen': frozen}, cfg['_test_stats'],
        SimpleNamespace(state_dict=lambda: {}), torch.device('cpu'))
    assert checkpoint['model_runtime']['sigma_frozen'] is frozen
    train.atomic_checkpoint(tmp_path/'sigma.pt', checkpoint)
    restored = train.get_builder('visual').build(cfg).eval()
    other = torch.optim.Adam(restored.parameters(), lr=.001)
    train.load_visual_checkpoint(tmp_path/'sigma.pt', cfg, restored, other,
        torch.optim.lr_scheduler.LambdaLR(other, lambda _: 1.), torch.amp.GradScaler('cuda', enabled=False),
        train.GroupBudgetSampler([4, 4], ['a', 'b'], 1), train.JointSelector(), torch.device('cpu'))
    assert all(layer.sigma_frozen is frozen for layer in restored.pose_head.decoder.layers)
    features = torch.randn(1, 20, 2051);valid = torch.ones(1, 20, dtype=torch.bool)
    times = torch.arange(20, dtype=torch.float64)[None]/40
    with torch.no_grad():
        a = original.forward_visual(features, valid, times)
        b = restored.forward_visual(features, valid, times)
    for key in ('x0_hat', 'pressure_hat', 'v_hat', 'gate_sigma'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    checkpoint['schedule']['sigma_frozen'] = not frozen
    with pytest.raises(ValueError, match='disagree'):
        train.restore_visual_runtime_state(restored, checkpoint)


def test_amp_backoff_replays_same_batch_rng_and_native_found_inf_without_budget(tmp_path):
    cfg = config(tmp_path)
    batch = grouped_collate([ToyGroups()[0]])
    model = student();reference = copy.deepcopy(model)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    other = torch.optim.Adam(reference.parameters(), lr=.001)
    # Native CPU GradScaler exercises found_inf/update state without CUDA.
    scaler = torch.amp.GradScaler('cpu', init_scale=2.)
    direct_scaler = torch.amp.GradScaler('cpu', init_scale=1.)
    handle = next(model.parameters()).register_hook(lambda grad: grad*float('inf'))
    state = {'attempt_steps': 0, 'successful_updates': 0, 'rgb_windows': 0,
             'last_sample_keys': batch['meta']['sample_key']}
    events = []
    rng = train.capture_training_rng(torch.device('cpu'))
    def overflow(event):
        events.append(event)
        assert not optimizer.state
        assert_state_equal(initial, model.state_dict())
        assert state['successful_updates'] == state['rgb_windows'] == 0
        handle.remove()
    result = train.visual_update_with_amp_retries(model, [batch], cfg, optimizer, scaler,
        torch.device('cpu'), 0, 4, state, overflow)
    after_rng = train.capture_training_rng(torch.device('cpu'))
    train.restore_training_rng(rng, torch.device('cpu'))
    direct = train.visual_accumulation_step(reference, [batch], cfg, other, direct_scaler,
                                            torch.device('cpu'), 0, 4)
    assert result['stepped'] and direct['stepped']
    assert_state_equal(model.state_dict(), reference.state_dict())
    assert_state_equal(optimizer.state_dict(), other.state_dict())
    assert_state_equal(after_rng, train.capture_training_rng(torch.device('cpu')))
    assert_state_equal(scaler.state_dict(), direct_scaler.state_dict())
    assert state['attempt_steps'] == 2 and state['amp_overflow_attempts'] == 1
    assert state['successful_updates'] == state['rgb_windows'] == 0
    assert events[0]['scale_before'] == 2. and events[0]['scale_after'] == 1.


@pytest.mark.parametrize('initial_scale,limit,error', [(1.,16,'unscaled'), (2.,0,'limit')])
def test_amp_backoff_hard_failure_at_scale_one_or_retry_cap(tmp_path, initial_scale, limit, error):
    cfg = config(tmp_path, max_amp_backoffs_per_update=limit)
    model = student();initial = copy.deepcopy(model.state_dict())
    next(model.parameters()).register_hook(lambda grad: grad*float('inf'))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    state = {'attempt_steps': 0, 'successful_updates': 0, 'rgb_windows': 0,
             'last_sample_keys': ['fixed']}
    with pytest.raises(FloatingPointError, match=error):
        train.visual_update_with_amp_retries(model, [grouped_collate([ToyGroups()[0]])], cfg,
            optimizer, torch.amp.GradScaler('cpu', init_scale=initial_scale),
            torch.device('cpu'), 0, 4, state)
    assert not optimizer.state
    assert_state_equal(initial, model.state_dict())
    assert state['successful_updates'] == state['rgb_windows'] == 0


@pytest.mark.parametrize('invalid', [True, -1, 17, 3.5])
def test_amp_backoff_cap_is_bounded_exact_integer(tmp_path, invalid):
    with pytest.raises(ValueError, match='integer from 0 through 16'):
        train.visual_update_with_amp_retries(None, [], {'max_amp_backoffs_per_update':invalid},
            None, None, torch.device('cpu'), 0, 4, {})


def _resource_eval_racer(path, identity, ready, finish, output):
    """Child holds a real live PID lease until the parent sees both results."""
    ledger = runner.SlotLedger(Path(path))
    key = None
    try:
        ready.wait(timeout=15)
        key = ledger.claim([(0, s) for s in range(3)], identity, job_type='eval',
                           job_class='eval', max_job_class_per_gpu=1, resource_version='CPU_envelope_v1')
        output.put((identity, key))
        if not finish.wait(timeout=15):
            raise TimeoutError('CPU race fixture cleanup timeout')
    finally:
        if key is not None:
            ledger.release(key, identity, 0)


def test_resource_eval_cap_is_atomic_across_processes(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    ready, finish, output = ctx.Barrier(3), ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=_resource_eval_racer,
                args=(str(tmp_path/'slots.json'), f'CPU_eval_{i}', ready, finish, output)) for i in range(2)]
    for child in children:
        child.start()
    try:
        ready.wait(timeout=15)
        rows = [output.get(timeout=15) for _ in children]
        assert sum(key is not None for _, key in rows) == 1
        live = [v for v in json.loads((tmp_path/'slots.json').read_text())['slots'].values() if v['state'] == 'active']
        assert len(live) == 1 and live[0]['job_class'] == 'eval'
        assert train.process_starttime(live[0]['pid']) == live[0]['starttime']
    finally:
        finish.set()
        for child in children:
            child.join(timeout=15)
            if child.is_alive():
                child.terminate();child.join(timeout=5)
        output.close();output.join_thread()
    assert all(child.exitcode == 0 for child in children)
    assert all(v['state'] == 'released' for v in json.loads((tmp_path/'slots.json').read_text())['slots'].values())
    assert not torch.cuda.is_initialized()


def test_resource_train_eval_share_total_and_eval_class_cap(tmp_path):
    ledger = runner.SlotLedger(tmp_path/'slots.json');slots = [(0, i) for i in range(3)]
    args = {'resource_version': 'CPU_envelope_v1'}
    leases = [(ledger.claim(slots, f'train{i}', **args), f'train{i}') for i in range(2)]
    key = ledger.claim(slots, 'eval1', job_type='n_calibration_eval', job_class='eval',
                       max_job_class_per_gpu=1, **args)
    leases.append((key, 'eval1'));assert key == '0:2'
    assert ledger.claim(slots, 'eval2', job_type='eval', max_job_class_per_gpu=1, **args) is None
    assert ledger.claim(slots, 'train2', **args) is None
    # A legitimate old alias lacking class still consumes the same eval cap.
    with ledger.transaction() as data:
        data['slots'][key].pop('job_class')
    ledger.release('0:0', 'train0', 0);leases.remove(('0:0', 'train0'))
    assert ledger.claim(slots, 'eval3', job_type='eval', max_job_class_per_gpu=1, **args) is None
    for key, identity in leases:
        ledger.release(key, identity, 0)
    key = ledger.claim(slots, 'eval4', job_type='eval', max_job_class_per_gpu=1, **args)
    assert key == '0:0';ledger.release(key, 'eval4', 0)


@pytest.mark.parametrize('cap', [None, 0, -1, True, 1.5, 4])
def test_resource_eval_cap_invalid_rejected_before_claim(tmp_path, cap):
    ledger = runner.SlotLedger(tmp_path/'slots.json')
    with pytest.raises(ValueError):
        ledger.claim([(0, i) for i in range(3)], 'bad', job_type='eval', max_job_class_per_gpu=cap)
    assert not ledger.path.exists()


def test_resource_stale_exact_starttime_and_release_isolation(tmp_path):
    ledger = runner.SlotLedger(tmp_path/'slots.json');slots = [(0, i) for i in range(2)]
    first = ledger.claim(slots, 'old', resource_version='V1')
    with ledger.transaction() as data:
        data['slots'][first]['starttime'] = 'wrong_reused_PID_starttime'
    second = ledger.claim(slots, 'new', resource_version='V1')
    assert first == second
    before = json.loads(ledger.path.read_text())
    with pytest.raises(RuntimeError, match="another"):
        ledger.release(second, 'different_identity', 0)
    assert json.loads(ledger.path.read_text()) == before
    third = ledger.claim(slots, 'also_live', resource_version='V1')
    assert third == '0:1'
    for key, identity in [(second, 'new'), (third, 'also_live')]:
        ledger.release(key, identity, 0)


def test_resource_unknown_class_version_and_total_mismatch_rejected(tmp_path):
    ledger = runner.SlotLedger(tmp_path/'slots.json');slots = [(0, i) for i in range(3)]
    key = ledger.claim(slots, 'owner', resource_version='V1')
    for version in ['V2', None]:
        with pytest.raises(RuntimeError, match='incompatible'):
            ledger.claim(slots, 'other', resource_version=version)
    with ledger.transaction() as data:
        data['slots'][key]['job_type'] = 'unknown_live_purpose'
    with pytest.raises(RuntimeError, match='unknown'):
        ledger.claim(slots, 'other', resource_version='V1')
    with ledger.transaction() as data:
        data['slots'][key]['job_type'] = 'train'
        data['slots'][key]['resource_version'] = None
    with pytest.raises(RuntimeError, match='incompatible'):
        ledger.claim(slots, 'other', resource_version='V1')
    ledger.release(key, 'owner', 0)
    leases = [(ledger.claim(slots, f'x{i}', resource_version='V1'), f'x{i}') for i in range(3)]
    with pytest.raises(RuntimeError, match='total slot'):
        ledger.claim([(0, 0), (0, 1)], 'smaller', resource_version='V1')
    for key, identity in leases:
        ledger.release(key, identity, 0)
    with pytest.raises(ValueError, match='conflicting'):
        ledger.claim(slots, 'wrong_class', job_type='n_calibration_eval', job_class='train', max_job_class_per_gpu=1)
    with pytest.raises(ValueError, match='complete'):
        ledger.claim([(0, 1)], 'single_preselected_slot')


def test_resource_slot_major_real_ledger_15_and_partial_gpus(tmp_path):
    # Shared helper is called by both the actual runner and benchmark protocol.
    import ast
    source = ast.parse(Path(runner.__file__).read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run_visual_plan')
    node = next(n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'slots' for t in n.targets))
    actual = eval(compile(ast.Expression(node.value), runner.__file__, 'eval'),
                  {'visual_slot_assignments': runner.visual_slot_assignments, 'gpus': list(range(7)), 'n': 3})
    ledger = runner.SlotLedger(tmp_path/'slots.json');leases = []
    for i in range(15):
        key = ledger.claim(actual, f'job{i}', resource_version='CPU_V1');assert key is not None
        leases.append((key, f'job{i}'))
    assert [sum(int(k.split(':')[0]) == g for k, _ in leases) for g in range(7)] == [3, 2, 2, 2, 2, 2, 2]
    for key, identity in leases:
        ledger.release(key, identity, 0)
    for count in range(1, 8):
        gpus = list(range(7-count, 7))
        for n in [1, 2, 3]:
            for jobs in [1, 5, 15]:
                assigned = runner.visual_slot_assignments(gpus, n, jobs)
                assert len(assigned) == min(jobs, count*n)
                occupancy = [sum(g == gpu for g, _ in assigned) for gpu in gpus]
                assert max(occupancy)-min(occupancy) <= 1 and max(occupancy) <= n
                assert all(g in gpus for g, _ in assigned)
    with pytest.raises(ValueError):runner.visual_slot_assignments([True], 1)
    with pytest.raises(ValueError):runner.visual_slot_assignments([0, 0], 1)
    with pytest.raises(ValueError):runner.visual_slot_assignments([0], True)
    assert not torch.cuda.is_initialized()


def test_resource_total_envelope_change_rejected_even_when_unoccupied(tmp_path):
    for old_n, new_n in [(2, 3), (3, 2)]:
        ledger = runner.SlotLedger(tmp_path/f'slots_{old_n}_{new_n}.json')
        key = ledger.claim([(0, i) for i in range(old_n)], 'original', resource_version='same_V')
        assert key == '0:0'  # Old envelope is not full; occupied slot lies in both sets.
        before = json.loads(ledger.path.read_text())
        with pytest.raises(RuntimeError, match='total slot'):
            ledger.claim([(0, i) for i in range(new_n)], 'different_total', job_type='eval',
                         max_job_class_per_gpu=1, resource_version='same_V')
        assert json.loads(ledger.path.read_text()) == before
        with ledger.transaction() as data:
            data['slots'][key].pop('total_slots_per_gpu')
        with pytest.raises(RuntimeError, match='total slot'):
            ledger.claim([(0, i) for i in range(old_n)], 'missing_proof', resource_version='same_V')
        ledger.release(key, 'original', 0)


def make_synthetic_cpu_resource_plan(tmp_path, *, n=3, eval_cap=1, gpus=(0,), version='cpu-resources.v1'):
    """Complete explicitly synthetic integrity chain; never a GPU authorization.

    The graph exercises file hashes/current production source, not measurement
    truth. Callers must keep NVML/worker launch stubbed in CPU boundary tests.
    """
    tmp_path = Path(tmp_path).resolve()
    if not tmp_path.is_relative_to(Path('/private/Tactile')):
        raise ValueError('Synthetic CPU fixtures must stay inside the workspace')
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path/'SYNTHETIC_CPU_ONLY_resource_chain'
    if root.exists():
        import uuid
        root = tmp_path/('SYNTHETIC_CPU_ONLY_resource_chain_'+uuid.uuid4().hex[:10])
    root.mkdir()
    marker = {'SYNTHETIC_CPU_ONLY': True, 'NOT_ACTUAL_MEASUREMENT_OR_AUTHORIZATION': True}
    def store(name, value, *, yaml_file=False):
        path = root/name
        path.write_text(yaml.safe_dump({**marker, **value}) if yaml_file else json.dumps({**marker, **value}, sort_keys=True))
        return {'path': str(path), 'sha256': runner.sha256_file(path)}
    fingerprints = {key: 'CPU_SYNTHETIC_'+key for key in
        ['manifest_hash','split_hash','stats_hash','stats_file_hash','frontend_hash',
         'frontend_runtime_hash','cache_index_hash','smpl_runtime_model_hash']}
    fingerprints['source_hash'] = train.visual_source_hash()
    selected = {**fingerprints, 'pressure_head':'time_pe', 'structure_hash':'CPU_SYNTHETIC_time_pe_structure',
                'visual_structure_version':'visual_v1', 'pressure_time_pe_enabled':True,
                'pressure_time_pe':'before_fusion','time_pe_scale':40.0,'d_model':256,'nhead':8,
                'tw':20,'pose_layers':6,'modal':'anysolev2','pose_repr':'6d'}
    final_config = store('SYNTHETIC_final_config.yaml', selected, yaml_file=True)
    protocol = {**marker, 'schema':'anysole.n-benchmark.v2','n_candidates':[n], 'single_gpu':list(gpus)[0],
                'seven_gpus':list(gpus),'single_warmup':50,'single_timed':200,'single_repeats':2,
                'seven_warmup':50,'seven_timed':50,'seven_repeats':2,'mixed_eval_per_gpu':eval_cap}
    protocol['plan_hash'] = runner.plan_hash(protocol)
    protocol_ref = store('SYNTHETIC_benchmark_plan.json', protocol)
    scripts = {}
    for name in ('protocol.py','benchmark.py','worker.py'):
        path = root/name;path.write_text('# SYNTHETIC_CPU_ONLY never executed\n')
        scripts[name] = runner.sha256_file(path)
    receipts = {}
    for field, value in [('todo080_gate',{'passed':True}),('independent_receipt',{'gate':'pass'}),
                         ('eval_cap_receipt',{'gate':'pass','scope':'PER_GPU_EVAL_CAP','production_source_sha256':fingerprints['source_hash']}),
                         ('queue_policy_receipt',{'gate':'pass','scope':'SLOT_MAJOR_15_TASK_QUEUE','production_source_sha256':fingerprints['source_hash']})]:
        ref = store('SYNTHETIC_'+field+'.json', value)
        receipts[field+'_path'], receipts[field+'_sha256'] = ref['path'], ref['sha256']
    authorization = store('SYNTHETIC_NOT_ACTUAL_authorization.json', {
        'scope':'TODO090_REAL_N_ONLY','approved_by':'root','measurement_authorized':True,
        'plan_hash':protocol['plan_hash'],'input_fingerprints':fingerprints,
        'final_config_path':final_config['path'],'final_config_sha256':final_config['sha256'],
        'script_sha256':scripts,'formal_update_horizon':5180,'formal_validation_every':104,
        'formal_checkpoint_every':104,'approved_formal_run_count':15,**receipts})
    records = []
    for phase in ('single','seven','mixed'):
        cards = [list(gpus)[0]] if phase == 'single' else list(gpus)
        warm, timed = 50, (200 if phase == 'single' else 50)
        for repeat in (1,2):
            workers = []
            for gpu, slot in runner.visual_slot_assignments(cards,n,15):
                mode = 'eval' if phase == 'mixed' and slot < eval_cap else 'train'
                worker = {**marker,'passed':True,'mode':mode,'physical_gpu':gpu,'n':n,
                          'input_fingerprints':fingerprints,'final_config_sha256':final_config['sha256']}
                if mode == 'train':
                    worker.update(warmup_updates=warm,timed_updates=timed,total_successful_updates=warm+timed,complete_val_save=True)
                else:
                    worker.update(optimizer_created=False,full_val_metrics={'CPU_SYNTHETIC':True},prediction_exports=1)
                workers.append(worker)
            records.append(store(f'SYNTHETIC_{phase}_n{n}_r{repeat}_summary.json',{
                'schema':protocol['schema'],'phase':phase,'n':n,'repeat':repeat,'all_pass':True,
                'failure_rate':0.,'memory_envelope':{'passed':True},'workers':workers,
                'benchmark_plan_hash':protocol['plan_hash'],'authorization_sha256':authorization['sha256'],
                'final_config_sha256':final_config['sha256'],'input_fingerprints':fingerprints}))
    report = store('SYNTHETIC_NOT_ACTUAL_measurement_report.json',{
        'schema':'anysole.resource-candidate.v1','frozen':False,'mixed_evaluation_passed':True,
        'gpus':list(gpus),'runs_per_gpu_candidate':n,'eval_runs_per_gpu_candidate':eval_cap,'formal_task_count':15,
        'input_fingerprints':fingerprints,'final_config':final_config,'selected_head_config_sha256':final_config['sha256'],
        'benchmark_plan':protocol_ref,'benchmark_plan_hash':protocol['plan_hash'],
        'authorization':authorization,'measurement_records':records})
    plan = {**marker,'schema':'anysole.visual-plan.v1','frozen':True,'candidate_only':False,
            'base_config':{**selected,'control_dir':str(tmp_path/'control')},'runs':[],
            'resources':{'frozen':True,'gpus':list(gpus),'runs_per_gpu':n,'eval_runs_per_gpu':eval_cap,
                         'version':version,'measurement':report}}
    plan['plan_hash'] = runner.plan_hash(plan)
    path = tmp_path/'SYNTHETIC_CPU_ONLY_resource_plan.json'
    path.write_text(json.dumps(plan,sort_keys=True))
    return path, plan


def _rewrite_synthetic_reference(reference, value, *, yaml_file=False):
    path = Path(reference['path'])
    path.write_text(yaml.safe_dump(value) if yaml_file else json.dumps(value, sort_keys=True))
    reference['sha256'] = runner.sha256_file(path)


def test_resource_binding_complete_chain_and_real_train_boundary(tmp_path, monkeypatch):
    path, plan = make_synthetic_cpu_resource_plan(tmp_path)
    assert runner.validate_visual_resource_plan(plan) == plan['resources']
    assert not torch.cuda.is_initialized()
    reached = []
    class BoundaryReached(Exception):pass
    def owners(*args):reached.append('validated_then_NVML_boundary');raise BoundaryReached
    monkeypatch.setattr(runner, 'check_visual_gpu_owners', owners)
    monkeypatch.setattr(runner.SlotLedger, 'claim', lambda *a, **kw: pytest.fail('No CPU test may claim for a real launch'))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No worker may launch'))
    with pytest.raises(BoundaryReached):runner.run_visual_plan(path)
    assert reached == ['validated_then_NVML_boundary']
    # Old-hash tampering is rejected before NVML, any claim or worker launch.
    reached.clear();plan['resources'].update(runs_per_gpu=4, eval_runs_per_gpu=2)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='hash'):runner.run_visual_plan(path)
    assert not reached


@pytest.mark.parametrize('damage', [
    'plan_schema','frozen_bool','no_measurement','measurement_hash','measurement_schema','unfinished_mixed',
    'n','cap','gpus','config_drift','protocol_hash','protocol_link','authorization_denied',
    'authorization_protocol','fingerprints','head','summary_hash','summary_incomplete','summary_protocol',
    'summary_n','missing_repeat','incomplete_train','incomplete_eval','relative_path','outside_path'])
def test_resource_binding_rejects_broken_chain_before_GPU(tmp_path, monkeypatch, damage):
    path, plan = make_synthetic_cpu_resource_plan(tmp_path)
    measurement = plan['resources']['measurement']
    report = json.loads(Path(measurement['path']).read_text())
    if damage == 'plan_schema':plan['schema'] = 'wrong'
    elif damage == 'frozen_bool':plan['resources']['frozen'] = 1
    elif damage == 'no_measurement':plan['resources']['measurement'] = None
    elif damage == 'measurement_hash':measurement['sha256'] = '0'*64
    elif damage == 'measurement_schema':report['schema'] = 'wrong'
    elif damage == 'unfinished_mixed':report['mixed_evaluation_passed'] = False
    elif damage == 'n':plan['resources']['runs_per_gpu'] = 2
    elif damage == 'cap':plan['resources']['eval_runs_per_gpu'] = 2
    elif damage == 'gpus':plan['resources']['gpus'] = [1]
    elif damage == 'config_drift':
        file = Path(report['final_config']['path']);file.write_text(file.read_text()+'\n# changed after hashing\n')
    elif damage == 'protocol_hash':report['benchmark_plan']['sha256'] = '0'*64
    elif damage == 'protocol_link':report['benchmark_plan_hash'] = '0'*64
    elif damage in ('authorization_denied','authorization_protocol'):
        auth = json.loads(Path(report['authorization']['path']).read_text())
        auth['measurement_authorized' if damage == 'authorization_denied' else 'plan_hash'] = False if damage == 'authorization_denied' else '0'*64
        _rewrite_synthetic_reference(report['authorization'], auth)
    elif damage == 'fingerprints':report['input_fingerprints']['source_hash'] = '0'*64
    elif damage == 'head':plan['base_config']['pressure_head'] = 'legacy'
    elif damage == 'summary_hash':report['measurement_records'][-1]['sha256'] = '0'*64
    elif damage in ('summary_incomplete','summary_protocol','summary_n','incomplete_train','incomplete_eval'):
        ref = report['measurement_records'][0 if damage == 'incomplete_train' else -1]
        summary = json.loads(Path(ref['path']).read_text())
        if damage == 'summary_incomplete':summary['all_pass'] = False
        elif damage == 'summary_protocol':summary['benchmark_plan_hash'] = '0'*64
        elif damage == 'summary_n':summary['n'] = 2
        elif damage == 'incomplete_train':summary['workers'][0]['complete_val_save'] = False
        else:summary['workers'][0]['prediction_exports'] = 0
        _rewrite_synthetic_reference(ref, summary)
    elif damage == 'missing_repeat':report['measurement_records'].pop()
    elif damage == 'relative_path':measurement['path'] = 'relative_file.json'
    elif damage == 'outside_path':measurement['path'] = '/tmp/OUTSIDE_WORKSPACE_DO_NOT_READ.json'
    if damage not in ('no_measurement','measurement_hash','relative_path','outside_path'):
        _rewrite_synthetic_reference(measurement, report)
    plan['plan_hash'] = runner.plan_hash(plan);path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'check_visual_gpu_owners', lambda *a: pytest.fail('NVML must not run on rejected provenance'))
    monkeypatch.setattr(runner.SlotLedger, 'claim', lambda *a, **kw: pytest.fail('No claim on rejected provenance'))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No worker on rejected provenance'))
    with pytest.raises(ValueError):runner.run_visual_plan(path)
    assert not torch.cuda.is_initialized()


def test_resource_binding_current_source_is_required(tmp_path, monkeypatch):
    _, plan = make_synthetic_cpu_resource_plan(tmp_path)
    monkeypatch.setattr(train, 'visual_source_hash', lambda:'different_current_source')
    with pytest.raises(ValueError, match='current production source'):
        runner.validate_visual_resource_plan(plan)


def test_resource_binding_script_symlink_cannot_read_outside_workspace(tmp_path, monkeypatch):
    _, plan = make_synthetic_cpu_resource_plan(tmp_path)
    report = json.loads(Path(plan['resources']['measurement']['path']).read_text())
    script = Path(report['benchmark_plan']['path']).parent/'worker.py'
    script.unlink();script.symlink_to('/opt/conda/envs/Tactile/bin/python')
    original = runner.sha256_file;checked = []
    def digest(path):checked.append(Path(path).resolve());return original(path)
    monkeypatch.setattr(runner,'sha256_file',digest)
    with pytest.raises(ValueError, match='script content mismatch'):
        runner.validate_visual_resource_plan(plan)
    assert Path('/opt/conda/envs/Tactile/bin/python').resolve() not in checked


def make_synthetic_cpu_short_resource_plan(tmp_path, *, damage=None):
    """Honest synthetic graph for the new short admission; never a GPU permit."""
    root = Path(tmp_path).resolve()/'SYNTHETIC_CPU_ONLY_short_chain'
    root.mkdir(parents=True)
    marker = {'SYNTHETIC_CPU_ONLY':True,'NOT_ACTUAL_MEASUREMENT_OR_AUTHORIZATION':True}
    def store(name, value, yaml_file=False):
        p=root/name
        p.write_text(yaml.safe_dump({**marker,**value}) if yaml_file else json.dumps({**marker,**value},sort_keys=True))
        return {'path':str(p),'sha256':runner.sha256_file(p)}
    fp={k:'CPU_SYNTHETIC_'+k for k in ('manifest_hash','split_hash','stats_hash','stats_file_hash',
        'frontend_hash','frontend_runtime_hash','cache_index_hash','smpl_runtime_model_hash')}
    fp['source_hash']=train.visual_source_hash()
    oldfp={**fp,'source_hash':'CPU_SYNTHETIC_OLD_SOURCE'}
    cfg={**fp,'pressure_head':'time_pe','structure_hash':'CPU_SYNTHETIC_STRUCTURE','lambda_mv':.2,
        'mv_pose_weight':1.,'mv_pressure_weight':1.,'pair_shuffle':False,'amp':True,'num_workers':0,
        'effective_batch_rgb_windows':256,'microbatch_rgb_windows':8,'deterministic_algorithms':True,'save_initial_checkpoint':False}
    config=store('SYNTHETIC_current.yaml',cfg,True)
    oldcfg={**cfg,'source_hash':oldfp['source_hash']}
    if damage=='scientific_config':oldcfg['lambda_mv']=.1
    oldconfig=store('SYNTHETIC_parent.yaml',oldcfg,True)
    oldprotocol={'schema':'anysole.n-benchmark.v2','n_candidates':[1,2,3],'single_gpu':0,
        'single_warmup':50,'single_timed':200,'single_repeats':2}
    oldprotocol['plan_hash']=runner.plan_hash({**marker,**oldprotocol})
    oldplan=store('SYNTHETIC_old_protocol.json',oldprotocol)
    oldauthorization=store('SYNTHETIC_old_authorization.json',{'scope':'TODO090_REAL_N_ONLY','approved_by':'root',
        'measurement_authorized':True,'plan_hash':oldprotocol['plan_hash'],'input_fingerprints':oldfp,
        'final_config_path':oldconfig['path'],'final_config_sha256':oldconfig['sha256']})
    parents=[]
    for n in (1,2,3):
        for repeat in (1,2):
            workers=[{'passed':True,'mode':'train','physical_gpu':0,'n':n,'warmup_updates':50,'timed_updates':200,
                'total_successful_updates':250,'complete_val_save':True,'input_fingerprints':oldfp,
                'final_config_sha256':oldconfig['sha256']} for _ in range(n)]
            parents.append(store(f'SYNTHETIC_parent_n{n}_r{repeat}.json',{'schema':oldprotocol['schema'],
                'phase':'single','n':n,'repeat':repeat,'all_pass':True,'failure_rate':0.,'memory_envelope':{'passed':True},
                'benchmark_plan_hash':oldprotocol['plan_hash'],'authorization_sha256':oldauthorization['sha256'],
                'final_config_sha256':oldconfig['sha256'],'input_fingerprints':fp if damage=='parent_source' else oldfp,'workers':workers}))
    if damage=='missing_parent':parents=parents[:-1]
    selection=store('SYNTHETIC_single_selection.json',{'schema':'anysole.inherited-single-selection.v1',
        'new_single_measurements':0,'selected_n':3,'seven_card_throughput_comparison':None,
        'protocol':oldplan,'authorization':oldauthorization,'final_config':oldconfig,'measurement_records':parents,
        'rows':[{'n':n,'eligible':True,'forecast_seconds':1000./n,'seven_card_rgb_per_second':None} for n in (1,2,3)]})
    acceptance=store('SYNTHETIC_parent_acceptance.json',{'gate':'pass','scope':'ACTUAL_FULL_SINGLE_PARENT_ONLY',
        'selected_candidate_for_short_confirmation':3,'old_source_hash':oldfp['source_hash'],
        'six_original_summary_references':[{**r,'n':1+i//2,'repeat':1+i%2,'independent_review_sha256':'CPU_SYNTHETIC'} for i,r in enumerate(parents)],
        'old_config':oldconfig,'old_permit':oldauthorization})
    bridge=store('SYNTHETIC_source_bridge.json',{'schema':'anysole.resource-admission-source-bridge.v1','gate':'pass',
        'old_source_hash':oldfp['source_hash'],'new_source_hash':fp['source_hash'],'learning_implementation_unchanged':True,
        'changed_production_paths':['anysole/train.py'] if damage=='source_bridge' else ['configs/tools/runner.py'],
        'config_changed_fields':['source_hash']})
    amendment=store('SYNTHETIC_user_amendment.json',{'schema':'anysole.user-test-scope-amendment.v1','approved_by':'root',
        'authority':'missing_user' if damage=='amendment' else 'latest_explicit_user_instruction'})
    protocol={'schema':'anysole.n-benchmark.user-shortened.v1','n_candidates':[3],'single_gpu':0,'seven_gpus':list(range(7)),
        'single_warmup':0,'single_timed':0,'single_repeats':0,'seven_warmup':5,'seven_timed':19 if damage=='workload' else 20,
        'seven_repeats':1,'mixed_warmup':5,'mixed_timed':20,'mixed_repeats':1,'mixed_eval_per_gpu':1,
        'seven_max_jobs':15,'higher_n_allowed':False,'eval_bootstrap_samples':1000,'user_amendment':amendment,
        'single_parent':{'selection':selection,'acceptance':acceptance,'source_bridge':bridge}}
    protocol['plan_hash']=runner.plan_hash({**marker,**protocol})
    plan_ref=store('SYNTHETIC_short_protocol.json',protocol)
    scripts={}
    for name in ('protocol.py','benchmark.py','worker.py'):
        p=root/name;p.write_text('# SYNTHETIC_CPU_ONLY never executed\n');scripts[name]=runner.sha256_file(p)
    receipts={}
    for name,value in [('todo080_gate',{'passed':True}),('independent_receipt',{'gate':'pass',
            'scope':'WRONG' if damage=='preparation' else 'USER_SHORTENED_N_PREPARATION',
            'source_hash':fp['source_hash'],'plan_hash':protocol['plan_hash']}),
            ('eval_cap_receipt',{'gate':'pass','scope':'PER_GPU_EVAL_CAP','production_source_sha256':fp['source_hash']}),
            ('queue_policy_receipt',{'gate':'pass','scope':'SLOT_MAJOR_15_TASK_QUEUE','production_source_sha256':fp['source_hash']})]:
        ref=store('SYNTHETIC_'+name+'.json',value);receipts[name+'_path'],receipts[name+'_sha256']=ref['path'],ref['sha256']
    auth=store('SYNTHETIC_NOT_ACTUAL_short_authorization.json',{'scope':'TODO090_REAL_N_ONLY','approved_by':'root',
        'measurement_authorized':True,'plan_hash':protocol['plan_hash'],'input_fingerprints':fp,
        'final_config_path':config['path'],'final_config_sha256':config['sha256'],'script_sha256':scripts,
        'formal_update_horizon':5180,'formal_validation_every':100,'formal_checkpoint_every':100,'approved_formal_run_count':15,
        'user_amendment_path':amendment['path'],'user_amendment_sha256':amendment['sha256'],
        'parent_single_acceptance_sha256':acceptance['sha256'],'source_bridge_sha256':bridge['sha256'],**receipts})
    records=[]
    for phase in ('seven','mixed'):
        workers=[]
        for gpu,slot in runner.visual_slot_assignments(list(range(7)),3,15):
            mode='eval' if phase=='mixed' and slot==0 else 'train'
            w={'passed':True,'mode':mode,'physical_gpu':gpu,'n':3,'input_fingerprints':fp,'final_config_sha256':config['sha256']}
            if mode=='train':w.update(warmup_updates=5,timed_updates=20,total_successful_updates=25,complete_val_save=True,
                timed_sigma_unfrozen=damage!='frozen_sigma',timed_lambda_mv=.2,
                timed_pair_count_min={'local23_pair':0 if damage=='zero_pair' else 1,'pressure_pair':1})
            else:w.update(optimizer_created=False,full_val_metrics={'SYNTHETIC':True},prediction_exports=104,
                bootstrap_samples=1000,surfaces=damage!='surface')
            workers.append(w)
        records.append(store(f'SYNTHETIC_short_{phase}.json',{'schema':protocol['schema'],'phase':phase,'n':3,
            'repeat':2 if damage=='repeat' else 1,'all_pass':True,'failure_rate':0.,'memory_envelope':{'passed':True},
            'workers':workers,'benchmark_plan_hash':protocol['plan_hash'],'authorization_sha256':auth['sha256'],
            'final_config_sha256':config['sha256'],'input_fingerprints':fp}))
    if damage=='missing_mixed':records=records[:1]
    report=store('SYNTHETIC_NOT_ACTUAL_short_measurement.json',{'schema':'anysole.resource-candidate.v1','frozen':False,
        'mixed_evaluation_passed':True,'gpus':list(range(7)),'runs_per_gpu_candidate':3,'eval_runs_per_gpu_candidate':1,
        'formal_task_count':15,'input_fingerprints':fp,'final_config':config,'selected_head_config_sha256':config['sha256'],
        'benchmark_plan':plan_ref,'benchmark_plan_hash':protocol['plan_hash'],'authorization':auth,'measurement_records':records})
    plan={**marker,'schema':'anysole.visual-plan.v1','frozen':True,'candidate_only':False,'runs':[],
        'base_config':{**cfg,'control_dir':str(root/'control')},'resources':{'frozen':True,'gpus':list(range(7)),
        'runs_per_gpu':3,'eval_runs_per_gpu':1,'version':'SYNTHETIC_SHORT_V1','measurement':report}}
    plan['plan_hash']=runner.plan_hash(plan);p=root/'SYNTHETIC_CPU_ONLY_short_plan.json';p.write_text(json.dumps(plan))
    return p,plan


def test_user_short_resource_positive_real_boundary(tmp_path,monkeypatch):
    path,plan=make_synthetic_cpu_short_resource_plan(tmp_path)
    assert runner.validate_visual_resource_plan(plan)==plan['resources']
    reached=[]
    class Boundary(Exception):pass
    def owners(*args):reached.append(True);raise Boundary
    monkeypatch.setattr(runner,'check_visual_gpu_owners',owners)
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('CPU-only, no Popen'))
    with pytest.raises(Boundary):runner.run_visual_plan(path)
    assert reached==[True] and not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['workload','scientific_config','missing_parent','parent_source','source_bridge',
    'amendment','preparation','frozen_sigma','zero_pair','surface','repeat','missing_mixed'])
def test_user_short_resource_rejects_before_GPU(tmp_path,monkeypatch,damage):
    path,plan=make_synthetic_cpu_short_resource_plan(tmp_path,damage=damage)
    monkeypatch.setattr(runner,'check_visual_gpu_owners',lambda *a:pytest.fail('Must reject before NVML'))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('Must reject before Popen'))
    with pytest.raises(ValueError):runner.run_visual_plan(path)
    assert not torch.cuda.is_initialized()
