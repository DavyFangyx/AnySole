"""Independent-GPU inference / bounded CPU overlay coordinator and audit.

The coordinator is the only events/summary writer. Camera terminals are the
durable work queue: losing the process never loses successful inference.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

from .controller import ControllerLock, wait_for_gpus, _pid_is_alive
from .dataset import _atomic_write_bytes
from .manifest_index import build_index, load_index, read_unit, identity, file_sha256
from .runner import _atomic_write_json, unit_root_for
from . import compact

PRODUCT_ROOT = Path(os.environ.get('VB_WORKSPACE_ROOT', str(Path(__file__).resolve().parents[2]))) / '3_Result/WHAM_baseline'
CODE_ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return dict(status='invalid', terminal=False, error='invalid JSON: '+str(exc))


def root_check(root):
    root = Path(root).resolve()
    if root != PRODUCT_ROOT and PRODUCT_ROOT not in root.parents:
        raise ValueError('all new pipeline outputs must be inside ' + str(PRODUCT_ROOT))
    return root


def fingerprint(args, idx):
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(args.smpl_model), Path(args.annotation_index)]
    wham = Path(args.wham_root)
    files += [wham / 'demo.py', wham / 'lib/models/preproc/detector.py',
              wham / 'lib/data/datasets/dataset_custom.py']
    files += list((wham / 'checkpoints').glob('*.pth*'))
    files += list((wham / 'checkpoints').glob('*.pt'))
    files += list((wham / 'checkpoints').glob('*.ckpt'))
    files += list((wham / 'lib').rglob('*.py'))
    files += list((wham / 'configs').rglob('*.py'))
    files += list((wham / 'configs/yamls').glob('*.yaml'))
    files += [p for p in (wham / 'dataset/body_models').glob('*') if p.is_file()]
    files += [Path(args.ffmpeg_bin), Path(args.wham_python)]
    if args.mock:
        files = list(Path(__file__).parent.glob('*.py'))
    hashes = {str(f.resolve()): file_sha256(f) for f in sorted(set(files))}
    config = dict(schema_version=1, manifest_sha256=idx['manifest_sha256'],
                  dataset_root=str(Path(args.dataset_root).resolve()),
                  wham_root=str(wham.resolve()), wham_python=str(Path(args.wham_python).resolve()),
                  ffmpeg_bin=str(Path(args.ffmpeg_bin).resolve()),
                  projection='f=sqrt(width^2+height^2), principal_point=(width/2,height/2)',
                  target_bbox=True, local_only=True, fps=args.fps, mock=args.mock,
                  output_layout=compact.LAYOUT if getattr(args,'compact_output',False) else 'legacy', files=hashes)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return digest, config, {p: identity(p) for p in hashes}


def terminal_paths(root, row):
    unit_root = unit_root_for(root, row, row['unit_index'])
    return unit_root, unit_root / 'status.json', unit_root / 'visualization/status.json'


def inference_valid(root, row, digest, mock=False):
    if compact.active(root) and compact.inference_valid(root,row,digest,mock):
        return True
    return cached_inference_valid(root,row,digest,mock)


def cached_inference_valid(root,row,digest,mock=False):
    unit_root, path, _ = terminal_paths(root, row)
    status = read_json(path)
    if not status or status.get('status') != 'succeeded' or status.get('config_hash') != digest:
        return False
    if mock:
        return (unit_root / 'wham_output/mock_result.json').is_file()
    from .validation import prediction_path, validate_prediction, validate_tracking
    try:
        validate_prediction(prediction_path(unit_root), row['frame_count'])
        validate_tracking(unit_root, row['frame_count'])
        command = read_json(unit_root / 'command.json')['command']
        if '--target_bboxes' not in command or '--estimate_local_only' not in command or '--calib' in command:
            return False
        sidecar = read_json(unit_root / 'input/target_bboxes.json')
        if sidecar['frame_count'] != row['frame_count'] or len(sidecar['frames']) != row['frame_count']:
            return False
        if [f['video_frame_index'] for f in sidecar['frames']] != list(range(row['frame_count'])):
            return False
        return True
    except Exception:
        return False


def visualization_valid(root, row, digest, mock=False):
    if compact.active(root):
        return compact.visualization_valid(root,row,digest,mock)
    unit_root, _, path = terminal_paths(root, row)
    status = read_json(path)
    if not status or status.get('status') != 'succeeded' or status.get('config_hash') != digest:
        return False
    if mock:
        return (path.parent / 'mock-overlay.json').is_file()
    from .overlay import verify_video, video_info, PROJECTION
    from .validation import prediction_path
    try:
        meta = read_json(path.parent / 'metadata.json')
        if meta['config_hash'] != digest or meta['projection_policy'] != PROJECTION or meta['joint_count'] != 24 or meta['track_id'] != 0:
            return False
        if meta['prediction_sha256'] != file_sha256(prediction_path(unit_root)):
            return False
        source=unit_root / 'input/sequence.mp4'
        if meta['video_sha256'] != file_sha256(source) or meta['video'] != video_info(source,decode=False):
            return False
        video = path.parent / 'smpl_joints_overlay.mp4'
        if file_sha256(video) != meta['output_sha256']:
            return False
        verify_video(video, meta['video'])
        if meta['video']['frames'] != row['frame_count']:
            return False
        return True
    except Exception:
        return False


def resource_snapshot(root):
    import psutil
    vm = psutil.virtual_memory()
    data = dict(host_used_bytes=vm.used, host_available_bytes=vm.available,
                disk_free_bytes=shutil.disk_usage(root).free)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        data['gpus'] = [dict(index=int(a), memory_mib=int(b), utilization=int(c))
                        for a, b, c in (line.split(',') for line in result.stdout.splitlines())]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        data['gpu_probe_error'] = True
    return data


def failure_kind(text):
    text = text.lower()
    if any(s in text for s in ('cuda out of memory', 'cuda error: out of memory', 'cudaoutofmemory')):
        return 'oom'
    if any(s in text for s in ('no space left', 'disk quota exceeded')):
        return 'disk'
    if any(s in text for s in ('input/output error', 'driver shutting down', 'gpu has fallen', 'timed out', 'signal 9', 'exit code -9')):
        return 'system'
    return 'ordinary'


def environment(gpu=None, slot=None, root=None):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1',
               PYTHONPATH=str(CODE_ROOT))
    if slot is not None:
        config = Path(root) / '.worker-config' / f'slot-{slot}'
        config.mkdir(parents=True, exist_ok=True)
        env['YOLO_CONFIG_DIR'] = str(config)
    return env


def execute(command, env, logfile, timeout, process_path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('ab') as log:
        process = subprocess.Popen(command, env=env, cwd=CODE_ROOT.parent, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _atomic_write_json(process_path, dict(pid=process.pid, pgid=process.pid, command=command, started_at=now()))
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError('worker timed out')


def infer(args, row, slot, digest, attempt, isolated=False):
    root, status_path, _ = terminal_paths(args.result_root, row)
    def publish():
        try:
            compact.publish_inference(args,row,digest)
            return dict(success=True,stage='inference',index=row['unit_index'],slot=slot,
                        attempt=attempt,isolated=isolated)
        except Exception as exc:
            # Keep the already verified raw inference terminal/cache intact.
            # Subsequent attempts retry publication without another GPU run.
            return dict(success=False,stage='inference',index=row['unit_index'],slot=slot,
                        attempt=attempt,isolated=isolated,error=str(exc),kind=failure_kind(str(exc)))
    if args.compact_output and cached_inference_valid(args.result_root,row,digest,args.mock):
        return publish()
    prior = read_json(status_path)
    if prior:
        _atomic_write_json(root / 'attempts' / f'inference-{prior.get("attempt",0)}-{uuid.uuid4().hex[:8]}.json', prior)
    gpu = args.gpus[slot % len(args.gpus)]
    command = [sys.executable if args.mock else args.wham_python, '-m', 'wham_baseline.runner', 'run',
               '--manifest', args.manifest, '--unit-index', str(row['unit_index']),
               '--result-root', args.result_root, '--wham-root', args.wham_root,
               '--force', '--fps', str(args.fps), '--ffmpeg-bin', args.ffmpeg_bin,
               '--config-hash', digest]
    if args.mock:
        command += ['--mock']
        if row['unit_index'] == args.mock_fail_unit and attempt <= args.mock_fail_attempts:
            command += ['--mock-fail']
    else:
        command += ['--use-target-bbox', '--annotation-index', args.annotation_index, '--validate-target-output']
    try:
        code = execute(command, environment(gpu, slot, args.result_root), root / 'worker.log',
                       args.worker_timeout, root / 'worker_process.json')
        status = read_json(status_path) or {}
        if code == 0 and inference_valid(args.result_root, row, digest, args.mock):
            if args.compact_output:
                return publish()
            return dict(success=True, stage='inference', index=row['unit_index'], slot=slot, attempt=attempt, isolated=isolated)
        text = (root / 'run.log').read_text(errors='replace')[-16000:] if (root / 'run.log').exists() else ''
        error = status.get('error', f'worker exit {code}')
        kind = failure_kind(text + '\n' + error)
    except Exception as exc:
        error, kind = str(exc), failure_kind(str(exc))
    # Preserve runner evidence and publish a failed terminal even after launch/validation errors.
    status = read_json(status_path) or {}
    status.update(status='failed', terminal=True, config_hash=digest, error=error,
                  failure_kind=kind, attempt=attempt, owner_pid=os.getpid())
    _atomic_write_json(status_path, status)
    return dict(success=False, stage='inference', index=row['unit_index'], slot=slot,
                attempt=attempt, error=error, kind=kind, isolated=isolated)


def visualize(args, row, digest, attempt):
    root, _, path = terminal_paths(args.result_root, row)
    prior = read_json(path)
    if prior:
        _atomic_write_json(root / 'attempts' / f'visualization-{uuid.uuid4().hex[:8]}.json', prior)
    if args.mock:
        success = not (row['unit_index'] == args.mock_viz_fail_unit and attempt <= args.mock_viz_fail_attempts)
        _atomic_write_json(path, dict(status='succeeded' if success else 'failed', terminal=True,
                                     config_hash=digest, owner_pid=os.getpid(), attempt=attempt))
        if success:
            _atomic_write_json(path.parent / 'mock-overlay.json', dict(frames=row['frame_count']))
            if args.compact_output:
                try:
                    compact.publish_visualization(args,row,digest)
                except Exception as exc:
                    return dict(success=False,stage='visualization',index=row['unit_index'],attempt=attempt,
                                kind=failure_kind(str(exc)),error=str(exc))
        return dict(success=success, stage='visualization', index=row['unit_index'], attempt=attempt,
                    kind='ordinary', error=None if success else 'injected mock visualization failure')
    from .validation import prediction_path
    try:
        video,prediction = compact.prepare_overlay(args,row) if args.compact_output else (root / 'input/sequence.mp4',prediction_path(root))
        command = [sys.executable, '-m', 'wham_baseline.overlay', '--video', str(video),
                   '--prediction', str(prediction), '--output-dir', str(path.parent),
                   '--smpl-model', args.smpl_model, '--ffmpeg-bin', args.ffmpeg_bin, '--config-hash', digest]
        code = execute(command, environment(), path.parent / 'worker.log', args.worker_timeout,
                       path.parent / 'worker_process.json')
        # Compact overlays use the published PKL, so legacy metadata validation
        # cannot require the deleted upstream cache at this stage.
        if code == 0 and (args.compact_output or visualization_valid(args.result_root, row, digest)):
            if args.compact_output:
                compact.publish_visualization(args,row,digest)
            return dict(success=True, stage='visualization', index=row['unit_index'], attempt=attempt)
        record = read_json(path) or {}
        error = record.get('error', f'overlay exit {code}')
    except Exception as exc:
        error = str(exc)
    kind = failure_kind(error)
    _atomic_write_json(path, dict(status='failed', terminal=True, config_hash=digest,
                                 attempt=attempt, error=error, failure_kind=kind))
    return dict(success=False, stage='visualization', index=row['unit_index'], attempt=attempt, error=error, kind=kind)


def audit(args, idx=None, digest=None, locked=False):
    compact.configure(args)
    root = root_check(args.result_root)
    idx = idx or load_index(args.manifest, verify_hash=True)
    if digest is None:
        digest, _, _ = fingerprint(args, idx)
    if not locked:
        with ControllerLock(root / 'pipeline.lock'):
            return audit(args, idx, digest, locked=True)
    failed, pending, issues, counts = [], [], [], Counter()
    outputs = set()
    for row in idx['units']:
        unit_root, ip, vp = terminal_paths(root, row)
        inf, viz = read_json(ip), read_json(vp)
        if args.compact_output:
            published = compact.state(root,row) or {}
            # Published terminals describe final files. Raw running workers
            # still take precedence so a live/orphan process remains visible.
            if published:
                if not inf or inf.get('status') != 'running':
                    inf = dict(published.get('inference',{}),config_hash=published.get('config_hash'))
                if not viz or viz.get('status') != 'running':
                    viz = dict(published.get('visualization',{}),config_hash=published.get('config_hash'))
        good_inf = compact.inference_valid(root,row,digest,args.mock) if args.compact_output else inference_valid(root, row, digest, args.mock)
        if args.compact_output and good_inf and not args.mock:
            try:
                good_inf = compact.verify_roi(args,row)
            except (OSError,ValueError,KeyError):
                good_inf = False
        good_viz = good_inf and visualization_valid(root, row, digest, args.mock)
        counts['inference_succeeded' if good_inf else 'inference_pending_or_failed'] += 1
        counts['visualization_succeeded' if good_viz else 'visualization_pending_or_failed'] += 1
        if good_inf:
            counts['inference_frames'] += row['frame_count']
        if good_viz:
            counts['visualization_frames'] += row['frame_count']
        reasons = []
        for stage, record, good in (('inference', inf, good_inf), ('visualization', viz, good_viz)):
            if record and record.get('config_hash') != digest:
                reasons.append(stage + ':config_hash_mismatch')
            if record and record.get('status') == 'running':
                reasons.append(stage + (':live_running' if _pid_is_alive(record.get('owner_pid')) else ':orphan_running'))
            if record and record.get('status') == 'failed':
                reasons.append(stage + ':failed:' + str(record.get('error', '')))
            if record and record.get('status') == 'invalid':
                reasons.append(stage + ':invalid_terminal:' + str(record.get('error', '')))
            if record and record.get('status') == 'succeeded' and not good:
                reasons.append(stage + ':invalid_output_or_missing_ROI_or_target')
        output = compact.directory(root,row)/'overlay.mp4' if args.compact_output else unit_root / 'visualization/smpl_joints_overlay.mp4'
        if output.exists():
            real = str(output.resolve())
            if real in outputs:
                reasons.append('duplicate_output')
            outputs.add(real)
        entry = dict(unit_index=row['unit_index'], unit_id=row['unit_id'], frames=row['frame_count'], reasons=reasons)
        if reasons:
            issues.append(entry)
        if any((record or {}).get('status') == 'failed' for record in (inf, viz)):
            failed.append(entry)
        if not good_viz:
            pending.append(entry)
    expected_dirs = {terminal_paths(root, r)[0].name for r in idx['units']}
    extra = sorted(p.name for p in (root / 'units').glob('*') if p.is_dir() and p.name not in expected_dirs)
    if args.compact_output:
        expected_public = {compact.short_name(r) for r in idx['units']}
        extra += sorted(p.name for p in root.parent.iterdir() if p.name != '.run' and p.name not in expected_public)
    result = dict(schema_version=1, at=now(), mock=args.mock, config_hash=digest,
                  manifest_sha256=idx['manifest_sha256'], expected_units=idx['unit_count'],
                  expected_frames=idx['frame_count'], counts=dict(counts), issues=issues,
                  unexpected_unit_outputs=extra, clean=not pending and not issues and not extra)
    for name, records in (('failed_units', failed), ('pending_units', pending)):
        _atomic_write_bytes(root / (name + '.jsonl'), ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode())
    _atomic_write_json(root / 'audit.json', result)
    summary = read_json(root / 'summary.json') or {}
    summary.update(audit=result, status='completed' if result['clean'] else 'incomplete')
    _atomic_write_json(root / 'summary.json', summary)
    return result


def plan(args, idx):
    result = dict(units=idx['unit_count'], frames=idx['frame_count'], methods=idx['methods'],
                  sessions=idx['session_count'], manifest_sha256=idx['manifest_sha256'],
                  gpus=args.gpus, workers_per_gpu=args.workers_per_gpu,
                  inference_slots=len(args.gpus)*args.workers_per_gpu, viz_workers=args.viz_workers,
                  viz_queue_limit=args.viz_queue_limit, dispatch='deterministic longest-first dynamic queue',
                  output_layout=compact.LAYOUT if args.compact_output else 'legacy',
                  full_inference_launched=False)
    if args.command in ('full-run', 'launch', 'dry-run'):
        if idx['unit_count'] != 3560 or idx['frame_count'] != 1399904 or idx['session_count'] != 445:
            raise ValueError('full manifest must be exactly 3560 units / 1399904 frames / 445 sessions')
        if idx['methods'] != {'method_1_offset': {'units':1780, 'frames':699952},
                              'method_2_projection': {'units':1780, 'frames':699952}}:
            raise ValueError('method accounting mismatch')
    return result


def run(args):
    compact.configure(args)
    root = root_check(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    idx = load_index(args.manifest, verify_hash=True)
    run_plan = plan(args, idx)
    if args.command == 'bounded-run' and (idx['unit_count'] > args.max_test_units or idx['frame_count'] > args.max_test_frames):
        raise ValueError('bounded manifest exceeds explicitly selected test limits')
    if args.command == 'full-run' and not args.approved_full_run:
        raise ValueError('full inference requires explicit user approval and --approved-full-run')
    digest, config, identities = fingerprint(args, idx)
    free_min = args.min_free_bytes if args.command != 'full-run' else max(10**12, args.min_free_bytes)
    if shutil.disk_usage(root).free < free_min:
        raise ValueError('insufficient free disk capacity')
    config_path = root / 'config.json'
    old = read_json(config_path)
    if old and old.get('config_hash') != digest:
        raise ValueError('config hash changed; use a fresh result root')
    run_id = uuid.uuid4().hex
    with ControllerLock(root / 'pipeline.lock', run_id=run_id):
        if not args.mock:
            wait_for_gpus(args.gpus, timeout=None if args.wait_for_gpu else 0, poll_interval=30)
        _atomic_write_json(config_path, dict(config_hash=digest, config=config))
        if args.compact_output:
            names = [compact.short_name(r) for r in idx['units']]
            if len(names) != len(set(names)):
                raise ValueError('compact output directory collision')
            _atomic_write_json(root/'layout.json',dict(layout=compact.LAYOUT))
        _atomic_write_json(root / 'run.json', dict(run_id=run_id, pid=os.getpid(), pgid=os.getpgid(0),
                           started_at=now(), command=sys.argv, config_hash=digest, plan=run_plan,
                           manifest_sha256=idx['manifest_sha256']))
        stop = [False]
        prior_signals = {}
        def stop_handler(signum, _frame):
            stop[0] = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            prior_signals[sig] = signal.signal(sig, stop_handler)
        rows = idx['units']
        states = {r['unit_index']: dict(inference='pending', visualization='pending') for r in rows}
        pending, backlog, oom = deque(), deque(), deque()
        attempts, viz_attempts = Counter(), Counter()
        started, last_monitor = time.monotonic(), 0.0
        rolling = deque()
        paused = None
        consecutive_system = 0
        inference_started = time.monotonic()
        inference_finished = None
        completed_frames = 0
        last_resources = [None]
        eventfile = (root / 'events.jsonl').open('a', encoding='utf-8')
        def emit(event_kind, **payload):
            entry = dict(at=now(), run_id=run_id, event=event_kind, **payload)
            eventfile.write(json.dumps(entry, sort_keys=True) + '\n')
            eventfile.flush()
            os.fsync(eventfile.fileno())
        def summary(resources=None):
            if resources is not None:
                last_resources[0]=resources
            elapsed = time.monotonic()-started
            counts = {stage: dict(Counter(s[stage] for s in states.values())) for stage in ('inference','visualization')}
            frame_states = {stage: Counter() for stage in counts}
            for i, state in states.items():
                for stage in counts:
                    frame_states[stage][state[stage]] += rows[i]['frame_count']
            frames = {stage: frame_states[stage].get('succeeded', 0) for stage in counts}
            t = time.monotonic()
            while rolling and rolling[0][0] < t-900:
                rolling.popleft()
            fps = sum(n for _,n in rolling) / min(max(elapsed,1),900)
            value = dict(at=now(), run_id=run_id, pid=os.getpid(), pgid=os.getpgid(0),
                         config_hash=digest, manifest_sha256=idx['manifest_sha256'],
                         status='paused' if paused or stop[0] else 'running', pause_reason=paused,
                         units=counts, frames=frames, frame_states={s:dict(c) for s,c in frame_states.items()},
                         total_units=len(rows), total_frames=idx['frame_count'],
                         rolling_fps=fps, eta_seconds=(idx['frame_count']-frames['inference'])/fps if fps else None,
                         elapsed_seconds=elapsed, viz_queue_depth=len(backlog), resources=last_resources[0],
                         inference_batch_seconds=(inference_finished or t)-inference_started,
                         inference_batch_fps=completed_frames/max((inference_finished or t)-inference_started,1))
            _atomic_write_json(root / 'summary.json', value)
            return value
        try:
            # Camera terminals are authoritative; validate before skipping either stage.
            for row in rows:
                i = row['unit_index']
                ur, ip, vp = terminal_paths(root, row)
                for path in (ip, vp):
                    record = read_json(path)
                    if record and record.get('status') == 'running':
                        proc = read_json(path.parent / 'worker_process.json')
                        if _pid_is_alive(record.get('owner_pid')) or (proc and _pid_is_alive(proc.get('pid'))):
                            raise RuntimeError('live orphan worker; wait for it before resuming: ' + str(path))
                        record.update(status='interrupted', terminal=False, interrupted_at=now())
                        _atomic_write_json(path, record)
                        emit('interrupted_recovered', path=str(path))
                if inference_valid(root, row, digest, args.mock):
                    states[i]['inference'] = 'succeeded'
                    if visualization_valid(root, row, digest, args.mock):
                        states[i]['visualization'] = 'succeeded'
                        emit('resume_skipped', index=i, stage='both')
                    elif not args.calibration_inference_only:
                        backlog.append(row)
                        emit('resume_skipped', index=i, stage='inference')
                else:
                    pending.append(row)
            pending = deque(sorted(pending, key=lambda r:(-r['frame_count'],r['unit_index'])))
            slots = list(range(len(args.gpus)*args.workers_per_gpu))
            running, viz_running = {}, {}
            with ThreadPoolExecutor(max_workers=len(slots)) as gpu_pool, ThreadPoolExecutor(max_workers=args.viz_workers) as viz_pool:
                while pending or backlog or running or viz_running or oom:
                    t = time.monotonic()
                    if t-last_monitor >= args.monitor_interval:
                        resources = resource_snapshot(root)
                        if resources['disk_free_bytes'] < free_min:
                            paused = 'disk capacity below required reserve'
                        try:
                            if identity(args.manifest) != idx['manifest_identity'] or any(identity(p)!=v for p,v in identities.items()):
                                paused = 'manifest/config asset identity changed'
                        except OSError:
                            paused = 'manifest/config asset missing'
                        value = summary(resources)
                        emit('monitor', **{k:value[k] for k in ('units','frames','frame_states','rolling_fps','eta_seconds','viz_queue_depth','resources')})
                        last_monitor = t
                    if stop[0] and not paused:
                        paused = 'interrupted by signal; draining active workers'
                    # OOM pauses all fresh dispatch, drains the batch, then retries once at n=1.
                    isolated = bool(oom and not running)
                    if isolated and not paused:
                        seen_gpus = set()
                        for slot in list(slots):
                            gpu = args.gpus[slot % len(args.gpus)]
                            if gpu in seen_gpus or not oom:
                                continue
                            seen_gpus.add(gpu)
                            row = oom.popleft()
                            slots.remove(slot)
                            states[row['unit_index']]['inference'] = 'running'
                            emit('dispatch', stage='inference', index=row['unit_index'], gpu=gpu, slot=slot, isolated=True)
                            running[gpu_pool.submit(infer,args,row,slot,digest,4,True)] = (row,slot)
                    if not paused and not oom:
                        # While an isolated wave runs, do not refill another slot on its GPU.
                        isolated_active = any(attempts[r['unit_index']] == -1 for r,_ in running.values())
                        if not isolated_active and len(backlog)+len(viz_running) < args.viz_queue_limit:
                            for slot in list(slots):
                                if not pending or len(backlog)+len(viz_running)+len(running) >= args.viz_queue_limit:
                                    break
                                row = pending.popleft()
                                slots.remove(slot)
                                i=row['unit_index']
                                attempts[i] += 1
                                states[i]['inference']='running'
                                emit('dispatch', stage='inference', index=i, gpu=args.gpus[slot%len(args.gpus)], slot=slot, isolated=False)
                                running[gpu_pool.submit(infer,args,row,slot,digest,attempts[i])] = (row,slot)
                    while backlog and len(viz_running)<args.viz_workers and paused not in ('disk error','disk capacity below required reserve','manifest/config asset identity changed','manifest/config asset missing'):
                        row = backlog.popleft()
                        i=row['unit_index']
                        viz_attempts[i] += 1
                        states[i]['visualization']='running'
                        emit('dispatch', stage='visualization', index=i, attempt=viz_attempts[i])
                        viz_running[viz_pool.submit(visualize,args,row,digest,viz_attempts[i])] = row
                    all_futures = set(running)|set(viz_running)
                    if not all_futures:
                        if paused:
                            break
                        continue
                    done,_ = wait(all_futures,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        outcome = future.result()
                        i=outcome['index']
                        if future in running:
                            row,slot=running.pop(future)
                            slots.append(slot)
                            states[i]['inference']='succeeded' if outcome['success'] else 'failed'
                            if outcome['success']:
                                completed_frames += row['frame_count']
                                rolling.append((time.monotonic(),row['frame_count']))
                                if not args.calibration_inference_only:
                                    backlog.append(row)
                                consecutive_system=0
                            elif outcome['kind']=='oom' and not outcome['isolated']:
                                oom.append(row)
                                attempts[i]=-1
                            elif outcome['kind']=='disk':
                                paused='disk error'
                            elif outcome['kind']=='system':
                                consecutive_system += 1
                                if consecutive_system>=3:
                                    paused='three consecutive system errors'
                                elif attempts[i]<3 and not outcome['isolated']:
                                    pending.append(row)
                            elif outcome['kind']=='ordinary' and attempts[i]<3 and not outcome['isolated']:
                                pending.append(row)
                        else:
                            row=viz_running.pop(future)
                            states[i]['visualization']='succeeded' if outcome['success'] else 'failed'
                            if not outcome['success']:
                                if outcome['kind']=='disk':
                                    paused='disk error'
                                elif viz_attempts[i]<3:
                                    backlog.append(row)
                        if not pending and not running and not oom and inference_finished is None:
                            inference_finished=time.monotonic()
                        if args.compact_output and not outcome['success']:
                            compact.failed(args,row,digest,outcome)
                        emit('terminal', **outcome)
                        summary()
                    if paused and not running and not viz_running:
                        break
            value=summary(resource_snapshot(root))
            emit('coordinator_finished', pause_reason=paused, summary=value)
            if paused:
                return 75
            result=audit(args,idx,digest,locked=True)
            if args.compact_output and result['clean']:
                compact.tidy(root)
            if args.calibration_inference_only:
                value=read_json(root/'summary.json')
                value['status']='inference_calibration_complete'
                _atomic_write_json(root/'summary.json',value)
                return 0 if result['counts'].get('inference_succeeded',0)==idx['unit_count'] else 1
            return 0 if result['clean'] else 1
        finally:
            eventfile.close()
            for sig,handler in prior_signals.items():
                signal.signal(sig,handler)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('dry-run','bounded-run','full-run','audit','status','launch'):
        s=sub.add_parser(name)
        s.add_argument('--dataset-root',default=str(PRODUCT_ROOT.parents[1]/'1_Dataset'))
        s.add_argument('--manifest',required=True)
        s.add_argument('--result-root',required=True)
        s.add_argument('--wham-root',default=str(PRODUCT_ROOT.parents[1]/'third_party/WHAM'))
        s.add_argument('--wham-python',default=os.environ.get('VB_WHAM_PYTHON','python'))
        s.add_argument('--ffmpeg-bin',default=str(Path(os.environ.get('VB_TACTILE_PREFIX',sys.prefix))/'bin/ffmpeg'))
        s.add_argument('--annotation-index',default=str(PRODUCT_ROOT/'index/annotations.sqlite'))
        s.add_argument('--smpl-model',default=str(PRODUCT_ROOT.parents[1]/'models/smpl/SMPL_NEUTRAL.pkl'))
        s.add_argument('--gpus',default='0,1,2,3,4,5,6')
        s.add_argument('--workers-per-gpu',type=int,choices=(1,2),default=2)
        s.add_argument('--viz-workers',type=int,choices=(2,4,8),default=2)
        s.add_argument('--viz-queue-limit',type=int,default=64)
        s.add_argument('--monitor-interval',type=float,default=60)
        s.add_argument('--worker-timeout',type=float,default=21600)
        s.add_argument('--min-free-bytes',type=int,default=10**9)
        s.add_argument('--fps',type=float,default=30)
        s.add_argument('--wait-for-gpu',action='store_true')
        s.add_argument('--resume',action='store_true',help='terminals are always validated for resume')
        s.add_argument('--use-target-bbox',action='store_true',help='required semantics, always enabled')
        s.add_argument('--approved-full-run',action='store_true')
        s.add_argument('--compact-output',action='store_true',help='short public outputs; discard verified successful caches')
        s.add_argument('--max-test-units',type=int,default=32)
        s.add_argument('--max-test-frames',type=int,default=10000)
        s.add_argument('--mock',action='store_true')
        s.add_argument('--calibration-inference-only',action='store_true')
        s.add_argument('--mock-fail-unit',type=int,default=-1)
        s.add_argument('--mock-fail-attempts',type=int,default=0)
        s.add_argument('--mock-viz-fail-unit',type=int,default=-1)
        s.add_argument('--mock-viz-fail-attempts',type=int,default=0)
    return p


def main(argv=None):
    p=parser()
    a=p.parse_args(argv)
    a.gpus=[int(g) for g in a.gpus.split(',')]
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or min(a.gpus)<0 or a.viz_queue_limit<len(a.gpus)*a.workers_per_gpu or a.monitor_interval<=0:
        p.error('invalid GPU, queue or monitor configuration')
    a.manifest=str(Path(a.manifest).resolve())
    a.result_root=str(root_check(a.result_root))
    compact.configure(a)
    if a.calibration_inference_only and a.command!='bounded-run':
        p.error('inference-only calibration is restricted to bounded-run')
    if a.command=='status':
        print(json.dumps(read_json(Path(a.result_root)/'summary.json'),indent=2))
        return 0
    idx=load_index(a.manifest,verify_hash=True)
    if a.command=='dry-run':
        print(json.dumps(plan(a,idx),indent=2))
        return 0
    if a.command=='audit':
        print(json.dumps(audit(a,idx),indent=2))
        return 0
    if a.command=='launch':
        plan(a,idx)
        if not a.approved_full_run:
            p.error('launch requires explicit user approval and --approved-full-run')
        root=Path(a.result_root)
        root.mkdir(parents=True,exist_ok=True)
        argv=list(argv if argv is not None else sys.argv[1:])
        argv[0]='full-run'
        command=[sys.executable,'-m','wham_baseline.pipeline',*argv]
        with (root/'coordinator.log').open('ab') as log:
            child=subprocess.Popen(command,env=environment(),stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        _atomic_write_json(root/'launch.json',dict(pid=child.pid,pgid=child.pid,command=command,
                           shell_command=shlex.join(command),started_at=now(),manifest_sha256=idx['manifest_sha256']))
        print(json.dumps(read_json(root/'launch.json'),indent=2))
        return 0
    return run(a)


if __name__=='__main__':
    raise SystemExit(main())
