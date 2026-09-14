"""Table 3/4 metrics for MotionPRO test evaluation."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

PELVIS_JOINT = 0
HEIGHT_AXIS = 1
CONTACT_HEIGHT_THRESH = 0.05
MM_SCALE = 1000.0
JITTER_SCALE = 1000.0
DEFAULT_FPS = 40.0
NAN = float('nan')

METRIC_NAMES = (
    'MPJPE',
    'PMPJPE',
    'PVE',
    'Accel',
    'WMPJPE',
    'WAMPJPE',
    'RTE',
    'Jitter',
    'WBCE',
)
CSV_FIELDS = ('session_id', 'n_valid_frames') + METRIC_NAMES


def _as_bool(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.ones((n,), dtype=bool)
    out = np.asarray(mask).reshape(-1).astype(bool)
    if out.shape[0] != n:
        raise ValueError(f'valid mask length {out.shape[0]} != {n}')
    return out


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return NAN
    return float(values.mean())


def pelvis_align(points: np.ndarray) -> np.ndarray:
    return points - points[:, PELVIS_JOINT:PELVIS_JOINT + 1]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Return R, t, scale mapping src -> dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f'umeyama expects (N,3) pairs, got {src.shape} and {dst.shape}')
    n = src.shape[0]
    if n == 0:
        return np.eye(3), np.zeros(3), 1.0
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
        S = S.copy()
        S[-1] *= -1
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = 1.0
    if with_scale and var_src > 0:
        scale = float(np.sum(S) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return R, t, scale


def apply_similarity(points: np.ndarray, R, t, scale: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return scale * points @ R.T + t


def first_valid_indices(valid: np.ndarray, n: int = 2) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return idx
    return idx[:n]


def last_valid_index(valid: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return None
    return int(idx[-1])


def _accel(points: np.ndarray, fps: float) -> np.ndarray:
    acc = np.full_like(points, np.nan, dtype=np.float64)
    if points.shape[0] < 3:
        return acc
    acc[1:-1] = (points[:-2] - 2.0 * points[1:-1] + points[2:]) * (fps ** 2)
    return acc


def _triple_valid(valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    if valid.shape[0] < 3:
        return out
    out[1:-1] = valid[:-2] & valid[1:-1] & valid[2:]
    return out


def _mean_l2(diff: np.ndarray, mask: np.ndarray) -> float:
    if mask is None:
        kept = diff.reshape(-1, diff.shape[-1])
    else:
        kept = diff[mask]
    if kept.size == 0:
        return NAN
    return float(np.mean(np.linalg.norm(kept.reshape(-1, kept.shape[-1]), axis=-1)))


def compute_session_metrics(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    pred_verts: Optional[np.ndarray] = None,
    gt_verts: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    fps: float = DEFAULT_FPS,
    session_id: str = '',
) -> Dict[str, float]:
    pred_j = np.asarray(pred_joints, dtype=np.float64)
    gt_j = np.asarray(gt_joints, dtype=np.float64)
    if pred_j.shape != gt_j.shape or pred_j.ndim != 3 or pred_j.shape[-1] != 3:
        raise ValueError(f'joints must be (T,J,3), got {pred_j.shape} and {gt_j.shape}')
    n_frames = pred_j.shape[0]
    keep = _as_bool(valid, n_frames)
    n_valid = int(keep.sum())
    out = {
        'session_id': session_id,
        'n_valid_frames': float(n_valid),
    }
    for name in METRIC_NAMES:
        out[name] = NAN
    if n_valid == 0:
        return out

    pred_pa = pelvis_align(pred_j)
    gt_pa = pelvis_align(gt_j)
    joint_err = np.linalg.norm(pred_pa - gt_pa, axis=-1)
    out['MPJPE'] = _nanmean(joint_err[keep]) * MM_SCALE

    pmpjpe = np.full((n_frames,), np.nan, dtype=np.float64)
    for t in np.flatnonzero(keep):
        aligned = apply_similarity(pred_j[t], *umeyama(pred_j[t], gt_j[t], with_scale=True))
        pmpjpe[t] = np.mean(np.linalg.norm(aligned - gt_j[t], axis=-1))
    out['PMPJPE'] = _nanmean(pmpjpe) * MM_SCALE

    if pred_verts is not None and gt_verts is not None:
        pred_v = np.asarray(pred_verts, dtype=np.float64)
        gt_v = np.asarray(gt_verts, dtype=np.float64)
        if pred_v.shape != gt_v.shape or pred_v.ndim != 3 or pred_v.shape[-1] != 3:
            raise ValueError(f'verts must be (T,V,3), got {pred_v.shape} and {gt_v.shape}')
        pred_v_pa = pred_v - pred_j[:, PELVIS_JOINT:PELVIS_JOINT + 1]
        gt_v_pa = gt_v - gt_j[:, PELVIS_JOINT:PELVIS_JOINT + 1]
        vert_err = np.linalg.norm(pred_v_pa - gt_v_pa, axis=-1)
        out['PVE'] = _nanmean(vert_err[keep]) * MM_SCALE

        ground = float(gt_v[keep, :, HEIGHT_AXIS].min())
        contact = (gt_v[:, :, HEIGHT_AXIS] - ground) < CONTACT_HEIGHT_THRESH
        contact[~keep] = False
        if np.any(contact):
            out['WBCE'] = float(np.mean(np.abs(pred_v[contact, HEIGHT_AXIS] - ground))) * MM_SCALE
    else:
        pred_v = None
        gt_v = None

    acc_mask = _triple_valid(keep)
    pred_acc = _accel(pred_pa, fps)
    gt_acc = _accel(gt_pa, fps)
    out['Accel'] = _mean_l2(pred_acc - gt_acc, acc_mask)

    world_acc = _accel(pred_j, fps)
    jitter = np.linalg.norm(world_acc, axis=-1)
    out['Jitter'] = _nanmean(jitter[acc_mask]) * JITTER_SCALE

    align_idx = first_valid_indices(keep, n=2)
    if align_idx.size > 0:
        src = pred_j[align_idx].reshape(-1, 3)
        dst = gt_j[align_idx].reshape(-1, 3)
        R, t, _ = umeyama(src, dst, with_scale=False)
        pred_w = apply_similarity(pred_j, R, t, 1.0)
        out['WMPJPE'] = _mean_l2(pred_w - gt_j, keep) * MM_SCALE
        last_t = last_valid_index(keep)
        if last_t is not None:
            out['RTE'] = float(np.linalg.norm(pred_w[last_t, PELVIS_JOINT] - gt_j[last_t, PELVIS_JOINT])) * MM_SCALE

    src_all = pred_j[keep].reshape(-1, 3)
    dst_all = gt_j[keep].reshape(-1, 3)
    R_all, t_all, _ = umeyama(src_all, dst_all, with_scale=False)
    pred_all = apply_similarity(pred_j, R_all, t_all, 1.0)
    out['WAMPJPE'] = _mean_l2(pred_all - gt_j, keep) * MM_SCALE
    return out


def summarize_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    valid_rows = [row for row in rows if int(row.get('n_valid_frames', 0)) > 0]
    summary = {
        'session_id': 'OVERALL',
        'n_valid_frames': float(sum(int(row['n_valid_frames']) for row in valid_rows)),
    }
    for name in METRIC_NAMES:
        summary[name] = NAN
    if not valid_rows:
        return summary

    weights = np.array([row['n_valid_frames'] for row in valid_rows], dtype=np.float64)
    for name in METRIC_NAMES:
        values = np.array([row[name] for row in valid_rows], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        if name == 'RTE':
            summary[name] = float(values[finite].mean())
        else:
            w = weights[finite]
            if w.sum() <= 0:
                continue
            summary[name] = float(np.average(values[finite], weights=w))
    return summary


def _fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return 'nan'
    return f'{value:.6f}'


def write_metrics_files(rows: Sequence[Mapping[str, float]], result_dir: Path, fps: float = DEFAULT_FPS):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(rows) + [summarize_metrics(rows)]
    csv_path = result_dir / 'test_metrics.csv'
    log_path = result_dir / 'test_metrics.log'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                'session_id': row['session_id'],
                'n_valid_frames': int(row['n_valid_frames']),
                **{name: _fmt(row[name]) for name in METRIC_NAMES},
            })
    lines = [
        'MotionPRO test metrics',
        f'eval_fps: {fps}',
        'units: MPJPE/PMPJPE/PVE/WMPJPE/WAMPJPE/RTE/WBCE in mm; Accel in m/s^2; Jitter in 1e-3 m/s^2',
        '',
    ]
    header = f"{'session_id':<16}{'n_valid':>8}" + ''.join(f'{name:>12}' for name in METRIC_NAMES)
    lines.append(header)
    for row in all_rows:
        line = f"{str(row['session_id']):<16}{int(row['n_valid_frames']):>8}"
        line += ''.join(f'{_fmt(row[name]):>12}' for name in METRIC_NAMES)
        lines.append(line)
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, log_path
