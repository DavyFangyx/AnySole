"""Final E9 trajectory organisation with five arms.

Arms: MotionPRO (v4), Step2Motion (entry), BiFoot (entry), BiFoot+Ramp,
BiFoot+TTA.  Tasks: SceneOcc (ShortCompletion, real VT) and HalfPlane
(LongCompletion, relative-preserving virtual VT).  Actions: 03/04/05/09.
"""
from __future__ import annotations
import argparse, csv, json, os, pickle, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP',
                                          'WenQuanYi Zen Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = str(Path(__file__).resolve().parents[2])
HERE = os.path.dirname(os.path.abspath(__file__))  # = SCRIPT_DIR
SW = ROOT + '/results/diagnostics/20261111_variant_sweep'
HYB = ROOT + '/results/diagnostics/20261108_hybrid'
HP_DIR = ROOT + '/results/diagnostics/20261111_halfplane_occ'
OUT_S = ROOT + '/results_display/Etest_crossfit_physimu_vgate/E9_Trajectory/ShortCompletion'
OUT_L = ROOT + '/results_display/Etest_crossfit_physimu_vgate/E9_Trajectory/LongCompletion'
sys.path[:0] = [ROOT + '/Baselines/Step2Motion/src', SW, HP_DIR, HERE, HYB,
                ROOT + '/results/diagnostics/20261110_unified_linear_ramp',
                ROOT + '/results/diagnostics/20261107_ti_step_stage1',
                ROOT + '/results/diagnostics/20261108_traj_sceneocc',
                ROOT + '/results/diagnostics/20261110_vtbridge',
                ROOT + '/results/diagnostics/20261111_s2m_next']

import eval_unified as U  # noqa: E402
import e9_final_protocol as P  # noqa: E402
import e9_halfplane_protocol as HP  # noqa: E402
import bifoot_theta_v as TV  # noqa: E402
import bifoot_theta_s_b as SB  # noqa: E402

FOLDS = ['fS5S6S11', 'fS5S11S12', 'fS6S7', 'fS5S6S8']
ACTIONS = ['03', '04', '05', '09']
ARMS = ['MotionPRO', 'Step2Motion', 'BiFoot', 'BiFoot-TTA']
METRICS = ['occ_ate_3d_rms', 'full_ate_3d_rms',
           'full_ate_3d_mean']
COL = {'GT': '#111111', 'MotionPRO': '#eb6834', 'Step2Motion': '#2a78d6',
       'BiFoot': '#1baf7a', 'BiFoot-Ramp': '#0d6b4a', 'BiFoot-TTA': '#7b4fa8'}
# GT reads as the reference (dashed, bold, drawn on top); baselines are
# thin solid lines; our method (BiFoot-TTA) is thicker than the other
# baselines so it doesn't get lost among them at a glance.
GT_STYLE = dict(color=COL['GT'], lw=1.8, ls=(0, (5, 2)), zorder=5)
ARM_LW = {'MotionPRO': 1.1, 'Step2Motion': 1.1, 'BiFoot': 1.1,
         'BiFoot-TTA': 1.6}
ARM_ZORDER = {'MotionPRO': 2, 'Step2Motion': 2, 'BiFoot': 2,
             'BiFoot-TTA': 4}
TEASERS = [('03', 'fS5S6S11', 'S5033'),
           ('04', 'fS6S7', 'S6041'),
           ('05', 'fS5S6S11', 'S11051'),
           ('09', 'fS6S7', 'S6092')]


def load_hybrid(fold):
    d = pickle.load(open(HYB + '/hybrid_data_%s_acontact.pkl' % fold, 'rb'))
    return {r['sid']: r for r in d['test']}


# --------------------------------------------------------------------------
# per-session rows
# --------------------------------------------------------------------------
def sceneocc_rows(device):
    rows = []
    src = json.load(open(HERE + '/e9_final_per_session.json'))
    m = {'MotionPRO-Entry': 'MotionPRO', 'Step2Motion-Entry': 'Step2Motion',
         'BiFoot-Entry': 'BiFoot', 'BiFoot-Ramp': 'BiFoot-Ramp'}
    for r in src:
        if r['arm'] in m:
            rows.append(dict(r, arm=m[r['arm']], proto='final'))
    for fold in FOLDS:
        p = SW + '/theta_sb_eval_%s_final_virtual.json' % fold
        if not os.path.isfile(p):
            print('missing TTA', p); continue
        for r in json.load(open(p)):
            if r['arm'] != 'TTA-pos+vel':
                continue
            rows.append(dict(fold=r['fold'], action=r['action'],
                             session=r['session'], arm='BiFoot-TTA',
                             proto='final',
                             **{k: r[k] for k in METRICS}))
    return rows


def halfplane_rows(device):
    rows = []
    for fold in FOLDS:
        p = SW + '/theta_sb_eval_%s_halfplane_virtual.json' % fold
        if not os.path.isfile(p):
            print('missing TTA', p); continue
        tta = json.load(open(p))
        for r in tta:
            if r['arm'] in ('Raw', 'Ramp', 'TTA-pos+vel'):
                amap = {'Raw': 'BiFoot', 'Ramp': 'BiFoot-Ramp',
                        'TTA-pos+vel': 'BiFoot-TTA'}[r['arm']]
                rows.append(dict(fold=r['fold'], action=r['action'],
                                 session=r['session'], arm=amap,
                                 proto='halfplane',
                                 **{k: r[k] for k in METRICS}))
        sids = sorted({r['session'] for r in tta if r['arm'] == 'Raw'})
        if not sids:
            continue
        s2m = HP.s2m_predictions(fold, device)
        by = load_hybrid(fold)
        for sid in sids:
            mp = HP.MANI + '/' + sid + '.json'
            if not os.path.isfile(mp) or sid not in s2m or sid not in by:
                continue
            man = json.load(open(mp))
            gt, vt, valid, J = HP.load_vt(fold, sid)
            n = len(gt)
            yaw = np.asarray(by[sid]['yaw'], float)
            if len(yaw) != n - 1:
                continue
            events = U.events_from_manifest(man, n)[0]
            if not events:
                continue
            vt_use = SB.virtual_vt(vt, gt, events, np.random.default_rng(0))
            arr = HP.map3(s2m[sid][0], s2m[sid][1], n)
            tr = HP.bifoot_arm(arr, yaw, vt_use, events, False)
            met = P.metrics(tr, gt, valid, events)
            rows.append(dict(fold=fold, action=str(man.get('action')),
                             session=sid, arm='Step2Motion',
                             proto='halfplane', **met))
            path = HP.MPRO_DIR % fold + '/' + sid + '.npz'
            if os.path.isfile(path):
                root, hips, _ = HP.load_baseline(path)
                if len(root) >= n:
                    tr = HP.course_arm(root[:n], vt_use, events, False)
                    met = P.metrics(tr, gt, valid, events)
                    rows.append(dict(fold=fold, action=str(man.get('action')),
                                     session=sid, arm='MotionPRO',
                                     proto='halfplane', **met))
    return rows


def rows_for(proto, device):
    return sceneocc_rows(device) if proto == 'final' else halfplane_rows(device)


# --------------------------------------------------------------------------
# tables / bars
# --------------------------------------------------------------------------
def write_tables(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    acts = ACTIONS
    # rows 每行 = 一条 (fold, session, arm) 记录（该记录内所有缺失区间的
    # 有效评分帧先并集计算 RMS）；下表 = 对这些记录等权 mean±std（ddof=0）。
    # 统计单位是 fold-session 记录，不是四折 std，也不是逐事件 std。
    for metric, name in [
            ('occ_ate_3d_rms', 'Main_Table1_E9_Final_OccAte3D_MeanStd.csv'),
            ('full_ate_3d_rms', 'Main_Table3_E9_Final_FullAte3D_RMS_MeanStd.csv')]:
        out = []
        for act in acts:
            rec = {'action': act}
            for arm in ARMS:
                v = [r[metric] for r in rows if r['arm'] == arm
                     and r['action'] == act and r.get(metric) is not None]
                rec[arm + '_mean'] = float(np.mean(v)) if v else None
                rec[arm + '_std'] = float(np.std(v)) if v else None
                rec[arm + '_n'] = len(v)
            out.append(rec)
        with open(os.path.join(out_dir, name), 'w', newline='',
                  encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
    keys = ['fold', 'action', 'session', 'arm'] + METRICS
    with open(os.path.join(out_dir, 'detail', 'Full_Table1_E9_Final_PerSession.csv'),
              'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            if r.get('action') not in ACTIONS:
                continue
            w.writerow({k: r.get(k) for k in keys})


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
def tta_traj(fold, sid, proto, head, device):
    cache = TV.load_cache(fold, 'test')
    if proto == 'halfplane':
        gt, vt, valid, J = HP.load_vt(fold, sid)
        man = json.load(open(HP.MANI + '/' + sid + '.json'))
        events = U.events_from_manifest(man, len(gt))[0]
        vt_use = SB.virtual_vt(vt, gt, events, np.random.default_rng(0))
    else:
        gt, vt, valid, J = P.load_vt(fold, sid)
        man = json.load(open(P.MANI + '/' + sid + '.json'))
        events = P.events_unified05(man, len(gt))
        vt_use = vt
    bf = P.bifoot_predictions(fold, SW + '/ckpt/bifoot_standalone_%s.pt' % fold,
                              device)
    by = load_hybrid(fold)
    yaw = np.asarray(by[sid]['yaw'], float)
    n = len(gt)
    arr = P.map3(bf[sid][0], bf[sid][1], n)
    d = arr.copy(); rng = np.random.default_rng(1)
    zt, zs, zb = SB.z_stats3(head, TV.load_cache(fold, 'train'), device)
    for ev in events:
        n0, e0 = ev['n0'], ev['e0']
        if e0 <= n0 or e0 >= n:
            continue
        u_exit = None
        M, _ = TV.event_target(arr, yaw, vt_use, n0, e0)
        if M is not None and e0 + 10 < n:
            u_exit = M.T @ SB.virtual_velocity(gt, e0, 10, rng)
        r = SB.tta_event3(head, cache, sid, n0, e0, arr, yaw, vt_use,
                          device, zt, zs, zb,
                          use_vel=True, u_exit=u_exit)
        if r is not None:
            d = SB.apply_correction3(d, n0, e0, r['theta'], r['s'], r['b'])
    tr = HP.bifoot_arm(d, yaw, vt_use, events, False)
    return gt, vt_use, valid, events, tr, man, yaw, arr


def arms_for_session(fold, sid, proto, head, device):
    gt, vt, valid, events, tr_tta, man, yaw, arr = tta_traj(
        fold, sid, proto, head, device)
    n = len(gt)
    s2m = P.s2m_predictions(fold, device)
    bf = P.bifoot_predictions(fold, SW + '/ckpt/bifoot_standalone_%s.pt' % fold,
                              device)
    arms = {'GT': gt}
    if sid in s2m:
        a = P.map3(s2m[sid][0], s2m[sid][1], n)
        if a is not None:
            arms['Step2Motion'] = HP.bifoot_arm(a, yaw, vt, events, False)
    path = (HP.MPRO_DIR if proto == 'halfplane' else P.MPRO_DIR) % fold
    path += '/' + sid + '.npz'
    if os.path.isfile(path):
        root, hips, _ = HP.load_baseline(path)
        if len(root) >= n:
            arms['MotionPRO'] = HP.course_arm(root[:n], vt, events, False)
    a = P.map3(bf[sid][0], bf[sid][1], n)
    arms['BiFoot'] = HP.bifoot_arm(a, yaw, vt, events, False)
    arms['BiFoot-Ramp'] = HP.bifoot_arm(a, yaw, vt, events, True)
    arms['BiFoot-TTA'] = tr_tta
    return gt, vt, valid, events, arms, man


def smooth_path(tr, med=9, win=31, poly=2):
    """Median + Savitzky-Golay smoothing for display only."""
    out = np.asarray(tr, float).copy()
    n = len(out)
    if n < 5:
        return out
    med = min(med, n - (1 - n % 2))
    win = min(win, n - (1 - n % 2))
    for j in range(out.shape[1]):
        x = out[:, j].copy()
        if not np.all(np.isfinite(x)):
            idx = np.arange(n)
            good = np.isfinite(x)
            if good.sum() < 2:
                continue
            x = np.interp(idx, idx[good], x[good])
        if med >= 3:
            x = medfilt(x, kernel_size=med)
        if win >= poly + 2:
            x = savgol_filter(x, win, poly)
        out[:, j] = x
    return out


def column_wedge(man):
    """cam3 through the column ground shadow wedge (mocap x-z)."""
    cal = json.load(open(man["calibration"]))
    col = man["column"]
    cx, cy = col["center_xy"]
    sx, sy = col["size_xy"]
    base = np.array([[cx + dx, cy + dy, 0.0] for dx in (-sx / 2, sx / 2)
                     for dy in (-sy / 2, sy / 2)], float)
    tr = cal["mocap_raw_to_checkerboard_world"]
    R = np.array(tr["R"])
    t = np.array(tr["t"]) / 100.0
    base_m = (base - t) @ R
    c = cal["cameras"]["cam3"]
    Rc = np.array(c["R"])
    tc = np.array(c["t"]) / 100.0
    Cm = ((-Rc.T @ tc) - t) @ R
    a = np.degrees(np.arctan2(base_m[:, 2] - Cm[2], base_m[:, 0] - Cm[0]))
    center = np.median(a)
    au = (a - center + 180.0) % 360.0 - 180.0 + center
    a0, a1 = float(au.min()), float(au.max())
    L = 12.0
    p1 = Cm[[0, 2]] + L * np.array([np.cos(np.radians(a0)),
                                    np.sin(np.radians(a0))])
    p2 = Cm[[0, 2]] + L * np.array([np.cos(np.radians(a1)),
                                    np.sin(np.radians(a1))])
    wedge = np.array([Cm[[0, 2]], p1, p2])
    return wedge, base_m[:, [0, 2]]


AXIS_GRAY = '#a7a7a7'    # 167,167,167 -- spines, ticks, tick/axis labels
OCC_FILL = '#e1ebf5'     # pale blue wash -- occlusion-region fill, both tasks
OCC_EDGE = AXIS_GRAY     # thin dashed boundary, same gray as the axes
OCC_HATCH = '////'
OBJ_FILL = '#5c5a54'     # physical occluder (column footprint): darker,
OBJ_ALPHA = 0.30         # never near-black so it can't read as GT


def draw_occlusion_region(ax, man, proto, gt, label_column=True):
    """Shade the no-light region with one consistent style across both
    tasks: a flat gray wash with a diagonal hatch and a thin dashed
    boundary, so SceneOcc's ground-shadow wedge and HalfPlane's half-plane
    region read as the same phenomenon (previously different alphas, and
    HalfPlane additionally drew a solid near-black `axhline` indistinguishable
    from the solid black GT line).
    """
    if proto == 'final' and 'calibration' in man and 'column' in man:
        wedge, foot = column_wedge(man)
        ax.fill(wedge[:, 0], wedge[:, 1], facecolor=OCC_FILL,
                edgecolor=OCC_EDGE, lw=0.8, ls='--', hatch=OCC_HATCH,
                zorder=0, label='vision lost (no light)')
        ax.fill(foot[:, 0], foot[:, 1], color=OBJ_FILL, alpha=OBJ_ALPHA,
                lw=0, zorder=1,
                label=('column (occluder)' if label_column else None))
        return foot
    zmax = max(float(np.nanmax(gt[:, 2])), 2.0) + 1.0
    ax.axhspan(0.5, zmax, facecolor=OCC_FILL, edgecolor=OCC_EDGE, lw=0,
               hatch=OCC_HATCH, zorder=0, label='vision lost (no light)')
    ax.axhline(0.5, color=OCC_EDGE, lw=0.8, ls='--', zorder=2)
    return None


def style_axes(ax):
    """Shared conference-figure chrome: hairline gray (167,167,167) spines
    (no boxed top/right) and matching tick marks/tick labels, but the X/Y
    axis-label text itself stays black (primary ink) at its given fontsize
    -- only the tick furniture is the muted gray. Thin solid gridlines
    behind the data.
    """
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_GRAY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS_GRAY, labelcolor=AXIS_GRAY, labelsize=8,
                   length=3, width=0.8)
    ax.xaxis.label.set_color('#0b0b0b')
    ax.yaxis.label.set_color('#0b0b0b')
    ax.grid(True, color='#e1e0d9', lw=0.6, zorder=-1)
    ax.set_axisbelow(True)


def set_view(ax, arrays, frac=0.12, floor=0.05):
    """Fit x/z limits to all given trajectories (GT + every drawn arm),
    with per-axis padding proportional to that axis's own span instead of
    a fixed metre pad. The old fixed 0.4 m pad was sized for the wide
    halfplane swings; on sessions where the real x-range is only ~0.15 m
    (walking roughly straight down the room) that pad was 2-3x the data
    itself, so the trajectory and the baseline-vs-GT differences in it
    were squeezed into a sliver of the frame. Bounds still come from the
    min/max of every array passed in (never a post-hoc re-centred crop),
    so nothing plotted here can end up clipped.
    """
    xs, ys = [], []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a, float)
        if a.size == 0:
            continue
        xs.append(a[:, 0])
        ys.append(a[:, 1])
    if not xs:
        return
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if len(xs) and len(ys):
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        padx = max(floor, frac * (x1 - x0))
        pady = max(floor, frac * (y1 - y0))
        ax.set_xlim(x0 - padx, x1 + padx)
        ax.set_ylim(y0 - pady, y1 + pady)


def plot_teaser(act, fold, sid, proto, head, device, out_path):
    gt, vt, valid, events, arms, man = arms_for_session(
        fold, sid, proto, head, device)
    mask = np.zeros(len(gt), bool)
    for e in events:
        mask[e['start']:e['end']] = True
    ok = mask & valid
    ates = {a: float(np.sqrt(np.mean(
        np.linalg.norm(tr[ok] - gt[ok], axis=1) ** 2)) * 1000)
        for a, tr in arms.items() if a != 'GT'}
    ev = max(events, key=lambda e: e['e0'] - e['n0'])
    plot_arms = dict(arms)
    if 'MotionPRO' in plot_arms:
        plot_arms['MotionPRO'] = smooth_path(plot_arms['MotionPRO'])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    ax.plot(gt[:, 0], gt[:, 2], **GT_STYLE, label='GT')
    for a in [x for x in ARMS if x in plot_arms]:
        t = plot_arms[a].copy(); t[~mask] = np.nan
        label = 'MotionPRO (smoothed)' if a == 'MotionPRO' else a
        ax.plot(t[:, 0], t[:, 2], color=COL[a], lw=ARM_LW[a],
                zorder=ARM_ZORDER[a], label=label)
    foot = draw_occlusion_region(ax, man, proto, gt, label_column=False)
    for i, e in enumerate(events):
        ax.scatter(vt[e['n0'], 0], vt[e['n0'], 2], marker='*', s=100,
                   color='#d62728', zorder=6,
                   label=('VT anchor n0' if i == 0 else None))
        ax.scatter(vt[e['e0'], 0], vt[e['e0'], 2], marker='X', s=65,
                   color='#1f4e9c', zorder=6,
                   label=('VT anchor e0' if i == 0 else None))
    set_view(ax, [gt[:, [0, 2]]]
             + [plot_arms[a][:, [0, 2]] for a in ARMS if a in plot_arms])
    ax.set_title('full session', fontsize=9); ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)'); style_axes(ax)
    ax.legend(frameon=False, fontsize=6)
    lo, hi = max(ev['n0'] - 10, 0), min(ev['e0'] + 10, len(gt))
    ax = axs[1]
    ax.plot(gt[lo:hi, 0], gt[lo:hi, 2], **GT_STYLE, label='GT')
    foot = draw_occlusion_region(ax, man, proto, gt)
    for a in [x for x in ARMS if x in plot_arms]:
        label = 'MotionPRO (smoothed)' if a == 'MotionPRO' else a
        ax.plot(plot_arms[a][lo:hi, 0], plot_arms[a][lo:hi, 2], color=COL[a],
                lw=ARM_LW[a], zorder=ARM_ZORDER[a], label=label)
    ax.scatter(vt[ev['n0'], 0], vt[ev['n0'], 2], marker='*', s=120,
               color='#d62728', zorder=6)
    ax.scatter(vt[ev['e0'], 0], vt[ev['e0'], 2], marker='X', s=75,
               color='#1f4e9c', zorder=6)
    set_view(ax, [gt[lo:hi, [0, 2]]]
             + [plot_arms[a][lo:hi, [0, 2]] for a in ARMS if a in plot_arms])
    ax.set_title('event zoom', fontsize=9); ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)'); style_axes(ax)
    ax = axs[2]
    ax.axvspan(ev['start'], ev['end'], color=OCC_FILL, lw=0, zorder=0)
    for a in [x for x in ARMS if x in arms]:
        e = np.linalg.norm(arms[a][lo:hi] - gt[lo:hi], axis=1) * 1000
        ax.plot(np.arange(lo, hi), e, color=COL[a], lw=ARM_LW[a],
                zorder=ARM_ZORDER[a], label=a)
    ax.axvline(ev['start'], color='#898781', lw=.8, ls=':')
    ax.axvline(ev['end'], color='#898781', lw=.8, ls=':')
    ax.set_title('3D root error', fontsize=9); ax.set_xlabel('frame')
    ax.set_ylabel('error (mm)'); style_axes(ax)
    txt = ' '.join('%s %.0f' % (a, ates[a]) for a in ARMS if a in ates)
    fig.suptitle('%s %s %s fold %s  occ ATE 3D: %s'
                 % (act, sid, proto, fold, txt), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.text(0.995, 0.008,
             'MotionPRO path smoothed for display; ATE/error on raw predictions.',
             ha='right', va='bottom', fontsize=5.5, color='0.45')
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print('wrote', out_path, {a: round(v, 1) for a, v in ates.items()})


def plot_fullsession(act, fold, sid, proto, head, device, out_path):
    """Standalone full-session panel (no axis labels; grid/ticks kept)."""
    gt, vt, valid, events, arms, man = arms_for_session(
        fold, sid, proto, head, device)
    mask = np.zeros(len(gt), bool)
    for e in events:
        mask[e['start']:e['end']] = True
    plot_arms = dict(arms)
    if 'MotionPRO' in plot_arms:
        plot_arms['MotionPRO'] = smooth_path(plot_arms['MotionPRO'])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(gt[:, 0], gt[:, 2], **GT_STYLE, label='GT')
    for a in [x for x in ARMS if x in plot_arms]:
        t = plot_arms[a].copy(); t[~mask] = np.nan
        label = 'MotionPRO (smoothed)' if a == 'MotionPRO' else a
        ax.plot(t[:, 0], t[:, 2], color=COL[a], lw=ARM_LW[a],
                zorder=ARM_ZORDER[a], label=label)
    foot = draw_occlusion_region(ax, man, proto, gt, label_column=False)
    for i, e in enumerate(events):
        ax.scatter(vt[e['n0'], 0], vt[e['n0'], 2], marker='*', s=110,
                   color='#d62728', zorder=6,
                   label=('VT anchor n0' if i == 0 else None))
        ax.scatter(vt[e['e0'], 0], vt[e['e0'], 2], marker='X', s=70,
                   color='#1f4e9c', zorder=6,
                   label=('VT anchor e0' if i == 0 else None))
    set_view(ax, [gt[:, [0, 2]]]
             + [plot_arms[a][:, [0, 2]] for a in ARMS if a in plot_arms])
    style_axes(ax)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print('wrote', out_path)


# One 2x2 panel per (task, action): row = task, each column a different
# action for that task. Single shared legend instead of one per panel.
GRID_CELLS = [
    ('final', 'fS5S6S11', 'S5033', '03'),
    ('final', 'fS6S7', 'S6092', '09'),
    ('halfplane', 'fS6S7', 'S6041', '04'),
    ('halfplane', 'fS5S6S11', 'S11051', '05'),
]
TASK_NAME = {'final': 'Column Occlusion', 'halfplane': 'Half-Room Blackout'}
ACTION_NAME = {'03': 'Slow Walk', '04': 'Fast Walk', '05': 'Circular Run',
              '09': 'Lateral Walk'}


def _draw_grid_panel(ax, proto, fold, sid, act, head, device):
    """Draw one GRID_CELLS panel (trajectory + occlusion region + VT
    anchors) onto `ax`. Shared by the 2x2 composite and the standalone
    per-panel figures so both stay pixel-identical except for legend/size.
    Returns the legend handle dict for this panel.
    """
    handles = {}
    gt, vt, valid, events, arms, man = arms_for_session(
        fold, sid, proto, head, device)
    mask = np.zeros(len(gt), bool)
    for e in events:
        mask[e['start']:e['end']] = True
    plot_arms = dict(arms)
    if 'MotionPRO' in plot_arms:
        plot_arms['MotionPRO'] = smooth_path(plot_arms['MotionPRO'])

    h, = ax.plot(gt[:, 0], gt[:, 2], **GT_STYLE)
    handles.setdefault('GT', h)
    for a in [x for x in ARMS if x in plot_arms]:
        t = plot_arms[a].copy(); t[~mask] = np.nan
        label = 'MotionPRO (smoothed)' if a == 'MotionPRO' else a
        h, = ax.plot(t[:, 0], t[:, 2], color=COL[a], lw=ARM_LW[a],
                    zorder=ARM_ZORDER[a])
        handles.setdefault(label, h)
    foot = draw_occlusion_region(ax, man, proto, gt, label_column=False)
    for patch in ax.patches:
        lbl = patch.get_label()
        if lbl == 'vision lost (no light)':
            handles.setdefault(lbl, patch)
    for i, e in enumerate(events):
        h = ax.scatter(vt[e['n0'], 0], vt[e['n0'], 2], marker='*', s=110,
                       color='#d62728', zorder=6)
        handles.setdefault('VT anchor: entry (n0)', h)
        h = ax.scatter(vt[e['e0'], 0], vt[e['e0'], 2], marker='X', s=70,
                       color='#1f4e9c', zorder=6)
        handles.setdefault('VT anchor: exit (e0)', h)
    set_view(ax, [gt[:, [0, 2]]]
             + [plot_arms[a][:, [0, 2]] for a in ARMS if a in plot_arms])
    style_axes(ax)
    return handles


LEGEND_ORDER = ['GT', 'MotionPRO (smoothed)', 'Step2Motion', 'BiFoot',
               'BiFoot-TTA', 'vision lost (no light)',
               'VT anchor: entry (n0)', 'VT anchor: exit (e0)']


def plot_grid2x2(head_by_fold, device, out_path):
    """2x2 trajectory grid: row 0 = SceneOcc (short occlusion), row 1 =
    HalfPlane (long occlusion); columns are two different demo actions per
    task. One shared legend for all four panels instead of a per-panel one.
    """
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.8))
    handles = {}
    for cell_i, (proto, fold, sid, act) in enumerate(GRID_CELLS):
        ax = axs[cell_i // 2, cell_i % 2]
        handles.update(_draw_grid_panel(ax, proto, fold, sid, act,
                                        head_by_fold[fold], device))
        ax.set_title('%s · %s' % (TASK_NAME[proto], ACTION_NAME[act]),
                     fontsize=13, fontweight='bold', pad=8, color='#0b0b0b')
        if cell_i // 2 == 1:
            ax.set_xlabel('X (m)', fontsize=9)
        if cell_i % 2 == 0:
            ax.set_ylabel('Y (m)', fontsize=9)

    fig.legend([handles[k] for k in LEGEND_ORDER if k in handles],
              [k for k in LEGEND_ORDER if k in handles],
              loc='lower center', ncol=4, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print('wrote', out_path)


def plot_grid_singles(head_by_fold, device, out_dir):
    """Same four panels as plot_grid2x2, each saved as its own standalone
    figure (own X/Y labels, no legend -- these are meant as single-panel
    paper figures, not a self-contained reference for curve identity).
    """
    for cell_i, (proto, fold, sid, act) in enumerate(GRID_CELLS):
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        _draw_grid_panel(ax, proto, fold, sid, act,
                         head_by_fold[fold], device)
        ax.set_title('%s · %s' % (TASK_NAME[proto], ACTION_NAME[act]),
                     fontsize=13, fontweight='bold', pad=8, color='#0b0b0b')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(
            out_dir, 'Main_Fig1_E9_Final_TrajectoryGrid_panel%d_%s_action%s.png'
            % (cell_i + 1, proto, act))
        fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        print('wrote', out_path)


def plot_grid_1x2(head_by_fold, device, cells, out_path):
    """Two GRID_CELLS panels side by side (1x2), one shared legend below
    -- same panel-drawing as plot_grid2x2/plot_grid_singles, just a
    different subset/layout.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 5.6))
    handles = {}
    for ax, (proto, fold, sid, act) in zip(axs, cells):
        handles.update(_draw_grid_panel(ax, proto, fold, sid, act,
                                        head_by_fold[fold], device))
        ax.set_title('%s · %s' % (TASK_NAME[proto], ACTION_NAME[act]),
                     fontsize=13, fontweight='bold', pad=8, color='#0b0b0b')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)

    fig.legend([handles[k] for k in LEGEND_ORDER if k in handles],
              [k for k in LEGEND_ORDER if k in handles],
              loc='lower center', ncol=4, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print('wrote', out_path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:2')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # SceneOcc
    rs = rows_for('final', device)
    write_tables(rs, OUT_S)
    # HalfPlane
    rl = rows_for('halfplane', device)
    write_tables(rl, OUT_L)
    # teasers and fullsession panels
    for i, (act, fold, sid) in enumerate(TEASERS):
        idx = 2 + i
        head = SB.load_head3(fold, device)
        plot_teaser(act, fold, sid, 'final', head, device,
                    os.path.join(OUT_S, 'Main_Fig%d_E9_Final_Teaser_action%s.png'
                                 % (idx, act)))
        plot_fullsession(act, fold, sid, 'final', head, device,
                         os.path.join(OUT_S,
                                      'Main_Fig%d_E9_Final_FullSession_action%s.png'
                                      % (idx, act)))
        head = SB.load_head3(fold, device)
        plot_teaser(act, fold, sid, 'halfplane', head, device,
                    os.path.join(OUT_L, 'Main_Fig%d_E9_Final_Teaser_action%s.png'
                                 % (idx, act)))
        plot_fullsession(act, fold, sid, 'halfplane', head, device,
                         os.path.join(OUT_L,
                                      'Main_Fig%d_E9_Final_FullSession_action%s.png'
                                      % (idx, act)))
    # one combined 2x2 figure: row = task, columns = two demo actions
    out_root = ROOT + '/results_display/Etest_crossfit_physimu_vgate/E9_Trajectory'
    head_by_fold = {fold: SB.load_head3(fold, device)
                    for fold in {c[1] for c in GRID_CELLS}}
    plot_grid2x2(head_by_fold, device,
                os.path.join(out_root, 'Main_Fig1_E9_Final_TrajectoryGrid.png'))


if __name__ == '__main__':
    main()
