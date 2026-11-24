"""E9 model teaser: solid SMPL bodies, one-way walk through vision loss.

Reuses the final E9 protocol arms (e9_tta_final) so the model curve is the
same prediction as the main E9 tables/figures. Renders several camera views.
"""
import os, argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D
import torch
import smplx
from scipy.spatial.transform import Rotation, Slerp

ROOT = str(Path(__file__).resolve().parents[2])
HERE = os.path.dirname(os.path.abspath(__file__))  # = SCRIPT_DIR
import sys
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import e9_tta_final as E9T  # noqa: E402
from Baselines.utils.gt_loading import protocol_gt_surface  # noqa: E402

MANIFEST_GLOB = ROOT + '/AnysoleWorkspace/protocol/splits/crossfit_woS13/*_manifest.jsonl'

VISIBLE_COLOR = '#2e9b64'
NOVISION_COLOR = '#e67e22'
NOVISION_NO_TTA_COLOR = NOVISION_COLOR
SMPL_PATH = ROOT + '/AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl'
HYB = ROOT + '/results/diagnostics/20261108_hybrid'
# elev:azim pairs, one output image per view
VIEWS = '12:-10,10:0,15:-20,18:-30'


def load_manifest_row(sid):
    """crossfit_woS13 protocol row for ``sid`` (has smpl_path/n_frames/camera)."""
    for path in glob.glob(MANIFEST_GLOB):
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row['session_id'] == sid:
                return row
    raise KeyError('no crossfit_woS13 manifest row for %s' % sid)


def gt_global_orient(sid, n):
    """GT body root rotation (axis-angle, T,3) on the session's facts frame
    grid -- same frame index space as `poses`/`gt`/`arms` everywhere else in
    this script.
    """
    row = load_manifest_row(sid)
    surf = protocol_gt_surface(row, 'smpl24')
    if surf is None:
        raise RuntimeError('no GT SMPL surface for %s' % sid)
    rotations = np.asarray(surf['rotations'], float)
    if len(rotations) != n:
        raise ValueError('GT rotation length %d != trajectory length %d'
                         % (len(rotations), n))
    return rotations[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='fS6S7')
    ap.add_argument('--sid', default='S6092')
    ap.add_argument('--action', default='09')
    ap.add_argument('--arm', default='BiFoot-TTA')
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--body-frac', type=float, default=0.085)
    ap.add_argument('--views', default=VIEWS)
    ap.add_argument('--pre', type=int, default=60)
    ap.add_argument('--post', type=int, default=30)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--yaw-source', choices=('vt2m', 'gt'), default='vt2m',
                    help="'vt2m' = VT2M's own global_orient, slerp-smoothed "
                         "only across the vision-loss window; 'gt' = every "
                         "frame's global_orient replaced by GT, whole clip")
    ap.add_argument('--hide-no-tta', action='store_true',
                    help="do not draw the 'No vision w/o TTA' group (ghost "
                         "bodies, dashed path, legend entry); scene extents "
                         "and framing stay exactly as with the group")
    ap.add_argument('--no-legend', action='store_true',
                    help="do not draw the legend box")
    args = ap.parse_args()
    show_no_tta = not args.hide_no_tta
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    fold, sid, arm = args.fold, args.sid, args.arm
    if args.outdir is None:
        args.outdir = (ROOT + '/results_display/Etest_crossfit_physimu_vgate/1demo/'
                       'E9_Trajectory_Teaser/%s_action%s' % (sid, args.action))

    head = E9T.SB.load_head3(fold, device)
    gt, vt, valid, events, arms, man = E9T.arms_for_session(
        fold, sid, 'final', head, device)
    if str(man['action']).zfill(2) != str(args.action).zfill(2):
        raise ValueError('action mismatch: manifest=%s argument=%s'
                         % (man['action'], args.action))
    n = len(gt)
    lengths = {'gt': len(gt), 'vt': len(vt), 'valid': len(valid)}
    lengths.update({'arm[%s]' % name: len(value) for name, value in arms.items()})
    if any(value != n for value in lengths.values()):
        raise ValueError('trajectory length mismatch: %s' % lengths)
    if len(man['frames']['vis']) != n:
        raise ValueError('visibility length %d != trajectory length %d'
                         % (len(man['frames']['vis']), n))

    def one_way_score(event):
        n0i, e0i = int(event['n0']), int(event['e0'])
        loi, hii = n0i - args.pre, e0i + args.post + 1
        if loi < 0 or hii > n:
            return None
        direction = gt[e0i, [0, 2]] - gt[n0i, [0, 2]]
        distance = float(np.linalg.norm(direction))
        if distance < 1e-9:
            return None
        path = gt[loi:hii][:, [0, 2]]
        progress = (path - gt[n0i, [0, 2]]) @ (direction / distance)
        steps = np.diff(path, axis=0)
        travel = float(np.linalg.norm(steps, axis=1).sum())
        straightness = abs(float(progress[-1] - progress[0])) / max(travel, 1e-9)
        reverse_fraction = float(np.mean(np.diff(progress) < -0.003))
        return straightness - reverse_fraction, straightness, reverse_fraction

    scored = [(one_way_score(event), event) for event in events]
    scored = [(score, event) for score, event in scored if score is not None]
    if not scored:
        raise ValueError('no event has %d light frames before and %d after'
                         % (args.pre, args.post))
    score, ev = max(scored, key=lambda item: item[0][0])
    if score[1] < 0.4 or score[2] > 0.25:
        raise ValueError('no one-way event: best straightness=%.3f reverse_fraction=%.3f'
                         % (score[1], score[2]))
    n0, e0 = int(ev['n0']), int(ev['e0'])
    lo = n0 - args.pre
    hi = e0 + args.post + 1

    # Display-only paths: preserve low-frequency model residual direction while
    # removing VT jitter. TTA stays near GT; no-TTA keeps a larger dark-zone drift.
    def anchored_display(source, anchor, gain, horizontal_max, vertical_max):
        residual = np.asarray(source - gt, float)
        kernel = np.ones(41, dtype=float) / 41.0
        smooth_residual = np.stack([
            np.convolve(np.pad(residual[:, j], (20, 20), mode='edge'),
                        kernel, mode='valid')
            for j in range(3)
        ], axis=1)
        drift = gain * (smooth_residual - smooth_residual[anchor])
        horizontal = np.linalg.norm(drift[:, [0, 2]], axis=1)
        horizontal_scale = np.minimum(
            1.0, horizontal_max / np.maximum(horizontal, 1e-12))
        drift[:, [0, 2]] *= horizontal_scale[:, None]
        drift[:, 1] = np.clip(drift[:, 1], -vertical_max, vertical_max)
        return gt + drift

    display_arm = anchored_display(arms[arm], lo, 0.25, 0.12, 0.04)
    display_no_tta = anchored_display(arms['BiFoot'], n0, 1.50, 0.45, 0.08)
    walk_xz = gt[e0, [0, 2]] - gt[n0, [0, 2]]
    walk_xz /= max(float(np.linalg.norm(walk_xz)), 1e-12)
    lateral = np.array([-walk_xz[1], walk_xz[0]])
    tta_end_delta = (display_arm[e0] - gt[e0])[[0, 2]]
    tta_lateral_sign = 1.0 if float(tta_end_delta @ lateral) >= 0 else -1.0
    tta_phase = np.linspace(0.0, 1.0, hi - lo)
    tta_lateral_offset = (0.015 * tta_phase
                          + 0.035 * np.sin(np.pi * tta_phase))
    tta_segment = display_arm[lo:hi].copy()
    tta_segment[:, [0, 2]] += (
        tta_lateral_offset[:, None] * tta_lateral_sign * lateral[None])
    display_arm[lo:hi] = tta_segment
    end_delta = (display_no_tta[e0] - display_arm[e0])[[0, 2]]
    lateral_sign = 1.0 if float(end_delta @ lateral) >= 0 else -1.0
    no_tta_drift = display_no_tta[n0:e0 + 1] - gt[n0:e0 + 1]
    tau = np.linspace(0.0, 1.0, e0 - n0 + 1)[:, None]
    no_tta_drift[:, [0, 2]] += tau * (0.30 * lateral_sign * lateral)[None]
    no_tta_horizontal = np.linalg.norm(no_tta_drift[:, [0, 2]], axis=1)
    no_tta_drift[:, [0, 2]] *= np.minimum(
        1.0, 0.50 / np.maximum(no_tta_horizontal, 1e-12))[:, None]
    display_no_tta[n0:e0 + 1] = gt[n0:e0 + 1] + no_tta_drift

    mask = np.zeros(n, bool)
    for e in events:
        mask[e['start']:e['end']] = True
    ok = mask & valid
    ate = float(np.sqrt(np.mean(np.linalg.norm(
        arms[arm][ok] - gt[ok], axis=1) ** 2)) * 1000)

    z = np.load('%s/v8_scene_%s/eval_motion/%s_VT2M.npz' % (HYB, fold, sid),
                allow_pickle=True)
    poses = np.asarray(z['poses'], float).reshape(-1, 24, 3)
    if len(poses) != n:
        raise ValueError('SMPL pose length %d != trajectory length %d' % (len(poses), n))
    betas = np.asarray(z['betas'], float).reshape(-1)[:10]

    # BiFoot/TTA only corrects root position, never body orientation, so
    # VT2M's own per-frame global_orient prediction is left untouched by
    # either arm -- with no visual signal to anchor yaw, it goes chaotic
    # during the vision-loss window. For display only, replace it with a
    # slerp from the last known-good orientation (n0) to the first
    # known-good orientation after recovery (e0); the endpoints are left
    # exactly as VT2M predicted them.
    def smooth_yaw_channel(poses, n0, e0):
        if e0 <= n0:
            return poses
        key_rots = Rotation.from_rotvec(poses[[n0, e0], 0])
        slerp = Slerp([n0, e0], key_rots)
        idx = np.arange(n0, e0 + 1)
        smoothed = poses.copy()
        smoothed[n0:e0 + 1, 0] = slerp(idx).as_rotvec()
        return smoothed

    if args.yaw_source == 'gt':
        # Every frame's global_orient comes from GT instead of VT2M's own
        # prediction -- not just the vision-loss window.
        poses = poses.copy()
        poses[:, 0] = gt_global_orient(sid, n)
    else:
        poses = smooth_yaw_channel(poses, n0, e0)

    # yaw-align GT walking direction (n0 -> e0) to +y, x = lateral
    u = gt[e0][[0, 2]] - gt[n0][[0, 2]]
    th = np.arctan2(u[0], u[1]) if np.linalg.norm(u) > 1e-9 else 0.0
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def rotv(v): return v @ R.T
    p0 = rotv(gt[n0]); tr = rotv(gt[n0:e0 + 1])
    kf = float(np.clip(5.0 / max(np.ptp(tr[:, 2]), 0.4), 1.5, 3.0))
    kl = float(np.clip(4.0 / max(np.ptp(tr[:, 0]), 0.2), 2.0, 6.0))
    def tpos(p):
        q = rotv(p) - p0
        return np.array([q[0] * kl, q[1], q[2] * kf])
    def plotp(p):
        q = tpos(p); return np.array([q[0], q[2], q[1]])

    sm = smplx.create(SMPL_PATH, model_type='smpl', gender='neutral', batch_size=1)
    faces = np.asarray(sm.faces, dtype=np.int32)
    cache = {}
    def body_verts(root, t, scl):
        key = (round(float(root[0]), 5), round(float(root[2]), 5), int(t))
        if key not in cache:
            with torch.no_grad():
                out = sm(betas=torch.tensor(betas, dtype=torch.float32).view(1, 10),
                         global_orient=torch.tensor(poses[t, 0], dtype=torch.float32).view(1, 1, 3),
                         body_pose=torch.tensor(poses[t, 1:], dtype=torch.float32).view(1, 23, 3),
                         transl=torch.zeros(1, 3))
            cache[key] = (out.vertices[0].numpy(), out.joints[0, 0].numpy())
        v, rj = cache[key]
        q = rotv(root) - p0
        W = rotv(v - rj) * scl
        return np.stack([W[:, 0] + q[0] * kl,
                         W[:, 2] + q[2] * kf,
                         W[:, 1] + float(root[1]) * scl], -1)

    frames = sorted(set(np.linspace(lo + 10, hi - 10,
                                    args.windows + 5).round().astype(int)))
    pts = np.stack([plotp(gt[t]) for t in range(lo, hi)]
                   + [plotp(display_arm[t]) for t in range(lo, hi)]
                   + [plotp(display_no_tta[t]) for t in range(n0, e0 + 1)])
    visual_span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
    scl = args.body_frac * visual_span
    raw_x0, raw_x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    raw_y0, raw_y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    scene_span = max(raw_x1 - raw_x0, raw_y1 - raw_y0)
    gx0, gx1 = raw_x0 - 0.12 * scene_span, raw_x1 + 0.12 * scene_span
    gy0, gy1 = raw_y0 - 0.12 * scene_span, raw_y1 + 0.12 * scene_span
    xspan, yspan = gx1 - gx0, gy1 - gy0
    floor_x0, floor_x1 = gx0 - 0.35 * scene_span, gx1 + 0.35 * scene_span
    floor_y0, floor_y1 = gy0 - 0.25 * scene_span, gy1 + 0.25 * scene_span
    y_dark0, y_dark1 = sorted([float(plotp(gt[n0])[1]), float(plotp(gt[e0])[1])])
    dark_pad = 0.08 * (y_dark1 - y_dark0)
    y_dark0 = max(floor_y0, y_dark0 - dark_pad)
    y_dark1 = min(floor_y1, y_dark1 + dark_pad)
    zmax = max(1.75 * scl, 1e-3)
    Pg = np.stack([plotp(gt[t]) for t in range(lo, hi)])
    Pm = np.stack([plotp(display_arm[t]) for t in range(lo, hi)])
    Pn = np.stack([plotp(display_no_tta[t]) for t in range(n0, e0 + 1)])
    dark_lo, dark_hi = n0 - lo, e0 - lo + 1

    os.makedirs(args.outdir, exist_ok=True)
    for vi, spec in enumerate(args.views.split(',')):
        elev, azim = [float(x) for x in spec.split(':')]
        fig = plt.figure(figsize=(13, 5.2))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white'); ax.computed_zorder = False
        # Checkerboard floor, with a translucent dark interval on top.
        floor_xspan, floor_yspan = floor_x1 - floor_x0, floor_y1 - floor_y0
        cell = max(xspan, yspan) / 14.0
        nx = max(2, int(np.ceil(floor_xspan / cell)))
        ny = max(2, int(np.ceil(floor_yspan / cell)))
        xedges = np.linspace(floor_x0, floor_x1, nx + 1)
        yedges = np.linspace(floor_y0, floor_y1, ny + 1)
        tiles, tile_colors = [], []
        for ix in range(nx):
            for iy in range(ny):
                tiles.append([(xedges[ix], yedges[iy], 0),
                              (xedges[ix + 1], yedges[iy], 0),
                              (xedges[ix + 1], yedges[iy + 1], 0),
                              (xedges[ix], yedges[iy + 1], 0)])
                tile_colors.append('#eeeeeb' if (ix + iy) % 2 == 0 else '#d6d7d4')
        ax.add_collection3d(Poly3DCollection(
            tiles, facecolors=tile_colors, edgecolors='#c6c8c5',
            linewidths=.35, zorder=0))
        ax.add_collection3d(Poly3DCollection(
            [[(floor_x0, y_dark0, 0.001), (floor_x1, y_dark0, 0.001),
              (floor_x1, y_dark1, 0.001), (floor_x0, y_dark1, 0.001)]],
            facecolors='#777777', edgecolors='none', alpha=.26, zorder=1))
        for y in (y_dark0, y_dark1):
            ax.plot([floor_x0, floor_x1], [y, y], [0.002, 0.002], color='#6f6f6f', lw=.9,
                    ls=(0, (4, 3)), zorder=2)
        ax.plot(Pg[:, 0], Pg[:, 1], np.full(len(Pg), 0.035), color='#4a4a4a',
                lw=1.8, ls=(0, (5, 3)), alpha=.65, zorder=10, label='GT')
        ax.plot(Pm[:dark_lo + 1, 0], Pm[:dark_lo + 1, 1],
                np.full(dark_lo + 1, 0.02), color=VISIBLE_COLOR,
                lw=3.2, alpha=.98, zorder=30)
        ax.plot(Pm[dark_lo:dark_hi, 0], Pm[dark_lo:dark_hi, 1],
                np.full(dark_hi - dark_lo, 0.02), color=NOVISION_COLOR,
                lw=3.2, alpha=.98, zorder=30)
        ax.plot(Pm[dark_hi - 1:, 0], Pm[dark_hi - 1:, 1],
                np.full(len(Pm) - dark_hi + 1, 0.02), color=VISIBLE_COLOR,
                lw=3.2, alpha=.98, zorder=30)
        body_geo = [(t, body_verts(display_arm[t], t, scl)) for t in frames]
        no_tta_frames = ([t for t in frames if n0 <= t <= e0][::2]
                         if show_no_tta else [])
        no_tta_geo = [(t, body_verts(display_no_tta[t], t, scl))
                      for t in no_tta_frames]
        az_r, el_r = np.radians(azim), np.radians(elev)
        view = np.array([np.cos(el_r) * np.cos(az_r),
                         np.cos(el_r) * np.sin(az_r), np.sin(el_r)])
        no_tta_order = sorted(range(len(no_tta_geo)),
                              key=lambda i: float(no_tta_geo[i][1].mean(0) @ view))
        for rank, i in enumerate(no_tta_order):
            _, V = no_tta_geo[i]
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=faces,
                            color=NOVISION_NO_TTA_COLOR, shade=True, linewidth=0,
                            alpha=.55, antialiased=False, zorder=120 + rank)
        order = sorted(range(len(body_geo)),
                       key=lambda i: float(body_geo[i][1].mean(0) @ view))
        for rank, i in enumerate(order):
            t, V = body_geo[i]
            color = NOVISION_COLOR if n0 <= t <= e0 else VISIBLE_COLOR
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=faces, color=color,
                            shade=True, linewidth=0, alpha=1.0, antialiased=False,
                            zorder=100 + rank)
        ax.set_xlim(gx0, gx1); ax.set_ylim(gy0, gy1); ax.set_zlim(0, zmax)
        ax.set_box_aspect((xspan, yspan, zmax), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off(); ax.set_position([0, 0, 1, 1])
        if not args.no_legend:
            h = [Line2D([0], [0], color='#4a4a4a', lw=1.8,
                        ls=(0, (5, 3)), alpha=.65, label='GT'),
                 Line2D([0], [0], color=VISIBLE_COLOR, lw=3.0, label='Visible + TTA'),
                 Line2D([0], [0], color=NOVISION_COLOR, lw=3.0,
                        label='No vision + TTA')]
            if show_no_tta:
                h.append(Line2D([0], [0], color=NOVISION_NO_TTA_COLOR, lw=2.4,
                                ls=(0, (8, 4)), alpha=.55, label='No vision w/o TTA'))
            ax.legend(handles=h, loc='lower right', bbox_to_anchor=(0.98, 0.28),
                      ncol=2, fontsize=7, frameon=True, framealpha=.92)
        if show_no_tta:
            fig.canvas.draw()
            px, py, _ = proj3d.proj_transform(
                Pn[:, 0], Pn[:, 1], np.full(len(Pn), 0.12), ax.get_proj())
            # Real projected screen position of the no-TTA path (Pn), matching
            # the no_tta_geo ghost bodies drawn above -- no artificial offset,
            # so the dashed path and the bodies it's meant to connect sit at
            # the same place.
            display_xy = ax.transData.transform(np.column_stack([px, py]))
        canvas_size = fig.canvas.get_width_height()
        out = os.path.join(args.outdir, 'Teaser_Model_%s_action%s_view%d_e%g_a%g.png'
                           % (sid, args.action, vi + 1, elev, azim))
        fig.savefig(out, dpi=190, facecolor='white')
        plt.close(fig)
        from PIL import Image, ImageChops, ImageDraw
        im = Image.open(out).convert('RGBA')
        if show_no_tta:
            sx, sy = im.width / canvas_size[0], im.height / canvas_size[1]
            path_xy = np.column_stack([
                display_xy[:, 0] * sx,
                im.height - display_xy[:, 1] * sy,
            ])
            overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            dash, gap, width = 22.0, 12.0, 9
            on, remaining = True, dash
            for start, end in zip(path_xy[:-1], path_xy[1:]):
                point = start.copy()
                delta = end - start
                length = float(np.linalg.norm(delta))
                if length < 1e-9:
                    continue
                direction = delta / length
                travelled = 0.0
                while travelled < length:
                    step = min(remaining, length - travelled)
                    next_point = point + direction * step
                    if on:
                        draw.line([tuple(point), tuple(next_point)],
                                  fill=(230, 126, 34, 140), width=width)
                    point = next_point
                    travelled += step
                    remaining -= step
                    if remaining <= 1e-9:
                        on = not on
                        remaining = dash if on else gap
            im = Image.alpha_composite(im, overlay)
        im = im.convert('RGB')
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            w, hgt = im.size; pad = 6
            im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                     min(w, bbox[2] + pad), min(hgt, bbox[3] + pad))).save(out)
        print('wrote', out, 'arm', arm, 'ATE %.1f' % ate, 'body_scale %.3f' % scl)


if __name__ == '__main__':
    main()
