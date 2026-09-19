"""Locate F6 false-contact segments in S11113 & S8023, and identify the trigger branch.

For every air->contact transition, print the trigger (vel_cont / pres_cont), the
h-veto status, and a 15-frame signal window around the transition.
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path("/data/fangyuxuan/projects/gait/results_display/script")
sys.path.insert(0, str(SCRIPT_DIR))
import contact_methods as cm  # noqa: E402

FPS = cm.FPS
V_LO, V_HI, A_THR, H_HIGH = cm.F6_V_LO, cm.F6_V_HI, cm.F6_A_THR, cm.F6_H_HIGH


def analyze(session_id: str, seq_root=cm.cli_common.DEFAULT_SEQ_ROOT):
    seq_dir = next(seq_root.glob(f"*/*/{session_id}"))
    ctx = cm.load_aligned(seq_dir)
    pipe = cm._f6_pipeline(ctx)
    # vertical velocity of the foot center (same filtered p as _f6_signals)
    names, pts = ctx["names"], ctx["pts"]
    ctx["_vz"] = {}
    for side in ("left", "right"):
        idx = [i for i, nm in enumerate(names)
               if nm.lower().startswith(f"{side}foot") or nm.lower().startswith(f"{side}toe")]
        fp = cm.median_filter(pts[:, idx, :], size=(3, 1, 1))
        p = fp.mean(axis=1)
        vz = np.zeros(p.shape[0], dtype=np.float64)
        vz[1:] = (p[1:, 2] - p[:-1, 2]) * FPS
        ctx["_vz"][side] = vz
    print(f"######## {session_id}  {seq_dir}  n={ctx['n']}")
    for side in ("left", "right"):
        sig = cm._f6_signals(ctx, side)
        p = pipe[side]
        v, az, h = sig["v"], sig["az"], sig["h"]
        vz = ctx["_vz"][side]
        state, loaded, corr = p["state"], p["loaded"], p["corrected"]
        print(f"\n===== {session_id} {side}  thr_lo={p['thr_lo']:.0f} thr_hi={p['thr_hi']:.0f} "
              f"calibrated={p['calibrated']} n_air={p['n_air']}")
        # contact runs
        runs, t, n = [], 0, len(v)
        while t < n:
            if state[t] == 1:
                s = t
                while t < n and state[t] == 1:
                    t += 1
                runs.append((s, t))
            else:
                t += 1
        print(f"  contact runs: " + "; ".join(
            f"[{s/FPS:.2f},{e/FPS:.2f})s h={h[s:e].min()*100:+.0f}~{h[s:e].max()*100:+.0f}cm "
            f"v_max={v[s:e].max():.2f}" for s, e in runs))
        # air runs
        air_runs, t = [], 0
        while t < n:
            if state[t] == 0:
                s = t
                while t < n and state[t] == 0:
                    t += 1
                air_runs.append((s, t))
            else:
                t += 1
        print(f"  air runs:    " + "; ".join(
            f"[{s/FPS:.2f},{e/FPS:.2f})s h={h[s:e].min()*100:+.0f}~{h[s:e].max()*100:+.0f}cm "
            f"v_min={v[s:e].min():.2f} v_max={v[s:e].max():.2f} corr_max={corr[s:e].max():.0f}"
            for s, e in air_runs))
        # for each air->contact transition: which branch fired
        for s, e in air_runs:
            if e >= n:
                continue
            tt = e  # first contact frame
            # recompute branch conditions at tt
            vel_cont = (v[tt] < V_LO and v[tt-1] < V_LO
                        and abs(az[tt]) < A_THR and abs(az[tt-1]) < A_THR)
            pres_cont = bool(loaded[tt]) and v[tt] < V_HI
            veto = (h[tt] >= H_HIGH and not loaded[tt])
            a, b = max(0, tt - 8), min(n, tt + 8)
            print(f"  air->contact @t={tt/FPS:.2f}s  vel_cont={vel_cont} pres_cont={pres_cont} "
                  f"h_veto_would={veto}  h={h[tt]*100:+.1f}cm  vz={vz[tt]:+.2f}")
            print(f"    {'t':>6} {'v':>6} {'vz':>7} {'az':>7} {'h':>6} {'corr':>7} {'ld':>3} {'st':>3}")
            for i in range(a, b):
                print(f"    {i/FPS:6.2f} {v[i]:6.2f} {vz[i]:7.2f} {az[i]:7.2f} {h[i]*100:6.1f} "
                      f"{corr[i]:7.1f} {int(loaded[i]):3d} {int(state[i]):3d}")


analyze("S11113")
analyze("S8023")
