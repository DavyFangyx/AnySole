"""Dump F6 signals for S11113 (left foot) to locate the mid-descent false contact."""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path("/data/fangyuxuan/projects/gait/results_display/script")
sys.path.insert(0, str(SCRIPT_DIR))
import contact_methods as cm  # noqa: E402

seq_dir = cm.cli_common.DEFAULT_SEQ_ROOT / "20260808" / "S11" / "S11113"
ctx = cm.load_aligned(seq_dir)
pipe = cm._f6_pipeline(ctx)

FPS = cm.FPS
for side in ("left", "right"):
    sig = cm._f6_signals(ctx, side)
    p = pipe[side]
    v, az, h = sig["v"], sig["az"], sig["h"]
    state, loaded = p["state"], p["loaded"]
    corr = p["corrected"]
    print(f"=== {side} (thr_lo={p['thr_lo']:.1f}, thr_hi={p['thr_hi']:.1f}, calibrated={p['calibrated']}, n_air={p['n_air']}) ===")
    print(f"{'t_s':>6} {'state':>5} {'v':>6} {'az':>7} {'h_cm':>6} {'corr':>7} {'loaded':>6}")
    for t in range(len(v)):
        mark = ""
        if side == "left" and state[t] == 1 and t > 0 and state[t - 1] == 0:
            mark = " <-- air->contact"
        print(f"{t/FPS:6.2f} {int(state[t]):5d} {v[t]:6.2f} {az[t]:7.2f} {h[t]*100:6.1f} {corr[t]:7.1f} {int(loaded[t]):6d}{mark}")
    # find runs where state==1 while h is high and v is high (i.e. suspicious mid-descent contacts)
    runs = []
    t = 0
    n = len(v)
    while t < n:
        if state[t] == 1:
            s = t
            while t < n and state[t] == 1:
                t += 1
            seg_h = h[s:t]
            seg_v = v[s:t]
            runs.append((s, t, seg_h.min(), seg_v.min(), seg_h.mean()))
        else:
            t += 1
    print(f"--- {side} contact runs (start_s, end_s, min_h_cm, min_v, mean_h_cm):")
    for r in runs:
        print(f"  {r[0]/FPS:6.2f} -> {r[1]/FPS:6.2f}  min_h={r[2]*100:6.1f}cm  min_v={r[3]:5.2f}  mean_h={r[4]*100:6.1f}cm")
