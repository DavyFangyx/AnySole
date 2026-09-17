"""Measure E3 exported-BVH discontinuity: per-frame jitter, seam jumps, vs GT.

- Pred: results/AnySole/E3_nocontact/predictions/eval_bvh/<S>_VT2M.bvh
- GT:   AnysoleWorkspace/sources/raw/<date>/mocap_ori_bvh/<S>/*.bvh
Both parsed with the display pipeline (z-up conversion is an orthogonal
transform, so displacement magnitudes are frame-of-reference independent).
"""
import sys
from pathlib import Path

import numpy as np

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT / "results_display" / "script"))
sys.path.insert(0, str(GAIT))
from bvh_aligner_pose import parse_bvh_aligner  # noqa: E402

PRED_DIR = GAIT / "results/AnySole/E3_nocontact/predictions/eval_bvh"
GT_ROOT = GAIT / "AnysoleWorkspace" / "sources" / "raw"


def find_gt_bvh(session: str) -> Path:
    cands = sorted(GT_ROOT.glob("*/mocap_ori_bvh/%s/*.bvh" % session))
    return cands[0] if cands else None


def jitter(joints):
    d = np.linalg.norm(np.diff(joints, axis=0), axis=-1)  # (T-1, J)
    return d * 10.0  # mm (BVH offsets are cm)


def main():
    sessions = ["S7013", "S10103", "S11023", "S13013", "S11053", "S11113"]
    print("%-8s %-5s %6s %6s %8s %8s %10s %10s %8s"
          % ("session", "type", "frames", "J", "mean_ji", "p99_ji", "max_ji", "seam_mean", "seam_n"))
    for s in sessions:
        pred_path = PRED_DIR / ("%s_VT2M.bvh" % s)
        if not pred_path.is_file():
            continue
        pred = parse_bvh_aligner(pred_path, trim_leading_seconds=0.0)  # no trim: seams at 20k
        pj = jitter(pred["joints"])
        # seams at window boundaries (tw=20, stride=20 -> every 20 frames)
        seams = np.arange(19, pj.shape[0], 20)
        seam_val = pj[seams]
        print("%-8s %-5s %6d %6d %8.2f %8.2f %10.2f %10.2f %8d"
              % (s, "pred", pred["joints"].shape[0], pred["joints"].shape[1],
                 pj.mean(), np.percentile(pj, 99), pj.max(),
                 seam_val.mean() if len(seam_val) else float("nan"), len(seam_val)))
        gt_path = find_gt_bvh(s)
        if gt_path is None:
            print("%-8s %-5s %s" % (s, "gt", "NOT FOUND"))
            continue
        gt = parse_bvh_aligner(gt_path, trim_leading_seconds=0.0)
        gj = jitter(gt["joints"])
        print("%-8s %-5s %6d %6d %8.2f %8.2f %10.2f %10.2f"
              % (s, "gt", gt["joints"].shape[0], gt["joints"].shape[1],
                 gj.mean(), np.percentile(gj, 99), gj.max(), gj[::20].mean()))
        # alignment-free spectral view: what fraction of pred frames exceed
        # 2x the GT 99th-percentile jitter
        frac = float((pj > 2 * np.percentile(gj, 99)).mean())
        print("%-8s %-5s  frames > 2x GT p99: %.2f%%" % (s, "ratio", frac))


if __name__ == "__main__":
    main()
