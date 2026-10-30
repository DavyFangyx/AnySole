"""AnySole TTA 用 per-camera 2D GT 预计算（SMPL-24 原生关节，全分辨率像素）。

GT 世界关节 = load_smpl 的 pose_6d + trans_m + offsets_m + parents 经项目 FK
（anysole.utils.geometry.fk_pose6d_np 只读复用）；投影 = HMR2.0 装配同链
（project_world_to_cam，含畸变）。输出 kp2d (N,24,3) px+conf。

用法（hmr2 env）：python anysole_tta/prep_kp2d24.py --split val
输出：results/baselines/HMR2.0/data/kp2d24/cam{N}/{sid}.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
from hmr2_tta.paths import RESULTS_ROOT, resolve_uri  # noqa: E402
from hmr2_tta.data.assemble_npz import (load_calibration_chain,  # noqa: E402
                                         load_manifest_rows, project_world_to_cam)
from hmr2_tta.paths import calibration_json  # noqa: E402
from anysole.data.smpl_io import load_smpl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    from anysole.utils.geometry import fk_pose6d_np
    rows = load_manifest_rows(3)
    index = json.loads((RESULTS_ROOT / "data/index" / f"{args.split}.json").read_text())
    sids = sorted({e["sid"] for e in index})
    for sid in sids:
        row = rows[sid]
        import re
        date = re.search(r"(\d{8})", row["video_path"]).group(1)
        from hmr2_tta.paths import facts_session_dir
        facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
        smpl = load_smpl(resolve_uri(row["smpl_path"]), query_t=facts["mocap_time_s"])
        n = len(facts["frame_id"])
        gt_pose = smpl["pose_6d"].reshape(n, 24, 6)
        gt_j = fk_pose6d_np(gt_pose, smpl["trans_m"], smpl["offsets_m"], smpl["parents"])
        for cam in (1, 2, 3, 4):
            out_path = RESULTS_ROOT / "data/kp2d24" / f"cam{cam}" / f"{sid}.npz"
            if out_path.exists():
                print(f"[skip] {sid} cam{cam}")
                continue
            cal = load_calibration_chain(calibration_json(date), cam)
            px = project_world_to_cam(gt_j, cal)  # (N,24,2)
            h, w = 1240.0, 1624.0
            conf = ((px[..., 0] >= 0) & (px[..., 0] < w) &
                    (px[..., 1] >= 0) & (px[..., 1] < h)).astype(np.float32)
            kp2d = np.concatenate([px, conf[..., None]], -1).astype(np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, kp2d=kp2d, frame_id=facts["frame_id"].astype(np.int64))
        print(f"[ok] {sid}")


if __name__ == "__main__":
    main()
