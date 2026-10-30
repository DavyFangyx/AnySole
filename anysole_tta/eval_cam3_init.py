"""V3_Bv1 cam3 单视角 init 基线（三臂比较的臂①：原方案单视角推理）。

与 run_tta.py 同口径（CONFIG_VT、前 16 窗口、窗口级 PA-MPJPE），只跑 cam3、
不做优化。四相机均值基线（run_tta.py 的 pa_init）混入了 1/2/4 相机分支——
模型按 D1 只吃 cam3 特征训练，本脚本给出可与生产 VT2M 对标的单视角臂。

用法（touch_gait env）：
  python anysole_tta/eval_cam3_init.py --device cuda:7
输出：results/baselines/HMR2.0/metrics/anysole_tta_stage1_cam3_init.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（anysole/anysole_tta/hmr2_tta）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hmr2_gait 根
from hmr2_tta.paths import RESULTS_ROOT  # noqa: E402
from anysole.types import CONFIG_VT  # noqa: E402
from anysole_tta.run_tta import (VAL_SESSIONS, TEST_SESSIONS,  # noqa: E402
                                 build_model, pa_mpjpe_np, session_date)


def run_session(model, sid, n_windows, device, split):
    from anysole.data.dataset import AnySoleDataset, collate_windows
    from anysole.train import condition_inputs, move_batch
    from anysole.utils.geometry import fk_pose6d

    ds = AnySoleDataset(split, session_ids=[sid], cam_id=3, stride=20)
    its = [ds[i] for i in range(len(ds))]
    starts = sorted(int(it["frame_start"].reshape(-1)[0]) for it in its)[:n_windows]
    items = {int(it["frame_start"].reshape(-1)[0]): it for it in its}
    pas = []
    for st in starts:
        it = items[st]
        batch = move_batch(collate_windows([it]), device)
        config_id = torch.full((1,), CONFIG_VT, device=device)
        v_feat, t_raw, t_phys, _ = condition_inputs(batch, config_id)
        with torch.no_grad():
            out = model(v_feat, t_raw, t_phys, config_id)
        kp = fk_pose6d(out["x0_hat"].reshape(1, -1, 144),
                       (out["trans_hat"] + it["trans_anchor"].to(device)).reshape(1, -1, 3),
                       it["offsets"].to(device), it["parents"].to(device))[0]
        pas.append(pa_mpjpe_np(kp.detach().cpu().numpy(), it["kp_gt"].cpu().numpy()))
    return float(np.mean(pas)), len(pas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sids = VAL_SESSIONS if args.split == "val" else TEST_SESSIONS
    out_path = args.out or str(RESULTS_ROOT / "metrics" /
                               f"anysole_tta_stage1_{args.split}_cam3_init.json")
    model, _ = build_model(args.device)
    summary = {"sessions": {}}
    for sid in sids:
        mean, n = run_session(model, sid, args.n_windows, args.device, args.split)
        summary["sessions"][sid] = {"pa_init_cam3": mean, "n": n}
        print(f"[{sid}] cam3 init: {mean:.1f} (n={n})")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    overall = float(np.mean([s["pa_init_cam3"] for s in summary["sessions"].values()]))
    print(f"overall cam3 init: {overall:.1f}")
    print("→", out_path)


if __name__ == "__main__":
    main()
