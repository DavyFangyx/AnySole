"""HMR2.0 TTA 阶段 2：AnySole 协议窗口级评估（touch_gait env）。

读阶段 1 落盘的 SMPL-24 相机系关节（predictions/tta_theta/{sid}.npz），
经评估侧标定旋转到世界系（R = (Rc@Rm)^T，仅评估用），按 AnySole 窗口
（tw=20、stride=20）与 kp_gt 计算四臂 PA-MPJPE（anysole.utils.metrics，
与 AnySole TTA 同一评估脚本口径）：
① per-view 初值均值 ② init 聚合 ③ per-view 精修均值 ④ virtual。

用法（touch_gait env）：
  python eval/eval_tta_windows.py --sessions S13011 --n-windows 16
输出：results/baselines/HMR2.0/metrics/zero/tta_theta_windows.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import RESULTS_ROOT, facts_session_dir  # noqa: E402
from hmr2_tta.data.assemble_npz import (date_of, load_calibration_chain,  # noqa: E402
                                         load_manifest_rows)
from anysole.utils.metrics import pelvis_align, pa_mpjpe  # noqa: E402


def pa_window(pred_w: np.ndarray, gt_w: np.ndarray) -> float:
    """(tw,24,3) 窗口级 PA-MPJPE（AnySole 口径，逐帧骨盆对齐 + Procrustes 均值）。"""
    p, _ = pelvis_align(pred_w)
    r, _ = pelvis_align(gt_w)
    return float(pa_mpjpe(p, r).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="S13011")
    ap.add_argument("--n-windows", type=int, default=16)
    args = ap.parse_args()

    rows = load_manifest_rows(3)
    summary = {"sessions": {}}
    for sid in args.sessions.split(","):
        date = date_of(rows[sid])
        d = np.load(RESULTS_ROOT / "predictions/tta_theta" / f"{sid}.npz")
        fid = d["frame_id"]
        from hmr2_tta.paths import calibration_json
        cal = load_calibration_chain(calibration_json(date), 3)
        g2c = cal["Rc"] @ cal["Rm"]                    # 世界→相机系（GT 口径）
        R_wc = g2c.T                                    # 相机系→世界（评估侧标定）
        res = {k: [] for k in ("pa_init", "pa_virt_init", "pa_refined", "pa_virtual")}
        for st in range(0, args.n_windows * 20, 20):
            sel = (fid >= st) & (fid < st + 20)
            if sel.sum() != 20:
                continue
            kp_gt = _load_window_gt(sid, st)
            if kp_gt is None:
                continue
            rot = lambda a: np.einsum("ij,tpj->tpi", R_wc, a[sel])
            init_v = np.stack([rot(d["joints_init_v"][sel, v]) for v in range(4)])
            res["pa_init"].append(float(np.mean(
                [pa_window(init_v[v], kp_gt) for v in range(4)])))
            res["pa_virt_init"].append(pa_window(rot(d["joints_virt_init"]), kp_gt))
            ref_v = np.stack([rot(d["joints_refined_v"][sel, v]) for v in range(4)])
            res["pa_refined"].append(float(np.mean(
                [pa_window(ref_v[v], kp_gt) for v in range(4)])))
            res["pa_virtual"].append(pa_window(rot(d["joints_virtual"]), kp_gt))
        summary["sessions"][sid] = {k: float(np.mean(v)) if v else None
                                    for k, v in res.items()}
        summary["sessions"][sid]["n"] = sum(
            1 for _ in range(0, args.n_windows * 20, 20))
        print(f"[{sid}] {summary['sessions'][sid]}")
    out = RESULTS_ROOT / "metrics/zero/tta_theta_windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("→", out)


def _load_window_gt(sid: str, st: int):
    """窗口 GT（AnySole 数据集口径）：kp_gt (tw,24,3) 世界系。"""
    from anysole.data.dataset import AnySoleDataset
    ds = AnySoleDataset("test", session_ids=[sid], cam_id=3, stride=20)
    for i in range(len(ds)):
        it = ds[i]
        if int(it["frame_start"].reshape(-1)[0]) == st:
            return it["kp_gt"].numpy()
    return None


if __name__ == "__main__":
    main()
