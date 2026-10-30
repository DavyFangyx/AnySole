"""HMR2.0 TTA 阶段 2：AnySole 协议窗口级评估（touch_gait env）。

读阶段 1 落盘的 SMPL-24 相机系关节（predictions/tta_theta/
{sid}_i{N}_cams{...}.npz），经评估侧标定旋转到世界系（R = (Rc@Rm)^T，
仅评估用），按 AnySole 窗口（tw=20、stride=20）与 kp_gt 计算四臂
（per-view 初值均值 / init 聚合 / per-view 精修均值 / virtual）的
三指标（PA-MPJPE / MPJPE / GlobalMPJPE，anysole.utils.metrics 唯一实现，
与 AnySole TTA 同一评估脚本口径）：
- PA-MPJPE：逐帧骨盆对齐 + Procrustes（anysole.utils.metrics.pa_mpjpe）
- MPJPE：逐帧骨盆对齐后欧氏（mean_point_error(pelvis_align)）
- GlobalMPJPE：原始世界系逐关节误差（无对齐；模型无轨迹头，世界平移
  来自 GT 骨盆锚——与 AnySole 线 gmpjpe 的 GT 锚定同口径类，登记）

视角组合（2026-10-30 与 AnySole 线对齐）：--i N 聚合全部 C(4,N) 组合，
std 列人口 = 组合间；i=4 单一组合无组合间 std（表注）。n = 实际成功
评估的窗口数。

用法（touch_gait env）：
  python hmr2_tta/eval/eval_tta_windows.py --sessions S13011 --n-windows 16 [--i 4]
输出：results/baselines/HMR2.0/metrics/zero/tta_theta_windows_i{N}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根
from hmr2_tta.paths import RESULTS_ROOT, calibration_json  # noqa: E402
from hmr2_tta.data.assemble_npz import date_of, load_calibration_chain, load_manifest_rows  # noqa: E402
from anysole.utils.metrics import mean_point_error, pa_mpjpe, pelvis_align  # noqa: E402


def window_metrics(pred_w: np.ndarray, gt_w: np.ndarray) -> dict:
    """(tw,24,3) 窗口三指标（AnySole 口径唯一实现）。"""
    p, _ = pelvis_align(pred_w)
    g, _ = pelvis_align(gt_w)
    return {
        "pa": float(pa_mpjpe(pred_w, gt_w).mean()),
        "mpjpe": float(mean_point_error(p, g, scale=1000.0).mean()),
        "gmpjpe": float(mean_point_error(pred_w, gt_w, scale=1000.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="S13011")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--i", type=int, default=None, choices=(1, 2, 3, 4),
                    help="视角数：聚合全部 C(4,i) 组合（组合间 std）；缺省 = 固定 4 视角")
    ap.add_argument("--cams", default=None, help="逗号分隔相机子集（如 1,2）；与 --i 互斥")
    args = ap.parse_args()

    if args.i is not None and args.cams:
        raise SystemExit("--i 与 --cams 互斥")
    if args.cams:
        cams_list = [tuple(int(x) for x in args.cams.split(",") if x.strip())]
    elif args.i is not None:
        cams_list = list(combinations((1, 2, 3, 4), args.i))
    else:
        cams_list = [(1, 2, 3, 4)]

    rows = load_manifest_rows(3)
    summary = {"i": args.i, "cams_list": [",".join(map(str, c)) for c in cams_list],
               "combos": {}, "sessions": {}}
    for sid in args.sessions.split(","):
        date = date_of(rows[sid])
        # 逐视角 R_wc：各相机各自的相机系→世界旋转（评估侧标定；模型输出
        # 在各自相机系，不能用单一相机的链旋转所有视角）
        R_wc = {}
        for cam in (1, 2, 3, 4):
            cal = load_calibration_chain(calibration_json(date), cam)
            g2c = cal["Rc"] @ cal["Rm"]        # 世界→相机系（GT 口径）
            R_wc[cam] = g2c.T
        for cams in cams_list:
            combo_key = ",".join(map(str, cams))
            n_views = len(cams)
            fname = f"{sid}_i{n_views}_cams{combo_key}.npz"
            d = np.load(RESULTS_ROOT / "predictions/tta_theta" / fname)
            fid = d["frame_id"]
            combo = {arm: {m: [] for m in ("pa", "mpjpe", "gmpjpe")}
                     for arm in ("pa_init", "pa_virt_init",
                                 "pa_refined", "pa_virtual")}
            combo_n = 0
            for st in range(0, args.n_windows * 20, 20):
                sel = (fid >= st) & (fid < st + 20)
                if sel.sum() != 20:
                    continue
                kp_gt = _load_window_gt(sid, st)
                if kp_gt is None:
                    continue
                def world_a(a, cam):
                    # 相机系→世界：先骨盆锚定（去模型空间骨盆偏移）再旋转，
                    # 加 GT 骨盆世界平移（模型无轨迹头，世界平移来自 GT 锚——
                    # 与 AnySole 线 gmpjpe 同口径类）
                    return np.einsum("ij,tpj->tpi", R_wc[cam],
                                     a - a[:, 0:1]) + kp_gt[:, 0:1]
                # i=1：init 臂即单视角值（其余臂不填）
                init_v = np.stack([world_a(d["joints_init_v"][sel, v], cams[v])
                                   for v in range(n_views)])
                init_m = [window_metrics(init_v[v], kp_gt) for v in range(n_views)]
                for m in ("pa", "mpjpe", "gmpjpe"):
                    combo["pa_init"][m].append(float(np.mean(
                        [w[m] for w in init_m])))
                if n_views > 1:
                    # virtual 臂：免标定下全局朝向为各视角混合、无单一相机系——
                    # 以各视角相机系分别旋转后平均（口径登记；PA 不受影响）
                    virt0_m = [window_metrics(world_a(d["joints_virt_init"][sel],
                                                     cams[v]), kp_gt)
                               for v in range(n_views)]
                    ref_v = np.stack([world_a(d["joints_refined_v"][sel, v], cams[v])
                                      for v in range(n_views)])
                    ref_m = [window_metrics(ref_v[v], kp_gt) for v in range(n_views)]
                    virt_m = [window_metrics(world_a(d["joints_virtual"][sel], cams[v]),
                                             kp_gt)
                              for v in range(n_views)]
                    for m in ("pa", "mpjpe", "gmpjpe"):
                        combo["pa_virt_init"][m].append(float(np.mean(
                            [w[m] for w in virt0_m])))
                        combo["pa_refined"][m].append(float(np.mean(
                            [w[m] for w in ref_m])))
                        combo["pa_virtual"][m].append(float(np.mean(
                            [w[m] for w in virt_m])))
                combo_n += 1
            combo_mean = {}
            for arm, ms in combo.items():
                combo_mean[arm] = {m: (float(np.mean(v)) if v else None)
                                   for m, v in ms.items()}
            combo_mean["n"] = combo_n
            summary["combos"][combo_key] = combo_mean
            summary["sessions"].setdefault(sid, {})[combo_key] = combo_mean
            print(f"[{sid} {combo_key}] {combo_mean}")
    # 组合间汇总（std 人口 = 组合间；单组合 → None）
    overall = {}
    for arm in ("pa_init", "pa_virt_init", "pa_refined", "pa_virtual"):
        overall[arm] = {}
        for m in ("pa", "mpjpe", "gmpjpe"):
            vals = [summary["combos"][ck][arm][m] for ck in summary["combos"]
                    if summary["combos"][ck].get(arm) is not None
                    and summary["combos"][ck][arm].get(m) is not None]
            overall[arm][m] = {"mean": float(np.mean(vals)) if vals else None,
                               "std": float(np.std(vals)) if len(vals) > 1 else None}
    summary["overall"] = overall
    print("overall:", overall)
    out = RESULTS_ROOT / "metrics/zero" / \
        f"tta_theta_windows_i{args.i or len(cams_list[0])}.json"
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
