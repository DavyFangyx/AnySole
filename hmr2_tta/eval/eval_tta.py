"""论文 TTA 复现评估（E3「TTA 原方法」证据源，2026-10-30 收口版，无标定）。

单命令 = TTA + AnySole 协议窗口评估（2026-10-31 合并：跑模型即算指标）。
同帧 V 视角 per-view token 优化 + virtual view；每窗在线按 AnySole 窗口
协议（tw=20、stride=20）与 GT（load_smpl + FK，与 AnySoleDataset 的
kp_gt 同源）计算四臂三指标（PA-MPJPE / MPJPE / GlobalMPJPE，
anysole.utils.metrics 唯一实现）：
- PA-MPJPE：逐帧骨盆对齐 + Procrustes（pa_mpjpe）
- MPJPE：逐帧骨盆对齐后欧氏（mean_point_error(pelvis_align)）
- GlobalMPJPE：原始世界系逐关节误差（无对齐；模型无轨迹头，世界平移
  来自 GT 骨盆锚——与 AnySole 线 gmpjpe 的 GT 锚定同口径类，登记）

视角组合（与 AnySole 线对齐）：--i N 跑全部 C(4,N) 组合，std 列人口 =
组合间；i=4 单一组合无组合间 std（表注）；i=1 单视角无 TTA（只 init 臂）。
n = 实际成功评估的窗口数。

设计（2026-10-30 收口 + 用户裁定）：
- 无标定：L_2D = 模型自带弱透视投影（deccam），零外部标定；
- L_2D 目标 = RTMPose-Halpe26 检测器（21 点监督子集，与 AnySole 线同一映射表）；
- 统一读出 SMPL-24：TTA 用原生 smplx 层（非 44 关节包装器）；
- 优化协议 = 论文：200 步、lr 0.1、20 步 warmup、梯度裁剪 0.1。

用法（hmr2 env）：
  python hmr2_tta/eval/eval_tta.py --split test --i 4 --device cuda:2
输出：results/baselines/HMR2.0/metrics/zero/tta_theta_windows_i{N}.json
"""
from __future__ import annotations
# ---- CPU 占用控制（2026-10-31）----
# TTA 驱动每优化步做 CPU 侧 numpy/scipy 计算（1-δ 滤波等）；numpy/torch
# 缺省线程数 = 全核，单进程会吃 5 核以上。与 configs 系统限定运算池同思路：
# 置单线程——并发多进程时各占 ~1 核，总 CPU 可控。
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根
from hmr2_tta.paths import (PROJECT_ROOT, RESULTS_ROOT, calibration_json,  # noqa: E402
                            facts_session_dir)
from hmr2_tta.data.assemble_npz import date_of, load_calibration_chain, load_manifest_rows  # noqa: E402
from hmr2_tta.models.loaders import load_hmr2_plain_with_token  # noqa: E402
from hmr2_tta.models.tta import TTAFrameSet, init_virtual_view, rot_to_6d  # noqa: E402
from anysole.utils.metrics import mean_point_error, pa_mpjpe, pelvis_align  # noqa: E402

CONF_THR = 0.5
SPLITS_CSV = PROJECT_ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
# Halpe-26 → SMPL-24 监督子集（与 AnySole TTA 同一映射表，21 点）
H26_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
             0, 20, 21, 22, 23, 24, 25]


def split_sessions(name: str) -> list[str]:
    """splits.csv 的 split 列会话（2026-10-31：SIDS 手写串作废）。"""
    rows = list(csv.DictReader(SPLITS_CSV.open(encoding="utf-8-sig", newline="")))
    return sorted({r[name].strip() for r in rows if r[name].strip()})


def detector_2d(kp: np.ndarray, score: np.ndarray, center: np.ndarray,
                patch_w: float) -> np.ndarray:
    """RTMPose-Halpe26 帧级检测 (26,2)px → (21,3) 归一化 crop + conf（监督子集序）。"""
    norm = (kp - center) / patch_w
    out = np.zeros((len(H26_ORDER), 3), dtype=np.float32)
    for i, h in enumerate(H26_ORDER):
        if score[h] >= CONF_THR:
            out[i, :2] = norm[h]
            out[i, 2] = score[h]
    return out


def window_metrics(pred_w: np.ndarray, gt_w: np.ndarray) -> dict:
    """(tw,24,3) 窗口三指标（AnySole 口径唯一实现）。"""
    p, _ = pelvis_align(pred_w)
    g, _ = pelvis_align(gt_w)
    return {
        "pa": float(pa_mpjpe(pred_w, gt_w).mean()),
        "mpjpe": float(mean_point_error(p, g, scale=1000.0).mean()),
        "gmpjpe": float(mean_point_error(pred_w, gt_w, scale=1000.0).mean()),
    }


def _combo_done(cams, sids_expected: int) -> bool:
    """该组合的产物文件已含全部会话 → 跳过（断点续跑）。"""
    import json as _json
    combo = ",".join(map(str, cams))
    p = RESULTS_ROOT / "metrics/zero" / \
        f"tta_theta_windows_i{len(cams)}_cams{combo}.json"
    if not p.is_file():
        return False
    try:
        d = _json.loads(p.read_text(encoding="utf-8"))
        n = max((len(v) for v in d.get("sessions", {}).values()), default=0)
        if n >= sids_expected:
            print(f"[skip] {combo} 已完成（{n} 会话）")
            return True
    except Exception:
        pass
    return False


def _launch_parallel(args, cams_list) -> None:
    """--gpu 分卡：逐组合子进程并行（本进程只调度，全部完成后退出）。"""
    import signal
    import subprocess
    import time
    gpus = [int(x) for x in args.gpu.split(",") if x.strip()]
    procs = []

    def _forward_signal(signum, frame):
        # Ctrl-C/SIGTERM 只发给本进程组；子进程在独立会话收不到——
        # 父进程收到后转杀全部子进程再退出（2026-10-31 修复孤儿进程问题）
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        raise SystemExit(f"interrupted (signal {signum})")
    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)
    for k, cams in enumerate(cams_list):
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--split", args.split, "--cams", ",".join(map(str, cams)),
               "--device", "cuda:%d" % gpus[k % len(gpus)]]
        if args.sessions:
            cmd += ["--sessions", args.sessions]
        if args.n_windows > 0:
            cmd += ["--n-windows", str(args.n_windows)]
        if args.steps != 200:
            cmd += ["--steps", str(args.steps)]
        if args.dump_kp:
            cmd += ["--dump-kp", str(args.dump_kp)]
        print("[launch]", ",".join(map(str, cams)), "->", "cuda:%d" % gpus[k % len(gpus)])
        procs.append(subprocess.Popen(cmd, start_new_session=True))
    # 完成即跳过：per-combo 文件已含全部会话的组合不再重发（断点续跑）
    sids_expected = len(args.sessions.split(",")) if args.sessions \
        else len(split_sessions(args.split))
    procs = [p for p, c in zip(procs, cams_list)
             if not _combo_done(c, sids_expected)]
    while any(p.poll() is None for p in procs):
        time.sleep(10)
    bad = [p for p in procs if p.returncode != 0]
    print(f"[launch] {len(procs) - len(bad)}/{len(procs)} 组合完成")
    if bad:
        raise SystemExit(f"{len(bad)} 个组合失败")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--sessions", default=None, help="逗号分隔；缺省 = split 全集")
    ap.add_argument("--n-windows", type=int, default=0,
                    help="限制前 N 窗（2026-11-01 起缺省 0 = 全量：每会话完整序列）")
    ap.add_argument("--i", type=int, default=None, choices=(1, 2, 3, 4),
                    help="视角数：跑全部 C(4,i) 组合；缺省 = 固定 4 视角")
    ap.add_argument("--cams", default=None,
                    help="逗号分隔相机子集（如 1,2）；与 --i/--n 互斥")
    ap.add_argument("--n", default=None,
                    help="逗号分隔视角数（如 2,3,4；缺省 = 全部 1,2,3,4）")
    ap.add_argument("--gpu", default=None,
                    help="逗号分隔卡号（如 1,2,3,7）：逐组合并行分卡，"
                         "单命令跑完全部组合（缺省 = --device 单卡）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--lreg", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--dump-kp", default=None,
                    help="目录：每窗落 {sid}_w{st}.npz（逐组合子目录；"
                         "joints=世界系 virtual 精修聚合 24 关节/gt/帧号）")
    args = ap.parse_args()

    if args.i is not None and args.cams:
        raise SystemExit("--i 与 --cams 互斥")
    if args.n is not None and args.cams:
        raise SystemExit("--n 与 --cams 互斥")
    if args.cams:
        cams_list = [tuple(int(x) for x in args.cams.split(",") if x.strip())]
    elif args.i is not None:
        cams_list = list(combinations((1, 2, 3, 4), args.i))
    elif args.n is not None:
        cams_list = [c for n in (int(x) for x in args.n.split(",") if x.strip())
                     for c in combinations((1, 2, 3, 4), n)]
    else:
        cams_list = [c for n in (1, 2, 3, 4) for c in combinations((1, 2, 3, 4), n)]
    if args.gpu:
        _launch_parallel(args, cams_list)
        return
    sids = (args.sessions.split(",") if args.sessions else split_sessions(args.split))

    model, cfg = load_hmr2_plain_with_token()
    model.eval().to(args.device)
    rows = load_manifest_rows(3)

    from hmr2.utils.geometry import rot6d_to_rotmat
    from anysole.data.smpl_io import load_smpl, resolve_smpl_path
    from anysole.utils.geometry import fk_pose6d_np

    summary = {"split": args.split, "i": args.i,
               "cams_list": [",".join(map(str, c)) for c in cams_list],
               "combos": {}, "sessions": {}}  # 单进程多组合：整体 json
    for sid in sids:
        row = rows[sid]
        date = date_of(row)
        facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
        # GT 世界 24 关节（与 AnySoleDataset kp_gt 同源：SMPL 标注 + FK）
        smpl = load_smpl(resolve_smpl_path({"smpl_path": row["smpl_path"],
                                            "session_id": sid},
                                           (PROJECT_ROOT / "AnysoleWorkspace",)),
                         query_t=facts["mocap_time_s"])
        gt_world = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"],
                                smpl["offsets_m"], smpl["parents"])   # (N,24,3)
        npz3 = np.load(RESULTS_ROOT / "data/npz/cam3" / f"{sid}.npz")
        index = json.loads((RESULTS_ROOT / "data/index/test.json").read_text())
        from hmr2.datasets.image_dataset import ImageDataset
        dsets, preds, dets = {}, {}, {}
        for cam in (1, 2, 3, 4):
            e = next(x for x in index if x["sid"] == sid and x["cam"] == cam)
            dsets[cam] = ImageDataset(cfg, e["npz"], img_dir="", train=False)
            d = np.load(RESULTS_ROOT / "data/rtmpose26" / f"cam{cam}" / f"{sid}.npz")
            dets[cam] = d
        # 逐视角 R_wc：各相机各自的相机系→世界旋转（评估侧标定）
        R_wc = {}
        for cam in (1, 2, 3, 4):
            cal = load_calibration_chain(calibration_json(date), cam)
            R_wc[cam] = (cal["Rc"] @ cal["Rm"]).T
        # 2026-11-01 用户裁定：全量运行（废弃「前 16 窗」协议）；
        # --n-windows N>0 显式限制前 N 窗（冒烟用）
        max_frames = args.n_windows * 20 if args.n_windows > 0 else len(gt_world)
        frames = list(range(min(max_frames, len(gt_world))))

        for cams in cams_list:
            combo_key = ",".join(map(str, cams))
            n_views = len(cams)
            combo = {arm: {m: [] for m in ("pa", "mpjpe", "gmpjpe")}
                     for arm in ("pa_init", "pa_virt_init", "pa_refined", "pa_virtual")}
            combo_n = 0
            buf = {"init": [], "refined": [], "virtual": [], "virt_init": []}
            for f in frames:
                tokens, inits, det2d = [], {"pose_6d": [], "betas": [], "cam": []}, []
                centers, patch_ws = [], []
                for cam in cams:
                    item = dsets[cam][f]
                    img = torch.from_numpy(item["img"]).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        out = model.forward_step({"img": img})
                    tokens.append(out["token_out"][0])
                    rot = torch.cat([out["pred_smpl_params"]["global_orient"],
                                     out["pred_smpl_params"]["body_pose"]], 1)[0]
                    inits["pose_6d"].append(rot_to_6d(rot).reshape(144).cpu())
                    inits["betas"].append(out["pred_smpl_params"]["betas"][0].cpu())
                    inits["cam"].append(out["pred_cam"][0].cpu())
                    pw, ctr = item["box_size"], item["box_center"]
                    centers.append(ctr)
                    patch_ws.append(pw)
                    det2d.append(detector_2d(dets[cam]["kp"][f],
                                             dets[cam]["score"][f], ctr, pw))
                if n_views == 1:
                    # 单视角：无 TTA（与 AnySole 线同语义），只 init 臂
                    rot_i0 = rot6d_to_rotmat(
                        inits["pose_6d"][0].to(args.device).unsqueeze(0)).view(1, 24, 3, 3)
                    with torch.no_grad():
                        j_init = _single_smpl24(model, rot_i0,
                                                inits["betas"][0].to(args.device).unsqueeze(0))
                    buf["init"].append(j_init[0:1, :24].cpu().numpy())
                    buf["refined"].append(j_init[0:1, :24].cpu().numpy())
                    buf["virtual"].append(j_init[0:1, :24].cpu().numpy())
                    buf["virt_init"].append(j_init[0:1, :24].cpu().numpy())
                else:
                    tta = TTAFrameSet(model, torch.stack(tokens).to(args.device),
                                      {"pose_6d": torch.stack(inits["pose_6d"]).to(args.device),
                                       "betas": torch.stack(inits["betas"]).to(args.device),
                                       "cam": torch.stack(inits["cam"]).to(args.device)},
                                      np.stack(det2d),
                                      np.stack(centers), np.stack(patch_ws),
                                      lambda_reg=args.lreg)
                    with torch.no_grad():
                        j_inits = []
                        for vi in range(n_views):
                            rot_i0 = rot6d_to_rotmat(
                                inits["pose_6d"][vi].to(args.device).unsqueeze(0)).view(1, 24, 3, 3)
                            j_i0, _ = tta._smpl_forward(
                                rot_i0, inits["betas"][vi].to(args.device).unsqueeze(0))
                            j_inits.append(j_i0[0, :24].detach().cpu().numpy())
                        virt0 = init_virtual_view(
                            torch.stack(inits["pose_6d"]).to(args.device),
                            torch.stack(inits["betas"]).to(args.device),
                            torch.stack(inits["cam"]).to(args.device))
                        rot_v0 = rot6d_to_rotmat(
                            virt0["pose_6d"].to(args.device).unsqueeze(0)).view(24, 3, 3)
                        j_v0, _ = tta._smpl_forward(rot_v0.unsqueeze(0),
                                                    virt0["betas"].to(args.device).unsqueeze(0))
                    tta.optimize(steps=args.steps, lr=args.lr, warmup=args.warmup,
                                 clip=args.clip)
                    out_t = tta.result()
                    buf["init"].append(np.stack(j_inits))
                    buf["refined"].append(
                        np.stack([out_t["joints_i"][v][:24] for v in range(n_views)]))
                    buf["virtual"].append(out_t["joints0"][0][:24])
                    buf["virt_init"].append(j_v0[0, :24].cpu().numpy())
                if len(buf["init"]) == 20:
                    st = f - 19
                    gt_w = gt_world[st:st + 20]
                    pelvis_gt = gt_w[:, 0:1]

                    def world_a(a, cam):
                        # 骨盆锚定 → 相机旋转 → + GT 骨盆世界平移（同口径类登记）
                        return np.einsum("ij,tpj->tpi", R_wc[cam],
                                         a - a[:, 0:1]) + pelvis_gt
                    init_w = np.stack(buf["init"])          # (20,V,24,3)
                    init_m = [window_metrics(world_a(init_w[:, v], cams[v]), gt_w)
                              for v in range(n_views)]
                    for m in ("pa", "mpjpe", "gmpjpe"):
                        combo["pa_init"][m].append(float(np.mean(
                            [w[m] for w in init_m])))
                    if n_views > 1:
                        virt0_w = np.stack(buf["virt_init"])   # (20,24,3)
                        ref_w = np.stack(buf["refined"])       # (20,V,24,3)
                        virt_w = np.stack(buf["virtual"])      # (20,24,3)
                        virt0_m = [window_metrics(world_a(virt0_w, cams[v]), gt_w)
                                   for v in range(n_views)]
                        ref_m = [window_metrics(world_a(ref_w[:, v], cams[v]), gt_w)
                                 for v in range(n_views)]
                        virt_m = [window_metrics(world_a(virt_w, cams[v]), gt_w)
                                  for v in range(n_views)]
                        for m in ("pa", "mpjpe", "gmpjpe"):
                            combo["pa_virt_init"][m].append(float(np.mean(
                                [w[m] for w in virt0_m])))
                            combo["pa_refined"][m].append(float(np.mean(
                                [w[m] for w in ref_m])))
                            combo["pa_virtual"][m].append(float(np.mean(
                                [w[m] for w in virt_m])))
                        if args.dump_kp:
                            # 落盘（与 run_tta --dump-kp 同口径）：世界系
                            # virtual 精修聚合（= pa_virtual 评估量，视角
                            # 间均值）+ GT + 帧号；逐组合子目录防覆盖
                            dump_dir = Path(args.dump_kp) / combo_key
                            dump_dir.mkdir(parents=True, exist_ok=True)
                            joints_w = np.mean(
                                [world_a(virt_w, cams[v]) for v in range(n_views)],
                                axis=0)
                            np.savez_compressed(
                                dump_dir / f"{sid}_w{st}.npz",
                                joints=joints_w.astype(np.float32),
                                gt=gt_w.astype(np.float32),
                                frame_ids=np.arange(st, st + 20, dtype=np.int64))
                    combo_n += 1
                    buf = {k: [] for k in buf}
            combo_mean = {}
            for arm, ms in combo.items():
                combo_mean[arm] = {m: (float(np.mean(v)) if v else None)
                                   for m, v in ms.items()}
            combo_mean["n"] = combo_n
            summary["combos"][combo_key] = combo_mean
            summary["sessions"].setdefault(sid, {})[combo_key] = combo_mean
            print(f"[{sid} {combo_key}] {combo_mean}", flush=True)
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
    if args.cams:
        # --cams 单组合：per-combo 文件
        combo_key = ",".join(map(str, cams_list[0]))
        out = RESULTS_ROOT / "metrics/zero" / \
            f"tta_theta_windows_i{len(cams_list[0])}_cams{combo_key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print("→", out)
    else:
        # 多组合：每个组合单独落盘 per-combo 文件（导出器按 glob 聚合）
        for cams in cams_list:
            combo_key = ",".join(map(str, cams))
            one = {"split": args.split, "i": len(cams),
                   "cams_list": [combo_key],
                   "combos": {combo_key: summary["combos"].get(combo_key, {})},
                   "sessions": {sid: {combo_key: summary["sessions"][sid][combo_key]}
                                for sid in summary["sessions"]
                                if combo_key in summary["sessions"][sid]},
                   "overall": {arm: {m: {"mean": summary["combos"]
                                         [combo_key][arm][m], "std": None}
                                     for m in ("pa", "mpjpe", "gmpjpe")}
                               for arm in ("pa_init", "pa_virt_init",
                                           "pa_refined", "pa_virtual")
                               if summary["combos"].get(combo_key, {})
                               .get(arm, {}).get("pa") is not None}}
            out = RESULTS_ROOT / "metrics/zero" / \
                f"tta_theta_windows_i{len(cams)}_cams{combo_key}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(one, indent=1), encoding="utf-8")
        print("→", str(RESULTS_ROOT / "metrics/zero") +
              "/tta_theta_windows_i*_cams*.json")


def _single_smpl24(model, rotmat, betas):
    """V=1 单视角 init 用：原生 SMPL-24 正演（按设备缓存复用层）。"""
    from hmr2_tta.models.tta import _smpl_layer
    layer = _smpl_layer(rotmat.device)
    return layer(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:],
                 betas=betas, pose2rot=False).joints


if __name__ == "__main__":
    main()
