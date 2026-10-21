#!/usr/bin/env python3
"""V5_1A §4.2 前置探针：相位/CoP 可读性（2026-10-19）.

零训练 ridge：从 T 侧 raw108 与 V 侧 v_tok（E v2.1 ckpt_best 冻结编码器）
读出每足 (cos, sin) 相位嵌入与 CoP 网格坐标。

- GT 相位：GT 足关节（10/11）世界高 < 地面 + 2cm 触地事件（D8 顶点口径
  同阈值；不用 f6_soft 磁盘标签——审计发现其为二值且 63/140 会话全 1
  退化）→ 每足归一化周期相位 φ = (t − td_k)/(td_{k+1} − td_k)，目标 =
  (cos 2πφ, sin 2πφ)；有效帧 = 首末触地之间、周期 ≥10 帧（每足独立掩码）；
- GT CoP：smplx 正演（smpl_vertices_from_archive_params）→ foot_heat
  4×12 顶点计数网格（复制自 results_display/script/d_test8_smpl_gt_cop.py
  foot_heat，provenance 注明）→ cop_from_grid（与触觉 CoP 同公式同单位
  0-1 网格坐标）；只统计接触帧（和值 > 0）；
- 特征：raw108 = [T_raw 96 | T_phys 12]（dataset 归一化口径）；v_tok =
  E encoders 单次前向（V5_1A 载体同源）；窗口级收集、按帧拼接
  （tw = stride 不重叠；窗帧范围取 dataset.valid_windows 保证 GT 对齐）；
- ridge：λ 网格 1e-3~1e5 对数 17 档；TRAIN 内 80/20 留出选参（会话级
  连续切分）→ 全量 TRAIN 重拟 → TEST（val ≡ test）打分；常数基线 =
  TRAIN 有效帧均值；
- 判据（V5_1A §4.2 逐通道独立）：启用门槛 = raw108 误差显著低于常数基线
  （逐会话配对 bootstrap 95% CI）；V 强语义 = v_tok 误差显著低于 raw108；
  停用门槛 = raw108 不优于常数。

用法（仓库根目录，touch_gait 环境）：
  python z_note/probes/probe_v5_phase_cop_readout.py --device cuda:3
  python z_note/probes/probe_v5_phase_cop_readout.py --device cuda:3 --max-sessions 2  # 冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.data.smpl_io import (
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.pressure import cop_from_grid
from anysole.data.workspace_adapter import load_shared_session
from anysole.eval import _load_model, resolve_device
from anysole.train import load_config
from anysole.utils.geometry import fk_pose6d_np

N_JOINTS = 24
ANKLE_L, ANKLE_R, FOOT_L, FOOT_R = 7, 8, 10, 11
GT_CONTACT_DZ = 0.02          # 触地阈值（D8 口径）
GT_ROWS, GT_COLS = 4, 12
GT_S0, GT_S1, GT_LAT = 0.10, 0.16, 0.045   # foot_heat 网格边界（D8 口径）
GT_ANKLE_DZ = 0.02
LAM_GRID = np.logspace(-3, 5, 17)
CHANNELS = ("phase_L", "phase_R", "cop_L", "cop_R")
READOUTS = ("raw108", "v_tok", "vt")


def foot_heat(verts, ankle, foot, floor_y, side="L"):
    """单帧单脚 GT 足底热图——复制自 d_test8_smpl_gt_cop.py foot_heat
    （2026-10-19，provenance：results_display/script/d_test8_smpl_gt_cop.py）。"""
    d = foot[[0, 2]] - ankle[[0, 2]]
    d = d / (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0]])
    rel = verts[:, [0, 2]] - ankle[[0, 2]][None]
    s = rel @ d
    lat = rel @ perp
    if side == "R":
        lat = -lat
    below = verts[:, 1] < float(ankle[1]) + GT_ANKLE_DZ
    keep = below & (s >= GT_S0 - 0.01) & (s <= GT_S1 + 0.01) & (np.abs(lat) <= GT_LAT + 0.01)
    contact = verts[:, 1] < floor_y + GT_CONTACT_DZ
    cols = np.clip(((GT_S1 - s) / (GT_S1 - GT_S0) * GT_COLS).astype(int), 0, GT_COLS - 1)
    rows = np.clip(((lat + GT_LAT) / (2 * GT_LAT) * GT_ROWS).astype(int), 0, GT_ROWS - 1)
    heat = np.zeros((GT_ROWS, GT_COLS), dtype=np.float32)
    np.add.at(heat, (rows[keep & contact], cols[keep & contact]), 1.0)
    return heat


def session_gt(sess: dict, seq_root: Path, corrected_offset: bool = False):
    """GT 相位（每足，运动学触地事件）与 GT CoP（每足，smplx
    正演 foot_heat → cop_from_grid），返回 dict of arrays (n_frames, ...)。

    corrected_offset=True（2026-10-20）：GT 采样时间走修正轴——与
    AnySoleDataset(corrected_offset=True) 同表同式（B 类平移 lag_eff×0.025，
    A/C/D 不动）。GT 的直接来源 = shared facts mocap_time_s + SMPL 归档
    （独立读取路径，非 dataset 内部 GT 缓存），因此此处必须显式应用同一
    修正，否则 GT 与载体窗内 pose 轴不一致。floor_y 取自 dataset session
    （载体 build 已按修正轴计算），随修正生效。
    """
    sid = sess["session_id"]
    shared = load_shared_session(sid, seq_root)
    n_frames = len(shared["frames"]["frame_id"])
    mocap_t = np.asarray(shared["frames"]["mocap_time_s"], dtype=np.float64)[:n_frames]
    if corrected_offset:
        from anysole.data.dataset import load_offset_correction
        mocap_t = mocap_t - load_offset_correction().get(sid, 0.0)
    floor = float(sess["floor_y"])
    smpl = load_smpl(Path(sess["smpl_path"]), query_t=mocap_t)
    joints = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"], smpl["offsets_m"], smpl["parents"])
    verts = smpl_vertices_from_archive_params(
        smpl["poses_aa"], smpl["model_trans_m"], smpl["betas"])
    # archive 覆盖可能短于 facts 帧轴（末会话尾部）：GT 数组统一补到 n_frames，
    # 尾部帧置无效（掩码 False、目标 NaN），保证窗切片恒为 tw。
    n_gt = len(smpl["pose_6d"])
    n_use = min(n_frames, n_gt)
    joints = joints[:n_use]

    out = {"floor": floor, "contact": {}, "phase": {}, "phase_mask": {}, "cop": {}, "cop_mask": {}}
    for i_side, side in enumerate(("L", "R")):
        foot_j, ankle_j = (FOOT_L, ANKLE_L) if side == "L" else (FOOT_R, ANKLE_R)
        # 触地事件：GT 足关节世界高 < floor + 2cm（D8 顶点口径同阈值，代理 =
        # 足关节）。不用 f6_soft 磁盘标签——2026-10-19 审计：标签为二值且
        # 63/140 会话全 1 退化（方案文档软值口径不符），运动学 GT 更干净。
        contact_use = joints[:, foot_j, 1] < floor + GT_CONTACT_DZ
        idx = np.where(np.diff(contact_use.astype(np.int8)) == 1)[0] + 1   # 上升沿 = 触地
        contact = np.zeros(n_frames, dtype=bool)
        contact[:n_use] = contact_use
        out["contact"][side] = contact
        n = n_frames
        phase = np.full((n, 2), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        if len(idx) >= 2:
            lo, hi = idx[0], idx[-1]
            for k, td in enumerate(idx[:-1]):
                td_next = idx[k + 1]
                span = max(td_next - td, 1)
                if span < 10:          # 周期过短（噪声触地）不计数
                    continue
                seg = np.arange(td, min(td_next, n_use))
                phi = (seg - td) / span
                phase[seg, 0] = np.cos(2 * np.pi * phi)
                phase[seg, 1] = np.sin(2 * np.pi * phi)
                mask[seg] = True
            mask[:lo] = False
            mask[hi:] = False
        out["phase"][side] = phase
        out["phase_mask"][side] = mask

        # GT CoP：foot_heat → cop_from_grid；接触帧且和值>0 有效
        cop = np.full((n, 2), np.nan, dtype=np.float32)
        cop_mask = np.zeros(n, dtype=bool)
        for i in range(n_use):
            heat = foot_heat(verts[i], joints[i, ankle_j], joints[i, foot_j], floor, side)
            if heat.sum() > 0 and contact[i]:
                cop[i] = cop_from_grid(heat.flatten())
                cop_mask[i] = True
        out["cop"][side] = cop
        out["cop_mask"][side] = cop_mask
    return out


def ridge_fit(X_tr, y_tr, lam):
    d = X_tr.shape[1]
    A = X_tr.T @ X_tr + lam * np.eye(d)
    b = X_tr.T @ y_tr
    return np.linalg.solve(A, b)


def ridge_loss(X, y, W, valid=None):
    """masked MSE：先取有效帧再算误差（NaN 目标 × 0 = NaN，不能靠掩码乘法）。"""
    if valid is None:
        return float(((X @ W - y) ** 2).mean())
    v = np.asarray(valid, dtype=bool)
    return float(((X[v] @ W - y[v]) ** 2).mean())


def bootstrap_paired(a, b, n_boot=10000, seed=0):
    """a, b 为逐会话配对数；H0 均值差 = 0；返回 (mean_diff, lo, hi)。"""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    mean = float(diffs.mean())
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    dist = diffs[idx].mean(axis=1)
    return mean, float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def collect(model, dataset, device, batch_size, seq_root, max_sessions=None,
            corrected_offset: bool = False):
    """逐帧特征 + GT（窗口级收集，GT 按 valid_windows 帧范围裁剪对齐）。

    返回 (feats, gt, mask, sess_seq, sess_counts)：会话按首次出现顺序
    （= loader 窗口顺序）登记，逐会话帧数供会话级操作对齐。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    feats = {r: {c: [] for c in CHANNELS} for r in READOUTS}
    gt = {c: [] for c in CHANNELS}
    mask = {c: [] for c in CHANNELS}
    sess_seq, sess_counts, seen, sess_err = [], {}, set(), set()
    sess_gt_cache = {}
    wcount = 0          # 全局窗口序号（批大小可变，不能用 k*B+i）
    with torch.inference_mode():
        for k, raw in enumerate(loader):
            batch = {kk: v.to(device) if torch.is_tensor(v) else v
                     for kk, v in raw.items()}
            B, tw = batch["pose_gt"].shape[:2]
            t_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
            config_id = torch.zeros(B, dtype=torch.long, device=device)
            v_tok, t_tok = model.encoders(batch["V_feat"], t_tac, config_id,
                                          V_hmr=batch.get("V_hmr"))
            raw108 = t_tac.cpu().numpy().reshape(-1, 108)
            v_tok_n = v_tok.cpu().numpy().reshape(-1, v_tok.shape[-1])
            vt_n = np.concatenate([v_tok_n, t_tok.cpu().numpy().reshape(-1, t_tok.shape[-1])], axis=-1)
            for i in range(B):
                sid = raw["session_id"][i]
                if max_sessions is not None:
                    if sid not in seen and len(seen) >= max_sessions:
                        continue
                    seen.add(sid)
                wi = wcount
                wcount += 1
                fi, left, right = dataset.valid_windows[wi]
                sess = dataset.sessions[fi]
                if sid not in sess_gt_cache:
                    try:
                        sess_gt_cache[sid] = session_gt(sess, seq_root, corrected_offset)
                        sess_seq.append(sid)
                        sess_counts[sid] = 0
                    except (FileNotFoundError, ValueError) as exc:
                        print("GT 不可用 %s: %s" % (sid, exc))
                        sess_err.add(sid)
                        sess_gt_cache[sid] = None
                        sess_seq.append(sid)
                        sess_counts[sid] = 0
                g = sess_gt_cache[sid]
                if g is None:
                    continue
                sl = slice(i * tw, (i + 1) * tw)
                sess_counts[sid] += tw
                for c in CHANNELS:
                    side = c.split("_")[1]
                    if c.startswith("phase"):
                        y = g["phase"][side][left:right]
                        m = g["phase_mask"][side][left:right]
                    else:
                        y = g["cop"][side][left:right]
                        m = g["cop_mask"][side][left:right]
                    gt[c].append(y)
                    mask[c].append(m)
                    feats["raw108"][c].append(raw108[sl])
                    feats["v_tok"][c].append(v_tok_n[sl])
                    feats["vt"][c].append(vt_n[sl])
    return feats, gt, mask, sess_seq, sess_counts


def run_probe(model, dataset_tr, dataset_te, device, batch_size, seq_root, max_sessions,
              corrected_offset: bool = False):
    F_tr, Y_tr, M_tr, sids_tr, cnt_tr = collect(model, dataset_tr, device, batch_size, seq_root, max_sessions, corrected_offset)
    F_te, Y_te, M_te, sids_te, cnt_te = collect(model, dataset_te, device, batch_size, seq_root, max_sessions, corrected_offset)

    def join(bag):
        return {c: np.concatenate(bag[c], axis=0).astype(np.float64) for c in CHANNELS}

    results, lam_chosen, sess_ci = {}, {}, {}
    for c in CHANNELS:
        for r in READOUTS:
            X = join(F_tr[r])[c]
            y = join(Y_tr)[c]
            m = join(M_tr)[c].astype(bool)
            X_te = join(F_te[r])[c]
            y_te = join(Y_te)[c]
            m_te = join(M_te)[c].astype(bool)
            n_tr = int(m.sum())
            n_te = int(m_te.sum())
            if n_tr < 50 or n_te < 5:
                results[(c, r)] = {"status": "no_data",
                                   "n_valid_train": n_tr, "n_valid_test": n_te}
                sess_ci[(c, r)] = {"per_session_mse_model": [], "per_session_mse_const": []}
                continue
            # 特征标准化（TRAIN 全局）+ ridge 只在有效帧上拟合
            mu, sd = X.mean(0), X.std(0) + 1e-8
            Xs = (X - mu) / sd
            ymu = y[m].mean(0)
            # 目标居中：ridge 无截距项，拟合 y − ymu、预测 + ymu
            # 选参：TRAIN 会话级 80/20 连续切分（前 80% 会话帧拟合、后 20% 验证）
            tr_len = int(sum(cnt_tr[s] for s in sids_tr[: int(len(sids_tr) * 0.8)]))
            tr_valid = m[:tr_len]
            va_valid = m[tr_len:]
            if not va_valid.any():
                results[(c, r)] = {"status": "no_data", "n_valid_train": n_tr,
                                   "n_valid_test": n_te, "reason": "empty_val"}
                sess_ci[(c, r)] = {"per_session_mse_model": [], "per_session_mse_const": []}
                continue
            best_lam, best_val = None, np.inf
            yc, yc_te = y - ymu, y_te - ymu
            for lam in LAM_GRID:
                W = ridge_fit(Xs[:tr_len][tr_valid], yc[:tr_len][tr_valid], lam)
                loss = ridge_loss(Xs[tr_len:], yc[tr_len:], W, va_valid)
                if loss < best_val:
                    best_val, best_lam = loss, lam
            lam_chosen[(c, r)] = best_lam
            W = ridge_fit(Xs[m], yc[m], best_lam)
            Xs_te = (X_te - mu) / sd
            err_te = ridge_loss(Xs_te, yc_te, W, m_te)
            err_base = float(((y_te[m_te] - ymu) ** 2).mean())
            results[(c, r)] = {"status": "ok", "lam": float(best_lam),
                               "mse_test": float(err_te), "mse_const": err_base,
                               "n_valid_train": n_tr, "n_valid_test": n_te}
            # 逐会话 TEST 误差（配对 CI 用；会话帧按 cnt_te 顺序切片）
            mod_li, con_li = [], []
            off = 0
            for sid in sids_te:
                nf = cnt_te[sid]
                mm = m_te[off:off + nf]
                if not mm.any():
                    off += nf
                    continue
                err_mod = ridge_loss(Xs_te[off:off + nf], yc_te[off:off + nf], W, mm)
                err_con = float(((yc_te[off:off + nf][mm]) ** 2).mean())
                mod_li.append(float(err_mod))
                con_li.append(err_con)
                off += nf
            sess_ci[(c, r)] = {"per_session_mse_model": mod_li, "per_session_mse_const": con_li}
    return results, lam_chosen, sess_ci


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "results/AnySole/V3_3E_f6_soft/"
                            "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                            "checkpoints/ckpt_best.pt")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tactile-input", default=None,
                    help="覆盖载体 saved config 的 tactile_input；None = 维持现行为 "
                         "（审计 A/B：raw108 / denoised108 / denoised108_v2 / denoised108_v3）")
    ap.add_argument("--corrected-offset", action="store_true", default=False,
                    help="GT 采样时间走修正轴（B 类平移 lag_eff×0.025，A/C/D 不动）："
                         "AnySoleDataset 构造与 session_gt 的 phase/CoP 目标同轴应用")
    args = ap.parse_args(argv)
    device = resolve_device(args.device)
    smoke = args.max_sessions is not None

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = load_config(REPO / "anysole" / "configs" / "v1.yaml")
    saved = checkpoint.get("config", {})
    tw = int(saved.get("tw", config["tw"]))
    model = _load_model(checkpoint, config, device).eval()
    seq_root = Path(config["seq_root"])

    def build(mode, column):
        return AnySoleDataset(
            mode=mode, seq_root=seq_root, split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]), window_length=tw,
            session_ids=(load_split_ids(Path(config["split_csv"]), column)
                         if mode == "eval" else None),
            contact_method=str(saved.get("contact_method", config.get("contact_method", "joint_and"))),
            tactile_input=args.tactile_input or str(saved.get("tactile_input", "raw108")),
            no_imu=bool(saved.get("no_imu", False)),
            v_input=str(saved.get("v_input", "hrnet")),
            f2_repr=bool(saved.get("f2_repr", False)),
            corrected_offset=bool(args.corrected_offset),
        )

    ds_tr = build("train", "train")
    ds_te = build("eval", "val")
    results, lam_chosen, sess_ci = run_probe(model, ds_tr, ds_te, device,
                                             args.batch_size, seq_root,
                                             args.max_sessions,
                                             bool(args.corrected_offset))
    verdict = {}
    for c in CHANNELS:
        for r in READOUTS:
            mse_m = sess_ci[(c, r)]["per_session_mse_model"]
            mse_c = sess_ci[(c, r)]["per_session_mse_const"]
            if len(mse_m) >= 3:
                verdict[(c, r, "vs_const")] = bootstrap_paired(mse_c, mse_m)
            else:
                verdict[(c, r, "vs_const")] = None
    for c in CHANNELS:
        a = sess_ci[(c, "v_tok")]["per_session_mse_model"]
        b = sess_ci[(c, "raw108")]["per_session_mse_model"]
        verdict[(c, "v_tok", "vs_raw108")] = bootstrap_paired(b, a) if len(a) >= 3 else None

    print("== 通道 × 读出（test，MSE；const = 常数基线；no_data = 有效帧不足）")
    for c in CHANNELS:
        for r in READOUTS:
            e = results[(c, r)]
            if e.get("status") != "ok":
                print("  %-8s %-7s no_data (n_train=%d n_test=%d)"
                      % (c, r, e.get("n_valid_train", 0), e.get("n_valid_test", 0)))
                continue
            print("  %-8s %-7s mse=%8.4f const=%8.4f lam=%.1e n_tr=%d n_te=%d"
                  % (c, r, e["mse_test"], e["mse_const"], e["lam"],
                     e["n_valid_train"], e["n_valid_test"]))
    print("== 判据（配对 bootstrap 95% CI；const−raw108 显著>0 = T 可读；raw108−v_tok 显著>0 = V 更强）")
    for c in CHANNELS:
        d1 = verdict[(c, "raw108", "vs_const")]
        d2 = verdict[(c, "vt", "vs_const")]
        d3 = verdict[(c, "v_tok", "vs_raw108")]
        s1 = "%+8.4f [%+8.4f,%+8.4f]" % d1 if d1 else "      n/a          "
        s2 = "%+8.4f" % d2[0] if d2 else "   n/a"
        s3 = "%+8.4f [%+8.4f,%+8.4f]" % d3 if d3 else "      n/a          "
        print("  %-8s T可读 %s | [v;t] %s | V更强 %s" % (c, s1, s2, s3))
    if not smoke:
        out = args.out or (REPO / "results/AnySole/probes/probe_v5_phase_cop_readout.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"provenance": {"ckpt": str(args.ckpt), "date": "2026-10-19",
                                  "tactile_input": args.tactile_input,
                                  "corrected_offset": bool(args.corrected_offset)},
                   "results": {f"{c}|{r}": results[(c, r)] for c in CHANNELS for r in READOUTS},
                   "verdict": {f"{c}|{r}|{k}": v for (c, r, k), v in verdict.items()},
                   "n_test_sessions": len(ds_te.session_ids)}
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print("wrote", out)


if __name__ == "__main__":
    main()
