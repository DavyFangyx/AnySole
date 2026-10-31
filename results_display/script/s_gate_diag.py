#!/usr/bin/env python3
"""TLR train 门值 + 逐组 D 诊断（V7_C门控修改方案 §7 硬决策点，2026-10-31）。

S 启动前的硬门槛数据：参照 TLR（V7_Cv5TLR_s1，ckpt_last ep740，权重
固定），在 train split 的原训练窗口网格上做「正常门控 + 四组分别关门」
（同窗内整组关门，其余门保持参照值）：
  - 根速度组：L_vel,t = mean_xyz((v_hat_t − vel_gt_t)^2)，单位 (m/s)^2；
    逐帧计算，不做时间均值、不经 cumsum。
  - 左/右下肢组：同侧腿足关节（左 1,4,7,10 / 右 2,5,8,11）骨盆对齐
    FK 位置误差（mm；骨盆 = joint 0 世界位置减去）。
  - 其余姿态组：全身 24 关节骨盆对齐 FK 位置误差（含骨盆旋转对下游
    关节的影响）。
逐帧 D_t,k = L_off,t − L_on,t（关门 − 正常）。
数值噪声：train 窗口子样本重复前向一次，逐组登记 n_k（|L1−L2| P95）。
预注册阈值（test 前登记，不按结果回改）：
  ε_k = max(3 × n_k, 1e-4 (m/s)^2 / 0.1 mm)——按组单位；
  绝对效应：|D|>ε_k 帧占比 ≥ 2% 且 ≥ 60% 会话出现 |D|>ε_k 帧；
  选择信息：D 跨帧 std > 3 × n_k，且逐窗 D std > ε_k 的窗口占比 ≥ 30%、
  出现此类窗口的会话占比 ≥ 60%；
  标签分布：τ_k = max(std_train(D_k), ε_k) 固定后 |y−0.5|>0.05 占比 ≥ 5%。
落盘（results_display/Etest/S_GateDiag/）：
  tlr_gate_diag.npz（逐窗逐帧门值/L_on/L_off/D/session/frame 索引/
  有效掩码）+ tlr_gate_diag.json（汇总 + 四行硬门槛逐组通过记录）。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/s_gate_diag.py --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_note/HMR2.0/hmr2_gait"))

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.data.contact_package import (  # noqa: E402
    VERSIONED_CONTACT_METHODS, corrected_contact_provenance)
from anysole.registry import get_builder, infer_anysole_paths  # noqa: E402
from configs.tools.common import effective_variant, load_registry  # noqa: E402
from anysole.train import load_config  # noqa: E402
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole_tta.phase_common import load_contact  # noqa: E402

OUT = REPO / "results_display/Etest/S_GateDiag"
MODEL = "V7_Cv5TLR"           # 参照（ckpt_last；--model 可换 V7_Cv7 重参照）
GROUP_NAMES = ("vel", "other", "left", "right")
GROUP_QUERIES = ((0,), tuple(range(1, 6)), (6, 8), (7, 9))
LEFT_JOINTS = (1, 4, 7, 10)
RIGHT_JOINTS = (2, 5, 8, 11)
# 预注册阈值（§7 硬门槛，见 docstring）
MIN_EFFECT_RATIO = 0.02
MIN_SESSION_COVERAGE = 0.60
MIN_WITHIN_WINDOWS = 0.30    # 窗内 D std > ε_k 的窗口占比下限
MIN_WITHIN_SESSIONS = 0.60    # 出现此类窗口的会话占比下限
MIN_LABEL_RATIO = 0.05
NOISE_MULT = 3.0
EPS_POS = 0.1                # FK 组最小效应下限（mm）
EPS_VEL = 1e-4               # 根速度组最小效应下限（(m/s)^2）


def build_model(device: torch.device, model_name: str):
    entry = get_builder(model_name)
    cfg = {"d_model": 256, "tw": 20, "dropout": 0.1, "pose_layers": 6,
           **entry.STRUCTURE, **entry.CONFIG_EXTRA}
    net = entry.build(cfg).to(device).eval()
    spec = load_registry()["models"]["%s_s1" % model_name]
    variant = effective_variant(spec)
    contact = "f6_soft_time_accel_state_smpl24_v1"
    ckpt_path = infer_anysole_paths(
        model_name, variant=variant,
        contact=contact, which="last")["ckpt"]
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict"))
    if state is None:
        raise RuntimeError("checkpoint without model state: %s" % ckpt_path)
    net.load_state_dict(state)
    return net, ckpt_path


def group_metrics(out, batch, device):
    """一次前向的逐组逐帧 L：(B, tw, 4)。vel=(m/s)^2，其余=mm。"""
    b = out["v_hat"].shape[0]
    l_vel = (out["v_hat"] - batch["vel_gt"].to(device)).square().mean(-1)
    fk = fk_pose6d(out["x0_hat"],
                   out["trans_hat"] + batch["trans_anchor"].to(device).view(b, 1, 3),
                   batch["offsets"].to(device), batch["parents"].to(device))
    fk = fk - fk[..., 0:1, :]                       # 骨盆对齐
    err = fk.square().sum(-1).sqrt() * 1000.0       # (B, tw, 24) m → mm
    return torch.stack([l_vel, err.mean(-1),
                        err[..., LEFT_JOINTS].mean(-1),
                        err[..., RIGHT_JOINTS].mean(-1)], dim=-1)  # (B, tw, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model", default=MODEL,
                        help="参照模型（默认 V7_Cv5TLR；V7_Cv7 = 无门重参照）")
    args = parser.parse_args()
    model_name = args.model
    device = torch.device(args.device)

    config = load_config(REPO / "anysole/configs/v1.yaml")
    config["contact_method"] = "f6_soft_time_accel_state_smpl24_v1"
    if config["contact_method"] in VERSIONED_CONTACT_METHODS:
        config.update(corrected_contact_provenance(config["contact_method"]))
    dataset = AnySoleDataset(
        mode="train",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=20,
        contact_method=str(config["contact_method"]),
        tactile_input="raw108", no_imu=False, v_input="hrnet",
        f2_repr=False, smpl_roots=None, cam_id=3,
        load_v5_1_gt=False, occ_mix=None, t_phys_v2=False, d8_heads=False,
        smpl_support_policy="exclude")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4,
                        collate_fn=collate_windows)
    model, ckpt_path = build_model(device, model_name)

    label_valid = {}
    sessions_seen = set()
    gates, l_on, l_off_k = [], [], [[] for _ in range(4)]
    valid = []
    metas = []  # (session_id, frame_start)
    with torch.no_grad():
        for batch in loader:
            cid = torch.zeros(batch["V_feat"].shape[0], dtype=torch.long,
                              device=device)
            v_feat = batch["V_feat"].to(device)
            t_raw = batch["T_raw"].to(device)
            t_phys = batch["T_phys"].to(device)
            out = model(v_feat, t_raw, t_phys, cid)
            g = out["route_gate"]                       # (B, tw, 10, 1)
            if g is None:
                g = torch.ones(v_feat.shape[0], 20, 10, 1, device=device)
            g4 = g[..., [0, 1, 6, 7], 0]                # vel/other/left/right
            gates.append(g4.cpu())
            l_on.append(group_metrics(out, batch, device).cpu())
            # 干预基线：TLR = 参照门值（其余组保持）；Cv7 无门 = 全 1
            #（该组 T 支路输出置零，标签 = 零门控偏置的裸 T 贡献，§7）。
            base = g if model_name != "V7_Cv7" else torch.ones_like(g)
            for k in range(4):
                override = base.clone()
                override[..., GROUP_QUERIES[k], :] = 0.0
                z = model(v_feat, t_raw, t_phys, cid, gate_override=override)
                l_off_k[k].append(
                    group_metrics(z, batch, device).cpu()[..., k].unsqueeze(-1))
            per_batch_valid = []
            for sid, start in zip(batch["session_id"],
                                  batch["frame_start"].tolist()):
                if sid not in label_valid:
                    _, _, lv = load_contact(sid)
                    label_valid[sid] = np.asarray(lv)
                idx = np.arange(start, start + 20)
                arr = label_valid[sid]
                per_batch_valid.append(arr[idx] if len(arr) > idx[-1]
                                       else np.zeros(20, dtype=bool))
            valid.append(np.stack(per_batch_valid))
            metas.extend(zip(batch["session_id"],
                             batch["frame_start"].tolist()))
            sessions_seen.update(batch["session_id"])
            print("batch %d/%d windows=%d" % (len(gates), len(loader),
                                              len(metas)), flush=True)

    gates = torch.cat(gates).numpy()                    # (N, tw, 4)
    l_on = torch.cat(l_on).numpy()                      # (N, tw, 4)
    l_off = np.stack([torch.cat(l_off_k[k]).numpy()[..., 0]
                      for k in range(4)], axis=-1)      # (N, tw, 4)
    valid = np.concatenate(valid)                       # (N, tw)
    d = l_off - l_on                                    # (N, tw, 4)
    n_win = gates.shape[0]

    # ---- 数值噪声：重复相同前向（首个 batch 子样本，5120 帧/组）----
    with torch.no_grad():
        for batch in loader:
            cid = torch.zeros(batch["V_feat"].shape[0], dtype=torch.long,
                              device=device)
            out2 = model(batch["V_feat"].to(device), batch["T_raw"].to(device),
                         batch["T_phys"].to(device), cid)
            l2 = group_metrics(out2, batch, device).cpu().numpy()
            break
    l1 = l_on[:l2.shape[0]]
    n_k = [float(np.percentile(np.abs(l2[:, :, k] - l1[:, :, k]), 95))
           for k in range(4)]
    eps = [max(NOISE_MULT * n_k[k], EPS_VEL if k == 0 else EPS_POS)
           for k in range(4)]

    # ---- §7 四行硬门槛逐组判定 ----
    checks = {}
    for k, name in enumerate(GROUP_NAMES):
        dk = d[:, :, k][valid]
        effect = np.abs(dk) > eps[k]
        sessions_effect = set()
        for i in range(n_win):
            f = valid[i]
            if f.any() and (np.abs(d[i, :, k][f]) > eps[k]).any():
                sessions_effect.add(metas[i][0])
        d_std = float(np.std(dk))
        # 选择信息：真正的逐窗 D std（窗内变化），不是跨窗拼接
        win_flags, win_by_session = [], {}
        for i in range(n_win):
            f = valid[i]
            if not f.any():
                continue
            flag = float(np.std(d[i, :, k][f])) > eps[k]
            win_flags.append(flag)
            win_by_session.setdefault(metas[i][0], []).append(flag)
        windows_ratio = float(np.mean(win_flags)) if win_flags else 0.0
        sessions_ratio = float(np.mean([any(v) for v in win_by_session.values()])) \
            if win_by_session else 0.0
        within_ratio = sessions_ratio
        tau = max(float(np.std(dk)), eps[k])
        y = 1.0 / (1.0 + np.exp(-dk / tau))
        label_ratio = float(np.mean(np.abs(y - 0.5) > 0.05))
        finite = bool(np.isfinite(dk).all())
        row2 = (float(np.mean(effect)) >= MIN_EFFECT_RATIO
                and len(sessions_effect) / max(len(sessions_seen), 1)
                >= MIN_SESSION_COVERAGE)
        row3 = (d_std > NOISE_MULT * n_k[k]
                and windows_ratio >= MIN_WITHIN_WINDOWS
                and within_ratio >= MIN_WITHIN_SESSIONS)
        row4 = label_ratio >= MIN_LABEL_RATIO
        checks[name] = {
            "n_k": n_k[k], "eps_k": eps[k], "tau_k": tau,
            "d_mean": float(np.mean(dk)), "d_std": d_std,
            "d_q25_median_q75": [float(np.percentile(dk, 25)),
                                  float(np.median(dk)),
                                  float(np.percentile(dk, 75))],
            "pos_ratio": float(np.mean(dk > eps[k])),
            "neg_ratio": float(np.mean(dk < -eps[k])),
            "near_zero_ratio": float(np.mean(np.abs(dk) <= eps[k])),
            "effect_ratio": float(np.mean(effect)),
            "session_coverage": len(sessions_effect) / max(len(sessions_seen), 1),
            "within_window_ratio": windows_ratio,
            "within_session_ratio": within_ratio,
            "label_ratio": label_ratio,
            "finite": finite,
            "gate_mean": float(gates[:, :, k][valid].mean()),
            "gate_q": [float(np.percentile(gates[:, :, k][valid], q))
                       for q in (5, 25, 50, 75, 95)],
            "gate_within_std": float(np.std(
                [gates[i, :, k][valid[i]].std() for i in range(n_win)
                 if valid[i].any()])) if valid.any() else 0.0,
            "gates_pass": {
                "数据与数值": finite, "绝对效应": bool(row2),
                "选择信息": bool(row3), "标签分布": bool(row4)},
            "S_ok": bool(finite and row2 and row3 and row4),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / ("tlr_gate_diag.npz" if model_name == "V7_Cv5TLR"
               else "cv7_gate_diag.npz"),
        gates=gates.astype(np.float32), l_on=l_on.astype(np.float32),
        l_off=l_off.astype(np.float32), d=d.astype(np.float32),
        valid=valid, session=np.array([m[0] for m in metas]),
        frame_start=np.array([m[1] for m in metas], dtype=np.int64))
    (OUT / ("tlr_gate_diag.json" if model_name == "V7_Cv5TLR"
            else "cv7_gate_diag.json")).write_text(json.dumps({
        "reference_model": model_name,
        "reference_ckpt": str(ckpt_path),
        "n_windows": int(n_win), "n_sessions": len(sessions_seen),
        "noise_n_k": n_k, "eps_k": eps,
        "thresholds": {"MIN_EFFECT_RATIO": MIN_EFFECT_RATIO,
                       "MIN_SESSION_COVERAGE": MIN_SESSION_COVERAGE,
                       "MIN_WITHIN_WINDOWS": MIN_WITHIN_WINDOWS,
                       "MIN_WITHIN_SESSIONS": MIN_WITHIN_SESSIONS,
                       "MIN_LABEL_RATIO": MIN_LABEL_RATIO,
                       "NOISE_MULT": NOISE_MULT},
        "groups": checks}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote %s" % OUT)
    for name in GROUP_NAMES:
        c = checks[name]
        print("  %-5s n=%.3g ε=%.3g D中位%+.3g 效应%.2f%% 会话覆盖%.0f%% "
              "窗内(窗%%/会话%%)%.0f/%.0f 标签%.1f%% → %s"
              % (name, c["n_k"], c["eps_k"], c["d_q25_median_q75"][1],
                 100 * c["effect_ratio"], 100 * c["session_coverage"],
                 100 * c["within_window_ratio"],
                 100 * c["within_session_ratio"], 100 * c["label_ratio"],
                 "PASS" if c["S_ok"] else "FAIL " + str(c["gates_pass"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
