#!/usr/bin/env python3
"""E0 复制捷径支配度分解（修改 03 步骤 3，2026-11-03）。

在 train 上对 V7_Cv5AR2M_s1 的训练反馈口径分解损失：
- teacher-forced 单步：L_total（模型输出、GT 反馈）vs L_copy（复制反馈帧
  作预测的同一损失），比值 1 − L_total/L_copy = 模型相对纯复制的净改进；
- 自由 rollout（长度 R ∈ {1, 5, 20}）：同一分解，反馈 = 模型自生成状态
  （窗口起点全零，与评估协议 ar2_session_chunks 同口径），复制臂 = 复制
  模型自身收到的反馈链（自持续）；两条曲线分叉点落图。
- 只读前向 + 复用 anysole.train.backward_v7_enum 的损失构造（mode_config
  的 t_probability 缩放、config_probs 加权），不反传；不写训练代码。

复制臂构造：out_copy 的三个运动槽位 = 反馈帧内容
  x0_hat[t] = prev_state[t, 1:145]、v_hat[t] = prev_state[t, 145:148]、
  dyaw_hat[t] = prev_state[t, :1]（零基线语义下 = 上一帧增量）、
  trans_hat = cumsum(v_copy)/FPS；观测/辅助头（vis_yaw/vis_gain/
  contact/route_gate 等）保持模型输出 → 对应损失项在两臂间相消，比值
  只度量运动槽位的捷径支配度。

口径：同窗口集 = 每 8 个有效窗口取 1（确定性），TF 与 rollout 两臂共用
（同帧同掩码）；损失逐窗口（B=1）口径——compute_losses 的 L_contact pos
权重按批统计，批组成会移动该单项；逐窗口口径下两臂接触项同构，分叉
比较无批语义污染（训练 minibatch bs=64 的接触项与逐窗口口径的差异为
接触单项量级，见 detail json 的 loss_semantics 登记）。帧有效性 =
smpl_support_policy=exclude（与评估协议同数据视图）。注意：TF 的帧 0
反馈 = batch prev_state（真实 GT 前驱 = 训练口径），rollout 的帧 0
反馈 = 全零（评估自由生成口径）——帧 0 的锚差异即两种口径的分叉来源
之一，如实保留。

产物（幂等）：
- results_display/Etest/E0_ReferenceFloors/Full_Table4_E0_TFShortcut.csv
- results_display/Etest/E0_ReferenceFloors/detail/e0_tf_shortcut.json
- results_display/Etest/E0_ReferenceFloors/detail/e0_tf_shortcut.png
用法（touch_gait env）：
  python results_display/script/e0_tf_shortcut.py --device cuda:5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.registry import get_builder  # noqa: E402
from anysole.train import condition_inputs, model_observation_kwargs, move_batch  # noqa: E402
from anysole.types import (  # noqa: E402
    CONFIG_T, CONFIG_V, CONFIG_VT, FPS, SPLIT_CSV)
from anysole.utils.losses import compute_losses, _wrap_angle  # noqa: E402

ETEST = REPO / "results_display/Etest"
OUT_DIR = ETEST / "E0_ReferenceFloors"
DETAIL = OUT_DIR / "detail"
CKPT_PATH = ("results/AnySole/V7_Cv5AR2M_f6_soft_time_accel_state_smpl24_v1/"
             "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs64_sd1/"
             "checkpoints/ckpt_last.pt")
CONTACT_METHOD = "f6_soft_time_accel_state_smpl24_v1"
# 补充格（2026-11-03）：V7_Cv5AR2 = 带反馈支路的 AR2（ar2_fb=True），
# TF 与自由 rollout 的分叉只在有反馈的模型上存在；AR2M（规格 03 指定）
# 为无反馈对照格（v7_c2ar2m：因果掩码、ar2_fb=False、prev_state 不消费）
# → 两曲线按构造重合，分叉 = 0 即其语义。
CKPT_ALT = ("results/AnySole/V7_Cv5AR2_f6_soft_time_accel_state_smpl24_v1/"
            "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs64_sd1/"
            "checkpoints/ckpt_last.pt")
ROLLOUT_LENS = (1, 5, 20)
SUBSAMPLE = 8                # 每 8 个有效窗口取 1（TF/rollout 同集）
MODES = (CONFIG_VT, CONFIG_V, CONFIG_T)
MODE_NAMES = {CONFIG_VT: "VT2M", CONFIG_V: "V2M", CONFIG_T: "T2M"}
LOSS_KEYS = ("L_pose", "L_traj", "L_dyaw", "L_cumyaw", "L_dyaw_td",
             "L_pose_vel", "L_visyaw", "L_yawcorr", "L_contact")
# 损失批内按帧切片的键（(1, tw, ...) 时序键）
TIME_KEYS = ("pose_gt", "dyaw_gt", "v_head_gt", "psi_gt", "vel_gt",
             "contact_gt", "prev_state")


def build_model(device, ckpt_rel):
    ckpt = torch.load(str(REPO / ckpt_rel), map_location="cpu")
    cfg = dict(ckpt["config"])
    for key in ("model_name", "epochs", "lr"):
        cfg.pop(key, None)
    part_joints = cfg.pop("part_joints", None)
    model = get_builder(ckpt["config"]["model_name"]).build(
        cfg, part_joints=part_joints).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def mode_weights(config):
    """backward_v7_enum 的 mode_config 构造（只读复用，不反传）。"""
    probs = [float(x) for x in config["config_probs"]]
    probs = [x / sum(probs) for x in probs]
    t_probability = probs[CONFIG_VT] + probs[CONFIG_T]
    mode_config = dict(config)
    for key in ("lambda_contact", "lambda_foot"):
        if key in mode_config:
            mode_config[key] = float(config[key]) / t_probability \
                if t_probability > 0 else 0.0
    return probs, mode_config


def _copy_out(out: dict, feedback: torch.Tensor) -> dict:
    """out_copy：三个运动槽位 = 反馈帧内容；其余保持模型输出。"""
    out_copy = dict(out)
    fb = feedback  # (B, tw, 148) [Δyaw, tilt+body, v_head]
    v_copy = fb[..., 145:148]
    out_copy["x0_hat"] = fb[..., 1:145]
    out_copy["v_hat"] = v_copy
    out_copy["trans_hat"] = torch.cumsum(v_copy, dim=1) / float(FPS)
    out_copy["dyaw_hat"] = fb[..., :1]
    return out_copy


def _slice_batch(batch: dict, R: int) -> dict:
    """batch → 前 R 帧的损失输入（时序键切片；其余原样）。"""
    out = {}
    for k, v in batch.items():
        if k in TIME_KEYS and torch.is_tensor(v):
            out[k] = v[:, :R]
        else:
            out[k] = v
    return out


def _slice_out(out: dict, R: int) -> dict:
    out_r = {}
    for k, v in out.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] >= R:
            out_r[k] = v[:, :R]
        else:
            out_r[k] = v
    return out_r


def tf_losses(model, batch, cid, mode_cfg):
    """teacher-forced 单步：GT 反馈前向 → 逐窗口两臂完整损失 dict + out。

    损失逐窗口（B=1）口径计算：compute_losses 的 L_contact pos 权重按
    批统计（训练按 minibatch bs=64 统计），批组成会移动该单项——逐窗口
    口径下 TF 与 rollout 两臂的接触项完全同构，分叉比较无批语义污染
    （与训练 minibatch 口径的差异为接触单项量级，在说明 md 登记）。"""
    with torch.inference_mode():
        out = model(batch["V_feat"], batch["T_raw"], batch["T_phys"], cid,
                    batch.get("session_id"), T_s2m=batch["T_s2m"],
                    prev_state=batch["prev_state"], causal=True,
                    **model_observation_kwargs(batch))
        B = batch["pose_gt"].shape[0]
        losses_total, losses_copy = [], []
        for i in range(B):
            b1 = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() >= 1
                      and v.shape[0] == B else v) for k, v in batch.items()}
            o1 = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() >= 1
                      and v.shape[0] == B else v) for k, v in out.items()}
            c1 = cid[i:i + 1]
            losses_total.append(compute_losses(o1, b1, c1, mode_cfg))
            out_copy = _copy_out(o1, batch["prev_state"][i:i + 1])
            losses_copy.append(compute_losses(out_copy, b1, c1, mode_cfg))
    return losses_total, losses_copy, out


def rollout_pass(model, batch, cid, dyaw_base):
    """自由 rollout 整窗串行（窗口起点全零 = 评估协议口径）→ 全窗 out
    (1,tw,*) 与反馈链 (1,tw,148)（复制臂素材）。"""
    tw = batch["pose_gt"].shape[1]
    ps = torch.zeros(1, tw, 148, device=batch["pose_gt"].device)
    keys = ("x0_hat", "v_hat", "dyaw_hat", "vis_yaw", "vis_gain",
            "contact_logits", "route_gate")
    stack = {k: [] for k in keys}
    chain = []
    with torch.inference_mode():
        for t in range(tw):
            chain.append(ps[0, t].clone())
            out_t = model(batch["V_feat"], batch["T_raw"], batch["T_phys"],
                          cid, batch.get("session_id"), T_s2m=batch["T_s2m"],
                          prev_state=ps, causal=True)
            for k in keys:
                v = out_t.get(k)
                stack[k].append(None if v is None else v[0, t].detach())
            d_total = (out_t["dyaw_hat"][0, t].detach() + ps[0, t, 0]
                       if dyaw_base == "history"
                       else out_t["dyaw_hat"][0, t].detach())
            rel = out_t["x0_hat"][0, t].detach()
            vh = out_t["v_hat"][0, t].detach()
            if t + 1 < tw:
                ps[0, t + 1] = torch.cat([d_total.reshape(1), rel, vh])
    out = {k: (torch.stack([x for x in v if x is not None]).unsqueeze(0)
               if any(x is not None for x in v) else None)
           for k, v in stack.items()}
    out["trans_hat"] = torch.cumsum(out["v_hat"], dim=1) / float(FPS)
    fb = torch.stack(chain).unsqueeze(0)            # (1, tw, 148)
    return out, fb


def _weight(weights, name, default):
    return float(weights.get(name, default))


def _prefix_loss_ar2_r1(out, batch, cid, weights):
    """R=1 前缀损失：训练损失中帧级可定义项（公式与
    _compute_losses_v7_ar2 逐项同源）；traj delta 项与 TD 项
    （l_dyaw_td/l_pose_vel）需 ≥2 帧 → 1 帧前缀排除并登记为 0。
    yaw_fix 关闭（本 ckpt 语义）：l_dyaw = mrad·MSE(dyaw_hat, dyaw_gt)。"""
    import torch.nn.functional as F
    pose_gt = batch["pose_gt"]
    t_present = (cid != CONFIG_V).to(pose_gt.device)
    v_rows = (cid != CONFIG_T).to(pose_gt.device)
    l_pose = F.mse_loss(out["x0_hat"], pose_gt)
    l_vel = F.mse_loss(out["v_hat"], batch["v_head_gt"])
    l_traj = _weight(weights, "traj_velocity_w", 1.0) * l_vel
    dyaw_hat = out["dyaw_hat"][..., 0]
    dyaw_gt = batch["dyaw_gt"]
    mrad = 1e3
    l_dyaw = mrad * F.mse_loss(dyaw_hat, dyaw_gt)
    l_cumyaw = mrad * F.mse_loss(torch.cumsum(dyaw_hat, dim=1),
                                 torch.cumsum(dyaw_gt, dim=1))
    l_dyaw_td = l_pose.new_zeros(())
    l_pose_vel = l_pose.new_zeros(())
    l_visyaw = l_pose.new_zeros(())
    l_yawcorr = l_pose.new_zeros(())
    l_world = l_pose.new_zeros(())
    if bool(v_rows.any()):
        psi_gt = batch["psi_gt"]
        vis_yaw = out["vis_yaw"][..., 0]
        err = _wrap_angle(vis_yaw - psi_gt)
        l_visyaw = err[v_rows].square().mean()
        gain = out["vis_gain"][..., 0]
        psi_pred = (batch["psi_base"].view(-1, 1)
                    + torch.cumsum(dyaw_hat, dim=1))
        corrected = psi_pred + gain * _wrap_angle(vis_yaw - psi_pred)
        l_yawcorr = _wrap_angle(corrected - psi_gt)[v_rows].square().mean()
    gt = batch["contact_gt"]
    pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
    pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
    air_w = _weight(weights, "lambda_air", 0.0)
    from anysole.utils.losses import _v5_1_contact_bce
    l_contact = _v5_1_contact_bce(out["contact_logits"], gt, t_present,
                                  air_w, pos_weight)
    total = (
        _weight(weights, "lambda_pose", 3.0) * l_pose
        + _weight(weights, "lambda_traj", 1.0) * l_traj
        + _weight(weights, "lambda_dyaw", 1.0) * l_dyaw
        + _weight(weights, "lambda_cumyaw", 1.0) * l_cumyaw
        + _weight(weights, "lambda_contact", 0.0) * l_contact
        + _weight(weights, "lambda_visyaw", 0.5) * l_visyaw
        + _weight(weights, "lambda_yawcorr", 0.5) * l_yawcorr
    )
    return {"L_pose": l_pose, "L_traj": l_traj, "L_dyaw": l_dyaw,
            "L_cumyaw": l_cumyaw, "L_dyaw_td": l_dyaw_td,
            "L_pose_vel": l_pose_vel, "L_visyaw": l_visyaw,
            "L_yawcorr": l_yawcorr, "L_contact": l_contact,
            "L_world": l_world, "loss": total}


def prefix_loss(out_r, batch_r, cid, mode_cfg, R):
    """R 帧前缀的同一损失：R=1 走帧级可定义项（无 delta/TD）；R≥2 走
    标准 compute_losses（traj delta 集合由函数按 R 内部筛选）。"""
    if R == 1:
        return _prefix_loss_ar2_r1(out_r, batch_r, cid, mode_cfg)
    return compute_losses(out_r, batch_r, cid, mode_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--ckpt", default=None,
                    help="ckpt 相对路径（缺省 = 规格 03 的 V7_Cv5AR2M_s1；"
                         "alt = V7_Cv5AR2_s1 带反馈补充格）")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.ckpt in (None, "alt"):
        ckpt_rel = CKPT_ALT if args.ckpt == "alt" else CKPT_PATH
    else:
        ckpt_rel = args.ckpt
    tag = ("ar2" if ckpt_rel == CKPT_ALT else "ar2m")
    print("[load] %s (tag=%s)" % (ckpt_rel, tag), flush=True)
    model, config = build_model(device, ckpt_rel)
    has_fb = bool(getattr(model.pose_head, "ar2_fb", False))
    probs, mode_cfg = mode_weights(config)
    dyaw_base = str(config.get("ar2_dyaw_base", "zero"))
    print("[cfg] probs=%s dyaw_base=%s obs_context=%d tw=%d ar2_fb=%s"
          % (probs, dyaw_base, config.get("obs_context", config["tw"]),
             config["tw"], has_fb), flush=True)

    ds = AnySoleDataset(
        "train", split_csv=SPLIT_CSV, cam_id=3, stride=20,
        contact_method=CONTACT_METHOD, tactile_input="raw108",
        v7_ar2=True, obs_context=int(config.get("obs_context", config["tw"])),
        smpl_support_policy=str(config.get("smpl_support_policy", "exclude")))
    idx_all = list(range(0, len(ds), SUBSAMPLE))
    subset = Subset(ds, idx_all)
    print("[data] windows=%d (subset %d)" % (len(ds), len(subset)), flush=True)

    acc = {"tf": {"total": 0.0, "copy": 0.0, "frames": 0.0},
           "tf_mode": {m: {"total": 0.0, "copy": 0.0, "frames": 0.0}
                       for m in MODE_NAMES.values()},
           "tf_comp": {k: {"total": 0.0, "copy": 0.0, "frames": 0.0}
                       for k in LOSS_KEYS},
           "tf_R": {R: {"total": 0.0, "copy": 0.0, "frames": 0.0}
                    for R in ROLLOUT_LENS},
           "ro": {R: {"total": 0.0, "copy": 0.0, "frames": 0.0}
                  for R in ROLLOUT_LENS}}

    def add(dst, l_total, l_copy, frames):
        dst["total"] += l_total * frames
        dst["copy"] += l_copy * frames
        dst["frames"] += frames

    # ---- teacher-forced 单步（批处理前向；损失逐窗口口径） ----
    loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate_windows)
    with torch.inference_mode():
        for bi, raw in enumerate(loader):
            batch = move_batch(raw, device)
            B, tw = batch["pose_gt"].shape[:2]
            for mode in MODES:
                cid = torch.full((B,), mode, dtype=torch.long, device=device)
                v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
                b = {**batch, "V_feat": v_feat, "T_raw": t_raw,
                     "T_phys": t_phys, "T_s2m": t_s2m}
                losses_total, losses_copy, out = tf_losses(
                    model, b, cid, mode_cfg)
                name = MODE_NAMES[mode]
                w = probs[mode]
                for i in range(B):
                    lt, lc = losses_total[i], losses_copy[i]
                    add(acc["tf"], w * lt["loss"].item(),
                        w * lc["loss"].item(), tw)
                    add(acc["tf_mode"][name], lt["loss"].item(),
                        lc["loss"].item(), tw)
                    for k in LOSS_KEYS:
                        add(acc["tf_comp"][k], lt[k].item(),
                            lc[k].item(), tw)
                    o1 = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() >= 1
                              and v.shape[0] == B else v) for k, v in out.items()}
                    b1 = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() >= 1
                              and v.shape[0] == B else v) for k, v in b.items()}
                    out_copy = _copy_out(o1, b["prev_state"][i:i + 1])
                    c1 = cid[i:i + 1]
                    for R in ROLLOUT_LENS:
                        batch_r = _slice_batch(b1, R)
                        l_t = prefix_loss(_slice_out(o1, R), batch_r, c1,
                                          mode_cfg, R)["loss"].item()
                        l_c = prefix_loss(_slice_out(out_copy, R), batch_r, c1,
                                          mode_cfg, R)["loss"].item()
                        add(acc["tf_R"][R], w * l_t, w * l_c, R)
            if bi % 50 == 0:
                print("[tf] batch %d/%d" % (bi, len(loader)), flush=True)

    # ---- 自由 rollout（同窗口集，逐窗串行；窗口起点全零） ----
    for i, di in enumerate(idx_all):
        item = ds[di]
        batch = move_batch(collate_windows([item]), device)
        B, tw = batch["pose_gt"].shape[:2]
        for mode in MODES:
            cid = torch.full((B,), mode, dtype=torch.long, device=device)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
            b = {**batch, "V_feat": v_feat, "T_raw": t_raw,
                 "T_phys": t_phys, "T_s2m": t_s2m}
            out, fb = rollout_pass(model, b, cid, dyaw_base)
            for R in ROLLOUT_LENS:
                batch_r = _slice_batch(b, R)
                l_total = prefix_loss(_slice_out(out, R), batch_r, cid,
                                      mode_cfg, R)["loss"].item()
                out_copy = _copy_out(_slice_out(out, R), fb[:, :R])
                l_copy = prefix_loss(out_copy, batch_r, cid,
                                     mode_cfg, R)["loss"].item()
                add(acc["ro"][R], probs[mode] * l_total,
                    probs[mode] * l_copy, B * R)
        if i % 200 == 0:
            print("[ro] window %d/%d" % (i, len(idx_all)), flush=True)

    # ---- 汇总与落盘 ----
    def ratio(total, copy):
        return 1.0 - total / max(copy, 1e-12)

    def mean(dst):
        return dst["total"] / max(dst["frames"], 1.0), \
            dst["copy"] / max(dst["frames"], 1.0)

    tf_total, tf_copy = mean(acc["tf"])
    out_json = {
        "checkpoint": ckpt_rel, "split": "train", "subsample": SUBSAMPLE,
        "ar2_fb": has_fb,
        "loss_semantics": "per-window (L_contact pos-weight per window; "
                          "training uses per-minibatch bs=64)",
        "n_windows": len(idx_all), "config_probs": probs,
        "dyaw_base": dyaw_base, "rollout_init": "zero (eval protocol)",
        "tf_frame0_feedback": "batch prev_state (GT predecessor)",
        "teacher_forced": {
            "L_total": tf_total, "L_copy": tf_copy,
            "ratio": ratio(tf_total, tf_copy), "frames": acc["tf"]["frames"],
            "prefix": {str(R): {
                "L_total": mean(v)[0], "L_copy": mean(v)[1],
                "ratio": ratio(*mean(v)), "frames": v["frames"]}
                for R, v in acc["tf_R"].items()},
            "modes": {m: {
                "L_total": mean(v)[0], "L_copy": mean(v)[1],
                "ratio": ratio(*mean(v))}
                for m, v in acc["tf_mode"].items()},
            "components": {k: {
                "L_total": mean(v)[0], "L_copy": mean(v)[1],
                "ratio": ratio(*mean(v))}
                for k, v in acc["tf_comp"].items()}},
        "rollout": {str(R): {
            "L_total": mean(v)[0], "L_copy": mean(v)[1],
            "ratio": ratio(*mean(v)), "frames": v["frames"]}
            for R, v in acc["ro"].items()},
    }
    DETAIL.mkdir(parents=True, exist_ok=True)
    out = DETAIL / ("e0_tf_shortcut_%s.json" % tag)
    out.write_text(json.dumps(out_json, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s" % out)
    print("[result] tf: L_total=%.6g L_copy=%.6g ratio=%.6g"
          % (tf_total, tf_copy, ratio(tf_total, tf_copy)))
    for R in ROLLOUT_LENS:
        v = out_json["rollout"][str(R)]
        print("[result] rollout R=%d: L_total=%.6g L_copy=%.6g ratio=%.6g"
              % (R, v["L_total"], v["L_copy"], v["ratio"]))

    # 主干表（E0 根层）：TF 全窗单步 + TF 前缀曲线 + rollout 曲线
    path = OUT_DIR / ("Full_Table4_E0_TFShortcut_%s.csv" % tag)
    rows = [["regime", "horizon", "L_total", "L_copy",
             "ratio_1_minus_Ltot_over_Lcopy", "frames"]]
    rows.append(["teacher_forced_full_window", tw, "%.8g" % tf_total,
                 "%.8g" % tf_copy, "%.6g" % ratio(tf_total, tf_copy),
                 int(acc["tf"]["frames"])])
    for R in ROLLOUT_LENS:
        v = out_json["teacher_forced"]["prefix"][str(R)]
        rows.append(["teacher_forced_prefix", R, "%.8g" % v["L_total"],
                     "%.8g" % v["L_copy"], "%.6g" % v["ratio"],
                     int(v["frames"])])
    for R in ROLLOUT_LENS:
        v = out_json["rollout"][str(R)]
        rows.append(["free_rollout", R, "%.8g" % v["L_total"],
                     "%.8g" % v["L_copy"], "%.6g" % v["ratio"],
                     int(v["frames"])])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("wrote %s" % path)

    # 分叉点图（detail/）：TF（前缀口径）与自由 rollout 两条曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ink, mut = "#1c1b19", "#6f6d66"
    xs = list(ROLLOUT_LENS)
    tf_curve = [out_json["teacher_forced"]["prefix"][str(R)]["ratio"]
                for R in ROLLOUT_LENS]
    ro_curve = [out_json["rollout"][str(R)]["ratio"] for R in ROLLOUT_LENS]
    ax.plot(xs, tf_curve, "-o", color="#2a78d6",
            label="teacher-forced (GT feedback)")
    ax.plot(xs, ro_curve, "-s", color="#eb6834",
            label="free rollout (self feedback)")
    ax.axhline(0.0, color=mut, lw=0.8, ls="--")
    ax.set_xlabel("horizon R (frames)")
    ax.set_ylabel("1 − L_total/L_copy (net gain over copy)")
    ax.set_xticks(xs)
    ax.set_title("V7_Cv5AR2M_s1 train: copy-shortcut share of the training objective",
                 fontsize=9, color=ink)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e4df", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = DETAIL / ("e0_tf_shortcut_%s.png" % tag)
    fig.savefig(png, dpi=160)
    print("wrote %s" % png)


if __name__ == "__main__":
    main()
