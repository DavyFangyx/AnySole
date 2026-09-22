"""ρ 网格 repr dump ↔ GT 对齐的读出工具（R_Test7 C2 / R_Test8 共用）。

repr dump 的帧序 = dataset 非重叠窗口序（与 anysole.rho_grid 生成时同一
dataset 构建口径），因此可以按 session 逐窗口对齐 GT（pose/contact/yaw）。

注意口径：ρ 网格目前只跑过 val/test（train 无 repr），ridge 只能
在落盘 split 上拟合/评测——任务书 C2/D1 的 train-fit 版本需先跑
`python -m anysole.rho_grid --split train`（若需要）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.train import move_batch
from anysole.utils.geometry import rot6d_to_rotmat

REPR_NAMES = ("F", "v_tok", "t_tok")


def repr_frames(repr_dir: Path, rV: int, rT: int, session_ids: List[str],
                seed: int = 0, names=REPR_NAMES) -> Dict[str, Dict[str, np.ndarray]]:
    """{sid: {name: (T, d)}}；缺产物报错。"""
    suffix = "" if seed == 0 else "_s%d" % seed
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sid in session_ids:
        path = repr_dir / ("%s_rhoV%d_rhoT%d%s_repr.npz" % (sid, rV, rT, suffix))
        if not path.is_file():
            raise SystemExit("repr 缺失：%s" % path)
        dump = np.load(path)
        out[sid] = {k: dump[k] for k in names if k in dump}
    return out


def gt_targets(dataset: AnySoleDataset, session_ids: List[str],
               device: str = "cpu") -> Dict[str, dict]:
    """{sid: {pose_gt (T,144), contact_gt (T,2), yaw_sincos (T,2),
    trans_gt (T,3), kp_gt (T,24,3), offsets (24,3), parents}}——按窗口序拼接。"""
    groups: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        sid = dataset[i]["session_id"]
        if sid in session_ids:
            groups.setdefault(sid, []).append(i)
    out: Dict[str, dict] = {}
    for sid, idxs in groups.items():
        raw = [dataset[i] for i in idxs]
        batch = move_batch(collate_windows(raw), torch.device(device))
        tw = batch["pose_gt"].shape[1]
        pose = batch["pose_gt"].reshape(-1, batch["pose_gt"].shape[-1])
        contact = batch["contact_gt"].reshape(-1, 2).float()
        # 根 yaw：6D root rotation → rotmat → atan2(+Z 轴在地面投影)
        root_6d = pose[:, :6]
        R = rot6d_to_rotmat(root_6d.reshape(-1, 1, 6))[:, 0]  # (T,3,3)
        fwd = R[:, :, 2]  # (T,3)
        yaw = torch.atan2(fwd[:, 0], fwd[:, 2])
        yaw_sincos = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        anchor = batch["trans_anchor"][:, None, :]
        out[sid] = {
            "pose_gt": pose.cpu().numpy(),
            "contact_gt": contact.cpu().numpy(),
            "yaw_sincos": yaw_sincos.cpu().numpy(),
            "gt_trans": (batch["trans_gt"] + anchor).reshape(-1, 3).cpu().numpy(),
            "kp_gt": batch["kp_gt"].reshape(-1, batch["kp_gt"].shape[-2],
                                            batch["kp_gt"].shape[-1]).cpu().numpy(),
            "offsets": batch["offsets"][0].cpu().numpy(),
            "parents": batch["parents"][0].cpu().numpy(),
            "n_windows": len(raw),
            "tw": tw,
        }
    return out


def align_stream(repr_by_sid: Dict[str, Dict[str, np.ndarray]],
                 targets: Dict[str, dict], stream: str) -> tuple:
    """对齐后返回 (X, Y_pose, Y_contact, Y_yaw, meta)。

    X = 每帧拼接（若流是每帧 k 个 token——如 F 的 2 token/帧——reshape 成 (T, k·d)，
    与 ridge_probe 的 512 维每帧口径一致）；帧数校验以 v_tok（每帧 1 token）为准。
    """
    xs, yp, yc, yy, metas = [], [], [], [], []
    for sid, frames in repr_by_sid.items():
        t = targets.get(sid)
        if t is None:
            raise SystemExit("GT 缺 session %s（dataset 与 repr 的 session 集不一致）" % sid)
        ref = frames.get("v_tok", frames[stream])
        n_frames = len(ref)
        if n_frames != len(t["pose_gt"]):
            raise SystemExit("session %s 帧数不一致：repr %d vs GT %d（window 口径变了？）"
                             % (sid, n_frames, len(t["pose_gt"])))
        x = frames[stream]
        if len(x) != n_frames:
            k = len(x) // n_frames
            if k * n_frames != len(x):
                raise SystemExit("session %s 流 %s 行数 %d 不是帧数 %d 的整数倍"
                                 % (sid, stream, len(x), n_frames))
            x = x.reshape(n_frames, k * x.shape[-1])
        xs.append(x)
        yp.append(t["pose_gt"])
        yc.append(t["contact_gt"])
        yy.append(t["yaw_sincos"])
        metas.append(t)
    return (np.concatenate(xs), np.concatenate(yp), np.concatenate(yc),
            np.concatenate(yy), metas)


def eval_pose_readout(pred, targets: Dict[str, dict], session_ids) -> dict:
    """解码姿态按帧 Procrustes + 部位聚合（mm）：overall/upper/lower/anklefoot/hands。"""
    import torch
    from anysole.utils.eval_protocol import (
        ANKLE_FOOT_JOINTS, HAND_JOINTS, LOWER_JOINTS, UPPER_JOINTS, _pa_align)
    from anysole.utils.geometry import fk_pose6d
    device = pred.device
    err = torch.zeros(24, device=device)
    pa = torch.zeros(24, device=device)
    n = 0
    idx = 0
    for sid in session_ids:
        t = targets.get(sid)
        if t is None:
            continue
        T = len(t["pose_gt"])
        pose = pred[idx: idx + T].reshape(1, T, -1)
        trans = torch.from_numpy(t["gt_trans"]).float().to(device)[None]
        kp = fk_pose6d(pose, trans,
                       torch.from_numpy(t["offsets"]).float().to(device)[None],
                       torch.from_numpy(t["parents"]).long().to(device)[None])[0]
        kp_gt = torch.from_numpy(t["kp_gt"]).float().to(device)
        e = torch.linalg.vector_norm(kp - kp_gt, dim=-1)      # (T,24)
        aligned = _pa_align(kp, kp_gt)
        p = torch.linalg.vector_norm(aligned - kp_gt, dim=-1)
        err += e.sum(dim=0)
        pa += p.sum(dim=0)
        n += T
        idx += T
    err /= n
    pa /= n
    out = {"overall": float(pa.mean().item()) * 1000.0}
    for agg, joints in (("upper", UPPER_JOINTS), ("lower", LOWER_JOINTS),
                        ("anklefoot", ANKLE_FOOT_JOINTS), ("hands", HAND_JOINTS)):
        flat = np.asarray(joints).ravel().tolist()
        out[agg] = float(pa[flat].mean().item()) * 1000.0
    return out


def stream_readout(frames_fit: Dict[str, Dict[str, np.ndarray]],
                   frames_eval: Dict[str, Dict[str, np.ndarray]],
                   targets: Dict[str, dict], stream: str,
                   ridge_lam: float, device) -> dict:
    """ridge（fit 会话上拟合 / eval 会话上评）解码 pose/contact/yaw。

    返回 {target: value}：pose_PA_{overall,upper,lower,anklefoot,hands}、
    contact_acc、yaw_MAE_deg。
    """
    import torch
    from utils.ridge_probe import ridge_apply, ridge_fit
    Xf, Ypf, Ycf, Yyf, _ = align_stream(frames_fit, targets, stream)
    Xe, Ype, Yce, Yye, _ = align_stream(frames_eval, targets, stream)
    Xf = torch.from_numpy(Xf).float().to(device)
    Xe = torch.from_numpy(Xe).float().to(device)
    eval_sids = list(frames_eval.keys())
    out: dict = {}
    for name, Yf, Ye, post in (
        ("pose", torch.from_numpy(Ypf).float().to(device),
         torch.from_numpy(Ype).float().to(device), "pose"),
        ("contact", torch.from_numpy(Ycf).float().to(device),
         torch.from_numpy(Yce).float().to(device), "contact"),
        ("yaw", torch.from_numpy(Yyf).float().to(device),
         torch.from_numpy(Yye).float().to(device), "yaw"),
    ):
        W = ridge_fit(Xf, Yf, ridge_lam, device)
        pred = ridge_apply(W, Xe, device)
        if post == "pose":
            for agg, v in eval_pose_readout(pred, targets, eval_sids).items():
                out["pose_PA_" + agg] = round(v, 2)
        elif post == "contact":
            out["contact_acc"] = round(
                ((pred > 0.5) == (Ye > 0.5)).float().mean().item(), 4)
        else:
            yaw_p = torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()
            yaw_g = torch.atan2(Ye[:, 0], Ye[:, 1]).cpu().numpy()
            d = yaw_p - yaw_g
            d = np.arctan2(np.sin(d), np.cos(d))
            out["yaw_MAE_deg"] = round(float(np.abs(d).mean() * 180.0 / np.pi), 2)
    return out
