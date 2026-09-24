"""ρ 网格生成器（missing-rate 任务书实验 1 / B3 生成器侧）。

机制：V/T 保留率网格 (rV, rT) ∈ ρ×ρ，每格 = 逐帧独立掷硬币把该帧 token 换成
null token（保留率 = 不换的概率；mask=True 的位置换 null）。config 恒为 VT，
mask 叠在 config 级替换之后 → 四角（100/100、100/0、0/100、0/0）与现有
VT2M/V2M/T2M/纯先验严格同机制；ρ=100%/0% 角与 fseries 的对应配置行一致 =
实现自检项（grid_metrics.json 的 corner_check）。

每格 × 每种子 × 每 session 落盘（默认 results_display/BTest/B3Test_rho_grid/）：
  - grid_metrics.json：全格协议指标（与 fseries 同构；复用
    eval_protocol._session_metrics，逐字节同源）
  - npz/<session>_rhoV<rV>_rhoT<rT>[_s<seed>].npz：eval_motion 同格式 SMPL npz
  - repr/<session>_rhoV<rV>_rhoT<rT>[_s<seed>]_repr.npz：F/t_tok/v_tok

mask 按 (session, cell, seed) 确定性生成（crc32 混种子，同 eval 鲁棒行口径）。
仅支持 anysolev2（回归式）checkpoint——mask 只接入了 v2 forward。

Usage (touch_gait env, 仓库根执行):
  python -m anysole.rho_grid --ckpt results/AnySole/V4B_joint_and/checkpoints/ckpt_last.pt \
      --split test --seeds 0
  python -m anysole.rho_grid --ckpt ... --split val --seeds 0,1,2
  # 冒烟：1 个 session × 粗网格
  python -m anysole.rho_grid --ckpt ... --split val --seeds 0 \
      --rhos 0,50,100 --limit-sessions 1 --skip-repr
"""
from __future__ import annotations

import argparse
import fcntl
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

import numpy as np
import torch

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.data.smpl_io import (
    pelvis_to_smpl_trans,
    smpl24_pose6d_to_poses,
    smpl_archive_metadata,
)
from anysole.eval import _load_model
from anysole.models import MODEL_ANYSOLEV2
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT, FPS, N_JOINTS, POSE_DIM
from anysole.utils.eval_protocol import _Accum, _pick_forward_axis, _session_metrics
from anysole.utils.geometry import f2_to_world, rot6d_to_rotmat, rotmat_to_6d

# 窗口缝 crossfade（同 eval.py 口径；回归式 FADE=4）
FADE = 4


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ρ 网格生成器（B3 / 任务书实验 1）")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "anysole" / "configs" / "v1.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val",
                        help="train 只用于落 repr/npz 供 ridge train-fit（无 fseries 自检）")
    parser.add_argument("--seeds", type=str, default="0",
                        help="逗号分隔的 mask 种子（val 用 0,1,2 估方差；test 单种子 0）")
    parser.add_argument("--rhos", type=str, default="0,20,40,60,80,100",
                        help="保留率网格（百分比，升序）")
    parser.add_argument("--rho-v", type=int, default=None,
                        help="只运行一个 V 保留率配置；与 --rho-t 一起使用")
    parser.add_argument("--rho-t", type=int, default=None,
                        help="只运行一个 T 保留率配置；与 --rho-v 一起使用")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "results_display" / "BTest" / "B3Test_rho_grid",
                        help="产物目录（grid_metrics.json + npz/ + repr/）")
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-npz", action="store_true", help="不落运动 npz")
    parser.add_argument("--skip-repr", action="store_true", help="不落表征 dump")
    parser.add_argument("--reuse", action="store_true",
                        help="产物齐全的 cell 直接跳过（指标沿用旧 grid_metrics.json）")
    return parser.parse_args(argv)


def parse_int_csv(value: str) -> List[int]:
    out = [int(x) for x in value.split(",") if x.strip()]
    if out != sorted(out):
        raise ValueError("--rhos/--seeds 必须升序：%s" % value)
    return out


def model_root_of(ckpt: Path) -> Path:
    return ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent


def cell_key(rV: int, rT: int) -> str:
    return "rhoV%d_rhoT%d" % (rV, rT)


def seed_suffix(seed: int) -> str:
    return "" if seed == 0 else "_s%d" % seed


def metric_config_of(rV: int, rT: int) -> int:
    """各格协议指标的配置口径（与 fseries 行同构的关键）：
    T 全缺（rT=0, rV>0）→ CONFIG_V（含 V2T 压力行，同 fseries V2M）；
    V 全缺（rV=0, rT>0）→ CONFIG_T（无压力行，同 fseries T2M）；
    其余 → CONFIG_VT（纯先验角 (0,0) 也走此口径，无 fseries 对应行）。"""
    if rT == 0 and rV > 0:
        return CONFIG_V
    if rV == 0 and rT > 0:
        return CONFIG_T
    return CONFIG_VT


def make_masks(bsz: int, tw: int, rV: int, rT: int, session_id: str,
               cell_seed: int) -> tuple:
    """确定性逐帧 mask：True = 该帧换成 null token。"""
    h = cell_seed ^ zlib.crc32(str(session_id).encode())
    rng = np.random.RandomState(h)
    mask_v = torch.from_numpy(rng.random_sample((bsz, tw)) < (1.0 - rV / 100.0))
    mask_t = torch.from_numpy(rng.random_sample((bsz, tw)) < (1.0 - rT / 100.0))
    return mask_v, mask_t


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = parse_int_csv(args.seeds)
    rhos = parse_int_csv(args.rhos)
    if (args.rho_v is None) != (args.rho_t is None):
        raise SystemExit("--rho-v 和 --rho-t 必须同时提供")
    rho_vs = [args.rho_v] if args.rho_v is not None else rhos
    rho_ts = [args.rho_t] if args.rho_t is not None else rhos
    device = resolve_device(args.device)
    config = load_config(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % args.ckpt)
    saved = checkpoint.get("config", {})
    if str(saved.get("modal")) != MODEL_ANYSOLEV2:
        raise SystemExit("rho_grid 仅支持 anysolev2（回归式）checkpoint，got %r"
                         % saved.get("modal"))
    model = _load_model(checkpoint, config, device)
    model.eval()

    tw = int(saved.get("tw", config["tw"]))
    contact_method = str(saved.get("contact_method", config.get("contact_method", "joint_and")))
    no_imu = bool(saved.get("no_imu", False))
    v_hmr_mode = str(saved.get("v_input", "hrnet")) == "hmr_gvhmr"
    f2_repr = bool(saved.get("f2_repr", False))
    session_ids = load_split_ids(Path(config["split_csv"]), args.split)
    if args.limit_sessions is not None:
        session_ids = session_ids[: args.limit_sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=tw,
        session_ids=session_ids,
        contact_method=contact_method,
        tactile_input=str(saved.get("tactile_input", "raw108")),
        no_imu=no_imu,
        v_input=str(saved.get("v_input", "hrnet")),
        f2_repr=f2_repr,
        smpl_roots=saved.get("smpl_roots", config.get("smpl_roots")),
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset contains no valid windows")
    forward_axis = _pick_forward_axis(dataset, device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = None if args.skip_npz else out_dir / "npz"
    repr_dir = None if args.skip_repr else out_dir / "repr"
    if npz_dir is not None:
        npz_dir.mkdir(parents=True, exist_ok=True)
    if repr_dir is not None:
        repr_dir.mkdir(parents=True, exist_ok=True)

    # session -> 窗口索引（dataset 顺序，同 eval 口径）
    groups: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset[i]["session_id"], []).append(i)

    metrics_path = out_dir / ("grid_metrics_%s.json" % args.split)
    prev = {}
    if metrics_path.is_file():
        try:
            prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    cells: Dict[str, dict] = prev.get("cells", {}) if isinstance(prev, dict) else {}

    n_cells = len(rho_vs) * len(rho_ts) * len(seeds)
    done = 0
    for rV in rho_vs:
        for rT in rho_ts:
            ck = cell_key(rV, rT)
            for seed in seeds:
                cell_seed = (seed * 1000003) ^ zlib.crc32(ck.encode())
                seed_key = "s%d" % seed
                if (args.reuse
                        and cells.get(seed_key, {}).get(ck) is not None
                        and _cell_products_ok(npz_dir, repr_dir, groups,
                                              rV, rT, seed, args)):
                    done += 1
                    print("[%d/%d] reuse %s s%d" % (done, n_cells, ck, seed))
                    continue
                metrics = _run_cell_inference(
                    model, dataset, groups, device, tw, forward_axis, rV, rT,
                    seed, cell_seed, v_hmr_mode, f2_repr, npz_dir, repr_dir, args,
                )
                cells.setdefault(seed_key, {})[ck] = metrics
                done += 1
                print("[%d/%d] %s s%d pa_mpjpe=%.2f mpjpe=%.2f contact_f1=%.4f"
                      % (done, n_cells, ck, seed,
                         metrics.get("pa_mpjpe_mm", float("nan")),
                         metrics.get("mpjpe_mm", float("nan")),
                         metrics.get("contact_f1", float("nan"))))

    payload = {
        "checkpoint": str(args.ckpt),
        "modal": str(saved.get("modal")),
        "contact_method": contact_method,
        "split": args.split,
        "seeds": list(seeds),
        "rhos": sorted(set(rho_vs + rho_ts)),
        "tw": tw,
        "forward_axis": "+Z" if forward_axis == 2 else "+X",
        "cells": cells,
    }
    _locked_merge_write(metrics_path, payload, args,
                        canonical=(out_dir / "grid_metrics.json")
                        if args.split == "val" else None)
    print("wrote %s" % metrics_path)
    return 0


def _locked_merge_write(metrics_path: Path, payload: dict, args,
                        canonical: Optional[Path] = None) -> None:
    """加锁合并写 grid_metrics：并发 cell conf 各自 merge 自己的 cell，不互相覆盖。

    锁在 <metrics_path>.lock 上；写前重读盘上最新文件并合并 cells/seeds/rhos，
    再用合并后的 cells 重算 corner_check（保证自检反映最终网格而非本次运行）。
    """
    lock_path = metrics_path.with_suffix(metrics_path.suffix + ".lock")
    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        prev = {}
        if metrics_path.is_file():
            try:
                prev = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prev = {}
        prev = prev if isinstance(prev, dict) else {}
        prev_cells = prev.get("cells", {}) if isinstance(prev.get("cells"), dict) else {}
        merged_cells: Dict[str, dict] = {key: dict(value)
                                         for key, value in prev_cells.items()}
        for seed_key, cell_metrics in payload["cells"].items():
            merged_cells.setdefault(seed_key, {}).update(cell_metrics)
        merged = dict(payload)
        merged["cells"] = merged_cells
        merged["seeds"] = sorted(set(prev.get("seeds") or []) | set(payload.get("seeds") or []))
        merged["rhos"] = sorted(set(prev.get("rhos") or []) | set(payload.get("rhos") or []))
        merged["corner_check"] = _corner_check(merged_cells, merged["seeds"], args)
        metrics_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        if canonical is not None:
            canonical.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")


def _cell_products_ok(npz_dir, repr_dir, groups, rV, rT, seed, args) -> bool:
    if not args.reuse:
        return False
    for session_id in groups:
        if npz_dir is not None and \
                not (npz_dir / ("%s_%s%s.npz" % (session_id, cell_key(rV, rT),
                                                 seed_suffix(seed)))).is_file():
            return False
        if repr_dir is not None and \
                not (repr_dir / ("%s_%s%s_repr.npz" % (session_id, cell_key(rV, rT),
                                                       seed_suffix(seed)))).is_file():
            return False
    return True


def _corner_check(cells: dict, seeds: List[int], args) -> dict:
    """角格 vs 现有 fseries 对应配置行（自检：必须一致）。

    只在全会话口径下有意义（--limit-sessions 冒烟时 diff 是预期的）。"""
    fseries_path = model_root_of(args.ckpt) / "metrics" / ("%s_fseries.json" % args.split)
    if not fseries_path.is_file():
        return {"note": "fseries 缺失，跳过自检：%s" % fseries_path}
    fseries = json.loads(fseries_path.read_text(encoding="utf-8"))["metrics"]
    check = {}
    if args.limit_sessions is not None:
        check["note"] = "--limit-sessions=%d 冒烟口径：diff 非 0 属预期" % args.limit_sessions
    seed_cells = cells.get("s%d" % seeds[0], {})
    for rV, rT, cfg in ((100, 0, "V2M"), (0, 100, "T2M"), (100, 100, "VT2M")):
        row = {}
        for metric in ("pa_mpjpe_mm", "mpjpe_mm", "contact_f1", "yaw_abs_deg", "root_rte_percent"):
            if metric not in fseries[cfg]:
                continue
            grid_val = seed_cells.get(cell_key(rV, rT), {}).get(metric)
            if grid_val is None:
                continue
            row[metric] = {"grid": round(grid_val, 4),
                           "fseries": round(fseries[cfg][metric], 4),
                           "diff": round(grid_val - fseries[cfg][metric], 4)}
        check[cfg] = row
    return check


def _run_cell_inference(model, dataset, groups, device, tw, forward_axis,
                        rV, rT, seed, cell_seed, v_hmr_mode, f2_repr,
                        npz_dir, repr_dir, args) -> dict:
    """一个 (cell, seed) 的全会话推理：协议指标 + 每 session npz/repr。"""
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    metric_config = metric_config_of(rV, rT)
    beta_by_session = {
        s["session_id"]: np.asarray(s.get("betas", np.zeros(10)), dtype=np.float32)
        for s in dataset.sessions
    }
    with torch.inference_mode():
        for session_id, idxs in groups.items():
            raw = [dataset[i] for i in idxs]
            batch = move_batch(collate_windows(raw), device)
            bsz = batch["pose_gt"].shape[0]
            config_id = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            mask_v, mask_t = make_masks(bsz, tw, rV, rT, session_id, cell_seed)
            out = model(v_feat, t_raw, t_phys, config_id,
                        batch.get("session_id"), T_s2m=t_s2m,
                        V_hmr=batch.get("V_hmr") if v_hmr_mode else None,
                        mask_v=mask_v, mask_t=mask_t)
            pred_pose = out["x0_hat"]
            anchor = batch["trans_anchor"][:, None, :]
            if f2_repr:
                pred_pose_w, pred_trans_w = f2_to_world(
                    pred_pose, out["v_hat"], batch["psi_anchor"], batch["trans_anchor"]
                )
                gt_pose_w, _ = f2_to_world(
                    batch["pose_gt"], batch["traj_gt_f2"], batch["psi_anchor"],
                    batch["trans_anchor"],
                )
            else:
                pred_pose_w = pred_pose
                pred_trans_w = out["trans_hat"] + anchor
                gt_pose_w = batch["pose_gt"]
            gt_trans_w = batch["trans_gt"] + anchor
            seq = {
                "pred_pose": pred_pose_w.reshape(-1, pred_pose_w.shape[-1]),
                "pred_trans": pred_trans_w.reshape(-1, 3),
                "gt_pose": gt_pose_w.reshape(-1, gt_pose_w.shape[-1]),
                "gt_trans": gt_trans_w.reshape(-1, 3),
                "kp_gt": batch["kp_gt"].reshape(-1, N_JOINTS, 3),
                "contact_gt": batch["contact_gt"].reshape(-1, 2),
                "floor_y": batch["floor_y"][0],
                "pressure_gt": batch["T_raw"].reshape(-1, batch["T_raw"].shape[-1]),
                "pressure_pred": (out["pressure_hat"].reshape(-1, out["pressure_hat"].shape[-1])
                                  if out.get("pressure_hat") is not None else None),
                "offsets": batch["offsets"][0],
                "parents": batch["parents"][0],
                "betas": batch["betas"][0],
            }
            _session_metrics(seq, tw, forward_axis, metric_config, accum, contact)
            _write_session_artifacts(npz_dir, repr_dir, session_id, rV, rT,
                                     seed, batch, pred_pose_w, pred_trans_w,
                                     gt_trans_w, beta_by_session.get(session_id),
                                     out, args)

    result = accum.means()
    p = contact["tp"] / max(contact["tp"] + contact["fp"], 1)
    r = contact["tp"] / max(contact["tp"] + contact["fn"], 1)
    result["contact_f1"] = 2.0 * p * r / max(p + r, 1e-8)
    result["contact_acc"] = (contact["tp"] + contact["tn"]) / max(
        contact["tp"] + contact["fp"] + contact["fn"] + contact["tn"], 1
    )
    result["contact_recall"] = contact["tp"] / max(contact["tp"] + contact["fn"], 1)
    result["air_recall"] = contact["tn"] / max(contact["tn"] + contact["fp"], 1)
    result["contact_balanced_acc"] = 0.5 * (
        result["contact_recall"] + result["air_recall"]
    )
    if "accel_mag_upper_ms2" in result:
        result["accel_dist_err_upper_ms2"] = abs(
            result["accel_mag_upper_ms2"] - result["accel_mag_upper_gt_ms2"]
        )
    return result


def _write_session_artifacts(npz_dir: Optional[Path], repr_dir: Optional[Path],
                             session_id: str, rV: int, rT: int, seed: int,
                             batch: dict, pred_pose_w, pred_trans_w, gt_trans_w,
                             betas, out: dict, args) -> None:
    """每 session：聚合窗口 → crossfade → 写 eval_motion 同格式 npz + repr dump。"""
    tw = batch["pose_gt"].shape[1]
    pose = pred_pose_w.reshape(-1, pred_pose_w.shape[-1]).cpu().numpy()
    trans = pred_trans_w.reshape(-1, 3).cpu().numpy()
    gt_trans = gt_trans_w.reshape(-1, 3).cpu().numpy()

    if npz_dir is not None and not args.skip_npz:
        n_windows = pose.shape[0] // tw
        if n_windows > 1 and FADE > 0:
            for w in range(1, n_windows):
                seam = w * tw
                if seam + FADE > pose.shape[0]:
                    break
                for j in range(FADE):
                    alpha = (j + 1) / (FADE + 1)
                    a, b = seam - FADE + j, seam + j
                    old_a, old_b = pose[a].copy(), pose[b].copy()
                    pose[a] = (1 - alpha) * old_a + alpha * old_b
                    pose[b] = (1 - alpha) * old_b + alpha * old_a
                    old_ta, old_tb = trans[a].copy(), trans[b].copy()
                    trans[a] = (1 - alpha) * old_ta + alpha * old_tb
                    trans[b] = (1 - alpha) * old_tb + alpha * old_ta
        R = rot6d_to_rotmat(torch.from_numpy(pose.reshape(-1, N_JOINTS, 6)).float())
        pose = rotmat_to_6d(R).reshape(-1, POSE_DIM).numpy()
        smpl_poses = smpl24_pose6d_to_poses(pose)
        np.savez_compressed(
            npz_dir / ("%s_%s%s.npz" % (session_id, cell_key(rV, rT), seed_suffix(seed))),
            poses=smpl_poses,
            trans=pelvis_to_smpl_trans(pose, trans, betas),
            pred_pelvis_trans=trans.astype(np.float32),
            gt_pelvis_trans=gt_trans.astype(np.float32),
            gt_trans=gt_trans.astype(np.float32),
            betas=betas if betas is not None else np.zeros((10,), dtype=np.float32),
            betas_source=np.asarray("ground_truth_session"),
            root_orient=smpl_poses[:, :3],
            pose_body=smpl_poses[:, 3:],
            mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
            source_frame_times_s=np.arange(len(pose), dtype=np.float32) / float(FPS),
            **smpl_archive_metadata(),
        )

    if repr_dir is not None and not args.skip_repr:
        dump = {}
        for name in ("F", "v_tok", "t_tok"):
            tensor = out.get(name)
            if tensor is not None:
                dump[name] = tensor.reshape(-1, tensor.shape[-1]).cpu().numpy()
        np.savez_compressed(
            repr_dir / ("%s_%s%s_repr.npz" % (session_id, cell_key(rV, rT),
                                              seed_suffix(seed))),
            **dump,
        )


if __name__ == "__main__":
    raise SystemExit(main())
