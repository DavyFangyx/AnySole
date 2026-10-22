"""V5_Av1 z 通道 GT（相位 / CoP）——口径唯一来源（2026-10-19）。

派生口径逐段复制自 `z_note/multimodal_fusion/V5_方案/
probe_v5_phase_cop_readout.py`（2026-10-19 实测落定的前置探针，V5_Av1
§4.2），provenance 注明：

- GT 相位（每足归一化周期相位）：GT 足关节（10/11）世界高 < floor_y
  + 2cm 触地事件（上升沿）→ φ=(t−td_k)/(td_{k+1}−td_k)，周期 ≥10 帧才
  计数，目标 = (cos 2πφ, sin 2πφ)；掩码 = 首末触地之间。63/140 会话为
  近静止数据（无摆动期）→ 相位掩码全空属正常，损失按掩码跳过。
  计算只需 FK 关节点 + floor_y（无 smplx）——dataset.__init__ 里按会话
  直接调用 :func:`derive_phase_gt`。
- GT CoP：smplx 正演（smpl_vertices_from_archive_params）→ foot_heat
  4×12 顶点计数网格（复制自 results_display/script/d_test8_smpl_gt_cop.py
  foot_heat，provenance 注明）→ cop_from_grid（与触觉 CoP 同公式同单位
  0-1 网格坐标）；掩码 = 接触帧且和值 > 0。

CoP 数据件设计（2026-10-19 裁定）：**独立构建脚本 + 装载函数**
（`python -m anysole.data.v5_1_gt --split train,val,test`），落盘
model_inputs/AnySole/{ADAPTER_VERSION}/gt_v5_1/{session_id}.npz；dataset
严格装载、缺失即报错并给出构建命令。选此而非 contact_adapter 的按需
build_one 模式，理由：(1) smplx 正演重——训练 num_workers>0 时每个
worker 各自构建 dataset，按需构建会在 __init__ 里触发 N 进程并发写
同一文件（ρ 网格合并竞态前科）；(2) 仓库纪律 = 训练不按需提取特征
（HRNet 缓存缺失即报错，extract_hrnet 前例）；(3) 相位派生廉价且
motion-based，直接在 dataset.__init__ 计算，无需落盘。
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from anysole.data.pressure import cop_from_grid
from anysole.data.smpl_io import (
    estimate_floor_y,
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.workspace_adapter import (
    ANYSOLE_INPUT_ROOT,
    PROTOCOL_SPLIT,
    load_shared_session,
    shared_smpl_path,
)
from anysole.types import SMPL_ROOTS
from anysole.utils.geometry import fk_pose6d_np

ANKLE_L, ANKLE_R, FOOT_L, FOOT_R = 7, 8, 10, 11
GT_CONTACT_DZ = 0.02          # 触地阈值（D8 口径）
GT_ROWS, GT_COLS = 4, 12
GT_S0, GT_S1, GT_LAT = 0.10, 0.16, 0.045   # foot_heat 网格边界（D8 口径）
GT_ANKLE_DZ = 0.02
MIN_CYCLE_SPAN = 10           # 相位周期最短帧数（噪声触地不计数）

V5_Av1_GT_ROOT = ANYSOLE_INPUT_ROOT / "gt_v5_1"


def derive_phase_gt(kp_gt: np.ndarray, floor_y: float) -> dict:
    """每足 (cos 2πφ, sin 2πφ) + 掩码——复制自 probe_v5_phase_cop_readout.py
    session_gt 相位段（2026-10-19）。kp_gt: (n_frames, 24, 3) 世界 FK。

    返回 {"phase": (n,2,2) [足 L/R, cos/sin] NaN 填充, "phase_mask":
    (n,2) bool}。掩码全空（近静止会话）属正常。"""
    n = kp_gt.shape[0]
    out_phase = np.full((n, 2, 2), np.nan, dtype=np.float32)
    out_mask = np.zeros((n, 2), dtype=bool)
    for i_side, side in enumerate(("L", "R")):
        foot_j = FOOT_L if side == "L" else FOOT_R
        # 触地事件：GT 足关节世界高 < floor + 2cm（D8 顶点口径同阈值，代理
        # = 足关节）。不用 f6_soft 磁盘标签——2026-10-19 审计：标签为二值
        # 且 63/140 会话全 1 退化，运动学 GT 更干净。
        contact_use = kp_gt[:, foot_j, 1] < float(floor_y) + GT_CONTACT_DZ
        idx = np.where(np.diff(contact_use.astype(np.int8)) == 1)[0] + 1   # 上升沿 = 触地
        if len(idx) < 2:
            continue
        lo, hi = idx[0], idx[-1]
        for k, td in enumerate(idx[:-1]):
            td_next = idx[k + 1]
            span = max(td_next - td, 1)
            if span < MIN_CYCLE_SPAN:      # 周期过短（噪声触地）不计数
                continue
            seg = np.arange(td, td_next)
            phi = (seg - td) / span
            out_phase[seg, i_side, 0] = np.cos(2 * np.pi * phi)
            out_phase[seg, i_side, 1] = np.sin(2 * np.pi * phi)
            out_mask[seg, i_side] = True
        out_mask[:lo, i_side] = False
        out_mask[hi:, i_side] = False
    return {"phase": out_phase, "phase_mask": out_mask}


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


def session_cop_gt(shared: dict, motion: dict, floor: float, n_frames: int):
    """单会话 GT CoP（每足，smplx 正演 foot_heat → cop_from_grid）——
    复制自 probe_v5_phase_cop_readout.py session_gt 的 CoP 段。

    返回 (cop (n,2,2) NaN 填充, cop_mask (n,2) bool, contact (n,2) bool)。"""
    joints = fk_pose6d_np(motion["pose_6d"], motion["trans_m"],
                          motion["offsets_m"], motion["parents"])
    verts = smpl_vertices_from_archive_params(
        motion["poses_aa"], motion["model_trans_m"], motion["betas"])
    # archive 覆盖可能短于 facts 帧轴（末会话尾部）：GT 数组统一补到
    # n_frames，尾部帧置无效（掩码 False、目标 NaN）。
    n_use = min(n_frames, len(motion["pose_6d"]))

    cop = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    cop_mask = np.zeros((n_frames, 2), dtype=bool)
    contact = np.zeros((n_frames, 2), dtype=bool)
    for i_side, side in enumerate(("L", "R")):
        foot_j, ankle_j = (FOOT_L, ANKLE_L) if side == "L" else (FOOT_R, ANKLE_R)
        contact_use = joints[:, foot_j, 1] < float(floor) + GT_CONTACT_DZ
        contact[:n_use, i_side] = contact_use
        # GT CoP：foot_heat → cop_from_grid；接触帧且和值>0 有效
        for i in range(n_use):
            heat = foot_heat(verts[i], joints[i, ankle_j], joints[i, foot_j], floor, side)
            if heat.sum() > 0 and contact_use[i]:
                cop[i, i_side] = cop_from_grid(heat.flatten())
                cop_mask[i, i_side] = True
    return cop, cop_mask, contact


def cop_path(session_id: str) -> Path:
    return V5_Av1_GT_ROOT / ("%s.npz" % session_id)


def build_one(session_id: str, seq_root: Path, *, force: bool = False) -> Path:
    """构建单会话 GT CoP 并落盘 model_inputs 区（原子替换，多进程安全）。"""
    output = cop_path(session_id)
    if output.is_file() and not force:
        return output
    shared = load_shared_session(session_id, seq_root)
    n_frames = len(shared["frames"]["frame_id"])
    mocap_t = np.asarray(shared["frames"]["mocap_time_s"], dtype=np.float64)[:n_frames]
    smpl_path = shared_smpl_path(shared, SMPL_ROOTS)
    motion = load_smpl(smpl_path, query_t=mocap_t)
    if motion["pose_6d"].shape[0] != n_frames:
        raise ValueError(
            "%s SMPL resample length %d != n_frames %d"
            % (session_id, motion["pose_6d"].shape[0], n_frames))
    joints = fk_pose6d_np(motion["pose_6d"], motion["trans_m"],
                          motion["offsets_m"], motion["parents"])
    floor = float(estimate_floor_y(joints))          # dataset 同口径 floor_y
    cop, cop_mask, contact = session_cop_gt(shared, motion, floor, n_frames)
    frame_id = np.asarray(shared["frames"]["frame_id"], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".%s.%d.tmp.npz" % (output.name, os.getpid()))
    np.savez_compressed(
        temporary,
        frame_id=frame_id,
        cop=cop.astype(np.float32),
        cop_mask=cop_mask.astype(np.uint8),
        contact=contact.astype(np.uint8),
        floor_y=np.asarray(floor, dtype=np.float32),
        source_session=np.asarray(shared["dir"]),
    )
    os.replace(temporary, output)
    print("wrote %s (n=%d, cop valid L/R=%d/%d)"
          % (output, n_frames, int(cop_mask[:, 0].sum()), int(cop_mask[:, 1].sum())))
    return output


def load_cop(session_id: str, n_frames: int) -> tuple:
    """装载会话 GT CoP：形状/帧轴校验。缺失即报错并给出构建命令。"""
    path = cop_path(session_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing V5_Av1 GT CoP cache {path}. Run "
            f"`python -m anysole.data.v5_1_gt --session {session_id}`."
        )
    data = dict(np.load(path, allow_pickle=True))
    cop = np.asarray(data["cop"], dtype=np.float32)
    mask = np.asarray(data["cop_mask"], dtype=np.uint8).astype(bool)
    frame_id = np.asarray(data["frame_id"])
    if cop.shape != (n_frames, 2, 2) or mask.shape != (n_frames, 2) \
            or len(frame_id) != n_frames:
        raise ValueError(
            "%s: cop shape %s / mask %s != n_frames %d — rebuild with "
            "`python -m anysole.data.v5_1_gt --session %s --force`"
            % (path, cop.shape, mask.shape, n_frames, session_id))
    return cop, mask


def _split_ids(split: str) -> list[str]:
    with Path(PROTOCOL_SPLIT).open(encoding="utf-8-sig", newline="") as handle:
        return [
            row[split].strip()
            for row in csv.DictReader(handle)
            if (row.get(split) or "").strip()
        ]


def main(argv: list[str] | None = None) -> int:
    from anysole.types import SEQ_ROOT  # local: avoid top-level import weight

    parser = argparse.ArgumentParser(description="Build V5_Av1 GT CoP caches from shared facts.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--session", action="append", default=[])
    scope.add_argument("--split", default=None,
                       help="comma-separated splits (train,val,test)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    sessions = list(args.session)
    if args.split:
        for split in [s.strip() for s in args.split.split(",") if s.strip()]:
            sessions.extend(_split_ids(split))
    if not sessions:
        raise SystemExit("provide --session or --split; refusing an implicit full rebuild")
    seen = set()
    for session_id in sessions:
        if session_id in seen:
            continue
        seen.add(session_id)
        try:
            build_one(session_id, Path(SEQ_ROOT), force=args.force)
        except Exception as exc:
            print("%s: FAILED %s" % (session_id, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
