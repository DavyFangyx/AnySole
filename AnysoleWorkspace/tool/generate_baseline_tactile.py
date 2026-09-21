"""三基线触觉转换落盘生成器（D_Test5 前置）。

把 AnySole 原始触觉逐帧转换成三个基线的触觉格式并落盘：

    MotionPRO   FRAPPE 实际输入的 96x96 压力图（与 image_pressure.py 同口径
                bilinear / align_corners=False，0-255）
    Step2Motion 16 通道/脚（process_gait 冻结池化口径：heel[0-7] + toe[8-15]；
                展示/审计产物，不替代原始协议）
    MMVP        pressure_tookit 与 VP-MoCap 共用同一套 31x11 insole/脚
                （点对点按 SMPL 模板足底系欧氏最近点映射，0-255）；
                toolkit 的 load_contact 与 FPP-Net 的 PED_tempKPCont
                读同一份 {'insole': [left, right]} 文件，故写两棵树

转换口径与 results_display/script/d_test5_baseline_tactile.py 逐函数一致（该
脚本是本生成器的可视化前端：产物齐全则跳过生成、缺则调用本脚本，--force
强制重生成）。单测不单独成文件，由前端的存在性检测 + 本脚本幂等跳过承担。

每 session 产物：
    AnysoleWorkspace/derived/MotionPRO/pressure_96/<sid>.npz                      {'pressure': (T,96,96)}
    AnysoleWorkspace/derived/Step2Motion/pressure_16ch/<sid>.npz                  {'left16','right16': (T,16)}
    AnysoleWorkspace/derived/pressure_tookit/images/<date>/<sub>/<sid>/insole/%03d.npy
    AnysoleWorkspace/derived/VP-MoCap/<date>/<sub>/<sid>/insole/%03d.npy          （同上，共用）
    AnysoleWorkspace/derived/baseline_tactile/<sid>/meta.json                     映射统计 + 产物路径

用法（仓库根目录，touch_gait 环境）：
    python AnysoleWorkspace/tool/generate_baseline_tactile.py --sessions S12072
    python AnysoleWorkspace/tool/generate_baseline_tactile.py --sessions S12072 --force
    python AnysoleWorkspace/tool/generate_baseline_tactile.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loguru import logger as log  # noqa: E402

MANIFEST = REPO_ROOT / "AnysoleWorkspace" / "manifests" / "session_manifest.jsonl"
SPLITS = REPO_ROOT / "AnysoleWorkspace" / "splits" / "default" / "splits.csv"
LAYOUT_DIR = REPO_ROOT / "AnysoleWorkspace" / "calibration" / "foot_sensor_layout"
ESSENTIALS = REPO_ROOT / "Baselines" / "VP-MoCap" / "FPP-Net" / "essentials" / "insole2cont"

OUT_MOTIONPRO = REPO_ROOT / "AnysoleWorkspace" / "derived" / "MotionPRO" / "pressure_96"
OUT_S2M = REPO_ROOT / "AnysoleWorkspace" / "derived" / "Step2Motion" / "pressure_16ch"
OUT_TOOLKIT = REPO_ROOT / "AnysoleWorkspace" / "derived" / "pressure_tookit" / "images"
OUT_FPP = REPO_ROOT / "AnysoleWorkspace" / "derived" / "VP-MoCap"
OUT_META = REPO_ROOT / "AnysoleWorkspace" / "derived" / "baseline_tactile"

# 冻结脚框：与 results_display/script/utils/render_common.py 的 LEFT/RIGHT_FOOT_BOX
# 逐值一致（生成器自持，不从 results_display 反向 import）。
LEFT_FOOT_BOX = (slice(40, 120), slice(6, 54))
RIGHT_FOOT_BOX = (slice(40, 120), slice(66, 114))

# ---------------------------------------------------------------- manifest / split

def manifest_record(session_id: str) -> dict:
    """从 session_manifest.jsonl 解析 session 记录（含 pressure_path / n_frames）。"""
    for line in Path(MANIFEST).open(encoding="utf-8"):
        rec = json.loads(line.strip())
        if rec.get("session_id") == session_id:
            return rec
    raise FileNotFoundError(f"{session_id} not in {MANIFEST}")


def resolve_session(session_id: str) -> dict:
    """session_id → {seq_dir, date, subject, n_frames}。date/subject 取自 pressure_path。"""
    rec = manifest_record(session_id)
    parts = Path(rec["pressure_path"]).parts
    cam_idx = parts.index("cam3")
    return {
        "session": session_id,
        "date": parts[cam_idx + 1],
        "subject": parts[cam_idx + 2],
        "n_frames_manifest": int(rec.get("n_frames", 0)),
        "seq_dir": REPO_ROOT / Path(*parts[:cam_idx + 4]),
    }


def split_sessions(split: str) -> list[str]:
    """splits.csv 某一列（train/val/test）的 session 列表；all = 三列并集。"""
    cols = [] if split == "all" else [split]
    if split == "all":
        cols = ["train", "val", "test"]
    sessions: list[str] = []
    with Path(SPLITS).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for col in cols:
                if row.get(col, "").strip():
                    sessions.append(row[col].strip())
    return sorted(set(sessions))


# ---------------------------------------------------------------- 几何（与 D_Test5 逐函数一致）

def sole_bbox(ess: dict, foot: str) -> tuple[float, float, float, float]:
    """模板足底包围盒 (xmin, xmax, zmin, zmax)。z：+z=脚尖，-z=脚跟。"""
    ids = ess["footIds" + foot]
    v = ess["v_template"][ids]
    return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 2].min()), float(v[:, 2].max())


def load_dots(foot: str) -> np.ndarray:
    """厂商布局 48 点 norm 坐标（(48,2)，行序 = 通道序）。"""
    name = {"L": "leftfoot", "R": "rightfoot"}[foot]
    return np.genfromtxt(LAYOUT_DIR / f"{name}_dots.csv", delimiter=",",
                         skip_header=1, usecols=(1, 2)).astype(np.float64)


def our_cell_positions(ess: dict, foot: str, mirror_x: bool = False) -> np.ndarray:
    """我方 48 格在模板足底系的 (x,z) 位置（(48,2)，索引 = 通道号）。"""
    dots = load_dots(foot)
    xmin, xmax, zmin, zmax = sole_bbox(ess, foot)
    xn = (dots[:, 0] - dots[:, 0].min()) / max(dots[:, 0].max() - dots[:, 0].min(), 1e-9)
    yn = (dots[:, 1] - dots[:, 1].min()) / max(dots[:, 1].max() - dots[:, 1].min(), 1e-9)
    if mirror_x:
        xn = 1.0 - xn
    x = xmin + xn * (xmax - xmin)
    z = zmax - yn * (zmax - zmin)                            # y 大端（脚尖）→ +z
    return np.stack([x, z], axis=1)


def mmvp_pixel_positions(ess: dict, foot: str) -> np.ndarray:
    """MMVP 31x11 每个像素的模板 (x,z) 位置，掩码外为 NaN。"""
    mapping = ess["insole2smpl" + foot]
    pos = np.full((31, 11, 2), np.nan)
    for v_str, rc_ in mapping.items():
        if not rc_[0].size:
            continue
        v = ess["v_template"][int(v_str)]
        for r, c in zip(rc_[0], rc_[1]):
            pos[r, c] = v[[0, 2]]
    return pos


def nearest_cell_map(ess: dict, foot: str, mirror_x: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """每个 MMVP 掩码像素 → 最近我方格的编号 + 距离统计（模板系，米）。"""
    from scipy.spatial import cKDTree
    ours = our_cell_positions(ess, foot, mirror_x)
    pos = mmvp_pixel_positions(ess, foot)
    mask = np.isfinite(pos[..., 0])
    tree = cKDTree(ours)
    dist, idx = tree.query(pos[mask])                        # 每个掩码像素 → 最近格
    cell_idx = np.full((31, 11), -1, dtype=int)
    cell_idx[mask] = idx
    stats = {
        "n_px": int(mask.sum()),
        "mean_mm": float(dist.mean() * 1000),
        "max_mm": float(dist.max() * 1000),
        "median_mm": float(np.median(dist) * 1000),
        "n_cells_used": int(np.unique(idx).size),
    }
    return cell_idx, mask, stats


# ---------------------------------------------------------------- 数据

def load_nine_piece(seq_dir: Path) -> dict:
    d = {}
    d["pressure"] = np.load(seq_dir / "pressure.npz")["pressure"].astype(np.float32)
    fake = seq_dir / "fake_mask.npy"
    d["fake"] = np.load(fake).astype(bool).reshape(-1) if fake.is_file() else \
        np.zeros(len(d["pressure"]), dtype=bool)
    return d


def crop_cells(pressure_t: np.ndarray) -> np.ndarray:
    """(T,160,120) -> (T,2,4,12)：冻结脚框读回 48 格（通道序 = 厂商 dot 序）。"""
    out = []
    for box in (LEFT_FOOT_BOX, RIGHT_FOOT_BOX):
        block = np.asarray(pressure_t[:, box[0], box[1]], dtype=np.float32)
        out.append(block.reshape(-1, 4, 20, 12, 4).mean(axis=(2, 4)))
    return np.stack(out, axis=1)


def load_essentials(base: Path) -> dict:
    d = {"maskL": np.loadtxt(base / "insoleMaskL.txt").astype(np.int32),
         "maskR": np.loadtxt(base / "insoleMaskR.txt").astype(np.int32),
         "insole2smplL": np.load(base / "insole2smplL.npy", allow_pickle=True).item(),
         "insole2smplR": np.load(base / "insole2smplR.npy", allow_pickle=True).item(),
         "footIdsL": np.loadtxt(base / "footL_ids.txt").astype(np.int32),
         "footIdsR": np.loadtxt(base / "footR_ids.txt").astype(np.int32)}
    import trimesh
    d["v_template"] = np.array(trimesh.load(base / "smpl_template.obj", process=False).vertices)
    return d


def pool_48_to_16(cells: np.ndarray) -> np.ndarray:
    """(4,12) -> 16 通道（process_gait 冻结口径）：heel(col6-11) 8 通道在前，toe 8 在后。"""
    grid = np.asarray(cells).reshape(4, 12)
    toe = grid[:, :6].reshape(4, 2, 3).mean(axis=-1).reshape(8)
    heel = grid[:, 6:].reshape(4, 2, 3).mean(axis=-1).reshape(8)
    return np.concatenate([heel, toe])


# ---------------------------------------------------------------- 产物路径

def product_paths(session_id: str) -> dict:
    """给定 session 的全部产物路径（存在性检测与生成共用同一处定义）。"""
    r = resolve_session(session_id)
    date, sub, sid = r["date"], r["subject"], session_id
    return {
        "motionpro_npz": OUT_MOTIONPRO / f"{sid}.npz",
        "s2m_npz": OUT_S2M / f"{sid}.npz",
        "toolkit_insole_dir": OUT_TOOLKIT / date / sub / sid / "insole",
        "fpp_insole_dir": OUT_FPP / date / sub / sid / "insole",
        "meta": OUT_META / f"{sid}.json",
    }


def products_complete(paths: dict) -> bool:
    """产物齐全判据：4 个文件/目录存在，且两棵 insole 帧数与 pressure_96 帧数一致。"""
    for p in (paths["motionpro_npz"], paths["s2m_npz"], paths["meta"]):
        if not p.is_file():
            return False
    for p in (paths["toolkit_insole_dir"], paths["fpp_insole_dir"]):
        if not p.is_dir():
            return False
    t = np.load(paths["motionpro_npz"], mmap_mode="r")["pressure"].shape[0]
    for p in (paths["toolkit_insole_dir"], paths["fpp_insole_dir"]):
        n = sum(1 for f in p.iterdir() if f.suffix == ".npy")
        if n != t:
            return False
    return True


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三基线触觉转换落盘生成器")
    p.add_argument("--sessions", type=str, default="",
                   help="逗号分隔的 session 列表（与 --split 二选一）")
    p.add_argument("--split", type=str, default="",
                   choices=["train", "val", "test", "all"],
                   help="按 splits.csv 列批量生成")
    p.add_argument("--mirror-x", action="store_true",
                   help="布局 x 轴（内外侧）相对模板 x 翻转（默认同向假定）")
    p.add_argument("--force", action="store_true",
                   help="产物已存在时仍强制重新生成（默认齐全则跳过）")
    return p.parse_args()


def generate_session(session_id: str, args: argparse.Namespace) -> dict:
    paths = product_paths(session_id)
    if products_complete(paths) and not args.force:
        log.info("{} 产物齐全，跳过（--force 覆盖）", session_id)
        return paths
    r = resolve_session(session_id)
    data = load_nine_piece(r["seq_dir"])
    ess = load_essentials(ESSENTIALS)
    cells = crop_cells(data["pressure"])                       # (T,2,4,12)
    valid = np.where(~data["fake"])[0]
    sid, n = session_id, len(cells)
    log.info("{}: {} 帧（有效 {}），4x12 值域 {:.0f}-{:.0f}",
             sid, n, len(valid), cells.min(), cells.max())

    # ---- 几何映射（一次性）：每掩码像素 → 最近我方格
    maps, dist_stats = {}, {}
    for foot in ("L", "R"):
        cell_idx, mask, stats = nearest_cell_map(ess, foot, args.mirror_x)
        maps[foot] = (cell_idx, mask)
        dist_stats[foot] = stats
        log.info("{}: 掩码像素 {}，最近格距离 均值 {:.1f}mm / 中位 {:.1f}mm / 最大 {:.1f}mm，"
                 "用到我方 {} / 48 格", foot, stats["n_px"], stats["mean_mm"],
                 stats["median_mm"], stats["max_mm"], stats["n_cells_used"])
    n_mask = int(maps["L"][1].sum() + maps["R"][1].sum())

    # ---- FPP weight（静立帧总压，与 D_Test5 同口径；全 fake session 回退全帧）
    if len(valid) == 0:
        log.warning("{}: 无有效帧（fake 全 True），静立帧选取回退全帧", sid)
        standing = int(np.argmax(cells.sum(axis=(1, 2, 3))))
    else:
        standing = valid[np.argmax(cells[valid].sum(axis=(1, 2, 3)))]
    weight = float(cells[standing].sum())
    pw = weight / n_mask
    log.info("FPP weight（静立帧 {}）= {:.1f}，每像素当量 pw = {:.4f}", standing, weight, pw)

    # ---- MotionPRO：96x96 bilinear（image_pressure.py 口径）
    import torch
    t = torch.from_numpy(np.ascontiguousarray(data["pressure"])).float()
    r96 = torch.nn.functional.interpolate(t.unsqueeze(1), size=(96, 96),
                                          mode="bilinear", align_corners=False).squeeze(1).numpy()

    # ---- Step2Motion：16 通道/脚（冻结池化）
    left16 = np.stack([pool_48_to_16(cells[i, 0]) for i in range(n)])
    right16 = np.stack([pool_48_to_16(cells[i, 1]) for i in range(n)])

    # ---- MMVP：逐帧 31x11/脚（nearest-cell），写两棵树
    toolkit_dir = paths["toolkit_insole_dir"]
    fpp_dir = paths["fpp_insole_dir"]
    for d in (toolkit_dir, fpp_dir):
        d.mkdir(parents=True, exist_ok=True)
    for fi in range(n):
        grid = np.zeros((31, 22), dtype=np.float32)
        for k, foot in enumerate(("L", "R")):
            cell_idx, mask = maps[foot]
            g = cells[fi, k].ravel()[cell_idx]
            g[~mask] = 0
            grid[:, k * 11:(k + 1) * 11] = g
        payload = {"insole": [grid[:, :11], grid[:, 11:]]}
        for d in (toolkit_dir, fpp_dir):
            np.save(d / f"{fi:03d}.npy", payload)

    # ---- 落盘 npz 与 meta
    paths["motionpro_npz"].parent.mkdir(parents=True, exist_ok=True)
    paths["s2m_npz"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths["motionpro_npz"], pressure=r96.astype(np.float32))
    np.savez(paths["s2m_npz"], left16=left16.astype(np.float32),
             right16=right16.astype(np.float32))

    meta = {
        "session": sid, "date": r["date"], "subject": r["subject"],
        "n_frames": n, "n_valid": int(len(valid)),
        "all_fake": bool(len(valid) == 0),
        "valid_frame_indices": [int(v) for v in valid],
        "mirror_x": bool(args.mirror_x),
        "mmvp_mask_px": n_mask,
        "mapping": "nearest-cell by euclidean distance in SMPL template sole frame",
        "our_grid": "vendor layout (calibration/foot_sensor_layout/*foot_dots.csv), "
                    "dot_index = channel+1 (verified); x/y extents aligned to per-foot sole bbox",
        "nearest_dist_mm": dist_stats,
        "fpp_weight": weight, "fpp_pw": pw, "fpp_weight_frame": int(standing),
        "motionpro_input": "bilinear 96x96 /255 (image_pressure.py 口径)",
        "s2m_channels": "16/foot, heel[0-7] then toe[8-15] (process_gait 冻结)",
        "products": {k: str(v) for k, v in paths.items()},
    }
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("wrote motionpro/s2m npz + insole {} 帧 x2 树 + meta", n)
    return paths


def main() -> None:
    args = parse_args()
    if args.sessions and args.split:
        raise SystemExit("--sessions 与 --split 互斥")
    sessions = ([s for s in args.sessions.split(",") if s] if args.sessions
                else split_sessions(args.split) if args.split
                else [])
    if not sessions:
        raise SystemExit("请给 --sessions 或 --split")
    for sid in sessions:
        generate_session(sid, args)
    log.info("done: {} sessions", len(sessions))


if __name__ == "__main__":
    main()
