"""HMR2.0 npz 数据装配：facts 帧轴 + CLIFF bbox + MoSh SMPL GT + 四相机标定 → 官方 ImageDataset npz 契约。

每个 (session, cam) 产出一个 npz；键口径与官方 npz 一致（image_dataset.py:46-206），
附加 frame_id/session_id/date/cam 供本实验线 wrapper 使用。

口径（与计划一致）：
- 帧集合 = frames.valid==1 且 fake==0，且 bbox 面积 > 0；
- body_pose (N,72) aa：23 个体关节保持 MoSh 值；global_orient 前 3 旋转到相机系
  （R_glob_cam = Rc @ Rm @ R_glob_mocap，Rc/tc = world_to_camera，Rm/tm = mocap_raw_to_checkerboard_world）；
- 3D 关键点 (N,44,4)：官方 SMPL 层正演（相机系 global_orient、不加平移，pelvis 对齐损失不依赖平移），conf 全 1；
- 2D 关键点 (N,44,3)：MoSh 世界系关节 + trans_m → 标定链全透视投影（含畸变），npz 存全分辨率像素坐标，
  裁剪归一化由官方 get_example 运行时完成；conf=1 当点在画幅内；
- center/scale：CLIFF bbox 推导，scale = max(w,h)*1.2 像素（官方加载时 /200）。

用法：
  python data/assemble_npz.py --sessions S10101,S8023              # 指定会话
  python data/assemble_npz.py --split train --cams 1,2,3,4         # 按 split 全量
  python data/assemble_npz.py --sessions S10101 --dry-run          # 只打印键/形状
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hmr2_gait 根
from hmr2_tta.paths import (  # noqa: E402
    RESULTS_ROOT, bbox_path, calibration_json, facts_session_dir, resolve_uri,
)

PRESSURE_CLIP = 1023.0  # anysole/types.py PRESSURE_CLIP
CHUNK = 512             # SMPL 正演分块（CPU）


def load_manifest_rows(cam: int) -> dict:
    """读 session manifest jsonl → {session_id: row}。

    per-camera manifest（session_manifest_cam{N}.jsonl）在 2026-10-24 清洗中
    收敛为统一 session_manifest.jsonl（cam3 文件已删）；缺 per-cam 文件时
    回退统一 manifest。
    """
    from AnysoleWorkspace.tool.workspace import resolve_uri as _resolve
    path = _resolve(f"protocol://manifests/session_manifest_cam{cam}.jsonl",
                    for_write=False, must_exist=False)
    if not path.is_file():
        path = _resolve("protocol://manifests/session_manifest.jsonl",
                        for_write=False, must_exist=True)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["session_id"]] = row
    return rows


def load_splits() -> dict:
    """splits.csv → {session_id: {splits}}。2026-10-22 起 val≠test（val=S12、test=S13+S14），一个 session 属恰一个 split。"""
    path = resolve_uri("protocol://splits/default/splits.csv")
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for key in ("train", "val", "test"):
                if row[key]:
                    out.setdefault(row[key], set()).add(key)
    return out


def date_of(row: dict) -> str:
    """video_path raw://rgb/20260808/... → 20260808（取首个 8 位数字 token）。"""
    import re
    m = re.search(r"(\d{8})", row["video_path"])
    if not m:
        raise ValueError(f"no date in video_path: {row['video_path']}")
    return m.group(1)


def load_calibration_chain(cal_path: Path, cam: int) -> dict:
    """标定链：K/D/Rc/tc（world_to_camera）+ Rm/tm（mocap_raw_to_checkerboard_world）。"""
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    camd = data["cameras"][f"cam{cam}"]
    K = np.asarray(camd["K"], dtype=np.float64)
    Rc = np.asarray(camd["R"], dtype=np.float64)
    tc = np.asarray(camd["t"], dtype=np.float64).reshape(3, 1)   # cm
    D = np.asarray(camd["D"], dtype=np.float64).reshape(-1)
    mc = data["mocap_raw_to_checkerboard_world"]
    Rm = np.asarray(mc["R"], dtype=np.float64)
    tm = np.asarray(mc["t"], dtype=np.float64).reshape(3, 1)     # cm
    if not np.allclose(Rc.T @ Rc, np.eye(3), atol=2e-3):
        raise ValueError(f"{cal_path}: cam{cam} R not orthonormal")
    if not np.allclose(Rm.T @ Rm, np.eye(3), atol=2e-3):
        raise ValueError(f"{cal_path}: mocap_raw_to_checkerboard R not orthonormal")
    return {"K": K, "D": D, "Rc": Rc, "tc": tc, "Rm": Rm, "tm": tm, "source": str(cal_path)}


def build_smpl_layer():
    """官方 SMPL 层（25 openpose + 19 extra = 44 关节），保证 GT 与预测端关节布局同构。"""
    from hmr2.models.smpl_wrapper import SMPL
    from hmr2_tta.paths import CACHE_4DHUMANS, SMPL_MODEL_PATH
    j19 = CACHE_4DHUMANS / "data/SMPL_to_J19.pkl"
    if not j19.is_file():
        raise FileNotFoundError(f"SMPL_to_J19.pkl 未下载: {j19}")
    layer = SMPL(str(SMPL_MODEL_PATH), joint_regressor_extra=str(j19))
    layer.eval()
    return layer


def smpl_forward(layer, poses_aa: np.ndarray, betas: np.ndarray, chunk: int = CHUNK):
    """(T,72) aa → joints (T,44,3) 模型空间（含 global_orient，无平移）。

    smplx 0.1.28 的 forward 恒等 rotmat 输入（pose2rot 参数被忽略），故先 aa→rotmat。
    """
    from scipy.spatial.transform import Rotation as R
    if len(poses_aa) == 0:
        return np.zeros((0, 44, 3), dtype=np.float32)
    joints = []
    for start in range(0, len(poses_aa), chunk):
        p = poses_aa[start:start + chunk].astype(np.float64)
        rot = R.from_rotvec(p.reshape(-1, 3)).as_matrix().reshape(len(p), 24, 3, 3)
        rot = torch.from_numpy(rot.astype(np.float32))
        b = torch.from_numpy(np.asarray(betas, dtype=np.float32))
        out = layer(global_orient=rot[:, :1], body_pose=rot[:, 1:], betas=b.expand(len(p), -1))
        joints.append(out.joints.detach().cpu().numpy())
    return np.concatenate(joints, axis=0)


def project_world_to_cam(joints_world: np.ndarray, cal: dict) -> np.ndarray:
    """MoSh 世界系关节 (N,44,3)[m] → 该相机像素坐标 (N,44,2)（含畸变）。"""
    n, j, _ = joints_world.shape
    x_cb = cal["Rm"] @ (100.0 * joints_world.reshape(-1, 3).T) + cal["tm"]    # cm
    x_cam = cal["Rc"] @ x_cb + cal["tc"]
    pts = x_cam.T.reshape(n, j, 3)
    u, _ = cv2.projectPoints(pts.reshape(-1, 3), np.zeros(3), np.zeros(3),
                             cal["K"], cal["D"])
    return u.reshape(n, j, 2)


def read_pressure(facts_dir: Path, n_frames: int) -> dict:
    """cam3 facts 树压力（四相机共享；同 40Hz 帧轴）→ T (N,96) 归一化 + 掩码。"""
    p = np.load(facts_dir / "pressure_48.npz")
    raw = np.concatenate([p["left48"], p["right48"]], axis=1).astype(np.float64)
    assert raw.shape == (n_frames, 96), f"pressure shape {raw.shape} != ({n_frames},96)"
    t = np.clip(raw, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP
    return {"T": t.astype(np.float32), "valid": p["valid"].astype(np.int8),
            "fake": p["fake"].astype(np.int8), "frame_id": p["frame_id"]}


def assemble_session(sid: str, cams: list[int], manifest_rows: dict,
                    smpl_layer, out_root: Path, dry_run: bool) -> dict:
    """装配一个 session 的 per-camera npz + 压力缓存。返回 stats。"""
    row = manifest_rows[sid]
    date = date_of(row)
    cal_path = calibration_json(date)
    smpl_path = resolve_uri(row["smpl_path"])

    # 帧轴 + SMPL GT（四相机共享）
    facts_ref = facts_session_dir(sid, 3)
    frames = np.load(facts_ref / "frames.npz")
    n = int(row["n_frames"])
    assert len(frames["frame_id"]) == n
    from anysole.data.smpl_io import load_smpl
    smpl = load_smpl(smpl_path, query_t=frames["mocap_time_s"])
    poses_aa = smpl["poses_aa"].reshape(n, 24, 3)
    betas = smpl["betas"]
    trans_m = smpl["trans_m"]  # (N,3) MoSh 世界系骨盆位置 [m]

    # 压力（cam3 树，四相机共享）
    pressure = read_pressure(facts_ref, n)
    pres_out = out_root / "pressure" / f"{sid}.npz"
    if not dry_run:
        pres_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pres_out, **pressure, session_id=np.array(sid))

    stats = {"session": sid, "date": date, "n_frames": n, "cameras": {}}
    for cam in cams:
        cam_stats = assemble_camera(sid, cam, n, row, date, cal_path, poses_aa,
                                    betas, trans_m, smpl_layer, out_root, dry_run)
        stats["cameras"][f"cam{cam}"] = cam_stats
    return stats


def assemble_camera(sid, cam, n, row, date, cal_path, poses_aa, betas, trans_m,
                    smpl_layer, out_root, dry_run) -> dict:
    facts_dir = facts_session_dir(sid, cam)
    frames = np.load(facts_dir / "frames.npz")
    bbox = np.load(bbox_path(sid, cam))
    assert bbox.shape == (n, 8), f"{sid} cam{cam}: bbox {bbox.shape} != ({n},8)"
    cal = load_calibration_chain(cal_path, cam)

    # 帧掩码：valid & ~fake & bbox 面积 > 0
    area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 4] - bbox[:, 2])
    valid = (frames["valid"].astype(bool) & ~frames["fake"].astype(bool) &
             (area > 0))
    idx = np.where(valid)[0]
    if len(idx) == 0:
        print(f"[warn] {sid} cam{cam}: 0 valid frames, skipping npz")
        return {"n_valid": 0, "n_dropped": n,
                "bbox_w_p50": float(np.percentile(bbox[:, 3] - bbox[:, 1], 50)),
                "bbox_h_p50": float(np.percentile(bbox[:, 4] - bbox[:, 2], 50)),
                "kp2d_inframe_frac": 0.0}

    # global_orient → 相机系（R_glob_cam = Rc @ Rm @ R_glob_mocap）
    from scipy.spatial.transform import Rotation as R
    rot_mats = R.from_rotvec(poses_aa[:, 0]).as_matrix()            # (N,3,3) mocap 系
    g2c = cal["Rc"] @ cal["Rm"]
    rot_cam = R.from_matrix(np.einsum("ij,njk->nik", g2c, rot_mats)).as_rotvec()
    body_pose = np.concatenate([rot_cam[:, None, :], poses_aa[:, 1:]], axis=1).reshape(n, 72)

    # 3D GT：相机系 orient 正演（模型空间，无平移）
    joints_3d = smpl_forward(smpl_layer, body_pose[idx], betas)     # (M,44,3)

    # 2D GT：MoSh 世界系（原 orient + trans_m）全透视投影
    joints_world = smpl_forward(smpl_layer, poses_aa[idx].reshape(-1, 72), betas) + \
        trans_m[idx][:, None, :]
    kp2d_pix = project_world_to_cam(joints_world, cal)
    h, w = 1240.0, 1624.0
    conf = ((kp2d_pix[..., 0] >= 0) & (kp2d_pix[..., 0] < w) &
            (kp2d_pix[..., 1] >= 0) & (kp2d_pix[..., 1] < h)).astype(np.float32)[..., None]
    kp2d = np.concatenate([kp2d_pix, conf], axis=-1).astype(np.float32)
    body_2d, extra_2d = kp2d[:, :25], kp2d[:, 25:]

    body_3d = np.concatenate([joints_3d[:, :25], np.ones((len(idx), 25, 1), np.float32)], -1)
    extra_3d = np.concatenate([joints_3d[:, 25:], np.ones((len(idx), 19, 1), np.float32)], -1)

    # bbox → center/scale
    x1, y1, x2, y2 = bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 4]
    center = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=1)[idx]
    scale = (np.maximum(x2 - x1, y2 - y1) * 1.2)[idx][:, None]

    # imgname：facts rgb 符号链绝对路径
    imgname = np.array([str(facts_dir / "rgb" / f"{int(f):06d}.jpg") for f in
                        frames["frame_id"][idx]], dtype="S")

    payload = {
        "imgname": imgname, "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "body_pose": body_pose[idx].astype(np.float32),
        "has_body_pose": np.ones(len(idx), np.float32),
        "betas": np.tile(betas.astype(np.float32), (len(idx), 1)),
        "has_betas": np.ones(len(idx), np.float32),
        "body_keypoints_2d": body_2d, "extra_keypoints_2d": extra_2d,
        "body_keypoints_3d": body_3d, "extra_keypoints_3d": extra_3d,
        "frame_id": frames["frame_id"][idx].astype(np.int64),
        "session_id": np.array(sid), "date": np.array(date), "cam": np.array(cam),
    }
    cam_stats = {"n_valid": int(len(idx)), "n_dropped": int(n - len(idx)),
                 "bbox_w_p50": float(np.percentile(x2 - x1, 50)),
                 "bbox_h_p50": float(np.percentile(y2 - y1, 50)),
                 "kp2d_inframe_frac": float(conf.mean())}
    if dry_run:
        return cam_stats
    out = out_root / "npz" / f"cam{cam}" / f"{sid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return cam_stats


def write_index(out_root: Path, split_map: dict):
    """按 splits.csv 聚合已装配的 npz → index/{split}.json。"""
    for split in ("train", "val", "test"):
        entries = []
        for cam in (1, 2, 3, 4):
            for npz in sorted((out_root / "npz" / f"cam{cam}").glob("*.npz")):
                sid = npz.stem
                if split not in split_map.get(sid, ()):
                    continue
                entries.append({"sid": sid, "cam": cam, "npz": str(npz),
                                "pressure": str(out_root / "pressure" / f"{sid}.npz")})
        if entries:
            (out_root / "index").mkdir(parents=True, exist_ok=True)
            (out_root / "index" / f"{split}.json").write_text(
                json.dumps(entries, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="", help="逗号分隔 session 列表")
    ap.add_argument("--split", default="", help="train/val/test/all")
    ap.add_argument("--cams", default="1,2,3,4")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "data"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cams = [int(c) for c in args.cams.split(",")]
    out_root = Path(args.out)
    split_map = load_splits()
    rows = {cid: load_manifest_rows(cid) for cid in cams}

    if args.sessions:
        sessions = args.sessions.split(",")
    elif args.split:
        sessions = [s for s, sp in split_map.items()
                    if args.split == "all" or args.split in sp]
    else:
        raise SystemExit("需要 --sessions 或 --split")

    smpl_layer = build_smpl_layer()
    all_stats = []
    for sid in sorted(sessions):
        row0 = rows[cams[0]].get(sid)
        if not row0 or not row0.get("smpl_path"):
            print(f"[skip] {sid}: no smpl_path in manifest")
            continue
        if args.skip_existing and all(
                (out_root / "npz" / f"cam{c}" / f"{sid}.npz").exists() for c in cams):
            print(f"[skip] {sid}: existing")
            continue
        st = assemble_session(sid, cams, rows[cams[0]], smpl_layer, out_root, args.dry_run)
        all_stats.append(st)
        nv = " ".join(f"cam{c}:{st['cameras'][f'cam{c}']['n_valid']}" for c in cams)
        print(f"[ok] {sid} ({st['date']}) n={st['n_frames']} {nv}")
    if not args.dry_run:
        write_index(out_root, split_map)
    stats_path = out_root / "stats.json"
    if not args.dry_run:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if stats_path.exists():
            merged = {s["session"]: s for s in json.loads(stats_path.read_text())}
        for st in all_stats:
            merged[st["session"]] = st
        stats_path.write_text(json.dumps(list(merged.values()), indent=1, default=str),
                              encoding="utf-8")
        print(f"stats → {stats_path}")


if __name__ == "__main__":
    main()
