"""40 Hz motion windows with cached HRNet features and raw insole tokens.

AnySole motion targets are native SMPL-24.  The directly exported BVH remains
the trusted source for precomputed contact labels and, only for the historical
``s2m50`` input, Step2Motion-compatible synthetic IMU channels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from anysole.data.smpl_io import estimate_floor_y, load_smpl, resolve_smpl_path
from anysole.data.pressure import normalize_raw

STRIKE_TH = 100.0   # 落地事件判定：足力和 >100g（CONTACT_SUM_THRESH 同值）
STRIKE_CELL_TH = 20.0  # 激活格阈值（g）


def derive_strike(left48: np.ndarray, right48: np.ndarray):
    """维度 3 着地模式三分类标签（T 侧派生，2026-10-21）：每足落地事件
    （力和跨过 STRIKE_TH 的首帧）按首触区沿足长位置分 0=趾先 / 1=中足 /
    2=跟先。返回 (strike (T,2) int64，非落地帧 -1；mask (T,2) uint8)。"""
    n = left48.shape[0]
    strike = np.full((n, 2), -1, dtype=np.int64)
    smask = np.zeros((n, 2), dtype=np.uint8)
    cols = np.arange(12, dtype=np.float32)[None, None, :]
    for f, raw in ((0, left48), (1, right48)):
        grid = np.clip(raw.reshape(-1, 4, 12), 0.0, None)
        force = grid.sum(axis=(1, 2))
        onset = (force[1:] > STRIKE_TH) & (force[:-1] <= STRIKE_TH)
        active = grid > STRIKE_CELL_TH
        cnt = active.sum(axis=(1, 2))
        mean_col = (active * cols).sum(axis=(1, 2)) / np.maximum(cnt, 1.0)
        cls = np.where(mean_col < 4.0, 0, np.where(mean_col < 8.0, 1, 2))
        idx = np.where(onset)[0] + 1
        strike[idx, f] = cls[idx]
        smask[idx, f] = 1
    return strike, smask
from anysole.data.pressure import physical_tokens
from anysole.data.v5_1_gt import derive_phase_gt, load_cop
from anysole.data.occ_mix import (
    SRC_OCC,
    OccMixSampler,
    default_occ_cache_root,
    load_occ_session,
)
from anysole.data.workspace_adapter import (
    load_contact_label,
    load_shared_session,
    shared_smpl_path,
)
from anysole.data.tactile_s2m import build_t_s2m, resolve_legacy_bvh_path
from anysole.utils.geometry import (
    fk_pose6d_np,
    heading_from_root_np,
    rot6d_to_rotmat_np,
    rotmat_to_6d_np,
    yaw_rotmat_np,
)
from anysole.types import (
    CONFIG_PROBS,
    FPS,
    HMR_CACHE_ROOT,
    HRNET_CACHE_ROOT,
    N_JOINTS,
    SEQ_ROOT,
    SPLIT_CSV,
    TW,
    V_FEAT_DIM,
    V_HMR_DIM,
    V_HMR_IMG_DIM,
    V_HMR_KP_DIM,
    V_HMR_MISC_DIM,
    V_HMR_ROT_DIM,
    SMPL_ROOTS,
)


def load_split_ids(split_csv: Path, column: str) -> List[str]:
    ids: List[str] = []
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError("%s missing column %s" % (split_csv, column))
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                ids.append(value)
    return ids


def find_session_dir(seq_root: Path, session_id: str) -> Path:
    matches = sorted(seq_root.glob("*/*/%s" % session_id))
    dirs = [path for path in matches if path.is_dir()]
    if not dirs:
        raise FileNotFoundError("No sequence dir for %s under %s" % (session_id, seq_root))
    return dirs[0]


def session_time_grid(meta: dict) -> np.ndarray:
    if "visual_time_s" in meta:
        return np.asarray(meta["visual_time_s"], dtype=np.float64)
    n = int(meta["n_frames"])
    start = float(meta["visual_start_s"])
    return start + np.arange(n, dtype=np.float64) / float(meta.get("target_fps", FPS))


def hrnet_cache_path(session_id: str, cache_root: Path = HRNET_CACHE_ROOT) -> Path:
    return Path(cache_root) / ("%s.pt" % session_id)


def hmr_cache_path(model: str, session_id: str, cam: int = 3) -> Path:
    """F1: per-session GVHMR cache written by anysole/data/extract_hmr.py
    (四相机扩展 D2: per-camera cache dir cam{N})."""
    return HMR_CACHE_ROOT / model / ("cam%d" % cam) / ("%s.pt" % session_id)


def assemble_v_hmr(hmr: dict, n_frames: int) -> np.ndarray:
    """(N, V_HMR_DIM) per-frame GVHMR channel from an hmr cache file.

    Layout: rot 76 (body_pose aa 63 + global_orient 3 + betas 10)
    | kp2d 51 | f_imgseq 1024 | misc 5 (bbox cx/w, cy/h, size/hypot + q_V 2).
    """
    rot = np.concatenate(
        [hmr["body_pose"], hmr["global_orient"], hmr["betas"]], axis=1
    ).astype(np.float32)
    kp = hmr["kp2d"].reshape(n_frames, -1).astype(np.float32)
    img = hmr["f_imgseq"].astype(np.float32)
    bbx = hmr["bbx_xys"].astype(np.float32)
    img_w = float(hmr.get("img_w", 1.0))
    img_h = float(hmr.get("img_h", 1.0))
    diag = float(np.hypot(img_w, img_h))
    misc = np.stack(
        [bbx[:, 0] / img_w, bbx[:, 1] / img_h, bbx[:, 2] / diag,
         hmr["q_v"][:, 0], hmr["q_v"][:, 1]],
        axis=1,
    ).astype(np.float32)
    out = np.concatenate([rot, kp, img, misc], axis=1).astype(np.float32)
    if out.shape != (n_frames, V_HMR_DIM):
        raise ValueError(
            "V_hmr shape %s != (%d, %d); rot/kp/img/misc = %d/%d/%d/%d"
            % (out.shape, n_frames, V_HMR_DIM, rot.shape[1], kp.shape[1], img.shape[1], misc.shape[1])
        )
    return out


def contact_label_path(seq_dir: Path, method: str) -> Path:
    """Contact label file for a session.

    This helper is legacy-only.  The default shared-facts loader uses
    ``workspace_adapter.load_contact_label`` and never reads labels from a
    MotionPRO sequence tree.
    """
    if str(method) == "tactile_abs":
        return Path(seq_dir) / "contact.npy"
    return Path(seq_dir) / ("contact_%s.npy" % str(method))


def sample_config_ids(
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    probs: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Sample config ids in the fixed order ``[VT, V, T]``.

    ``probs`` is configurable for training experiments while the old call
    ``sample_config_ids(batch_size, generator)`` remains valid.
    """
    values = CONFIG_PROBS if probs is None else tuple(float(value) for value in probs)
    if len(values) != 3:
        raise ValueError("config probabilities must contain exactly 3 values: [VT, V, T]")
    if any(value < 0.0 for value in values):
        raise ValueError("config probabilities must be non-negative")
    total = sum(values)
    if not total > 0.0:
        raise ValueError("config probabilities must have a positive sum")
    probs = torch.tensor(values, dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.multinomial(probs, batch_size, replacement=True, generator=generator)


def collate_windows(samples: Sequence[dict]) -> dict:
    out: Dict[str, object] = {}
    keys = samples[0].keys()
    for key in keys:
        if key in ("session_id",):
            out[key] = [sample[key] for sample in samples]
        else:
            out[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return out


class AnySoleDataset(Dataset):
    def __init__(
        self,
        mode: str,
        seq_root: Path = SEQ_ROOT,
        split_csv: Path = SPLIT_CSV,
        cache_root: Path = HRNET_CACHE_ROOT,
        window_length: int = TW,
        session_ids: Optional[Sequence[str]] = None,
        allow_missing_video: bool = False,
        contact_method: str = "f6_soft",
        stride: Optional[int] = None,
        tactile_input: str = "raw108",
        no_imu: bool = False,
        v_input: str = "hrnet",
        v_hmr_model: str = "gvhmr",
        f2_repr: bool = False,
        smpl_roots: Optional[Sequence[Path]] = None,
        cam_id: int = 3,
        load_v5_1_gt: bool = False,
        d8_heads: bool = False,
        occ_mix: Optional[dict] = None,
        t_phys_v2: bool = False,
    ):
        self.mode = mode
        self.window_length = int(window_length)
        self.tactile_input = str(tactile_input)
        self.t_phys_v2 = bool(t_phys_v2)
        self.cam_id = int(cam_id)
        # 四相机扩展 D2：cam_id 切换缓存目录——GVHMR 经 hmr_cache_path(cam)；
        # HRNet 经 cache_root 重定向到 <hrnet_root>/cam{N}（extract_hrnet
        # --cam-id N 的落盘布局；默认值指向 cam3，显式传入的 cache_root 不动）。
        if cache_root == HRNET_CACHE_ROOT:
            cache_root = HRNET_CACHE_ROOT.parent / ("cam%d" % self.cam_id)
        if self.tactile_input not in ("raw108", "s2m50"):
            raise ValueError("tactile_input must be 'raw108' or 's2m50'")
        if no_imu and self.tactile_input != "s2m50":
            raise ValueError("no_imu requires tactile_input='s2m50'")
        # --no-imu: T_s2m is built without the synthesized IMU channels
        # (38-dim instead of 50-dim; see tactile_s2m.py).
        self.no_imu = bool(no_imu)
        # F1: hmr_gvhmr reads the GVHMR cache as the visual input (V_hmr);
        # the HRNet cache is still loaded (L_Vrec target / V-only fallback).
        self.v_input = str(v_input)
        if self.v_input not in ("hrnet", "hmr_gvhmr"):
            raise ValueError("v_input must be 'hrnet' or 'hmr_gvhmr', got %r" % self.v_input)
        self.v_hmr_model = str(v_hmr_model)
        # F2a: pose_gt root 6D becomes the tilt (heading removed), and the
        # trajectory target becomes the 4-dim heading-frame quantity.
        self.f2_repr = bool(f2_repr)
        self.smpl_roots = tuple(Path(p) for p in (smpl_roots or SMPL_ROOTS))
        # E6.3: window stride. None = non-overlapping (stride == window_length,
        # the pre-E6 behavior). Training may use stride=1 (every frame
        # alignment, Step2Motion-style); continuation eval uses tw//2.
        self.stride = int(stride) if stride is not None else self.window_length
        if not 1 <= self.stride <= self.window_length:
            raise ValueError("stride must be in [1, window_length]")
        self.seq_root = Path(seq_root)
        self.cache_root = Path(cache_root)
        self.allow_missing_video = allow_missing_video
        self.contact_method = str(contact_method)
        # V5_Av1 z 通道 GT（2026-10-19）：相位运动学派生（__init__ 内，
        # kp_gt + floor_y 同口径）；CoP = smplx 正演数据件（独立构建脚本
        # anysole.data.v5_1_gt 落盘，缺失即报错——训练不按需构建）。
        self.load_v5_1_gt = bool(load_v5_1_gt)
        self.d8_heads = bool(d8_heads)
        # V6_Av1 occ_mix（2026-10-19）：窗口级结构化丢弃采样器，默认关
        # （None = 现行为逐位不变）。启用后逐会话装载 occ010 双缓存、
        # __getitem__ 采样窗源并挂标签（occ_src/occ_v_zeroed/occ_t_zeroed）。
        self.occ_sampler = OccMixSampler(**occ_mix) if occ_mix else None

        if session_ids is None:
            column = {"train": "train", "eval": "val", "val": "val", "test": "test"}[mode]
            session_ids = load_split_ids(split_csv, column)
        self.session_ids = list(session_ids)

        self.sessions: List[dict] = []
        self.valid_windows: List[tuple] = []
        smpl_cache: Dict[str, Path] = {}

        for session_id in self.session_ids:
            seq_dir = find_session_dir(self.seq_root, session_id)
            shared = None
            if (seq_dir / "session.json").is_file():
                shared = load_shared_session(session_id, self.seq_root)
                shared_frames = shared["frames"]
                shared_meta = shared["meta"]
                n_frames = len(shared_frames["frame_id"])
                t_grid = np.asarray(shared_frames["visual_time_s"], dtype=np.float64)
                t_mocap = np.asarray(shared_frames["mocap_time_s"], dtype=np.float64)
                meta = dict(shared_meta)
                meta.update({
                    "n_frames": n_frames,
                    "date": shared_meta.get("date", ""),
                    "subject": shared_meta.get("subject", ""),
                    "offset_s": float(shared_meta.get("offset_s", 0.0)),
                    "bvh_path": shared_meta.get("source_files", {}).get("bvh", ""),
                })
                try:
                    smpl_path = shared_smpl_path(shared, self.smpl_roots)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"AnySole requires SMPL motion for {session_id}; {exc}. "
                        "BVH is supported only by Step2Motion."
                    ) from exc
            else:
                raise FileNotFoundError(
                    f"{session_id}: missing shared facts session.json under {seq_dir}")
            motion = load_smpl(smpl_path, query_t=t_mocap)
            if motion["pose_6d"].shape[0] != n_frames:
                raise ValueError(
                    "%s SMPL resample length %d != n_frames %d"
                    % (session_id, motion["pose_6d"].shape[0], n_frames)
                )
            if shared is not None:
                shared_pressure = shared["pressure"]
                left48 = np.asarray(shared_pressure["left48"], dtype=np.float32)
                right48 = np.asarray(shared_pressure["right48"], dtype=np.float32)
                t_raw_source = np.concatenate([left48, right48], axis=1)
                pressure = {
                    "T_raw": t_raw_source,
                    "T_phys": physical_tokens(
                        normalize_raw(left48), normalize_raw(right48),
                        t_phys_v2=self.t_phys_v2,
                    ),
                    "left48": left48,
                    "right48": right48,
                }
                try:
                    contact = load_contact_label(shared, self.contact_method)
                except FileNotFoundError:
                    # Contact labels are model-private derived inputs.  A
                    # cleaned workspace may legitimately not contain them;
                    # materialize the requested method from shared facts on
                    # first use, then load it through the same validation path.
                    from anysole.data.contact_adapter import build_one
                    print(
                        "building missing AnySole contact label: "
                        f"session={session_id} method={self.contact_method}"
                    )
                    build_one(session_id, [self.contact_method], shared_root=self.seq_root)
                    contact = load_contact_label(shared, self.contact_method)
                fake_mask = np.maximum(
                    np.asarray(shared["frames"]["fake"], dtype=np.uint8),
                    1 - np.asarray(shared["frames"]["valid"], dtype=np.uint8),
                )

            cache_path = hrnet_cache_path(session_id, self.cache_root)
            if cache_path.is_file():
                v_feat = torch.load(cache_path, map_location="cpu")
                if torch.is_tensor(v_feat):
                    v_feat = v_feat.float().cpu().numpy()
                else:
                    v_feat = np.asarray(v_feat, dtype=np.float32)
            elif self.allow_missing_video:
                if self.occ_sampler is not None:
                    # 零 V 会话无真实遮挡缓存语义（occ_mix.py 挂钩点 1b）：
                    # allow_missing_video 回退路径不挂 occ_mix。
                    raise ValueError(
                        "occ_mix is incompatible with allow_missing_video "
                        "(session %s has no cam3 cache, hence no occ cache)"
                        % session_id)
                v_feat = np.zeros((n_frames, V_FEAT_DIM), dtype=np.float32)
            else:
                raise FileNotFoundError(
                    "Missing AnySole HRNet+bbox cache %s. Run `python -m anysole.data.extract_hrnet --cam-id 3`."
                    % cache_path
                )
            if v_feat.shape != (n_frames, V_FEAT_DIM):
                raise ValueError(
                    "%s V_feat shape %s != (%d, %d)" % (session_id, v_feat.shape, n_frames, V_FEAT_DIM)
                )

            # F1: GVHMR visual channel (only when the run consumes it).
            if self.v_input == "hmr_gvhmr":
                hmr_path = hmr_cache_path(self.v_hmr_model, session_id, self.cam_id)
                if not hmr_path.is_file():
                    raise FileNotFoundError(
                        "Missing GVHMR cache %s. Run `python -m anysole.data.extract_hmr` "
                        "(downloads required, see extract_hmr.py header)." % hmr_path
                    )
                hmr = torch.load(hmr_path, map_location="cpu")
                v_hmr = assemble_v_hmr(hmr, n_frames)
            else:
                v_hmr = None

            kp = fk_pose6d_np(motion["pose_6d"], motion["trans_m"], motion["offsets_m"], motion["parents"])
            # Native SMPL feet are joint centers, not sole vertices, and the
            # archive world origin is not guaranteed to lie exactly on the
            # floor.  This floor reference is only for the SMPL prediction-side
            # contact metric/loss.  Ground-truth contact labels remain
            # precomputed by the unchanged directly exported BVH pipeline.
            floor_y = estimate_floor_y(kp)
            t_raw_norm = normalize_raw(pressure["T_raw"])
            # E6.6a is an explicit historical auxiliary input.  Its IMU comes
            # from the directly exported BVH-23/ToeBase chain, exactly as in
            # Step2Motion; it must never be synthesized from the SMPL target.
            # Default raw108 runs do not parse BVH or synthesize IMU at all.
            if self.tactile_input == "s2m50":
                t_s2m = build_t_s2m(
                    t_raw_norm,
                    bvh_path=None if self.no_imu else resolve_legacy_bvh_path(meta),
                    query_t=None if self.no_imu else t_mocap,
                    no_imu=self.no_imu,
                )
            else:
                t_s2m = np.zeros((n_frames, 50), dtype=np.float32)
            # E6.1: root-local positions of the 23 non-root joints, in the
            # session frame-0 root frame (Step2Motion initial_global_rot
            # convention).  World(t) = R_init @ local(t), so turns stay in the
            # local yaw and are learned from the condition.
            root_rot = rot6d_to_rotmat_np(
                motion["pose_6d"].reshape(n_frames, N_JOINTS, 6)[:, 0:1, :]
            )[:, 0]  # (T,3,3) world root rotation
            root_rot_init = root_rot[0].astype(np.float32)
            rel = kp[:, 1:, :] - kp[:, 0:1, :]  # (T,23,3) world-relative
            pose_pos = np.einsum(
                "ij,tpj->tpi", root_rot_init.T, rel
            ).reshape(n_frames, -1).astype(np.float32)
            trans = motion["trans_m"]
            vel = np.zeros_like(trans)
            # Store true forward differences in m/s.  The first frame has no
            # predecessor and is therefore the zero increment.
            if n_frames > 1:
                vel[1:] = (trans[1:] - trans[:-1]) * float(FPS)

            # F2a: heading/tilt representation (per session, before windows).
            if self.f2_repr:
                root_rot = rot6d_to_rotmat_np(
                    motion["pose_6d"].reshape(n_frames, N_JOINTS, 6)[:, 0]
                )
                psi = heading_from_root_np(root_rot)  # (T,), unwrapped
                r_yaw = yaw_rotmat_np(psi)
                tilt = np.einsum("tji,tjk->tik", r_yaw, root_rot)  # R_yaw^T @ R_root
                pose_f2 = motion["pose_6d"].copy()
                pose_f2[:, :6] = rotmat_to_6d_np(tilt)
                # psi_dot / v_h use the true forward differences; the first
                # frame of the sequence has no predecessor and gets ZERO
                # motion (not the plan's copy-next-frame: zero keeps the
                # f2->world roundtrip exact at frame 0, since the window
                # anchor is that same frame's world state).
                psi_dot = np.zeros_like(psi)
                vel_xy = np.zeros((n_frames, 2), dtype=np.float64)
                if n_frames > 1:
                    psi_dot[1:] = (psi[1:] - psi[:-1]) * float(FPS)
                    vel_xy[1:] = (trans[1:, [0, 2]] - trans[:-1, [0, 2]]) * float(FPS)
                v_h = np.einsum("tji,tj->ti", r_yaw, np.stack(
                    [vel_xy[:, 0], np.zeros_like(vel_xy[:, 0]), vel_xy[:, 1]], axis=1
                ))[:, [0, 2]]  # R_yaw^T @ v_xy, y padded
                traj_f2 = np.concatenate(
                    [psi_dot[:, None], v_h, trans[:, 1:2]], axis=1
                ).astype(np.float32)
            else:
                psi = None
                traj_f2 = None

            session = {
                "session_id": session_id,
                "motion_format": "smpl",
                "smpl_path": str(smpl_path),
                "V_feat": v_feat.astype(np.float32),
                "T_raw": t_raw_norm,
                "T_phys": pressure["T_phys"],
                "T_s2m": t_s2m,
                "pose_gt": pose_f2 if self.f2_repr else motion["pose_6d"],
                "pose_gt_pos": pose_pos,
                "traj_gt_f2": traj_f2,
                "psi": psi,
                "root_rot_init": root_rot_init,
                "trans_global": trans.astype(np.float32),
                "vel_gt": vel.astype(np.float32),
                "kp_gt": kp.astype(np.float32),
                "floor_y": floor_y,
                "contact_gt": contact.astype(np.float32),
                "offsets": motion["offsets_m"].astype(np.float32),
                "parents": motion["parents"].astype(np.int64),
                "betas": motion.get("betas", np.zeros(10, dtype=np.float32)).astype(np.float32),
                "fake_mask": fake_mask,
            }
            if self.occ_sampler is not None:
                # occ_mix 挂钩点 1b：occ010 双缓存装载（同帧轴强制核对，
                # 缺失/不符立即报错）。
                session["V_feat_occ"] = load_occ_session(
                    session_id,
                    occ_root=default_occ_cache_root(self.cache_root),
                    cam3_array=v_feat)
            if self.d8_heads:
                # 维度 3 着地模式标签（2026-10-21）：T 侧落地事件首触区
                # 三分类（0=趾先 / 1=中足 / 2=跟先），仅落地帧有标签。
                strike, strike_mask = derive_strike(
                    pressure["left48"], pressure["right48"])
                session["strike_gt"] = strike
                session["strike_mask"] = strike_mask
            if self.load_v5_1_gt:
                # V5_Av1 z 通道 GT：相位（探针同口径运动学派生，直接算）+
                # CoP（smplx 数据件装载，接触帧专用）。掩码全空（近静止
                # 会话）属正常——损失按掩码跳过。
                phase = derive_phase_gt(kp, float(floor_y))
                cop, cop_mask = load_cop(session_id, n_frames)
                session["phase_gt"] = phase["phase"]
                session["phase_mask"] = phase["phase_mask"]
                session["cop_gt"] = cop
                session["cop_mask"] = cop_mask
            if self.v_input == "hmr_gvhmr":
                session["V_hmr"] = v_hmr.astype(np.float32)
            file_index = len(self.sessions)
            if n_frames >= self.window_length:
                n_windows = (n_frames - self.window_length) // self.stride + 1
            else:
                n_windows = 0
            for window_idx in range(n_windows):
                left = window_idx * self.stride
                right = left + self.window_length
                if fake_mask[left:right].any():
                    continue
                self.valid_windows.append((file_index, left, right))
            self.sessions.append(session)

        print("%s sessions=%d windows=%d contact_method=%s tactile_input=%s no_imu=%s" % (
            self.mode, len(self.sessions), len(self.valid_windows), self.contact_method,
            self.tactile_input, self.no_imu))

    def __len__(self) -> int:
        return len(self.valid_windows)

    def __getitem__(self, idx: int) -> dict:
        file_index, left, right = self.valid_windows[idx]
        session = self.sessions[file_index]

        def crop(name: str) -> torch.Tensor:
            value = session[name][left:right]
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor

        # occ_mix 挂钩点 1c：crop 装配之前的窗口级采样（整窗一致）。
        aug = self.occ_sampler.sample_one() if self.occ_sampler is not None else None
        item = {
            "V_feat": crop("V_feat_occ") if (aug is not None and aug.v_src == SRC_OCC) else crop("V_feat"),
            "T_raw": crop("T_raw"),
            "T_phys": crop("T_phys"),
            "T_s2m": crop("T_s2m"),
            "pose_gt": crop("pose_gt"),
            "pose_gt_pos": crop("pose_gt_pos"),
            "vel_gt": crop("vel_gt"),
            "kp_gt": crop("kp_gt"),
            "contact_gt": crop("contact_gt"),
            "floor_y": torch.tensor(session["floor_y"], dtype=torch.float32),
            "offsets": torch.from_numpy(session["offsets"]).float(),
            "parents": torch.from_numpy(session["parents"]).long(),
            "betas": torch.from_numpy(session["betas"]).float(),
            "root_rot_init": torch.from_numpy(session["root_rot_init"]).float(),
            "session_id": session["session_id"],
            "frame_start": torch.tensor(left, dtype=torch.long),
        }
        if aug is not None:
            item.update(aug.batch_fields())
        if self.v_input == "hmr_gvhmr":
            item["V_hmr"] = crop("V_hmr")
        if self.d8_heads:
            item["strike_gt"] = crop("strike_gt")
            item["strike_mask"] = crop("strike_mask")
        if self.load_v5_1_gt:
            item["phase_gt"] = crop("phase_gt")
            item["phase_mask"] = crop("phase_mask")
            item["cop_gt"] = crop("cop_gt")
            item["cop_mask"] = crop("cop_mask")
        if self.f2_repr:
            item["traj_gt_f2"] = crop("traj_gt_f2")
            item["psi_anchor"] = torch.tensor(
                session["psi"][max(left - 1, 0)], dtype=torch.float32
            )
        # Trajectory targets are relative to the frame immediately before the
        # window.  Dividing by FPS integrates m/s back to meters.
        item["trans_gt"] = torch.cumsum(item["vel_gt"], dim=0) / float(FPS)
        anchor_index = max(left - 1, 0)
        item["trans_anchor"] = torch.from_numpy(
            np.ascontiguousarray(session["trans_global"][anchor_index])
        ).float()
        # ``betas`` is session metadata used by eval_protocol/SMPL export; it
        # is not passed as a model input.  Keep the guard for actual SMPL or
        # theta tensors, but do not reject the metadata required by evaluation.
        banned = [key for key in item if key.lower() in ("smpl", "theta") or "smpl" in key.lower()]
        if banned:
            raise RuntimeError("SMPL fields leaked into batch: %s" % banned)
        return item
