"""S2 前置检查（2026-11-07，门控联合设计方案 §S2 前置检查，零训练）。

把 Cv7 的 traj_ATE 收益（Δ=VT−V，四折 occ000）按 GT 朝向坐标系分解成
前进（forward，身体 +Z）与侧向（lateral，身体 +X）两个分量，判断"侧向
分量完全没有收益、可以硬屏蔽"这个假设是否成立。

口径：复刻 ``eval_protocol.py`` 的 ``root_ate_unanchored_mm``（窗边界用 GT
锚重置，链式累计窗内相对位移误差）——同一个算法，只是把标量范数换成
按 GT 逐帧航向 psi 分解的两个有符号分量，和标量版逐位对账（范数应相等）。

GT 航向来自 AnySoleDataset 的 ``pose_gt`` 根 6D（``heading_from_root_np``），
按 session_id 对齐 predictions/eval_motion 的 ``frame_indices``。

用法（touch_gait env，仓库根）：
  python results_display/script/s2_traj_ate_lateral_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from anysole.train import load_config  # noqa: E402
from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.utils.geometry import (  # noqa: E402
    rot6d_to_rotmat, heading_from_root_np, yaw_rotmat_np)

TW = 20
FAM_FOLDS = (
    ("fS5S6S11", "anysole/configs/v1_fold_S5S6S11.yaml", "V7_Cv7_fS5S6S11_s1", "V7_Cv5_fS5S6S11_s1"),
    ("fS5S11S12", "anysole/configs/v1_fold_S5S11S12.yaml", "V7_Cv7_fS5S11S12_s1", "V7_Cv5_fS5S11S12_s1"),
    ("fS6S7S13", "anysole/configs/v1_fold_S6S7S13.yaml", "V7_Cv7_fS6S7S13_s1", "V7_Cv5_fS6S7S13_s1"),
    ("fS5S6S8", "anysole/configs/v1_fold_S5S6S8.yaml", "V7_Cv7_fS5S6S8_s1", "V7_Cv5_fS5S6S8_s1"),
)


def load_gt_heading(config_path: str) -> dict[str, dict]:
    """Build AnySoleDataset(test) for one fold, return per-session GT psi/root."""
    config = load_config(REPO / config_path)
    tw = int(config.get("tw", TW))
    split_csv = Path(config["split_csv"])
    session_ids = load_split_ids(split_csv, "test")
    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=split_csv,
        cache_root=Path(config["cache_root"]), window_length=tw,
        session_ids=session_ids, contact_method=str(config["contact_method"]),
        tactile_input="raw108", no_imu=False, v_input="hrnet",
        f2_repr=False, smpl_roots=config.get("smpl_roots"),
        cam_id=int(config.get("cam_id", 3)), t_phys_v2=False,
        stride=None, obs_context=tw, smpl_support_policy="exclude")
    out = {}
    for s in ds.sessions:
        pg = s["pose_gt"].reshape(-1, 24, 6)
        root_rotmat = rot6d_to_rotmat(torch.from_numpy(pg[:, 0, :])).numpy()
        psi = heading_from_root_np(root_rotmat)
        out[s["session_id"]] = {"psi": psi, "root_pos": s["kp_gt"][:, 0, :]}
    return out


def unanchored_decompose(pred_root: np.ndarray, gt_root: np.ndarray,
                          psi: np.ndarray, tw: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate root_ate_unanchored_mm's windowed GT-reset chaining, but
    return the per-frame error vector decomposed into (forward, lateral)
    instead of collapsing to a scalar norm. Returns (fwd_err_m, lat_err_m),
    each (T,) signed, meters. Norm(fwd,lat,vertical) reproduces the scalar
    metric (vertical dropped here since the gate groups split horizontal
    fwd/lat from vertical already)."""
    t = pred_root.shape[0]
    gt_anchors = gt_root
    n_windows = (t + tw - 1) // tw
    unanchored = np.zeros_like(pred_root)
    for w in range(n_windows):
        start, stop = w * tw, min((w + 1) * tw, t)
        anchor = gt_anchors[max(start - 1, 0)]
        base = unanchored[start - 1] if start > 0 else gt_anchors[0]
        unanchored[start:stop] = base + (pred_root[start:stop] - anchor)
    err = unanchored - gt_anchors  # (T,3) world
    r = yaw_rotmat_np(psi)  # (T,3,3), maps body -> world
    err_body = np.einsum("tij,tj->ti", np.transpose(r, (0, 2, 1)), err)
    # body axes: x = lateral (right), z = forward (per yaw_rotmat_np convention)
    lat = err_body[:, 0]
    fwd = err_body[:, 2]
    return fwd, lat


def session_frames(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    # NOTE: ``joint_xyz_world`` is ``world_z_up`` (per its own
    # ``joint_coordinate_system`` field) while ``pred_pelvis_trans`` /
    # ``gt_pelvis_trans`` are the SMPL y-up world frame ("world motion
    # preserved") that matches the dataset's ``kp_gt``/``trans_global`` and
    # ``heading_from_root_np``'s +Z-forward convention. Mixing the two here
    # silently produced ~meter-scale "errors" on first pass — use the
    # y-up pelvis trans fields for both pred and GT root position.
    return {
        "pred_root": d["pred_pelvis_trans"],
        "gt_root": d["gt_pelvis_trans"],
        "frame_indices": d["frame_indices"],
        "session_id": str(d["session_id"]),
    }


def main():
    rows = []
    for fold, cfg_path, cv7_cell, cv5_cell in FAM_FOLDS:
        gt_by_session = load_gt_heading(cfg_path)
        for model, cell in (("Cv7", cv7_cell), ("Cv5", cv5_cell)):
            motion_dir = REPO / "results/cells" / cell / "predictions/eval_motion"
            for cond in ("V2M", "VT2M"):
                fwd_vals, lat_vals, total_vals = [], [], []
                n_sessions = 0
                for f in sorted(motion_dir.glob(f"*_{cond}.npz")):
                    sid = f.stem[: -len(f"_{cond}")]
                    if sid not in gt_by_session:
                        continue
                    gt = gt_by_session[sid]
                    data = session_frames(f)
                    fi = data["frame_indices"]
                    psi = gt["psi"][fi]
                    gt_root = data["gt_root"]  # use the npz's own GT (y-up, matches pred)
                    pred_root = data["pred_root"]
                    fwd, lat = unanchored_decompose(pred_root, gt_root, psi, TW)
                    fwd_vals.append(np.abs(fwd).mean() * 1000.0)
                    lat_vals.append(np.abs(lat).mean() * 1000.0)
                    total_vals.append(
                        np.linalg.norm(
                            np.stack([fwd, lat], -1), axis=-1).mean() * 1000.0)
                    n_sessions += 1
                rows.append({
                    "fold": fold, "model": model, "cond": cond,
                    "n_sessions": n_sessions,
                    "fwd_mm": float(np.mean(fwd_vals)) if fwd_vals else float("nan"),
                    "lat_mm": float(np.mean(lat_vals)) if lat_vals else float("nan"),
                    "total_mm": float(np.mean(total_vals)) if total_vals else float("nan"),
                })

    print(f"{'fold':12} {'model':5} {'cond':5} {'n':>3} {'fwd_mm':>8} {'lat_mm':>8} {'total_mm':>9}")
    for r in rows:
        print(f"{r['fold']:12} {r['model']:5} {r['cond']:5} {r['n_sessions']:>3} "
              f"{r['fwd_mm']:>8.2f} {r['lat_mm']:>8.2f} {r['total_mm']:>9.2f}")

    print()
    print("=== Delta(VT-V) per fold, per model ===")
    by_key = {(r["fold"], r["model"], r["cond"]): r for r in rows}
    for fold, _, _, _ in FAM_FOLDS:
        for model in ("Cv7", "Cv5"):
            v = by_key.get((fold, model, "V2M"))
            vt = by_key.get((fold, model, "VT2M"))
            if v is None or vt is None:
                continue
            dfwd = vt["fwd_mm"] - v["fwd_mm"]
            dlat = vt["lat_mm"] - v["lat_mm"]
            dtot = vt["total_mm"] - v["total_mm"]
            print(f"{fold:12} {model:5} dFwd={dfwd:+8.3f} dLat={dlat:+8.3f} dTotal={dtot:+8.3f} "
                  f"lat_share={abs(dlat)/(abs(dfwd)+abs(dlat)+1e-9):.3f}")

    print()
    print("=== Mean across 4 folds ===")
    for model in ("Cv7", "Cv5"):
        dfwds, dlats, dtots = [], [], []
        for fold, _, _, _ in FAM_FOLDS:
            v = by_key.get((fold, model, "V2M"))
            vt = by_key.get((fold, model, "VT2M"))
            if v is None or vt is None:
                continue
            dfwds.append(vt["fwd_mm"] - v["fwd_mm"])
            dlats.append(vt["lat_mm"] - v["lat_mm"])
            dtots.append(vt["total_mm"] - v["total_mm"])
        print(f"{model:5} mean dFwd={np.mean(dfwds):+8.3f} mean dLat={np.mean(dlats):+8.3f} "
              f"mean dTotal={np.mean(dtots):+8.3f} lat_share={abs(np.mean(dlats))/(abs(np.mean(dfwds))+abs(np.mean(dlats))+1e-9):.3f}")


if __name__ == "__main__":
    main()
