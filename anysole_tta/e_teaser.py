"""E6/E7 teaser 图生成：按时间间距抽取可见人体 SMPL 快照，偏航恒定（侧视）。

--mode e6：读 {cell} 逐帧 SMPL 姿态（global_orient 置零 = 偏航恒 0），
          N 帧网格渲染，每帧标注时间与左右脚接触状态（f6_soft）。
--mode e7：AnySole-U 五阶段装配关节（V/T/VT 读 cell、Vi/ViT 读 TTA dump），
          逐帧去偏航（髋轴对齐 + 速度/朝向消歧），N 帧骨架渲染，
          每帧标注时间与输入阶段。

--count N：按总运动序列时间的 1/N 间距取 N 个快照（每段中心最近有效帧）。

用法（touch_gait env）：
  python anysole_tta/e_teaser.py --mode e6 --session S14033 --count 10
  python anysole_tta/e_teaser.py --mode e7 --session S14033 --count 10
输出：results_display/Etest/{E6_ContactPhaseAnalysis,E7_DynamicModalityInputs}/
Full_Teaser_{E6,E7}_{sid}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import PROJECT_ROOT  # noqa: E402
from anysole_tta.phase_common import (CELL_ROOT, load_contact,  # noqa: E402
                                      load_pred_joints, phase_labels)
from anysole_tta.e7_driver import (TTA_DUMP_ROOT, stage_of_frame,  # noqa: E402
                                   STAGES)
from anysole.types import JOINT_NAMES, JOINT_PARENTS  # noqa: E402

SMPL_MODEL = PROJECT_ROOT / "AnysoleWorkspace/assets/third_party/smpl/SMPL_NEUTRAL.pkl"
E6_OUT = PROJECT_ROOT / "results_display/Etest/E6_ContactPhaseAnalysis"
E7_OUT = PROJECT_ROOT / "results_display/Etest/E7_DynamicModalityInputs"
FPS = 40.0


def sample_indices(valid: np.ndarray, n: int, count: int) -> list[int]:
    """N 段每段中心时刻最近的有效帧。"""
    targets = (np.arange(count) + 0.5) * n / count
    valid_idx = np.where(valid)[0]
    if not len(valid_idx):
        raise SystemExit("no valid frames")
    return [int(valid_idx[np.abs(valid_idx - t).argmin()]) for t in targets]


def _rot_y(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def unyaw_joints(joints: np.ndarray, forward: np.ndarray | None = None,
                 velocity: np.ndarray | None = None) -> np.ndarray:
    """去偏航：髋轴对齐 +X；朝向/速度消前后歧义（朝 +Z）。"""
    hip = joints[2] - joints[1]
    angle = np.arctan2(hip[2], hip[0])
    out = joints @ _rot_y(-angle).T
    z = forward @ _rot_y(-angle).T if forward is not None else None
    if z is None and velocity is not None:
        z = velocity @ _rot_y(-angle).T
    if z is not None and z[2] < 0:
        out = out @ _rot_y(np.pi).T
    return out


def e6_mesh(ax, vertices: np.ndarray, faces: np.ndarray) -> None:
    """单帧 SMPL 网格（native +Y up → matplotlib +Z up 轴交换，侧视）。"""
    import matplotlib.pyplot as plt
    ax.plot_trisurf(vertices[:, 0], vertices[:, 2], vertices[:, 1],
                    triangles=faces, color=(0.72, 0.72, 0.76), shade=True,
                    linewidth=0, antialiased=False)
    lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 1.9)
    ax.set_box_aspect((2, 2, 1.9))
    ax.view_init(elev=8, azim=0)
    ax.axis("off")


def e7_skeleton(ax, joints: np.ndarray) -> None:
    """单帧骨架（去偏航后 2D 侧视：x = 水平、y = 垂直）。"""
    ax.plot([joints[p, 0] for p in JOINT_PARENTS if p >= 0],
            [joints[p, 1] for p in JOINT_PARENTS if p >= 0],
            color="#555555", lw=2, zorder=1)
    for p, name in enumerate(JOINT_NAMES):
        if p == 0:
            continue
        child = joints[p]
        par = joints[JOINT_PARENTS[p]]
        ax.plot([par[0], child[0]], [par[1], child[1]], color="#555555",
                lw=2, zorder=1)
    ax.scatter(joints[:, 0], joints[:, 1], s=18, c="#2b6cb0", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("e6", "e7"))
    ap.add_argument("--session", default="S14033")
    ap.add_argument("--count", type=int, default=10,
                    help="快照个数（按总时长 1/N 间距抽取）")
    ap.add_argument("--cell", default="V7_Cv5_s1")
    ap.add_argument("--cond", default="VT2M", choices=("V2M", "VT2M", "T2M"))
    ap.add_argument("--model", default="V7_Cv5")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.mode == "e6":
        d = np.load(CELL_ROOT / args.cell / "predictions/eval_motion" /
                    f"{args.session}_{args.cond}.npz", allow_pickle=True)
        poses = d["poses"].astype(np.float64)          # (T,72) aa，native
        betas = d["betas"].astype(np.float64)          # (10,)
        vmask = d["valid_mask"]
        idx = sample_indices(vmask, len(poses), args.count)
        poses = poses[idx].copy()
        poses[:, :3] = 0.0                             # 偏航恒 0（侧视）
        import smplx
        import torch
        layer = smplx.create(str(SMPL_MODEL), "smpl", gender="neutral",
                             batch_size=len(idx), num_betas=10)
        with torch.no_grad():
            out = layer(global_orient=torch.from_numpy(
                            poses[:, :3].astype(np.float32)),
                        body_pose=torch.from_numpy(
                            poses[:, 3:].astype(np.float32)),
                        betas=torch.from_numpy(
                            np.tile(betas, (len(idx), 1)).astype(np.float32)))
        verts = out.vertices.numpy()
        faces = layer.faces
        contact, _, _ = load_contact(args.session)
        phases = phase_labels(contact, fps=FPS)
        fig, axes = plt.subplots(1, len(idx), figsize=(2.2 * len(idx), 2.6),
                                 subplot_kw={"projection": "3d"})
        for ax, i in zip(axes, idx):
            j = int(np.where(np.asarray(idx) == i)[0][0])
            e6_mesh(ax, verts[j], faces)
            ph = phases[i]
            l = "stance" if contact[i, 0] > 0.5 else "swing"
            r = "stance" if contact[i, 1] > 0.5 else "swing"
            ax.set_title(f"t={i / FPS:.1f}s  L:{l} R:{r}", fontsize=9)
        fig.suptitle(f"E6 teaser — {args.session} {args.cell} {args.cond}",
                     y=0.98)
        out_path = E6_OUT / f"Full_Teaser_E6_{args.session}_{args.cond}.png"
    else:
        # E7：五阶段装配关节（V/T/VT = cell，Vi/ViT = TTA dump）
        from anysole_tta.e7_driver import _gt_native
        n = len(_gt_native(args.session))
        joints_all = np.full((n, 24, 3), np.nan)
        vmask = np.zeros(n, dtype=bool)
        for stage in STAGES:
            si = STAGES.index(stage)
            if stage in ("V", "T", "VT"):
                cond = {"V": "V2M", "T": "T2M", "VT": "VT2M"}[stage]
                pred, _, vm = load_pred_joints(args.cell, args.session, cond)
            else:
                dump_dir = TTA_DUMP_ROOT / f"{args.model}_{stage}"
                pred = np.full((n, 24, 3), np.nan)
                vm = np.zeros(n, dtype=bool)
                for path in sorted(dump_dir.glob(f"{args.session}_w*.npz")):
                    dp = np.load(path)
                    st = int(path.stem.split("_w")[-1])
                    seg = slice(st, min(st + len(dp["joints"]), n))
                    pred[seg] = dp["joints"][:seg.stop - seg.start]
                    vm[seg] = True
            sel = stage_of_frame(np.arange(n), n) == si
            joints_all[sel] = pred[sel]
            vmask[sel] = vm[sel]
        idx = sample_indices(vmask, n, args.count)
        # 逐帧去偏航：朝向用 cell 姿态根（V/T/VT 段），Vi/ViT 用骨盆速度
        d = np.load(CELL_ROOT / args.cell / "predictions/eval_motion" /
                    f"{args.session}_VT2M.npz", allow_pickle=True)
        poses_ref = d["poses"].astype(np.float64)
        from scipy.spatial.transform import Rotation as R
        root_rot = R.from_rotvec(poses_ref[:, :3]).as_matrix()   # (T,3,3)
        fig, axes = plt.subplots(1, len(idx), figsize=(2.2 * len(idx), 2.6))
        for ax, i in zip(axes, idx):
            joints = joints_all[i] - joints_all[i, 0:1]
            forward = root_rot[i] @ np.array([0.0, 0.0, 1.0])
            if i + 1 < n and np.isfinite(joints_all[i + 1, 0]).all():
                velocity = joints_all[i + 1, 0] - joints_all[i, 0]
            else:
                velocity = None
            unyawed = unyaw_joints(joints, forward=forward, velocity=velocity)
            e7_skeleton(ax, unyawed)
            si = int(stage_of_frame(np.array([i]), n)[0])
            ax.set_title(f"t={i / FPS:.1f}s [{STAGES[si]}]", fontsize=9)
        fig.suptitle(f"E7 teaser — {args.session} AnySole-U ({args.model})",
                     y=0.98)
        out_path = E7_OUT / f"Full_Teaser_E7_{args.session}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("→", out_path)


if __name__ == "__main__":
    main()
