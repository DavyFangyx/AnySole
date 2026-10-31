"""E7 动态模态可用性下的统一运动重建（实验计划 E7 口径，2026-10-27 用户裁定）。

时间轴：连续序列均分 5 阶段（各 1/5 时长，窗口边界切换）——
  V（单相机）→ T（单触觉）→ VT（相机+触觉）→ Vi（多相机）→ ViT（多相机+触觉）

两条线（同一序列、同一阶段划分）：
- AnySole-U：V7_Cv4（单门）/ V7_Cv5（双门）同一 ckpt——V/T/VT 阶段直接读
  {model}_stat cell 逐帧预测（V2M/T2M/VT2M，ckpt_last）；Vi 阶段 =
  run_tta --v-only；ViT 阶段 = run_tta（CONFIG_VT）——均 --ckpt last、
  与 cell 口径一致。
- 基线组合（外部基线，SMPL-24 GT 同口径）：V 阶段 = WHAM（smpl24 协议
  直出，2026-10-30 接入）；T 阶段 = Step2Motion（smpl24 协议直出，
  2026-10-28 切换，与 MotionPRO 导出同构）；VT 阶段 = MotionPRO
  （smpl24 vs SMPL-24 GT）；Vi/ViT 阶段 = EasyMocap 标定多视角基线
  （n=4 四相机，2026-10-30 用户裁定接入——多视角参考线，与 E3 表 4 线 1
  同源，easymocap_baseline.py 缓存）。

阶段划分 = 窗口对齐（stride20 窗口网格，与 run_tta --win-frac 选窗规则
一致；阶段边界不落窗口中部，无缺口）。

指标：踝足 MPJPE（mm，骨盆对齐；跨骨架一致口径）。产物：
z_note/E_Test/E7_动态输入/{e7_table.json, E7_动态输入重建.md,
e7_phase_{walk,run,squat}.png}

用法（touch_gait env）：
  python anysole_tta/e7_driver.py --actions walk            # 冒烟
  python anysole_tta/e7_driver.py                           # walk+run+squat 全量
  python anysole_tta/e7_driver.py --force-tta               # 重生成 TTA dump
（TTA dump 已有则自动跳过）"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import PROJECT_ROOT, RESULTS_ROOT  # noqa: E402
from anysole_tta.phase_common import (CELL_ROOT, load_gt_joints,  # noqa: E402
                                      load_pred_joints)

OUT_DIR = PROJECT_ROOT / "results_display/Etest/E7_DynamicModalityInputs"
FPS = 40.0
STAGES = ("V", "T", "VT", "Vi", "ViT")
STAGE_ZH = {"V": "单相机", "T": "单触觉", "VT": "相机+触觉",
            "Vi": "多相机", "ViT": "多相机+触觉"}
STAGE_FRAC = {s: (i / 5, (i + 1) / 5) for i, s in enumerate(STAGES)}
ANYSOLE_MODELS = ["V7_Cv4", "V7_Cv5"]   # run_tta 模型名；cell = {m}_s1
SEQUENCES = {
    "walk": ["S14033"],
    "run": ["S13082"],
    "squat": ["S13061", "S13062", "S13063", "S14061", "S14062", "S14063"],
}
TTA_DUMP_ROOT = RESULTS_ROOT / "predictions/e7_tta"
SMPL_ANKLE_FOOT = (7, 8, 10, 11)
S2M_ROOT = PROJECT_ROOT / "results/baselines/Step2Motion/predictions/gait_model_smpl24/eval_motion"
MPRO_ROOT = PROJECT_ROOT / "results/baselines/MotionPRO/predictions/carpet_v2/eval_motion"
WHAM_ROOT = PROJECT_ROOT / "results/baselines/WHAM/predictions/local_gen/eval_motion"
EMC_ROOT = RESULTS_ROOT / "predictions/easymocap"   # EasyMocap n=4 四相机缓存


def _display_to_native(points: np.ndarray) -> np.ndarray:
    """z-up display（eval.py _smpl_yup_to_display）→ native y-up：(x, z, −y)。"""
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)


def stage_of_frame(t: np.ndarray, n: int) -> np.ndarray:
    """窗口对齐的阶段划分（窗口网格 = stride20 非重叠，窗口起始帧定阶段；
    与 run_tta --win-frac 的选窗规则一致——阶段边界不落窗口中部，无缺口）。"""
    s = np.minimum((t // 20) * 20, max(0, (n - 20) // 20 * 20))
    return np.minimum(4, (5 * s // n).astype(np.int64))


def run_tta_subprocess(model: str, stage: str, sids: list[str], device: str,
                       force: bool = False) -> Path:
    dump_dir = TTA_DUMP_ROOT / f"{model}_{stage}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    if not force and any(dump_dir.glob("*.npz")):
        print(f"[tta] skip {model} {stage}（已有 dump）")
        return dump_dir
    cmd = [sys.executable, str(Path(__file__).parent / "run_tta.py"),
           "--model", model, "--sessions", ",".join(sids),
           "--win-frac", "%s,%s" % STAGE_FRAC[stage],
           "--dump-kp", str(dump_dir), "--ckpt", "last",
           "--device", device,
           "--out", str(dump_dir / f"metrics_{model}_{stage}.json")]
    if stage == "Vi":
        cmd.append("--v-only")
    print("[tta]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))
    return dump_dir


def load_anysole_stage(model: str, stage: str, sid: str, n: int) -> np.ndarray:
    """AnySole-U 单阶段逐帧踝足 MPJPE（mm）：cell 预测或 TTA dump。"""
    if stage in ("V", "T", "VT"):
        cond = {"V": "V2M", "T": "T2M", "VT": "VT2M"}[stage]
        pred, _, vmask = load_pred_joints(f"{model}_s1", sid, cond)
        err = _ankle_foot_mpjpe(pred, sid, pelvis=(1, 2), n=n)
        return err, vmask[:n]
    dump_dir = TTA_DUMP_ROOT / f"{model}_{stage}"
    err = np.full(n, np.nan, dtype=np.float64)
    vmask = np.zeros(n, dtype=bool)
    for path in sorted(dump_dir.glob(f"{sid}_w*.npz")):
        d = np.load(path)
        st = int(path.stem.split("_w")[-1])
        seg = slice(st, min(st + len(d["joints"]), n))
        gt = _gt_native(sid)
        e = _ankle_foot_mpjpe(d["joints"], sid, pelvis=(1, 2),
                              gt=gt[st:st + len(d["joints"])])
        err[seg] = e[:seg.stop - seg.start]
        vmask[seg] = True
    return err, vmask


_gt_cache: dict[str, np.ndarray] = {}


def _gt_native(sid: str) -> np.ndarray:
    if sid not in _gt_cache:
        _gt_cache[sid] = load_gt_joints(sid)
    return _gt_cache[sid]


def _ankle_foot_mpjpe(pred: np.ndarray, sid: str, *, pelvis: tuple,
                      gt: np.ndarray | None = None, n: int | None = None
                      ) -> np.ndarray:
    """pred（native）→ 逐帧踝足 MPJPE（mm）。pelvis 对齐按各自骨架骨盆。"""
    g = _gt_native(sid) if gt is None else gt
    n = min(len(pred), len(g)) if n is None else min(n, len(pred), len(g))
    p = pred[:n] - pred[:n, pelvis].mean(axis=1, keepdims=True)
    q = g[:n] - g[:n, (1, 2)].mean(axis=1, keepdims=True)
    err = np.linalg.norm(p - q, axis=-1)  # (n,24)
    return err[:, SMPL_ANKLE_FOOT].mean(axis=1) * 1000.0


def load_baseline_stage(stage: str, sid: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """基线组合单阶段逐帧踝足 MPJPE（mm，SMPL-24 GT 同口径）。

    V = WHAM（smpl24 协议直出，2026-10-30 接入）；T = Step2Motion
    （smpl24 协议直出，2026-10-28 切换后与 MotionPRO 同口径）；
    VT = MotionPRO（smpl24 vs SMPL-24 GT，display→native 转换）；
    Vi/ViT = EasyMocap 标定多视角 n=4 四相机（2026-10-30 用户裁定接入；
    缓存关节已为 native y-up 世界系，与 GT 同基，无需 display 转换）。
    """
    if stage in ("Vi", "ViT"):
        d = np.load(EMC_ROOT / f"{sid}_n4_1234.npz")
        err = _ankle_foot_mpjpe(d["joints"], sid, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = True
        return err, vout
    if stage == "V":
        # WHAM：world_z_up 显示系 → native；同 MotionPRO 口径
        d = np.load(WHAM_ROOT / f"{sid}.npz", allow_pickle=True)
        joints = _display_to_native(d["joint_xyz_world"])   # (T,24,3) native
        vmask = d["valid_mask"]
        err = _ankle_foot_mpjpe(joints, sid, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = vmask[:len(err)]
        return err, vout
    if stage == "T":
        # Step2Motion 2026-10-28 已切 smpl24 协议直出（world_z_up/smpl24，
        # 与 MotionPRO 导出同构）；GT 同用 facts SMPL-24 FK
        d = np.load(S2M_ROOT / f"{sid}.npz", allow_pickle=True)
        joints = _display_to_native(d["joint_xyz_world"])   # (T,24,3) native
        vmask = d["valid_mask"]
        err = _ankle_foot_mpjpe(joints, sid, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = vmask[:len(err)]
        return err, vout
    d = np.load(MPRO_ROOT / f"{sid}.npz", allow_pickle=True)
    joints = _display_to_native(d["joint_xyz_world"])       # (T,24,3) native
    vmask = d["valid_mask"]
    err = _ankle_foot_mpjpe(joints, sid, pelvis=(1, 2), n=n)
    vout = np.zeros(n, dtype=bool)
    vout[:len(err)] = vmask[:len(err)]
    return err, vout


def _seg_sma(e: np.ndarray, w: int = 20) -> np.ndarray:
    """逐连续有效段滑动均值，NaN 段保持 NaN（防无数据段出现假曲线）。"""
    out = np.full_like(e, np.nan)
    idx = np.where(np.isfinite(e))[0]
    if not len(idx):
        return out
    splits = np.where(np.diff(idx) > 1)[0]
    for lo_i, hi_i in zip(np.concatenate([[0], splits + 1]),
                          np.concatenate([splits, [len(idx) - 1]])):
        lo, hi = idx[lo_i], idx[hi_i]
        seg = e[lo:hi + 1]
        out[lo:hi + 1] = (np.convolve(seg, np.ones(w) / w, mode="same")
                          if len(seg) >= w else seg)
    return out


def render_figure(action: str, sids: list[str],
                  curves: dict[str, dict[str, tuple]], out_png: Path) -> None:
    """每动作图：每会话一个子图，x=时间，AnySole 两模型曲线，
    背景 = 5 阶段带。图内英文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    ncols = 2
    nrows = max(1, (len(sids) + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows),
                             squeeze=False)
    # 阶段带 = 5 类色按时间轴顺序取连续色（已校验 CVD/对比度）；带内文字标名，
    # 身份不只靠颜色。曲线仍用 C0/C1/C2。
    stage_colors = {"V": "#2a78d6", "T": "#eb6834", "VT": "#1baf7a",
                    "Vi": "#eda100", "ViT": "#e87ba4"}
    stage_labels = {"V": "V (mono-cam)", "T": "T (tactile)",
                    "VT": "VT (cam+tactile)", "Vi": "Vi (multi-cam)",
                    "ViT": "ViT (multi-cam+tactile)"}
    for ax, sid in zip(axes.flat, sids):
        n = len(_gt_native(sid))
        t = np.arange(n) / FPS
        st = stage_of_frame(np.arange(n), n)
        ax.set_xlim(0, t[-1])
        # 阶段带边界 = stage_of_frame 实际窗口边界（与曲线分段同源 → 严格对齐）；
        # 带间留 1 帧缝使「分离的时间块」可读；带内短标（窄带可容纳），
        # 全名走图例
        for si, stage in enumerate(STAGES):
            sel = np.flatnonzero(st == si)
            t0 = sel[0] / FPS + (0.5 / FPS if si else 0)
            t1 = (sel[-1] + 1) / FPS - (0.5 / FPS if si < len(STAGES) - 1 else 0)
            ax.axvspan(t0, t1, color=stage_colors[stage], alpha=0.22, lw=0,
                       label=stage_labels[stage])
            ax.text((t0 + t1) / 2, 0.97, stage,
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color="#3c3b38",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])
        for ci, label in enumerate(curves[sid]):
            err, vmask = curves[sid][label]
            e = np.where(vmask, err, np.nan)
            # 逐阶段分段绘制：阶段边界处断线（各阶段 = 独立推理，输出不衔接）
            for si, stage in enumerate(STAGES):
                sel = st == si
                seg = e[sel]
                if not np.isfinite(seg).any():
                    continue
                t_seg = t[sel]
                ax.plot(t_seg, seg, color=f"C{ci}", alpha=0.25, lw=0.8,
                        label=label if si == 0 else None)
                ax.plot(t_seg, _seg_sma(seg), color=f"C{ci}", lw=1.6,
                        label=label if si == 0 else None)
        ax.set_title(sid)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ankle-foot MPJPE (mm)")
        ax.set_ylim(0, 400)
    for ax in axes.flat[len(sids):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7)
    fig.suptitle(f"E7 dynamic input availability — {action}", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default=",".join(SEQUENCES))
    ap.add_argument("--models", default=",".join(ANYSOLE_MODELS))
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--force-tta", action="store_true",
                    help="无视已有 TTA dump 重新生成")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    actions = [a for a in args.actions.split(",") if a.strip()]
    models = [m for m in args.models.split(",") if m.strip()]
    sids_all = sorted({s for a in actions for s in SEQUENCES[a]})
    # gate：AnySole 统计 cell 逐帧预测未产出时给可读报错（等队列 v7_train
    # confs 补跑 test 评估后重跑）
    missing = []
    for model in models:
        for sid in sids_all:
            for cond in ("V2M", "T2M", "VT2M"):
                p = CELL_ROOT / f"{model}_s1" / "predictions/eval_motion" / f"{sid}_{cond}.npz"
                if not p.is_file():
                    missing.append(str(p))
    if missing:
        raise SystemExit("[E7 gate] AnySole 统计预测未产出（等队列补跑 test 评估后重跑）：\n"
                         + "\n".join(missing[:6]))
    table: dict[str, dict] = {}
    curves: dict[str, dict[str, dict[str, tuple]]] = {}

    # AnySole-U 的 Vi/ViT 阶段 = TTA（先统一生成 dump；已有则跳过）
    for model in models:
        for stage in ("Vi", "ViT"):
            run_tta_subprocess(model, stage, sids_all, args.device,
                               force=args.force_tta)

    for action in actions:
        sids = SEQUENCES[action]
        stage_agg: dict[str, dict[str, list[float]]] = {
            stage: {} for stage in STAGES}
        for sid in sids:
            n = len(_gt_native(sid))
            st = stage_of_frame(np.arange(n), n)
            # 窗口覆盖掩码：最后一个完整窗口之后的无窗口尾巴不参与任何阶段
            covered = np.arange(n) < (((n - 20) // 20) * 20 + 20)
            curves.setdefault(sid, {})
            for model in models:
                per_frame = np.full(n, np.nan)
                per_valid = np.zeros(n, dtype=bool)
                for stage in STAGES:
                    e, v = load_anysole_stage(model, stage, sid, n)
                    sel = (st == STAGES.index(stage)) & covered
                    per_frame[sel] = e[sel]
                    per_valid[sel] = v[sel]
                    key = f"AnySole {model.replace('V7_C', 'C')}"
                    vals = e[sel][v[sel]]
                    if vals.size:
                        stage_agg[stage].setdefault(key, []).append(vals.mean())
                curves[sid][f"AnySole {model.replace('V7_C', 'C')}"] = \
                    (per_frame, per_valid)
            # 基线组合（V/Vi/ViT 暂无基线 → NaN 段由 _seg_sma 留白）
            per_frame = np.full(n, np.nan)
            per_valid = np.zeros(n, dtype=bool)
            for stage in STAGES:
                e, v = load_baseline_stage(stage, sid, n)
                sel = (st == STAGES.index(stage)) & covered
                per_frame[sel] = e[sel]
                per_valid[sel] = v[sel]
                vals = e[sel][v[sel]]
                if vals.size:
                    stage_agg[stage].setdefault("baseline_combo", []).append(vals.mean())
            curves[sid]["baseline_combo"] = (per_frame, per_valid)
        table[action] = {"sessions": sids,
                         "stages": {s: {k: float(np.mean(v))
                                        for k, v in stage_agg[s].items()}
                                    for s in STAGES}}
        if not args.no_figure:
            fig_no = {"walk": 1, "run": 2, "squat": 3}[action]
            render_figure(action, sids, curves,
                          Path(args.out_dir) /
                          f"Full_Fig{fig_no}_E7_Stages_{action}.png")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e7_table.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# E7 动态模态可用性下的统一运动重建\n",
             "口径：canonical test 子集（walk 1 / run 1 / squat 6，共 8 会话）；ckpt = ckpt_last（TTA 同步 --ckpt last）；"
             "指标 = 踝足 MPJPE（mm，骨盆对齐，跨骨架一致）；"
             "时间轴 = 序列均分 5 阶段（各 1/5，窗口边界切换）："
             "V（单相机）→ T（单触觉）→ VT（相机+触觉）→ Vi（多相机）→ "
             "ViT（多相机+触觉）。\n"]
    lines.append("| 系统 | %s |" % " | ".join("%s %s" % (s, STAGE_ZH[s])
                                             for s in STAGES))
    lines.append("|" + "---|" * (len(STAGES) + 1))
    systems = ["AnySole Cv4", "AnySole Cv5", "baseline_combo"]
    for action in actions:
        lines.append("")
        lines.append("## %s（%s）\n" % (action, "/".join(table[action]["sessions"])))
        for sys_name in systems:
            cells = []
            for stage in STAGES:
                v = table[action]["stages"][stage].get(sys_name)
                cells.append("—" if v is None else "%.1f" % v)
            lines.append("| %s | %s |" % (sys_name, " | ".join(cells)))

    lines.append("")
    lines.append("基线组合（外部基线，SMPL-24 GT 同口径）：V=WHAM、"
                 "T=Step2Motion、VT=MotionPRO 均 smpl24 协议直出 vs SMPL-24 GT；"
                 "Vi/ViT 暂无基线（—）。阶段间对比以各系统自身协议 GT 为基准。\n")
    (out_dir / "E7_动态输入重建.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    main()
