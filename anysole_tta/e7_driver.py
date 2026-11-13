"""E7 动态模态可用性下的统一运动重建（实验计划 E7 口径，2026-10-27 用户裁定）。

时间轴：连续序列均分 5 阶段（各 1/5 时长，窗口边界切换）——
  V（单相机）→ T（单触觉）→ VT（相机+触觉）→ Vi（多相机）→ ViT（多相机+触觉）

两条线（同一序列、同一阶段划分）：
- AnySole-U：V7_Cv5（主模型，2026-11-01 用户裁定 E7 只留 Cv5，与 E3
  TTA 线同口径）——V/T/VT 阶段直接读 {model}_stat cell 逐帧预测
  （V2M/T2M/VT2M，ckpt_last）；Vi/ViT 阶段读 E3 全量 TTA 产物树
  （predictions/anysole_tta_stage1_{model}_test/：Vi = ③ 线组合
  1,2,3,4、ViT = ④ 线组合 1,2,3,4_contact；2026-11-01 用户裁定：
  E3 全量运行、E7 只读对应时间段，E7 不调度 TTA）。
- 基线组合（外部基线，SMPL-24 GT 同口径）：V 阶段 = WHAM（smpl24 协议
  直出，2026-10-30 接入）；T 阶段 = Step2Motion（smpl24 协议直出，
  2026-10-28 切换，与 MotionPRO 导出同构）；VT 阶段 = MotionPRO
  （smpl24 vs SMPL-24 GT）；Vi/ViT 阶段 = EasyMocap 标定多视角基线
  （n=4 四相机，2026-10-30 用户裁定接入——多视角参考线，与 E3 表 4 线 1
  同源，easymocap_baseline.py 缓存）。

阶段划分 = 窗口对齐（stride20 窗口网格，与 run_tta --win-frac 选窗规则
一致；阶段边界不落窗口中部，无缺口）。

指标：位置误差逐帧 mm（--metric：ankle_foot_mpjpe = 踝足 MPJPE 骨盆对齐；
pa_mpjpe = 全身逐帧相似对齐，项目口径）。产物（2026-11-02 起按指标子文件夹）：
results_display/Etest/E7_DynamicModalityInputs/{ankle_foot_mpjpe,pa_mpjpe}/{e7_table.json,
E7_动态输入重建.md, Full_FigN_E7_Stages_{action}.png}

动作 = canonical test 全集（splits.csv，36 会话，按动作分组，与 E6 同口径）。

用法（touch_gait env）：
  python anysole_tta/e7_driver.py --models V7_Cv5 --actions walk   # 冒烟
  python anysole_tta/e7_driver.py --models V7_Cv5                 # test 全部动作组
  python anysole_tta/e7_driver.py --metric pa_mpjpe
（Vi/ViT 直接读 E3 全量 TTA 产物树，E7 不调度 TTA）"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import PROJECT_ROOT, RESULTS_ROOT  # noqa: E402
from anysole_tta.phase_common import (CELL_ROOT, METRICS, action_of,  # noqa: E402
                                      load_gt_joints, load_pred_joints)

OUT_BASE = PROJECT_ROOT / "results_display/Etest/E7_DynamicModalityInputs"
FPS = 40.0
STAGES = ("V", "T", "VT", "Vi", "ViT")
STAGE_ZH = {"V": "单相机", "T": "单触觉", "VT": "相机+触觉",
            "Vi": "多相机", "ViT": "多相机+触觉"}
STAGE_FRAC = {s: (i / 5, (i + 1) / 5) for i, s in enumerate(STAGES)}
ANYSOLE_MODELS = ["V7_Cv5"]   # run_tta 模型名；cell = {m}_s1（E7 只留 Cv5）
# 动作分组（与 E6 同口径：canonical test 36 会话按 manifest action 分组）
ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}
GROUP_ZH = {"walk": "走路", "run": "跑步", "squat": "深蹲",
            "stand_shift": "站立重心移", "single_stand": "单脚站立",
            "jump_jack": "开合跳", "lateral": "横移", "sit_stand": "坐下站起",
            "stairs": "上楼梯"}


def build_session_sets() -> dict[str, list[str]]:
    """canonical test 36 会话按动作组归类（splits.csv，与 E6 同口径）。"""
    import csv
    test = []
    with open(PROJECT_ROOT / "AnysoleWorkspace/protocol/splits/default/"
              "splits.csv") as f:
        for row in csv.DictReader(f):
            if row["test"]:
                test.append(row["test"])
    sets: dict[str, list[str]] = {}
    for sid in sorted(test):
        a = action_of(sid)
        sets.setdefault(ACTION_GROUPS.get(a, a), []).append(sid)
    return sets
TTA_DUMP_ROOT = RESULTS_ROOT / "predictions"   # E3 全量 TTA 产物树（E7 直接读）
# run_tta 逐组合 dump 子目录（combo_key + _contact 后缀，同 run_tta 口径）
DUMP_SUB = {"Vi": "1,2,3,4", "ViT": "1,2,3,4_contact"}
SMPL_ANKLE_FOOT = (7, 8, 10, 11)
S2M_ROOT = PROJECT_ROOT / "results/baselines/Step2Motion/predictions/gait_model_smpl24/eval_motion"
# 2026-11-13: MotionPRO v4 AnySole-bbox held-out predictions (canonical test 36:
# S13 -> fS5S11S12 ckpt, S14 -> fS6S7 ckpt). Replaces the retired carpet_v2 tree.
MPRO_ROOT = PROJECT_ROOT / "results/baselines/MotionPRO/predictions/carpet_v4_anysolebbox_heldout/eval_motion"
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


def load_anysole_stage(model: str, stage: str, sid: str, n: int,
                       metric: str = "ankle_foot_mpjpe") -> np.ndarray:
    """AnySole-U 单阶段逐帧位置误差（mm）：cell 预测或 TTA dump。"""
    if stage in ("V", "T", "VT"):
        cond = {"V": "V2M", "T": "T2M", "VT": "VT2M"}[stage]
        pred, _, vmask = load_pred_joints(f"{model}_s1", sid, cond)
        err = _frame_error(pred, sid, metric=metric, pelvis=(1, 2), n=n)
        return err, vmask[:n]
    dump_dir = TTA_DUMP_ROOT / f"anysole_tta_stage1_{model}_test" / DUMP_SUB[stage]
    err = np.full(n, np.nan, dtype=np.float64)
    vmask = np.zeros(n, dtype=bool)
    for path in sorted(dump_dir.glob(f"{sid}_w*.npz")):
        d = np.load(path)
        st = int(path.stem.split("_w")[-1])
        seg = slice(st, min(st + len(d["joints"]), n))
        gt = _gt_native(sid)
        e = _frame_error(d["joints"], sid, metric=metric, pelvis=(1, 2),
                         gt=gt[st:st + len(d["joints"])])
        err[seg] = e[:seg.stop - seg.start]
        vmask[seg] = True
    return err, vmask


_gt_cache: dict[str, np.ndarray] = {}


def _gt_native(sid: str) -> np.ndarray:
    if sid not in _gt_cache:
        _gt_cache[sid] = load_gt_joints(sid)
    return _gt_cache[sid]


def _frame_error(pred: np.ndarray, sid: str, *, metric: str, pelvis: tuple,
                 gt: np.ndarray | None = None, n: int | None = None
                 ) -> np.ndarray:
    """pred（native）→ 逐帧位置误差（mm）。

    metric = ankle_foot_mpjpe：踝足 MPJPE（骨盆对齐按各自骨架骨盆）；
    metric = pa_mpjpe：全身逐帧相似对齐（项目 pa_mpjpe 口径，含尺度）。
    """
    g = _gt_native(sid) if gt is None else gt
    n = min(len(pred), len(g)) if n is None else min(n, len(pred), len(g))
    p, q = pred[:n], g[:n]
    if metric == "pa_mpjpe":
        from anysole_tta.phase_common import pa_mpjpe_per_frame_safe
        return pa_mpjpe_per_frame_safe(p, q)
    p = p - p[:, pelvis].mean(axis=1, keepdims=True)
    q = q - q[:, (1, 2)].mean(axis=1, keepdims=True)
    err = np.linalg.norm(p - q, axis=-1)  # (n,24)
    return err[:, SMPL_ANKLE_FOOT].mean(axis=1) * 1000.0


def load_baseline_stage(stage: str, sid: str, n: int,
                        metric: str = "ankle_foot_mpjpe") -> tuple[np.ndarray, np.ndarray]:
    """基线组合单阶段逐帧位置误差（mm，SMPL-24 GT 同口径）。

    V = WHAM（smpl24 协议直出，2026-10-30 接入）；T = Step2Motion
    （smpl24 协议直出，2026-10-28 切换后与 MotionPRO 同口径）；
    VT = MotionPRO（smpl24 vs SMPL-24 GT，display→native 转换）；
    Vi/ViT = EasyMocap 标定多视角 n=4 四相机（2026-10-30 用户裁定接入；
    缓存关节已为 native y-up 世界系，与 GT 同基，无需 display 转换）。
    """
    if stage in ("Vi", "ViT"):
        d = np.load(EMC_ROOT / f"{sid}_n4_1234.npz")
        err = _frame_error(d["joints"], sid, metric=metric, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = True
        return err, vout
    if stage == "V":
        # WHAM：world_z_up 显示系 → native；同 MotionPRO 口径
        d = np.load(WHAM_ROOT / f"{sid}.npz", allow_pickle=True)
        joints = _display_to_native(d["joint_xyz_world"])   # (T,24,3) native
        vmask = d["valid_mask"]
        err = _frame_error(joints, sid, metric=metric, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = vmask[:len(err)]
        return err, vout
    if stage == "T":
        # Step2Motion 2026-10-28 已切 smpl24 协议直出（world_z_up/smpl24，
        # 与 MotionPRO 导出同构）；GT 同用 facts SMPL-24 FK
        d = np.load(S2M_ROOT / f"{sid}.npz", allow_pickle=True)
        joints = _display_to_native(d["joint_xyz_world"])   # (T,24,3) native
        vmask = d["valid_mask"]
        err = _frame_error(joints, sid, metric=metric, pelvis=(1, 2), n=n)
        vout = np.zeros(n, dtype=bool)
        vout[:len(err)] = vmask[:len(err)]
        return err, vout
    d = np.load(MPRO_ROOT / f"{sid}.npz", allow_pickle=True)
    joints = _display_to_native(d["joint_xyz_world"])       # (T,24,3) native
    vmask = d["valid_mask"]
    err = _frame_error(joints, sid, metric=metric, pelvis=(1, 2), n=n)
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
                  curves: dict[str, dict[str, tuple]], out_png: Path,
                  metric: str = "ankle_foot_mpjpe") -> None:
    """每动作图：每会话一个子图，x=时间，AnySole 两模型曲线，
    背景 = 5 阶段带。图内英文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    ylabel = {"ankle_foot_mpjpe": "ankle-foot MPJPE (mm)",
              "pa_mpjpe": "PA-MPJPE (mm)"}[metric]
    ylim = {"ankle_foot_mpjpe": (0, 400), "pa_mpjpe": (0, 150)}[metric]
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
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
    for ax in axes.flat[len(sids):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7)
    fig.suptitle(f"E7 dynamic input availability — {action} ({metric})", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_best_sections(tiles, out_png):
    """单面板 tiles 原样拼贴 3×3 合并图（2026-11-01 用户裁定：一动作一面板，
    子图不重绘直接搬原图）；无胜出会话的动作红框标记。"""
    from PIL import Image, ImageDraw

    imgs = []
    for path, red in tiles:
        if not path.is_file():
            raise FileNotFoundError(f"missing panel figure: {path}")
        imgs.append((Image.open(path).convert("RGB"), red))
    w, h = imgs[0][0].size
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    gap = 12
    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap,
                               rows * h + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, red) in enumerate(imgs):
        r, c = divmod(i, cols)
        x = gap + c * (w + gap)
        y = gap + r * (h + gap)
        canvas.paste(img, (x, y))
        if red:
            draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2],
                           outline="red", width=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print("[png] %s" % out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default=None,
                    help="动作组逗号分隔（缺省 = canonical test 全部动作组）")
    ap.add_argument("--models", default=",".join(ANYSOLE_MODELS))
    ap.add_argument("--metric", choices=METRICS, default="ankle_foot_mpjpe",
                    help="位置误差指标（2026-11-02 裁定：E7 按指标建子文件夹；"
                         "缺省 ankle_foot_mpjpe = 现行踝足口径）")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(OUT_BASE / args.metric)

    session_sets = build_session_sets()
    actions = ([a for a in args.actions.split(",") if a.strip()]
               if args.actions else list(session_sets))
    unknown = [a for a in actions if a not in session_sets]
    if unknown:
        raise SystemExit(f"unknown actions: {unknown}（可选：{list(session_sets)}）")
    models = [m for m in args.models.split(",") if m.strip()]
    one_panels: dict[str, tuple[str, bool]] = {}
    sids_all = sorted({s for a in actions for s in session_sets[a]})
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
    missing_bl = []
    for sid in sids_all:
        for root, name in ((WHAM_ROOT, f"{sid}.npz"), (S2M_ROOT, f"{sid}.npz"),
                           (MPRO_ROOT, f"{sid}.npz"),
                           (EMC_ROOT, f"{sid}_n4_1234.npz")):
            if not (Path(root) / name).is_file():
                missing_bl.append(str(Path(root) / name))
    if missing_bl:
        raise SystemExit("[E7 gate] 外部基线产物缺失：\n"
                         + "\n".join(missing_bl[:6]))
    table: dict[str, dict] = {}
    curves: dict[str, dict[str, dict[str, tuple]]] = {}

    # Vi/ViT = 读 E3 全量 TTA 产物（③/④ 落盘树）；缺失 → 该段 NaN（不阻断）
    for model in models:
        for stage in ("Vi", "ViT"):
            d = TTA_DUMP_ROOT / f"anysole_tta_stage1_{model}_test" / DUMP_SUB[stage]
            n_npz = len(list(d.glob("*.npz"))) if d.is_dir() else 0
            if n_npz == 0:
                print(f"[E7 warn] {model} {stage} dump 缺失（{d}）——该段将为空窗")

    for action in actions:
        sids = session_sets[action]
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
                    e, v = load_anysole_stage(model, stage, sid, n, args.metric)
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
                e, v = load_baseline_stage(stage, sid, n, args.metric)
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
        n_sess = len(sids)
        scores = np.full(n_sess, -np.inf)
        for si in range(n_sess):
            parts = []
            for s in STAGES:
                c = stage_agg[s].get("AnySole Cv5")
                b = stage_agg[s].get("baseline_combo")
                if c is not None and b is not None and si < len(c) and si < len(b):
                    parts.append(b[si] - c[si])
            scores[si] = float(np.mean(parts)) if parts else -np.inf
        best_si = int(np.argmax(scores))
        one_panels[action] = (sids[best_si], not (scores[best_si] > 0))
        print(f"[{action}] sessions={len(sids)}")
        for sys_name in ("AnySole Cv5", "baseline_combo"):
            cells = []
            for s in STAGES:
                v = table[action]["stages"][s].get(sys_name)
                cells.append("—" if v is None else "%.1f" % v)
            print("  %-15s %s" % (sys_name, " | ".join(
                "%s %s" % (s, c) for s, c in zip(STAGES, cells))))
        if not args.no_figure:
            fig_no = actions.index(action) + 1
            fig_path = (Path(args.out_dir) / "supplementary" /
                        f"Full_Fig{fig_no}_E7_Stages_{action}.png")
            render_figure(action, sids, curves, fig_path, args.metric)
            print("[png] %s" % fig_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figure:
        tiles = []
        for action in actions:
            best_sid, red = one_panels[action]
            tmp = out_dir / f"_one_{action}.png"
            render_figure(action, [best_sid], curves, tmp, args.metric)
            tiles.append((tmp, red))
        render_best_sections(tiles, out_dir / "Full_Fig10_E7_BestSections.png")
        for tmp, _ in tiles:
            tmp.unlink()
    (out_dir / "detail").mkdir(parents=True, exist_ok=True)
    (out_dir / "detail" / "e7_table.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    main()
