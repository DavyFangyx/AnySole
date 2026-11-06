"""E6 运动状态下的触觉增益与门控效果（实验计划 E6 口径）。

消费 V7_Cv4_s1（单门）/ V7_Cv5_s1（双门）/ V7_Cv7_s1（无门控，枚举）
训练 cell 的逐帧预测 + f6_soft 接触标签 + facts GT：
- 逐帧左右踝足误差（骨盆对齐，mm/帧/侧），Δ = VT 误差 − V 误差（负值 = 触觉有帮助；
  与 E2/E5 的 Δ(VT−V) 口径一致，2026-10-28 用户裁定）
- 相位状态机：稳定支撑/摆动/落脚附近/离地附近（±0.1s 近窗）
- 动作 = canonical test 全集（splits.csv，36 会话，按实际出现的动作分组，逐动作表）
- 产物：z_note/E_Test/E6_接触相位/{e6_table.json, E6_接触相位分析.md,
  e6_phase_{group}.png}（2026-11-02 起按指标分子文件夹：
  results_display/Etest/E6_ContactPhaseAnalysis/{ankle_foot_mpjpe,pa_mpjpe}/…）

用法（touch_gait env）：
  python anysole_tta/e6_phase_analysis.py --models V7_Cv4_s1,V7_Cv5_s1,V7_Cv7_s1
  python anysole_tta/e6_phase_analysis.py --models V7_Cv5_s1 --actions walk
  python anysole_tta/e6_phase_analysis.py --metric pa_mpjpe
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hmr2_gait 根（anysole_tta 命名空间包）
from hmr2_tta.paths import PROJECT_ROOT  # noqa: E402
from anysole_tta.phase_common import (CELL_ROOT, LEFT_ANKLE_FOOT,  # noqa: E402
                                      METRICS, PHASE_NAMES, PHASE_NAMES_ZH,
                                      RIGHT_ANKLE_FOOT,
                                      load_contact, load_gt_joints,
                                      load_pred_joints, metric_error,
                                      phase_labels)
from anysole.data.smpl_io import estimate_floor_y  # noqa: E402

OUT_BASE = PROJECT_ROOT / "results_display/Etest/E6_ContactPhaseAnalysis"
FPS = 40.0
CONDS = ("V2M", "VT2M")
DEFAULT_MODELS = ["V7_Cv4_s1", "V7_Cv5_s1", "V7_Cv7_s1"]
# 01..11 → 组名/中文名（走路=03/04、跑步=05/08、深蹲=06）
ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}
ACTION_NAMES_ZH = {"01": "站立重心移", "02": "单脚站立", "03": "折返慢走",
                   "04": "折返快走", "05": "绕圈跑", "06": "深蹲", "07": "开合跳",
                   "08": "折返跑", "09": "横移", "10": "坐下站起", "11": "上楼梯"}
GROUP_ZH = {"walk": "走路", "run": "跑步", "squat": "深蹲",
            "stand_shift": "站立重心移", "single_stand": "单脚站立",
            "jump_jack": "开合跳", "lateral": "横移", "sit_stand": "坐下站起",
            "stairs": "上楼梯"}

# 多指标相位表（2026-11-02 扩展）：位置误差之外，逐相位统计足垂直误差 /
# 地面穿透 / 接触准确率。G = VT − V；vert/pen 为负 = 触觉有帮助，
# contact 为正 = 触觉有帮助（方向随指标）。
METRICS_EXTRA = ("vert", "pen", "contact")


def foot_vertical_error_per_frame(pred: np.ndarray, gt: np.ndarray,
                                  foot_joints: tuple[int, ...]) -> np.ndarray:
    """骨盆对齐后逐帧 |Δy|（该脚 2 关节均值，mm/帧）。"""
    from anysole.utils.metrics import pelvis_align
    p, _ = pelvis_align(pred)
    g, _ = pelvis_align(gt)
    err_y = np.abs(p[..., 1] - g[..., 1])[:, foot_joints].mean(axis=1) * 1000.0
    return err_y


def foot_ground_pen_per_frame(pred: np.ndarray, floor: float,
                              foot_joints: tuple[int, ...]) -> np.ndarray:
    """逐帧地面穿透 = max(0, floor − 该脚踝足最低点)（mm/帧）。"""
    ys = pred[:, foot_joints, 1]
    return np.maximum(0.0, floor - ys.min(axis=1)) * 1000.0


def predicted_contact_per_frame(pred: np.ndarray, floor: float,
                                fps: float = 40.0) -> np.ndarray:
    """soft_contact_from_keypoints（losses.py）的 numpy 同口径镜像 →
    (T,2) bool 预测接触：足高与足速 soft 概率乘积 > 0.5。"""
    out = np.zeros((len(pred), 2), dtype=bool)
    for fi, joints in enumerate((LEFT_ANKLE_FOOT, RIGHT_ANKLE_FOOT)):
        foot = pred[:, joints, :]
        height = foot[..., 1].min(axis=1) - floor
        speed = np.zeros(len(pred), dtype=np.float64)
        if len(pred) > 1:
            speed = np.concatenate([
                np.linalg.norm(foot[1:] - foot[:-1], axis=-1).mean(axis=1)[:1],
                np.linalg.norm(foot[1:] - foot[:-1], axis=-1).mean(axis=1)]) * fps
        h_prob = 1.0 / (1.0 + np.exp(-np.clip((0.05 - height) / 0.02, -700, 700)))
        s_prob = 1.0 / (1.0 + np.exp(-np.clip((0.20 - speed) / 0.05, -700, 700)))
        out[:, fi] = (h_prob * s_prob) > 0.5
    return out


def build_session_sets() -> dict[str, list[str]]:
    """E6 评估会话按动作组归类：默认 canonical test 36 会话；环境变量 E6_SPLIT_CSV / E6_SPLIT_COLUMN
    可改读其他 split（如 splits_action_8020.csv 的 val 列，2026-11-04），不设即旧行为逐位不变。"""
    import os
    split_csv = os.environ.get("E6_SPLIT_CSV", "splits.csv")
    column = os.environ.get("E6_SPLIT_COLUMN", "test")
    test = []
    with open(PROJECT_ROOT / "AnysoleWorkspace/protocol/splits/default" / split_csv) as f:
        for row in csv.DictReader(f):
            if row[column]:
                test.append(row[column])
    sets: dict[str, list[str]] = {}
    for sid in sorted(test):
        a = _action_of(sid)
        sets.setdefault(ACTION_GROUPS.get(a, a), []).append(sid)
    return sets


def _action_of(sid: str) -> str:
    with open(PROJECT_ROOT / "AnysoleWorkspace/protocol/manifests/"
              "session_manifest.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["session_id"] == sid:
                return row["action"]
    raise KeyError(sid)


def require_cells(model: str) -> list[str]:
    missing = []
    for sids in build_session_sets().values():
        for sid in sids:
            for cond in CONDS:
                p = CELL_ROOT / model / "predictions/eval_motion" / f"{sid}_{cond}.npz"
                if not p.exists():
                    missing.append(str(p))
    return missing


_gt_cache: dict[str, np.ndarray] = {}


def load_session(model: str, sid: str, metric: str = "ankle_foot_mpjpe") -> dict:
    """逐帧 per 脚 V/VT 误差与相位 + 多指标扩展（垂直误差/穿透/预测接触）。

    metric = ankle_foot_mpjpe：逐帧踝足 MPJPE（mm，左右脚各自系列）；
    metric = pa_mpjpe：逐帧全关节相似对齐 MPJPE（mm；左右系列为同一全身
    误差，仅相位条件不同——表/图模式与 ankle_foot_mpjpe 相同）。
    """
    if metric not in METRICS:
        raise ValueError("unknown metric: " + metric)
    gt = _gt_cache.get(sid)
    if gt is None:
        gt = load_gt_joints(sid)
        _gt_cache[sid] = gt
    contact, contact_ids, label_valid = load_contact(sid)
    if not np.array_equal(contact_ids, np.arange(len(gt))):
        raise ValueError("Contact/GT frame grid mismatch: " + sid)
    phases = phase_labels(contact, fps=FPS)
    floor = float(estimate_floor_y(gt))
    out = {"phase": phases, "valid": label_valid.copy(),
           "floor": floor, "contact_gt": contact > 0.5}
    for cond in CONDS:
        pred, ids, valid = load_pred_joints(model, sid, cond)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != len(pred) or len(valid) != len(pred) or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(gt)):
            raise ValueError("Prediction frame grid mismatch: " + model + "/" + sid)
        aligned = np.full_like(gt, np.nan)
        aligned[ids] = pred
        frame_valid = np.zeros(len(gt), dtype=bool)
        frame_valid[ids] = valid
        frame_valid &= np.isfinite(aligned).all(axis=(1, 2))
        out["valid"] &= frame_valid
        for series in ("left", "right"):
            out[cond + "_" + series] = metric_error(aligned, gt, metric, series)
        out[cond + "_vert"] = np.stack([
            foot_vertical_error_per_frame(aligned, gt, LEFT_ANKLE_FOOT),
            foot_vertical_error_per_frame(aligned, gt, RIGHT_ANKLE_FOOT)],
            axis=1)  # (T,2) mm
        out[cond + "_pen"] = np.stack([
            foot_ground_pen_per_frame(aligned, floor, LEFT_ANKLE_FOOT),
            foot_ground_pen_per_frame(aligned, floor, RIGHT_ANKLE_FOOT)],
            axis=1)  # (T,2) mm
        out[cond + "_cpred"] = predicted_contact_per_frame(aligned, floor)
    return out


def aggregate(sessions: dict[str, dict]) -> dict:
    """{sid: data} → {foot: {phase: {V, VT, G}}}（mm，均值）。"""
    result = {}
    for foot in ("left", "right"):
        acc = {ph: {"V": [], "VT": [], "G": []} for ph in range(4)}
        for data in sessions.values():
            v = data[f"V2M_{foot}"][:len(data["phase"])]
            vt = data[f"VT2M_{foot}"][:len(data["phase"])]
            valid = data["valid"][:len(data["phase"])]
            for ph in range(4):
                sel = (data["phase"][:, 0 if foot == "left" else 1] == ph) & valid
                if sel.any():
                    acc[ph]["V"].append(v[sel].mean())
                    acc[ph]["VT"].append(vt[sel].mean())
                    acc[ph]["G"].append((vt[sel] - v[sel]).mean())
        result[foot] = {ph: ({k: float(np.mean(vals))
                              for k, vals in acc[ph].items()}
                             if acc[ph]["V"] else None)
                        for ph in range(4)}
    return result


def aggregate_extra(sessions: dict[str, dict]) -> dict:
    """{sid: data} → {foot: {phase: {vert/pen/contact: {V,VT,G}|None}}}。
    垂直误差/地面穿透 mm、接触准确率 %；会话均值再跨会话平均。
    G = VT − V（vert/pen 为负 = 触觉有帮助，contact 为正 = 有帮助）。"""
    result = {}
    for foot in ("left", "right"):
        fi = 0 if foot == "left" else 1
        acc = {ph: {m: {"V": [], "VT": [], "G": []} for m in METRICS_EXTRA}
               for ph in range(4)}
        for data in sessions.values():
            n = len(data["phase"])
            valid = data["valid"][:n]
            phase = data["phase"][:n, fi]
            for ph in range(4):
                sel = (phase == ph) & valid
                if not sel.any():
                    continue
                for m in ("vert", "pen"):
                    v = data["V2M_" + m][:n][sel, fi]
                    vt = data["VT2M_" + m][:n][sel, fi]
                    acc[ph][m]["V"].append(v.mean())
                    acc[ph][m]["VT"].append(vt.mean())
                    acc[ph][m]["G"].append((vt - v).mean())
                gt = data["contact_gt"][:n][sel, fi]
                cv = float((data["V2M_cpred"][:n][sel, fi] == gt).mean() * 100.0)
                cvt = float((data["VT2M_cpred"][:n][sel, fi] == gt).mean() * 100.0)
                acc[ph]["contact"]["V"].append(cv)
                acc[ph]["contact"]["VT"].append(cvt)
                acc[ph]["contact"]["G"].append(cvt - cv)
        result[foot] = {
            ph: ({m: {k: float(np.mean(vals)) for k, vals in acc[ph][m].items()}
                  for m in METRICS_EXTRA} if acc[ph]["vert"]["V"] else None)
            for ph in range(4)}
    return result


def render_table(agg: dict[str, dict[str, dict]]) -> str:
    """agg = {model: {foot: {phase: {V,VT,G}|None}}} → md 表。"""
    lines = ["| 模型 | 脚 | 稳定支撑 | 摆动 | 落脚附近 | 离地附近 |",
             "|---|---|---|---|---|---|"]
    for model in agg:
        for foot in ("left", "right"):
            cells = []
            for ph in range(4):
                d = agg[model][foot][ph]
                cells.append("—" if d is None else
                             "%.1f / %.1f / %+.1f" % (d["V"], d["VT"], d["G"]))
            lines.append("| %s | %s | %s |" % (model, foot, " | ".join(cells)))
    return "\n".join(lines)


def render_figure(action: str, sids: list[str], data: dict[str, dict[str, dict]],
                  out_dir: Path, metric: str = "ankle_foot_mpjpe") -> None:
    """每动作 × 每会话一张图（2026-10-31 用户裁定：1×2 上下分布，上=左脚、
    下=右脚，左右脚分开不合并不平均）：每面板 x=时间、y=该脚 Δ = VT − V、
    各模型曲线 + 该脚相位带。图内英文（项目口径）。

    metric = pa_mpjpe：曲线为全身逐帧 PA-MPJPE Δ（左右面板同一曲线），
    面板标题注明相位条件脚。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(next(iter(next(iter(data.values())).values())))
    phase_colors = {0: "#d9d9d9", 1: "#ffffff", 2: "#f4a582", 3: "#92c5de"}
    panel_titles = {"ankle_foot_mpjpe": {"left": "left foot", "right": "right foot"},
                    "pa_mpjpe": {"left": "full body — left-foot phases",
                                 "right": "full body — right-foot phases"}}
    ylim = {"ankle_foot_mpjpe": (-60, 60), "pa_mpjpe": (-30, 30)}[metric]
    for sid in sids:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                                 squeeze=True)
        d = data[sid]
        any_data = next(iter(d.values()))
        t = np.arange(len(any_data["phase"])) / FPS
        for foot_i, (ax, foot) in enumerate(zip(axes, ("left", "right"))):
            ax.set_xlim(0, t[-1])
            ph = any_data["phase"][:, foot_i]
            for k in range(4):
                sel = ph == k
                if sel.any():
                    boundaries = np.flatnonzero(np.diff(np.r_[False, sel, False].astype(np.int8)))
                    for number, (start, stop) in enumerate(zip(boundaries[::2], boundaries[1::2])):
                        ax.axvspan(t[start], t[stop - 1] + 1.0 / FPS, ymin=0.0, ymax=0.08,
                                   color=phase_colors[k], alpha=0.6, lw=0,
                                   label=PHASE_NAMES[k] if number == 0 else None)
            for mi, (model, mdata) in enumerate(d.items()):
                g = mdata[f"VT2M_{foot}"] - mdata[f"V2M_{foot}"]
                w = 20
                sma = np.convolve(np.nan_to_num(g), np.ones(w) / w, mode="same")
                ax.plot(t, g, color=f"C{mi}", alpha=0.25, lw=0.8)
                ax.plot(t, sma, color=f"C{mi}", lw=1.6,
                        label=model if foot_i == 0 else None)
            ax.axhline(0, color="k", lw=0.6, ls="--")
            ax.set_title(panel_titles[metric][foot], fontsize=10)
            ax.set_ylabel("Δ = VT - V (mm)")
            ax.set_ylim(*ylim)
        axes[-1].set_xlabel("time (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(7, n_models),
                   fontsize=8, frameon=False)
        fig.suptitle(f"E6 tactile gain by contact phase — {action} — {sid}",
                     y=0.99)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"Full_Fig_{action}_{sid}.png", dpi=150)
        plt.close(fig)



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--ckpt", choices=("last", "best"), default="last",
                    help="ckpt 口径（best 口径 stat cells 未注册，暂只支持 last）")
    ap.add_argument("--actions", default=None,
                    help="动作组逗号分隔（缺省 = canonical test 全部动作组）")
    ap.add_argument("--metric", choices=METRICS, default="ankle_foot_mpjpe",
                    help="位置误差指标（2026-11-02 裁定：E6 按指标建子文件夹；"
                         "缺省 ankle_foot_mpjpe = 现行踝足口径）")
    ap.add_argument("--allow-partial", action="store_true",
                    help="跳过缺评估产物的模型（默认硬报错）")
    ap.add_argument("--skip-missing-sessions", action="store_true",
                    help="按 split 建会话集后，剔除此批模型没有预测的会话"
                         "（eval 侧已排除的会话，如 S12102；默认硬报错）")
    ap.add_argument("--contact-method", default="f6_soft_time_accel_state_smpl24_v1",
                    choices=("f6_soft", "f6_soft_time_accel_v1", "f6_soft_time_accel_state_v2",
                             "f6_soft_time_accel_state_smpl24_v1"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(OUT_BASE / args.metric)
    if args.ckpt != "last":
        raise SystemExit("E6 目前只支持 --ckpt last（best 口径 stat cells 未注册）")

    from anysole_tta import phase_common
    phase_common.CONTACT_METHOD = args.contact_method
    if args.contact_method != "f6_soft_time_accel_state_smpl24_v1" and Path(args.out_dir) == OUT_BASE / args.metric:
        args.out_dir = str(OUT_BASE / (args.metric + "_" + args.contact_method))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        if not (CELL_ROOT / model).is_dir():
            raise SystemExit(
                f"[E6 gate] cell 不存在：{CELL_ROOT / model}\n"
                f"模型名须为 canonical（如 {', '.join(DEFAULT_MODELS)}）")
        missing = require_cells(model)
        if missing:
            msg = (f"[E6 gate] {model} 缺评估产物：\n" + "\n".join(missing[:5]))
            if args.allow_partial:
                print(msg)
                models.remove(model)
            elif not args.skip_missing_sessions:
                raise SystemExit(msg)
    if not models:
        raise SystemExit("no model with complete eval products")

    session_sets = build_session_sets()
    if args.skip_missing_sessions:
        pruned = {}
        for action, sids in session_sets.items():
            keep = [s for s in sids
                    if all((CELL_ROOT / m / "predictions/eval_motion" /
                            ("%s_%s.npz" % (s, c))).is_file()
                           for m in models for c in CONDS)]
            if len(keep) != len(sids):
                print("[E6] %s: 剔除 %d 个无预测会话 %s" % (
                    action, len(sids) - len(keep),
                    sorted(set(sids) - set(keep))))
            if keep:
                pruned[action] = keep
        session_sets = pruned
    actions = ([a for a in args.actions.split(",") if a.strip()]
               if args.actions else list(session_sets))
    unknown = [a for a in actions if a not in session_sets]
    if unknown:
        raise SystemExit(f"unknown actions: {unknown}（可选：{list(session_sets)}）")

    table = {}
    metrics_table = {}
    for action in actions:
        sids = session_sets[action]
        data = {sid: {} for sid in sids}
        for model in models:
            for sid in sids:
                data[sid][model] = load_session(model, sid, args.metric)
        agg = {model: aggregate({sid: data[sid][model] for sid in sids})
               for model in models}
        agg_extra = {model: aggregate_extra({sid: data[sid][model]
                                             for sid in sids})
                     for model in models}
        table[action] = {"sessions": sids, "table": agg}
        metrics_table[action] = {"sessions": sids, "table": agg_extra}
        if not args.no_figure:
            render_figure(action, sids, data, Path(args.out_dir) / "detail" / "per_session",
                          args.metric)
        print(f"[{action}] sessions={len(sids)}")
        for model in models:
            for foot in ("left", "right"):
                row = agg[model][foot]
                cells = []
                for ph in range(4):
                    if row[ph] is None:
                        cells.append("%s —" % PHASE_NAMES_ZH[ph])
                    else:
                        cells.append("%s V%.1f/VT%.1f/Δ%+.1f" % (
                            PHASE_NAMES_ZH[ph], row[ph]["V"], row[ph]["VT"],
                            row[ph]["G"]))
                print(f"  {model} {foot}: " + " | ".join(cells))
                row2 = agg_extra[model][foot]
                cells2 = []
                for ph in range(4):
                    d = row2[ph]
                    cells2.append("%s —" % PHASE_NAMES_ZH[ph] if d is None
                                  else "%s 接触%.0f/%.0f/Δ%+.0f%%" % (
                                      PHASE_NAMES_ZH[ph], d["contact"]["V"],
                                      d["contact"]["VT"], d["contact"]["G"]))
                print(f"  {model} {foot} contact: " + " | ".join(cells2))

    out_dir = Path(args.out_dir)
    detail_dir = out_dir / "detail"
    supp_dir = out_dir / "supplementary"
    (detail_dir / "per_session").mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    (detail_dir / "e6_table.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    (detail_dir / "e6_table_metrics.json").write_text(
        json.dumps(metrics_table, indent=1, ensure_ascii=False), encoding="utf-8")
    metric_desc = {"ankle_foot_mpjpe": "踝足 MPJPE，骨盆对齐",
                   "pa_mpjpe": "全身逐帧相似变换对齐 PA-MPJPE（项目 pa_mpjpe 口径，"
                              "含尺度；左右行 = 同一全身误差分别按左/右脚相位条件统计）"}[args.metric]
    md_lines = ["# E6 接触相位下的触觉增益与门控效果\n",
                "口径：canonical test（splits.csv，36 会话，按实际出现的动作分组）；ckpt = ckpt_last；"
                "模型 = V7_Cv4（单门）/ V7_Cv5（双门）/ V7_Cv7（无门控，枚举）；"
                "Δ = VT 误差 − V 误差（%s，mm；负值 = 触觉有帮助，"
                "与 E2/E5 的 Δ(VT−V) 口径一致）；"
                "相位 = f6_soft 接触标签 ±0.1s 近窗。\n" % metric_desc]
    for action in actions:
        sids = table[action]["sessions"]
        label = f"{GROUP_ZH.get(action, action)} {action}（{'/'.join(sids)}）"
        md_lines.append(f"## {label}\n")
        md_lines.append(render_table(table[action]["table"]))
        md_lines.append("")
    if "walk" in actions or "run" in actions:
        md_lines.append("表注：走路/跑步在 canonical test 各仅 1 条会话，样本量小，"
                        "数字作观察不作统计结论。\n")
    md_lines.append("## 多指标相位表（2026-11-02 扩展）\n")
    md_lines.append(
        "除踝足位置误差外，逐相位同时统计三组指标（数字见 "
        "Full_Table2_E6_PhaseMetrics.csv，由 e6_table_metrics.json 导出）：\n"
        "- 足垂直误差：骨盆对齐 |Δy|（该脚 2 关节均值，mm）；\n"
        "- 地面穿透：max(0, floor − 该脚踝足最低点)，floor = GT 全关节 y 5 分位"
        "（estimate_floor_y 同口径，mm）；\n"
        "- 足高判据接触准确率：预测接触 = 足高/足速 soft 概率乘积 > 0.5"
        "（soft_contact_from_keypoints 同口径），对 f6_soft 标签（%）。"
        "注意 cell 级 test.json 的 VT/T 接触指标用的是模型触觉接触头"
        "（V41ContactHead，未逐帧落盘），本表 V/VT 统一用足高判据，"
        "两者口径不同不可直接比较。\n"
        "G = VT − V：垂直误差/地面穿透为负 = 触觉有帮助，接触准确率为正 = "
        "触觉有帮助（方向随指标）。会话均值再跨会话平均。\n")
    (detail_dir / "E6_接触相位分析.md").write_text(
        "\n".join(md_lines), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    main()
