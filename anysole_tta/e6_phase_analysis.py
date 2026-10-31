"""E6 运动状态下的触觉增益与门控效果（实验计划 E6 口径）。

消费 V7_Cv4_s1（单门）/ V7_Cv5_s1（双门）/ V7_Cv7_s1（无门控，枚举）
训练 cell 的逐帧预测 + f6_soft 接触标签 + facts GT：
- 逐帧左右踝足误差（骨盆对齐，mm/帧/侧），Δ = VT 误差 − V 误差（负值 = 触觉有帮助；
  与 E2/E5 的 Δ(VT−V) 口径一致，2026-10-28 用户裁定）
- 相位状态机：稳定支撑/摆动/落脚附近/离地附近（±0.1s 近窗）
- 动作 = canonical test 全集（splits.csv，36 会话，按实际出现的动作分组，逐动作表）
- 产物：z_note/E_Test/E6_接触相位/{e6_table.json, E6_接触相位分析.md,
  e6_phase_{group}.png}

用法（touch_gait env）：
  python anysole_tta/e6_phase_analysis.py --models V7_Cv4_s1,V7_Cv5_s1,V7_Cv7_s1
  python anysole_tta/e6_phase_analysis.py --models V7_Cv5_s1 --actions walk
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
from anysole_tta.phase_common import (CELL_ROOT, PHASE_NAMES, PHASE_NAMES_ZH,  # noqa: E402
                                      foot_mpjpe_per_frame, load_contact,
                                      load_gt_joints, load_pred_joints,
                                      phase_labels)

OUT_DIR = PROJECT_ROOT / "results_display/Etest/E6_ContactPhaseAnalysis"
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


def build_session_sets() -> dict[str, list[str]]:
    """canonical test 36 会话按动作组归类。"""
    test = []
    with open(PROJECT_ROOT / "AnysoleWorkspace/protocol/splits/default/"
              "splits.csv") as f:
        for row in csv.DictReader(f):
            if row["test"]:
                test.append(row["test"])
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


def load_session(model: str, sid: str) -> dict:
    """逐帧 per 脚 V/VT 误差与相位。"""
    gt = load_gt_joints(sid)
    contact, contact_ids, label_valid = load_contact(sid)
    if not np.array_equal(contact_ids, np.arange(len(gt))):
        raise ValueError("Contact/GT frame grid mismatch: " + sid)
    phases = phase_labels(contact, fps=FPS)
    out = {"phase": phases, "valid": label_valid.copy()}
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
        left, right = foot_mpjpe_per_frame(aligned, gt)
        out[cond + "_left"] = left
        out[cond + "_right"] = right
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
                  out_dir: Path) -> None:
    """每动作 × 每会话一张图（2026-10-31 用户裁定：1×2 上下分布，上=左脚、
    下=右脚，左右脚分开不合并不平均）：每面板 x=时间、y=该脚 Δ = VT − V、
    各模型曲线 + 该脚相位带。图内英文（项目口径）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(next(iter(next(iter(data.values())).values())))
    phase_colors = {0: "#d9d9d9", 1: "#ffffff", 2: "#f4a582", 3: "#92c5de"}
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
            ax.set_title("%s foot" % foot, fontsize=10)
            ax.set_ylabel("Δ = VT - V (mm)")
            ax.set_ylim(-60, 60)
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
    ap.add_argument("--allow-partial", action="store_true",
                    help="跳过缺评估产物的模型（默认硬报错）")
    ap.add_argument("--contact-method", default="f6_soft_time_accel_state_smpl24_v1",
                    choices=("f6_soft", "f6_soft_time_accel_v1", "f6_soft_time_accel_state_v2",
                             "f6_soft_time_accel_state_smpl24_v1"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    if args.ckpt != "last":
        raise SystemExit("E6 目前只支持 --ckpt last（best 口径 stat cells 未注册）")

    from anysole_tta import phase_common
    phase_common.CONTACT_METHOD = args.contact_method
    if args.contact_method != "f6_soft_time_accel_state_smpl24_v1" and Path(args.out_dir) == OUT_DIR:
        args.out_dir = str(OUT_DIR.parent / (OUT_DIR.name + "_" + args.contact_method))
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
            else:
                raise SystemExit(msg)
    if not models:
        raise SystemExit("no model with complete eval products")

    session_sets = build_session_sets()
    actions = ([a for a in args.actions.split(",") if a.strip()]
               if args.actions else list(session_sets))
    unknown = [a for a in actions if a not in session_sets]
    if unknown:
        raise SystemExit(f"unknown actions: {unknown}（可选：{list(session_sets)}）")

    table = {}
    for action in actions:
        sids = session_sets[action]
        data = {sid: {} for sid in sids}
        for model in models:
            for sid in sids:
                data[sid][model] = load_session(model, sid)
        agg = {model: aggregate({sid: data[sid][model] for sid in sids})
               for model in models}
        table[action] = {"sessions": sids, "table": agg}
        if not args.no_figure:
            render_figure(action, sids, data, Path(args.out_dir))
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e6_table.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    md_lines = ["# E6 接触相位下的触觉增益与门控效果\n",
                "口径：canonical test（splits.csv，36 会话，按实际出现的动作分组）；ckpt = ckpt_last；"
                "模型 = V7_Cv4（单门）/ V7_Cv5（双门）/ V7_Cv7（无门控，枚举）；"
                "Δ = VT 误差 − V 误差（踝足 MPJPE，骨盆对齐，mm；负值 = 触觉有帮助，"
                "与 E2/E5 的 Δ(VT−V) 口径一致）；"
                "相位 = f6_soft 接触标签 ±0.1s 近窗。\n"]
    for action in actions:
        sids = table[action]["sessions"]
        label = f"{GROUP_ZH.get(action, action)} {action}（{'/'.join(sids)}）"
        md_lines.append(f"## {label}\n")
        md_lines.append(render_table(table[action]["table"]))
        md_lines.append("")
    if "walk" in actions or "run" in actions:
        md_lines.append("表注：走路/跑步在 canonical test 各仅 1 条会话，样本量小，"
                        "数字作观察不作统计结论。\n")
    (out_dir / "E6_接触相位分析.md").write_text(
        "\n".join(md_lines), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    main()
