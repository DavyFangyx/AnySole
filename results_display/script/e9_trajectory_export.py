"""E9 场景遮挡轨迹可视化（单视角 cam3，2026-11-08）。

产物 → results_display/Etest_crossfit/E9_TrajectorySceneOcc/
  Main_Table1_E9_SceneOcc_OccAte.csv       遮挡内 ATE（动作均值）
  Main_Table2_E9_SceneOcc_FullAte.csv      全程 ATE（动作均值）
  Main_Fig1-3_E9_SceneOcc_Teaser_action*.png/pdf
  Main_Fig4_E9_SceneOcc_OccAte.png         遮挡内 ATE 柱图
  Full_Table1_E9_SceneOcc_PerSession.csv
  Full_Table2_E9_SceneOcc_CalibAblation.csv
  Full_Table3_E9_SceneOcc_ModalityDecomp.csv
  Supplementary_Table1_E9_Blackout_fS6S7.csv
  Supplementary_Fig1_E9_Blackout_fS6S7.png
  detail/E9_轨迹可视化_说明.md

口径：事件外 = VT；事件内从 VT 最后可见帧交接；Hybrid = 纯学习 IMU 臂；
Hybrid+VTα = 与遮挡后 VT 预测逐帧位移融合；VT 臂 = occluded-V prior + T，
不是“遮挡中还能看见人”。S13 用 fS5S11S12、S14 用 fS6S7 的固定 epoch 模型。
"""
import csv
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = "/data/fangyuxuan/projects/gait"
OUT = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc"
DETAIL = os.path.join(OUT, "detail")
SUPP = os.path.join(OUT, "supplementary")
HYB = ROOT + "/results/diagnostics/20261108_hybrid"
TRAJ = ROOT + "/results/diagnostics/20261108_traj_sceneocc"
sys.path.insert(0, TRAJ)
sys.path.insert(0, HYB)
import trajectory_scene_occlusion as T  # noqa: E402
import hybrid_sceneocc_eval as H  # noqa: E402

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
SIDS = ["S14033", "S14091", "S14092", "S14093",
        "S13082", "S13092", "S13093"]
REP = {"03": "S14033", "08": "S13082", "09": "S14092"}
LABEL = {"03": "shuttle walk", "08": "shuttle run", "09": "lateral walk"}
TAG = "_last"
CALIB_FRAMES = 40
FPS = 40.0

COLORS = {
    "GT": "#0b0b0b",
    "Step2Motion": "#2a78d6",
    "MotionPRO": "#eb6834",
    "Anysolo-TI": "#1baf7a",
    "Anysolo-I": "#eda100",
    "VT": "#8a6d3b",
    "Hybrid": "#7b52ab",
    "Hybrid+VT0.75": "#d64545",
    "Hybrid+VT0.50": "#c58b2a",
}
MAIN_ARMS = ["Step2Motion", "Anysolo-TI", "VT", "Hybrid", "Hybrid+VT0.75"]
ALL_ARMS = ["GT", "Step2Motion", "MotionPRO", "Anysolo-TI", "Anysolo-I",
            "VT", "Hybrid", "Hybrid+VT0.75", "Hybrid+VT0.50"]


def load_session(sid):
    man = json.load(open(os.path.join(T.MANI, sid + ".json")))
    zvt = np.load(os.path.join(T.SCENE7, sid + "_VT2M.npz"))
    gt = zvt["gt_pelvis_trans"].astype(np.float64)
    vt = zvt["pred_pelvis_trans"].astype(np.float64)
    ro = np.asarray(zvt["root_orient"])
    valid = zvt["valid_mask"].astype(bool)
    n = len(gt)
    hips = T.gt_hips_from_fit(sid, n)
    return man, gt, vt, ro, valid, hips


def build_arms(tag=TAG):
    data = pickle.load(open(os.path.join(HYB, "hybrid_data_scene7.pkl"), "rb"))
    by_sid = {r["sid"]: r for r in data["test"]}
    models = {}
    device = DEVICE
    arms, rows, meta = {}, [], {}
    for sid in SIDS:
        man, gt, vt, ro, valid, hips = load_session(sid)
        ev = man["full_events"]
        subject = "S13" if sid.startswith("S13") else "S14"
        fold = H.MODEL_FOR[subject]
        if fold not in models:
            models[fold] = H.load_model(fold, tag, device)
        model, mu, sd, ck = models[fold]
        hr = by_sid.get(sid)
        if hr is not None and ck.get("feat_mode", "full") == "tilearn":
            hr = dict(hr); hr["X"] = hr["X"][:, :15]
        a = {}
        masks = {"GT": valid, "Anysolo-TI": valid, "Anysolo-I": valid,
                 "VT": valid, "Hybrid": valid, "Hybrid+VT0.75": valid,
                 "Hybrid+VT0.50": valid}
        a["GT"] = gt
        a["Step2Motion"], masks["Step2Motion"] = T.load_baseline(
            os.path.join(T.S2M, sid + ".npz"), gt, hips)
        a["MotionPRO"], masks["MotionPRO"] = T.load_baseline(
            os.path.join(T.MPRO, sid + ".npz"), gt, hips)
        a["Anysolo-TI"] = T.assemble_anysolo(sid, "TI", vt, ro, gt, ev)
        a["Anysolo-I"] = T.assemble_anysolo(sid, "I", vt, ro, gt, ev)
        a["VT"] = vt[:, [0, 2]].copy()
        for alpha, name in ((1.0, "Hybrid"), (0.75, "Hybrid+VT0.75"),
                            (0.50, "Hybrid+VT0.50")):
            a[name] = H.assemble(sid, hr, model, mu, sd, vt, ro, ev, device,
                                 alpha, calib="block",
                                 calib_frames=CALIB_FRAMES)
        arms[sid] = a
        meta[sid] = {"action": man["action"], "events": ev, "n": len(gt),
                     "valid": valid}
        occ = np.zeros(len(gt), bool)
        for e in ev:
            occ[max(int(e["start"]), 0):min(int(e["end"]), len(gt))] = True
        for arm, tr in a.items():
            vm = masks.get(arm, valid)
            full, _ = T.ates(tr, gt, vm)
            tr2 = tr if tr.shape[1] == 2 else tr[:, [0, 2]]
            d = np.linalg.norm(tr2 - gt[:, [0, 2]], axis=-1)
            ok = occ & vm
            rows.append({"action": man["action"], "session": sid, "arm": arm,
                         "full_ate_mm": full,
                         "occ_ate_mm": float(np.sqrt(np.mean(d[ok] ** 2)) * 1000)
                         if ok.any() else None,
                         "n_frames": len(gt), "n_occ": int(occ.sum())})
    return arms, rows, meta


def write_csv(path, rows, fields=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def agg_action(rows, key, arms):
    out = []
    for act in ("03", "08", "09"):
        rec = {"action": act, "n": sum(1 for r in rows if r["action"] == act
                                        and r["arm"] == arms[0])}
        for arm in arms:
            vv = [r[key] for r in rows
                  if r["action"] == act and r["arm"] == arm and r[key] is not None]
            rec[arm] = float(np.mean(vv)) if vv else None
        out.append(rec)
    return out


def p2(arr):
    return arr if arr.shape[1] == 2 else arr[:, [0, 2]]


def plot_teaser(arms, meta, action):
    sid = REP[action]
    a = arms[sid]
    ev = meta[sid]["events"]
    gt = a["GT"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    g2 = p2(gt)
    for e in ev:
        s, t = max(int(e["start"]), 0), min(int(e["end"]), len(g2))
        ax.plot(g2[s:t, 0], g2[s:t, 1], color="#2b2b2b", lw=13, alpha=0.18,
                zorder=1, solid_capstyle="round")
    ax.plot([], [], color="#2b2b2b", lw=10, alpha=0.18,
            label="vision lost (GT)")
    n = len(gt)
    samples = np.linspace(5, n - 6, 7).astype(int)
    for i in samples:
        g = gt[i, [0, 2]]
        for arm in ("Step2Motion", "Anysolo-TI", "Hybrid", "Hybrid+VT0.75"):
            p = p2(a[arm])[i]
            ax.plot([g[0], p[0]], [g[1], p[1]], color="#9b9a94", lw=0.4,
                    alpha=0.45, zorder=2)
    for arm in ("Step2Motion", "Anysolo-TI", "VT", "Hybrid+VT0.75",
                "Hybrid", "GT"):
        tr = p2(a[arm])
        ls = "--" if arm in ("GT", "Hybrid") else "-"
        ax.plot(tr[:, 0], tr[:, 1], color=COLORS[arm],
                lw=2.4 if arm == "GT" else (2.2 if arm == "Hybrid" else 2.0),
                ls=ls, label=arm, zorder=5 if arm == "Hybrid" else 3)
        for e in ev:
            s, t = max(int(e["start"]), 0), min(int(e["end"]), n)
            ax.plot(tr[s:t, 0], tr[s:t, 1], color=COLORS[arm], lw=3.2,
                    alpha=0.9, zorder=6 if arm == "Hybrid" else 4, ls=ls)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Trajectory under scene occlusion — action %s (%s), %s"
                 % (action, LABEL[action], sid))
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "Main_Fig%s_E9_SceneOcc_Teaser_action%s.%s"
                                 % ({"03": "1", "08": "2", "09": "3"}[action],
                                    action, ext)), dpi=220)
    plt.close(fig)
    print("wrote teaser action", action)


def plot_teaser_baselines(arms, meta, action):
    """原五臂基线版（S2M / MotionPRO / Anysolo-TI / Anysolo-I / GT）。"""
    sid = REP[action]
    a = arms[sid]
    ev = meta[sid]["events"]
    gt = a["GT"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    g2 = p2(gt)
    for e in ev:
        s, t = max(int(e["start"]), 0), min(int(e["end"]), len(g2))
        ax.plot(g2[s:t, 0], g2[s:t, 1], color="#2b2b2b", lw=13, alpha=0.18,
                zorder=1, solid_capstyle="round")
    ax.plot([], [], color="#2b2b2b", lw=10, alpha=0.18,
            label="vision lost (GT)")
    n = len(gt)
    samples = np.linspace(5, n - 6, 7).astype(int)
    for i in samples:
        g = gt[i, [0, 2]]
        for arm in ("Step2Motion", "MotionPRO", "Anysolo-TI", "Anysolo-I"):
            p = p2(a[arm])[i]
            ax.plot([g[0], p[0]], [g[1], p[1]], color="#9b9a94", lw=0.4,
                    alpha=0.5, zorder=2)
    for arm in ("Step2Motion", "MotionPRO", "Anysolo-TI", "Anysolo-I", "GT"):
        tr = p2(a[arm])
        ax.plot(tr[:, 0], tr[:, 1], color=COLORS[arm],
                lw=2.4 if arm == "GT" else 2.0,
                ls="--" if arm == "GT" else "-", label=arm, zorder=3)
        for e in ev:
            s, t = max(int(e["start"]), 0), min(int(e["end"]), n)
            ax.plot(tr[s:t, 0], tr[s:t, 1], color=COLORS[arm], lw=3.2,
                    alpha=0.9, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Baselines under scene occlusion — action %s (%s), %s"
                 % (action, LABEL[action], sid))
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(
        SUPP, "Supplementary_Fig%s_E9_SceneOcc_Baselines_action%s.png"
        % ({"03": "2", "08": "3", "09": "4"}[action], action)), dpi=220)
    plt.close(fig)
    print("wrote baseline teaser action", action)


def plot_bar(rows):
    agg = {r["action"]: r for r in agg_action(rows, "occ_ate_mm", MAIN_ARMS)}
    acts = ("03", "08", "09")
    x = np.arange(len(acts))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for k, arm in enumerate(MAIN_ARMS):
        vals = [agg[a][arm] for a in acts]
        ax.bar(x + (k - 2) * width, vals, width, label=arm,
               color=COLORS[arm], alpha=0.9)
        for xx, v in zip(x + (k - 2) * width, vals):
            ax.text(xx, v + 4, "%.0f" % v, ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["03 %s" % LABEL["03"], "08 %s" % LABEL["08"],
                        "09 %s" % LABEL["09"]])
    ax.set_ylabel("occlusion ATE (mm)")
    ax.set_title("Scene occlusion — inside full_event (lower is better)")
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Main_Fig4_E9_SceneOcc_OccAte.png"), dpi=200)
    plt.close(fig)
    print("wrote bar fig")


def calibration_ablation(arms, meta):
    rows = []
    for sid in SIDS:
        man, gt, vt, ro, valid, hips = load_session(sid)
        ev = man["full_events"]
        subject = "S13" if sid.startswith("S13") else "S14"
        # 复用 build_arms 已算好的 block 轨迹，vtyaw 单独重算
        model, mu, sd, ck = models_for(sid)
        hr = pickle.load(open(os.path.join(HYB, "hybrid_data_scene7.pkl"), "rb"))
        hr = {r["sid"]: r for r in hr["test"]}[sid]
        tr_v = H.assemble(sid, hr, model, mu, sd, vt, ro, ev,
                          DEVICE, 1.0, calib="vtyaw")
        occ = np.zeros(len(gt), bool)
        for e in ev:
            occ[max(int(e["start"]), 0):min(int(e["end"]), len(gt))] = True
        def oa(tr):
            tr2 = tr if tr.shape[1] == 2 else tr[:, [0, 2]]
            d = np.linalg.norm(tr2 - gt[:, [0, 2]], axis=-1)
            ok = occ & valid
            return float(np.sqrt(np.mean(d[ok] ** 2)) * 1000)
        rows.append({"action": man["action"], "session": sid,
                     "vtyaw_occ_ate_mm": oa(tr_v),
                     "block_occ_ate_mm": oa(arms[sid]["Hybrid"]),
                     "delta_mm": oa(arms[sid]["Hybrid"]) - oa(tr_v)})
    return rows


_MODELS = {}


def models_for(sid):
    subject = "S13" if sid.startswith("S13") else "S14"
    fold = H.MODEL_FOR[subject]
    if fold not in _MODELS:
        _MODELS[fold] = H.load_model(fold, TAG, DEVICE)
    return _MODELS[fold]


def modality_decomp():
    rows = []
    for sid in SIDS:
        man, gt, vt, ro_vt, valid, hips = load_session(sid)
        ev = man["full_events"]
        for arm in ("V2M", "T2M", "VT2M"):
            z = np.load(os.path.join(T.SCENE7, "%s_%s.npz" % (sid, arm)))
            pr = z["pred_pelvis_trans"].astype(np.float64)
            ro = np.asarray(z["root_orient"])
            n = min(len(pr), len(gt))
            traj = vt[:n, [0, 2]].copy()
            for e in ev:
                a = max(int(e["start"]), 1)
                b = min(int(e["end"]), n - 1)
                n0 = a - 1
                th = np.radians(T.aa_yaw(ro_vt[n0]) - T.aa_yaw(ro[n0]))
                d = pr[a:b + 1, [0, 2]] - pr[n0, [0, 2]]
                traj[a:b + 1] = vt[n0, [0, 2]] + T.rot2(d, th)
            occ = np.zeros(n, bool)
            for e in ev:
                occ[max(int(e["start"]), 0):min(int(e["end"]), n)] = True
            d = np.linalg.norm(traj - gt[:n, [0, 2]], axis=-1)
            ok = occ & valid[:n]
            rows.append({"action": man["action"], "session": sid, "arm": arm,
                         "occ_ate_mm": float(np.sqrt(np.mean(d[ok] ** 2)) * 1000)})
    return rows


def blackout_table():
    fp = os.path.join(HYB, "results_hybrid_fS6S7_last.json")
    if not os.path.isfile(fp):
        return [], []
    j = json.load(open(fp))
    rows = []
    order = ["TI-learn", "Hybrid", "TI-contact", "TI", "VT", "V-freeze", "V", "I"]
    for lev in ("20", "40", "80", "160", "320"):
        d = j["folds"]["fS6S7"]["levels"][lev]
        rec = {"level_s": int(lev) / FPS, "level_frames": int(lev)}
        for a in order:
            rec[a] = d["Hybrid"] if a == "Hybrid" else d["ref"].get(a)
        rows.append(rec)
    return rows, j


def plot_blackout(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = [r["level_s"] for r in rows]
    for arm, color, ls in (("Hybrid", COLORS["Hybrid"], "-"),
                           ("TI-learn", "#2a78d6", "--"),
                           ("TI-contact", "#1baf7a", "-."),
                           ("VT", "#8a6d3b", ":")):
        ys = [r.get(arm) for r in rows]
        ax.plot(xs, ys, marker="o", color=color, ls=ls, lw=1.8, label=arm)
    ax.set_xlabel("blackout length (s)")
    ax.set_ylabel("ATE (mm)")
    ax.set_title("Blackout ladder — fS6S7 fold (fixed epoch)")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SUPP, "Supplementary_Fig1_E9_Blackout_fS6S7.png"),
                dpi=200)
    plt.close(fig)
    print("wrote blackout fig")


def write_report(rows, calib, decomp, black_rows):
    occ = agg_action(rows, "occ_ate_mm", ALL_ARMS)
    lines = ["# E9 场景遮挡轨迹可视化（2026-11-08）", "",
             "## 口径", "",
             "- 单视角 cam3；柱子按标定合成进 cam3（`cam3_occ_scene`），"
             "`full_event` = 人体可见度 ≤ 0.05 的连续区间。",
             "- 事件外 = VT；事件内从 VT 最后可见帧位置/朝向交接。",
             "- **Hybrid** = 纯学习 IMU 臂（44 维：12 IMU + yaw 3 + F2 前端 12 "
             "+ 压力块 10 + 压力时钟 6 + v_phys 1；yaw 帧 2D 位移）；"
             "遮挡内完全不用 VT。",
             "- **Hybrid+VTα** = 遮挡内 `α·Hybrid + (1−α)·VT` 的逐帧位移融合；"
             "α=1 即纯 Hybrid，α=0 即纯 VT。",
             "- **VT** = 主模型在 `cam3_occ_scene` 上的输出。全遮挡时 V 分支"
             "吃的是被柱子污染的特征，V2M/T2M/VT2M 分解表明它输出的是"
             "**learned prior**，不是视觉证据；不得写成“遮挡中视觉仍有效”。",
             "- 帧标定：用可见块 [n0−40, n0] 网格搜索 ψ0，使模型可见块位移"
             "对齐 VT 可见块位移；仅用可见数据，无 GT。",
             "- 模型：S13 用 fS5S11S12（val=S13）、S14 用 fS6S7（val=S14）"
             "的固定 epoch（200）模型；受试者未进 train，但属 held-out-subject "
             "validation，不是四折 test。",
             "- 动作 03/08 各 1 个会话，09 有 5 个；不做显著性主张。",
             "- 基线（S2M/MotionPRO/Anysolo）按当前 `eval_motion` 重算；与 "
             "2026-11-08 旧表相差 <10 mm（S2M 213.9 vs 209.0、303.4 vs 294.3、"
             "218.6 vs 216.6），结论不变。MotionPRO 09 发散（4552.7），"
             "表中保留、不剔除。", "",
             "## 遮挡内 ATE（动作均值，mm）", "",
             "| 动作 | n | " + " | ".join(ALL_ARMS) + " |",
             "|---|---|" + "---|" * len(ALL_ARMS)]
    for rec in occ:
        vals = ["%.1f" % rec[a] if rec[a] is not None else "—"
                for a in ALL_ARMS]
        lines.append("| %s | %d | %s |" % (rec["action"], rec["n"],
                                           " | ".join(vals)))
    occ_map = {rec["action"]: rec for rec in occ}
    cal_map = {}
    for act in ("03", "08", "09"):
        vv = [r for r in calib if r["action"] == act]
        cal_map[act] = (float(np.mean([r["vtyaw_occ_ate_mm"] for r in vv])),
                        float(np.mean([r["block_occ_ate_mm"] for r in vv])))
    lines += ["", "## 读数", "",
              "- 纯 Hybrid vs S2M（遮挡内，mm）：03 %.1f vs %.1f；08 %.1f vs %.1f；"
              "09 %.1f vs %.1f。03/09 低于 S2M，08 高于 S2M；"
              % (occ_map["03"]["Hybrid"], occ_map["03"]["Step2Motion"],
                 occ_map["08"]["Hybrid"], occ_map["08"]["Step2Motion"],
                 occ_map["09"]["Hybrid"], occ_map["09"]["Step2Motion"]),
              "- Hybrid+VT0.75 vs S2M：03 %.1f vs %.1f；08 %.1f vs %.1f；"
              "09 %.1f vs %.1f。三动作均低于 S2M，但 VT 是 occluded-V prior + T，"
              "不是纯 IMU；"
              % (occ_map["03"]["Hybrid+VT0.75"], occ_map["03"]["Step2Motion"],
                 occ_map["08"]["Hybrid+VT0.75"], occ_map["08"]["Step2Motion"],
                 occ_map["09"]["Hybrid+VT0.75"], occ_map["09"]["Step2Motion"]),
              "- 可见块帧标定（动作均值 vtyaw → block，mm）：03 %.1f→%.1f；"
              "08 %.1f→%.1f；09 %.1f→%.1f。不标定会把 body yaw/course 的 "
              "~90° 差当成模型误差；"
              % (cal_map["03"][0], cal_map["03"][1], cal_map["08"][0],
                 cal_map["08"][1], cal_map["09"][0], cal_map["09"][1]),
              "- V2M/T2M/VT2M 分解见 Full_Table3：V2M 是运动先验（逐帧方向极抖），"
              "T2M 明显更弱，VT 是 V 先验 + T 相位；不得写成“遮挡中 V 仍看见人”。",
              "", "## 产物", "",
              "- `Main_Table1/2`：遮挡内 / 全程 ATE（动作均值）",
              "- `Main_Fig1-3`：teaser（03/08/09）",
              "- `Main_Fig4`：遮挡内 ATE 柱图",
              "- `Supplementary_Fig2-4`：原五臂基线 teaser",
              "- `Full_Table1`：逐会话全臂",
              "- `Full_Table2`：帧标定消融（vtyaw vs block）",
              "- `Full_Table3`：V2M/T2M/VT2M 分解",
              "- `Supplementary_*`：fS6S7 黑屏阶梯"]
    os.makedirs(DETAIL, exist_ok=True)
    with open(os.path.join(DETAIL, "E9_轨迹可视化_说明.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DETAIL, exist_ok=True)
    os.makedirs(SUPP, exist_ok=True)
    arms, rows, meta = build_arms()
    write_csv(os.path.join(OUT, "Full_Table1_E9_SceneOcc_PerSession.csv"), rows)
    write_csv(os.path.join(OUT, "Main_Table1_E9_SceneOcc_OccAte.csv"),
              agg_action(rows, "occ_ate_mm", ALL_ARMS))
    write_csv(os.path.join(OUT, "Main_Table2_E9_SceneOcc_FullAte.csv"),
              agg_action(rows, "full_ate_mm", ALL_ARMS))
    calib = calibration_ablation(arms, meta)
    write_csv(os.path.join(OUT, "Full_Table2_E9_SceneOcc_CalibAblation.csv"),
              calib)
    decomp = modality_decomp()
    write_csv(os.path.join(OUT, "Full_Table3_E9_SceneOcc_ModalityDecomp.csv"),
              decomp)
    black_rows, black_json = blackout_table()
    if black_rows:
        write_csv(os.path.join(SUPP, "Supplementary_Table1_E9_Blackout_fS6S7.csv"),
                  black_rows)
        with open(os.path.join(DETAIL, "blackout_fS6S7.json"), "w") as f:
            json.dump(black_json, f, ensure_ascii=False, indent=2)
        plot_blackout(black_rows)
    with open(os.path.join(DETAIL, "sceneocc_rows.json"), "w") as f:
        json.dump({"rows": rows, "calib": calib, "decomp": decomp}, f,
                  ensure_ascii=False, indent=2)
    for act in ("03", "08", "09"):
        plot_teaser(arms, meta, act)
        plot_teaser_baselines(arms, meta, act)
    plot_bar(rows)
    write_report(rows, calib, decomp, black_rows)
    print("E9 ->", OUT)


if __name__ == "__main__":
    main()
