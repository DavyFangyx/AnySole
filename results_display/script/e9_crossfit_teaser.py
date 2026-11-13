"""E9 四折 crossfit 轨迹可视化（2026-11-09）。

- 空间遮挡：cam3 通过柱子的地面阴影楔形整片调暗（不是 GT 轨迹带）。
- 主图：每个动作一个代表会话（Main_Fig2-6）。
- 全量：每折每个 scene-occ test 会话一张，放 V7_Cv5/figures、V8_Cv5/figures。
- 额外：per-session ATE 图 Full_Fig2。
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
V7 = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc/V7_Cv5"
V8 = ROOT + "/results_display/Etest_crossfit_v8/E9_TrajectorySceneOcc/V8_Cv5"
HYB = ROOT + "/results/diagnostics/20261108_hybrid"
E5 = ROOT + "/results/diagnostics/20261107_e5_sceneocc_full"
MANI = os.path.join(E5, "manifests")
S2M_ROOT = ROOT + "/results/baselines/Step2Motion/predictions"
TRAJ = ROOT + "/results/diagnostics/20261108_traj_sceneocc"
sys.path.insert(0, TRAJ)
sys.path.insert(0, HYB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trajectory_scene_occlusion as T  # noqa: E402
import hybrid_sceneocc_eval as H  # noqa: E402
from e9_crossfit_export import s2m_vt_glue  # noqa: E402

FOLDS = ["fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"]
BASES = {"V7_Cv5": V7, "V8_Cv5": V8}
BASE_DIRS = {
    "V7_Cv5": os.path.join(E5, "%s_scene", "eval_motion"),
    "V8_Cv5": os.path.join(HYB, "v8_scene_%s", "eval_motion"),
}
REPS = [
    ("03", "fS5S6S8", "S5031"),
    ("04", "fS5S11S12", "S12042"),
    ("05", "fS5S6S11", "S11051"),
    ("08", "fS5S6S8", "S5081"),
    ("09", "fS6S7", "S6091"),
]
COLORS = {"Step2Motion": "#2a78d6", "Step2Motion-VT": "#5b7ac2",
          "VT": "#8a6d3b", "Hybrid": "#7b52ab", "GT": "#0b0b0b"}
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
_CACHE = {}


def get_fold(fold):
    if fold not in _CACHE:
        model, mu, sd, ck = H.load_model(fold, "_last", DEVICE)
        data = pickle.load(open(os.path.join(HYB, "hybrid_data_%s.pkl" % fold), "rb"))
        _CACHE[fold] = (model, mu, sd, ck,
                        {r["sid"]: r for r in data["test"]})
    return _CACHE[fold]


def load_base(fold, sid, base):
    z = np.load(os.path.join(BASE_DIRS[base] % fold, sid + "_VT2M.npz"))
    return (z["gt_pelvis_trans"].astype(np.float64),
            z["pred_pelvis_trans"].astype(np.float64),
            np.asarray(z["root_orient"]), z["valid_mask"].astype(bool))


def column_wedge(man):
    """cam3 通过柱子的地面阴影楔形（mocap x-z）。"""
    cal = json.load(open(man["calibration"]))
    col = man["column"]
    cx, cy = col["center_xy"]; sx, sy = col["size_xy"]
    base = np.array([[cx + dx, cy + dy, 0.0] for dx in (-sx / 2, sx / 2)
                     for dy in (-sy / 2, sy / 2)], float)
    tr = cal["mocap_raw_to_checkerboard_world"]
    R = np.array(tr["R"]); t = np.array(tr["t"]) / 100.0
    base_m = (base - t) @ R
    c = cal["cameras"]["cam3"]
    Rc = np.array(c["R"]); tc = np.array(c["t"]) / 100.0
    Cm = ((-Rc.T @ tc) - t) @ R
    a = np.degrees(np.arctan2(base_m[:, 2] - Cm[2], base_m[:, 0] - Cm[0]))
    center = np.median(a)
    au = (a - center + 180.0) % 360.0 - 180.0 + center
    a0, a1 = float(au.min()), float(au.max())
    L = 12.0
    p1 = Cm[[0, 2]] + L * np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
    p2 = Cm[[0, 2]] + L * np.array([np.cos(np.radians(a1)), np.sin(np.radians(a1))])
    wedge = np.array([Cm[[0, 2]], p1, p2])
    return wedge, base_m[:, [0, 2]]


def arms_for(fold, sid, base):
    man = json.load(open(os.path.join(MANI, sid + ".json")))
    ev = man["full_events"]
    gt, vt, ro, valid = load_base(fold, sid, base)
    n = len(gt)
    model, mu, sd, ck, by_sid = get_fold(fold)
    hips = T.gt_hips_from_fit(sid, n)
    s2m, _ = T.load_baseline(
        os.path.join(S2M_ROOT, "gait_model_smpl24_%s" % fold,
                     "eval_motion", sid + ".npz"), gt, hips)
    hr = by_sid.get(sid)
    if ck.get("feat_mode", "full") == "tilearn" and hr is not None:
        hr = dict(hr); hr["X"] = hr["X"][:, :15]
    vis = np.asarray(man["frames"]["vis"], float)
    hyb = H.assemble(sid, hr, model, mu, sd, vt, ro, ev, DEVICE, 1.0,
                     calib="block", calib_frames=40,
                     vis=vis, end_glue=True)
    s2m_vt = s2m_vt_glue(s2m[:, [0, 2]], vt, vis, ev)
    return man, ev, gt, {
        "GT": gt, "Step2Motion": s2m, "Step2Motion-VT": s2m_vt,
        "VT": vt[:, [0, 2]], "Hybrid": hyb,
    }


def p2(a):
    return a if a.shape[1] == 2 else a[:, [0, 2]]


def plot(out_path, base, action, fold, sid, man, ev, gt, arms):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    wedge, foot = column_wedge(man)
    ax.fill(wedge[:, 0], wedge[:, 1], color="#2b2b2b", alpha=0.13,
            zorder=0, label="vision lost (spatial)")
    ax.fill(foot[:, 0], foot[:, 1], color="#2b2b2b", alpha=0.45, zorder=1,
            label="column")
    n = len(gt)
    samples = np.linspace(5, n - 6, 7).astype(int)
    for i in samples:
        g = gt[i, [0, 2]]
        for arm in ("Step2Motion-VT", "Hybrid", "VT"):
            p = p2(arms[arm])[i]
            ax.plot([g[0], p[0]], [g[1], p[1]], color="#9b9a94", lw=0.4,
                    alpha=0.45, zorder=2)
    for arm in ("Step2Motion-VT", "VT", "Hybrid", "GT"):
        tr = p2(arms[arm])
        ls = "--" if arm == "GT" else "-"
        ax.plot(tr[:, 0], tr[:, 1], color=COLORS[arm],
                lw=2.4 if arm == "GT" else 2.0, ls=ls, label=arm, zorder=3)
        for e in ev:
            s, t = max(int(e["start"]), 0), min(int(e["end"]), n)
            ax.plot(tr[s:t, 0], tr[s:t, 1], color=COLORS[arm], lw=3.2,
                    alpha=0.9, ls=ls, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    # 以 GT/各臂 + 柱脚为视图范围；楔形 fill 由 axes 裁剪，避免被 12m 楔形拉爆
    xs, ys = [], []
    for arm in ("GT", "Step2Motion-VT", "VT", "Hybrid"):
        tr = p2(arms[arm])
        xs.extend(tr[:, 0]); ys.extend(tr[:, 1])
    xs.extend(foot[:, 0]); ys.extend(foot[:, 1])
    pad = 0.4
    ax.set_xlim(float(np.min(xs)) - pad, float(np.max(xs)) + pad)
    ax.set_ylim(float(np.min(ys)) - pad, float(np.max(ys)) + pad)
    ax.grid(True, color="#e6e5e1", lw=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("E9 four-fold — action %s, %s (%s, %s)" %
                 (action, sid, fold, base))
    ax.legend(loc="best", frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def per_session_ate(root, base):
    src = os.path.join(root, "Full_Table1_E9_Crossfit_PerSession.csv")
    rows = [r for r in read_csv(src) if r["base"] == base]
    arms = ["Step2Motion-VT", "Hybrid", "VT"]
    colors = {"Step2Motion-VT": COLORS["Step2Motion-VT"], "Hybrid": COLORS["Hybrid"],
              "VT": COLORS["VT"]}
    sessions = sorted({(r["fold"], r["action"], r["session"]) for r in rows})
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(sessions))
    for k, arm in enumerate(arms):
        vals = []
        for f, a, s in sessions:
            rr = [r for r in rows if r["fold"] == f and r["session"] == s and r["arm"] == arm]
            vals.append(float(rr[0]["occ_ate_mm"]) if rr and rr[0]["occ_ate_mm"] else np.nan)
        ax.plot(x + (k - 1) * 0.18, vals, "o", ms=3.5, color=colors[arm], label=arm)
    labels = ["%s/%s/%s" % (f.replace("f", ""), a, s) for f, a, s in sessions]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("occlusion ATE (mm)")
    ax.set_title("%s — per-session (all four-fold test sessions)" % base)
    ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "Full_Fig2_E9_Crossfit_PerSessionATE.png"), dpi=200)
    plt.close(fig)
    print("wrote per-session ATE", root)


def main():
    # main representative figures
    for base, root in BASES.items():
        for k, (action, fold, sid) in enumerate(REPS):
            man, ev, gt, arms = arms_for(fold, sid, base)
            plot(os.path.join(root, "Main_Fig%d_E9_Crossfit_Teaser_action%s.png"
                              % (k + 2, action)),
                 base, action, fold, sid, man, ev, gt, arms)
        per_session_ate(root, base)
    # all sessions, one figures/ folder per base
    from ti_step_dataset import fold_sets
    sets = fold_sets()
    for base, root in BASES.items():
        figdir = os.path.join(root, "figures")
        os.makedirs(figdir, exist_ok=True)
        for fold in FOLDS:
            sids = [s for s in sorted(sets[fold]["test"])
                    if os.path.isfile(os.path.join(MANI, s + ".json"))]
            for sid in sids:
                man, ev, gt, arms = arms_for(fold, sid, base)
                plot(os.path.join(figdir, "%s_%s_action%s.png"
                                  % (fold, sid, man["action"])),
                     base, man["action"], fold, sid, man, ev, gt, arms)
                print("figure", base, fold, sid)
    print("done")


if __name__ == "__main__":
    main()
