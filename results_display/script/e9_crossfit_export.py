"""E9 四折 crossfit 场景遮挡轨迹导出（2026-11-08）。

四折 LSO（fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8），每折 test 的 scene-occ
manifest 会话；V7_Cv5 / V8_Cv5 双基座；S2M 用逐折 ckpt 的 per-fold 预测。
产物：E9_TrajectorySceneOcc/crossfit/
  Main_Table1_E9_Crossfit_OccAte.csv
  Main_Table2_E9_Crossfit_FullAte.csv
  Main_Table3_E9_Crossfit_FoldAction.csv
  Full_Table1_E9_Crossfit_PerSession.csv
  Main_Fig1_E9_Crossfit_OccAte.png
  detail/E9_四折Crossfit说明.md
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

ROOT = "/data/fangyuxuan/projects/gait"
OUT = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc/crossfit"
DETAIL = os.path.join(OUT, "detail")
HYB = ROOT + "/results/diagnostics/20261108_hybrid"
E5 = ROOT + "/results/diagnostics/20261107_e5_sceneocc_full"
MANI = os.path.join(E5, "manifests")
TRAJ = ROOT + "/results/diagnostics/20261108_traj_sceneocc"
S2M_ROOT = ROOT + "/results/baselines/Step2Motion/predictions"
sys.path.insert(0, TRAJ)
sys.path.insert(0, HYB)
import trajectory_scene_occlusion as T  # noqa: E402
import hybrid_sceneocc_eval as H  # noqa: E402

FOLDS = ["fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"]
BASES = ["V7_Cv5", "V8_Cv5"]
BASE_DIRS = {
    "V7_Cv5": os.path.join(E5, "%s_scene", "eval_motion"),
    "V8_Cv5": os.path.join(HYB, "v8_scene_%s", "eval_motion"),
}
ARMS = ["Step2Motion", "Anysolo-TI", "Anysolo-I", "VT",
        "Hybrid", "Hybrid+VT0.50", "Hybrid+VT0.75"]
COLORS = {"Step2Motion": "#2a78d6", "Anysolo-TI": "#1baf7a",
          "Anysolo-I": "#eda100", "VT": "#8a6d3b", "Hybrid": "#7b52ab",
          "Hybrid+VT0.50": "#c58b2a", "Hybrid+VT0.75": "#d64545"}
CALIB_FRAMES = 40


def fold_sets():
    from ti_step_dataset import fold_sets as fs
    return fs()


def load_session(sid, vt_path):
    zvt = np.load(vt_path)
    gt = zvt["gt_pelvis_trans"].astype(np.float64)
    vt = zvt["pred_pelvis_trans"].astype(np.float64)
    ro = np.asarray(zvt["root_orient"])
    valid = zvt["valid_mask"].astype(bool)
    jv = np.asarray(zvt["joint_xyz_world"])
    n = len(gt)
    hips = T.gt_hips_from_fit(sid, n)
    return gt, vt, ro, valid, hips, jv


def _align_at(pred, pred_hips, target, target_hips, i0):
    def yaw(h):
        return np.degrees(np.arctan2(h[0], h[2])) if len(h) >= 3 \
            else np.degrees(np.arctan2(h[0], h[1]))
    a = yaw(pred_hips[i0])
    b = yaw(target_hips[i0])
    th = np.radians(b - a)
    c, s = np.cos(th), np.sin(th)
    v = pred[:, [0, 2]] - pred[i0, [0, 2]]
    rot = np.stack([c * v[:, 0] + s * v[:, 1],
                    -s * v[:, 0] + c * v[:, 1]], -1)
    return rot + target[i0, [0, 2]]


def load_s2m_vt(path, vt_joints, vt, ev):
    """S2M 事件起点按 VT 位置 + 髋线 yaw 对齐（公平 VT-start 臂）。"""
    z = np.load(path, allow_pickle=True)
    J = np.asarray(z["joint_xyz_world"])
    root = T.zup_to_yup(J[:, 0])
    hips = T.zup_to_yup(J[:, 1] - J[:, 2])
    m = np.asarray(z["valid_mask"], bool)
    n = min(len(root), len(vt))
    root, hips, m = root[:n], hips[:n], m[:n]
    vj = T.zup_to_yup(vt_joints[:n])
    vh = vj[:, 1] - vj[:, 2]
    i0 = max(int(ev[0]["start"]) - 1, 0)
    traj = _align_at(root, hips, vt, vh, i0)
    for e in ev:
        a = max(int(e["start"]), 1)
        b = min(int(e["end"]), n - 1)
        n0 = a - 1
        seg = _align_at(root, hips, vt, vh, n0)
        traj[a:b + 1] = seg[a:b + 1]
    return traj, m


def occ_mask(ev, n):
    m = np.zeros(n, bool)
    for e in ev:
        m[max(int(e["start"]), 0):min(int(e["end"]), n)] = True
    return m


def metrics(tr, gt, valid, ev):
    tr2 = tr if tr.shape[1] == 2 else tr[:, [0, 2]]
    d = np.linalg.norm(tr2 - gt[:, [0, 2]], axis=-1)
    om = occ_mask(ev, len(gt))
    ok = om & valid
    full = float(np.sqrt(np.mean((d[valid]) ** 2)) * 1000) if valid.any() else None
    occ = float(np.sqrt(np.mean((d[ok]) ** 2)) * 1000) if ok.any() else None
    return full, occ


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DETAIL, exist_ok=True)
    sets = fold_sets()
    rows = []
    for fold in FOLDS:
        sids = [s for s in sets[fold]["test"]
                if os.path.isfile(os.path.join(MANI, s + ".json"))]
        sids = sorted(sids)
        print(fold, "test scene sessions", len(sids))
        # hybrid data/model for this fold
        hdata = pickle.load(open(os.path.join(HYB, "hybrid_data_%s.pkl" % fold), "rb"))
        by_sid = {r["sid"]: r for r in hdata["test"]}
        models = {}
        for base in BASES:
            for sid in sids:
                man = json.load(open(os.path.join(MANI, sid + ".json")))
                ev = man["full_events"]
                vt_path = os.path.join(BASE_DIRS[base] % fold, sid + "_VT2M.npz")
                if not os.path.isfile(vt_path):
                    print("  missing VT", base, sid)
                    continue
                gt, vt, ro, valid, hips, jv = load_session(sid, vt_path)
                hr = by_sid.get(sid)
                if hr is None:
                    print("  missing hybrid features", sid)
                    continue
                if fold not in models:
                    models[fold] = H.load_model(fold, "_last", D := __import__("torch").device(
                        "cuda:3" if __import__("torch").cuda.is_available() else "cpu"))
                model, mu, sd, ck = models[fold]
                if ck.get("feat_mode", "full") == "tilearn":
                    hr = dict(hr); hr["X"] = hr["X"][:, :15]
                a = {}
                a["VT"] = vt[:, [0, 2]].copy()
                s2m_path = os.path.join(S2M_ROOT, "gait_model_smpl24_%s" % fold,
                                        "eval_motion", sid + ".npz")
                if os.path.isfile(s2m_path):
                    a["Step2Motion"], m_s2m = T.load_baseline(s2m_path, gt, hips)
                else:
                    print("  missing S2M", fold, sid)
                    m_s2m = valid
                a["Anysolo-TI"] = T.assemble_anysolo(sid, "TI", vt, ro, gt, ev)
                a["Anysolo-I"] = T.assemble_anysolo(sid, "I", vt, ro, gt, ev)
                for alpha, name in ((1.0, "Hybrid"), (0.50, "Hybrid+VT0.50"),
                                    (0.75, "Hybrid+VT0.75")):
                    a[name] = H.assemble(sid, hr, model, mu, sd, vt, ro, ev,
                                         D, alpha, calib="block",
                                         calib_frames=CALIB_FRAMES)
                for arm, tr in a.items():
                    vm = m_s2m if arm in ("Step2Motion", "Step2Motion-VT") else valid
                    full, occ = metrics(tr, gt, vm, ev)
                    rows.append({"fold": fold, "base": base, "action": man["action"],
                                 "session": sid, "arm": arm,
                                 "full_ate_mm": full, "occ_ate_mm": occ})
    # per fold/base/action means
    agg = {}
    for r in rows:
        key = (r["fold"], r["base"], r["action"])
        agg.setdefault(key, {}).setdefault(r["arm"], []).append(r["occ_ate_mm"])
    fold_rows = []
    for (fold, base, act), d in sorted(agg.items()):
        rec = {"fold": fold, "base": base, "action": act,
               "n": len(d.get("Hybrid", []))}
        for arm in ARMS:
            vv = [x for x in d.get(arm, []) if x is not None]
            rec[arm] = float(np.mean(vv)) if vv else None
        fold_rows.append(rec)
    # four-fold mean/std per base/action (fold-level)
    mean_rows = []
    for base in BASES:
        for act in sorted({r["action"] for r in fold_rows}):
            rr = [r for r in fold_rows if r["base"] == base and r["action"] == act]
            rec = {"base": base, "action": act, "n_folds": len(rr)}
            for arm in ARMS:
                vv = [r[arm] for r in rr if r[arm] is not None]
                rec[arm + "_mean"] = float(np.mean(vv)) if vv else None
                rec[arm + "_std"] = float(np.std(vv)) if vv else None
            mean_rows.append(rec)
    # write
    def wcsv(path, rs):
        if not rs:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)
    wcsv(os.path.join(OUT, "Main_Table1_E9_Crossfit_OccAte_FoldAction.csv"), fold_rows)
    wcsv(os.path.join(OUT, "Main_Table2_E9_Crossfit_OccAte_MeanStd.csv"), mean_rows)
    wcsv(os.path.join(OUT, "Full_Table1_E9_Crossfit_PerSession.csv"), rows)
    # bar: fold-level mean by action, two bases
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    acts = sorted({r["action"] for r in mean_rows})
    x = np.arange(len(acts)); width = 0.13
    for ax, base in zip(axes, BASES):
        rr = {r["action"]: r for r in mean_rows if r["base"] == base}
        for k, arm in enumerate(["Step2Motion", "Anysolo-TI", "VT", "Hybrid",
                                 "Hybrid+VT0.50", "Hybrid+VT0.75"]):
            vals = [rr[a][arm + "_mean"] for a in acts]
            errs = [rr[a][arm + "_std"] for a in acts]
            ax.bar(x + (k - 2.5) * width, vals, width, color=COLORS[arm],
                   label=arm if base == BASES[0] else None,
                   yerr=errs, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(acts)
        ax.set_title("%s — fold mean±std" % base)
        ax.grid(True, axis="y", color="#e6e5e1", lw=.6)
    axes[0].set_ylabel("occlusion ATE (mm)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Main_Fig1_E9_Crossfit_OccAte.png"), dpi=200)
    plt.close(fig)
    # report
    lines = ["# E9 四折 crossfit 场景遮挡轨迹（2026-11-08）", "",
             "## 口径", "",
             "- 四折 LSO woS13；每折 test 的 scene-occ manifest 会话（03/05/08/09）。",
             "- V7_Cv5：E5 四折 scene 预测；V8_Cv5：四折 V8 ckpt 在 `cam3_occ_scene` 上的新推理。",
             "- S2M：逐折 ckpt 的 per-fold 预测（`gait_model_smpl24_<fold>`），不是零样本。",
             "- Hybrid：逐折 `_last` 模型；S13/S14 不进对应折 train；四折 test 受试者留出。",
             "- 指标：遮挡内 ATE（full_event 并集）；四折报 fold-level mean±std，不池化。",
             "- VT 是 occluded-V prior，不是视觉证据；α 固定 0.50/0.75。", "",
             "## 四折 fold-level mean±std（mm）", "",
             "| 基座 | 动作 | " + " | ".join(ARMS) + " |",
             "|---|---|" + "---|" * len(ARMS)]
    for r in mean_rows:
        lines.append("| %s | %s | %s |" % (r["base"], r["action"], " | ".join(
            "%.1f±%.1f" % (r[a + "_mean"], r[a + "_std"])
            if r[a + "_mean"] is not None else "—" for a in ARMS)))
    lines += ["", "## 文件", "",
              "- `Main_Table1_E9_Crossfit_OccAte_FoldAction.csv`：逐折×动作",
              "- `Main_Table2_E9_Crossfit_OccAte_MeanStd.csv`：四折 mean±std",
              "- `Full_Table1_E9_Crossfit_PerSession.csv`：逐会话全臂",
              "- `Main_Fig1_E9_Crossfit_OccAte.png`：双基座柱图"]
    open(os.path.join(DETAIL, "E9_四折Crossfit说明.md"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
