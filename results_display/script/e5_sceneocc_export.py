"""E5 scene-occlusion full-run export (4-fold LSO, V7_Cv5).

T2M reference rows: after this exporter, run
``results_display/script/e5_t_bifoot_reference_replace.py`` to replace the
legacy T2M reference with the shared BiFoot branch.

Reads:
  results/diagnostics/20261107_e5_sceneocc_full/<fold>_{scene,clean}/metrics/
      test.json, window_detail_test.json, session_detail_test.json
  results/diagnostics/20261107_e5_sceneocc_full/manifests/<session>.json

Writes to results_display/Etest_crossfit/E5_SceneOcclusion/:
  Main_Table1_SceneOcc_TactileGain.csv       Δ(VT−V) per fold / mean±std
  Main_Table2_SceneOcc_Degradation.csv      scene − clean per fold / mean±std
  Main_Table3_SceneOcc_Absolute.csv         V/VT/T absolute PA/MPJPE/Global
  Main_Table4_SceneOcc_PerAction.csv        per-action full/overall Δ
  Main_Table5_SceneOcc_VisibilityBins.csv   clean/partial/full bins
  Main_Table6_SceneOcc_Gate.csv             post-hoc visibility gate / oracle
  Main_Table7_SceneOcc_PerSessionFull.csv   per-session full-window V/VT/T
  Full_Table1_SceneOcc_Pooled.csv           long table: fold × mode × cond × metric
  Main_Fig1_SceneOcc_Degradation.png        4-panel summary figure
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results/diagnostics/20261107_e5_sceneocc_full"
MANIFEST = DIAG / "manifests"
OUT = ROOT / "results_display/Etest_crossfit/E5_SceneOcclusion"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODES = ("clean", "scene")
CONDS = ("V2M", "VT2M", "T2M")

# (test.json key, display name)
METRICS = (
    ("PA-MPJPE", "PA-MPJPE"),
    ("MPJPE", "MPJPE"),
    ("GlobalMPJPE（GT Anchored）", "GlobalMPJPE"),
    ("traj_ATE", "traj_ATE"),
    ("FootSliding（GT Anchored）", "FootSliding"),
    ("GroundPen（GT Anchored）", "GroundPen"),
    ("PA-MPJPE_anklefoot", "PA-MPJPE_anklefoot"),
)
MAIN_METRICS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "FootSliding", "GroundPen")

# session_detail keys → display names
SESSION_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "PA-MPJPE_anklefoot": "PA-MPJPE_anklefoot",
}

STATE_ORDER = ("clean", "partial", "full")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_test(fold: str, mode: str) -> dict:
    return load_json(DIAG / ("%s_%s" % (fold, mode)) / "metrics/test.json")["metrics"]


def load_sessions(fold: str, mode: str, cond: str) -> dict:
    data = load_json(DIAG / ("%s_%s" % (fold, mode)) / "metrics/session_detail_test.json")
    return {row["session_id"]: row for row in data["conditions"][cond]}


def load_windows(fold: str, mode: str, cond: str) -> list:
    data = load_json(DIAG / ("%s_%s" % (fold, mode)) / "metrics/window_detail_test.json")
    c = data["conditions"][cond]
    return [dict(session_id=sid, start=int(start), n_frames=int(n),
                 **{k: c["metrics"][k][i] for k in c["metrics"]})
            for i, (sid, start, n) in enumerate(zip(c["session_ids"], c["frame_starts"], c["n_frames"]))]


def load_manifests() -> dict:
    out = {}
    for p in MANIFEST.glob("S*.json"):
        m = load_json(p)
        out[m["session_id"]] = m
    return out


def wmean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return float("nan")
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) == 1:
        return "%.3f" % vals[0]
    return "%.3f±%.3f" % (float(np.mean(vals)), float(np.std(vals)))


def fmt(v):
    if v is None or not np.isfinite(float(v)):
        return ""
    return round(float(v), 3)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("[csv] %s (%d rows)" % (path, len(rows)))


def metric_of(block: dict, name: str):
    for key, label in METRICS:
        if label == name:
            return block.get(key)
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = load_manifests()
    state = {}
    for sid, m in man.items():
        for w in m["windows"]:
            state[(sid, int(w["start"]))] = w

    # ---------- long pooled table ----------
    pooled = {}  # (fold,mode,cond) -> test metrics block
    rows_long = []
    for fold in FOLDS:
        for mode in MODES:
            m = load_test(fold, mode)
            for cond in CONDS:
                block = m.get(cond, {})
                pooled[(fold, mode, cond)] = block
                for key, label in METRICS:
                    rows_long.append([fold, mode, cond, label, fmt(block.get(key))])
    write_csv(OUT / "Full_Table1_SceneOcc_Pooled.csv",
              ["fold", "mode", "cond", "metric", "value"], rows_long)

    # ---------- Main Table 1: Δ(VT−V) scene ----------
    rows = []
    for fold in FOLDS:
        row = [fold]
        for key, label in METRICS:
            v = pooled[(fold, "scene", "V2M")].get(key)
            vt = pooled[(fold, "scene", "VT2M")].get(key)
            row.append(fmt(None if v is None or vt is None else float(vt) - float(v)))
        rows.append(row)
    row = ["mean±std"]
    for j, (key, label) in enumerate(METRICS):
        vals = []
        for fold in FOLDS:
            v = pooled[(fold, "scene", "V2M")].get(key)
            vt = pooled[(fold, "scene", "VT2M")].get(key)
            if v is not None and vt is not None:
                vals.append(float(vt) - float(v))
        row.append(mean_std(vals))
    rows.append(row)
    write_csv(OUT / "Main_Table1_SceneOcc_TactileGain.csv",
              ["fold"] + [label for _, label in METRICS], rows)

    # ---------- Main Table 2: scene − clean ----------
    rows = []
    for fold in FOLDS:
        row = [fold]
        for cond in ("V2M", "VT2M"):
            for key, label in METRICS:
                s = pooled[(fold, "scene", cond)].get(key)
                c = pooled[(fold, "clean", cond)].get(key)
                row.append(fmt(None if s is None or c is None else float(s) - float(c)))
        rows.append(row)
    row = ["mean±std"]
    for cond in ("V2M", "VT2M"):
        for key, label in METRICS:
            vals = []
            for fold in FOLDS:
                s = pooled[(fold, "scene", cond)].get(key)
                c = pooled[(fold, "clean", cond)].get(key)
                if s is not None and c is not None:
                    vals.append(float(s) - float(c))
            row.append(mean_std(vals))
    rows.append(row)
    header = ["fold"] + ["%s Δ(scene−clean) %s" % (cond[:2], label)
                         for cond in ("V2M", "VT2M") for _, label in METRICS]
    write_csv(OUT / "Main_Table2_SceneOcc_Degradation.csv", header, rows)

    # ---------- Main Table 3: absolute PA/MPJPE/Global ----------
    rows = []
    for fold in FOLDS:
        row = [fold]
        for metric in ("PA-MPJPE", "MPJPE", "GlobalMPJPE"):
            for cond in CONDS:
                block = pooled[(fold, "scene", cond)]
                row.append(fmt(metric_of(block, metric)))
        rows.append(row)
    row = ["mean±std"]
    for metric in ("PA-MPJPE", "MPJPE", "GlobalMPJPE"):
        for cond in CONDS:
            vals = [metric_of(pooled[(fold, "scene", cond)], metric) for fold in FOLDS]
            row.append(mean_std(vals))
    rows.append(row)
    header = ["fold"] + ["%s %s" % (metric, cond) for metric in
                         ("PA-MPJPE", "MPJPE", "GlobalMPJPE") for cond in CONDS]
    write_csv(OUT / "Main_Table3_SceneOcc_Absolute.csv", header, rows)

    # ---------- per-session full windows ----------
    per_session = defaultdict(dict)
    for fold in FOLDS:
        for cond in CONDS:
            for row in load_windows(fold, "scene", cond):
                w = state.get((row["session_id"], row["start"]))
                if w is None or w["state"] != "full":
                    continue
                per_session[(fold, row["session_id"])].setdefault(cond, []).append(row["pa_mpjpe_mm"])
    rows = []
    for (fold, sid), vals in sorted(per_session.items()):
        if "V2M" not in vals or "VT2M" not in vals or "T2M" not in vals:
            continue
        v = float(np.mean(vals["V2M"])); vt = float(np.mean(vals["VT2M"])); t = float(np.mean(vals["T2M"]))
        rows.append([fold, sid, man[sid]["action"], len(vals["V2M"]),
                     fmt(v), fmt(vt), fmt(t), fmt(vt - v), fmt(t - v)])
    write_csv(OUT / "Main_Table7_SceneOcc_PerSessionFull.csv",
              ["fold", "session", "action", "n_full_windows", "PA_V", "PA_VT", "PA_T",
               "VT−V", "T−V"], rows)

    # ---------- per-action ----------
    action_rows = []
    for action in ("03", "04", "05", "08", "09"):
        per_metric = {m: [] for m in MAIN_METRICS}
        n_by_fold = []
        for fold in FOLDS:
            sv = load_sessions(fold, "scene", "V2M")
            svt = load_sessions(fold, "scene", "VT2M")
            sids = [sid for sid in sv if man[sid]["action"] == action and sid in svt]
            n_by_fold.append(len(sids))
            if not sids:
                continue
            for metric in MAIN_METRICS:
                key = SESSION_KEYS[metric]
                v = wmean([sv[s]["metrics"][key] for s in sids],
                          [sv[s]["n_frames"] for s in sids])
                vt = wmean([svt[s]["metrics"][key] for s in sids],
                           [svt[s]["n_frames"] for s in sids])
                per_metric[metric].append(vt - v)
        cells = [mean_std(per_metric[m]) for m in MAIN_METRICS]
        action_rows.append([action, "/".join(str(n) for n in n_by_fold)] + cells)
    write_csv(OUT / "Main_Table4_SceneOcc_PerAction.csv",
              ["action", "n_per_fold"] + list(MAIN_METRICS), action_rows)

    # ---------- visibility bins ----------
    bin_rows = []
    for st in STATE_ORDER:
        cells = []
        for metric, wkey in (("PA-MPJPE", "pa_mpjpe_mm"),
                             ("MPJPE", "mpjpe_mm"),
                             ("GlobalMPJPE", "global_mpjpe_mm")):
            for mode, cond in (("clean", "V2M"), ("scene", "V2M"),
                               ("scene", "VT2M"), ("scene", "T2M")):
                vals = []
                for fold in FOLDS:
                    rs = [r for r in load_windows(fold, mode, cond)
                          if state.get((r["session_id"], r["start"]), {}).get("state") == st]
                    if rs:
                        vals.append(float(np.mean([r[wkey] for r in rs])))
                cells.append(mean_std(vals))
        # deltas per fold
        for metric, wkey in (("PA-MPJPE", "pa_mpjpe_mm"),):
            d_v, d_vt = [], []
            for fold in FOLDS:
                sv = { (r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "V2M")
                       if state.get((r["session_id"], r["start"]), {}).get("state") == st }
                svt = { (r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "VT2M")
                        if state.get((r["session_id"], r["start"]), {}).get("state") == st }
                c = { (r["session_id"], r["start"]): r for r in load_windows(fold, "clean", "V2M")
                      if state.get((r["session_id"], r["start"]), {}).get("state") == st }
                keys = set(sv) & set(svt) & set(c)
                if not keys:
                    continue
                d_v.append(np.mean([sv[k][wkey] - c[k][wkey] for k in keys]))
                d_vt.append(np.mean([svt[k][wkey] - c[k][wkey] for k in keys]))
            cells.append(mean_std(d_v))
            cells.append(mean_std(d_vt))
        bin_rows.append([st] + cells)
    header = ["state"]
    for metric in ("PA-MPJPE", "MPJPE", "GlobalMPJPE"):
        header += ["%s cleanV" % metric, "%s sceneV" % metric,
                   "%s sceneVT" % metric, "%s sceneT" % metric]
    header += ["ΔPA sceneV−clean", "ΔPA sceneVT−clean"]
    write_csv(OUT / "Main_Table5_SceneOcc_VisibilityBins.csv", header, bin_rows)

    # per-fold clean/partial/full PA, to show the dose-response directly
    rows = []
    for fold in FOLDS:
        for st in STATE_ORDER:
            vals, n_win = {}, 0
            for cond in ("V2M", "VT2M", "T2M"):
                rs = [r for r in load_windows(fold, "scene", cond)
                      if state.get((r["session_id"], r["start"]), {}).get("state") == st]
                vals[cond] = float(np.mean([r["pa_mpjpe_mm"] for r in rs])) if rs else float("nan")
                n_win = len(rs)
            rows.append([fold, st, n_win,
                         fmt(vals["V2M"]), fmt(vals["VT2M"]), fmt(vals["T2M"]),
                         fmt(vals["VT2M"] - vals["V2M"])])
    write_csv(OUT / "Main_Table5b_SceneOcc_FullBinPerFold.csv",
              ["fold", "state", "n_windows", "PA_V", "PA_VT", "PA_T", "VT−V"], rows)

    # ---------- gate / oracle ----------
    gate_rows = []
    for label in ("V", "VT", "T", "gate@0.05", "gate@0.20", "gate@0.50", "oracle min(V,T)"):
        vals_full, vals_all = [], []
        for fold in FOLDS:
            v = { (r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "V2M") }
            vt = { (r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "VT2M") }
            t = { (r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "T2M") }
            keys = set(v) & set(vt) & set(t)
            full = [k for k in keys if state.get(k, {}).get("state") == "full"]
            def pick(k):
                if label == "V": return v[k]["pa_mpjpe_mm"]
                if label == "VT": return vt[k]["pa_mpjpe_mm"]
                if label == "T": return t[k]["pa_mpjpe_mm"]
                if label == "oracle min(V,T)": return min(v[k]["pa_mpjpe_mm"], t[k]["pa_mpjpe_mm"])
                thr = float(label.split("@")[1])
                return t[k]["pa_mpjpe_mm"] if state[k]["vis_mean"] < thr else v[k]["pa_mpjpe_mm"]
            if full:
                vals_full.append(np.mean([pick(k) for k in full]))
            if keys:
                vals_all.append(np.mean([pick(k) for k in keys]))
        gate_rows.append([label, mean_std(vals_full), mean_std(vals_all)])
    write_csv(OUT / "Main_Table6_SceneOcc_Gate.csv",
              ["condition", "full-window PA", "all-window PA"], gate_rows)

    plot_figure(man, state)
    print("export complete -> %s" % OUT)
    return 0


def plot_figure(man: dict, state: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF = "#fcfcfb"
    INK = "#0b0b0b"
    MUT = "#52514e"
    GRID = "#e1e0d9"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    NEUTRAL = "#52514e"

    def style(ax):
        ax.set_facecolor(SURF)
        ax.grid(alpha=0.5, lw=0.6, color=GRID)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUT, labelsize=9)

    # per-fold bin means
    def bin_means(mode, cond, st, key="pa_mpjpe_mm"):
        vals, xs = [], []
        for fold in FOLDS:
            rs = [r for r in load_windows(fold, mode, cond)
                  if state.get((r["session_id"], r["start"]), {}).get("state") == st]
            if rs:
                vals.append(float(np.mean([r[key] for r in rs])))
                xs.append(float(np.mean([state[(r["session_id"], r["start"])]["vis_mean"] for r in rs])))
        return np.asarray(vals), (float(np.mean(xs)) if xs else np.nan)

    states = STATE_ORDER
    x = []
    curves = {(m, c): [] for m, c in (("clean", "V2M"), ("scene", "V2M"),
                                      ("scene", "VT2M"), ("scene", "T2M"))}
    for st in states:
        for (mode, cond), arr in curves.items():
            v, xv = bin_means(mode, cond, st)
            arr.append(v)
        x.append(bin_means("scene", "V2M", st)[1])
    x = np.asarray(x)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), facecolor=SURF)
    ax = axes[0, 0]
    style(ax)
    specs = [
        (("clean", "V2M"), NEUTRAL, "--o", "V clean (reference)"),
        (("scene", "V2M"), BLUE, "-o", "V scene"),
        (("scene", "VT2M"), ORANGE, "-s", "VT scene"),
        (("scene", "T2M"), AQUA, "-^", "T scene"),
    ]
    for key, color, marker, label in specs:
        vals = np.asarray([np.mean(a) if len(a) else np.nan for a in curves[key]])
        ax.plot(x, vals, marker, color=color, lw=2.0 if key[0] == "scene" else 1.4,
                ms=7, mfc="white" if key == ("clean", "V2M") else color, label=label)
        # fold dots
        for i, st in enumerate(states):
            v, _ = bin_means(*key, st)
            if len(v):
                ax.plot(np.full(len(v), x[i]), v, "o", color=color, ms=3, alpha=0.25)
        ax.annotate(label.split(" (")[0], (x[-1], vals[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, color=color, va="center")
    ax.set_xlim(1.03, -0.03)
    ax.set_xlabel("Mean visible fraction", fontsize=10, color=MUT)
    ax.set_ylabel("PA-MPJPE (mm, 20-frame window)", fontsize=10, color=MUT)
    ax.set_title("A. Error vs. visibility", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[0, 1]
    style(ax)
    for cond, color, marker, label in (("V2M", BLUE, "-o", "V scene − V clean"),
                                       ("VT2M", ORANGE, "-s", "VT scene − VT clean")):
        vals = []
        for st in states:
            v = []
            for fold in FOLDS:
                s = [r for r in load_windows(fold, "scene", cond)
                     if state.get((r["session_id"], r["start"]), {}).get("state") == st]
                c = [r for r in load_windows(fold, "clean", cond)
                     if state.get((r["session_id"], r["start"]), {}).get("state") == st]
                if s and c:
                    v.append(float(np.mean([r["pa_mpjpe_mm"] for r in s]) -
                                  np.mean([r["pa_mpjpe_mm"] for r in c])))
            vals.append(np.mean(v) if v else np.nan)
        ax.plot(x, vals, marker, color=color, lw=2.0, ms=7, label=label)
        ax.annotate(label.split(" scene")[0], (x[-1], vals[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, color=color, va="center")
    ax.axhline(0, color="#9a9890", lw=0.9, ls=":")
    ax.set_xlim(1.03, -0.03)
    ax.set_xlabel("Mean visible fraction", fontsize=10, color=MUT)
    ax.set_ylabel("Δ PA scene − clean (mm)", fontsize=10, color=MUT)
    ax.set_title("B. Degradation attributable to the occluder", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # per-action full-window
    ax = axes[1, 0]
    style(ax)
    actions = ("03", "04", "05", "08", "09")
    for cond, color, marker, label in (("V2M", BLUE, "o", "V"),
                                       ("VT2M", ORANGE, "s", "VT"),
                                       ("T2M", AQUA, "^", "T")):
        means, foldpts = [], []
        for action in actions:
            vals = []
            for fold in FOLDS:
                rs = [r for r in load_windows(fold, "scene", cond)
                      if state.get((r["session_id"], r["start"]), {}).get("state") == "full"
                      and man[r["session_id"]]["action"] == action]
                if rs:
                    vals.append(float(np.mean([r["pa_mpjpe_mm"] for r in rs])))
            means.append(np.mean(vals) if vals else np.nan)
            foldpts.append(vals)
        xs = np.arange(len(actions)) + {"V2M": -0.18, "VT2M": 0.0, "T2M": 0.18}[cond]
        ax.plot(xs, means, marker, color=color, lw=1.4, ms=8, label=label)
        for xi, vs in zip(xs, foldpts):
            if vs:
                ax.plot(np.full(len(vs), xi), vs, "o", color=color, ms=3, alpha=0.25)
    ax.set_xticks(np.arange(len(actions)))
    ax.set_xticklabels(["03\nwalk", "04\nwalk-fast", "05\ncircle", "08\nrun", "09\nlateral"])
    ax.set_ylabel("PA-MPJPE full-window (mm)", fontsize=10, color=MUT)
    ax.set_title("C. Per-action full-occlusion error", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # gate headroom
    ax = axes[1, 1]
    style(ax)
    labels = ("V", "VT", "T", "gate\nvis<0.2", "oracle\nmin(V,T)")
    colors = (BLUE, ORANGE, AQUA, NEUTRAL, NEUTRAL)
    markers = ("o", "s", "^", "D", "P")
    for j, (lab, color, marker) in enumerate(zip(labels, colors, markers)):
        vals = []
        for fold in FOLDS:
            v = {(r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "V2M")}
            vt = {(r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "VT2M")}
            t = {(r["session_id"], r["start"]): r for r in load_windows(fold, "scene", "T2M")}
            keys = [k for k in (set(v) & set(vt) & set(t))
                    if state.get(k, {}).get("state") == "full"]
            if not keys:
                continue
            if lab == "V": vals.append(np.mean([v[k]["pa_mpjpe_mm"] for k in keys]))
            elif lab == "VT": vals.append(np.mean([vt[k]["pa_mpjpe_mm"] for k in keys]))
            elif lab == "T": vals.append(np.mean([t[k]["pa_mpjpe_mm"] for k in keys]))
            elif lab.startswith("gate"):
                vals.append(np.mean([t[k]["pa_mpjpe_mm"] if state[k]["vis_mean"] < 0.2
                                     else v[k]["pa_mpjpe_mm"] for k in keys]))
            else:
                vals.append(np.mean([min(v[k]["pa_mpjpe_mm"], t[k]["pa_mpjpe_mm"]) for k in keys]))
        ax.plot([j] * len(vals), vals, "o", color=color, ms=3, alpha=0.25)
        if vals:
            ax.plot([j], [np.mean(vals)], marker, color=color, ms=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("PA-MPJPE full-window (mm)", fontsize=10, color=MUT)
    ax.set_title("D. Post-hoc visibility gate / oracle", fontsize=11, color=INK, loc="left")

    fig.suptitle("E5 scene occlusion (4-fold LSO, V7_Cv5): graded degradation; "
                 "tactile gain grows with severity, but VT still trails T-only at full occlusion",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "Main_Fig1_SceneOcc_Degradation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("[png] %s" % out)


if __name__ == "__main__":
    raise SystemExit(main())
