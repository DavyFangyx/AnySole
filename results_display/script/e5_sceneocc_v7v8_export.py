"""Export the completed E5 scene-occlusion cells for V7 and V8.

Sources:
  V7_Cv5: results/diagnostics/20261107_e5_sceneocc_full/<fold>_{scene,clean}/
  V7_Cv7 / V8_*: results/cells/<model>_<fold>_s1_eval{scene,sceneclean}/

Outputs:
  results_display/Etest_crossfit/E5_OcclusionRobustness/SceneOcc/      (V7_Cv5 + V7_Cv7)
  results_display/Etest_crossfit_v8/E5_OcclusionRobustness/SceneOcc/   (six V8 models)

Tables mirror the existing V7_Cv5 scene export; all folds are 4-fold LSO,
fS6S7 is the S13-out fold (V7_Cv7 uses the fS6S7S13 checkpoint cell).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results/diagnostics/20261107_e5_sceneocc_full"
CELLS = ROOT / "results/cells"
MANIFEST = DIAG / "manifests"
OUT_V7 = ROOT / "results_display/Etest_crossfit/E5_OcclusionRobustness/SceneOcc"
OUT_V8 = ROOT / "results_display/Etest_crossfit_v8/E5_OcclusionRobustness/SceneOcc"

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODES = ("clean", "scene")
CONDS = ("V2M", "VT2M", "T2M")

MODEL_ORDER_V7 = ("V7_Cv5", "V7_Cv7")
MODEL_ORDER_V8 = ("V8_Cv5", "V8_Cv4", "V8_Cv2", "V8_Cv6", "V8_Cv7", "V8_Cv8")

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
SESSION_KEYS = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "GlobalMPJPE": "global_mpjpe_mm",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "PA-MPJPE_anklefoot": "PA-MPJPE_anklefoot",
}
ACTION_NAMES = {"03": "折返走慢", "04": "折返走快", "05": "绕圈跑",
                "08": "折返跑", "09": "横移"}
STATE_ORDER = ("clean", "partial", "full")


def fold_map(model: str) -> dict:
    if model == "V7_Cv7":
        return {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
                "fS6S7": "fS6S7S13", "fS5S6S8": "fS5S6S8"}
    return {f: f for f in FOLDS}


def metric_dir(model: str, fold: str, mode: str) -> Path:
    if model == "V7_Cv5":
        return DIAG / ("%s_%s" % (fold, mode)) / "metrics"
    suffix = "evalscene" if mode == "scene" else "evalsceneclean"
    cell = "%s_%s_s1_%s" % (model, fold_map(model)[fold], suffix)
    return CELLS / cell / "metrics"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_test(model: str, fold: str, mode: str) -> dict:
    return load_json(metric_dir(model, fold, mode) / "test.json")["metrics"]


def load_sessions(model: str, fold: str, mode: str, cond: str) -> dict:
    data = load_json(metric_dir(model, fold, mode) / "session_detail_test.json")
    return {r["session_id"]: r for r in data["conditions"][cond]}


def load_windows(model: str, fold: str, mode: str, cond: str) -> list:
    data = load_json(metric_dir(model, fold, mode) / "window_detail_test.json")
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
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else float("nan")


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


def table1(models, out: Path):
    header = ["模型", "折"] + [label for _, label in METRICS]
    rows = []
    for model in models:
        for fold in FOLDS:
            v = load_test(model, fold, "scene")
            row = [model, fold]
            for key, label in METRICS:
                a, b = v.get("V2M", {}).get(key), v.get("VT2M", {}).get(key)
                row.append(fmt(None if a is None or b is None else float(b) - float(a)))
            rows.append(row)
        row = [model, "mean±std"]
        for key, label in METRICS:
            vals = []
            for fold in FOLDS:
                v = load_test(model, fold, "scene")
                a, b = v.get("V2M", {}).get(key), v.get("VT2M", {}).get(key)
                if a is not None and b is not None:
                    vals.append(float(b) - float(a))
            row.append(mean_std(vals))
        rows.append(row)
    write_csv(out / "Main_Table1_SceneOcc_TactileGain.csv", header, rows)


def table2(models, out: Path):
    header = ["模型", "折"]
    for cond in ("V2M", "VT2M"):
        header += ["%s Δ(scene−clean) %s" % (cond[:2], label) for _, label in METRICS]
    rows = []
    for model in models:
        for fold in FOLDS:
            s = load_test(model, fold, "scene")
            c = load_test(model, fold, "clean")
            row = [model, fold]
            for cond in ("V2M", "VT2M"):
                for key, label in METRICS:
                    a, b = s.get(cond, {}).get(key), c.get(cond, {}).get(key)
                    row.append(fmt(None if a is None or b is None else float(a) - float(b)))
            rows.append(row)
        row = [model, "mean±std"]
        for cond in ("V2M", "VT2M"):
            for key, label in METRICS:
                vals = []
                for fold in FOLDS:
                    s = load_test(model, fold, "scene").get(cond, {}).get(key)
                    c = load_test(model, fold, "clean").get(cond, {}).get(key)
                    if s is not None and c is not None:
                        vals.append(float(s) - float(c))
                row.append(mean_std(vals))
        rows.append(row)
    write_csv(out / "Main_Table2_SceneOcc_Degradation.csv", header, rows)


def table3(models, out: Path):
    header = ["模型", "折"]
    for metric in ("PA-MPJPE", "MPJPE", "GlobalMPJPE"):
        for cond in CONDS:
            header.append("%s %s" % (metric, cond))
    rows = []
    for model in models:
        for fold in FOLDS:
            s = load_test(model, fold, "scene")
            rows.append([model, fold] + [fmt(metric_of(s.get(c, {}), m))
                                         for m in ("PA-MPJPE", "MPJPE", "GlobalMPJPE")
                                         for c in CONDS])
        row = [model, "mean±std"]
        for metric in ("PA-MPJPE", "MPJPE", "GlobalMPJPE"):
            for cond in CONDS:
                vals = [metric_of(load_test(model, f, "scene").get(cond, {}), metric)
                        for f in FOLDS]
                row.append(mean_std(vals))
        rows.append(row)
    write_csv(out / "Main_Table3_SceneOcc_Absolute.csv", header, rows)


def table4(models, out: Path, man: dict):
    header = ["模型", "动作", "每折会话数"] + list(MAIN_METRICS)
    rows = []
    for model in models:
        for action in ("03", "04", "05", "08", "09"):
            per_metric = {m: [] for m in MAIN_METRICS}
            ns = []
            for fold in FOLDS:
                sv = load_sessions(model, fold, "scene", "V2M")
                svt = load_sessions(model, fold, "scene", "VT2M")
                sids = [sid for sid in sv if sid in svt
                        and man[sid]["action"] == action]
                ns.append(len(sids))
                if not sids:
                    continue
                for metric in MAIN_METRICS:
                    key = SESSION_KEYS[metric]
                    v = wmean([sv[s]["metrics"][key] for s in sids],
                              [sv[s]["n_frames"] for s in sids])
                    vt = wmean([svt[s]["metrics"][key] for s in sids],
                               [svt[s]["n_frames"] for s in sids])
                    per_metric[metric].append(vt - v)
            rows.append([model, "%s %s" % (action, ACTION_NAMES[action]),
                         "/".join(str(n) for n in ns)]
                        + [mean_std(per_metric[m]) for m in MAIN_METRICS])
    write_csv(out / "Main_Table4_SceneOcc_PerAction.csv", header, rows)


def table5(models, out: Path, man: dict):
    header = ["模型", "状态", "n窗/折", "PA_V", "PA_VT", "PA_T", "VT−V"]
    rows = []
    for model in models:
        for st in STATE_ORDER:
            vals = {"V2M": [], "VT2M": [], "T2M": []}
            ns = []
            for fold in FOLDS:
                counts = []
                for cond in CONDS:
                    rs = [r for r in load_windows(model, fold, "scene", cond)
                          if man.get(r["session_id"], {}).get("windows") and
                          next((w for w in man[r["session_id"]]["windows"]
                                if int(w["start"]) == r["start"]), {}).get("state") == st]
                    if rs:
                        vals[cond].append(float(np.mean([r["pa_mpjpe_mm"] for r in rs])))
                        counts.append(len(rs))
                if counts:
                    ns.append(int(np.mean(counts)))
            rows.append([model, st, "/".join(str(n) for n in ns),
                         mean_std(vals["V2M"]), mean_std(vals["VT2M"]),
                         mean_std(vals["T2M"]),
                         mean_std([vt - v for v, vt in zip(vals["V2M"], vals["VT2M"])])])
    write_csv(out / "Main_Table5_SceneOcc_VisibilityBins.csv", header, rows)


def table5b(models, out: Path, man: dict):
    """Per-fold clean/partial/full PA for every model (dose-response detail)."""
    header = ["模型", "折", "状态", "n窗", "PA_V", "PA_VT", "PA_T", "VT−V"]
    rows = []
    for model in models:
        for fold in FOLDS:
            for st in STATE_ORDER:
                vals = {}
                n = 0
                for cond in CONDS:
                    rs = [r for r in load_windows(model, fold, "scene", cond)
                          if next((w for w in man[r["session_id"]]["windows"]
                                   if int(w["start"]) == r["start"]), {}).get("state") == st]
                    vals[cond] = float(np.mean([r["pa_mpjpe_mm"] for r in rs])) if rs else None
                    n = len(rs)
                if vals.get("V2M") is None:
                    continue
                rows.append([model, fold, st, n,
                             fmt(vals["V2M"]), fmt(vals["VT2M"]), fmt(vals["T2M"]),
                             fmt(vals["VT2M"] - vals["V2M"])])
    write_csv(out / "Main_Table5b_SceneOcc_FullBinPerFold.csv", header, rows)


def table6(models, out: Path, man: dict):
    header = ["模型", "条件", "full窗 PA", "all窗 PA"]
    rows = []
    for model in models:
        for label in ("V", "VT", "T", "gate@0.20", "oracle min(V,T)"):
            full_vals, all_vals = [], []
            for fold in FOLDS:
                v = {(r["session_id"], r["start"]): r for r in load_windows(model, fold, "scene", "V2M")}
                vt = {(r["session_id"], r["start"]): r for r in load_windows(model, fold, "scene", "VT2M")}
                t = {(r["session_id"], r["start"]): r for r in load_windows(model, fold, "scene", "T2M")}
                keys = set(v) & set(vt) & set(t)
                full = [k for k in keys
                        if next((w for w in man[k[0]]["windows"] if int(w["start"]) == k[1]),
                                {}).get("state") == "full"]
                def pick(k):
                    if label == "V": return v[k]["pa_mpjpe_mm"]
                    if label == "VT": return vt[k]["pa_mpjpe_mm"]
                    if label == "T": return t[k]["pa_mpjpe_mm"]
                    if label.startswith("gate"):
                        vis = next(w["vis_mean"] for w in man[k[0]]["windows"]
                                   if int(w["start"]) == k[1])
                        return t[k]["pa_mpjpe_mm"] if vis < 0.20 else v[k]["pa_mpjpe_mm"]
                    return min(v[k]["pa_mpjpe_mm"], t[k]["pa_mpjpe_mm"])
                if full:
                    full_vals.append(float(np.mean([pick(k) for k in full])))
                if keys:
                    all_vals.append(float(np.mean([pick(k) for k in keys])))
            rows.append([model, label, mean_std(full_vals), mean_std(all_vals)])
    write_csv(out / "Main_Table6_SceneOcc_Gate.csv", header, rows)


def table7(models, out: Path, man: dict):
    rows = []
    for model in models:
        for fold in FOLDS:
            by = defaultdict(dict)
            for cond in CONDS:
                for r in load_windows(model, fold, "scene", cond):
                    if next((w for w in man[r["session_id"]]["windows"]
                             if int(w["start"]) == r["start"]), {}).get("state") == "full":
                        by[r["session_id"]].setdefault(cond, []).append(r["pa_mpjpe_mm"])
            for sid, vals in sorted(by.items()):
                if not all(c in vals for c in CONDS):
                    continue
                v = float(np.mean(vals["V2M"])); vt = float(np.mean(vals["VT2M"])); t = float(np.mean(vals["T2M"]))
                rows.append([model, fold, sid, man[sid]["action"], len(vals["V2M"]),
                             fmt(v), fmt(vt), fmt(t), fmt(vt - v), fmt(t - v)])
    write_csv(out / "Main_Table7_SceneOcc_PerSessionFull.csv",
              ["模型", "折", "会话", "动作", "n_full", "PA_V", "PA_VT", "PA_T", "VT−V", "T−V"], rows)


def full_pooled(models, out: Path):
    rows = []
    for model in models:
        for fold in FOLDS:
            for mode in MODES:
                m = load_test(model, fold, mode)
                for cond in CONDS:
                    for key, label in METRICS:
                        rows.append([model, fold, mode, cond, label,
                                     fmt(m.get(cond, {}).get(key))])
    write_csv(out / "Full_Table1_SceneOcc_Pooled.csv",
              ["模型", "折", "模式", "条件", "指标", "值"], rows)


def plot(models, out: Path, man: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF = "#fcfcfb"; INK = "#0b0b0b"; MUT = "#52514e"; GRID = "#e1e0d9"
    BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

    def style(ax):
        ax.set_facecolor(SURF); ax.grid(alpha=0.5, lw=0.6, color=GRID)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUT, labelsize=9)

    # full-bin PA per model, V/VT/T
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), facecolor=SURF)
    ax = axes[0]; style(ax)
    xs = np.arange(len(models))
    for cond, color, marker, lab in (("V2M", BLUE, "o", "V"),
                                     ("VT2M", ORANGE, "s", "VT"),
                                     ("T2M", AQUA, "^", "T")):
        means, foldpts = [], []
        for model in models:
            vals = []
            for fold in FOLDS:
                rs = [r for r in load_windows(model, fold, "scene", cond)
                      if next((w for w in man[r["session_id"]]["windows"]
                               if int(w["start"]) == r["start"]), {}).get("state") == "full"]
                if rs:
                    vals.append(float(np.mean([r["pa_mpjpe_mm"] for r in rs])))
            means.append(float(np.mean(vals)) if vals else np.nan)
            foldpts.append(vals)
        off = {"V2M": -0.18, "VT2M": 0.0, "T2M": 0.18}[cond]
        ax.plot(xs + off, means, marker, color=color, lw=1.4, ms=8, label=lab)
        for xi, vs in zip(xs + off, foldpts):
            if vs:
                ax.plot(np.full(len(vs), xi), vs, "o", color=color, ms=3, alpha=0.25)
    ax.set_xticks(xs); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("PA-MPJPE full-window (mm)", fontsize=10, color=MUT)
    ax.set_title("A. Full-occlusion error", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[1]; style(ax)
    means, foldpts = [], []
    for model in models:
        vals = []
        for fold in FOLDS:
            v = {(r["session_id"], r["start"]): r for r in load_windows(model, fold, "scene", "V2M")}
            vt = {(r["session_id"], r["start"]): r for r in load_windows(model, fold, "scene", "VT2M")}
            keys = [k for k in (set(v) & set(vt))
                    if next((w for w in man[k[0]]["windows"] if int(w["start"]) == k[1]),
                            {}).get("state") == "full"]
            if keys:
                vals.append(float(np.mean([vt[k]["pa_mpjpe_mm"] - v[k]["pa_mpjpe_mm"] for k in keys])))
        means.append(float(np.mean(vals)) if vals else np.nan)
        foldpts.append(vals)
    ax.bar(xs, means, color=BLUE, alpha=0.75, width=0.55)
    for xi, vs in zip(xs, foldpts):
        if vs:
            ax.plot(np.full(len(vs), xi), vs, "o", color=NEUTRAL, ms=4, alpha=0.55)
    ax.axhline(0, color="#9a9890", lw=0.9, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Δ(VT−V) full-window PA (mm)", fontsize=10, color=MUT)
    ax.set_title("B. Tactile gain at full occlusion", fontsize=11, color=INK, loc="left")
    fig.suptitle("E5 scene occlusion: model summary (4-fold LSO, mean; dots = folds)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "Main_Fig1_SceneOcc_ModelSummary.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("[png] %s" % path)

    # degradation curve for the first model in the group
    model = models[0]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor=SURF); style(ax)
    xvals, curves = [], {"V2M": [], "VT2M": [], "T2M": []}
    for st in STATE_ORDER:
        for cond in curves:
            vals = []
            for fold in FOLDS:
                rs = [r for r in load_windows(model, fold, "scene", cond)
                      if next((w for w in man[r["session_id"]]["windows"]
                               if int(w["start"]) == r["start"]), {}).get("state") == st]
                if rs:
                    vals.append(float(np.mean([r["pa_mpjpe_mm"] for r in rs])))
            curves[cond].append(float(np.mean(vals)) if vals else np.nan)
        vs = []
        for fold in FOLDS:
            rs = [r for r in load_windows(model, fold, "scene", "V2M")
                  if next((w for w in man[r["session_id"]]["windows"]
                           if int(w["start"]) == r["start"]), {}).get("state") == st]
            vs.extend([w["vis_mean"] for r in rs
                       for w in man[r["session_id"]]["windows"]
                       if int(w["start"]) == r["start"]])
        xvals.append(float(np.mean(vs)) if vs else np.nan)
    for cond, color, marker, lab in (("V2M", BLUE, "o", "V"),
                                     ("VT2M", ORANGE, "s", "VT"),
                                     ("T2M", AQUA, "^", "T")):
        ax.plot(xvals, curves[cond], "-" + marker, color=color, lw=2, ms=7, label=lab)
    ax.set_xlim(1.05, -0.05)
    ax.set_xlabel("Mean visible fraction", fontsize=10, color=MUT)
    ax.set_ylabel("PA-MPJPE (mm, 20-frame window)", fontsize=10, color=MUT)
    ax.set_title("%s: error vs. visibility" % model, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    path = out / "Main_Fig2_SceneOcc_DegradationCurve.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("[png] %s" % path)


def export(models, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifests()
    table1(models, out)
    table2(models, out)
    table3(models, out)
    table4(models, out, man)
    table5(models, out, man)
    table5b(models, out, man)
    table6(models, out, man)
    table7(models, out, man)
    full_pooled(models, out)
    plot(models, out, man)
    print("export complete -> %s" % out)


def main() -> int:
    # Preserve the old V7_Cv5-only scene outputs before overwriting root tables.
    v7_old = [p for p in OUT_V7.glob("Main_*") if p.is_file()]
    if v7_old:
        keep = OUT_V7 / "V7_Cv5_only"
        keep.mkdir(parents=True, exist_ok=True)
        for p in v7_old:
            target = keep / p.name
            if not target.exists():
                shutil.copy2(p, target)
        print("[archive] old V7_Cv5-only files -> %s" % keep)
    export(MODEL_ORDER_V7, OUT_V7)
    export(MODEL_ORDER_V8, OUT_V8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
