#!/usr/bin/env python3
"""E5 V8：遮挡下触觉增益的身体部位 SMPL 热力图（2026-11-09）。

口径
----
- 四折 LSO woS13（fS5S6S11 / fS5S11S12 / fS6S7 / fS5S6S8），只评 test。
- 对每个会话取 V2M 与 VT2M 的 ``predictions/eval_motion/<sid>_{V,VT}2M.npz``，
  用 ``frames.npz`` 的 ``mocap_time_s`` 采样 GT SMPL（与 eval_protocol 同源），
  逐帧做 24 关节相似变换对齐（PA-MPJPE 口径），得到 24 个关节的逐帧误差。
- 折级 = 该折所有 test 会话按帧数加权平均；四折再取 mean±std。
- ``Δ = VT − V``（mm）：负 = 触觉有帮助，正 = 触觉有害。
- 热力图把 24 关节 Δ 通过 SMPL skinning weights 平滑铺到 6890 顶点，
  在零位姿 canonical mesh 上渲染。蓝 = 触觉增益，红 = 触觉伤害，灰 = 0。

产物
----
``results_display/Etest_crossfit_v8/E5_OcclusionRobustness/SMPLHeatmap/``
  Main_Fig_SMPL_Gain_Front.png      Cv5 vs Cv7 × 六档，正面 3/4
  Main_Fig_SMPL_Gain_Back.png       背面 3/4
  Main_Fig_SMPL_Gain_Occ040.png     occ040 机制特写（正/侧/背）
  panels/                           单格 PNG
  detail/SMPL_Gain_PerJoint_FoldLevel.csv
  detail/SMPL_Gain_RegionSummary.csv
  detail/SMPL_Gain_Validation.csv  与 E5 Table2 的 occ040 对账
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anysole.data.smpl_io import _rest_joints, load_smpl  # noqa: E402
from anysole.types import JOINT_HIERARCHY, JOINT_NAMES  # noqa: E402
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402
from anysole.utils.metrics import similarity_align  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402
from utils.motion_io import smpl_yup_to_display  # noqa: E402
from utils.smpl_mesh import (_body_model, render_mesh_frame, smpl_faces,  # noqa: E402
                             smpl_vertices)

CELLS = REPO / "results" / "cells"
SPLITS = REPO / "AnysoleWorkspace" / "protocol" / "splits" / "default"
SESSIONS = REPO / "AnysoleWorkspace" / "shared" / "facts" / "sessions" / "cam3"
OUT = REPO / "results_display" / "Etest_crossfit_v8" / "E5_OcclusionRobustness" / "SMPLHeatmap"
E5_GAIN_TABLE = (REPO / "results_display" / "Etest_crossfit_v8" /
                 "E5_OcclusionRobustness" / "Full_Table2_E5_TactileGain.csv")

FOLDS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12.csv",
    "fS5S11S12": "splits_test_S5S11S12_val_S13.csv",
    "fS6S7": "splits_test_S6S7_val_S14.csv",
    "fS5S6S8": "splits_test_S5S6S8_val_S7.csv",
}
LEVELS = ("000", "020", "040", "060", "080", "100")
LEVEL_LABEL = {"000": "clean", "020": "occ020", "040": "occ040",
               "060": "occ060", "080": "occ080", "100": "occ100"}
MODEL_LABEL = {"V8_Cv5": "V8_Cv5 (double gate)", "V8_Cv7": "V8_Cv7 (no gate)"}

# Diverging pair from the project palette: blue (benefit) <-> red (harm),
# neutral gray at zero.  Equal step count per arm.
CMAP_STOPS = [
    (0.00, "#0d366b"),
    (0.15, "#1c5cab"),
    (0.30, "#2a78d6"),
    (0.42, "#86b6ef"),
    (0.50, "#f0efec"),
    (0.58, "#f3a6a4"),
    (0.70, "#e34948"),
    (0.85, "#b02a2a"),
    (1.00, "#7f1d1d"),
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


# ---------------------------------------------------------------- sessions
def _test_ids(fold: str) -> list:
    with (SPLITS / FOLDS[fold]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sorted({str(row["test"]).strip() for row in rows
                   if str(row.get("test", "")).strip()})


def _session_dir(sid: str) -> Path:
    hits = sorted(SESSIONS.glob("*/*/%s/session.json" % sid))
    if not hits:
        raise FileNotFoundError("shared session not found: %s" % sid)
    return hits[0].parent


def _smpl_path(meta: dict, sid: str) -> Path:
    value = str(meta.get("source_files", {}).get("smpl", ""))
    if value:
        try:
            path = resolve_uri(value, must_exist=True)
            if path.is_file():
                return path
        except Exception:
            pass
    for root in (Path("/data/lizhe/projects/Tactile/Mocap"),):
        hits = sorted(root.glob("**/%s/motion_neutral_smpl.npz" % sid))
        if hits:
            return hits[0]
    raise FileNotFoundError("SMPL archive not found for %s (source=%r)" % (sid, value))


class SessionCache:
    """GT joints + frame validity, loaded once per test session."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def get(self, sid: str) -> dict:
        if sid in self._cache:
            return self._cache[sid]
        session_dir = _session_dir(sid)
        meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        frames = np.load(session_dir / "frames.npz")
        t_mocap = np.asarray(frames["mocap_time_s"], dtype=np.float64)
        valid = (np.asarray(frames["valid"]).astype(bool)
                 & ~np.asarray(frames["fake"]).astype(bool))
        motion = load_smpl(_smpl_path(meta, sid), query_t=t_mocap)
        joints = fk_pose6d_np(motion["pose_6d"], motion["trans_m"],
                              motion["offsets_m"], motion["parents"])
        if joints.shape[0] != valid.size:
            raise ValueError("%s: GT joints %d != frames %d"
                             % (sid, joints.shape[0], valid.size))
        info = {"joints": joints, "valid": valid,
                "t_mocap": t_mocap, "source": str(_smpl_path(meta, sid))}
        self._cache[sid] = info
        return info


# ---------------------------------------------------------------- compute
def _pred_joints_native(npz_path: Path) -> tuple:
    data = np.load(npz_path, allow_pickle=True)
    display = np.asarray(data["joint_xyz_world"], dtype=np.float64)
    # eval.py stores z-up display joints; inverse of (x, -z, y).
    native = np.stack([display[..., 0], display[..., 2], -display[..., 1]], axis=-1)
    return native, np.asarray(data["valid_mask"]).astype(bool)


def compute_level(model: str, level: str, cache: SessionCache) -> list:
    suffix = "" if level == "000" else "_evalocc%s" % level
    rows = []
    for fold in FOLDS:
        sums = {"V2M": np.zeros(24, dtype=np.float64),
                "VT2M": np.zeros(24, dtype=np.float64)}
        n_frames = 0
        for sid in _test_ids(fold):
            info = cache.get(sid)
            cell = CELLS / ("%s_%s_s1%s" % (model, fold, suffix))
            paths = {cond: cell / "predictions" / "eval_motion" / ("%s_%s.npz" % (sid, cond))
                     for cond in ("V2M", "VT2M")}
            if not all(path.is_file() for path in paths.values()):
                continue
            mask = info["valid"].copy()
            preds = {}
            for cond, path in paths.items():
                joints, valid = _pred_joints_native(path)
                if joints.shape[0] != mask.size:
                    raise ValueError("%s/%s: predictions %d != frames %d"
                                     % (sid, cond, joints.shape[0], mask.size))
                preds[cond] = joints
                mask &= valid
            if not mask.any():
                continue
            gt = info["joints"][mask]
            for cond, joints in preds.items():
                aligned = similarity_align(joints[mask], gt)
                error = np.linalg.norm(aligned - gt, axis=-1) * 1000.0
                sums[cond] += error.sum(axis=0)
            n_frames += int(mask.sum())
        if not n_frames:
            continue
        for cond in ("V2M", "VT2M"):
            mean = sums[cond] / n_frames
            for index, name in enumerate(JOINT_NAMES):
                rows.append({
                    "模型": model, "档位": level, "折": fold,
                    "关节索引": index, "关节名": name,
                    "条件": cond, "误差_mm": float(mean[index]),
                    "帧数": n_frames,
                })
    return rows


# ---------------------------------------------------------------- aggregate
def aggregate(rows: list) -> tuple:
    """Return per-joint 4-fold mean/std and per-region summary rows."""
    table: dict = {}
    for row in rows:
        table.setdefault((row["模型"], row["档位"], row["折"], row["条件"]), {})[
            row["关节名"]] = float(row["误差_mm"])
    per_joint = []
    for model in sorted({row["模型"] for row in rows}):
        for level in LEVELS:
            folds = [fold for fold in FOLDS
                     if (model, level, fold, "V2M") in table]
            if not folds:
                continue
            # fold-paired delta: keep fold axis
            v_fold = np.asarray([[table[(model, level, fold, "V2M")][name]
                                  for name in JOINT_NAMES] for fold in folds])
            vt_fold = np.asarray([[table[(model, level, fold, "VT2M")][name]
                                   for name in JOINT_NAMES] for fold in folds])
            delta_fold = vt_fold - v_fold
            for index, name in enumerate(JOINT_NAMES):
                per_joint.append({
                    "模型": model, "档位": level, "关节索引": index,
                    "关节名": name,
                    "V_mean_mm": float(v_fold[:, index].mean()),
                    "VT_mean_mm": float(vt_fold[:, index].mean()),
                    "Delta_mean_mm": float(delta_fold[:, index].mean()),
                    "Delta_std_mm": float(delta_fold[:, index].std(ddof=0)),
                    "n_folds": len(folds),
                })
    regions = []
    for model in sorted({row["模型"] for row in rows}):
        for level in LEVELS:
            folds = [fold for fold in FOLDS
                     if (model, level, fold, "V2M") in table]
            if not folds:
                continue
            v_fold = np.asarray([[table[(model, level, fold, "V2M")][name]
                                  for name in JOINT_NAMES] for fold in folds])
            vt_fold = np.asarray([[table[(model, level, fold, "VT2M")][name]
                                   for name in JOINT_NAMES] for fold in folds])
            for region, joints in JOINT_HIERARCHY.items():
                idx = list(joints)
                delta = vt_fold[:, idx].mean(axis=1) - v_fold[:, idx].mean(axis=1)
                regions.append({
                    "模型": model, "档位": level, "区域": region,
                    "V_mean_mm": float(v_fold[:, idx].mean()),
                    "VT_mean_mm": float(vt_fold[:, idx].mean()),
                    "Delta_mean_mm": float(delta.mean()),
                    "Delta_std_mm": float(delta.std(ddof=0)),
                    "n_folds": len(folds),
                })
    return per_joint, regions


# ---------------------------------------------------------------- validation
def validate_occ040(regions: list) -> list:
    if not E5_GAIN_TABLE.is_file():
        return []
    wanted = {("PA-MPJPE_anklefoot↓", "anklefoot"), ("PA-MPJPE_upper↓", "upper")}
    computed = {(row["模型"], row["档位"], row["区域"]): row
                for row in regions}
    out = []
    with E5_GAIN_TABLE.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("档位") != "occ040" or row.get("行") != "Δ(VT−V)":
                continue
            for column, region in wanted:
                if column not in row:
                    continue
                key = (row["模型"], "040", region)
                if key not in computed:
                    continue
                ours = computed[key]["Delta_mean_mm"]
                ref = float(str(row[column]).split("±")[0])
                out.append({"模型": row["模型"], "区域": region,
                            "表值_mm": ref, "复算_mm": round(ours, 4),
                            "差_mm": round(ours - ref, 4)})
    return out


# ---------------------------------------------------------------- render
def _canonical_mesh() -> tuple:
    body = _body_model()
    rest = _rest_joints({"betas": np.zeros(10, dtype=np.float64)})
    pose = np.zeros((1, 24, 3), dtype=np.float64)
    trans = rest[0][None]
    verts = smpl_vertices(pose, trans, np.zeros(10), rest)[0]
    weights = np.asarray(body["weights"], dtype=np.float64)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return verts, smpl_faces(), weights


def _vertex_values(delta_joint: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Smooth per-joint values over the mesh with SMPL skinning weights."""
    return weights @ np.asarray(delta_joint, dtype=np.float64)


def _colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tactile_gain", CMAP_STOPS)


def _crop_body(image: np.ndarray, pad: int = 10) -> np.ndarray:
    """Trim render padding so the mesh fills its panel."""
    background = np.asarray([int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)])
    mask = np.any(np.abs(image.astype(np.int16) - background) > 6, axis=2)
    ys, xs = np.where(mask)
    if not len(ys):
        return image
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, image.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def _render_one(verts_display: np.ndarray, faces: np.ndarray,
                values: np.ndarray, vmax: float, azim: float, elev: float,
                size: int = 480) -> np.ndarray:
    cmap = _colormap()
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
    colors = cmap(norm(values))[:, :3]
    image = render_mesh_frame(
        verts_display, faces, vertex_colors=colors, ambient=0.85,
        light_intensity=0.8, bg_color=SURFACE, size=(size, size),
        elev=elev, azim=azim)
    return _crop_body(image)


def _panel_values(per_joint: list, model: str, level: str,
                  weights: np.ndarray) -> np.ndarray:
    rows = [row for row in per_joint
            if row["模型"] == model and row["档位"] == level]
    rows.sort(key=lambda row: row["关节索引"])
    if len(rows) != 24:
        raise ValueError("%s/%s: expected 24 joints, got %d" % (model, level, len(rows)))
    delta = np.asarray([row["Delta_mean_mm"] for row in rows], dtype=np.float64)
    return _vertex_values(delta, weights)


def _compose_grid(images: np.ndarray, row_labels: list, col_labels: list,
                  vmax: float, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_rows, n_cols = images.shape[:2]
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.25 * n_cols + 1.6, 2.55 * n_rows + 0.9),
                             facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(images[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10.5, color=INK, pad=4)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11.5, color=INK,
                                labelpad=8)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.02, ha="left", y=0.995)
    fig.text(0.5, 0.012,
             "Δ = PA-MPJPE(VT) − PA-MPJPE(V), mm; blue = tactile benefit, "
             "red = tactile harm; joint errors smoothed by SMPL skinning weights; "
             "4-fold LSO mean.",
             fontsize=8.5, color=MUTED, ha="center")
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=_colormap())
    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("ΔPA-MPJPE (mm)", fontsize=9.5, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.055, right=0.885, top=0.89, bottom=0.10,
                        wspace=0.035, hspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def render(per_joint: list, models: list, vmax: Optional[float] = None) -> None:
    verts, faces, weights = _canonical_mesh()
    verts_display = smpl_yup_to_display(verts)
    values = {(model, level): _panel_values(per_joint, model, level, weights)
              for model in models for level in LEVELS}
    if vmax is None:
        peak = max(float(np.abs(v).max()) for v in values.values())
        vmax = float(np.ceil(peak / 5.0) * 5.0)
        vmax = max(vmax, 5.0)
    print("[scale] vmax = ±%.1f mm" % vmax)

    views = {
        "front": dict(azim=-64.0, elev=8.0),
        "back": dict(azim=116.0, elev=8.0),
        "side": dict(azim=2.0, elev=6.0),
    }
    for view_name, cfg in views.items():
        images = np.empty((len(models), len(LEVELS)), dtype=object)
        for r, model in enumerate(models):
            for c, level in enumerate(LEVELS):
                image = _render_one(verts_display, faces, values[(model, level)],
                                    vmax, **cfg)
                images[r, c] = image
                panel = OUT / "panels" / ("%s_%s_%s.png"
                                          % (model, LEVEL_LABEL[level], view_name))
                panel.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.fromarray(image).save(panel)
        title = "Tactile effect by body region under 2D occlusion — %s view" % view_name
        _compose_grid(images, [MODEL_LABEL[m] for m in models],
                      [LEVEL_LABEL[level] for level in LEVELS],
                      vmax, title,
                      OUT / ("Main_Fig_SMPL_Gain_%s.png" % view_name.capitalize()))

    # occ040 mechanism close-up: front / side / back per model.
    close = np.empty((len(models), 3), dtype=object)
    for r, model in enumerate(models):
        for c, view_name in enumerate(("front", "side", "back")):
            close[r, c] = _render_one(verts_display, faces,
                                      values[(model, "040")], vmax,
                                      **views[view_name])
    _compose_grid(close, [MODEL_LABEL[m] for m in models],
                  ["front 3/4", "side", "back 3/4"], vmax,
                  "occ040: body-region structure of tactile gain and harm",
                  OUT / "Main_Fig_SMPL_Gain_Occ040.png")


# ---------------------------------------------------------------- main
def write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[key] for key in header])


def main() -> int:
    global LEVELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="V8_Cv5,V8_Cv7",
                        help="comma-separated model families")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--vmax", type=float, default=20.0,
                        help="fixed symmetric color scale in mm (default: 20; "
                             "values outside clip)")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    LEVELS = tuple(levels)

    detail = OUT / "detail"
    detail.mkdir(parents=True, exist_ok=True)
    joint_path = detail / "SMPL_Gain_PerJoint_FoldLevel.csv"
    region_path = detail / "SMPL_Gain_RegionSummary.csv"

    if not args.render_only:
        cache = SessionCache()
        rows = []
        for model in models:
            for level in LEVELS:
                print("[compute] %s %s" % (model, LEVEL_LABEL.get(level, level)))
                rows.extend(compute_level(model, level, cache))
        write_csv(joint_path,
                  ["模型", "档位", "折", "关节索引", "关节名", "条件", "误差_mm", "帧数"],
                  rows)
        _, regions = aggregate(rows)
        write_csv(region_path,
                  ["模型", "档位", "区域", "V_mean_mm", "VT_mean_mm",
                   "Delta_mean_mm", "Delta_std_mm", "n_folds"],
                  regions)
        checks = validate_occ040(regions)
        if checks:
            write_csv(detail / "SMPL_Gain_Validation.csv",
                      ["模型", "区域", "表值_mm", "复算_mm", "差_mm"], checks)
            for row in checks:
                print("[check] %s %s  table=%.3f ours=%.3f diff=%+.3f"
                      % (row["模型"], row["区域"], row["表值_mm"],
                         row["复算_mm"], row["差_mm"]))
        if args.compute_only:
            return 0
    else:
        if not joint_path.is_file():
            raise SystemExit("missing %s; run without --render-only first" % joint_path)
        rows = []
        with joint_path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows.append({
                    "模型": row["模型"], "档位": row["档位"], "折": row["折"],
                    "关节索引": int(row["关节索引"]), "关节名": row["关节名"],
                    "条件": row["条件"], "误差_mm": float(row["误差_mm"]),
                    "帧数": int(row["帧数"]),
                })

    per_joint, _ = aggregate(rows)
    render(per_joint, models, vmax=args.vmax)
    print("[done] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
