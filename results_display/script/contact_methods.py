"""Contact-label variants for the Test5 method comparison (Test5 方案对比).

Each method defines a per-frame, per-foot contact label (1 = on ground).
Generated labels are stored next to ``contact.npy`` as
``contact_<method>.npy`` (same (n, 10) layout, only cols 6/7 filled), so
they can later be swapped into ``contact_gt`` for training without touching
the original files.

Methods (``METHODS`` registry):

    tactile_abs 触觉绝对阈值: 48格压力和 > 100 (原 contact.npy 口径)
    bvh_soft    BVH 运动学, 与 losses.soft_contact_from_keypoints 同构:
                sigmoid((0.05-h)/0.02) * sigmoid((0.20-v)/0.05) > 0.5
    bvh_h       BVH 高度判据: h < 5cm 判接触, 4/6cm 滞回防抖
    tactile_gmm 触觉自适应阈值: 逐 session 逐脚 log1p(压力和) 双峰 GMM,
                取谷值; 单峰(拖步/常高)则不切, 判全接触
    tactile_rel 触觉相对阈值: thr = min + 0.25 * (max - min)
    pat_offset  患者级偏移补偿: thr = 该患者所有 session 摆动相压力和的
                中位数 + 100 (摆动相由 BVH 自动挑出, 免人工); 无摆动相
                的患者退化为绝对阈值
    joint_or    bvh_h ∨ tactile_gmm (任一判离地即离地)
    joint_and   bvh_h ∧ tactile_gmm (两者都判离地才离地)

The ``--report`` step scores every method against ``bvh_h`` as reference
(视觉判据: 脚离地) and writes ``comparison.csv`` / ``comparison.png``
into the Test5 display root.  Additionally, each method folder receives a
``diff_vs_bvh_h.csv``: every contiguous frame interval where that method
disagrees with the reference (session, foot, frame range, direction, and
in-interval height/speed/sum context), sorted by span.

Usage (from the repository root):
    python results_display/script/contact_methods.py                     # 全部 session, 全部方法 + 对比报告
    python results_display/script/contact_methods.py --session S11023    # 单个 session
    python results_display/script/contact_methods.py --methods bvh_h --no-report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cli_common  # noqa: E402
from anysole.data.pressure import load_session_pressure  # noqa: E402
from anysole.types import CONTACT_SUM_THRESH  # noqa: E402
from bvh_aligner_pose import parse_bvh_aligner  # noqa: E402
from loguru import logger as log  # noqa: E402

FPS = 40.0
N_CONTACT_COLS = 10
LEFT_COL, RIGHT_COL = 6, 7
AIR_H = 0.05  # airborne threshold (m), matches losses.py
HYST_LO, HYST_HI = 0.04, 0.06  # bvh_h Schmitt band
REL_ALPHA = 0.25  # tactile_rel: thr = min + alpha * (max - min)


# --- BVH / pressure loading -------------------------------------------------


def seq_dirs(seq_root: Path) -> list[Path]:
    return sorted(p for p in Path(seq_root).glob("*/*/S*") if p.is_dir())


def load_aligned(seq_dir: Path) -> dict:
    """Meta + BVH world joints (m) + 48-cell CSV sums on the 40 Hz grid."""
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    n = int(meta["n_frames"])
    t_grid = float(meta["visual_start_s"]) + np.arange(n, dtype=np.float64) / FPS
    out = load_session_pressure(meta, t_grid)
    parsed = parse_bvh_aligner(cli_common.resolve_path(meta["bvh_path"]), trim_leading_seconds=0.0)
    joints_src, frame_time, names = parsed["joints"], parsed["frame_time"], parsed["names"]
    t_mocap = t_grid - float(meta["offset_s"])
    src_t = np.arange(joints_src.shape[0], dtype=np.float64) * float(frame_time)
    pts = np.empty((n,) + joints_src.shape[1:], dtype=np.float64)
    clipped = np.clip(t_mocap, src_t[0], src_t[-1])
    for j in range(joints_src.shape[1]):
        for a in range(3):
            pts[:, j, a] = np.interp(clipped, src_t, joints_src[:, j, a])
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0:
        pts = pts * 0.01
    floor = float(np.percentile(pts[:, :, 2], 5))
    return {
        "meta": meta,
        "n": n,
        "pts": pts,
        "names": names,
        "floor": floor,
        "sums_l": out["left48"].sum(axis=1).astype(np.float64),
        "sums_r": out["right48"].sum(axis=1).astype(np.float64),
    }


def foot_signal(ctx: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(height, speed) of a foot: height = min z over foot+toe joints minus
    the session floor; speed = mean joint displacement in m/s (40 Hz grid)."""
    names, pts, floor = ctx["names"], ctx["pts"], ctx["floor"]
    idx = [
        i for i, nm in enumerate(names)
        if nm.lower().startswith(f"{side}foot") or nm.lower().startswith(f"{side}toe")
    ]
    if not idx:
        raise ValueError(f"No {side} foot joints matched by name; names={names}")
    foot = pts[:, idx, :]
    height = foot[:, :, 2].min(axis=1) - floor
    speed = np.zeros(foot.shape[0], dtype=np.float64)
    speed[1:] = np.linalg.norm(foot[1:] - foot[:-1], axis=2).mean(axis=1) * FPS
    return height.astype(np.float64), speed


def soft_prob(height: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Same formula as anysole/losses.py:soft_contact_from_keypoints."""
    hp = 1.0 / (1.0 + np.exp(-(0.05 - height) / 0.02))
    sp = 1.0 / (1.0 + np.exp(-(0.20 - speed) / 0.05))
    return hp * sp


# --- Method definitions -----------------------------------------------------


def m_bvh_soft(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, side in enumerate(("left", "right")):
        h, v = foot_signal(ctx, side)
        out[:, k] = (soft_prob(h, v) > 0.5).astype(np.float32)
    return out


def m_bvh_h(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, side in enumerate(("left", "right")):
        h, _ = foot_signal(ctx, side)
        prev = 1.0  # sessions start grounded
        for t in range(h.shape[0]):
            if h[t] >= HYST_HI:
                prev = 0.0
            elif h[t] <= HYST_LO:
                prev = 1.0
            out[t, k] = prev
    return out


def _gmm_valley(sums: np.ndarray):
    """Two-cluster GMM valley on log1p sums; None when unimodal (all contact)."""
    x = np.log1p(np.maximum(sums, 0.0).astype(np.float64))
    if x.size < 40 or float(np.ptp(x)) < 1.0:
        return None
    c0, c1 = float(np.percentile(x, 20)), float(np.percentile(x, 80))
    if c0 == c1:
        return None
    m = np.ones(x.size, dtype=bool)
    for _ in range(50):
        new_m = np.abs(x - c1) < np.abs(x - c0)
        if np.array_equal(new_m, m):
            break
        m = new_m
        if m.sum() == 0 or (~m).sum() == 0:
            return None
        c0, c1 = float(x[~m].mean()), float(x[m].mean())
    if c0 > c1:
        c0, c1 = c1, c0
    w1 = float(m.mean())
    if w1 < 0.12 or w1 > 0.88:
        return None  # no meaningful two-phase gait
    s0, s1 = float(x[~m].std()), float(x[m].std())
    if (c1 - c0) < 2.0 * max(s0 + s1, 1e-3):
        return None
    valley = (c0 * s1 + c1 * s0) / max(s0 + s1, 1e-9)
    return float(np.expm1(valley))


def m_tactile_gmm(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, sums in enumerate((ctx["sums_l"], ctx["sums_r"])):
        thr = _gmm_valley(sums)
        out[:, k] = 1.0 if thr is None else (sums > thr).astype(np.float32)
    return out


def m_tactile_rel(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, sums in enumerate((ctx["sums_l"], ctx["sums_r"])):
        thr = float(sums.min()) + REL_ALPHA * float(sums.ptp())
        out[:, k] = (sums > thr).astype(np.float32)
    return out


def _joint(method_bvh: np.ndarray, method_tact: np.ndarray, mode: str) -> np.ndarray:
    air_bvh, air_tact = method_bvh <= 0.5, method_tact <= 0.5
    air = air_bvh | air_tact if mode == "or" else air_bvh & air_tact
    return (~air).astype(np.float32)


def m_joint_or(ctx: dict) -> np.ndarray:
    return _joint(m_bvh_h(ctx), m_tactile_gmm(ctx), "or")


def m_joint_and(ctx: dict) -> np.ndarray:
    return _joint(m_bvh_h(ctx), m_tactile_gmm(ctx), "and")


def m_tactile_abs(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, sums in enumerate((ctx["sums_l"], ctx["sums_r"])):
        out[:, k] = (sums > CONTACT_SUM_THRESH).astype(np.float32)
    return out


def m_pat_offset(ctx: dict, patient_offsets: dict | None) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    patient = ctx["meta"]["subject"]
    for k, (side, sums) in enumerate((("left", ctx["sums_l"]), ("right", ctx["sums_r"]))):
        offset = None
        if patient_offsets is not None:
            offset = patient_offsets.get(patient, {}).get(side)
        thr = CONTACT_SUM_THRESH if offset is None else float(offset) + CONTACT_SUM_THRESH
        out[:, k] = (sums > thr).astype(np.float32)
    return out


METHODS: dict[str, dict] = {
    "tactile_abs": {"desc": "触觉绝对阈值: 48格压力和 > 100 (原 contact.npy 口径)", "fn": lambda ctx, po: m_tactile_abs(ctx)},
    "bvh_soft": {"desc": "BVH 运动学 (模型同构 sigmoid 乘积 > 0.5)", "fn": lambda ctx, po: m_bvh_soft(ctx)},
    "bvh_h": {"desc": "BVH 高度 h<5cm + 4/6cm 滞回", "fn": lambda ctx, po: m_bvh_h(ctx)},
    "tactile_gmm": {"desc": "触觉 log1p 双峰 GMM 谷值阈值 (单峰=全接触)", "fn": lambda ctx, po: m_tactile_gmm(ctx)},
    "tactile_rel": {"desc": "触觉相对阈值 min + 0.25*(max-min)", "fn": lambda ctx, po: m_tactile_rel(ctx)},
    "pat_offset": {"desc": "患者级偏移补偿: 摆动相压力和的中位数 + 100", "fn": m_pat_offset},
    "joint_or": {"desc": "bvh_h ∨ tactile_gmm (任一判离地)", "fn": lambda ctx, po: m_joint_or(ctx)},
    "joint_and": {"desc": "bvh_h ∧ tactile_gmm (都判离地才离地)", "fn": lambda ctx, po: m_joint_and(ctx)},
}
METHOD_NAMES = list(METHODS)


def patient_offsets(seq_dirs_list: list[Path]) -> dict:
    """Per patient per foot: median pressure sum during that patient's
    BVH-airborne frames across all their sessions (pat_offset template)."""
    by_patient: dict[str, list[Path]] = {}
    for seq_dir in seq_dirs_list:
        meta = json.loads((seq_dir / "align_meta.json").read_text())
        by_patient.setdefault(meta["subject"], []).append(seq_dir)
    offsets: dict[str, dict[str, float]] = {}
    for patient, dirs in by_patient.items():
        air_sums: dict[str, list[float]] = {"left": [], "right": []}
        for seq_dir in dirs:
            try:
                ctx = load_aligned(seq_dir)
            except (FileNotFoundError, ValueError) as exc:
                log.warning(f"{seq_dir.name}: load_aligned failed ({exc}); skipping for offset template")
                continue
            for side, sums in (("left", ctx["sums_l"]), ("right", ctx["sums_r"])):
                h, _ = foot_signal(ctx, side)
                air_sums[side].extend(float(v) for v in sums[h >= AIR_H])
        patient_off = {}
        for side, vals in air_sums.items():
            patient_off[side] = float(np.median(vals)) if vals else None
        offsets[patient] = patient_off
        log.info(f"patient {patient}: offset template L={patient_off['left']} R={patient_off['right']}")
    return offsets


def apply_method(ctx: dict, method: str, po: dict | None) -> tuple[np.ndarray, tuple[float | None, float | None]]:
    """(n, 10) contact matrix (cols 6/7 filled) plus the per-foot sum
    thresholds the method used (None for kinematic/BVH methods)."""
    labels = METHODS[method]["fn"](ctx, po)
    contact = np.zeros((ctx["n"], N_CONTACT_COLS), dtype=np.float32)
    contact[:, LEFT_COL] = labels[:, 0]
    contact[:, RIGHT_COL] = labels[:, 1]
    thr_l = thr_r = None
    if method == "tactile_abs":
        thr_l = thr_r = CONTACT_SUM_THRESH
    elif method == "tactile_gmm":
        thr_l, thr_r = _gmm_valley(ctx["sums_l"]), _gmm_valley(ctx["sums_r"])
    elif method == "tactile_rel":
        thr_l = float(ctx["sums_l"].min()) + REL_ALPHA * float(ctx["sums_l"].ptp())
        thr_r = float(ctx["sums_r"].min()) + REL_ALPHA * float(ctx["sums_r"].ptp())
    elif method == "pat_offset":
        patient = ctx["meta"]["subject"]
        if po is not None and patient in po:
            off = po[patient]
            thr_l = (off["left"] + CONTACT_SUM_THRESH) if off["left"] is not None else CONTACT_SUM_THRESH
            thr_r = (off["right"] + CONTACT_SUM_THRESH) if off["right"] is not None else CONTACT_SUM_THRESH
        else:
            thr_l = thr_r = CONTACT_SUM_THRESH
    return contact, (thr_l, thr_r)


def compute_contact(seq_dir: Path, method: str, po: dict | None) -> tuple[np.ndarray, tuple[float | None, float | None]]:
    """(n, 10) contact matrix with cols 6/7 filled by the method."""
    return apply_method(load_aligned(seq_dir), method, po)


def method_thresholds(seq_dir: Path, method: str) -> tuple[float | None, float | None] | None:
    """Per-foot sum thresholds stored alongside the labels (None if absent)."""
    sidecar = Path(seq_dir) / f"contact_{method}.json"
    if not sidecar.is_file():
        return None
    data = json.loads(sidecar.read_text())
    return data.get("thr_l"), data.get("thr_r")


# --- Label generation --------------------------------------------------------


def ensure_labels(dirs: list[Path], methods: list[str], force: bool = False) -> None:
    """Generate missing ``contact_<method>.npy`` for the given session dirs.

    ``load_aligned`` (BVH resample + CSV load) runs once per session and the
    context is shared across methods."""
    methods = [m for m in methods if m in METHODS]
    if not dirs or not methods:
        return
    po = patient_offsets(dirs) if "pat_offset" in methods else None
    n_written = n_skipped = 0
    for seq_dir in dirs:
        ctx = None
        for method in methods:
            out_path = seq_dir / f"contact_{method}.npy"
            if out_path.is_file() and not force:
                n_skipped += 1
                continue
            try:
                if ctx is None:
                    ctx = load_aligned(seq_dir)
                contact, (thr_l, thr_r) = apply_method(ctx, method, po)
            except (FileNotFoundError, ValueError) as exc:
                log.warning(f"{seq_dir.name} [{method}]: {exc}")
                continue
            np.save(out_path, contact)
            (seq_dir / f"contact_{method}.json").write_text(
                json.dumps({"method": method, "thr_l": thr_l, "thr_r": thr_r})
            )
            n_written += 1
    if n_written:
        log.info(f"labels: wrote {n_written}, skipped {n_skipped} ({len(dirs)} sessions x {len(methods)} methods)")


def generate(args: argparse.Namespace) -> None:
    root = Path(cli_common.resolve_path(args.seq_root))
    dirs = seq_dirs(root)
    if args.session:
        wanted = set(cli_common.split_csv_arg(args.session))
        dirs = [d for d in dirs if d.name in wanted]
    if not dirs:
        raise SystemExit(f"No sequence dirs under {root}")
    methods = [m for m in METHOD_NAMES if args.methods == "all" or m in args.methods]
    ensure_labels(dirs, methods, force=args.force)


# --- Comparison report vs bvh_h reference -----------------------------------


REFERENCE = "bvh_h"


def load_labels(seq_dir: Path, method: str) -> np.ndarray | None:
    path = seq_dir / f"contact_{method}.npy"
    if not path.is_file():
        return None
    raw = np.load(path).astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] < 8:
        return None
    return raw[:, [LEFT_COL, RIGHT_COL]] > 0.5


def report(args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk).is_file():
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    root = Path(cli_common.resolve_path(args.seq_root))
    dirs = seq_dirs(root)
    if args.session:
        wanted = set(cli_common.split_csv_arg(args.session))
        dirs = [d for d in dirs if d.name in wanted]
    methods = METHOD_NAMES if args.methods == "all" else [m for m in METHOD_NAMES if m in args.methods]

    rows = []
    for seq_dir in dirs:
        ref = load_labels(seq_dir, REFERENCE)
        if ref is None:
            log.warning(f"{seq_dir.name}: no reference ({REFERENCE}) labels; skipped")
            continue
        for method in methods:
            lab = load_labels(seq_dir, method)
            if lab is None:
                continue
            row = {"method": method, "session": seq_dir.name, "n": int(ref.shape[0])}
            for k, side in enumerate(("l", "r")):
                same = (lab[:, k] == ref[:, k]).mean()
                air_ref = ~ref[:, k]
                grd_ref = ref[:, k]
                fc = float((lab[:, k] & air_ref).sum() / max(air_ref.sum(), 1))
                fa = float((~lab[:, k] & grd_ref).sum() / max(grd_ref.sum(), 1))
                row[f"acc_{side}"] = round(float(same), 4)
                row[f"fc_{side}"] = round(fc, 4)
                row[f"fa_{side}"] = round(fa, 4)
                row[f"rate_{side}"] = round(float(lab[:, k].mean()), 4)
            rows.append(row)

    csv_path = out_dir / "comparison.csv"
    fields = ["method", "session", "n", "acc_l", "acc_r", "fc_l", "fc_r", "fa_l", "fa_r", "rate_l", "rate_r"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {csv_path} ({len(rows)} rows)")

    # Aggregate per method: mean accuracy and pooled false rates vs reference.
    agg = {}
    for row in rows:
        entry = agg.setdefault(row["method"], {"n": 0, "acc": 0.0, "fc_num": 0, "fc_den": 0, "fa_num": 0, "fa_den": 0})
        entry["n"] += 1
        entry["acc"] += (row["acc_l"] + row["acc_r"]) / 2.0
    # Pooled false rates need frame counts; recompute from per-session margins.
    pooled = {m: {"fc_num": 0, "fc_den": 0, "fa_num": 0, "fa_den": 0} for m in agg}
    for seq_dir in dirs:
        ref = load_labels(seq_dir, REFERENCE)
        if ref is None:
            continue
        for method in methods:
            lab = load_labels(seq_dir, method)
            if lab is None:
                continue
            air_ref, grd_ref = ~ref, ref
            pooled[method]["fc_num"] += int((lab & air_ref).sum())
            pooled[method]["fc_den"] += int(air_ref.sum())
            pooled[method]["fa_num"] += int((~lab & grd_ref).sum())
            pooled[method]["fa_den"] += int(grd_ref.sum())
    names = [m for m in methods if m in agg]
    accs = [agg[m]["acc"] / max(agg[m]["n"], 1) for m in names]
    fcs = [pooled[m]["fc_num"] / max(pooled[m]["fc_den"], 1) for m in names]
    fas = [pooled[m]["fa_num"] / max(pooled[m]["fa_den"], 1) for m in names]
    # One fixed entity order across all panels (sorted by agreement once).
    order = sorted(range(len(names)), key=lambda i: -accs[i])
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    fcs = [fcs[i] for i in order]
    fas = [fas[i] for i in order]

    bar_color = "#4C72B0"  # single hue: identity comes from the axis labels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    series = (
        ("与 BVH 高度参考的一致率 (越高越好)", accs),
        ("假接触率: 参考离地但判接触 (越低越好)", fcs),
        ("假离地率: 参考接地但判离地 (越低越好)", fas),
    )
    for ax, (title, values) in zip(axes, series):
        ys = np.arange(len(names))
        ax.barh(ys, values, color=bar_color, height=0.62)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max(1.0, max(values) * 1.15))
        ax.grid(True, axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=9)
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.suptitle(f"Test5 contact 方案对比 (参考={REFERENCE}, {len(dirs)} sessions)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = out_dir / "comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {png_path}")


# --- Per-interval difference list vs the bvh_h reference --------------------


MERGE_GAP = 3  # merge same-direction disagreement runs separated by <= this many frames


def diff_intervals(ref: np.ndarray, lab: np.ndarray) -> list[dict]:
    """Contiguous frame ranges where ``lab`` disagrees with ``ref``.

    Runs of the same direction separated by <= ``MERGE_GAP`` frames are
    merged into one interval (gaps are absorbed into the span).  ``direction``
    is "fc" (false contact: reference airborne, method says contact) or "fa"
    (false airborne: reference grounded, method says airborne)."""
    out = []
    d = ref != lab
    if not d.any():
        return out
    pad = np.concatenate(([False], d, [False]))
    starts = np.where(pad[1:] & ~pad[:-1])[0]
    ends = np.where(~pad[1:] & pad[:-1])[0]
    cur = None
    for a, b in zip(starts, ends):
        direction = "fc" if not ref[a] else "fa"
        if cur is not None and cur["direction"] == direction and a - cur["end"] <= MERGE_GAP:
            cur["end"] = int(b)
        else:
            cur = {"start": int(a), "end": int(b), "direction": direction}
            out.append(cur)
    return out


C_FC = "#2a78d6"  # categorical slot 1: false contact
C_FA = "#eb6834"  # categorical slot 2: false airborne


def diff_report(args: argparse.Namespace, out_dir: Path) -> None:
    """Per-method difference lists and distribution charts.

    Each method folder gets:
    - ``diff_vs_bvh_h.csv``: one row per disagreement interval vs the
      reference (session, foot, time range in session-relative seconds -- the
      same ``t`` the animation footer shows -- direction, and in-interval
      height/speed/sum context), sorted by span.
    - ``diff_by_subject.png`` / ``diff_by_sequence.png``: stacked fc/fa
      difference-rate bars per subject (S5..S14) and per action sequence,
      so the differences vs bvh_h are located per group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk).is_file():
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    root = Path(cli_common.resolve_path(args.seq_root))
    dirs = seq_dirs(root)
    if args.session:
        wanted = set(cli_common.split_csv_arg(args.session))
        dirs = [d for d in dirs if d.name in wanted]
    methods = METHOD_NAMES if args.methods == "all" else [m for m in METHOD_NAMES if m in args.methods]

    rows_by_method = {m: [] for m in methods}
    # session -> (subject, sequence); per method per session: fc/fa/frame counts
    sid_subject, sid_seq = {}, {}
    counts = {m: {} for m in methods}
    for seq_dir in dirs:
        try:
            ref = load_labels(seq_dir, REFERENCE)
            if ref is None:
                log.warning(f"{seq_dir.name}: no reference ({REFERENCE}) labels; skipped")
                continue
            ctx = load_aligned(seq_dir)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"{seq_dir.name}: {exc}")
            continue
        subject = str(ctx["meta"]["subject"])  # "S7" / "S14"
        body = seq_dir.name[len("S") + len(subject) - 1:]  # sequence(2) + trial(1)
        sid_subject[seq_dir.name] = subject
        sid_seq[seq_dir.name] = body[:2]
        for method in methods:
            lab = load_labels(seq_dir, method)
            if lab is None:
                continue
            fc_total = fa_total = 0
            for k, side in enumerate(("L", "R")):
                h, v = foot_signal(ctx, {"L": "left", "R": "right"}[side])
                sums = ctx["sums_l"] if side == "L" else ctx["sums_r"]
                fc_total += int(((~ref[:, k]) & lab[:, k]).sum())
                fa_total += int((ref[:, k] & (~lab[:, k])).sum())
                for it in diff_intervals(ref[:, k], lab[:, k]):
                    seg = slice(it["start"], it["end"])
                    rows_by_method[method].append({
                        "session": seq_dir.name,
                        "foot": side,
                        "start_s": round(it["start"] / FPS, 2),
                        "end_s": round(it["end"] / FPS, 2),
                        "span_s": round((it["end"] - it["start"]) / FPS, 2),
                        "diff_frames": int((ref[seg, k] != lab[seg, k]).sum()),
                        "direction": it["direction"],
                        "h_median_m": round(float(np.median(h[seg])), 3),
                        "v_median_ms": round(float(np.median(v[seg])), 3),
                        "sum_median": round(float(np.median(sums[seg])), 1),
                    })
            counts[method][seq_dir.name] = (fc_total, fa_total, int(ref.shape[0]))

    subjects = sorted(set(sid_subject.values()), key=lambda s: int(s[1:]))
    seqs = sorted(set(sid_seq.values()))

    fields = ["session", "foot", "start_s", "end_s", "span_s", "diff_frames",
              "direction", "h_median_m", "v_median_ms", "sum_median"]
    log.info("diff summary (vs bvh_h), per-method lists under Test5_contact/<method>/:")
    for method in methods:
        rows = rows_by_method[method]
        rows.sort(key=lambda r: (-r["span_s"], r["session"], r["foot"], r["start_s"]))
        if method == REFERENCE:
            log.info(f"  {method:12s} 参考本身，无差异清单")
            continue
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        csv_path = method_dir / "diff_vs_bvh_h.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        fc = sum(r["diff_frames"] for r in rows if r["direction"] == "fc")
        fa = sum(r["diff_frames"] for r in rows if r["direction"] == "fa")
        sessions = {r["session"] for r in rows}
        log.info(
            f"  {method:12s} -> {csv_path.relative_to(REPO_ROOT)}  sessions={len(sessions):3d} "
            f"intervals={len(rows):4d} fc_frames={fc:6d} fa_frames={fa:6d}"
        )
        _plot_diff_groups(method, method_dir, counts[method], sid_subject, sid_seq, subjects, seqs)


def _plot_diff_groups(method: str, method_dir: Path, counts: dict, sid_subject: dict,
                      sid_seq: dict, subjects: list, seqs: list) -> None:
    """Two stacked-bar charts in the method folder: fc/fa difference rates
    per subject and per action sequence (normalized by group frame counts)."""
    import matplotlib.pyplot as plt

    for label, group_keys, key_fn, fname in (
        ("用户", subjects, lambda s: sid_subject[s], "diff_by_subject.png"),
        ("动作序列", seqs, lambda s: sid_seq[s], "diff_by_sequence.png"),
    ):
        grouped = {key: {"fc": 0, "fa": 0, "n": 0} for key in group_keys}
        for session, (fc_c, fa_c, n) in counts.items():
            key = key_fn(session)
            if key not in grouped:
                continue
            grouped[key]["fc"] += fc_c
            grouped[key]["fa"] += fa_c
            grouped[key]["n"] += n
        fc_rates = [grouped[k]["fc"] / max(grouped[k]["n"], 1) for k in group_keys]
        fa_rates = [grouped[k]["fa"] / max(grouped[k]["n"], 1) for k in group_keys]
        total = np.asarray(fc_rates) + np.asarray(fa_rates)
        overall = float(total.sum() / len(group_keys))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ys = np.arange(len(group_keys))
        ax.bar(ys, fc_rates, color=C_FC, width=0.68, label="fc 假接触（参考离地但判接触）")
        ax.bar(ys, fa_rates, bottom=fc_rates, color=C_FA, width=0.68, label="fa 假离地（参考接地但判离地）")
        ax.axhline(overall, color="#8C8C94", linestyle="--", linewidth=1.0)
        ax.text(len(group_keys) - 0.35, min(overall + 0.012, 0.95), f"mean={overall:.3f}",
                ha="right", va="bottom", fontsize=9, color="#5A5A60")
        for i, t in enumerate(total):
            ax.text(i, t + 0.008, f"{t:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(ys)
        ax.set_xticklabels(group_keys, fontsize=10)
        ax.set_ylim(0, min(1.0, max(total.max() * 1.18, 0.1)))
        ax.set_ylabel("差异帧占比")
        ax.set_title(f"{method} 与 {REFERENCE} 的差异 × {label}分布", fontsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = method_dir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-method contact labels and compare them.")
    cli_common.add_common_args(
        parser,
        seq_root=True,
        session=True,
        split_csv=False,
        split=False,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        force=True,
        out_dir=False,
    )
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--no-report", action="store_true", help="Skip the comparison report.")
    parser.add_argument("--no-diff", action="store_true", help="Skip the per-interval difference list (method_diffs.csv).")
    parser.add_argument("--report-only", action="store_true", help="Only write the comparison report (labels must exist).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.report_only:
        generate(args)
    if not args.no_report:
        out_dir = cli_common.DISPLAY_ROOT / "Test5_contact"
        out_dir.mkdir(parents=True, exist_ok=True)
        report(args, out_dir)
        if not args.no_diff:
            diff_report(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
