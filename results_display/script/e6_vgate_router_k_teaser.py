#!/usr/bin/env python3
"""E6 teaser：单会话内 router 的 T 分支权重 k 随时间与遮挡变化（2026-11-14）。

构图
----
- 4 行 = clean / occ020 / occ050 / occ100；6 列 = 同一会话时间 6 等分的
  GT SMPL 动作快照。
- 每个身体按当前时刻的逐部位 k 着色（蓝=V，红=T，灰=0.5），经 SMPL
  skinning weights 平滑；同一列四行使用同一 GT 姿态，只替换视觉条件。
- 底部是整段会话的 f6_soft 双足接触带，竖虚线对齐 6 个快照时刻；
  横轴为时间。
- 只读 checkpoint / 缓存，不训练。

口径
----
- 模型：``V8_Cv7_PhysIMU_Router``，默认 checkpoint
  ``results/diagnostics/20261112_v8_gate_frozen/router_ft_enum_fS5S6S11.pt``。
- 默认会话 S11051（run，fS5S6S11 test）；四档使用完全相同的帧，只替换视觉缓存。
- k = σ(s_v − s_t)，逐帧逐组；四组 = 根速度 / 根姿态＋躯干头臂 / 左腿左脚 /
  右腿右脚。快照颜色取该帧前后 ±(smooth//2) 帧的组均值。

产物（默认 E6_ContactPhaseAnalysis/teaser/）
  Main_Fig_E6_RouterK_Teaser_<Action>_<sid>.png
  detail/E6_RouterK_Teaser_<sid>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
for _entry in (REPO, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from anysole.data.dataset import AnySoleDataset  # noqa: E402
from anysole.data.smpl_io import load_smpl  # noqa: E402
from anysole.types import CONFIG_VT  # noqa: E402
from anysole.phase_common import action_of, load_contact  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

DIAG = REPO / "results" / "diagnostics" / "20261112_v8_gate_frozen"
sys.path.insert(0, str(DIAG))
from train_router_gate import build_router  # noqa: E402

import e5_v8_smpl_gain_heatmap as heat  # noqa: E402
from e6_vgate_router_k_map import _joint_values  # noqa: E402

DEFAULT_OUT = (REPO / "results_display" / "Etest_crossfit_physimu_vgate"
               / "E6_ContactPhaseAnalysis" / "detail" / "router_k")
DEFAULT_SPLIT = (REPO / "AnysoleWorkspace" / "protocol" / "splits" / "default"
                 / "splits_test_S5S6S11_val_S12.csv")
DEFAULT_VISUAL = (REPO / "AnysoleWorkspace" / "model_inputs" / "AnySole"
                  / "adapter_v1" / "visual" / "hrnet")

GROUPS = ("root_velocity", "upper_body", "left_leg_foot", "right_leg_foot")
GROUP_SLOTS = (0, 1, 6, 7)
CONDS = (
    ("clean", "cam3"),
    ("occ020", "cam3_occ_lower020"),
    ("occ050", "cam3_occ_lower050"),
    ("occ100", "cam3_occ_lower100"),
)
ACTION_GROUPS = {
    "01": "stand-shift", "02": "single-stand", "03": "walk", "04": "walk",
    "05": "run", "06": "squat", "07": "jump-jack", "08": "run",
    "09": "lateral", "10": "sit-stand", "11": "stairs",
}
CMAP_STOPS = [
    (0.00, "#0d366b"),
    (0.25, "#2a78d6"),
    (0.50, "#dedddb"),
    (0.75, "#e34948"),
    (1.00, "#7f1d1d"),
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
N_SNAPSHOTS = 6


def _colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("router_k", CMAP_STOPS)


def _load_model(checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = build_router(cfg)
    model.load_state_dict(ck["model"], strict=True)
    return model.eval(), cfg


def _session_dir(sid: str) -> Path:
    hits = sorted((REPO / "AnysoleWorkspace" / "shared" / "facts" / "sessions"
                   / "cam3").glob("*/*/%s/session.json" % sid))
    if not hits:
        raise FileNotFoundError("session.json not found for %s" % sid)
    return hits[0].parent


def _collect(model, cfg, sid: str, split_csv: Path, visual_root: Path) -> dict:
    """→ k per condition, time, contact, valid, GT SMPL motion."""
    session_dir = _session_dir(sid)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = np.load(session_dir / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    t = t - t[0]
    valid = (np.asarray(frames["valid"]).astype(bool)
             & ~np.asarray(frames["fake"]).astype(bool))
    contact, _, contact_valid = load_contact(sid)
    smpl_path = resolve_uri(meta["source_files"]["smpl"], must_exist=True)
    motion = load_smpl(smpl_path, query_t=np.asarray(frames["mocap_time_s"]))
    n = min(len(t), len(valid), len(contact), len(motion["poses_aa"]))

    out = {}
    for cond, cache in CONDS:
        ds = AnySoleDataset(
            mode="eval", window_length=int(cfg["tw"]), session_ids=[sid],
            cache_root=visual_root / cache,
            contact_method=str(cfg["contact_method"]),
            tactile_input=str(cfg["tactile_input"]), split_csv=split_csv)
        session = ds.sessions[0]
        if cond != "clean":
            path = visual_root / cache / (sid + ".pt")
            if not path.is_file():
                raise FileNotFoundError(path)
            arr = torch.load(path, map_location="cpu", weights_only=False)
            arr = (arr.float().numpy() if torch.is_tensor(arr)
                   else np.asarray(arr, dtype=np.float32))
            if arr.shape != session["V_feat"].shape:
                raise ValueError("%s: %s vs %s" % (path, arr.shape,
                                                   session["V_feat"].shape))
            session["V_feat"] = arr
        full = np.full((len(session["V_feat"]), 4), np.nan, dtype=np.float32)
        for batch in DataLoader(ds, batch_size=16, shuffle=False):
            with torch.inference_mode():
                nb = batch["V_feat"].shape[0]
                cid = torch.full((nb,), CONFIG_VT, dtype=torch.long)
                v_tok, t_tok = model.encoders(
                    batch["V_feat"], batch["T_raw"], cid,
                    T_s2m=batch["T_s2m"], T_phys=batch["T_phys"])
                present = torch.ones(nb, dtype=v_tok.dtype)
                k = model._route_router_gate(
                    v_tok, t_tok, batch["T_phys"], batch["T_s2m"],
                    present, present)[:, :, list(GROUP_SLOTS), 0].numpy()
            for i, left in enumerate(batch["frame_start"].tolist()):
                full[left:left + int(cfg["tw"])] = k[i]
        out[cond] = full[:n]
        print("[k] %-6s %s frames=%d finite=%d"
              % (cond, sid, len(full),
                 int(np.isfinite(out[cond][:, 0]).sum())), flush=True)
    return {"k": out, "t": t[:n], "contact": contact[:n], "valid": valid[:n],
            "motion": motion, "meta": meta}


def _snapshot_indices(k: dict, valid: np.ndarray, count: int) -> np.ndarray:
    finite = valid.copy()
    for cond, _ in CONDS:
        finite &= np.all(np.isfinite(k[cond]), axis=1)
    candidates = np.flatnonzero(finite)
    if len(candidates) < count:
        raise ValueError("only %d valid frames for %d snapshots"
                         % (len(candidates), count))
    picks = candidates[np.round(np.linspace(
        0, len(candidates) - 1, count)).astype(int)]
    return np.unique(picks)


def _body_azim(poses: np.ndarray, frame: int, offset: float = 20.0) -> float:
    """相机放在动作前方 + offset，避免同一行里转到背面。"""
    rot = SciRotation.from_rotvec(poses[frame, 0]).as_matrix()
    forward = rot @ np.array([0.0, 0.0, 1.0])
    display = np.array([forward[0], -forward[2], forward[1]])
    return float(np.degrees(np.arctan2(display[1], display[0])) + offset)


def _render_body(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray,
                 rest: dict, faces: np.ndarray, weights: np.ndarray,
                 k_frame: np.ndarray, frame: int, size: int = 380) -> np.ndarray:
    from matplotlib.colors import Normalize

    per_joint = _joint_values(k_frame)
    vertex_values = heat._vertex_values(per_joint, weights)
    colors = _colormap()(Normalize(0.0, 1.0, clip=True)(vertex_values))[:, :3]
    verts = heat.smpl_vertices(poses[frame:frame + 1],
                               trans[frame:frame + 1], betas, rest)[0]
    verts_display = heat.smpl_yup_to_display(verts)
    image = heat.render_mesh_frame(
        verts_display, faces, vertex_colors=colors, ambient=0.85,
        light_intensity=0.8, bg_color=SURFACE, size=(size, size),
        elev=8.0, azim=_body_azim(poses, frame))
    return heat._crop_body(image)


def _plot(data: dict, sid: str, action: str, fold: str, checkpoint: Path,
          path: Path, smooth: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    k = data["k"]
    t = data["t"]
    contact = data["contact"]
    motion = data["motion"]
    poses = np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3)
    trans = np.asarray(motion["trans_m"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)[:10]
    rest = heat._rest_joints({"betas": betas})
    _, faces, weights = heat._canonical_mesh()
    picks = _snapshot_indices(k, data["valid"], N_SNAPSHOTS)
    half = max(0, smooth // 2)

    fig = plt.figure(figsize=(17.2, 11.0), facecolor=SURFACE)
    gs = GridSpec(len(CONDS) + 1, len(picks), figure=fig,
                  height_ratios=[1.0] * len(CONDS) + [0.42],
                  hspace=0.03, wspace=0.015,
                  left=0.045, right=0.925, top=0.895, bottom=0.09)
    for row, (cond, _) in enumerate(CONDS):
        for col, frame in enumerate(picks):
            frame = int(frame)
            lo, hi = max(0, frame - half), min(len(t), frame + half + 1)
            k_frame = np.nanmean(k[cond][lo:hi], axis=0)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(_render_body(poses, trans, betas, rest, faces, weights,
                                   k_frame, frame))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title("t = %.1f s" % t[frame], fontsize=9.5,
                             color=INK, pad=3)
            if col == 0:
                ax.set_ylabel(cond, fontsize=10.5, color=INK, rotation=0,
                              labelpad=14, va="center")

    # 底部接触带：整段会话，竖线对齐快照。
    # 快照列是等宽列的“中心”，因此把接触带两端各补半个时间步的空白，
    # 使 t_i 正好落在第 i 列中心，而不是把接触数据拉伸到列中心。
    step = (t[-1] - t[0]) / max(len(picks) - 1, 1)
    pad = int(round(contact.shape[0] * (step / 2.0) / (t[-1] - t[0])))
    band = np.pad(contact.T.astype(np.float32), ((0, 0), (pad, pad)),
                  constant_values=0.0)
    axc = fig.add_subplot(gs[-1, :])
    axc.imshow(band, aspect="auto",
               cmap=plt.matplotlib.colors.ListedColormap(
                   ["#fcfcfb", "#b9c6cf"]), vmin=0.0, vmax=1.0,
               interpolation="nearest",
               extent=[t[0] - step / 2.0, t[-1] + step / 2.0, 0, 2])
    axc.set_xlim(t[0] - step / 2.0, t[-1] + step / 2.0)
    axc.set_yticks([0.5, 1.5])
    axc.set_yticklabels(["L", "R"], fontsize=8.0, color=MUTED)
    axc.set_xticks([t[f] for f in picks])
    axc.set_xticklabels(["%.1f" % t[f] for f in picks], fontsize=8.5,
                        color=MUTED)
    for frame in picks:
        axc.axvline(t[frame], color="#8a8980", lw=0.8, ls=":")
    axc.tick_params(length=2, pad=2)
    for spine in axc.spines.values():
        spine.set_visible(False)
    axc.set_ylabel("contact", fontsize=8.5, color=MUTED, labelpad=6)
    axc.set_xlabel("time (s)", fontsize=9.5, color=MUTED, labelpad=4)

    cax = fig.add_axes([0.935, 0.27, 0.012, 0.50])
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=_colormap())
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("T-branch weight k", fontsize=9.0, color=MUTED)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)

    fig.suptitle("Router T-branch weight at six moments of one %s trial (%s) "
                 "under increasing visual occlusion" % (action, sid),
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.975)
    fig.text(0.48, 0.925,
             "Session %s (%s fold); GT SMPL pose at each moment; "
             "k = σ(s_V − s_T), k=0 → V only, k=1 → T only; "
             "colour = per-part k averaged over ±%d ms; "
             "bottom band = f6_soft foot contact."
             % (sid, fold, int(round(half * 1000.0 / 40.0))),
             fontsize=8.4, color=MUTED, ha="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("[png] %s" % path)


def _write_csv(path: Path, k: dict, t: np.ndarray, contact: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "condition", *GROUPS, "contact_l", "contact_r"])
        for cond, _ in CONDS:
            for i in range(len(t)):
                writer.writerow(
                    ["%.4f" % t[i], cond,
                     *["%.6f" % x if np.isfinite(x) else "" for x in k[cond][i]],
                     int(contact[i, 0] > 0.5), int(contact[i, 1] > 0.5)])
    print("[csv] %s" % path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", default="S11051")
    ap.add_argument("--checkpoint", type=Path,
                    default=DIAG / "router_ft_enum_fS5S6S11.pt")
    ap.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "teaser")
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--smooth", type=int, default=9,
                    help="snapshot k averaging window in frames (1 = raw)")
    args = ap.parse_args()

    model, cfg = _load_model(args.checkpoint)
    data = _collect(model, cfg, args.session_id, args.split_csv, args.visual_root)
    print("[meta] %s frames=%d fps=%s quality=%s"
          % (args.session_id, len(data["t"]), data["meta"].get("fps"),
             data["meta"].get("quality")))
    action = ACTION_GROUPS.get(action_of(args.session_id), "motion")

    _write_csv(args.out_dir / "detail" / ("E6_RouterK_Teaser_%s.csv"
                                          % args.session_id),
               data["k"], data["t"], data["contact"])
    _plot(data, args.session_id, action, args.fold, args.checkpoint,
          args.out_dir / ("Main_Fig_E6_RouterK_Teaser_%s_%s.png"
                          % (action.capitalize(), args.session_id)), args.smooth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
