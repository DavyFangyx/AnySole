#!/usr/bin/env python3
"""SMPL 足底 vs 地面（Y=0 口径）+ smpl_contact vs 接触标签逐帧对比（2026-11-07）。

问题链：
  1. 摆动帧压力非零（fmax 0.75-0.79，约 1/3 格非零）——是鞋垫足-鞋压力伪影，
     还是摆动标签本身不准（脚其实还在地面）？
  2. SMPL GT 是否真的穿透地面（GroundPen）？若有穿透则 SMPL 接触不可信。
     若无（只有轻微悬浮），SMPL 足底高度是更干净的接触量。

地面参考：
  y0      = 世界 Y=0（10-19 审计裁定判读口径）
  est     = estimate_floor_y（关节中心 5 分位，天然高于地面，仅作偏差报告）

相位（phase_common 同口径，margin=0.1s）：
  stance=0 swing=1 strike=2 off=3；稳定摆动 = phase==1（剔除落脚/离地边缘）

标签集：
  offfix    = f6_soft_offfix（文档15/修改线当前默认）
  f6_soft   = 训练默认
  smpl24st  = f6_soft_time_accel_state_smpl24_v1（TTA 评估链）

输出（.tmp_scripts/smpl_ground_pen_out/）：
  penetration.csv  逐会话逐脚 h_y0 分布 + 穿透率（h<0）+ est 偏差
  label_cmp.csv     逐会话逐脚×标签集：一致率 + 稳定摆动帧内 SMPL 接触率/压力
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "AnysoleWorkspace" / "tool"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anysole.data.smpl_io import (  # noqa: E402
    estimate_floor_y,
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.workspace_adapter import (  # noqa: E402
    load_shared_session,
    shared_smpl_path,
)
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402
from contact_labels import _smpl_foot_heights, SMPL_CONTACT_DZ  # noqa: E402
from anysole_tta.phase_common import phase_labels  # noqa: E402

LABEL_ROOT = REPO / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/labels"
LABEL_SETS = {
    "offfix": LABEL_ROOT / "f6_soft_offfix",
    "f6_soft": LABEL_ROOT / "f6_soft",
    "smpl24st": LABEL_ROOT / "f6_soft_time_accel_state_smpl24_v1/labels",
}
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
OUT = REPO / ".tmp_scripts" / "smpl_ground_pen_out"
OUT.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = {"01": "stand_shift", "02": "single_stand", "03": "walk",
                "04": "walk", "05": "run", "06": "squat", "07": "jump_jack",
                "08": "run", "09": "lateral", "10": "sit_stand", "11": "stairs"}


def fold_test_sessions() -> list[str]:
    out: set[str] = set()
    for fold in FOLDS:
        path = REPO / "AnysoleWorkspace/protocol/splits/default" / f"splits_sceneocc_{fold}.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        out |= {r["test"].strip() for r in rows if r["test"].strip()}
    return sorted(out)


def manifest() -> dict[str, tuple[str, str]]:
    rows = list(csv.DictReader(
        (REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.csv")
        .open(encoding="utf-8-sig")))
    return {r["session_id"]: (r["subject_id"], r["action"]) for r in rows}


def load_labels(root: Path, sid: str, n: int, facts_fid: np.ndarray) -> tuple | None:
    d = np.load(root / f"{sid}.npz", allow_pickle=False)
    fid = np.asarray(d["frame_id"])
    pos = {int(f): i for i, f in enumerate(facts_fid)}
    lab_idx = [pos[int(f)] for f in fid if int(f) in pos]
    src_idx = [i for i, f in enumerate(fid) if int(f) in pos]
    if not lab_idx:
        return None
    contact = np.asarray(d["contact"], dtype=np.float32)
    lab = np.full((n, 2), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    v = np.asarray(d["valid"], dtype=bool)
    lab[np.asarray(lab_idx)] = contact[np.asarray(src_idx)]
    valid[np.asarray(lab_idx)] = v[np.asarray(src_idx)]
    if "source_time_supported" in d:
        sts = np.asarray(d["source_time_supported"], dtype=bool)
        valid[np.asarray(lab_idx)] &= sts[np.asarray(src_idx)]
    return lab, valid


def run_session(sid: str) -> dict | None:
    shared = load_shared_session(sid)
    frames = shared["frames"]
    n = int(len(frames["frame_id"]))
    facts_fid = np.asarray(frames["frame_id"], dtype=np.int64)[:n]
    mocap_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)[:n]
    left48 = np.asarray(shared["pressure"]["left48"], dtype=np.float32)[:n]
    right48 = np.asarray(shared["pressure"]["right48"], dtype=np.float32)[:n]

    smpl = load_smpl(shared_smpl_path(shared), query_t=mocap_t)
    qv = smpl["query_valid"]
    joints_yup = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"],
                              smpl["offsets_m"], smpl["parents"])
    verts = smpl_vertices_from_archive_params(smpl["poses_aa"],
                                              smpl["model_trans_m"],
                                              smpl["betas"])
    est_floor = float(estimate_floor_y(joints_yup))
    h_y0 = _smpl_foot_heights(verts, joints_yup, 0.0)   # 相对 Y=0（绝对高度）
    labels = {name: load_labels(root, sid, n, facts_fid)
              for name, root in LABEL_SETS.items()}
    labels = {k: v for k, v in labels.items() if v is not None}
    return {"h": h_y0, "labels": labels, "qv": qv, "est_floor": est_floor,
            "sums": (left48.sum(axis=1), right48.sum(axis=1)),
            "cells": (left48, right48)}


def main() -> None:
    man = manifest()
    sids = fold_test_sessions()
    pen_rows, cmp_rows = [], []
    for sid in sids:
        try:
            ctx = run_session(sid)
        except Exception as exc:
            print(f"{sid}: FAILED {exc}")
            continue
        subject, action = man[sid]
        h, qv = ctx["h"], ctx["qv"]
        for k, side in enumerate(("L", "R")):
            hh = h[:, k]
            sel = qv & np.isfinite(hh) & (hh < 1.0)
            h_sel = hh[sel]
            if sel.sum() < 10:
                continue
            smpl_cont = h_sel < SMPL_CONTACT_DZ
            pen_rows.append({
                "session": sid, "subject": subject, "action": action, "foot": side,
                "n_valid": int(sel.sum()),
                "h_min": float(h_sel.min()), "h_p1": float(np.percentile(h_sel, 1)),
                "h_p5": float(np.percentile(h_sel, 5)),
                "h_med": float(np.median(h_sel)), "h_p95": float(np.percentile(h_sel, 95)),
                "h_max": float(h_sel.max()),
                "pen_rate": float((h_sel < 0).mean()),
                "pen_depth_mm": float(-h_sel[h_sel < 0].mean() * 1000) if (h_sel < 0).any() else 0.0,
                "smpl_rate": float(smpl_cont.mean()),
                "est_floor_mm": ctx["est_floor"] * 1000,
            })
            sums = ctx["sums"][k]
            cells = np.asarray(ctx["cells"][k])
            for lname, (lab, valid) in ctx["labels"].items():
                cc = lab[:, k]
                vv = valid & sel
                use = vv & np.isfinite(cc)
                if use.sum() < 10:
                    continue
                cc_use = cc[use] > 0.5
                h_use = h_sel[use[sel]]
                smpl_c = h_use < SMPL_CONTACT_DZ
                air = ~cc_use
                grd = cc_use
                # 相位（剔除边缘的稳定摆动/支撑）
                two = np.zeros((cc_use.shape[0], 2), dtype=np.float32)
                two[:, k] = np.where(cc_use, 1.0, 0.0)
                phase = phase_labels(two)[:, k].astype(int)
                swing = phase == 1
                strike_off = (phase == 2) | (phase == 3)
                # 各子集内 SMPL 接触率与压力
                def block(mask):
                    if not mask.any():
                        return (0, 0, float("nan"), float("nan"), float("nan"),
                                float("nan"), float("nan"))
                    return (int(mask.sum()),
                            int((h_use[mask] < SMPL_CONTACT_DZ).sum()),
                            float(np.median(h_use[mask]) * 1000),
                            float(np.median(sums[use][mask])),
                            float(np.mean(sums[use][mask])),
                            float(cells[use][mask].max(axis=1).mean()),
                            float((sums[use][mask] == 0).mean()))
                n_air, air_smpl, air_h, air_sm, air_smean, air_cm, air_z = block(air)
                n_sw, sw_smpl, sw_h, sw_sm, sw_smean, sw_cm, sw_z = block(swing)
                n_so, so_smpl, so_h, so_sm, so_smean, so_cm, so_z = block(strike_off)
                n_grd, grd_smpl, grd_h, grd_sm, grd_smean, grd_cm, grd_z = block(grd)
                n_sc, _, sc_h, sc_sm, sc_smean, sc_cm, sc_z = block(swing & smpl_c)
                n_sa, _, sa_h, sa_sm, sa_smean, sa_cm, sa_z = block(swing & ~smpl_c)
                cmp_rows.append({
                    "session": sid, "subject": subject, "action": action,
                    "foot": side, "label": lname,
                    "n_valid": int(use.sum()),
                    "agree": float((smpl_c == cc_use).mean()),
                    "n_air": n_air, "air_smpl_contact": air_smpl,
                    "air_smpl_contact_rate": air_smpl / max(n_air, 1),
                    "air_h_med_mm": air_h, "air_sum_med": air_sm,
                    "air_sum_mean": air_smean, "air_cellmax_mean": air_cm,
                    "air_zero_sum_rate": air_z,
                    "n_swing": n_sw, "swing_smpl_contact": sw_smpl,
                    "swing_smpl_contact_rate": sw_smpl / max(n_sw, 1),
                    "swing_h_med_mm": sw_h, "swing_sum_med": sw_sm,
                    "swing_sum_mean": sw_smean, "swing_cellmax_mean": sw_cm,
                    "swing_zero_sum_rate": sw_z,
                    "n_strike_off": n_so, "so_smpl_contact_rate": so_smpl / max(n_so, 1),
                    "so_h_med_mm": so_h, "so_sum_med": so_sm,
                    "n_sw_cont": n_sc, "sw_cont_h_med_mm": sc_h,
                    "sw_cont_sum_med": sc_sm, "sw_cont_sum_mean": sc_smean,
                    "sw_cont_cellmax_mean": sc_cm, "sw_cont_zero_rate": sc_z,
                    "n_sw_air": n_sa, "sw_air_h_med_mm": sa_h,
                    "sw_air_sum_med": sa_sm, "sw_air_sum_mean": sa_smean,
                    "sw_air_cellmax_mean": sa_cm, "sw_air_zero_rate": sa_z,
                    "n_grd": n_grd, "grd_smpl_contact": grd_smpl,
                    "grd_smpl_air_rate": (n_grd - grd_smpl) / max(n_grd, 1),
                    "grd_h_med_mm": grd_h, "grd_sum_mean": grd_smean,
                    "grd_cellmax_mean": grd_cm,
                })
        print(f"{sid} done")

    for name, rows in (("penetration", pen_rows), ("label_cmp", cmp_rows)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- 聚合打印 ----
    print("\n== penetration（Y=0 口径）全体 ==")
    p = pen_rows
    print(f"sessions={len(set(r['session'] for r in p))} feet={len(p)}")
    for kk in ("h_min", "h_p1", "h_p5", "h_med", "h_p95"):
        vals = [r[kk] for r in p]
        print(f"  {kk}: median over feet {np.median(vals):+.3f} m, min {min(vals):+.3f}")
    print(f"  pen_rate: median {np.median([r['pen_rate'] for r in p]):.4f}, "
          f"max {max(r['pen_rate'] for r in p):.4f}")
    print(f"  pen_depth_mm (h<0 帧): median {np.median([r['pen_depth_mm'] for r in p]):.1f}, "
          f"max {max(r['pen_depth_mm'] for r in p):.1f}")
    print(f"  est_floor_mm: median {np.median([r['est_floor_mm'] for r in p]):.1f} "
          f"(estimate_floor_y 高于 Y=0 的量)")

    print("\n== label_cmp 按标签集（池化）==")
    for lname in ("offfix", "f6_soft", "smpl24st"):
        c = [r for r in cmp_rows if r["label"] == lname]
        if not c:
            continue
        tot = sum(r["n_valid"] for r in c)
        n_air = sum(r["n_air"] for r in c)
        n_sw = sum(r["n_swing"] for r in c)
        n_so = sum(r["n_strike_off"] for r in c)
        agree = sum(r["agree"] * r["n_valid"] for r in c) / tot
        a_smpl = sum(r["air_smpl_contact"] for r in c)
        sw_smpl = sum(r["swing_smpl_contact"] for r in c)
        so_smpl = sum(r["so_smpl_contact_rate"] * r["n_strike_off"] for r in c)
        print(f"  {lname:9s} valid={tot} agree={agree:.4f}")
        print(f"      air={n_air} 中 SMPL=接触 {a_smpl} ({a_smpl/max(n_air,1):.4f})")
        print(f"      swing(稳定)={n_sw} 中 SMPL=接触 {sw_smpl} ({sw_smpl/max(n_sw,1):.4f})")
        print(f"      strike/off={n_so} 中 SMPL=接触率加权 {so_smpl/max(n_so,1):.4f}")

    print("\n== 稳定摆动帧（offfix）按动作：SMPL 接触率 / 高度 / 压力 ==")
    for act in sorted({r["action"] for r in cmp_rows if r["label"] == "offfix"}):
        c = [r for r in cmp_rows if r["label"] == "offfix" and r["action"] == act]
        n_sw = sum(r["n_swing"] for r in c)
        if n_sw == 0:
            continue
        sw_smpl = sum(r["swing_smpl_contact"] for r in c) / n_sw
        h_med = np.average([r["swing_h_med_mm"] for r in c],
                           weights=[r["n_swing"] for r in c])
        sum_med = np.average([r["swing_sum_med"] for r in c],
                             weights=[r["n_swing"] for r in c])
        zrate = sum(r["swing_zero_sum_rate"] * r["n_swing"] for r in c) / n_sw
        print(f"  {ACTION_NAMES.get(act, act):12s} swing={n_sw:5d} "
              f"smpl_contact={sw_smpl:.3f} h_med={h_med:+.1f}mm "
              f"sum_med={sum_med:6.1f} zero_rate={zrate:.3f}")

    print("\n== 稳定摆动帧（offfix）内子集对比：SMPL 判接触 vs 判离地 的压力 ==")
    tot = {"cont": [0, 0.0, 0.0, 0.0, 0.0], "air": [0, 0.0, 0.0, 0.0, 0.0]}  # n, h*mm, sum_med, sum_mean, zero
    for act in sorted({r["action"] for r in cmp_rows if r["label"] == "offfix"}):
        c = [r for r in cmp_rows if r["label"] == "offfix" and r["action"] == act]
        for grp, col in (("cont", "n_sw_cont"), ("air", "n_sw_air")):
            cc = [r for r in c if r[col] > 0]
            n = sum(r[col] for r in cc)
            if n == 0:
                continue
            h = sum(r[f"sw_{grp}_h_med_mm"] * r[col] for r in cc) / n
            sm = sum(r[f"sw_{grp}_sum_med"] * r[col] for r in cc) / n
            smean = sum(r[f"sw_{grp}_sum_mean"] * r[col] for r in cc) / n
            z = sum(r[f"sw_{grp}_zero_rate"] * r[col] for r in cc) / n
            tot[grp][0] += n; tot[grp][1] += h * n; tot[grp][2] += sm * n
            tot[grp][3] += smean * n; tot[grp][4] += z * n
            print(f"  {ACTION_NAMES.get(act, act):12s} swing_{grp:4s} n={n:5d} "
                  f"h_med={h:+.1f}mm sum_med={sm:6.1f} sum_mean={smean:6.1f} zero={z:.3f}")
    for grp in ("cont", "air"):
        n = tot[grp][0]
        if n == 0:
            continue
        h, sm, smean, z = tot[grp][1:]
        print(f"  {'TOTAL':12s} swing_{grp:4s} n={n:5d} "
              f"h_med={h/n:+.1f}mm sum_med={sm/n:6.1f} sum_mean={smean/n:6.1f} zero={z/n:.3f}")


if __name__ == "__main__":
    main()
