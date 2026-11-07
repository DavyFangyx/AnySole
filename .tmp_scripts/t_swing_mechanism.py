#!/usr/bin/env python3
"""T 作用 V 机制检验：逐窗伤害×摆动比例 + 压力饱和统计（2026-11-07）。

Part A（假设 a 行为面）：Cv7/Cv5 四折 window_detail_test.json 逐窗
harm = PA(VT) − PA(V)，join f6_soft 相位 → 窗内 swing 比例，相关+分箱，
按动作（walk/run）拆分。

Part B（假设 a 输入面）：shared facts 压力 clip/1023 归一后按动作×相位
统计饱和率（max≥0.99）、零格率、均值。

会话帧对齐：contact 标签 npz 与 facts 帧同长同序（逐位验证后使用）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole_tta.phase_common import load_contact, phase_labels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "results" / "cells"
OUT = REPO / ".tmp_scripts" / "t_swing_mechanism_out"
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
MODELS = ("V7_Cv7", "V7_Cv5")
ACTION_GROUPS = {"03": "walk", "04": "walk", "05": "run", "08": "run",
                 "06": "squat", "01": "stand_shift", "02": "single_stand",
                 "07": "jump_jack", "09": "lateral", "10": "sit_stand",
                 "11": "stairs"}

_session_action: dict[str, str] = {}
with open(REPO / "AnysoleWorkspace/protocol/manifests/"
          "session_manifest.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        _session_action[row["session_id"]] = ACTION_GROUPS.get(
            row["action"], row["action"])


def session_contact(sid: str) -> tuple[np.ndarray, np.ndarray]:
    contact, frame_id, valid = load_contact(sid)
    assert frame_id[0] == 0 and np.all(np.diff(frame_id) == 1), \
        f"{sid}: frame_id not contiguous arange"
    return contact, valid


def session_vlat(sid: str) -> np.ndarray:
    """GT 逐帧侧向速度幅值 (m/frame)：根速度水平分量中垂直于朝向的部分。

    朝向 = 骨盆→脊柱水平投影（GT 关节，世界系 +Y up，无需再载 smpl）。
    """
    from anysole_tta.phase_common import load_gt_joints
    j = load_gt_joints(sid)                      # (T,24,3) m
    root = j[:, 0]
    v = np.diff(root, axis=0, prepend=root[:1])  # (T,3) m/frame
    fwd = j[:, 3] - j[:, 0]                      # 骨盆→脊柱
    fwd[:, 1] = 0.0
    fn = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd = fwd / np.maximum(fn, 1e-6)
    vh = v.copy()
    vh[:, 1] = 0.0
    along = (vh * fwd).sum(axis=1, keepdims=True) * fwd
    lat = vh - along
    return np.linalg.norm(lat, axis=1).astype(np.float32)


def load_pressure(sid: str) -> np.ndarray:
    from anysole.data.workspace_adapter import load_shared_session
    shared = load_shared_session(sid)
    p = shared["pressure"]
    l = np.asarray(p["left48"], dtype=np.float32)
    r = np.asarray(p["right48"], dtype=np.float32)
    n = max(len(l), len(r))
    t = np.concatenate([l[:n], r[:n]], axis=1) if l.shape[0] == r.shape[0] else (
        np.concatenate([l, np.zeros((n - len(l), 48), np.float32)], axis=1)
        if len(l) < n else
        np.concatenate([np.zeros((n - len(r), 48), np.float32), r], axis=1))
    return np.clip(t, 0.0, 1023.0) / 1023.0


def part_a():
    print("=== Part A: window harm x swing fraction ===")
    rows = []
    corr_rows = []
    bin_rows = []
    for fold in FOLDS:
        for model in MODELS:
            cell = f"{model}_{fold}_s1"
            wpath = CELLS / cell / "metrics" / "window_detail_test.json"
            if not wpath.is_file() and fold == "fS6S7":
                cell = f"{model}_fS6S7S13_s1_s13out"
                wpath = CELLS / cell / "metrics" / "window_detail_test.json"
            if not wpath.is_file():
                print(f"  missing {cell}")
                continue
            d = json.loads(wpath.read_text())
            vt = d["conditions"]["VT2M"]
            v = d["conditions"]["V2M"]
            sids = vt["session_ids"]
            assert sids == v["session_ids"]
            fs = np.asarray(vt["frame_starts"], dtype=int)
            nf = np.asarray(vt["n_frames"], dtype=int)
            harm_full = np.asarray(vt["metrics"]["pa_mpjpe_mm"]) - \
                np.asarray(v["metrics"]["pa_mpjpe_mm"])
            harm_af = np.asarray(vt["metrics"]["PA-MPJPE_anklefoot"]) - \
                np.asarray(v["metrics"]["PA-MPJPE_anklefoot"])
            harm_lf = np.asarray(vt["metrics"]["PA-MPJPE_l_foot"]) - \
                np.asarray(v["metrics"]["PA-MPJPE_l_foot"])
            harm_rf = np.asarray(vt["metrics"]["PA-MPJPE_r_foot"]) - \
                np.asarray(v["metrics"]["PA-MPJPE_r_foot"])
            harm_upper = np.asarray(vt["metrics"]["PA-MPJPE_upper"]) - \
                np.asarray(v["metrics"]["PA-MPJPE_upper"])
            harm_lower = np.asarray(vt["metrics"]["PA-MPJPE_lower"]) - \
                np.asarray(v["metrics"]["PA-MPJPE_lower"])

            contact_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            press_cache: dict[str, np.ndarray] = {}
            vlat_cache: dict[str, np.ndarray] = {}
            swing_frac = np.zeros(len(sids), dtype=np.float32)
            flight_frac = np.zeros(len(sids), dtype=np.float32)
            strike_frac = np.zeros(len(sids), dtype=np.float32)
            off_frac = np.zeros(len(sids), dtype=np.float32)
            press_grad = np.zeros(len(sids), dtype=np.float32)
            press_mag = np.zeros(len(sids), dtype=np.float32)
            v_lat = np.zeros(len(sids), dtype=np.float32)
            act = np.empty(len(sids), dtype=object)
            for i, (sid, s0, n) in enumerate(zip(sids, fs, nf)):
                if sid not in contact_cache:
                    contact_cache[sid] = session_contact(sid)
                if sid not in press_cache:
                    press_cache[sid] = load_pressure(sid)
                if sid not in vlat_cache:
                    vlat_cache[sid] = session_vlat(sid)
                p = press_cache[sid]
                pseg = p[s0:s0 + n]
                press_grad[i] = float(np.abs(np.diff(pseg, axis=0)).mean())
                press_mag[i] = float(pseg.mean())
                vl = vlat_cache[sid]
                v_lat[i] = float(vl[s0:s0 + n].mean()) if len(vl) >= s0 + n \
                    else float(vl[s0:].mean())
                c, valid = contact_cache[sid]
                ph = phase_labels(c)
                seg = ph[s0:s0 + n]
                vseg = valid[s0:s0 + n]
                ok = vseg.astype(bool)
                if ok.sum() == 0:
                    swing_frac[i] = flight_frac[i] = strike_frac[i] = np.nan
                else:
                    seg = seg[ok]
                    swing_frac[i] = (seg == 1).mean()
                    strike_frac[i] = (seg == 2).mean()
                    off_frac[i] = (seg == 3).mean()
                    flight_frac[i] = ((seg == 1).all(axis=1)).mean()
                act[i] = _session_action.get(sid, "?")

            feats = {"swing": swing_frac, "strike": strike_frac,
                     "off": off_frac, "mag": press_mag,
                     "grad": press_grad, "vlat": v_lat}
            harms = {"full": harm_full, "anklefoot": harm_af,
                     "l_foot": harm_lf, "r_foot": harm_rf,
                     "upper": harm_upper, "lower": harm_lower}
            with open(OUT / "A_feature_corr.csv", "a", newline="") as fh:
                w = csv.writer(fh)
                for fn, fv in feats.items():
                    sel = np.isfinite(fv)
                    for aname in ("ALL", "walk", "run", "lateral"):
                        m = sel & ((act == aname) if aname != "ALL"
                                   else np.ones(len(sids), bool))
                        if m.sum() < 20:
                            continue
                        for pn, ph_ in harms.items():
                            h = ph_[m]
                            w.writerow([model, fold, pn, fn, aname, int(m.sum()),
                                        round(float(np.corrcoef(h, fv[m])[0, 1]), 3)])

            ok = ~np.isnan(swing_frac)
            for name, harm in harms.items():
                # 压力动态/幅值相关（全动作池化）
                h, g, m = harm[ok], press_grad[ok], press_mag[ok]
                rows.append([model, fold, f"{name}~grad", len(h),
                             float(np.corrcoef(h, g)[0, 1])])
                rows.append([model, fold, f"{name}~mag", len(h),
                             float(np.corrcoef(h, m)[0, 1])])
                for aname in ("walk", "run"):
                    mm = ok & (act == aname)
                    if mm.sum() < 20:
                        continue
                    hh, gg, mh = harm[mm], press_grad[mm], press_mag[mm]
                    corr_rows.append([model, fold, f"{name}~grad", aname,
                                      mm.sum(),
                                      float(np.corrcoef(hh, gg)[0, 1]),
                                      float(hh.mean()), float(gg.mean())])
                    corr_rows.append([model, fold, f"{name}~mag", aname,
                                      mm.sum(),
                                      float(np.corrcoef(hh, mh)[0, 1]),
                                      float(hh.mean()), float(mh.mean())])
                h, s = harm[ok], swing_frac[ok]
                r = float(np.corrcoef(h, s)[0, 1])
                rows.append([model, fold, name, len(h), r])
                for aname in ("walk", "run"):
                    m = ok & (act == aname)
                    if m.sum() < 20:
                        continue
                    hh, ss = harm[m], swing_frac[m]
                    corr_rows.append([model, fold, name, aname, m.sum(),
                                      float(np.corrcoef(hh, ss)[0, 1]),
                                      float(hh.mean()), float(ss.mean())])
                # 分箱：swing 四分位 → 均值 harm
                q = np.quantile(s, [0.25, 0.5, 0.75])
                bins = np.digitize(s, q)
                for b in range(4):
                    m = bins == b
                    if m.sum() == 0:
                        continue
                    bin_rows.append([model, fold, name, f"bin{b}", int(m.sum()),
                                     float(s[m].mean()), float(h[m].mean())])

    with open(OUT / "A_window_corr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "part", "n", "pearson"])
        w.writerows(rows)
    with open(OUT / "A_window_corr_byaction.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "part", "action", "n", "pearson",
                    "mean_harm", "mean_swing"])
        w.writerows(corr_rows)
    with open(OUT / "A_window_bins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "part", "bin", "n",
                    "mean_swing", "mean_harm"])
        w.writerows(bin_rows)
    print(f"  rows={len(rows)} by-action rows={len(corr_rows)} "
          f"bin rows={len(bin_rows)}")


def part_b():
    print("=== Part B: pressure stats by action x phase ===")
    sids = set()
    for fold in FOLDS:
        cell = f"V7_Cv5_{fold}_s1"
        wpath = CELLS / cell / "metrics" / "window_detail_test.json"
        if not wpath.is_file() and fold == "fS6S7":
            wpath = CELLS / "V7_Cv5_fS6S7S13_s1_s13out" / "metrics" / "window_detail_test.json"
        if wpath.is_file():
            d = json.loads(wpath.read_text())
            sids.update(d["conditions"]["VT2M"]["session_ids"])
    sids = sorted(sids)
    print(f"  sessions: {len(sids)}")
    rows = []
    for sid in sids:
        act = _session_action.get(sid, "?")
        try:
            c, valid = session_contact(sid)
            p = load_pressure(sid)
        except Exception as exc:
            print(f"  {sid} FAIL {type(exc).__name__}: {exc}")
            continue
        n = min(len(c), len(p))
        c, valid, p = c[:n], valid[:n], p[:n]
        ph = phase_labels(c)
        frame_max = p.max(axis=1)
        frame_sum = p.sum(axis=1)
        zero_cells = (p == 0).mean(axis=1)
        for ph_name, ph_id in (("Stance", 0), ("Swing", 1),
                               ("Strike", 2), ("Off", 3)):
            for foot in (0, 1):
                m = (ph[:, foot] == ph_id) & valid.astype(bool)
                if m.sum() < 5:
                    continue
                rows.append([
                    sid, act, ph_name, foot,
                    int(m.sum()),
                    float((frame_max[m] >= 0.99).mean()),
                    float(frame_max[m].mean()),
                    float(frame_sum[m].mean()),
                    float(zero_cells[m].mean()),
                ])
    with open(OUT / "B_pressure_by_phase.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "action", "phase", "foot", "n_frames",
                    "sat_rate", "mean_framemax", "mean_framesum",
                    "zero_cell_rate"])
        w.writerows(rows)
    print(f"  rows={len(rows)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        part_a()
    if which in ("all", "b"):
        part_b()
