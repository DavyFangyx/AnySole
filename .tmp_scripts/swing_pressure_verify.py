#!/usr/bin/env python3
"""核对：腾空帧（SMPL 足底确实离地）触觉压力是否仍非零（2026-11-11）。

对象 = results_display/DataTest/D3Test_contact 的 36 个 session。
腾空判据用 SMPL 足底最低顶点离地高度（Y=0 绝对口径，避开 estimate_floor_y 偏差）：
  air_strong  h > 5cm   （明显腾空，远超 2cm 鞋底尺度）
  air_mid     h > 3cm
  stance      h < 2cm   （SMPL_CONTACT_DZ 接触口径）
压力 = facts pressure_48.npz 每脚 48 格原始和。

输出 swing_pressure_verify_out/verify.csv（逐 session×脚）与 summary.csv（按动作汇总）。
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
    load_smpl,
    smpl_vertices_from_archive_params,
)
from anysole.data.workspace_adapter import (  # noqa: E402
    load_shared_session,
    shared_smpl_path,
)
from anysole.utils.geometry import fk_pose6d_np  # noqa: E402
from contact_labels import _smpl_foot_heights  # noqa: E402

DISPLAY = REPO / "results_display/DataTest/D3Test_contact"
OUT = REPO / ".tmp_scripts/swing_pressure_verify_out"
OUT.mkdir(parents=True, exist_ok=True)

H_AIR_STRONG, H_AIR_MID, H_STANCE = 0.05, 0.03, 0.02
THR_TACTILE = 100.0
ACTION_NAMES = {"01": "stand_shift", "02": "single_stand", "03": "walk",
                "04": "walk", "05": "run", "06": "squat", "07": "jump_jack",
                "08": "run", "09": "lateral", "10": "sit_stand", "11": "stairs"}


def manifest() -> dict[str, tuple[str, str]]:
    rows = list(csv.DictReader(
        (REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.csv")
        .open(encoding="utf-8-sig")))
    return {r["session_id"]: (r["subject_id"], r["action"]) for r in rows}


def d3_sessions() -> list[str]:
    return sorted(p.stem.replace("_contact", "")
                  for p in (DISPLAY / "tactile_abs/gif").glob("*_contact.gif"))


def block(sums: np.ndarray, cells: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {}
    s = sums[mask]
    c = cells[mask]
    return {
        "n": int(mask.sum()),
        "sum_med": float(np.median(s)),
        "sum_mean": float(np.mean(s)),
        "sum_p10": float(np.percentile(s, 10)),
        "sum_p90": float(np.percentile(s, 90)),
        "zero_rate": float((s <= 0).mean()),
        "gt100_rate": float((s > THR_TACTILE).mean()),
        "gt1000_rate": float((s > 1000).mean()),
        "cellmax_mean": float(c.max(axis=1).mean()),
        "cell_nonzero_rate": float((c > 0).mean()),
    }


def best_lag(h: np.ndarray, s: np.ndarray, max_lag: int = 8) -> tuple[int, float]:
    """压力和对足底高度的互相关最佳滞后（帧）；负值 = 压力滞后于高度。"""
    hh = h - h.mean()
    ss = s - s.mean()
    best, bl = None, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = hh[lag:], ss[:len(ss) - lag]
        else:
            a, b = hh[:lag], ss[-lag:]
        if len(a) < 30 or a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if best is None or abs(r) > abs(best):
            best, bl = r, lag
    return bl, (best or 0.0)


def main() -> None:
    man = manifest()
    rows = []
    for sid in d3_sessions():
        try:
            shared = load_shared_session(sid)
            frames = shared["frames"]
            n = int(len(frames["frame_id"]))
            mocap_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)[:n]
            smpl = load_smpl(shared_smpl_path(shared), query_t=mocap_t)
            qv = np.asarray(smpl["query_valid"], dtype=bool)
            joints_yup = fk_pose6d_np(smpl["pose_6d"], smpl["trans_m"],
                                      smpl["offsets_m"], smpl["parents"])
            verts = smpl_vertices_from_archive_params(smpl["poses_aa"],
                                                      smpl["model_trans_m"],
                                                      smpl["betas"])
            h_all = _smpl_foot_heights(verts, joints_yup, 0.0)  # 相对 Y=0
        except Exception as exc:  # noqa: BLE001
            print(f"{sid}: SKIP {exc}")
            continue
        sums = (np.asarray(shared["pressure"]["left48"], dtype=np.float32)[:n].sum(axis=1),
                np.asarray(shared["pressure"]["right48"], dtype=np.float32)[:n].sum(axis=1))
        cells = (np.asarray(shared["pressure"]["left48"], dtype=np.float32)[:n],
                 np.asarray(shared["pressure"]["right48"], dtype=np.float32)[:n])
        subject, action = man[sid]
        for k, side in enumerate(("L", "R")):
            h = h_all[:, k]
            ok = qv & np.isfinite(h) & (h < 1.0)
            air_s = ok & (h > H_AIR_STRONG)
            air_m = ok & (h > H_AIR_MID)
            st = ok & (h < H_STANCE)
            b_s = block(sums[k], cells[k], st)
            b_a = block(sums[k], cells[k], air_s)
            b_m = block(sums[k], cells[k], air_m)
            if not b_s or not b_a:
                continue
            lag, r = best_lag(h[ok], sums[k][ok])
            row = {
                "session": sid, "subject": subject,
                "action": action, "action_name": ACTION_NAMES.get(action, action),
                "foot": side,
                "n_valid": int(ok.sum()),
                "h_air_strong_med_mm": float(np.median(h[air_s]) * 1000),
                "h_stance_med_mm": float(np.median(h[st]) * 1000),
                "air_sum_mean": b_a["sum_mean"], "stance_sum_mean": b_s["sum_mean"],
                "air_over_stance": b_a["sum_mean"] / max(b_s["sum_mean"], 1e-9),
                "lag_frames": lag, "corr_h_sum": r,
            }
            for tag, b in (("air", b_a), ("airmid", b_m), ("stance", b_s)):
                for key, val in b.items():
                    row[f"{tag}_{key}"] = val
            rows.append(row)
            print(f"{sid} {side}: air n={b_a['n']} med={b_a['sum_med']:.0f} "
                  f"stance med={b_s['sum_med']:.0f} air/stance={b_a['sum_mean']/max(b_s['sum_mean'],1e-9):.2f} "
                  f"air>100={b_a['gt100_rate']:.0%} air_zero={b_a['zero_rate']:.0%}")

    if not rows:
        print("no rows")
        return
    cols = list(rows[0])
    with (OUT / "verify.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 按动作加权汇总
    groups = {}
    for r in rows:
        groups.setdefault(r["action_name"], []).append(r)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["action", "n_feet", "air_frames", "stance_frames",
                    "air_sum_mean", "stance_sum_mean", "air_over_stance",
                    "air_zero_rate", "air_gt100_rate", "air_cellmax_mean",
                    "stance_zero_rate", "median_lag_frames"])
        for name, rs in sorted(groups.items()):
            na = sum(r["air_n"] for r in rs)
            ns = sum(r["stance_n"] for r in rs)
            am = sum(r["air_sum_mean"] * r["air_n"] for r in rs) / max(na, 1)
            sm = sum(r["stance_sum_mean"] * r["stance_n"] for r in rs) / max(ns, 1)
            w.writerow([
                name, len(rs), na, ns, f"{am:.1f}", f"{sm:.1f}",
                f"{am / max(sm, 1e-9):.3f}",
                f"{sum(r['air_zero_rate'] * r['air_n'] for r in rs) / max(na, 1):.3f}",
                f"{sum(r['air_gt100_rate'] * r['air_n'] for r in rs) / max(na, 1):.3f}",
                f"{sum(r['air_cellmax_mean'] * r['air_n'] for r in rs) / max(na, 1):.1f}",
                f"{sum(r['stance_zero_rate'] * r['stance_n'] for r in rs) / max(ns, 1):.3f}",
                f"{np.median([r['lag_frames'] for r in rs]):.0f}",
            ])

    # 全局
    na = sum(r["air_n"] for r in rows)
    ns = sum(r["stance_n"] for r in rows)
    am = sum(r["air_sum_mean"] * r["air_n"] for r in rows) / max(na, 1)
    sm = sum(r["stance_sum_mean"] * r["stance_n"] for r in rows) / max(ns, 1)
    print("\n=== 全局 ===")
    print(f"feet={len(rows)} air_frames={na} stance_frames={ns}")
    print(f"air_sum_mean={am:.1f}  stance_sum_mean={sm:.1f}  ratio={am/max(sm,1e-9):.3f}")
    print(f"air_zero_rate={sum(r['air_zero_rate']*r['air_n'] for r in rows)/max(na,1):.3f}")
    print(f"air_gt100_rate={sum(r['air_gt100_rate']*r['air_n'] for r in rows)/max(na,1):.3f}")
    print(f"air_cell_nonzero_rate={sum(r['air_cell_nonzero_rate']*r['air_n'] for r in rows)/max(na,1):.3f}")
    print(f"air_cellmax_mean={sum(r['air_cellmax_mean']*r['air_n'] for r in rows)/max(na,1):.1f}")
    print(f"median_lag_frames={np.median([r['lag_frames'] for r in rows]):.0f}")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
