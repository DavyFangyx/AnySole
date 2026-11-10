#!/usr/bin/env python3
"""Audit baseline input data against the current shared facts / raw canonical.

Checks per session:
  1. shared facts pressure_48 vs hf-raw pressure npz (processed48, absolute wall clock)
  2. MotionPRO adapter_v2 carpet vs re-render from current facts
  3. MotionPRO adapter_v1 contact.npy vs recomputed soft-f6 from current facts
  4. MotionPRO adapter_v1 smpl.npy vs recomputed interpolation from current facts
  5. shared MMVP 31x11 insole vs recomputed from current facts pressure_48
  6. facts contact_f6_soft.npy vs recomputed from current facts
  7. shared facts frames vs hf-raw (mtime only report)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Baselines/MotionPRO"))
sys.path.insert(0, str(ROOT / "AnysoleWorkspace/tool"))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402
from tactile_pipeline.align import nearest_indices  # noqa: E402
from lib.util.carpet_v2 import render_carpet  # noqa: E402


def split_sessions(name: str) -> list[str]:
    rows = list(csv.DictReader(open(ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{name}.csv",
                                    encoding="utf-8-sig")))
    return [r["test"].strip() for r in rows if (r.get("test") or "").strip()]


def check(sid: str, cam: int = 3) -> dict:
    out = {"session": sid}
    session = A.load_shared_session(sid, cam)
    sdir = Path(session["dir"])
    meta = session["meta"]
    prov = meta["pressure_provenance"]
    frames = session["frames"]
    pressure = session["pressure"]
    n = len(frames["frame_id"])
    date, subject = sdir.parts[-3], sdir.parts[-2]

    # ---- 1. facts vs raw processed48 (absolute wall clock)
    raw = ROOT / f"AnysoleWorkspace/hf-raw/pressure/data/dataset/{date}/{subject}/{sid}"
    wall0 = float(prov["video_frame0_wall_s"])
    fps = float(meta["fps"])
    targets = wall0 + np.arange(n) / fps
    raw_diff = {}
    for side, key in (("left", "left48"), ("right", "right48")):
        z = np.load(raw / f"{side}.npz", allow_pickle=True)
        vals = np.asarray(z["processed48"], dtype=np.float32)
        wall = np.asarray(z["wall_time_day_s"], dtype=np.float64)
        idx = nearest_indices(wall, targets)
        ref = vals[idx]
        cur = np.asarray(pressure[key], dtype=np.float32)
        raw_diff[side] = {
            "max_abs": float(np.abs(cur - ref).max()),
            "n_mismatch": int(np.count_nonzero(cur != ref)),
            "frames": int(n),
        }
    out["facts_vs_raw"] = raw_diff

    # ---- 2. MotionPRO carpet (adapter_v2) vs re-render current facts
    v2 = ROOT / f"AnysoleWorkspace/model_inputs/MotionPRO/adapter_v2_carpet/cam{cam}/{date}/{subject}/{sid}"
    geom_path = ROOT / f"AnysoleWorkspace/model_inputs/MotionPRO/adapter_v2_carpet/geometry_cam{cam}.json"
    geom = json.loads(geom_path.read_text())
    ctx = A._context(session)
    tracks = A.foot_tracks(ctx)
    carpet = render_carpet(tracks, pressure["left48"], pressure["right48"], geom, A._paint_footprint)
    stored = np.load(v2 / "pressure.npz")["pressure"]
    out["carpet_vs_adapter_v2"] = {
        "shape_match": bool(stored.shape == carpet.shape),
        "max_abs": float(np.abs(stored.astype(np.float64) - carpet).max()) if stored.shape == carpet.shape else None,
        "n_pixel_mismatch": int(np.count_nonzero(stored != carpet)) if stored.shape == carpet.shape else None,
        "n_frames": int(carpet.shape[0]),
    }

    # ---- 3. MotionPRO contact vs recomputed soft-f6 from current facts
    contact_new = A.soft_f6_contact(ctx)
    contact_old = np.load(v2 / "contact.npy")
    out["contact_vs_adapter"] = {
        "shape_match": bool(contact_old.shape == contact_new.shape),
        "n_mismatch": int(np.count_nonzero(contact_old != contact_new)),
        "n_elements": int(contact_new.size),
        "rates_old": [float(x) for x in contact_old[:, 6].mean(axis=0).reshape(-1)] if contact_old.ndim == 2 else None,
        "rates_new": [float(x) for x in contact_new[:, 6].mean(axis=0).reshape(-1)],
    }

    # ---- 4. MotionPRO smpl vs recomputed interpolation from current facts
    src = A.load_smpl_source(session)
    q = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    go, mo = A._interp_to_grid(src["times"], src["poses"][:, :3], q)
    bp, mb = A._interp_to_grid(src["times"], src["poses"][:, 3:], q)
    tr, mt = A._interp_to_grid(src["times"], src["trans"], q)
    old_smpl = np.load(v2 / "smpl.npy", allow_pickle=True).item()
    out["smpl_vs_adapter"] = {
        "betas_max_abs": float(np.abs(old_smpl["betas"] - src["betas"][:10]).max()),
        "global_orient_max_abs": float(np.abs(old_smpl["global_orient"] - go).max()),
        "body_pose_max_abs": float(np.abs(old_smpl["body_pose"] - bp).max()),
        "transl_max_abs": float(np.abs(old_smpl["transl"] - tr).max()),
    }

    # ---- 5. MMVP 31x11 vs recomputed from current facts
    from AnysoleWorkspace.tool import build_shared as B
    mmvp = ROOT / f"AnysoleWorkspace/shared/representations/tactile/mmvp_31x11/v1/{date}/{subject}/{sid}/insole"
    maps = {foot: B.mmvp_map(foot) for foot in ("L", "R")}
    diffs = []
    for frame in (0, n // 2, n - 1):
        feet = []
        for foot, key in (("L", "left48"), ("R", "right48")):
            indices, mask = maps[foot]
            grid = np.zeros((31, 11), dtype=np.float32)
            grid[mask] = pressure[key][frame][indices[mask]]
            feet.append(grid)
        ref = np.stack(feet, axis=0)
        cur = np.load(mmvp / f"{frame:06d}.npy")
        diffs.append(float(np.abs(ref - cur).max()))
    out["mmvp_vs_facts"] = {"max_abs": max(diffs), "frames_checked": [0, n // 2, n - 1]}

    # ---- 6. facts contact_f6_soft vs recomputed
    cpath = sdir / "contact_f6_soft.npy"
    if cpath.is_file():
        c_old = np.load(cpath)
        out["facts_contact_vs_recompute"] = {
            "shape": list(c_old.shape),
            "n_mismatch": int(np.count_nonzero(c_old != contact_new)),
            "n_elements": int(c_old.size),
        }
    else:
        out["facts_contact_vs_recompute"] = {"missing": True}

    return out


def main() -> None:
    folds = {
        "fS5S6S11": ("splits_test_S5S6S11_val_S12", ["S11011", "S6011", "S8011"]),
        "fS5S11S12": ("splits_test_S5S11S12_val_S13", ["S12011", "S6111", "S11013"]),
        "fS6S7": ("splits_test_S6S7_val_S14", ["S6012", "S7011", "S11012"]),
        "fS5S6S8": ("splits_test_S5S6S8_val_S7", ["S5011", "S8011", "S6013"]),
    }
    results = []
    for fold, (split_name, sids) in folds.items():
        test_sessions = set(split_sessions(split_name))
        for sid in sids:
            if sid not in test_sessions:
                print(f"skip {sid}: not in {split_name} test", flush=True)
                continue
            try:
                r = check(sid)
                r["fold"] = fold
                results.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {sid}: {type(exc).__name__}: {exc}", flush=True)
    out = ROOT / ".tmp_scripts" / "audit_data_fairness.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
