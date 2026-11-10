#!/usr/bin/env python3
"""Recompute soft-f6 contact from current facts (BVH24) for all sessions and
compare against the frozen facts contact_f6_soft.npy."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AnysoleWorkspace/tool"))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402


def norm(name: str) -> str:
    return name.lower().replace("_", "")


def names_for(ctx, side):
    return [i for i, n in enumerate(ctx["names"])
            if norm(n).startswith(f"{side}foot") or norm(n).startswith(f"{side}toe")]


def foot_signal(ctx, side):
    idx = names_for(ctx, side)
    if not idx:
        raise ValueError(f"no {side} foot joints")
    foot = ctx["pts"][:, idx, :]
    height = foot[:, :, 2].min(axis=1) - float(ctx["floor"])
    speed = np.zeros(len(foot))
    speed[1:] = np.linalg.norm(foot[1:] - foot[:-1], axis=2).mean(axis=1) * A.FPS
    return height.astype(np.float64), speed


def f6_signals(ctx, side):
    idx = names_for(ctx, side)
    filtered = median_filter(ctx["pts"][:, idx, :], size=(3, 1, 1))
    center = filtered.mean(axis=1)
    velocity = np.zeros_like(center)
    velocity[1:] = (center[1:] - center[:-1]) * A.FPS
    speed = np.linalg.norm(velocity, axis=1)
    az = np.zeros(len(center))
    az[2:] = (center[2:, 2] - center[:-2, 2]) / 2.0 * A.FPS * A.FPS
    height, _ = foot_signal(ctx, side)
    return {"v": speed, "az": az, "h": height}


def joint_index(names, side, suffix):
    ln = [norm(n) for n in names]
    for pref in (f"{side}{suffix}", f"{side}{suffix}base"):
        if pref in ln:
            return ln.index(pref)
    if suffix == "toe":
        for i, n in enumerate(ln):
            if n.startswith(f"{side}foot") and "endsite" in n:
                return i
    for i, n in enumerate(ln):
        if n.startswith(f"{side}{suffix}") and "endsite" not in n:
            return i
    raise ValueError((side, suffix, names))


A._foot_signal = foot_signal
A._f6_signals = f6_signals
A._joint_index = joint_index


def main() -> None:
    rows = list(csv.DictReader(open(
        ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv",
        encoding="utf-8-sig")))
    sessions = sorted({r[c].strip() for r in rows for c in ("train", "val", "test")
                       if (r.get(c) or "").strip()})
    results = []
    for k, sid in enumerate(sessions):
        try:
            session = A.load_shared_session(sid)
            ctx = A._context(session)
            new = A.soft_f6_contact(ctx)
            old = np.load(Path(session["dir"]) / "contact_f6_soft.npy")
            diff = int(np.count_nonzero(old != new))
            row = {"session": sid, "n_mismatch": diff, "n": int(new.size),
                   "old_rate_L": float(old[:, 6].mean()), "old_rate_R": float(old[:, 7].mean()),
                   "new_rate_L": float(new[:, 6].mean()), "new_rate_R": float(new[:, 7].mean())}
        except Exception as exc:  # noqa: BLE001
            row = {"session": sid, "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        if (k + 1) % 20 == 0:
            print(f"{k+1}/{len(sessions)}", flush=True)
    out = ROOT / ".tmp_scripts" / "contact_f6_bvh24_audit.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    bad = [r for r in results if r.get("n_mismatch")]
    err = [r for r in results if r.get("error")]
    print(f"done sessions={len(results)} mismatch_sessions={len(bad)} errors={len(err)}")
    print("mismatch sample:", bad[:5])
    print("error sample:", err[:5])


if __name__ == "__main__":
    main()
