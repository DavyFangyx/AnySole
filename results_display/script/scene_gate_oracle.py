#!/usr/bin/env python3
"""Scene-occlusion gate oracle: does a V-trust gate have headroom on the new E5 scene?

Loads the scene cache (cam3_occ_scene) for one fold's test split, runs
V / T / ∅ / VT, computes the per-window oracle V-trust g* and the half-window
cross-fit (w, a, b) upper bounds, and reports them by scene visibility state.

Usage:
  python results_display/script/scene_gate_oracle.py --fold fS5S6S11 --device cuda:6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ab_decomp import run_rows  # noqa: E402
from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from leak_check_crossfit import crossfit_ab, crossfit_w  # noqa: E402
from stage2_gate_pilot import oracle_g  # noqa: E402

OUT = REPO / "results/diagnostics/20261107_scene_gate_oracle"


def load_states(fold: str) -> dict:
    man_dir = REPO / "results/diagnostics/20261107_e5_sceneocc_full/manifests"
    states = {}
    for p in man_dir.glob("S*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        for w in m["windows"]:
            states[(m["session_id"], int(w["start"]))] = w
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    cell = "V8_Cv5_%s_s1" % args.fold
    cfg = "anysole/configs/v1_sceneocc_%s.yaml" % args.fold
    net, ckpt, tp = build_model(cell, device)
    print("[ckpt] %s" % ckpt)
    loader = build_loader(cfg, "test", args.batch_size, t_phys_v2=tp)
    r = run_rows(net, loader, device, "test", "scene")
    g = oracle_g(r["p_v"], r["p_t"], r["gt"]).mean(axis=1)
    e_v, e_t, e_vt = r["e_v"], r["e_t"], r["e_vt"]
    e_w = crossfit_w(r["p_v"], r["p_t"], r["gt"])
    e_ab, b_cf, b_pw = crossfit_ab(r["p_v"], r["p_t"], r["p_0"], r["gt"])
    states = load_states(args.fold)
    state_names = np.asarray([states.get((s, int(st)), {}).get("state", "?")
                              for s, st in zip(r["session"], r["start"])])
    print("[scene/test] n=%d VT=%.2f V=%.2f T=%.2f | g* mean %.2f p05 %.2f p95 %.2f"
          % (len(g), float(np.mean(e_vt)), float(np.mean(e_v)),
             float(np.mean(e_t)), float(np.mean(g)),
             float(np.percentile(g, 5)), float(np.percentile(g, 95))))
    print("  crossfit w gain vs VT = %+.2f mm; ab gain = %+.2f mm; b_cf=%.2f"
          % (float(np.mean(e_vt - e_w)), float(np.mean(e_vt - e_ab)), b_cf))
    for st in ("clean", "partial", "full"):
        m = state_names == st
        if not m.any():
            continue
        print("  %-7s n=%3d g*=%.2f VT=%.1f V=%.1f T=%.1f | w gain=%+.1f ab gain=%+.1f"
              % (st, int(m.sum()), float(g[m].mean()), float(e_vt[m].mean()),
                 float(e_v[m].mean()), float(e_t[m].mean()),
                 float(np.mean((e_vt - e_w)[m])), float(np.mean((e_vt - e_ab)[m]))))
    out = {
        "fold": args.fold, "n": int(len(g)),
        "g_star_mean": float(np.mean(g)),
        "g_star_p05": float(np.percentile(g, 5)),
        "g_star_p95": float(np.percentile(g, 95)),
        "e_vt": float(np.mean(e_vt)), "e_v": float(np.mean(e_v)),
        "e_t": float(np.mean(e_t)),
        "gain_w_crossfit": float(np.mean(e_vt - e_w)),
        "gain_ab_crossfit": float(np.mean(e_vt - e_ab)),
        "b_crossfit": float(b_cf),
        "by_state": {},
    }
    for st in ("clean", "partial", "full"):
        m = state_names == st
        if m.any():
            out["by_state"][st] = {
                "n": int(m.sum()), "g_star": float(g[m].mean()),
                "e_vt": float(e_vt[m].mean()), "e_v": float(e_v[m].mean()),
                "e_t": float(e_t[m].mean()),
                "gain_w": float(np.mean((e_vt - e_w)[m])),
                "gain_ab": float(np.mean((e_vt - e_ab)[m])),
            }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ("%s.json" % args.fold)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] %s" % (args.out_dir / ("%s.json" % args.fold)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
