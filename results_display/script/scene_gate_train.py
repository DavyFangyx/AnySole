#!/usr/bin/env python3
"""Scene-trained V-trust gate pilot.

Trains the same GateHead (v_tok, t_tok, IG → g*) on the scene-occlusion cache
for one fold, then evaluates on the held-out scene test split:
  - ghat ~ g* (Spearman), overall and by clean/partial/full state
  - realized lower-body PA from mixing V/T with ghat, versus VT
  - crossfit oracle headroom (from ab_decomp) for reference

Usage:
  python results_display/script/scene_gate_train.py \
      --fold fS5S11S12 --device cuda:6
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
from ig_contact_pilot import ContactHead  # noqa: E402
from stage2_gate_pilot import (IG_HEAD, GateHead, feats, run_split,  # noqa: E402
                               spearman, train_gate, win_lower_pa)

OUT = REPO / "results/diagnostics/20261107_scene_gate_train"


def load_states() -> dict:
    states = {}
    for p in (REPO / "results/diagnostics/20261107_e5_sceneocc_full/manifests").glob("S*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        for w in m["windows"]:
            states[(m["session_id"], int(w["start"]))] = w["state"]
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S11S12")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    cell = "V8_Cv5_%s_s1" % args.fold
    cfg = args.cfg or ("anysole/configs/v1_sceneocc_all_%s.yaml" % args.fold)
    net, ckpt, tp = build_model(cell, device)
    print("[ckpt] %s" % ckpt)
    head = ContactHead(256).to(device)
    head.load_state_dict(torch.load(IG_HEAD, map_location="cpu"))
    head.eval(); head.requires_grad_(False)

    data, rows = {}, {}
    for split in ("train", "val", "test"):
        loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
        data[split] = run_split(net, head, loader, device, split, "scene")
        print("[%s] n=%d" % (split, len(data[split]["g_star"])))
        loader = build_loader(cfg, split, args.batch_size, t_phys_v2=tp)
        rows[split] = run_rows(net, loader, device, split, "scene")
        assert len(rows[split]["e_vt"]) == len(data[split]["g_star"])

    gate = train_gate(data["train"], data["val"], device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gate.state_dict(), args.out_dir / ("gate_head_%s.pt" % args.fold))

    states = load_states()
    out = {"fold": args.fold, "n": {}, "test": {}}
    for split in ("train", "val", "test"):
        r = data[split]
        with torch.no_grad():
            ghat = gate(torch.from_numpy(feats(r)).float().to(device)).cpu().numpy()
        mix = (1.0 - ghat.reshape(-1, 1, 1, 1)) * rows[split]["p_t"] + \
              ghat.reshape(-1, 1, 1, 1) * rows[split]["p_v"]
        e_gate = win_lower_pa(mix, rows[split]["gt"])
        st = np.asarray([states.get((s, int(t)), "?")
                         for s, t in zip(r["session"], r["start"])])
        rec = {
            "n": int(len(ghat)),
            "g_star": float(r["g_star"].mean()),
            "ghat": float(ghat.mean()),
            "spearman": spearman(ghat, r["g_star"]),
            "e_vt": float(r["e_vt"].mean()),
            "e_gate": float(e_gate.mean()),
            "gain_vs_vt": float(np.mean(r["e_vt"] - e_gate)),
        }
        rec["by_state"] = {}
        for s in ("clean", "partial", "full"):
            m = st == s
            if m.any():
                rec["by_state"][s] = {
                    "n": int(m.sum()),
                    "g_star": float(r["g_star"][m].mean()),
                    "ghat": float(ghat[m].mean()),
                    "spearman": spearman(ghat[m], r["g_star"][m]),
                    "e_vt": float(r["e_vt"][m].mean()),
                    "e_gate": float(e_gate[m].mean()),
                    "gain_vs_vt": float(np.mean((r["e_vt"] - e_gate)[m])),
                }
        out[split] = rec
        print("[%s] g*=%.2f ghat=%.2f rho=%+.3f VT=%.2f gate=%.2f gain=%+.2f"
              % (split, rec["g_star"], rec["ghat"], rec["spearman"],
                 rec["e_vt"], rec["e_gate"], rec["gain_vs_vt"]))
        for s, v in rec["by_state"].items():
            print("  %-7s n=%3d g*=%.2f ghat=%.2f rho=%+.3f VT=%.1f gate=%.1f gain=%+.1f"
                  % (s, v["n"], v["g_star"], v["ghat"], v["spearman"],
                     v["e_vt"], v["e_gate"], v["gain_vs_vt"]))
    (args.out_dir / ("%s.json" % args.fold)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] %s" % (args.out_dir / ("%s.json" % args.fold)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
