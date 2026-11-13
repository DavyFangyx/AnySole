#!/usr/bin/env python3
"""Diagnose what the V8_Cv8U gate K has learned.

Questions:
  * Is K a value-arbitration signal?  Ideal: K should correlate with
    ``err_V - err_T`` (T better -> larger K).
  * Or is K an action-difficulty detector?  Check correlation with
    ``err_V + err_T`` / ``err_VT`` and between-action variance.
  * Or is K subject-specific?  Check between-subject variance (eta^2).

Usage:
  PYTHONPATH=. python results_display/script/v8_cv8u_gate_diagnose.py \
      --ckpt <ckpt.pt> --split-csv <fold csv> --split test \
      --manifest AnysoleWorkspace/protocol/manifests/session_manifest.csv \
      --out results/diagnostics/.../gate_diag.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset
from anysole.registry import get_builder
from anysole.train import condition_inputs

REPO = Path(__file__).resolve().parents[2]


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def _eta2(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    total = values.var()
    if total <= 0:
        return float("nan")
    within = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        within += len(v) * v.var()
    return float(1.0 - within / (len(values) * total))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--manifest", default=str(
        REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.csv"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    entry = get_builder(cfg["model_name"])
    model = entry.build({**cfg, "d_model": int(cfg["d_model"]),
                         "tw": int(cfg["tw"]),
                         "pose_layers": int(cfg.get("pose_layers", 6)),
                         "dropout": float(cfg.get("dropout", 0.1))})
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()

    ds = AnySoleDataset(mode=("eval" if args.split == "val" else "test"),
                        window_length=int(cfg["tw"]),
                        split_csv=Path(args.split_csv),
                        contact_method=str(cfg["contact_method"]),
                        tactile_input=str(cfg["tactile_input"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # session -> action/subject
    meta = {}
    with open(args.manifest, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            meta[row["session_id"]] = (row.get("action", ""), row.get("subject_id", ""))

    rec = {k: [] for k in ("session", "frame", "err_v", "err_t", "err_vt", "k")}
    k_slots = []
    with torch.no_grad():
        for batch in loader:
            bsz = batch["V_feat"].shape[0]
            out = {}
            for name, cidv in (("v", 1), ("t", 2), ("vt", 0)):
                cid = torch.full((bsz,), cidv, dtype=torch.long)
                v, tr, tp, ts = condition_inputs(batch, cid)
                o = model(v.cuda(), tr.cuda(), tp.cuda(), cid.cuda(),
                          T_s2m=ts.cuda())
                out[name] = o
                if name == "vt":
                    k_slots.append(o["unc_gate"][..., 0].cpu().numpy())
            gt = batch["pose_gt"].cuda()
            for name in ("v", "t", "vt"):
                err = ((out[name]["x0_hat"] - gt) ** 2).mean(dim=(1, 2))
                rec["err_" + name].extend(err.cpu().tolist())
            k = out["vt"]["unc_gate"][..., 0].mean(dim=(1, 2))
            rec["k"].extend(k.cpu().tolist())
            rec["session"].extend([str(s) for s in batch["session_id"]])
            rec["frame"].extend(batch["frame_start"].tolist())

    err_v = np.array(rec["err_v"]); err_t = np.array(rec["err_t"])
    err_vt = np.array(rec["err_vt"]); k = np.array(rec["k"])
    subjects = [meta.get(s, ("", ""))[1] for s in rec["session"]]
    actions = [meta.get(s, ("", ""))[0] for s in rec["session"]]
    rel = err_v - err_t          # >0: T better
    diff = err_v + err_t         # difficulty proxy
    k_slots = np.concatenate(k_slots, axis=0)  # (N,tw,10)

    report = {
        "n_windows": int(len(k)),
        "K_mean": float(k.mean()), "K_std": float(k.std()),
        "err_v_mean": float(err_v.mean()), "err_t_mean": float(err_t.mean()),
        "corr": {
            "K_err_V": _corr(k, err_v),
            "K_err_T": _corr(k, err_t),
            "K_err_VT": _corr(k, err_vt),
            "K_errV_minus_errT": _corr(k, rel),
            "K_errV_plus_errT": _corr(k, diff),
            "K_abs_errV_minus_errT": _corr(k, np.abs(rel)),
            "spearman_K_rel": _spearman(k, rel),
            "spearman_K_diff": _spearman(k, diff),
        },
        "eta2_subject": _eta2(k, subjects),
        "eta2_action": _eta2(k, actions),
        "K_by_subject": {s: float(k[np.array(subjects) == s].mean())
                         for s in sorted(set(subjects))},
        "K_by_action": {a: float(k[np.array(actions) == a].mean())
                        for a in sorted(set(actions))},
        "K_slot_mean": {str(i): float(k_slots[..., i].mean())
                        for i in range(k_slots.shape[-1])},
    }
    out = Path(args.out) if args.out else (
        REPO / "results" / "diagnostics" / "20261112_v8_cv8u_pilot"
        / ("gate_diag_%s.json" % args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
